"""
audit_existing_results.py
=========================
Post-hoc audit of the N50 R3 net16v2 CCG experiment results.

Checks:
  1. AP completeness  (each JSON has exactly N_SAMPLES=50 scenarios)
  2. AP MILP vs PyTorch cost cross-check
  3. Convergence quality: AP cost distribution per iteration
  4. Summary verification (all referenced files exist, fields plausible)
  5. Writes audit_existing_results.json
  6. Prints concise table to terminal
  7. Issues a final recommendation

Does NOT rerun any part of the experiment.
"""

import os, sys, json, glob
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# ─────────────────────────────────────────────────────────────
# CONFIG  (must match run_all_net16_v2.py exactly)
# ─────────────────────────────────────────────────────────────
BASE_DIR   = "/data/home/aviyac/retailers_problem/R3/N50/saainit_net16_v2"
N_SAMPLES  = 50
SEEDS      = list(range(10))
R          = 3
c, h, b    = 0.25, 0.05, 0.5
SCS_EPS    = 1e-4
SCS_MAX    = 200_000
ABS_TOL    = 1e-3   # threshold for "large" absolute mismatch
REL_TOL    = 1e-3   # threshold for "large" relative mismatch

# ─────────────────────────────────────────────────────────────
# MODEL  (copy of the class from run_all_net16_v2.py)
# ─────────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R * 2 + 1, 16), nn.ReLU(),
            nn.Linear(16, 16),         nn.ReLU(),
            nn.Linear(16, R)
        )
        self.Q10 = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.Q1r = nn.Parameter(torch.zeros(R, dtype=torch.float32))
        x       = cp.Variable(R)
        z_param = cp.Parameter(R)
        Q10_p   = cp.Parameter(1)
        cons    = [cp.sum(x) <= Q10_p + 1e-6, x >= 0]
        self.proj_layer = CvxpyLayer(
            cp.Problem(cp.Minimize(cp.sum_squares(x - z_param)), cons),
            parameters=[z_param, Q10_p], variables=[x]
        )

    def per_sample_cost(self, xi1, xi2):
        N   = xi1.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        net_in = torch.cat([xi1,
                            q1r.unsqueeze(0).expand(N, -1),
                            q10.expand(N, 1)], dim=1)
        z = self.fc_layers(net_in)
        q2, = self.proj_layer(z, q10.expand(N, 1),
                              solver_args={"solve_method": "SCS",
                                           "eps": SCS_EPS,
                                           "max_iters": SCS_MAX})
        stage1 = c * (q10 + torch.sum(q1r)) + h * q10
        t1 = b * (xi2 + xi1 - q2 - q1r) - h * q2
        t2 = h * (q1r - xi1 - xi2)
        t3 = b * (xi1 - q1r) - h * xi2
        return stage1 + torch.sum(torch.maximum(torch.maximum(t1, t2), t3), dim=1)


def load_model(ckpt_path):
    saved = torch.load(ckpt_path, map_location="cpu")
    sd    = saved["model_state_dict"]
    m     = Model()
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m, saved.get("best_loss", float("nan"))


def ckpt_for_ap(seed, ap_iter):
    """Return checkpoint path used to generate AP iter `ap_iter`."""
    if ap_iter == 1:
        return os.path.join(BASE_DIR, f"seed_{seed}_nominal.pth")
    else:
        return os.path.join(BASE_DIR, f"seed_{seed}_mp_iter{ap_iter-1}.pth")


def ap_json_path(seed, it):
    return os.path.join(BASE_DIR, f"seed_{seed}_ap_iter{it}.json")


def summary_path(seed):
    return os.path.join(BASE_DIR, f"seed_{seed}_summary.json")

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
DIVIDER = "─" * 90

def fmt_arr(a, decimals=4):
    return "[" + ", ".join(f"{x:.{decimals}f}" for x in a) + "]"

# ─────────────────────────────────────────────────────────────
# MAIN AUDIT
# ─────────────────────────────────────────────────────────────
print(DIVIDER)
print("AUDIT: N50 R3 net16v2  —  post-hoc result verification")
print(DIVIDER)

audit = {
    "config": {"N_SAMPLES": N_SAMPLES, "SEEDS": SEEDS, "R": R,
               "abs_tol": ABS_TOL, "rel_tol": REL_TOL},
    "ap_completeness": {},
    "ap_crosscheck":   {},
    "convergence_quality": {},
    "summary_verification": {},
    "flags": {
        "incomplete_ap_files": [],
        "large_mismatch_cases": [],
        "missing_files": [],
    }
}

# ══════════════════════════════════════════════════════════════
# PASS 1 — AP completeness + cost distribution
# ══════════════════════════════════════════════════════════════
print("\n[1/4]  AP completeness & cost distributions")
print(DIVIDER)
header = f"{'Seed':>4} {'Iter':>4} {'#Sc':>4} {'Missing?':>10}  {'Mean':>8} {'Max':>8} {'p90':>8} {'p95':>8} {'Std':>8}"
print(header)
print("─" * len(header))

for seed in SEEDS:
    audit["ap_completeness"][seed] = {}
    # discover all AP iters for this seed
    ap_files = sorted(glob.glob(os.path.join(BASE_DIR, f"seed_{seed}_ap_iter*.json")))
    iters = [int(os.path.basename(f).split("ap_iter")[1].replace(".json","")) for f in ap_files]

    for it in sorted(iters):
        path = ap_json_path(seed, it)
        with open(path) as f:
            scenarios = json.load(f)

        n_sc   = len(scenarios)
        found_ids = {s["ball_id"] for s in scenarios}
        missing   = sorted(set(range(N_SAMPLES)) - found_ids)

        costs  = [s["max_cost"] for s in scenarios]
        c_mean = float(np.mean(costs))
        c_max  = float(np.max(costs))
        c_p90  = float(np.percentile(costs, 90))
        c_p95  = float(np.percentile(costs, 95))
        c_std  = float(np.std(costs))

        entry = {
            "n_scenarios": n_sc,
            "missing_ball_ids": missing,
            "complete": n_sc == N_SAMPLES,
            "ap_cost_mean": c_mean,
            "ap_cost_max":  c_max,
            "ap_cost_p90":  c_p90,
            "ap_cost_p95":  c_p95,
            "ap_cost_std":  c_std,
        }
        audit["ap_completeness"][seed][it] = entry

        miss_str = "OK" if not missing else f"MISSING {len(missing)}"
        print(f"{seed:>4} {it:>4} {n_sc:>4} {miss_str:>10}  "
              f"{c_mean:>8.4f} {c_max:>8.4f} {c_p90:>8.4f} {c_p95:>8.4f} {c_std:>8.4f}")

        if missing:
            audit["flags"]["incomplete_ap_files"].append(
                {"seed": seed, "iter": it, "n_scenarios": n_sc,
                 "missing_ball_ids": missing})

# ══════════════════════════════════════════════════════════════
# PASS 2 — AP MILP vs PyTorch cross-check
# ══════════════════════════════════════════════════════════════
print(f"\n[2/4]  AP MILP vs PyTorch cross-check  (tols: abs={ABS_TOL}, rel={REL_TOL})")
print(DIVIDER)
header2 = (f"{'Seed':>4} {'Iter':>4} {'#Sc':>4} {'MaxAbsErr':>12} {'MeanAbsErr':>12} "
           f"{'MaxRelErr':>12} {'#LargeAbs':>10} {'#LargeRel':>10}")
print(header2)
print("─" * len(header2))

for seed in SEEDS:
    audit["ap_crosscheck"][seed] = {}
    ap_files = sorted(glob.glob(os.path.join(BASE_DIR, f"seed_{seed}_ap_iter*.json")))
    iters = sorted([int(os.path.basename(f).split("ap_iter")[1].replace(".json",""))
                    for f in ap_files])

    for it in iters:
        ckpt_path = ckpt_for_ap(seed, it)
        if not os.path.exists(ckpt_path):
            audit["ap_crosscheck"][seed][it] = {"error": f"checkpoint missing: {ckpt_path}"}
            audit["flags"]["missing_files"].append(ckpt_path)
            print(f"{seed:>4} {it:>4}  — CHECKPOINT MISSING: {os.path.basename(ckpt_path)}")
            continue

        model, _ = load_model(ckpt_path)

        path = ap_json_path(seed, it)
        with open(path) as f:
            scenarios = json.load(f)

        abs_errs, rel_errs = [], []
        large_abs_cases, large_rel_cases = [], []

        for s in scenarios:
            bid   = s["ball_id"]
            wx1   = torch.tensor([s["worst_xi1"]], dtype=torch.float32)   # (1, R)
            wx2   = torch.tensor([s["worst_xi2"]], dtype=torch.float32)   # (1, R)
            milp_cost = float(s["max_cost"])

            with torch.no_grad():
                pt_cost = model.per_sample_cost(wx1, wx2).item()

            ae = abs(pt_cost - milp_cost)
            re = ae / (abs(milp_cost) + 1e-12)
            abs_errs.append(ae)
            rel_errs.append(re)

            if ae > ABS_TOL:
                large_abs_cases.append(
                    {"seed": seed, "iter": it, "ball_id": bid,
                     "milp_cost": milp_cost, "pt_cost": pt_cost,
                     "abs_err": ae, "rel_err": re})
            if re > REL_TOL:
                large_rel_cases.append(
                    {"seed": seed, "iter": it, "ball_id": bid,
                     "milp_cost": milp_cost, "pt_cost": pt_cost,
                     "abs_err": ae, "rel_err": re})

        max_ae  = float(np.max(abs_errs))   if abs_errs else float("nan")
        mean_ae = float(np.mean(abs_errs))  if abs_errs else float("nan")
        max_re  = float(np.max(rel_errs))   if rel_errs else float("nan")

        entry = {
            "n_checked":         len(abs_errs),
            "max_abs_err":       max_ae,
            "mean_abs_err":      mean_ae,
            "max_rel_err":       max_re,
            "n_large_abs":       len(large_abs_cases),
            "n_large_rel":       len(large_rel_cases),
            "large_abs_cases":   large_abs_cases[:10],   # store at most 10
            "large_rel_cases":   large_rel_cases[:10],
        }
        audit["ap_crosscheck"][seed][it] = entry
        audit["flags"]["large_mismatch_cases"].extend(large_abs_cases)

        flag = " *** LARGE MISMATCH ***" if large_abs_cases or large_rel_cases else ""
        print(f"{seed:>4} {it:>4} {len(abs_errs):>4} "
              f"{max_ae:>12.6f} {mean_ae:>12.6f} {max_re:>12.6f} "
              f"{len(large_abs_cases):>10} {len(large_rel_cases):>10}{flag}")

# ══════════════════════════════════════════════════════════════
# PASS 3 — Convergence quality: delta & per-ball distribution
# ══════════════════════════════════════════════════════════════
print(f"\n[3/4]  Convergence quality (AP cost distribution per iteration)")
print(DIVIDER)
print(f"{'Seed':>4} {'Iter':>4} {'AP_mean':>9} {'MP_loss':>9} "
      f"{'delta_pct':>10} {'AP_max':>9} {'AP_p90':>8} {'Conv?':>6}")
print("─" * 72)

for seed in SEEDS:
    audit["convergence_quality"][seed] = {}
    summ_p = summary_path(seed)
    if not os.path.exists(summ_p):
        audit["convergence_quality"][seed] = {"error": "summary missing"}
        continue
    with open(summ_p) as f:
        summ = json.load(f)

    traj = {r["iter"]: r for r in summ.get("ccg_trajectory", [])}

    ap_files = sorted(glob.glob(os.path.join(BASE_DIR, f"seed_{seed}_ap_iter*.json")))
    iters = sorted([int(os.path.basename(ff).split("ap_iter")[1].replace(".json",""))
                    for ff in ap_files])

    for it in iters:
        path = ap_json_path(seed, it)
        with open(path) as f:
            scenarios = json.load(f)
        costs = [s["max_cost"] for s in scenarios]
        ap_mean = float(np.mean(costs))
        ap_max  = float(np.max(costs))
        ap_p90  = float(np.percentile(costs, 90))

        mp_loss    = float("nan")
        delta_pct  = float("nan")
        converged  = False
        if it in traj:
            mp_loss   = traj[it]["mp_loss"]
            delta_pct = traj[it]["delta_pct"]
            converged = traj[it].get("converged", False)

        audit["convergence_quality"][seed][it] = {
            "ap_mean":    ap_mean,
            "ap_max":     ap_max,
            "ap_p90":     ap_p90,
            "mp_loss":    mp_loss,
            "delta_pct":  delta_pct,
            "converged":  converged,
        }

        conv_str = "YES" if converged else "no"
        print(f"{seed:>4} {it:>4} {ap_mean:>9.4f} {mp_loss:>9.4f} "
              f"{delta_pct:>10.2f} {ap_max:>9.4f} {ap_p90:>8.4f} {conv_str:>6}")

# ══════════════════════════════════════════════════════════════
# PASS 4 — Summary verification
# ══════════════════════════════════════════════════════════════
print(f"\n[4/4]  Summary file verification")
print(DIVIDER)
print(f"{'Seed':>4} {'Conv':>5} {'CCGit':>6} {'NomLoss':>9} {'MPLoss':>9} "
      f"{'Q10':>7} {'AllFiles?':>10} {'Time(h)':>8}")
print("─" * 72)

for seed in SEEDS:
    summ_p = summary_path(seed)
    if not os.path.exists(summ_p):
        audit["summary_verification"][seed] = {"error": "summary JSON missing"}
        audit["flags"]["missing_files"].append(summ_p)
        print(f"{seed:>4}  — SUMMARY JSON MISSING")
        continue

    with open(summ_p) as f:
        summ = json.load(f)

    # check all referenced files exist
    missing_ref = []
    nom_p = os.path.join(BASE_DIR, f"seed_{seed}_nominal.pth")
    if not os.path.exists(nom_p):
        missing_ref.append(nom_p)

    n_iters = summ.get("ccg_iters", 0)
    for it in range(1, n_iters + 1):
        pp = ap_json_path(seed, it)
        if not os.path.exists(pp):
            missing_ref.append(pp)
        # MP checkpoint exists for non-final converged iters
        if it < n_iters or not summ.get("converged", False):
            mp_p = os.path.join(BASE_DIR, f"seed_{seed}_mp_iter{it}.pth")
            if not os.path.exists(mp_p):
                missing_ref.append(mp_p)

    files_ok = "OK" if not missing_ref else f"MISS {len(missing_ref)}"
    time_h   = summ.get("total_time_s", 0) / 3600

    audit["summary_verification"][seed] = {
        "converged":      summ.get("converged"),
        "ccg_iters":      n_iters,
        "nominal_loss":   summ.get("nominal_loss"),
        "final_mp_loss":  summ.get("final_mp_loss"),
        "final_Q10":      summ.get("final_Q10"),
        "final_Q1r":      summ.get("final_Q1r"),
        "total_time_s":   summ.get("total_time_s"),
        "missing_files":  missing_ref,
    }
    if missing_ref:
        audit["flags"]["missing_files"].extend(missing_ref)

    conv_str = "YES" if summ.get("converged") else "NO"
    print(f"{seed:>4} {conv_str:>5} {n_iters:>6} "
          f"{summ.get('nominal_loss', float('nan')):>9.4f} "
          f"{summ.get('final_mp_loss', float('nan')):>9.4f} "
          f"{summ.get('final_Q10', float('nan')):>7.3f} "
          f"{files_ok:>10} {time_h:>8.2f}h")
    if missing_ref:
        for mf in missing_ref:
            print(f"       MISSING: {os.path.basename(mf)}")

# ══════════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════════
n_incomplete = len(audit["flags"]["incomplete_ap_files"])
n_mismatch   = len(audit["flags"]["large_mismatch_cases"])
n_missing    = len(set(audit["flags"]["missing_files"]))

# collect seeds / iters affected
affected_seeds = set()
for item in audit["flags"]["incomplete_ap_files"]:
    affected_seeds.add(item["seed"])
for item in audit["flags"]["large_mismatch_cases"]:
    affected_seeds.add(item["seed"])

reasons = []
if n_incomplete > 0:
    reasons.append(f"{n_incomplete} AP file(s) have < {N_SAMPLES} scenarios "
                   f"→ AP mean is computed on a subset (biased upward if hardest balls failed)")
if n_mismatch > 0:
    reasons.append(f"{n_mismatch} scenarios show |PyTorch cost - MILP cost| > {ABS_TOL} "
                   f"→ the MILP and the model being trained are inconsistent")
if n_missing > 0:
    reasons.append(f"{n_missing} referenced file(s) are missing from disk")

# Count total AP files
total_ap = sum(
    len(audit["ap_completeness"].get(s, {}))
    for s in SEEDS
)

if n_incomplete == 0 and n_mismatch == 0 and n_missing == 0:
    recommendation = "OK_TO_USE_EXISTING_RESULTS"
    reasons.append("All AP files have exactly 50 scenarios, "
                   "PyTorch costs match MILP costs within tolerance, "
                   "and all referenced files exist.")
elif len(affected_seeds) <= 3 and n_mismatch == 0:
    recommendation = "PARTIAL_RERUN_NEEDED"
elif n_mismatch > 0 or n_incomplete > total_ap * 0.3:
    recommendation = "FULL_RERUN_RECOMMENDED"
else:
    recommendation = "PARTIAL_RERUN_NEEDED"

audit["recommendation"] = {
    "verdict":          recommendation,
    "reasons":          reasons,
    "affected_seeds":   sorted(affected_seeds),
    "n_incomplete_ap":  n_incomplete,
    "n_mismatch":       n_mismatch,
    "n_missing_files":  n_missing,
}

print(f"\n{'═'*90}")
print(f"  RECOMMENDATION:  {recommendation}")
print(f"{'═'*90}")
for r in reasons:
    print(f"  • {r}")
if affected_seeds:
    print(f"  Affected seeds: {sorted(affected_seeds)}")
print(f"  Total AP files checked: {total_ap}")
print(f"  Incomplete AP files:    {n_incomplete}")
print(f"  Large-mismatch cases:   {n_mismatch}")
print(f"  Missing files:          {n_missing}")
print(f"{'═'*90}\n")

# ══════════════════════════════════════════════════════════════
# WRITE JSON
# ══════════════════════════════════════════════════════════════
# Convert int keys to strings for JSON
def intkeys(d):
    if isinstance(d, dict):
        return {str(k): intkeys(v) for k, v in d.items()}
    if isinstance(d, list):
        return [intkeys(i) for i in d]
    return d

out_path = os.path.join(BASE_DIR, "audit_existing_results.json")
with open(out_path, "w") as f:
    json.dump(intkeys(audit), f, indent=2)
print(f"Audit written to: {out_path}")
