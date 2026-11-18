"""
audit_mean_gap_check.py
=======================
Targeted audit: does the SCS approximation error materially shift the AP *mean*,
and would any convergence decision have changed if we used PyTorch-evaluated costs
instead of MILP-reported costs for the AP mean?

Outputs:
  - terminal table
  - audit_R3_N50_mean_gap_check.json
"""

import os, sys, json, glob
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR  = "/data/home/aviyac/retailers_problem/R3/N50/saainit_net16_v2"
SEEDS     = list(range(10))
R         = 3
c, h, b   = 0.25, 0.05, 0.5
SCS_EPS   = 1e-4
SCS_MAX   = 200_000
CONV_THRESH_PCT = 1.0   # same as in run_all_net16_v2.py

# ─────────────────────────────────────────────────────────────
# MODEL
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
    m     = Model()
    m.load_state_dict(saved["model_state_dict"], strict=False)
    m.eval()
    return m


def ckpt_for_ap(seed, ap_iter):
    if ap_iter == 1:
        return os.path.join(BASE_DIR, f"seed_{seed}_nominal.pth")
    return os.path.join(BASE_DIR, f"seed_{seed}_mp_iter{ap_iter-1}.pth")


def ap_json_path(seed, it):
    return os.path.join(BASE_DIR, f"seed_{seed}_ap_iter{it}.json")

# ─────────────────────────────────────────────────────────────
# LOAD SUMMARY TRAJECTORIES ONCE
# ─────────────────────────────────────────────────────────────
def load_trajectory(seed):
    """Returns dict: iter -> {mp_loss, delta_pct, converged}"""
    p = os.path.join(BASE_DIR, f"seed_{seed}_summary.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        summ = json.load(f)
    return {r["iter"]: r for r in summ.get("ccg_trajectory", [])}

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
DIV = "─" * 110

print(DIV)
print("AUDIT: AP mean gap check — does SCS error shift AP_mean enough to change convergence?")
print(DIV)

header = (f"{'Seed':>4} {'Iter':>4} | "
          f"{'AP_mean_MILP':>13} {'AP_mean_Torch':>14} {'AbsErr':>8} {'RelErr%':>8} | "
          f"{'MP_loss':>9} {'dPct_MILP':>10} {'dPct_Torch':>11} | "
          f"{'ConvChng?':>10}")
print(header)
print("─" * len(header))

all_rows = []
conv_changes = []

for seed in SEEDS:
    traj = load_trajectory(seed)
    ap_files = sorted(glob.glob(os.path.join(BASE_DIR, f"seed_{seed}_ap_iter*.json")))
    iters = sorted([int(os.path.basename(f).split("ap_iter")[1].replace(".json",""))
                    for f in ap_files])

    for it in iters:
        ckpt_path = ckpt_for_ap(seed, it)
        if not os.path.exists(ckpt_path):
            print(f"{seed:>4} {it:>4} | CHECKPOINT MISSING: {os.path.basename(ckpt_path)}")
            continue

        model = load_model(ckpt_path)

        with open(ap_json_path(seed, it)) as f:
            scenarios = json.load(f)

        milp_costs  = []
        torch_costs = []
        for s in scenarios:
            milp_costs.append(float(s["max_cost"]))
            wx1 = torch.tensor([s["worst_xi1"]], dtype=torch.float32)
            wx2 = torch.tensor([s["worst_xi2"]], dtype=torch.float32)
            with torch.no_grad():
                torch_costs.append(model.per_sample_cost(wx1, wx2).item())

        ap_milp  = float(np.mean(milp_costs))
        ap_torch = float(np.mean(torch_costs))
        abs_err  = abs(ap_milp - ap_torch)
        rel_err  = abs_err / (abs(ap_milp) + 1e-12) * 100

        # MP loss and delta percentages
        traj_rec = traj.get(it, {})
        mp_loss  = traj_rec.get("mp_loss", float("nan"))

        if not np.isnan(mp_loss) and mp_loss != 0:
            d_milp  = (ap_milp  - mp_loss) / abs(mp_loss) * 100
            d_torch = (ap_torch - mp_loss) / abs(mp_loss) * 100
        else:
            d_milp = d_torch = float("nan")

        # Convergence change check (only meaningful for it > 1, which is when
        # the code's convergence check fires)
        conv_milp  = (not np.isnan(d_milp))  and (d_milp  < CONV_THRESH_PCT)
        conv_torch = (not np.isnan(d_torch)) and (d_torch < CONV_THRESH_PCT)
        chng = (conv_milp != conv_torch) and (it > 1)
        chng_str = "*** YES ***" if chng else "no"

        row = {
            "seed": seed, "iter": it,
            "ap_mean_milp":    ap_milp,
            "ap_mean_torch":   ap_torch,
            "ap_mean_abs_err": abs_err,
            "ap_mean_rel_err_pct": rel_err,
            "mp_loss":         mp_loss,
            "delta_pct_milp":  d_milp,
            "delta_pct_torch": d_torch,
            "conv_milp":       conv_milp,
            "conv_torch":      conv_torch,
            "conv_decision_changed": chng,
        }
        all_rows.append(row)
        if chng:
            conv_changes.append(row)

        mp_str = f"{mp_loss:9.4f}" if not np.isnan(mp_loss) else f"{'N/A':>9}"
        dm_str = f"{d_milp:+10.3f}" if not np.isnan(d_milp) else f"{'N/A':>10}"
        dt_str = f"{d_torch:+11.3f}" if not np.isnan(d_torch) else f"{'N/A':>11}"
        print(f"{seed:>4} {it:>4} | "
              f"{ap_milp:>13.4f} {ap_torch:>14.4f} {abs_err:>8.5f} {rel_err:>7.4f}% | "
              f"{mp_str} {dm_str} {dt_str} | "
              f"{chng_str:>10}")

# ─────────────────────────────────────────────────────────────
# WORST 10 by AP_mean_abs_error
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("WORST 10 seed/iter pairs by AP_mean_abs_error:")
print(f"{'─'*60}")
sorted_rows = sorted(all_rows, key=lambda r: r["ap_mean_abs_err"], reverse=True)
print(f"{'Seed':>4} {'Iter':>4}  {'AbsErr':>9}  {'RelErr%':>8}  {'AP_MILP':>10}  {'AP_Torch':>10}")
print("─" * 55)
for row in sorted_rows[:10]:
    print(f"{row['seed']:>4} {row['iter']:>4}  "
          f"{row['ap_mean_abs_err']:>9.5f}  {row['ap_mean_rel_err_pct']:>7.4f}%  "
          f"{row['ap_mean_milp']:>10.4f}  {row['ap_mean_torch']:>10.4f}")

# ─────────────────────────────────────────────────────────────
# CONVERGENCE DECISION CHANGES
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"Convergence decision changes (MILP vs Torch, threshold={CONV_THRESH_PCT}%):")
print(f"{'─'*60}")
if conv_changes:
    for row in conv_changes:
        direction = ("MILP→conv, Torch→NOT conv"
                     if row["conv_milp"] and not row["conv_torch"]
                     else "Torch→conv, MILP→NOT conv")
        print(f"  seed={row['seed']}, iter={row['iter']}: {direction} "
              f"(d_MILP={row['delta_pct_milp']:+.3f}%, d_Torch={row['delta_pct_torch']:+.3f}%)")
else:
    print("  None — convergence decisions are identical under MILP and Torch AP means.")

# ─────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────
abs_errs = [r["ap_mean_abs_err"] for r in all_rows]
rel_errs = [r["ap_mean_rel_err_pct"] for r in all_rows]

print(f"\n{'─'*60}")
print("AP mean error summary across all seed/iter pairs:")
print(f"{'─'*60}")
print(f"  N pairs          : {len(all_rows)}")
print(f"  Mean abs error   : {np.mean(abs_errs):.5f}")
print(f"  Median abs error : {np.median(abs_errs):.5f}")
print(f"  Max abs error    : {np.max(abs_errs):.5f}")
print(f"  p95 abs error    : {np.percentile(abs_errs, 95):.5f}")
print(f"  Mean rel error%  : {np.mean(rel_errs):.4f}%")
print(f"  Max rel error%   : {np.max(rel_errs):.4f}%")
print(f"  Conv decisions changed: {len(conv_changes)}")

# ─────────────────────────────────────────────────────────────
# FINAL RECOMMENDATION
# ─────────────────────────────────────────────────────────────
n_chng   = len(conv_changes)
max_abs  = float(np.max(abs_errs))
max_rel  = float(np.max(rel_errs))

reasons = []
if n_chng == 0:
    reasons.append("No convergence decisions change when using PyTorch AP means instead of MILP AP means.")
else:
    seeds_affected = sorted({r['seed'] for r in conv_changes})
    reasons.append(f"{n_chng} convergence decision(s) change: seeds {seeds_affected}.")

reasons.append(f"Max AP-mean absolute error: {max_abs:.5f} (on costs ~17-23, i.e. "
               f"{max_abs/20*100:.3f}% of typical cost).")
reasons.append(f"Max AP-mean relative error: {max_rel:.4f}%.")

if n_chng == 0 and max_rel < 0.5:
    verdict = "OK_TO_USE_EXISTING_RESULTS"
    reasons.append("SCS averaging effect keeps AP-mean errors well below the 1% convergence threshold.")
elif n_chng > 0 and len({r['seed'] for r in conv_changes}) <= 3:
    verdict = "PARTIAL_RERUN_NEEDED"
else:
    verdict = "FULL_RERUN_RECOMMENDED"

print(f"\n{'═'*70}")
print(f"  RECOMMENDATION:  {verdict}")
print(f"{'═'*70}")
for r in reasons:
    print(f"  • {r}")
print(f"{'═'*70}\n")

# ─────────────────────────────────────────────────────────────
# SAVE JSON
# ─────────────────────────────────────────────────────────────
def nan_safe(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: nan_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [nan_safe(i) for i in x]
    return x

out = {
    "config": {"SEEDS": SEEDS, "R": R, "CONV_THRESH_PCT": CONV_THRESH_PCT},
    "per_iter_rows":     [nan_safe(r) for r in all_rows],
    "worst10_by_abs_err": [nan_safe(r) for r in sorted_rows[:10]],
    "convergence_changes": [nan_safe(r) for r in conv_changes],
    "summary_stats": {
        "n_pairs":             len(all_rows),
        "mean_abs_err":        float(np.mean(abs_errs)),
        "median_abs_err":      float(np.median(abs_errs)),
        "max_abs_err":         float(np.max(abs_errs)),
        "p95_abs_err":         float(np.percentile(abs_errs, 95)),
        "mean_rel_err_pct":    float(np.mean(rel_errs)),
        "max_rel_err_pct":     float(np.max(rel_errs)),
        "n_conv_changes":      n_chng,
    },
    "recommendation": {
        "verdict":  verdict,
        "reasons":  reasons,
    },
}
out_path = os.path.join(BASE_DIR, "audit_R3_N50_mean_gap_check.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"Audit written to: {out_path}")
