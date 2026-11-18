"""
audit_all_remaining.py
======================
Generalized completeness + mean-gap audit for all remaining net16_v2 experiments:

  Group 1: R3/N200  — rho_sweep_net16_v2/rho_*/
  Group 2: R3/N800  — rho_sweep_net16_v2/rho_*/ + saainit_net16_v2/
  Group 3: R10/N50  — rho_sweep_net16_v2/rho_*/ + saainit_net16_v2/
  Group 4: R10/N200 — rho_sweep_net16_v2/rho_*/ + saainit_net16_v2/

For every AP JSON it checks:
  1. COMPLETENESS — are all N scenario slots filled?
  2. MEAN-GAP    — does the PyTorch forward pass agree with the MILP AP mean
                   closely enough that no convergence decision would change?

Notes on checkpoints:
  saainit dirs : AP iter 1 → seed_{s}_nominal.pth
  rho_sweep    : AP iter 1 → NO checkpoint saved (skip cross-check for iter 1)
  all dirs     : AP iter k>1 → seed_{s}_mp_iter{k-1}.pth

Outputs per group (written to the group's parent directory):
  audit_mean_gap_check.json   — full row-by-row results + summary

Usage:
  source /data/home/aviyac/ap_env/bin/activate
  python3 audit_all_remaining.py
"""

import os, sys, glob, json, re, time
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
BASE          = "/data/home/aviyac/retailers_problem"
C, H, B       = 0.25, 0.05, 0.5
SCS_EPS       = 1e-4
SCS_MAX       = 200_000
CONV_THRESH   = 1.0          # percent, same as CCG_CONV_THRESH_PCT

# Config groups: (label, parent_dir, R, N_SAMPLES)
GROUPS = [
    ("R3/N200",  f"{BASE}/R3/N200",  3,  200),
    ("R3/N800",  f"{BASE}/R3/N800",  3,  800),
    ("R10/N50",  f"{BASE}/R10/N50",  10,  50),
    ("R10/N200", f"{BASE}/R10/N200", 10, 200),
]

# ─────────────────────────────────────────────────────────────
# PARAMETERISED MODEL
# ─────────────────────────────────────────────────────────────
_model_classes = {}

def _make_model_class(R_val):
    """Create a Model class wired to R_val retailers (cached per R)."""
    if R_val in _model_classes:
        return _model_classes[R_val]

    class Model(nn.Module):
        def __init__(self_m):
            super().__init__()
            self_m.fc_layers = nn.Sequential(
                nn.Linear(R_val * 2 + 1, 16), nn.ReLU(),
                nn.Linear(16, 16),             nn.ReLU(),
                nn.Linear(16, R_val)
            )
            self_m.Q10 = nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self_m.Q1r = nn.Parameter(torch.zeros(R_val, dtype=torch.float32))
            x       = cp.Variable(R_val)
            z_p     = cp.Parameter(R_val)
            q10_p   = cp.Parameter(1)
            cons    = [cp.sum(x) <= q10_p + 1e-6, x >= 0]
            self_m.proj_layer = CvxpyLayer(
                cp.Problem(cp.Minimize(cp.sum_squares(x - z_p)), cons),
                parameters=[z_p, q10_p], variables=[x]
            )

        def per_sample_cost(self_m, xi1, xi2):
            """Batched forward pass.  xi1, xi2 : (N, R) float32 tensors."""
            N   = xi1.shape[0]
            q10 = torch.nn.functional.softplus(self_m.Q10)
            q1r = torch.nn.functional.softplus(self_m.Q1r)
            net_in = torch.cat([xi1,
                                q1r.unsqueeze(0).expand(N, -1),
                                q10.expand(N, 1)], dim=1)
            z   = self_m.fc_layers(net_in)
            q2, = self_m.proj_layer(z, q10.expand(N, 1),
                                    solver_args={"solve_method": "SCS",
                                                 "eps": SCS_EPS,
                                                 "max_iters": SCS_MAX})
            stage1 = C * (q10 + torch.sum(q1r)) + H * q10
            t1 = B * (xi2 + xi1 - q2 - q1r) - H * q2
            t2 = H * (q1r - xi1 - xi2)
            t3 = B * (xi1 - q1r) - H * xi2
            return stage1 + torch.sum(torch.maximum(torch.maximum(t1, t2), t3), dim=1)

    _model_classes[R_val] = Model
    return Model


def load_model(R, ckpt_path):
    M = _make_model_class(R)()
    saved = torch.load(ckpt_path, map_location="cpu")
    M.load_state_dict(saved["model_state_dict"], strict=False)
    M.eval()
    return M


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_ap(path):
    with open(path) as f:
        return json.load(f)


def load_summary_traj(exp_dir, seed):
    p = os.path.join(exp_dir, f"seed_{seed}_summary.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    return {r["iter"]: r for r in d.get("ccg_trajectory", [])}


def ckpt_path_for(exp_dir, seed, ap_iter, is_saainit):
    """Return the checkpoint path to use for a given AP iteration."""
    if ap_iter == 1:
        if is_saainit:
            return os.path.join(exp_dir, f"seed_{seed}_nominal.pth")
        else:
            return None   # rho_sweep: no nominal saved
    return os.path.join(exp_dir, f"seed_{seed}_mp_iter{ap_iter - 1}.pth")


def completeness_check(scenarios, expected_N):
    found = {s["ball_id"] for s in scenarios}
    missing = [i for i in range(expected_N) if i not in found]
    incomplete = [s["ball_id"] for s in scenarios
                  if not s.get("worst_xi1") or len(s["worst_xi1"]) == 0]
    return {
        "n_found":     len(scenarios),
        "expected_N":  expected_N,
        "n_missing":   len(missing),
        "n_incomplete": len(incomplete),
        "ok":          len(missing) == 0 and len(incomplete) == 0
                        and len(scenarios) == expected_N
    }


# ─────────────────────────────────────────────────────────────
# DISCOVER EXPERIMENT DIRECTORIES
# ─────────────────────────────────────────────────────────────

def discover_exp_dirs(parent_dir):
    """
    Returns list of (exp_dir, is_saainit) tuples under parent_dir.
    Covers  *_net16_v2/  (saainit_net16_v2) and
            rho_sweep_net16_v2/rho_*/
    """
    results = []

    # saainit_net16_v2 (flat dir)
    saainit_dir = os.path.join(parent_dir, "saainit_net16_v2")
    if os.path.isdir(saainit_dir) and glob.glob(os.path.join(saainit_dir, "seed_*_ap_iter*.json")):
        results.append((saainit_dir, True))

    # rho_sweep_net16_v2/rho_*  (one subdir per rho value)
    sweep_root = os.path.join(parent_dir, "rho_sweep_net16_v2")
    if os.path.isdir(sweep_root):
        rho_dirs = sorted(glob.glob(os.path.join(sweep_root, "rho_*")))
        for rd in rho_dirs:
            if os.path.isdir(rd) and glob.glob(os.path.join(rd, "seed_*_ap_iter*.json")):
                results.append((rd, False))

    return results


# ─────────────────────────────────────────────────────────────
# AUDIT ONE EXPERIMENT DIRECTORY
# ─────────────────────────────────────────────────────────────

def audit_exp_dir(exp_dir, is_saainit, R, expected_N, label=""):
    """
    Run completeness + mean-gap check for every (seed, iter) AP file in exp_dir.
    Returns list of row dicts.
    """
    rows = []
    tag  = os.path.basename(exp_dir)

    # Discover seeds
    ap_files = sorted(glob.glob(os.path.join(exp_dir, "seed_*_ap_iter*.json")))
    if not ap_files:
        print(f"  [{label}/{tag}] No AP files found – skipping.")
        return rows

    seed_iter_pairs = []
    for fp in ap_files:
        m = re.match(r"seed_(\d+)_ap_iter(\d+)\.json", os.path.basename(fp))
        if m:
            seed_iter_pairs.append((int(m.group(1)), int(m.group(2)), fp))
    seed_iter_pairs.sort()

    seeds_seen = sorted({s for s, _, _ in seed_iter_pairs})

    # Load all trajectories upfront
    trajs = {s: load_summary_traj(exp_dir, s) for s in seeds_seen}

    # Process each (seed, iter)
    for seed, ap_iter, ap_fp in seed_iter_pairs:
        # ── completeness ──────────────────────────────────────
        scenarios = load_ap(ap_fp)
        comp      = completeness_check(scenarios, expected_N)

        # ── mean-gap cross-check ──────────────────────────────
        ckpt = ckpt_path_for(exp_dir, seed, ap_iter, is_saainit)

        if ckpt is None:
            # rho_sweep iter 1: no checkpoint available
            gap_status  = "SKIPPED_NO_CKPT"
            ap_milp_mean = ap_torch_mean = abs_err = rel_err_pct = float("nan")
            mp_loss = delta_pct_milp = delta_pct_torch = float("nan")
            conv_milp = conv_torch = conv_changed = False
        elif not os.path.exists(ckpt):
            gap_status  = "SKIPPED_CKPT_MISSING"
            ap_milp_mean = ap_torch_mean = abs_err = rel_err_pct = float("nan")
            mp_loss = delta_pct_milp = delta_pct_torch = float("nan")
            conv_milp = conv_torch = conv_changed = False
        else:
            # Run batched forward pass
            model     = load_model(R, ckpt)
            xi1 = torch.tensor([s["worst_xi1"] for s in scenarios], dtype=torch.float32)
            xi2 = torch.tensor([s["worst_xi2"] for s in scenarios], dtype=torch.float32)
            with torch.no_grad():
                tc = model.per_sample_cost(xi1, xi2).numpy()

            ap_milp_mean  = float(np.mean([s["max_cost"] for s in scenarios]))
            ap_torch_mean = float(np.mean(tc))
            abs_err       = abs(ap_milp_mean - ap_torch_mean)
            rel_err_pct   = abs_err / (abs(ap_milp_mean) + 1e-12) * 100

            # Convergence check
            traj_rec = trajs[seed].get(ap_iter, {})
            mp_loss  = traj_rec.get("mp_loss", float("nan"))
            if not np.isnan(mp_loss) and mp_loss != 0:
                delta_pct_milp  = (ap_milp_mean  - mp_loss) / abs(mp_loss) * 100
                delta_pct_torch = (ap_torch_mean - mp_loss) / abs(mp_loss) * 100
            else:
                delta_pct_milp = delta_pct_torch = float("nan")

            conv_milp    = (not np.isnan(delta_pct_milp))  and delta_pct_milp  < CONV_THRESH
            conv_torch   = (not np.isnan(delta_pct_torch)) and delta_pct_torch < CONV_THRESH
            conv_changed = (conv_milp != conv_torch) and (ap_iter > 1)
            gap_status   = "OK"

        rows.append({
            "exp_dir":         exp_dir,
            "is_saainit":      is_saainit,
            "seed":            seed,
            "iter":            ap_iter,
            # completeness
            "comp_ok":         comp["ok"],
            "n_found":         comp["n_found"],
            "n_missing":       comp["n_missing"],
            "n_incomplete":    comp["n_incomplete"],
            # mean-gap
            "gap_status":      gap_status,
            "ap_mean_milp":    ap_milp_mean,
            "ap_mean_torch":   ap_torch_mean,
            "abs_err":         abs_err,
            "rel_err_pct":     rel_err_pct,
            "mp_loss":         mp_loss,
            "delta_pct_milp":  delta_pct_milp,
            "delta_pct_torch": delta_pct_torch,
            "conv_milp":       conv_milp,
            "conv_torch":      conv_torch,
            "conv_changed":    conv_changed,
        })

    return rows


# ─────────────────────────────────────────────────────────────
# SUMMARISE + WRITE OUTPUT
# ─────────────────────────────────────────────────────────────

def nan_safe(vals, fn):
    v = [x for x in vals if not np.isnan(x)]
    return fn(v) if v else float("nan")


def summarise_and_save(group_label, parent_dir, all_rows):
    DIV = "─" * 100

    # Filter to rows that have a gap check
    gap_rows = [r for r in all_rows if r["gap_status"] == "OK"]
    comp_bad = [r for r in all_rows if not r["comp_ok"]]
    conv_chg = [r for r in gap_rows if r["conv_changed"]]

    abs_errs = [r["abs_err"]     for r in gap_rows]
    rel_errs = [r["rel_err_pct"] for r in gap_rows]

    print(f"\n{DIV}")
    print(f"GROUP: {group_label}   ({len(all_rows)} AP files, "
          f"{len(gap_rows)} cross-checked, "
          f"{len(all_rows)-len(gap_rows)} skipped)")
    print(DIV)

    # Completeness
    if comp_bad:
        print(f"  COMPLETENESS ISSUES ({len(comp_bad)}):")
        for r in comp_bad:
            print(f"    {os.path.relpath(r['exp_dir'], parent_dir)!s:35s}  "
                  f"seed={r['seed']} iter={r['iter']}  "
                  f"found={r['n_found']} missing={r['n_missing']}")
    else:
        print(f"  Completeness : ALL OK ({len(all_rows)} files, all full)")

    # Mean-gap summary
    if gap_rows:
        print(f"  AP-mean abs error  : mean={nan_safe(abs_errs,np.mean):.5f}  "
              f"max={nan_safe(abs_errs,np.max):.5f}  "
              f"p95={nan_safe(abs_errs, lambda v: np.percentile(v,95)):.5f}")
        print(f"  AP-mean rel error% : mean={nan_safe(rel_errs,np.mean):.4f}%  "
              f"max={nan_safe(rel_errs,np.max):.4f}%")
        print(f"  Conv decisions changed: {len(conv_chg)}")
    else:
        print("  Mean-gap check: no rows cross-checked (all checkpoints missing/skipped)")

    # Verdict
    n_comp_bad = len(comp_bad)
    n_conv_chg = len(conv_chg)
    max_rel    = nan_safe(rel_errs, np.max)

    if n_comp_bad == 0 and n_conv_chg == 0 and (np.isnan(max_rel) or max_rel < 0.5):
        verdict = "OK_TO_USE_EXISTING_RESULTS"
    elif n_conv_chg > 0:
        verdict = "CONVERGENCE_CHANGES_DETECTED"
    elif n_comp_bad > 0:
        verdict = "COMPLETENESS_ISSUES"
    else:
        verdict = "REVIEW_RECOMMENDED"

    print(f"\n  VERDICT: {verdict}")

    # Worst 5 by abs error
    if gap_rows:
        print(f"\n  Worst 5 by AP-mean abs error:")
        for r in sorted(gap_rows, key=lambda x: x["abs_err"], reverse=True)[:5]:
            reldir = os.path.relpath(r['exp_dir'], parent_dir)
            print(f"    {reldir!s:35s}  seed={r['seed']} iter={r['iter']}  "
                  f"absErr={r['abs_err']:.5f}  relErr={r['rel_err_pct']:.4f}%")

    # Save JSON
    out_path = os.path.join(parent_dir, "audit_mean_gap_check.json")
    summary = {
        "group":              group_label,
        "n_total_ap_files":   len(all_rows),
        "n_cross_checked":    len(gap_rows),
        "n_skipped":          len(all_rows) - len(gap_rows),
        "n_comp_issues":      n_comp_bad,
        "n_conv_changed":     n_conv_chg,
        "max_abs_err":        nan_safe(abs_errs, np.max),
        "mean_abs_err":       nan_safe(abs_errs, np.mean),
        "max_rel_err_pct":    nan_safe(rel_errs, np.max),
        "mean_rel_err_pct":   nan_safe(rel_errs, np.mean),
        "verdict":            verdict,
        "rows":               all_rows,
    }
    # Convert any numpy floats to Python floats for JSON serialisation
    def _fix(obj):
        if isinstance(obj, dict):
            return {k: _fix(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_fix(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj
    with open(out_path, "w") as f:
        json.dump(_fix(summary), f, indent=2)
    print(f"\n  Saved → {out_path}")
    return verdict


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    overall_results = {}
    t_start = time.time()

    for group_label, parent_dir, R, expected_N in GROUPS:
        print(f"\n{'='*100}")
        print(f"PROCESSING GROUP: {group_label}  (R={R}, N={expected_N})")
        print(f"{'='*100}")

        exp_dirs = discover_exp_dirs(parent_dir)
        if not exp_dirs:
            print(f"  No experiment directories found under {parent_dir}  – skipping.")
            continue

        print(f"  Found {len(exp_dirs)} experiment directory/ies:")
        for ed, is_sa in exp_dirs:
            relp = os.path.relpath(ed, parent_dir)
            print(f"    {'[saainit]' if is_sa else '[rho_swp]':12s}  {relp}")

        all_rows = []
        for idx, (exp_dir, is_saainit) in enumerate(exp_dirs):
            relp = os.path.relpath(exp_dir, parent_dir)
            ap_count = len(glob.glob(os.path.join(exp_dir, "seed_*_ap_iter*.json")))
            print(f"\n  [{idx+1}/{len(exp_dirs)}] {relp}  ({ap_count} AP files) ...", flush=True)
            t0 = time.time()
            rows = audit_exp_dir(exp_dir, is_saainit, R, expected_N,
                                 label=f"{group_label}/{relp}")
            dt = time.time() - t0
            all_rows.extend(rows)
            gap_ok = sum(1 for r in rows if r["gap_status"] == "OK")
            chg    = sum(1 for r in rows if r["conv_changed"])
            print(f"    done in {dt:.1f}s  |  {len(rows)} pairs, "
                  f"{gap_ok} cross-checked, {chg} conv-change", flush=True)

        verdict = summarise_and_save(group_label, parent_dir, all_rows)
        overall_results[group_label] = verdict

    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"OVERALL SUMMARY  (total elapsed {elapsed/60:.1f} min)")
    print(f"{'='*100}")
    all_ok = True
    for label, verdict in overall_results.items():
        flag = "" if verdict == "OK_TO_USE_EXISTING_RESULTS" else "  <-- ATTENTION"
        print(f"  {label:<12s}  {verdict}{flag}")
        if verdict != "OK_TO_USE_EXISTING_RESULTS":
            all_ok = False
    print()
    if all_ok:
        print("  → All groups OK_TO_USE_EXISTING_RESULTS")
    else:
        print("  → Some groups need attention – see per-group JSON files.")


if __name__ == "__main__":
    main()
