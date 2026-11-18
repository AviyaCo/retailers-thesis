"""
collect_timing.py  —  Summarise timing results from timing_test/
Reads: timing_test/rho_XXX/seed_Y_timing.json

Prints per-run times and averages for:
  - SAA LP solve
  - Nominal training
  - CCG total
  - Total wall time
"""

import os, json, glob
import numpy as np

BASE = "/data/home/aviyac/retailers_problem/R3/N50/timing_test"

files = sorted(glob.glob(os.path.join(BASE, "rho_*", "seed_*_timing.json")))
if not files:
    print("No timing.json files found yet.")
    exit(0)

rows = []
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    rows.append(d)

# ── per-run table ────────────────────────────────────────────────────────────
print(f"{'rho':>8}  {'seed':>4}  {'saa_s':>6}  {'nom_min':>7}  "
      f"{'ccg_min':>7}  {'ccg_it':>6}  {'total_min':>9}  conv")
print("-" * 65)
for d in sorted(rows, key=lambda x: (x['rho'], x['seed'])):
    print(f"{d['rho']:8.3f}  {d['seed']:4d}  "
          f"{d['saa_time_s']:6.1f}  "
          f"{d['nominal_time_s']/60:7.1f}  "
          f"{d['ccg_time_s']/60:7.1f}  "
          f"{d['ccg_iters']:6d}  "
          f"{d['total_time_s']/60:9.1f}  "
          f"{'Y' if d['converged'] else 'N'}")

# ── summary by rho ────────────────────────────────────────────────────────────
print("\nSummary by rho (mean over seeds):")
print(f"{'rho':>8}  {'n':>2}  {'nom_min':>7}  {'ccg_min':>7}  {'total_min':>9}  {'ccg_iters':>9}")
print("-" * 55)
for rho in sorted(set(d['rho'] for d in rows)):
    sub = [d for d in rows if d['rho'] == rho]
    nm  = np.mean([d['nominal_time_s'] for d in sub]) / 60
    cm  = np.mean([d['ccg_time_s']     for d in sub]) / 60
    tm  = np.mean([d['total_time_s']   for d in sub]) / 60
    itr = np.mean([d['ccg_iters']      for d in sub])
    print(f"{rho:8.3f}  {len(sub):2d}  {nm:7.1f}  {cm:7.1f}  {tm:9.1f}  {itr:9.1f}")

# ── overall ───────────────────────────────────────────────────────────────────
print(f"\nOverall ({len(rows)} runs):")
print(f"  Mean nominal:  {np.mean([d['nominal_time_s'] for d in rows])/60:.1f} min")
print(f"  Mean CCG:      {np.mean([d['ccg_time_s']     for d in rows])/60:.1f} min")
print(f"  Mean total:    {np.mean([d['total_time_s']   for d in rows])/60:.1f} min")
