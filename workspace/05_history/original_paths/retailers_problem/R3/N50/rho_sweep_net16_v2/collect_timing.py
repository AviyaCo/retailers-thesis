"""collect_timing.py -- Print per-seed timing from the timing test run (seeds 0-2, --force).
"""
import os, json, glob, numpy as np

base  = "/data/home/aviyac/retailers_problem/R3/N50/rho_sweep_net16_v2"
rhos  = [0.001, 0.300, 1.500]

def rho_tag(rho):
    s = f"{rho:.6f}".rstrip("0")
    parts = s.split(".")
    if len(parts) > 1 and len(parts[1]) < 3:
        s += "0" * (3 - len(parts[1]))
    return s.replace(".", "p")

print(f"\n{'Rho':>8}  {'Seed':>4}  {'CCG':>4}  {'Total':>8}  {'AP/iter':>8}  {'MP/iter':>8}")
print("-" * 55)

for rho in rhos:
    tag   = rho_tag(rho)
    paths = sorted(glob.glob(os.path.join(base, f"rho_{tag}", "seed_[012]_summary.json")))
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if not d.get("completed"):
            print(f"  {rho:>8}  seed={d['seed']}  NOT DONE")
            continue
        traj  = d["ccg_trajectory"]
        n_it  = d["ccg_iters"]
        total = d["total_time_s"]
        ap_times = [t.get("ap_time_s", 0) for t in traj]
        mp_times = [t.get("mp_time_s", 0) for t in traj if t.get("mp_time_s", 0) > 0]
        ap_mean  = np.mean(ap_times) if ap_times else 0
        mp_mean  = np.mean(mp_times) if mp_times else 0
        print(f"  {rho:>8g}  {d['seed']:>4d}  {n_it:>4d}  "
              f"{total:>7.0f}s  {ap_mean:>7.0f}s  {mp_mean:>7.0f}s")

print()
print("Summary by rho (mean over seeds 0-2):")
print(f"{'Rho':>8}  {'CCG mean':>9}  {'Total mean':>11}  {'AP/iter':>9}  {'MP/iter':>9}")
print("-" * 55)
for rho in rhos:
    tag   = rho_tag(rho)
    paths = sorted(glob.glob(os.path.join(base, f"rho_{tag}", "seed_[012]_summary.json")))
    totals, ccgs, aps, mps = [], [], [], []
    for p in paths:
        with open(p) as f: d = json.load(f)
        if not d.get("completed"): continue
        traj = d["ccg_trajectory"]
        totals.append(d["total_time_s"])
        ccgs.append(d["ccg_iters"])
        aps += [t.get("ap_time_s", 0) for t in traj]
        mps += [t.get("mp_time_s", 0) for t in traj if t.get("mp_time_s", 0) > 0]
    if totals:
        print(f"  {rho:>8g}  {np.mean(ccgs):>9.1f}  {np.mean(totals):>10.0f}s  "
              f"{np.mean(aps):>8.0f}s  {np.mean(mps):>8.0f}s")
