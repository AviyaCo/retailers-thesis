#!/usr/bin/env python3
"""
relaunch_incomplete.py  --  Relaunch missing + non-converged R10 N50 jobs
=========================================================================
Deletes summaries for completed-but-not-converged cases, then spawns all
(rho, seed) pairs that don't yet have a completed summary, with max_iter=25.
"""
import os, json, glob, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

PYTHON  = "/data/home/aviyac/ap_env/bin/python3"
BASE    = "/data/home/aviyac/retailers_problem/R10/N50/rho_sweep_net16_v2"
SCRIPT  = f"{BASE}/run_rho_seed_net16_v2_N50_R10.py"
LOG_DIR = f"{BASE}/logs"
MAX_ITER    = 25
MAX_PARALLEL = int(sys.argv[1]) if len(sys.argv) > 1 else 20

RHOS = [0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150, 0.200,
        0.300, 0.400, 0.450, 0.500, 0.600, 0.750, 0.800, 1.000, 1.500]
SEEDS = list(range(10))

os.makedirs(LOG_DIR, exist_ok=True)

def rho_tag(rho):
    s = f"{rho:.6f}".rstrip("0")
    parts = s.split(".")
    if len(parts[1]) < 3:
        parts[1] = parts[1].ljust(3, "0")
    return ".".join(parts).replace(".", "p")

# ── Remove completed-but-not-converged summaries ──────────────────────────────
print("Scanning for non-converged summaries to remove...")
removed = 0
for p in glob.glob(f"{BASE}/rho_*/seed_*_summary.json"):
    try:
        with open(p) as f:
            d = json.load(f)
        if d.get("completed", False) and not d.get("converged", False):
            os.remove(p)
            print(f"  removed {p}")
            removed += 1
    except Exception:
        pass
print(f"Removed {removed} non-converged summaries.\n")

# ── Build list of (rho, seed) that need running ───────────────────────────────
jobs = []
for rho in RHOS:
    tag = rho_tag(rho)
    for seed in SEEDS:
        summ = f"{BASE}/rho_{tag}/seed_{seed}_summary.json"
        done = False
        if os.path.exists(summ):
            try:
                with open(summ) as f:
                    d = json.load(f)
                done = d.get("completed", False)
            except Exception:
                pass
        if not done:
            jobs.append((rho, seed, tag))

print(f"Jobs to run: {len(jobs)} (skipping {170 - len(jobs)} already completed)\n")

def run_job(rho, seed, tag):
    log_path = f"{LOG_DIR}/relaunch_rho{rho}_seed{seed}.txt"
    with open(log_path, "w") as lf:
        result = subprocess.run(
            [PYTHON, SCRIPT, "--rho", str(rho), "--seed", str(seed),
             "--max_iter", str(MAX_ITER)],
            stdout=lf, stderr=lf
        )
    summ = f"{BASE}/rho_{tag}/seed_{seed}_summary.json"
    conv = False
    if os.path.exists(summ):
        try:
            with open(summ) as f:
                d = json.load(f)
            conv = d.get("converged", False)
        except Exception:
            pass
    return rho, seed, result.returncode, conv

# ── Launch with thread pool ────────────────────────────────────────────────────
done_count = 0
failed = []
t0 = time.time()
with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
    futures = {pool.submit(run_job, rho, seed, tag): (rho, seed)
               for rho, seed, tag in jobs}
    for fut in as_completed(futures):
        rho, seed, rc, conv = fut.result()
        done_count += 1
        elapsed = (time.time() - t0) / 60
        status = "converged" if conv else ("FAILED(rc)" if rc != 0 else "NOT_CONV")
        print(f"[{done_count:3d}/{len(jobs)}  {elapsed:5.1f}min]  "
              f"rho={rho:.3f} seed={seed}  {status}")
        if not conv:
            failed.append((rho, seed))

print(f"\n=== Done ===")
print(f"Converged: {len(jobs)-len(failed)}/{len(jobs)}")
if failed:
    print(f"Still not converged ({len(failed)}):")
    for rho, seed in sorted(failed):
        print(f"  rho={rho:.3f} seed={seed}")
