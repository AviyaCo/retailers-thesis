"""
sro_fa_sweep.py  —  SRO-FA sweep for R=3, N=200
=================================================
Same method as R3/N50/sro_fa/sro_fa_sweep.py but with N_TRAIN=200 and the
rho grid matching the R3/N200 neural sweep.
"""

import os, json, time
import numpy as np
from scipy.stats import truncnorm
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

# ── problem constants ─────────────────────────────────────────────────────────
R         = 3
N_TRAIN   = 200
N_TEST    = 10000
SEED_TEST = 9999
c, h, b   = 0.25, 0.05, 0.5

# Same rho grid as neural R3/N200 sweep
RHOS = [
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2,
    0.3, 0.4, 0.45, 0.5, 0.6, 0.75, 0.8, 1.0, 1.5,
]
SEEDS = list(range(10))

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(OUT_DIR, "sro_fa_results_N200_R3.json")

# ── data generation ───────────────────────────────────────────────────────────
def gen(N, seed):
    np.random.seed(seed)
    a   = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a2  = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2   # (N, R) each

# ── partition building ────────────────────────────────────────────────────────
def build_partitions(xi1_tr, delta):
    N = xi1_tr.shape[0]

    balls_lo = np.maximum(0.0, xi1_tr - delta)  # (N, R)
    balls_hi = xi1_tr + delta                    # (N, R)

    partitions = [(np.zeros(R), np.full(R, np.inf))]

    repeat = 0
    while len(partitions) < N and repeat < 2:
        for r in range(R):
            exponent = 1.0 / (R - r)

            new_parts = list(partitions)
            for _ in range(len(partitions)):
                part_lo, part_hi = new_parts.pop(0)

                in_mask = (np.all(xi1_tr >= part_lo, axis=1) &
                           np.all(xi1_tr <= part_hi, axis=1))
                in_idx  = np.where(in_mask)[0]

                if len(in_idx) == 0:
                    continue

                data_r = sorted(xi1_tr[in_idx, r])
                n_r    = len(data_r)

                num_cuts  = max(int(n_r ** exponent), 1)
                numpoints = max(int(n_r / num_cuts), 1)

                if r == R - 1 or repeat > 0:
                    numpoints = 1

                new_num_cuts = int(np.ceil(n_r / numpoints))

                last_hi_r = part_lo[r]
                for j in range(new_num_cuts):
                    split_lo = part_lo.copy()
                    split_hi = part_hi.copy()

                    if j > 0:
                        split_lo[r] = last_hi_r

                    end_idx = numpoints * (j + 1)
                    if end_idx < n_r:
                        split_hi[r] = (data_r[end_idx - 1] + data_r[end_idx]) / 2.0
                    else:
                        split_hi[r] = part_hi[r]

                    last_hi_r = split_hi[r]

                    if split_hi[r] > split_lo[r]:
                        new_parts.append((split_lo, split_hi))

            partitions = new_parts

        repeat += 1

    K = len(partitions)

    part_lo_arr = np.array([p[0] for p in partitions])  # (K, R)
    part_hi_arr = np.array([p[1] for p in partitions])  # (K, R)

    K_map = []
    for j in range(N):
        intersect = (np.all(part_lo_arr <= balls_hi[j], axis=1) &
                     np.all(part_hi_arr >= balls_lo[j], axis=1))
        K_map.append(list(np.where(intersect)[0]))

    return balls_lo, balls_hi, partitions, K_map, K

# ── solve SRO-FA LP ───────────────────────────────────────────────────────────
def solve_sro_fa(xi1_tr, xi2_tr, delta, partitions, K_map, K):
    N = xi1_tr.shape[0]

    xi1_lo = np.maximum(0.0, xi1_tr - delta)
    xi1_hi = xi1_tr + delta
    xi2_lo = np.maximum(0.0, xi2_tr - delta)
    xi2_hi = xi2_tr + delta

    part_lo_arr  = np.array([p[0] for p in partitions])
    part_hi_clip = np.where(np.isinf(np.array([p[1] for p in partitions])),
                            1e9, np.array([p[1] for p in partitions]))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    m = gp.Model(env=env)

    Q10  = m.addVar(lb=0.0, name="Q10")
    Q1rv = m.addVars(R, lb=0.0, name="Q1r")
    Q2v  = m.addVars(R, K, lb=0.0, name="Q2")
    objv = m.addVars(N, lb=-GRB.INFINITY, name="obj")

    for k in range(K):
        m.addConstr(gp.quicksum(Q2v[r, k] for r in range(R)) <= Q10)

    for j in range(N):
        for k in K_map[j]:
            xi1_lo_k = np.maximum(xi1_lo[j], part_lo_arr[k])
            xi1_hi_k = np.minimum(xi1_hi[j], part_hi_clip[k])

            u_vars = []
            for r in range(R):
                u = m.addVar(lb=-GRB.INFINITY, name=f"u_{j}_{r}_{k}")
                m.addConstr(
                    u >= b * (xi2_hi[j,r] + xi1_hi_k[r] - Q2v[r,k] - Q1rv[r])
                         - h * Q2v[r,k])
                m.addConstr(
                    u >= h * (Q1rv[r] - xi2_lo[j,r] - xi1_lo_k[r]))
                m.addConstr(
                    u >= b * (xi1_hi_k[r] - Q1rv[r]) - h * xi2_lo[j,r])
                u_vars.append(u)

            m.addConstr(objv[j] >= gp.quicksum(u_vars))

    m.setObjective(
        c * (Q10 + gp.quicksum(Q1rv[r] for r in range(R))) + h * Q10
        + (1.0 / N) * gp.quicksum(objv[j] for j in range(N)),
        GRB.MINIMIZE)

    t0 = time.time()
    m.optimize()
    solve_time = time.time() - t0

    if m.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        m.dispose(); env.dispose()
        raise RuntimeError(f"Gurobi status {m.status}")

    Q10_val = Q10.X
    Q1r_val = np.array([Q1rv[r].X for r in range(R)])
    Q2_val  = np.array([[Q2v[r,k].X for k in range(K)] for r in range(R)])
    obj_val = m.ObjVal

    m.dispose(); env.dispose()
    return Q10_val, Q1r_val, Q2_val, obj_val, solve_time

# ── evaluate on test data ─────────────────────────────────────────────────────
def evaluate(xi1_test, xi2_test, Q10, Q1r, Q2_partitions, partitions):
    part_lo_arr = np.array([p[0] for p in partitions])
    part_hi_arr = np.array([p[1] for p in partitions])

    in_part     = (xi1_test[:, None, :] >= part_lo_arr[None, :, :]) & \
                  (xi1_test[:, None, :] <= part_hi_arr[None, :, :])
    in_part_all = np.all(in_part, axis=2)
    part_assign = np.argmax(in_part_all, axis=1)

    q2 = Q2_partitions[:, part_assign].T   # (N_test, R)

    s1 = c * (Q10 + Q1r.sum()) + h * Q10
    v  = np.maximum(
            b * (xi1_test + xi2_test - Q1r - q2) - h * q2,
            np.maximum(
                h * (Q1r - xi1_test - xi2_test),
                b * (xi1_test - Q1r) - h * xi2_test))
    return float(s1 + v.sum(axis=1).mean())

# ── main sweep ────────────────────────────────────────────────────────────────
print(f"SRO-FA sweep: R={R}, N_train={N_TRAIN}, N_test={N_TEST}")
print(f"Rho values ({len(RHOS)}): {RHOS}")
print(f"Output: {OUT_JSON}\n")

print(f"Generating test data: N={N_TEST}, seed={SEED_TEST}")
xi1_test, xi2_test = gen(N_TEST, SEED_TEST)

results = {}
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as f:
        results = json.load(f)
    print(f"[Cache] Loaded existing results\n")

grand_t0 = time.time()

for rho in RHOS:
    rho_key = str(rho)
    if rho_key not in results:
        results[rho_key] = {}

    done = [s for s in SEEDS if str(s) in results[rho_key]]
    if len(done) == len(SEEDS):
        oos_mean = np.mean([results[rho_key][str(s)]["oos"] for s in SEEDS])
        print(f"[Cache] rho={rho}: all {len(SEEDS)} seeds done  OOS_mean={oos_mean:.4f} — skipping")
        continue

    print(f"\n=== rho={rho} ===")
    for seed in SEEDS:
        seed_key = str(seed)
        if seed_key in results[rho_key]:
            r_ = results[rho_key][seed_key]
            print(f"  [Cache] seed={seed}  K={r_['K']}  IS={r_['is']:.4f}"
                  f"  OOS={r_['oos']:.4f}  total={r_['time_total']:.1f}s")
            continue

        t0 = time.time()
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)

        t_part = time.time()
        balls_lo, balls_hi, partitions, K_map, K = build_partitions(xi1_tr, rho)
        t_part = time.time() - t_part

        t_lp = time.time()
        Q10, Q1r, Q2, obj_is, solve_time = solve_sro_fa(
            xi1_tr, xi2_tr, rho, partitions, K_map, K)
        t_lp = time.time() - t_lp

        t_ev = time.time()
        oos = evaluate(xi1_test, xi2_test, Q10, Q1r, Q2, partitions)
        t_ev = time.time() - t_ev

        wall = time.time() - t0
        print(f"  seed={seed}: K={K}  IS={obj_is:.4f}  OOS={oos:.4f}"
              f"  | partition={t_part:.2f}s  LP={solve_time:.2f}s"
              f"  eval={t_ev:.2f}s  total={wall:.2f}s", flush=True)

        results[rho_key][seed_key] = {
            "is":             float(obj_is),
            "oos":            float(oos),
            "K":              int(K),
            "Q10":            float(Q10),
            "Q1r":            Q1r.tolist(),
            "time_partition": float(t_part),
            "time_lp":        float(solve_time),
            "time_eval":      float(t_ev),
            "time_total":     float(wall),
        }

        with open(OUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Saved] {OUT_JSON}", flush=True)

grand_wall = time.time() - grand_t0
print(f"\n[Done] Total elapsed: {grand_wall:.1f}s  ({grand_wall/60:.1f} min)")
print(f"Results: {OUT_JSON}")

print("\n--- Summary (mean OOS over seeds) ---")
print(f"{'rho':>8}  {'OOS_mean':>10}  {'IS_mean':>10}  {'K_mean':>7}  {'n_seeds':>7}")
for rho in RHOS:
    rho_key = str(rho)
    if rho_key not in results or not results[rho_key]:
        continue
    oos_l = [results[rho_key][str(s)]["oos"] for s in SEEDS if str(s) in results[rho_key]]
    is_l  = [results[rho_key][str(s)]["is"]  for s in SEEDS if str(s) in results[rho_key]]
    k_l   = [results[rho_key][str(s)]["K"]   for s in SEEDS if str(s) in results[rho_key]]
    print(f"{rho:>8g}  {np.mean(oos_l):>10.4f}  {np.mean(is_l):>10.4f}"
          f"  {np.mean(k_l):>7.1f}  {len(oos_l):>7}")
