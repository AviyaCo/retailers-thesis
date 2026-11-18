"""
sro_fa_shipping_sweep.py  —  SRO-FA method for N=200 with fixed shipping
=============================================================================
Implements the Sample Robust Optimization - Finite Adaptability (SRO-FA) from
the supervisor's Julia code, adapted to the fixed-shipping variant.

Problem: R=3 retailers, N=200 training samples, fixed shipping cost f=0.1.
  - Same rho (= delta_N) grid as the neural robust sweep.
  - Both methods use identical L-inf ball: xi +/- rho per dimension.
  - Evaluates OOS on 10k test set (seed=9999).

Saves: sro_fa_shipping_results_N200_R3.json  (incremental, safe to interrupt/restart)
"""

import os, json, platform, socket, time
from datetime import datetime, timezone
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
f_ship    = 0.1

# Same epsilon grid as experiment_inventory_pooling.jl:
# vcat(0, 0.001:0.001:0.009, 0.01:0.01:0.09, 0.1:0.1:1.5)
RHOS = (
    [0.0]
    + [round(x, 3) for x in np.arange(0.001, 0.010, 0.001)]
    + [round(x, 2) for x in np.arange(0.01, 0.10, 0.01)]
    + [round(x, 1) for x in np.arange(0.1, 1.51, 0.1)]
)
SEEDS = list(range(10))
if os.environ.get("SRO_FA_RHOS"):
    RHOS = [float(value) for value in os.environ["SRO_FA_RHOS"].split(",")]
if os.environ.get("SRO_FA_SEEDS"):
    SEEDS = [int(value) for value in os.environ["SRO_FA_SEEDS"].split(",")]

OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.environ.get(
    "SRO_FA_OUT_JSON",
    os.path.join(OUT_DIR, "sro_fa_shipping_results_N200_R3.json"),
)
RUNTIME_JSON = os.environ.get(
    "SRO_FA_RUNTIME_JSON",
    os.path.splitext(OUT_JSON)[0] + "_runtime.json",
)


def atomic_json_dump(path, payload):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary_path, path)


def status_name(status_code):
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }
    return names.get(status_code, f"STATUS_{status_code}")

# ── data generation (identical to neural sweep) ───────────────────────────────
def gen(N, seed):
    np.random.seed(seed)
    a   = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a2  = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2   # (N, R) each

# ── partition building  (port of BuildSampleUncertaintySet) ───────────────────
def build_partitions(xi1_tr, delta):
    """
    Creates an L-inf ball of radius delta around each training point and
    builds a rectangular partition of the xi1-space, then maps balls to
    intersecting partitions.

    Port of BuildSampleUncertaintySet() from inventory_pooling_new.jl.

    Args:
        xi1_tr : (N, R) first-stage training demands
        delta  : float, L-inf ball radius (= rho)

    Returns:
        balls_lo  : (N, R)  lower bounds of xi1-balls
        balls_hi  : (N, R)  upper bounds of xi1-balls
        partitions: list of K tuples (lo, hi), each ndarray (R,)
        K_map     : list of N lists; K_map[j] = indices of partitions
                    that intersect ball j
    """
    N = xi1_tr.shape[0]

    # xi1 ball bounds
    balls_lo = np.maximum(0.0, xi1_tr - delta)  # (N, R)
    balls_hi = xi1_tr + delta                    # (N, R)

    # Start: one partition covering [0, inf)^R
    partitions = [(np.zeros(R), np.full(R, np.inf))]

    repeat = 0
    while len(partitions) < N and repeat < 2:
        for r in range(R):
            # exponent: 1/(R-r+1) in Julia 1-indexed = 1/(R-r) in Python 0-indexed
            exponent = 1.0 / (R - r)

            # Process each partition: dequeue from front, enqueue splits at back
            new_parts = list(partitions)
            for _ in range(len(partitions)):
                part_lo, part_hi = new_parts.pop(0)   # dequeue

                # Training points inside this partition (by xi1)
                in_mask = (np.all(xi1_tr >= part_lo, axis=1) &
                           np.all(xi1_tr <= part_hi, axis=1))
                in_idx  = np.where(in_mask)[0]

                if len(in_idx) == 0:
                    # Empty partition — drop it
                    continue

                data_r = sorted(xi1_tr[in_idx, r])
                n_r    = len(data_r)

                num_cuts  = max(int(n_r ** exponent), 1)
                numpoints = max(int(n_r / num_cuts), 1)

                # Last dimension or second repeat → one point per split
                if r == R - 1 or repeat > 0:
                    numpoints = 1

                new_num_cuts = int(np.ceil(n_r / numpoints))

                # Generate splits along dimension r
                last_hi_r = part_lo[r]
                for j in range(new_num_cuts):
                    split_lo    = part_lo.copy()
                    split_hi    = part_hi.copy()

                    if j > 0:
                        split_lo[r] = last_hi_r

                    end_idx = numpoints * (j + 1)
                    if end_idx < n_r:
                        # Midpoint between end-of-chunk and start-of-next-chunk
                        split_hi[r] = (data_r[end_idx - 1] + data_r[end_idx]) / 2.0
                    else:
                        split_hi[r] = part_hi[r]

                    last_hi_r = split_hi[r]

                    if split_hi[r] > split_lo[r]:
                        new_parts.append((split_lo, split_hi))

            partitions = new_parts

        repeat += 1

    K = len(partitions)
    if K > N:
        raise RuntimeError(f"Too many partitions: K={K} > N={N}")

    # Build K_map: ball j intersects partition k iff balls_lo[j] <= part_hi
    #              AND part_lo <= balls_hi[j]  (for all dimensions)
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
    """
    Port of SolveInventoryPoolingProblem() with fixed shipping cost.
    Uses binary z[r,k] = 1 iff partition-policy Q2[r,k] is positive.

    Objective: c*(Q10 + sum Q1r) + h*Q10 + (1/N)*sum_j obj[j]

    Worst-case cost constraints (analytically, within each ball-partition cell):
      u[j,r,k] >= b*(xi2_hi[j,r] + xi1_hi_k[j,r] - Q2[r,k] - Q1r[r]) - h*Q2[r,k]
      u[j,r,k] >= h*(Q1r[r] - xi2_lo[j,r] - xi1_lo_k[j,r])
      u[j,r,k] >= b*(xi1_hi_k[j,r] - Q1r[r]) - h*xi2_lo[j,r]

    Returns: Q10, Q1r (R,), Q2 (R, K), obj_val (IS), solve_time
    """
    N = xi1_tr.shape[0]

    # Ball bounds for xi1 and xi2 (L-inf, clamped to >= 0 for lower)
    xi1_lo = np.maximum(0.0, xi1_tr - delta)  # (N, R)
    xi1_hi = xi1_tr + delta                    # (N, R)
    xi2_lo = np.maximum(0.0, xi2_tr - delta)  # (N, R)
    xi2_hi = xi2_tr + delta                    # (N, R)

    part_lo_arr = np.array([p[0] for p in partitions])  # (K, R)
    part_hi_arr = np.array([p[1] for p in partitions])  # (K, R)
    # Replace inf with a large finite value for clipping (doesn't affect LP)
    part_hi_clip = np.where(np.isinf(part_hi_arr), 1e9, part_hi_arr)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    m.setParam("MIPGap", 0.01)
    m.setParam("TimeLimit", 3000)
    m.setParam("Presolve", 1)
    m.setParam("Method", 3)
    m.setParam("Crossover", 0)
    m.setParam("Threads", 2)

    Q10  = m.addVar(lb=0.0, name="Q10")
    Q1rv = m.addVars(R, lb=0.0, name="Q1r")
    Q2v  = m.addVars(R, K, lb=0.0, name="Q2")
    Zv   = m.addVars(R, K, vtype=GRB.BINARY, name="z_ship")
    objv = m.addVars(N, lb=-GRB.INFINITY, name="obj")

    # Budget: sum_r Q2[r,k] <= Q10 for each partition k
    big_m = 1000.0
    for k in range(K):
        m.addConstr(gp.quicksum(Q2v[r, k] for r in range(R)) <= Q10)
        for r in range(R):
            m.addConstr(Q2v[r, k] <= big_m * Zv[r, k])

    # Worst-case cost constraints
    for j in range(N):
        for k in K_map[j]:
            # Effective xi1 bounds within ball j ∩ partition k
            xi1_lo_k = np.maximum(xi1_lo[j], part_lo_arr[k])    # (R,)
            xi1_hi_k = np.minimum(xi1_hi[j], part_hi_clip[k])   # (R,)

            u_vars = []
            for r in range(R):
                u = m.addVar(lb=-GRB.INFINITY, name=f"u_{j}_{r}_{k}")
                # Constraint 1: worst-case backorder (max xi1, max xi2)
                m.addConstr(
                    u >= b * (xi2_hi[j,r] + xi1_hi_k[r] - Q2v[r,k] - Q1rv[r])
                         - h * Q2v[r,k])
                # Constraint 2: worst-case holding (min xi1, min xi2)
                m.addConstr(
                    u >= h * (Q1rv[r] - xi2_lo[j,r] - xi1_lo_k[r]))
                # Constraint 3: stage-1 backorder + stage-2 holding (max xi1, min xi2)
                m.addConstr(
                    u >= b * (xi1_hi_k[r] - Q1rv[r]) - h * xi2_lo[j,r])
                u_vars.append(u)

            # obj[j] >= sum_r u[j,r,k] + fixed shipping for this partition k
            m.addConstr(
                objv[j] >= gp.quicksum(u_vars)
                + f_ship * gp.quicksum(Zv[r, k] for r in range(R))
            )

    m.setObjective(
        c * (Q10 + gp.quicksum(Q1rv[r] for r in range(R))) + h * Q10
        + (1.0 / N) * gp.quicksum(objv[j] for j in range(N)),
        GRB.MINIMIZE)

    t0 = time.perf_counter()
    m.optimize()
    solve_time = time.perf_counter() - t0

    status_code = int(m.Status)
    sol_count = int(m.SolCount)
    mip_gap = float(m.MIPGap) if sol_count > 0 else None
    obj_bound = float(m.ObjBound) if sol_count > 0 else None
    node_count = float(m.NodeCount)
    gurobi_runtime = float(m.Runtime)

    if status_code not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        m.dispose(); env.dispose()
        raise RuntimeError(f"Gurobi status {status_code}")
    if sol_count <= 0:
        m.dispose(); env.dispose()
        raise RuntimeError(f"Gurobi status {status_code} with no incumbent")

    Q10_val = Q10.X
    Q1r_val = np.array([Q1rv[r].X for r in range(R)])
    Q2_val  = np.array([[Q2v[r,k].X for k in range(K)] for r in range(R)])  # (R, K)
    Z_val = np.array([[Zv[r,k].X for k in range(K)] for r in range(R)])
    obj_val = m.ObjVal

    m.dispose(); env.dispose()
    return (
        Q10_val, Q1r_val, Q2_val, Z_val, obj_val, solve_time,
        status_code, sol_count, mip_gap, obj_bound, node_count, gurobi_runtime,
    )

# ── evaluate policy on test data ──────────────────────────────────────────────
def evaluate(xi1_test, xi2_test, Q10, Q1r, Q2_partitions, partitions):
    """
    Assign each test point xi1_test[i] to a partition, then compute mean cost
    using the assigned Q2[:, k].  Equivalent to EveluateInventoryPoolingProblem.

    Uses the same scenario_costs formula as the neural sweep for consistency.
    """
    K = len(partitions)
    part_lo_arr = np.array([p[0] for p in partitions])  # (K, R)
    part_hi_arr = np.array([p[1] for p in partitions])  # (K, R)

    # Vectorised partition assignment: (N_test, K) -> first match
    in_part = (xi1_test[:, None, :] >= part_lo_arr[None, :, :]) & \
              (xi1_test[:, None, :] <= part_hi_arr[None, :, :])   # (N_test, K, R)
    in_part_all = np.all(in_part, axis=2)                          # (N_test, K)
    part_assign = np.argmax(in_part_all, axis=1)                   # (N_test,)

    # Gather Q2 for each test point's partition
    q2 = Q2_partitions[:, part_assign].T   # (N_test, R)

    # scenario_costs with hard shipping by the Julia threshold Q2 > 1e-10
    s1 = c * (Q10 + Q1r.sum()) + h * Q10
    z_ship = (q2 > 1e-10).astype(float)
    ship = f_ship * z_ship.sum(axis=1)
    v  = np.maximum(
            b * (xi1_test + xi2_test - Q1r - q2) - h * q2,
            np.maximum(
                h * (Q1r - xi1_test - xi2_test),
                b * (xi1_test - Q1r) - h * xi2_test))
    return float((s1 + ship + v.sum(axis=1)).mean())

# ── main sweep ────────────────────────────────────────────────────────────────
print(f"SRO-FA shipping sweep: R={R}, N_train={N_TRAIN}, N_test={N_TEST}, f_ship={f_ship}")
print(f"Rho values ({len(RHOS)}): {RHOS}")
print(f"Output: {OUT_JSON}\n")

print(f"Generating test data: N={N_TEST}, seed={SEED_TEST}")
test_started = time.perf_counter()
xi1_test, xi2_test = gen(N_TEST, SEED_TEST)
test_generation_time = time.perf_counter() - test_started

# Load cache
results = {}
if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as f:
        results = json.load(f)
    print(f"[Cache] Loaded existing results from {OUT_JSON}\n")

grand_t0 = time.perf_counter()
run_metadata = {
    "experiment": "fixed_shipping_sro_fa_solver_status_audit",
    "status": "running",
    "n_train": N_TRAIN,
    "n_test": N_TEST,
    "test_seed": SEED_TEST,
    "training_seeds": SEEDS,
    "rhos": RHOS,
    "fixed_shipping_cost": f_ship,
    "solver": "Gurobi",
    "gurobi_version": list(gp.gurobi.version()),
    "target_mip_gap": 0.01,
    "time_limit_s": 3000,
    "gurobi_threads": 2,
    "sweep_workers": 1,
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "logical_cpu_count": os.cpu_count(),
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "test_generation_time_s": test_generation_time,
    "completed_jobs": 0,
}
atomic_json_dump(RUNTIME_JSON, run_metadata)

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

        t0 = time.perf_counter()
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)

        # 1) Build partition structure
        t_part = time.perf_counter()
        balls_lo, balls_hi, partitions, K_map, K = build_partitions(xi1_tr, rho)
        t_part = time.perf_counter() - t_part

        # 2) Solve SRO-FA LP
        t_lp = time.perf_counter()
        Q10, Q1r, Q2, Z, obj_is, solve_time, status_code, sol_count, mip_gap, obj_bound, node_count, gurobi_runtime = solve_sro_fa(
            xi1_tr, xi2_tr, rho, partitions, K_map, K)
        t_lp = time.perf_counter() - t_lp

        # 3) Evaluate OOS on 10k test set
        t_ev = time.perf_counter()
        oos = evaluate(xi1_test, xi2_test, Q10, Q1r, Q2, partitions)
        t_ev = time.perf_counter() - t_ev

        wall = time.perf_counter() - t0
        print(f"  seed={seed}: K={K}  IS={obj_is:.4f}  OOS={oos:.4f}"
              f"  | partition={t_part:.2f}s  LP={solve_time:.2f}s"
              f"  eval={t_ev:.2f}s  total={wall:.2f}s", flush=True)

        results[rho_key][seed_key] = {
            "is":             float(obj_is),
            "oos":            float(oos),
            "K":              int(K),
            "Q10":            float(Q10),
            "Q1r":            Q1r.tolist(),
            "z_ship_sum":      float(Z.sum()),
            "time_partition": float(t_part),
            "time_lp":        float(solve_time),
            "time_eval":      float(t_ev),
            "time_total":     float(wall),
            "solver_status":  status_code,
            "solver_status_name": status_name(status_code),
            "sol_count":      sol_count,
            "mip_gap":        mip_gap,
            "obj_bound":      obj_bound,
            "node_count":     node_count,
            "gurobi_runtime_s": gurobi_runtime,
        }

        atomic_json_dump(OUT_JSON, results)
        run_metadata["completed_jobs"] += 1
        run_metadata["current_invocation_elapsed_time_s"] = time.perf_counter() - grand_t0
        atomic_json_dump(RUNTIME_JSON, run_metadata)
        print(f"  [Saved] {OUT_JSON}", flush=True)

grand_wall = time.perf_counter() - grand_t0
run_metadata["status"] = "completed"
run_metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
run_metadata["current_invocation_elapsed_time_s"] = grand_wall
atomic_json_dump(RUNTIME_JSON, run_metadata)
print(f"\n[Done] Total elapsed: {grand_wall:.1f}s  ({grand_wall/60:.1f} min)")
print(f"Results: {OUT_JSON}")

# Print summary table
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
