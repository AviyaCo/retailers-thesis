"""
compute_lb_10k.py  --  Compute SAA-LP clairvoyant lower bound on N=10000 test set.
====================================================================================
Solves the full joint LP (optimize Q10, Q1r, Q2 jointly) on the 10k test draw,
then patches sweep_eval_net16_v2_10k.json with the correct lb_oos value.
Replaces the placeholder that was copied from the 2k eval.

Estimated runtime: 10–30 min (Gurobi LP with ~60k variables).
"""

import json, os, time
import numpy as np
from scipy.stats import truncnorm
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

SWEEP_DIR  = "/data/home/aviyac/retailers_problem/R3/N50/rho_sweep_net16_v2"
NOMINAL_DIR = "/data/home/aviyac/retailers_problem/R3/N50/saainit_net16_v2"
OUT_JSON   = os.path.join(SWEEP_DIR, "sweep_eval_net16_v2_10k.json")

N_TEST  = 10000
SEED_TEST = 9999
R       = 3
N_TRAIN = 50
c, h, b = 0.25, 0.05, 0.5
SEEDS   = list(range(10))

# ── data ──────────────────────────────────────────────────────────────────────
def gen(N, seed):
    np.random.seed(seed)
    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

def mean_cost(Q10, Q1r, Q2, xi1, xi2):
    s1 = c * (Q10 + Q1r.sum()) + h * Q10
    v  = np.maximum(b * (xi1 + xi2 - Q1r - Q2) - h * Q2,
          np.maximum(h * (Q1r - xi1 - xi2),
                     b * (xi1 - Q1r) - h * xi2))
    return float((s1 + v.sum(axis=1)).mean())

def saa_lp_full(xi1, xi2):
    N = xi1.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag", 1); env.start()
    m = gp.Model(env=env)
    m.setParam("Method", 2)        # barrier — fast for large LPs
    m.setParam("Crossover", 0)     # skip crossover (we only need obj value)
    Q10v = m.addVar(lb=0.0, name="Q10")
    Q1rv = m.addVars(R, lb=0.0, name="Q1r")
    Q2v  = m.addVars(N, R, lb=0.0, name="Q2")
    vv   = m.addVars(N, R, lb=-GRB.INFINITY, name="v")
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i, r] for r in range(R)) <= Q10v)
        for r in range(R):
            m.addConstr(vv[i,r] >= b*(xi1[i,r]+xi2[i,r]-Q1rv[r]-Q2v[i,r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1rv[r] - xi1[i,r] - xi2[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1[i,r] - Q1rv[r]) - h*xi2[i,r])
    m.setObjective(
        (c + h) * Q10v + c * gp.quicksum(Q1rv[r] for r in range(R))
        + (1.0/N) * gp.quicksum(vv[i,r] for i in range(N) for r in range(R)),
        GRB.MINIMIZE)
    m.optimize()
    Q10_opt = Q10v.X
    Q1r_opt = np.array([Q1rv[r].X for r in range(R)])
    Q2_opt  = np.array([[Q2v[i,r].X for r in range(R)] for i in range(N)])
    obj     = m.ObjVal
    return Q10_opt, Q1r_opt, Q2_opt, obj

# ── generate test data ─────────────────────────────────────────────────────────
print(f"Generating test data: N={N_TEST}, seed={SEED_TEST}")
xi1_test, xi2_test = gen(N_TEST, SEED_TEST)

# ── solve LP ──────────────────────────────────────────────────────────────────
print(f"\nSolving SAA-LP full LB on N={N_TEST} test scenarios ...")
print("(~60k variables, barrier method — expect 10-30 min)")
t0 = time.time()
Q10_lb, Q1r_lb, Q2_lb, obj_val = saa_lp_full(xi1_test, xi2_test)
elapsed = time.time() - t0
lb_oos = mean_cost(Q10_lb, Q1r_lb, Q2_lb, xi1_test, xi2_test)
print(f"\nDone in {elapsed/60:.1f} min")
print(f"SAA-LP LB OOS (N=10k): {lb_oos:.5f}  (Gurobi obj={obj_val:.5f})")
print(f"Q10={Q10_lb:.4f}  Q1r={Q1r_lb}")

# ── per-seed IS LB (on training data, small LP — fast) ────────────────────────
print("\nComputing per-seed IS lower bounds on training data (fast) ...")

def saa_lp_full_quiet(xi1, xi2):
    N = xi1.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model(env=env)
    Q10v = m.addVar(lb=0.0)
    Q1rv = m.addVars(R, lb=0.0)
    Q2v  = m.addVars(N, R, lb=0.0)
    vv   = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i, r] for r in range(R)) <= Q10v)
        for r in range(R):
            m.addConstr(vv[i,r] >= b*(xi1[i,r]+xi2[i,r]-Q1rv[r]-Q2v[i,r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1rv[r] - xi1[i,r] - xi2[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1[i,r] - Q1rv[r]) - h*xi2[i,r])
    m.setObjective(
        (c + h) * Q10v + c * gp.quicksum(Q1rv[r] for r in range(R))
        + (1.0/N) * gp.quicksum(vv[i,r] for i in range(N) for r in range(R)),
        GRB.MINIMIZE)
    m.optimize()
    return np.array([Q1rv[r].X for r in range(R)]), \
           np.array([[Q2v[i,r].X for r in range(R)] for i in range(N)]), \
           Q10v.X

saa_lb_is = []
for seed in SEEDS:
    xi1_tr, xi2_tr = gen(N_TRAIN, seed)
    Q1r_tr, Q2_tr, Q10_tr = saa_lp_full_quiet(xi1_tr, xi2_tr)
    saa_lb_is.append(mean_cost(Q10_tr, Q1r_tr, Q2_tr, xi1_tr, xi2_tr))
    print(f"  seed {seed}: IS LB = {saa_lb_is[-1]:.5f}")

# ── patch the 10k JSON ────────────────────────────────────────────────────────
if not os.path.exists(OUT_JSON):
    print(f"\nERROR: {OUT_JSON} not found. Run eval_sweep_net16_v2_10k.py first.")
    exit(1)

with open(OUT_JSON) as f:
    data = json.load(f)

data["benchmarks"]["saa_lb"] = {
    "oos": [lb_oos] * len(SEEDS),
    "is":  saa_lb_is,
}
data["benchmarks"]["_full_lp_10k"] = True

with open(OUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

print(f"\nPatched {OUT_JSON}")
print(f"SAA-LP LB (N=10k): {lb_oos:.5f}")
print(f"Mean IS LB: {np.mean(saa_lb_is):.5f}")
print("\nNow re-run:  ap_env/bin/python3 retailers_problem/plot_N50_net16v2.py")
