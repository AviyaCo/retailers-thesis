"""
evaluate_N50_final.py
=====================
Full out-of-sample evaluation (N=10000, seed=456) for N50 / RHO=0.3 pipeline.

Methods compared:
  1. SAA-LP              — nominal first-stage (N=50), clairvoyant optimal Q2 (LP lower bound)
  2. SAA-kNN (k=1)       — nominal first-stage, nearest-neighbour Q2 from 50 training samples
  3. SAA-kNN (k=5)       — nominal first-stage, 5-NN inverse-distance-weighted Q2
  4. Neural nominal      — thesis_checkpoint_N50.pth  (SAA training, best of grid search)
  5. Neural N50 iter4    — converged CCG N50 (RHO_1=0.3, RHO_2=0.3)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
from scipy.spatial.distance import cdist

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

# ============================================================
# CONFIG
# ============================================================
R          = 3
c          = 0.25
h          = 0.05
b          = 0.5
N_TEST     = 10000
N_TRAIN    = 50
SEED_TEST  = 456
SEED_TRAIN = 42

# SAA-LP first-stage (from results_nominal_N50_summary.txt, N=50, seed=42)
Q10_SAA = 23.1579
Q1r_SAA = np.array([7.2418, 7.1840, 6.9041])

CHECKPOINTS = {
    "neural_nominal": "/data/home/aviyac/retailers_problem/R3/N50/code_check/run_N50_rho03/thesis_checkpoint_N50.pth",
    "N50_iter4":      "/data/home/aviyac/retailers_problem/R3/N50/code_check/run_N50_rho03/thesis_checkpoint_mp_iter4_N50.pth",
}

# ============================================================
# DATA
# ============================================================
def generate_data(N, seed):
    np.random.seed(seed)
    xi1 = np.maximum(np.random.normal(6.0, 2.5, (N, R)), 0.0)
    xi2 = np.maximum(np.random.normal(2 * (6.0 - xi1) ** 2, 2.5), 0.0)
    return xi1, xi2

print("Generating data...")
xi1_test,  xi2_test  = generate_data(N_TEST,  SEED_TEST)
xi1_train, xi2_train = generate_data(N_TRAIN, SEED_TRAIN)

xi1_test_t = torch.tensor(xi1_test, dtype=torch.float32)

print(f"  Test  N={N_TEST}, seed={SEED_TEST}:  xi1 mean={xi1_test.mean():.3f}, xi2 mean={xi2_test.mean():.3f}")
print(f"  Train N={N_TRAIN}, seed={SEED_TRAIN}: xi1 mean={xi1_train.mean():.3f}, xi2 mean={xi2_train.mean():.3f}")

# ============================================================
# COST HELPERS
# ============================================================
def vr_np(xi1, xi2, Q1r, Q2):
    return np.maximum(
        b * (xi1 + xi2 - Q1r - Q2) - h * Q2,
        np.maximum(
            h * (Q1r - xi1 - xi2),
            b * (xi1 - Q1r) - h * xi2
        )
    )

def total_cost(Q10, Q1r, Q2, xi1, xi2):
    stage1 = c * (Q10 + Q1r.sum()) + h * Q10
    Vr     = vr_np(xi1, xi2, Q1r, Q2)
    return stage1 + Vr.sum(axis=1)

def report(label, Q10, Q1r, Q2, xi1, xi2):
    costs = total_cost(Q10, Q1r, Q2, xi1, xi2)
    viol  = np.maximum(Q2.sum(axis=1) - Q10, 0.0)
    cap_pct = (viol <= 1e-5).mean() * 100
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Q10:            {Q10:.4f}")
    print(f"  Q1r:            {np.round(Q1r, 4)}")
    print(f"  Mean cost:      {costs.mean():.4f}")
    print(f"  Std  cost:      {costs.std():.4f}")
    print(f"  Percentiles:    p25={np.percentile(costs,25):.2f}  "
          f"p50={np.percentile(costs,50):.2f}  "
          f"p75={np.percentile(costs,75):.2f}  "
          f"p95={np.percentile(costs,95):.2f}  "
          f"p99={np.percentile(costs,99):.2f}")
    print(f"  Cap. feasible:  {cap_pct:.1f}%")
    print(f"  Max (WC) cost:  {costs.max():.4f}")
    return costs, cap_pct

results = {}

# ============================================================
# SECTION 1: SAA-LP — clairvoyant optimal Q2 (lower bound)
# ============================================================
print("\n\n[1/5] SAA-LP — nominal first-stage (N=50), clairvoyant optimal Q2")
print("  Solving joint recourse LP for all test samples ...")

env0 = gp.Env(empty=True); env0.setParam("OutputFlag", 0); env0.start()
m0 = gp.Model(env=env0)
Q2v = m0.addVars(N_TEST, R, lb=0.0, name="Q2")
vv  = m0.addVars(N_TEST, R, lb=-GRB.INFINITY, name="v")
for i in range(N_TEST):
    m0.addConstr(gp.quicksum(Q2v[i,r] for r in range(R)) <= Q10_SAA)
    for r in range(R):
        m0.addConstr(vv[i,r] >= b*(xi1_test[i,r]+xi2_test[i,r]-Q2v[i,r]-Q1r_SAA[r]) - h*Q2v[i,r])
        m0.addConstr(vv[i,r] >= h*(Q1r_SAA[r]-xi1_test[i,r]-xi2_test[i,r]))
        m0.addConstr(vv[i,r] >= b*(xi1_test[i,r]-Q1r_SAA[r]) - h*xi2_test[i,r])
m0.setObjective(gp.quicksum(vv[i,r] for i in range(N_TEST) for r in range(R)), GRB.MINIMIZE)
m0.optimize()
Q2_saa_opt = np.array([[Q2v[i,r].X for r in range(R)] for i in range(N_TEST)])
costs_saa, cap_saa = report("SAA-LP (N=50) — clairvoyant optimal Q2 [LB]",
                             Q10_SAA, Q1r_SAA, Q2_saa_opt, xi1_test, xi2_test)
results["SAA-LP (clairvoyant Q2)"] = (Q10_SAA, Q1r_SAA.copy(), costs_saa.mean(), costs_saa.std(), cap_saa, costs_saa.max())

# ============================================================
# SECTION 2: Optimal Q2 for training samples (for kNN)
# ============================================================
print("\n\n[2/5] Computing optimal Q2 for 50 training scenarios (for kNN lookup) ...")

env1 = gp.Env(empty=True); env1.setParam("OutputFlag", 0); env1.start()
m1 = gp.Model(env=env1)
Q2tr = m1.addVars(N_TRAIN, R, lb=0.0, name="Q2")
vtr  = m1.addVars(N_TRAIN, R, lb=-GRB.INFINITY, name="v")
for i in range(N_TRAIN):
    m1.addConstr(gp.quicksum(Q2tr[i,r] for r in range(R)) <= Q10_SAA)
    for r in range(R):
        m1.addConstr(vtr[i,r] >= b*(xi1_train[i,r]+xi2_train[i,r]-Q2tr[i,r]-Q1r_SAA[r]) - h*Q2tr[i,r])
        m1.addConstr(vtr[i,r] >= h*(Q1r_SAA[r]-xi1_train[i,r]-xi2_train[i,r]))
        m1.addConstr(vtr[i,r] >= b*(xi1_train[i,r]-Q1r_SAA[r]) - h*xi2_train[i,r])
m1.setObjective(gp.quicksum(vtr[i,r] for i in range(N_TRAIN) for r in range(R)), GRB.MINIMIZE)
m1.optimize()
Q2_train_opt = np.array([[Q2tr[i,r].X for r in range(R)] for i in range(N_TRAIN)])
print(f"  Done. Q2_train mean={Q2_train_opt.mean():.3f}, sum-per-sample mean={Q2_train_opt.sum(axis=1).mean():.3f}")

# ============================================================
# SECTION 3: SAA-kNN
# ============================================================
print("\n\nComputing pairwise xi1 distances ...")
dists = cdist(xi1_test, xi1_train, metric='euclidean')
print(f"  Distance matrix shape: {dists.shape}")
print(f"  Mean nearest distance: {dists.min(axis=1).mean():.4f}")

def knn_Q2(k, dists, Q2_train, Q10):
    N_test = dists.shape[0]
    knn_idx   = np.argpartition(dists, k, axis=1)[:, :k]
    knn_dists = np.take_along_axis(dists, knn_idx, axis=1)
    eps     = 1e-9
    inv_d   = 1.0 / np.maximum(knn_dists, eps)
    weights = inv_d / inv_d.sum(axis=1, keepdims=True)
    Q2_neighbors = Q2_train[knn_idx]
    Q2_raw = (weights[:, :, np.newaxis] * Q2_neighbors).sum(axis=1)
    s = Q2_raw.sum(axis=1, keepdims=True)
    scale = np.where(s > Q10, Q10 / s, 1.0)
    return Q2_raw * scale

print("\n[3/5] SAA-kNN k=1")
Q2_knn1 = knn_Q2(1, dists, Q2_train_opt, Q10_SAA)
costs_knn1, cap_knn1 = report("SAA-kNN k=1 — nearest-neighbour Q2 (N=50 training set)",
                               Q10_SAA, Q1r_SAA, Q2_knn1, xi1_test, xi2_test)
results["SAA-kNN (k=1)"] = (Q10_SAA, Q1r_SAA.copy(), costs_knn1.mean(), costs_knn1.std(), cap_knn1, costs_knn1.max())

print("\n[4/5] SAA-kNN k=5")
Q2_knn5 = knn_Q2(5, dists, Q2_train_opt, Q10_SAA)
costs_knn5, cap_knn5 = report("SAA-kNN k=5 — 5-NN inverse-distance-weighted Q2 (N=50 training set)",
                               Q10_SAA, Q1r_SAA, Q2_knn5, xi1_test, xi2_test)
results["SAA-kNN (k=5)"] = (Q10_SAA, Q1r_SAA.copy(), costs_knn5.mean(), costs_knn5.std(), cap_knn5, costs_knn5.max())

# ============================================================
# SECTION 4–5: Neural policies (Gurobi QP projection)
# ============================================================
class PredictAndProjectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R * 2 + 1, 32), nn.ReLU(),
            nn.Linear(32, 32),        nn.ReLU(),
            nn.Linear(32, R)
        )
        self.Q10 = nn.Parameter(torch.zeros(1,  dtype=torch.float32))
        self.Q1r = nn.Parameter(torch.zeros(R,  dtype=torch.float32))

    @torch.no_grad()
    def get_Q2_gurobi(self, xi1_t):
        N   = xi1_t.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()

        q1r_in = torch.tensor(q1r, dtype=torch.float32).unsqueeze(0).expand(N, -1)
        q10_in = torch.full((N, 1), q10)
        net_in = torch.cat([xi1_t, q1r_in, q10_in], dim=1)
        z_np   = self.fc_layers(net_in).numpy()

        env_q = gp.Env(empty=True); env_q.setParam("OutputFlag", 0); env_q.start()
        mq = gp.Model(env=env_q)
        xv = mq.addVars(R, lb=0.0, name="x")
        mq.addConstr(gp.quicksum(xv[r] for r in range(R)) <= q10, name="cap")
        mq.update()

        Q2_np = np.zeros_like(z_np)
        for i in range(N):
            obj = gp.QuadExpr()
            for r in range(R):
                obj += xv[r] * xv[r] - 2.0 * float(z_np[i, r]) * xv[r]
            mq.setObjective(obj, GRB.MINIMIZE)
            mq.optimize()
            if mq.Status == GRB.OPTIMAL:
                Q2_np[i] = [xv[r].X for r in range(R)]
            else:
                xc = np.maximum(z_np[i], 0.0)
                s  = xc.sum()
                Q2_np[i] = xc * (q10 / s) if s > q10 else xc
        return Q2_np, q10, q1r

def eval_neural(checkpoint_path, label, idx, total):
    if not os.path.exists(checkpoint_path):
        print(f"\n  [SKIP] {checkpoint_path} not found.")
        return None, None, None, None
    print(f"\n[{idx}/{total}] Loading {os.path.basename(checkpoint_path)} ...")
    model = PredictAndProjectModel()
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"  Projecting {N_TEST} samples via Gurobi QP (exact) ...")
    Q2_np, Q10, Q1r = model.get_Q2_gurobi(xi1_test_t)
    costs, cap = report(label, Q10, Q1r, Q2_np, xi1_test, xi2_test)
    return Q10, Q1r, costs, cap

Q10_n, Q1r_n, costs_n, cap_n = eval_neural(
    CHECKPOINTS["neural_nominal"],
    "Neural nominal N50 — SAA training, best of grid search", 4, 5)
if costs_n is not None:
    results["Neural nominal (N=50)"] = (Q10_n, Q1r_n, costs_n.mean(), costs_n.std(), cap_n, costs_n.max())

Q10_r, Q1r_r, costs_r, cap_r = eval_neural(
    CHECKPOINTS["N50_iter4"],
    "Neural N50 iter4 — CONVERGED CCG (RHO_1=0.3, RHO_2=0.3)", 5, 5)
if costs_r is not None:
    results["Neural N50 iter4 (robust)"] = (Q10_r, Q1r_r, costs_r.mean(), costs_r.std(), cap_r, costs_r.max())

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
print("\n\n" + "=" * 120)
print("FINAL COMPARISON TABLE  (N_test=10000, seed=456, Gurobi QP projection)")
print(f"Training: N={N_TRAIN}, seed={SEED_TRAIN} | RHO_1=0.3, RHO_2=0.3")
print("=" * 120)
header = f"  {'Method':<44} {'Q10':>7} {'Mean Cost':>10} {'Std Cost':>10} {'Worst Cost':>11} {'Cap.Feas%':>10}  Notes"
print(header)
print("-" * 120)

notes = {
    "SAA-LP (clairvoyant Q2)":      "← clairvoyant LB (knows ξ2)",
    "SAA-kNN (k=1)":                "← realistic kNN policy",
    "SAA-kNN (k=5)":                "← realistic kNN policy",
    "Neural nominal (N=50)":        "← no robustness, SAA training",
    "Neural N50 iter4 (robust)":    "← CCG converged (RHO=0.3)",
}
for name, (q10, q1r, mean_c, std_c, cap, max_c) in results.items():
    note = notes.get(name, "")
    print(f"  {name:<44} {q10:>7.3f} {mean_c:>10.4f} {std_c:>10.4f} {max_c:>11.4f} {cap:>9.1f}%  {note}")

print("-" * 120)

ref_lb = results.get("SAA-LP (clairvoyant Q2)", (None,)*6)[2]
if ref_lb is not None:
    print(f"\n  Gap vs SAA-LP clairvoyant LB ({ref_lb:.4f}):")
    for name, (q10, q1r, mean_c, std_c, cap, max_c) in results.items():
        if "clairvoyant" not in name.lower():
            gap = (mean_c - ref_lb) / abs(ref_lb) * 100
            print(f"    {name:<44} +{gap:.2f}%")

    print(f"\n  Worst-case cost vs clairvoyant LB:")
    ref_wc = results["SAA-LP (clairvoyant Q2)"][5]
    for name, (q10, q1r, mean_c, std_c, cap, max_c) in results.items():
        gap_wc = (max_c - ref_wc) / abs(ref_wc) * 100
        print(f"    {name:<44} {max_c:.4f}  ({gap_wc:+.2f}%)")

print("\n[Done] evaluate_N50_final.py complete.")
