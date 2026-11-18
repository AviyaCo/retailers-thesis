"""
eval_N200.py — Out-of-sample evaluation for N=200 nominal model
===============================================================
Compares Neural nominal (N=200) vs SAA-kNN (N=200) on N=2000 test samples,
and prints a side-by-side table with the N=50 results for context.

N=50 numbers come from eval_seeds_results (seeds 0-4 average).
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

R = 3; c = 0.25; h = 0.05; b = 0.5
N_TRAIN   = 200
SEED_TRAIN = 42
N_TEST    = 2000
SEED_TEST = 999

CKPT_N200 = "/data/home/aviyac/retailers_problem/R3/N200/code_check/run_N200_rho03/thesis_checkpoint_N200.pth"

# ── data ──────────────────────────────────────────────────────────────────────
def generate_data(N, seed):
    np.random.seed(seed)
    xi1 = np.maximum(np.random.normal(6.0, 2.5, (N, R)), 0.0)
    xi2 = np.maximum(np.random.normal(2*(6.0-xi1)**2, 2.5), 0.0)
    return xi1, xi2

print("Generating data...")
xi1_test, xi2_test   = generate_data(N_TEST,  SEED_TEST)
xi1_train, xi2_train = generate_data(N_TRAIN, SEED_TRAIN)
xi1_test_t = torch.tensor(xi1_test, dtype=torch.float32)
print(f"  Test   N={N_TEST},  seed={SEED_TEST}:  xi1={xi1_test.mean():.3f}, xi2={xi2_test.mean():.3f}")
print(f"  Train  N={N_TRAIN}, seed={SEED_TRAIN}: xi1={xi1_train.mean():.3f}, xi2={xi2_train.mean():.3f}")

# ── cost ──────────────────────────────────────────────────────────────────────
def vr_np(xi1, xi2, Q1r, Q2):
    return np.maximum(
        b*(xi1+xi2-Q1r-Q2) - h*Q2,
        np.maximum(h*(Q1r-xi1-xi2), b*(xi1-Q1r) - h*xi2))

def total_cost(Q10, Q1r, Q2, xi1, xi2):
    return c*(Q10+Q1r.sum()) + h*Q10 + vr_np(xi1,xi2,Q1r,Q2).sum(axis=1)

def report(label, Q10, Q1r, Q2, xi1, xi2):
    costs = total_cost(Q10, Q1r, Q2, xi1, xi2)
    viol  = np.maximum(Q2.sum(axis=1) - Q10, 0.0)
    cap   = (viol <= 1e-5).mean()*100
    print(f"\n  {label}")
    print(f"    Q10={Q10:.4f}  Q1r={np.round(Q1r,3)}")
    print(f"    Mean={costs.mean():.4f}  Std={costs.std():.4f}  "
          f"Max={costs.max():.4f}  Cap={cap:.1f}%")
    return costs.mean(), costs.std(), costs.max()

# ── SAA LP on training data ───────────────────────────────────────────────────
def solve_saa(xi1_tr, xi2_tr):
    N = xi1_tr.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag",0); env.start()
    m = gp.Model(env=env)
    Q10v = m.addVar(lb=0.0)
    Q1rv = m.addVars(R, lb=0.0)
    Q2v  = m.addVars(N, R, lb=0.0)
    vv   = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i,r] for r in range(R)) <= Q10v)
        for r in range(R):
            m.addConstr(vv[i,r] >= b*(xi1_tr[i,r]+xi2_tr[i,r]-Q2v[i,r]-Q1rv[r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1rv[r]-xi1_tr[i,r]-xi2_tr[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1_tr[i,r]-Q1rv[r]) - h*xi2_tr[i,r])
    obj = (c*(Q10v + gp.quicksum(Q1rv[r] for r in range(R))) + h*Q10v
           + (1.0/N)*gp.quicksum(vv[i,r] for i in range(N) for r in range(R)))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    return Q10v.X, np.array([Q1rv[r].X for r in range(R)]), m.ObjVal

def saa_train_optimal_Q2(Q10, Q1r, xi1_tr, xi2_tr):
    N = xi1_tr.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag",0); env.start()
    m = gp.Model(env=env)
    Q2v = m.addVars(N, R, lb=0.0); vv = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i,r] for r in range(R)) <= Q10)
        for r in range(R):
            m.addConstr(vv[i,r] >= b*(xi1_tr[i,r]+xi2_tr[i,r]-Q2v[i,r]-Q1r[r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1r[r]-xi1_tr[i,r]-xi2_tr[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1_tr[i,r]-Q1r[r]) - h*xi2_tr[i,r])
    m.setObjective(gp.quicksum(vv[i,r] for i in range(N) for r in range(R)), GRB.MINIMIZE)
    m.optimize()
    return np.array([[Q2v[i,r].X for r in range(R)] for i in range(N)])

def saa_lp_clairvoyant(Q10, Q1r, xi1, xi2):
    N = xi1.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag",0); env.start()
    m = gp.Model(env=env)
    Q2v = m.addVars(N, R, lb=0.0); vv = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i,r] for r in range(R)) <= Q10)
        for r in range(R):
            m.addConstr(vv[i,r] >= b*(xi1[i,r]+xi2[i,r]-Q2v[i,r]-Q1r[r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1r[r]-xi1[i,r]-xi2[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1[i,r]-Q1r[r]) - h*xi2[i,r])
    m.setObjective(gp.quicksum(vv[i,r] for i in range(N) for r in range(R)), GRB.MINIMIZE)
    m.optimize()
    return np.array([[Q2v[i,r].X for r in range(R)] for i in range(N)])

def knn_Q2(xi1_test_, xi1_train, Q2_train, Q10, k=5):
    diff  = xi1_test_[:,None,:] - xi1_train[None,:,:]
    dists = np.sqrt((diff**2).sum(axis=2))
    idx   = np.argpartition(dists, k, axis=1)[:,:k]
    d     = np.take_along_axis(dists, idx, axis=1)
    w     = 1.0/np.maximum(d, 1e-9); w /= w.sum(axis=1, keepdims=True)
    Q2r   = (w[:,:,None]*Q2_train[idx]).sum(axis=1)
    Q2r   = np.maximum(Q2r, 0.0)
    s     = Q2r.sum(axis=1, keepdims=True)
    return Q2r * np.where(s > Q10, Q10/s, 1.0)

# ── neural ────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R*2+1, 32), nn.ReLU(),
            nn.Linear(32, 32),    nn.ReLU(),
            nn.Linear(32, R))
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))

    @torch.no_grad()
    def get_Q2(self, xi1_t):
        N   = xi1_t.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        net_in = torch.cat([xi1_t,
                            torch.tensor(q1r).unsqueeze(0).expand(N,-1),
                            torch.full((N,1), q10)], dim=1)
        z  = self.fc_layers(net_in).numpy()
        Q2 = np.maximum(z, 0.0)
        for i in range(N):
            if Q2[i].sum() > q10:
                lo, hi = 0.0, float(z[i].max())
                for _ in range(60):
                    mid = (lo+hi)/2
                    if np.maximum(z[i]-mid,0.0).sum() > q10: lo = mid
                    else: hi = mid
                Q2[i] = np.maximum(z[i]-hi, 0.0)
        return Q2, q10, q1r

# ── run ───────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  N=200 EVALUATION  (test N={N_TEST}, seed={SEED_TEST})")
print(f"{'='*65}")

# SAA on N=200 training data
print("\n[1] Solving SAA LP on N=200 training data...")
t0 = time.time()
Q10_saa, Q1r_saa, saa_obj = solve_saa(xi1_train, xi2_train)
print(f"    SAA obj={saa_obj:.4f}  Q10={Q10_saa:.4f}  Q1r={np.round(Q1r_saa,3)}  ({time.time()-t0:.1f}s)")

# SAA clairvoyant LB
print("\n[2] SAA-LP clairvoyant LB...")
Q2_lb = saa_lp_clairvoyant(Q10_saa, Q1r_saa, xi1_test, xi2_test)
lb_mean, lb_std, lb_max = report("SAA-LP clairvoyant (LB)", Q10_saa, Q1r_saa, Q2_lb, xi1_test, xi2_test)

# SAA-kNN k=5
print("\n[3] SAA-kNN k=5...")
Q2_tr_opt = saa_train_optimal_Q2(Q10_saa, Q1r_saa, xi1_train, xi2_train)
Q2_knn    = knn_Q2(xi1_test, xi1_train, Q2_tr_opt, Q10_saa, k=5)
knn_mean, knn_std, knn_max = report("SAA-kNN k=5", Q10_saa, Q1r_saa, Q2_knn, xi1_test, xi2_test)

# Neural nominal N=200
print("\n[4] Neural nominal (N=200)...")
model = Model()
model.load_state_dict(torch.load(CKPT_N200, map_location="cpu")["model_state_dict"], strict=False)
model.eval()
Q2_nom, q10n, q1rn = model.get_Q2(xi1_test_t)
nom_mean, nom_std, nom_max = report("Neural nominal (N=200)", q10n, q1rn, Q2_nom, xi1_test, xi2_test)

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*65}")
print(f"SUMMARY — N=200  (test N={N_TEST}, seed={SEED_TEST})")
print(f"{'='*65}")
print(f"\n  {'Method':<32} {'Mean cost':>10} {'Std':>8}  {'Gap vs LB':>10}")
print(f"  {'-'*62}")
for name, mean_c, std_c in [
    ("SAA-LP clairvoyant (LB)",  lb_mean,  lb_std),
    ("SAA-kNN k=5  (N=200)",     knn_mean, knn_std),
    ("Neural nominal (N=200)",   nom_mean, nom_std),
]:
    gap = (mean_c - lb_mean)/abs(lb_mean)*100
    print(f"  {name:<32} {mean_c:>10.4f} {std_c:>8.4f}  {gap:>+9.2f}%")

# N=50 results for comparison (from eval_seeds avg seeds 0-4)
print(f"\n  --- N=50 reference (avg seeds 0–4) ---")
n50_ref = [
    ("SAA-kNN k=5  (N=50)",    21.924),
    ("Neural nominal (N=50)",  21.578),
]
for name, mean_c in n50_ref:
    gap_vs_lb = (mean_c - lb_mean)/abs(lb_mean)*100
    print(f"  {name:<32} {mean_c:>10.4f}           {gap_vs_lb:>+9.2f}% (vs N=200 LB)")

print(f"\n  Neural improvement N=50→N=200: "
      f"{nom_mean - 21.578:+.4f}  ({(nom_mean-21.578)/21.578*100:+.2f}%)")
print(f"  kNN    improvement N=50→N=200: "
      f"{knn_mean - 21.924:+.4f}  ({(knn_mean-21.924)/21.924*100:+.2f}%)")
print(f"\n[Done]")
