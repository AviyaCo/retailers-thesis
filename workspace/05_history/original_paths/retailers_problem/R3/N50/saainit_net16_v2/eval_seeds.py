"""
eval_seeds.py — Out-of-sample evaluation for seeds 0-9 (saainit_net16_v2)
==========================================================================
For each seed evaluates 4 policies on N=2000 fresh test samples (seed=999):
  1. SAA-LP (clairvoyant LB) — seed's Q10/Q1r, optimal Q2 using test xi2
  2. SAA-kNN (k=5)           — seed's Q10/Q1r, Q2 from 5-NN on training data
  3. Neural nominal           — seed_{k}_nominal.pth
  4. Neural robust            — seed_{k}_mp_iter{final}.pth

Distribution: TruncatedNormal (matching paper's Julia code)
Network width: 16
"""

import os, json, time
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

BASE_DIR = "/data/home/aviyac/retailers_problem/R3/N50/saainit_net16_v2"

# ── problem constants ───────────────────────────────────────────────────────
R = 3; c = 0.25; h = 0.05; b = 0.5
N_TRAIN = 50     # training set size per seed
N_TEST  = 2000   # out-of-sample evaluation set (never seen during training)
SEED_TEST = 999

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# final MP iter per seed — update after runs complete
FINAL_MP_ITER = {0: 4, 1: 4, 2: 4, 3: 5, 4: 5, 5: 4, 6: 5, 7: 4, 8: 3, 9: 7}

# ── data generation (TruncatedNormal — matches paper's Julia code) ──────────
def generate_data(N, seed):
    np.random.seed(seed)
    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

print(f"Generating test data: N={N_TEST}, seed={SEED_TEST}")
xi1_test, xi2_test = generate_data(N_TEST, SEED_TEST)
print(f"  xi1 mean={xi1_test.mean():.3f}, xi2 mean={xi2_test.mean():.3f}")

# ── cost helpers ─────────────────────────────────────────────────────────────
def vr_np(xi1, xi2, Q1r, Q2):
    return np.maximum(
        b*(xi1 + xi2 - Q1r - Q2) - h*Q2,
        np.maximum(h*(Q1r - xi1 - xi2),
                   b*(xi1 - Q1r) - h*xi2))

def total_cost(Q10, Q1r, Q2, xi1, xi2):
    stage1 = c*(Q10 + Q1r.sum()) + h*Q10
    return stage1 + vr_np(xi1, xi2, Q1r, Q2).sum(axis=1)

def report(label, Q10, Q1r, Q2, xi1, xi2):
    costs = total_cost(Q10, Q1r, Q2, xi1, xi2)
    viol  = np.maximum(Q2.sum(axis=1) - Q10, 0.0)
    cap   = (viol <= 1e-5).mean() * 100
    print(f"    {label}")
    print(f"      Q10={Q10:.4f}  Q1r={np.round(Q1r,3)}")
    print(f"      Mean cost={costs.mean():.4f}  Std={costs.std():.4f}  "
          f"Max={costs.max():.4f}  Cap.feas={cap:.1f}%")
    return costs.mean(), costs.std(), costs.max()

def insample_mean(Q10, Q1r, Q2, xi1, xi2):
    return total_cost(Q10, Q1r, Q2, xi1, xi2).mean()

# ── SAA clairvoyant LP ───────────────────────────────────────────────────────
def saa_lp_clairvoyant(Q10, Q1r, xi1, xi2):
    """Optimal Q2 given test xi2 — lower bound."""
    N = xi1.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model(env=env)
    Q2v = m.addVars(N, R, lb=0.0)
    vv  = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i,r] for r in range(R)) <= Q10)
        for r in range(R):
            m.addConstr(vv[i,r] >= b*(xi1[i,r]+xi2[i,r]-Q2v[i,r]-Q1r[r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1r[r]-xi1[i,r]-xi2[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1[i,r]-Q1r[r]) - h*xi2[i,r])
    m.setObjective(gp.quicksum(vv[i,r] for i in range(N) for r in range(R)), GRB.MINIMIZE)
    m.optimize()
    return np.array([[Q2v[i,r].X for r in range(R)] for i in range(N)])

# ── kNN Q2 ───────────────────────────────────────────────────────────────────
def knn_Q2(xi1_test_, xi1_train, Q2_train, Q10, k=5):
    """Predict Q2 for test samples via k-NN on training (xi1, Q2*) pairs."""
    diff = xi1_test_[:, None, :] - xi1_train[None, :, :]   # (N_test, N_train, R)
    dists = np.sqrt((diff**2).sum(axis=2))                  # (N_test, N_train)
    knn_idx   = np.argpartition(dists, k, axis=1)[:, :k]
    knn_dists = np.take_along_axis(dists, knn_idx, axis=1)
    eps = 1e-9
    inv_d   = 1.0 / np.maximum(knn_dists, eps)
    weights = inv_d / inv_d.sum(axis=1, keepdims=True)
    Q2_raw  = (weights[:, :, None] * Q2_train[knn_idx]).sum(axis=1)
    Q2_raw  = np.maximum(Q2_raw, 0.0)
    s       = Q2_raw.sum(axis=1, keepdims=True)
    Q2_raw *= np.where(s > Q10, Q10 / s, 1.0)
    return Q2_raw

def saa_train_optimal_Q2(Q10, Q1r, xi1_tr, xi2_tr):
    """Solve SAA recourse LP on training data to get optimal (xi1_tr, Q2*) pairs."""
    N = xi1_tr.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model(env=env)
    Q2v = m.addVars(N, R, lb=0.0)
    vv  = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i,r] for r in range(R)) <= Q10)
        for r in range(R):
            m.addConstr(vv[i,r] >= b*(xi1_tr[i,r]+xi2_tr[i,r]-Q2v[i,r]-Q1r[r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1r[r]-xi1_tr[i,r]-xi2_tr[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1_tr[i,r]-Q1r[r]) - h*xi2_tr[i,r])
    m.setObjective(gp.quicksum(vv[i,r] for i in range(N) for r in range(R)), GRB.MINIMIZE)
    m.optimize()
    return np.array([[Q2v[i,r].X for r in range(R)] for i in range(N)])

# ── neural model (width=16) ──────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R*2+1, 16), nn.ReLU(),
            nn.Linear(16, 16),    nn.ReLU(),
            nn.Linear(16, R)
        )
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))

    @torch.no_grad()
    def get_Q2(self, xi1_t):
        """Analytical projection via binary search."""
        N   = xi1_t.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        q1r_in = torch.tensor(q1r).unsqueeze(0).expand(N, -1)
        q10_in = torch.full((N, 1), q10)
        net_in = torch.cat([xi1_t, q1r_in, q10_in], dim=1)
        z = self.fc_layers(net_in).numpy()
        Q2 = np.maximum(z, 0.0)
        for i in range(N):
            if Q2[i].sum() > q10:
                lo, hi = 0.0, float(z[i].max())
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if np.maximum(z[i] - mid, 0.0).sum() > q10:
                        lo = mid
                    else:
                        hi = mid
                Q2[i] = np.maximum(z[i] - hi, 0.0)
        return Q2, q10, q1r

def load_neural(path):
    m = Model()
    ck = torch.load(path, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.eval()
    return m

# ── main evaluation loop ──────────────────────────────────────────────────────
xi1_test_t = torch.tensor(xi1_test, dtype=torch.float32)

all_results = {}

for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"  SEED {seed}")
    print(f"{'='*70}")

    # Load summary
    summ_path = os.path.join(BASE_DIR, f"seed_{seed}_summary.json")
    if not os.path.exists(summ_path):
        print(f"  [SKIP] no summary.json for seed {seed}")
        continue
    with open(summ_path) as f:
        summ = json.load(f)
    Q10_saa = summ["saa_Q10"]
    Q1r_saa = np.array(summ["saa_Q1r"])
    final_mp = FINAL_MP_ITER[seed]

    print(f"  SAA: Q10={Q10_saa:.4f}, Q1r={np.round(Q1r_saa,3)}, final_mp_iter={final_mp}")

    # --- 1. SAA LB (clairvoyant) ---
    print(f"\n  [1] SAA-LP clairvoyant LB ...")
    t0 = time.time()
    Q2_lb = saa_lp_clairvoyant(Q10_saa, Q1r_saa, xi1_test, xi2_test)
    print(f"      (LP solved in {time.time()-t0:.1f}s)")
    lb_mean, lb_std, lb_max = report("SAA-LP (clairvoyant, LB)", Q10_saa, Q1r_saa, Q2_lb, xi1_test, xi2_test)

    # --- 2. SAA kNN (k=5) ---
    print(f"\n  [2] SAA-kNN k=5 ...")
    xi1_tr, xi2_tr = generate_data(N_TRAIN, seed)
    Q2_tr_opt = saa_train_optimal_Q2(Q10_saa, Q1r_saa, xi1_tr, xi2_tr)
    Q2_knn = knn_Q2(xi1_test, xi1_tr, Q2_tr_opt, Q10_saa, k=5)
    knn_mean, knn_std, knn_max = report("SAA-kNN (k=5)", Q10_saa, Q1r_saa, Q2_knn, xi1_test, xi2_test)

    # --- 3. Neural nominal ---
    print(f"\n  [3] Neural nominal ...")
    nom_path = os.path.join(BASE_DIR, f"seed_{seed}_nominal.pth")
    m_nom = load_neural(nom_path)
    Q2_nom, Q10_nom, Q1r_nom = m_nom.get_Q2(xi1_test_t)
    nom_mean, nom_std, nom_max = report("Neural nominal", Q10_nom, Q1r_nom, Q2_nom, xi1_test, xi2_test)

    # --- 4. Neural robust (final MP iter) ---
    print(f"\n  [4] Neural robust (mp_iter{final_mp}) ...")
    rob_path = os.path.join(BASE_DIR, f"seed_{seed}_mp_iter{final_mp}.pth")
    if not os.path.exists(rob_path):
        print(f"  [SKIP] {rob_path} not found — check FINAL_MP_ITER dict")
        continue
    m_rob = load_neural(rob_path)
    Q2_rob, Q10_rob, Q1r_rob = m_rob.get_Q2(xi1_test_t)
    rob_mean, rob_std, rob_max = report("Neural robust", Q10_rob, Q1r_rob, Q2_rob, xi1_test, xi2_test)

    # --- In-sample costs (on training set) ---
    xi1_tr_t = torch.tensor(xi1_tr, dtype=torch.float32)
    lb_is   = insample_mean(Q10_saa, Q1r_saa, Q2_tr_opt, xi1_tr, xi2_tr)
    Q2_knn_tr = knn_Q2(xi1_tr, xi1_tr, Q2_tr_opt, Q10_saa, k=5)
    knn_is  = insample_mean(Q10_saa, Q1r_saa, Q2_knn_tr, xi1_tr, xi2_tr)
    Q2_nom_tr, _, _ = m_nom.get_Q2(xi1_tr_t)
    nom_is  = insample_mean(Q10_nom, Q1r_nom, Q2_nom_tr, xi1_tr, xi2_tr)
    Q2_rob_tr, _, _ = m_rob.get_Q2(xi1_tr_t)
    rob_is  = insample_mean(Q10_rob, Q1r_rob, Q2_rob_tr, xi1_tr, xi2_tr)

    # --- Mini table for this seed ---
    print(f"\n  {'Method':<28} {'InSample':>9} {'Mean':>9} {'Std':>9} {'Max':>9}  {'Gap vs LB':>10}")
    print(f"  {'-'*80}")
    for name, is_c, mean_c, std_c, max_c in [
        ("SAA-LP clairvoyant (LB)",  lb_is,  lb_mean,  lb_std,  lb_max),
        ("SAA-kNN k=5",              knn_is, knn_mean, knn_std, knn_max),
        ("Neural nominal",           nom_is, nom_mean, nom_std, nom_max),
        ("Neural robust",            rob_is, rob_mean, rob_std, rob_max),
    ]:
        gap = (mean_c - lb_mean) / abs(lb_mean) * 100
        print(f"  {name:<28} {is_c:>9.4f} {mean_c:>9.4f} {std_c:>9.4f} {max_c:>9.4f}  {gap:>+9.2f}%")

    all_results[seed] = {
        "lb":  (lb_mean,  lb_std,  lb_max,  lb_is),
        "knn": (knn_mean, knn_std, knn_max, knn_is),
        "nom": (nom_mean, nom_std, nom_max, nom_is),
        "rob": (rob_mean, rob_std, rob_max, rob_is),
    }

# ── aggregate table ───────────────────────────────────────────────────────────
if all_results:
    print(f"\n\n{'='*80}")
    print(f"AGGREGATE  (seeds {list(all_results.keys())}, N_TEST={N_TEST}, seed_test={SEED_TEST})")
    print(f"{'='*80}")

    completed = list(all_results.keys())
    for method_key, method_name in [("lb","SAA-LP clairvoyant (LB)"), ("knn","SAA-kNN k=5"),
                                      ("nom","Neural nominal"), ("rob","Neural robust")]:
        means   = [all_results[s][method_key][0] for s in completed]
        stds    = [all_results[s][method_key][1] for s in completed]
        is_vals = [all_results[s][method_key][3] for s in completed]
        print(f"\n  {method_name}")
        print(f"    Per-seed in-sample: {[f'{v:.4f}' for v in is_vals]}")
        print(f"    In-sample avg:      mean={np.mean(is_vals):.4f}  std={np.std(is_vals):.4f}")
        print(f"    Per-seed OOS means: {[f'{m:.4f}' for m in means]}")
        print(f"    OOS avg:            mean={np.mean(means):.4f}  std={np.std(means):.4f}")

print(f"\n[Done]")
