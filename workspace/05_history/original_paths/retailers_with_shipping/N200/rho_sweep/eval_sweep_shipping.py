"""
eval_sweep_shipping.py  --  Evaluate shipping rho sweep: IS + OOS cost vs RHO
===============================================================================
For every completed rho_*/seed_*_summary.json, loads the final robust MP
checkpoint and computes:
    - In-sample cost  (N=200 training set, hard shipping)
  - Out-of-sample cost (N=2000 test set, seed=999, hard shipping)

Flat-line benchmarks (computed once, cached):
  - SAA MILP true lower bound (optimise over Q10,Q1r,Q2,z_ship jointly)
  - kNN-SAA (k=5): use SAA-MILP Q2 as training set, kNN map xi1->Q2

Saves: sweep_eval_shipping.json   (incremental, re-runnable)
"""

import os, json, argparse
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

NOMINAL_DIR = "/data/home/aviyac/retailers_with_shipping/N200"
SWEEP_DIR   = "/data/home/aviyac/retailers_with_shipping/N200/rho_sweep"
OUT_PATH    = os.path.join(SWEEP_DIR, "sweep_eval_shipping.json")

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Recompute all sweep entries")
parser.add_argument("--force-rhos", nargs="*", type=float, default=[],
                    help="Recompute only the listed rho values, e.g. --force-rhos 0.4 0.5")
args = parser.parse_args()

SEEDS    = list(range(10))   # use whatever are available
N_TRAIN  = 200
N_TEST   = 2000
SEED_TEST = 999

R        = 3
c, h, b  = 0.25, 0.05, 0.5
f_ship   = 0.1

# ── data ──────────────────────────────────────────────────────────────────────
def gen(N, seed):
    np.random.seed(seed)
    a   = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a2  = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

# ── cost ──────────────────────────────────────────────────────────────────────
def scenario_costs(Q10, Q1r, Q2, xi1, xi2):
    """Per-scenario total cost with hard shipping. Q2: (N,R). Returns (N,)."""
    z_ship = (Q2 > 1e-6).astype(float)          # (N, R)
    ship   = f_ship * z_ship.sum(axis=1)         # (N,)
    v = np.maximum(b * (xi1 + xi2 - Q1r - Q2) - h * Q2,
        np.maximum(h * (Q1r - xi1 - xi2),
                   b * (xi1 - Q1r) - h * xi2))  # (N, R)
    s1 = c * (Q10 + Q1r.sum()) + h * Q10        # scalar
    return s1 + ship + v.sum(axis=1)

def mean_cost(Q10, Q1r, Q2, xi1, xi2):
    return float(scenario_costs(Q10, Q1r, Q2, xi1, xi2).mean())

# ── Gurobi QP projection ──────────────────────────────────────────────────────
_grb_env = None
def _grb():
    global _grb_env
    if _grb_env is None:
        _grb_env = gp.Env(empty=True)
        _grb_env.setParam("OutputFlag", 0)
        _grb_env.start()
    return _grb_env

def gurobi_project(z_row, q10):
    m  = gp.Model(env=_grb())
    m.setParam("OutputFlag", 0)
    q2 = m.addVars(R, lb=0.0)
    m.addConstr(gp.quicksum(q2[r] for r in range(R)) <= q10)
    m.setObjective(
        gp.quicksum((q2[r] - float(z_row[r]))**2 for r in range(R)),
        GRB.MINIMIZE)
    m.optimize()
    result = (np.array([q2[r].X for r in range(R)])
              if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
              else np.zeros(R))
    m.dispose()
    return result

# ── model ─────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R * 2 + 1, 16), nn.ReLU(),
            nn.Linear(16, 16),         nn.ReLU(),
            nn.Linear(16, R))
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))

    @torch.no_grad()
    def get_Q2(self, xi1_t):
        N   = xi1_t.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        ni  = torch.cat([xi1_t,
                         torch.tensor(q1r, dtype=torch.float32).unsqueeze(0).expand(N, -1),
                         torch.full((N, 1), q10)], dim=1)
        z   = self.fc_layers(ni).numpy()
        Q2  = np.array([gurobi_project(z[i], q10) for i in range(N)])
        return Q2, q10, q1r

def load_model(path):
    m = Model()
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd["model_state_dict"], strict=False)
    m.eval()
    return m

# ── exact IS (MP objective with Gurobi Q2) ───────────────────────────────────
def exact_mp_is(state_dict, ap_json_paths, xi1_np, xi2_np):
    m = Model()
    m.load_state_dict(state_dict, strict=False)
    m.eval()
    q10 = torch.nn.functional.softplus(m.Q10).item()
    q1r = torch.nn.functional.softplus(m.Q1r).detach().numpy()

    adv1, adv2 = {}, {}
    for path in ap_json_paths:
        for s in json.load(open(path)):
            bid = s["ball_id"]
            adv1.setdefault(bid, []).append(np.array(s["worst_xi1"]))
            adv2.setdefault(bid, []).append(np.array(s["worst_xi2"]))
    if not adv1:
        xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
        Q2, _, _ = m.get_Q2(xi1_t)
        return mean_cost(q10, q1r, Q2, xi1_np, xi2_np)

    max_cols = max(len(v) for v in adv1.values())
    c1 = [xi1_np.copy() for _ in range(max_cols)]
    c2 = [xi2_np.copy() for _ in range(max_cols)]
    for bid in adv1:
        for k, (a1, a2) in enumerate(zip(adv1[bid], adv2[bid])):
            c1[k][bid] = a1
            c2[k][bid] = a2

    all1 = [xi1_np] + c1
    all2 = [xi2_np] + c2
    col_costs = []
    for cx1, cx2 in zip(all1, all2):
        Q2, _, _ = m.get_Q2(torch.tensor(cx1, dtype=torch.float32))
        col_costs.append(scenario_costs(q10, q1r, Q2, cx1, cx2))
    return float(np.stack(col_costs, axis=1).max(axis=1).mean())

# ── SAA MILP full (true lower bound) ─────────────────────────────────────────
def saa_milp_full(xi1, xi2):
    N = xi1.shape[0]
    BIG_M = float((xi1 + xi2).max()) + 1.0   # tight: max single-retailer demand
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model(env=env)
    Q10v = m.addVar(lb=0.0)
    Q1rv = m.addVars(R, lb=0.0)
    Q2v  = m.addVars(N, R, lb=0.0)
    zs   = m.addVars(N, R, vtype=GRB.BINARY)
    vv   = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i,r] for r in range(R)) <= Q10v)
        for r in range(R):
            m.addConstr(Q2v[i,r] <= BIG_M * zs[i,r])
            m.addConstr(Q2v[i,r] >= 1e-6 * zs[i,r])
            m.addConstr(vv[i,r] >= b*(xi1[i,r]+xi2[i,r]-Q1rv[r]-Q2v[i,r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1rv[r] - xi1[i,r] - xi2[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1[i,r] - Q1rv[r]) - h*xi2[i,r])
    m.setObjective(
        (c+h)*Q10v + c*gp.quicksum(Q1rv[r] for r in range(R))
        + (1.0/N)*gp.quicksum(
            vv[i,r] + f_ship*zs[i,r]
            for i in range(N) for r in range(R)),
        GRB.MINIMIZE)
    m.optimize()
    Q10_opt = Q10v.X
    Q1r_opt = np.array([Q1rv[r].X for r in range(R)])
    Q2_opt  = np.array([[Q2v[i,r].X for r in range(R)] for i in range(N)])
    return Q10_opt, Q1r_opt, Q2_opt

# ── kNN helper ────────────────────────────────────────────────────────────────
def knn_Q2(xi1_te, xi1_tr, Q2_tr, Q10, k=5):
    diff  = xi1_te[:, None, :] - xi1_tr[None, :, :]
    dists = np.sqrt((diff**2).sum(2))
    idx   = np.argpartition(dists, k, axis=1)[:, :k]
    d     = np.take_along_axis(dists, idx, axis=1)
    w     = 1.0 / np.maximum(d, 1e-9)
    w    /= w.sum(1, keepdims=True)
    Q2    = (w[:, :, None] * Q2_tr[idx]).sum(1)
    Q2    = np.maximum(Q2, 0.0)
    s     = Q2.sum(1, keepdims=True)
    return Q2 * np.where(s > Q10, Q10 / s, 1.0)

# ── generate test data ────────────────────────────────────────────────────────
print(f"Generating test data: N={N_TEST}, seed={SEED_TEST}")
xi1_test, xi2_test = gen(N_TEST, SEED_TEST)
xi1_test_t = torch.tensor(xi1_test, dtype=torch.float32)

# ── load existing cache ───────────────────────────────────────────────────────
_cache = {}
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        _cache = json.load(f)
    print(f"[Cache] Loaded existing results from {OUT_PATH}")

# ── benchmarks ────────────────────────────────────────────────────────────────
if "benchmarks" in _cache and _cache["benchmarks"].get("_done"):
    benchmarks = _cache["benchmarks"]
    print("[Cache] Reusing benchmarks.")
else:
    print("\nComputing per-seed SAA MILP lower bound + kNN benchmarks ...")
    # Per-seed IS lower bound + kNN OOS
    # Note: use stored saa_milp_obj from checkpoint where available.
    saa_lb_is, saa_knn_is, saa_knn_oos = [], [], []
    for seed in SEEDS:
        nom_path = os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth")
        if not os.path.exists(nom_path):
            print(f"  seed {seed}: nominal not ready, skipping")
            saa_lb_is.append(None); saa_knn_is.append(None); saa_knn_oos.append(None)
            continue
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)
        ck = torch.load(nom_path, map_location="cpu", weights_only=False)

        # IS lower bound: from saved saa_milp_obj (already computed during training)
        saa_lb_is.append(float(ck["saa_milp_obj"]))

        # kNN: solve MILP on training set -> use optimal Q2 as kNN training set
        Q10_s   = float(ck["saa_Q10"])
        Q1r_s   = np.array(ck["saa_Q1r"], dtype=float)
        print(f"  seed {seed}: MILP on training set (N={N_TRAIN}) ...", flush=True)
        Q10_lb_tr, Q1r_lb_tr, Q2_lb_tr = saa_milp_full(xi1_tr, xi2_tr)

        Q2_knn_tr = knn_Q2(xi1_tr, xi1_tr, Q2_lb_tr, Q10_lb_tr, k=5)
        saa_knn_is.append(mean_cost(Q10_s, Q1r_s, Q2_knn_tr, xi1_tr, xi2_tr))

        Q2_knn_te = knn_Q2(xi1_test, xi1_tr, Q2_lb_tr, Q10_s, k=5)
        saa_knn_oos.append(mean_cost(Q10_s, Q1r_s, Q2_knn_te, xi1_test, xi2_test))

        print(f"  seed {seed}: IS LB={saa_lb_is[-1]:.4f}  kNN OOS={saa_knn_oos[-1]:.4f}")

    benchmarks = {
        "_done": True,
        "saa_lb":  {"is": saa_lb_is,  "oos": saa_lb_is},   # IS LB as flat ref
        "saa_knn": {"is": saa_knn_is, "oos": saa_knn_oos},
    }
    _cache["benchmarks"] = benchmarks
    with open(OUT_PATH, "w") as f:
        json.dump({**_cache, "benchmarks": benchmarks}, f, indent=2)
    print(f"[Saved benchmarks] -> {OUT_PATH}")

# ── discover rho dirs ─────────────────────────────────────────────────────────
rho_dirs = {}
for d in sorted(os.listdir(SWEEP_DIR)):
    full = os.path.join(SWEEP_DIR, d)
    if d.startswith("rho_") and os.path.isdir(full):
        tag = d[4:]
        rho_val = float(tag.replace("p", "."))
        rho_dirs[rho_val] = full

all_rhos = sorted(rho_dirs)
print(f"\nFound {len(all_rhos)} rho directories: {all_rhos}")

# ── per-rho evaluation ────────────────────────────────────────────────────────
sweep_res = dict(_cache.get("sweep", {}))

for rho in all_rhos:
    rho_dir = rho_dirs[rho]
    force_this_rho = args.force or any(abs(rho - r) <= 1e-12 for r in args.force_rhos)
    avail_seeds = [s for s in SEEDS
                   if os.path.exists(os.path.join(rho_dir, f"seed_{s}_summary.json"))]
    cached = sweep_res.get(str(rho), {})
    if not force_this_rho and cached.get("n_seeds", 0) >= len(avail_seeds) and len(avail_seeds) > 0:
        print(f"[Cache] rho={rho:.3f}: {cached['n_seeds']} seeds — skipping")
        continue

    rob_is, rob_oos = [], []
    nom_is, nom_oos = [], []
    n_ok = 0

    for seed in SEEDS:
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)
        xi1_tr_t = torch.tensor(xi1_tr, dtype=torch.float32)
        nom_path = os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth")
        if not os.path.exists(nom_path):
            continue

        m_nom = load_model(nom_path)
        Q2_n_tr, q10n, q1rn = m_nom.get_Q2(xi1_tr_t)
        nom_is.append(mean_cost(q10n, q1rn, Q2_n_tr, xi1_tr, xi2_tr))
        Q2_n_te, _, _ = m_nom.get_Q2(xi1_test_t)
        nom_oos.append(mean_cost(q10n, q1rn, Q2_n_te, xi1_test, xi2_test))

        summ_file = os.path.join(rho_dir, f"seed_{seed}_summary.json")
        if not os.path.exists(summ_file):
            print(f"  [MISSING] rho={rho:.3f} seed={seed}")
            continue
        summ = json.load(open(summ_file))
        fmp = summ.get("final_mp_iter", 0)
        ccg_iters = summ.get("ccg_iters", fmp)

        if fmp == 0:
            rob_is.append(nom_is[-1])
            rob_oos.append(nom_oos[-1])
            n_ok += 1
            continue

        rob_path = os.path.join(rho_dir, f"seed_{seed}_mp_iter{fmp}.pth")
        if not os.path.exists(rob_path):
            print(f"  [MISSING MODEL] rho={rho:.3f} seed={seed}")
            continue

        ap_paths = [os.path.join(rho_dir, f"seed_{seed}_ap_iter{k}.json")
                    for k in range(1, ccg_iters+1)
                    if os.path.exists(os.path.join(rho_dir, f"seed_{seed}_ap_iter{k}.json"))]
        print(f"  IS: rho={rho:.3f} seed={seed} ({len(ap_paths)} AP cols)...", flush=True)
        rob_is_val = exact_mp_is(
            torch.load(rob_path, map_location="cpu")["model_state_dict"],
            ap_paths, xi1_tr, xi2_tr)
        rob_is.append(rob_is_val)

        m_rob = load_model(rob_path)
        Q2_r_te, q10r, q1rr = m_rob.get_Q2(xi1_test_t)
        rob_oos.append(mean_cost(q10r, q1rr, Q2_r_te, xi1_test, xi2_test))
        n_ok += 1

    sweep_res[str(rho)] = {
        "rob_is":  rob_is,
        "rob_oos": rob_oos,
        "nom_is":  nom_is,
        "nom_oos": nom_oos,
        "n_seeds": n_ok,
    }
    print(f"rho={rho:.3f}: IS={np.mean(rob_is):.4f}  OOS={np.mean(rob_oos):.4f}  (n={n_ok})")

    out = {"rhos": sorted(float(k) for k in sweep_res),
           "sweep": sweep_res, "benchmarks": benchmarks,
           "meta": {"N_train": N_TRAIN, "N_test": N_TEST,
                    "seed_test": SEED_TEST, "seeds": SEEDS}}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [Saved] {OUT_PATH}")

# final save
out = {"rhos": sorted(float(k) for k in sweep_res),
       "sweep": sweep_res, "benchmarks": benchmarks,
       "meta": {"N_train": N_TRAIN, "N_test": N_TEST,
                "seed_test": SEED_TEST, "seeds": SEEDS}}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[Done] {OUT_PATH}")
