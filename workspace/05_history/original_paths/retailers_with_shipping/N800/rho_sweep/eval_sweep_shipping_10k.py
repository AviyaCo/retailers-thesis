"""
eval_sweep_shipping_10k.py  --  OOS / IS evaluation for shipping rho sweep.
============================================================================
For every completed rho_*/seed_*_summary.json, loads the final robust MP
checkpoint and computes:
  - IS cost  : exact MP objective (max over nominal + adversarial scenarios)
               using HARD shipping  (z_ship = 1 iff q2[r] > 1e-6)
  - OOS cost : mean cost on N_TEST=10000 test scenarios, hard shipping

Benchmarks (computed once per seed from nominal checkpoints):
  - SAA-NN OOS  : nominal model evaluated on test set (hard shipping)
    - SAA-kNN OOS : kNN recourse map built from the training SAA-MILP recourse,
                                    with k selected by leave-one-out training validation
  - SAA-MILP    : clairvoyant MILP lower bound (cached in seed_*_summary.json)

Output: sweep_eval_shipping_10k.json   (incremental, re-runnable)
"""

import os, json, argparse
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

NOMINAL_DIR = "/data/home/aviyac/retailers_with_shipping/N800"
SWEEP_DIR   = "/data/home/aviyac/retailers_with_shipping/N800/rho_sweep"
OUT_PATH    = os.path.join(SWEEP_DIR, "sweep_eval_shipping_10k.json")

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Recompute all sweep entries")
parser.add_argument("--force-rhos", nargs="*", type=float, default=[],
                    help="Recompute only the listed rho values, e.g. --force-rhos 0.4 0.5")
args = parser.parse_args()

SEEDS     = list(range(10))
N_TRAIN   = 800
N_TEST    = 10000
SEED_TEST = 9999

R       = 3
c, h, b = 0.25, 0.05, 0.5
f_ship  = 0.1
KNN_K_GRID = [k for k in (1, 3, 5, 7, 9, 11, 15, 20, 25, 35, 50, 75, 100, 150, 200, 300, 500, 800) if k < N_TRAIN]
if N_TRAIN - 1 not in KNN_K_GRID:
    KNN_K_GRID.append(N_TRAIN - 1)
KNN_SELECTION_VERSION = "shipping_per_seed_loo_grid_v1"

# ── Data ──────────────────────────────────────────────────────────────────────
def gen(N, seed):
    np.random.seed(seed)
    a   = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a2  = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

# ── Cost with hard shipping ────────────────────────────────────────────────────
def scenario_costs(Q10, Q1r, Q2, xi1, xi2):
    """Per-scenario cost with hard (binary) shipping. Returns (N,)."""
    z_ship = (Q2 > 1e-6).astype(float)
    ship   = f_ship * z_ship.sum(axis=1)
    v = np.maximum(b * (xi1 + xi2 - Q1r - Q2) - h * Q2,
        np.maximum(h * (Q1r - xi1 - xi2),
                   b * (xi1 - Q1r) - h * xi2))
    s1 = c * (Q10 + float(Q1r.sum())) + h * Q10
    return s1 + ship + v.sum(axis=1)

def mean_cost(Q10, Q1r, Q2, xi1, xi2):
    return float(scenario_costs(Q10, Q1r, Q2, xi1, xi2).mean())

# ── Gurobi projection: Q2* = argmin ||Q2-z||^2  s.t. Q2>=0, sum(Q2)<=Q10 ─────
_grb_env = None
def _grb():
    global _grb_env
    if _grb_env is None:
        _grb_env = gp.Env(empty=True)
        _grb_env.setParam("OutputFlag", 0)
        _grb_env.start()
    return _grb_env

def _project(z_row, q10):
    m  = gp.Model(env=_grb())
    m.setParam("OutputFlag", 0)
    q2 = m.addVars(R, lb=0.0)
    m.addConstr(gp.quicksum(q2[r] for r in range(R)) <= q10)
    m.setObjective(
        gp.quicksum((q2[r] - float(z_row[r]))**2 for r in range(R)),
        GRB.MINIMIZE)
    m.optimize()
    res = np.array([q2[r].X for r in range(R)]) if m.status == GRB.OPTIMAL else np.zeros(R)
    m.dispose()
    return res

# ── Model (must match run_rho_seed.py exactly) ────────────────────────────────
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
                         torch.tensor(q1r).unsqueeze(0).expand(N, -1),
                         torch.full((N, 1), q10)], dim=1)
        z   = self.fc_layers(ni).numpy()
        Q2  = np.array([_project(z[i], q10) for i in range(N)])
        return Q2, q10, q1r

def load_model(path):
    m = Model()
    ck = torch.load(path, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.eval()
    return m

# ── Exact IS: max over nominal + adversarial columns, hard shipping ────────────
def exact_mp_is(state_dict, ap_json_paths, xi1_np, xi2_np):
    m = Model()
    m.load_state_dict(state_dict, strict=False)
    m.eval()
    q10 = torch.nn.functional.softplus(m.Q10).item()
    q1r = torch.nn.functional.softplus(m.Q1r).detach().numpy()

    adv_xi1, adv_xi2 = {}, {}
    for path in ap_json_paths:
        with open(path) as f:
            for s in json.load(f):
                bid = s["ball_id"]
                adv_xi1.setdefault(bid, []).append(np.array(s["worst_xi1"]))
                adv_xi2.setdefault(bid, []).append(np.array(s["worst_xi2"]))

    max_cols = max(len(v) for v in adv_xi1.values()) if adv_xi1 else 0
    all_xi1 = [xi1_np] + [xi1_np.copy() for _ in range(max_cols)]
    all_xi2 = [xi2_np] + [xi2_np.copy() for _ in range(max_cols)]
    for bid in adv_xi1:
        for k, (ax1, ax2) in enumerate(zip(adv_xi1[bid], adv_xi2[bid])):
            all_xi1[k + 1][bid] = ax1
            all_xi2[k + 1][bid] = ax2

    col_costs = []
    for cx1, cx2 in zip(all_xi1, all_xi2):
        Q2_col, _, _ = m.get_Q2(torch.tensor(cx1, dtype=torch.float32))
        col_costs.append(scenario_costs(q10, q1r, Q2_col, cx1, cx2))

    per_sample_max = np.stack(col_costs, axis=1).max(axis=1)
    return float(per_sample_max.mean())

# ── SAA-MILP recourse + kNN benchmark ─────────────────────────────────────────
def saa_milp_full(xi1, xi2):
    N = xi1.shape[0]
    big_m = float((xi1 + xi2).max()) + 1.0
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    q10 = m.addVar(lb=0.0)
    q1r = m.addVars(R, lb=0.0)
    q2 = m.addVars(N, R, lb=0.0)
    z_ship = m.addVars(N, R, vtype=GRB.BINARY)
    v = m.addVars(N, R, lb=-GRB.INFINITY)

    for i in range(N):
        m.addConstr(gp.quicksum(q2[i, r] for r in range(R)) <= q10)
        for r in range(R):
            m.addConstr(q2[i, r] <= big_m * z_ship[i, r])
            m.addConstr(q2[i, r] >= 1e-6 * z_ship[i, r])
            m.addConstr(v[i, r] >= b * (xi1[i, r] + xi2[i, r] - q1r[r] - q2[i, r]) - h * q2[i, r])
            m.addConstr(v[i, r] >= h * (q1r[r] - xi1[i, r] - xi2[i, r]))
            m.addConstr(v[i, r] >= b * (xi1[i, r] - q1r[r]) - h * xi2[i, r])

    m.setObjective(
        (c + h) * q10
        + c * gp.quicksum(q1r[r] for r in range(R))
        + (1.0 / N) * gp.quicksum(
            v[i, r] + f_ship * z_ship[i, r]
            for i in range(N) for r in range(R)
        ),
        GRB.MINIMIZE,
    )
    m.optimize()
    q10_opt = float(q10.X)
    q1r_opt = np.array([q1r[r].X for r in range(R)], dtype=float)
    q2_opt = np.array([[q2[i, r].X for r in range(R)] for i in range(N)], dtype=float)
    m.dispose()
    env.dispose()
    return q10_opt, q1r_opt, q2_opt

def knn_Q2(xi1_te, xi1_tr, Q2_tr, Q10, k=5):
    k = int(k)
    if k < 1 or k > xi1_tr.shape[0]:
        raise ValueError(f"k must be in [1, {xi1_tr.shape[0]}], got {k}")
    diff = xi1_te[:, None, :] - xi1_tr[None, :, :]
    dists = np.sqrt((diff ** 2).sum(2))
    idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
    d = np.take_along_axis(dists, idx, axis=1)
    w = 1.0 / np.maximum(d, 1e-9)
    w /= w.sum(1, keepdims=True)
    Q2 = (w[:, :, None] * Q2_tr[idx]).sum(1)
    Q2 = np.maximum(Q2, 0.0)
    s = Q2.sum(1, keepdims=True)
    scale = np.ones_like(s)
    np.divide(Q10, s, out=scale, where=s > Q10)
    return Q2 * scale

def knn_Q2_loo(xi1_tr, Q2_tr, Q10, k):
    k = int(k)
    n_train = xi1_tr.shape[0]
    if k < 1 or k >= n_train:
        raise ValueError(f"k must be in [1, {n_train - 1}] for leave-one-out, got {k}")
    diff = xi1_tr[:, None, :] - xi1_tr[None, :, :]
    dists = np.sqrt((diff ** 2).sum(2))
    np.fill_diagonal(dists, np.inf)
    idx = np.argpartition(dists, k - 1, axis=1)[:, :k]
    d = np.take_along_axis(dists, idx, axis=1)
    w = 1.0 / np.maximum(d, 1e-9)
    w /= w.sum(1, keepdims=True)
    Q2 = (w[:, :, None] * Q2_tr[idx]).sum(1)
    Q2 = np.maximum(Q2, 0.0)
    s = Q2.sum(1, keepdims=True)
    scale = np.ones_like(s)
    np.divide(Q10, s, out=scale, where=s > Q10)
    return Q2 * scale

def summarize_knn_grid(knn_by_k):
    for vals in knn_by_k.values():
        vals["mean_is"] = float(np.mean(vals["is"]))
        vals["mean_val"] = float(np.mean(vals["val"]))
        vals["mean_oos"] = float(np.mean(vals["oos"]))

# ── Test data ─────────────────────────────────────────────────────────────────
print(f"Generating test data: N={N_TEST}, seed={SEED_TEST}")
xi1_test, xi2_test = gen(N_TEST, SEED_TEST)
xi1_test_t = torch.tensor(xi1_test, dtype=torch.float32)

# ── Load cache ────────────────────────────────────────────────────────────────
_cache = {}
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        _cache = json.load(f)
    print(f"[Cache] Loaded existing results from {OUT_PATH}")

# ── Benchmarks ────────────────────────────────────────────────────────────────
if "benchmarks" in _cache and "saa_nn" in _cache["benchmarks"] and "saa_milp" in _cache["benchmarks"]:
    benchmarks = _cache["benchmarks"]
    print("[Cache] Reusing benchmarks.")
else:
    print("\nComputing SAA-NN benchmarks ...")
    saa_nn_is, saa_nn_oos, saa_milp_vals = [], [], []
    for seed in SEEDS:
        nom_path = os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth")
        m_nom    = load_model(nom_path)
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)
        Q2_tr, q10, q1r = m_nom.get_Q2(torch.tensor(xi1_tr, dtype=torch.float32))
        saa_nn_is.append(mean_cost(q10, q1r, Q2_tr, xi1_tr, xi2_tr))
        Q2_te, q10, q1r = m_nom.get_Q2(xi1_test_t)
        saa_nn_oos.append(mean_cost(q10, q1r, Q2_te, xi1_test, xi2_test))
        with open(os.path.join(NOMINAL_DIR, f"seed_{seed}_summary.json")) as f:
            saa_milp_vals.append(json.load(f)["saa_milp"])
        print(f"  seed {seed}: IS={saa_nn_is[-1]:.4f}  OOS={saa_nn_oos[-1]:.4f}  "
              f"SAA-MILP={saa_milp_vals[-1]:.4f}")
    benchmarks = {
        "saa_nn":   {"is": saa_nn_is,   "oos": saa_nn_oos},
        "saa_milp": {"is": saa_milp_vals},
    }

knn_current = (
    benchmarks.get("_knn_k_selection") == KNN_SELECTION_VERSION
    and "selected_k_by_seed" in benchmarks.get("saa_knn", {})
    and len(benchmarks["saa_knn"]["selected_k_by_seed"]) == len(SEEDS)
    and len(benchmarks["saa_knn"].get("oos", [])) == len(SEEDS)
)

if not knn_current:
    print("\nSelecting SAA-kNN k by leave-one-out training validation; evaluating on 10k test set ...")
    knn_by_k = {str(k): {"is": [], "val": [], "oos": []} for k in KNN_K_GRID}
    selected_k_by_seed, selected_is, selected_val, selected_oos = [], [], [], []
    for seed in SEEDS:
        nom_path = os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth")
        ck = torch.load(nom_path, map_location="cpu", weights_only=False)
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)
        q10_s = float(ck["saa_Q10"])
        q1r_s = np.array(ck["saa_Q1r"], dtype=float)

        print(f"  seed {seed}: training SAA-MILP recourse + selecting k ...", flush=True)
        _, _, q2_lb = saa_milp_full(xi1_tr, xi2_tr)
        for k in KNN_K_GRID:
            q2_knn_tr = knn_Q2(xi1_tr, xi1_tr, q2_lb, q10_s, k=k)
            knn_by_k[str(k)]["is"].append(mean_cost(q10_s, q1r_s, q2_knn_tr, xi1_tr, xi2_tr))

            q2_knn_val = knn_Q2_loo(xi1_tr, q2_lb, q10_s, k=k)
            knn_by_k[str(k)]["val"].append(mean_cost(q10_s, q1r_s, q2_knn_val, xi1_tr, xi2_tr))

            q2_knn_te = knn_Q2(xi1_test, xi1_tr, q2_lb, q10_s, k=k)
            knn_by_k[str(k)]["oos"].append(mean_cost(q10_s, q1r_s, q2_knn_te, xi1_test, xi2_test))

        best_k = min(KNN_K_GRID, key=lambda k: (knn_by_k[str(k)]["val"][-1], k))
        selected_k_by_seed.append(best_k)
        selected_is.append(knn_by_k[str(best_k)]["is"][-1])
        selected_val.append(knn_by_k[str(best_k)]["val"][-1])
        selected_oos.append(knn_by_k[str(best_k)]["oos"][-1])
        print(f"    best k={best_k}  LOO={selected_val[-1]:.4f}  OOS={selected_oos[-1]:.4f}")

    summarize_knn_grid(knn_by_k)
    print("  kNN grid mean LOO: " + ", ".join(
        f"k={k}: {knn_by_k[str(k)]['mean_val']:.4f}" for k in KNN_K_GRID))
    print("  kNN grid mean 10k OOS: " + ", ".join(
        f"k={k}: {knn_by_k[str(k)]['mean_oos']:.4f}" for k in KNN_K_GRID))
    selected_k_counts = {str(k): selected_k_by_seed.count(k) for k in sorted(set(selected_k_by_seed))}
    selected_k = selected_k_by_seed[0] if len(selected_k_counts) == 1 else None
    print(f"  Selected k by seed: {selected_k_by_seed}")
    print(f"  Mean selected 10k OOS: {np.mean(selected_oos):.4f}")

    benchmarks["_knn_k_selection"] = KNN_SELECTION_VERSION
    benchmarks["knn_k_grid"] = KNN_K_GRID
    benchmarks["saa_knn"] = {
        "is": selected_is,
        "val": selected_val,
        "oos": selected_oos,
        "selected_k": selected_k,
        "selected_k_by_seed": selected_k_by_seed,
        "selected_k_counts": selected_k_counts,
        "selection_metric": "per_seed_loo_validation_cost",
    }
    benchmarks["saa_knn_by_k"] = knn_by_k
    with open(OUT_PATH, "w") as f:
        json.dump({"sweep": _cache.get("sweep", {}), "benchmarks": benchmarks,
                   "meta": {"N_train": N_TRAIN, "N_test": N_TEST,
                             "seed_test": SEED_TEST, "seeds": SEEDS}}, f, indent=2)
    print(f"[Saved SAA-kNN CV benchmarks] {OUT_PATH}")

# ── Discover rho directories ──────────────────────────────────────────────────
rho_dirs = {}
for entry in sorted(os.listdir(SWEEP_DIR)):
    full = os.path.join(SWEEP_DIR, entry)
    if entry.startswith("rho_") and os.path.isdir(full):
        rho = float(entry[4:].replace("p", "."))
        rho_dirs[rho] = full
rho_dirs[0.0] = NOMINAL_DIR   # rho=0: nominal model

all_rhos = sorted(rho_dirs.keys())
print(f"\nFound {len(all_rhos)} rho values: {all_rhos}")

# ── Evaluate ──────────────────────────────────────────────────────────────────
sweep_res = dict(_cache.get("sweep", {}))

for rho in all_rhos:
    rho_dir = rho_dirs[rho]
    is_nominal = (rho == 0.0)
    force_this_rho = args.force or any(abs(rho - r) <= 1e-12 for r in args.force_rhos)

    completed_seeds = []
    for seed in SEEDS:
        if is_nominal:
            completed_seeds.append(seed)
            continue
        summ_path = os.path.join(rho_dir, f"seed_{seed}_summary.json")
        if not os.path.exists(summ_path):
            continue
        with open(summ_path) as f:
            if json.load(f).get("completed") is True:
                completed_seeds.append(seed)

    if len(completed_seeds) < len(SEEDS):
        sweep_res.pop(str(rho), None)
        print(f"[SKIP INCOMPLETE] rho={rho:.4f}: {len(completed_seeds)}/{len(SEEDS)} seeds complete")
        continue

    cached = sweep_res.get(str(rho), {})
    if not force_this_rho and cached.get("n_seeds", 0) == len(SEEDS):
        print(f"[Cache] rho={rho:.4f}: {cached['n_seeds']} seeds — skipping")
        continue

    rob_is, rob_oos = [], []
    n_ok = 0

    for seed in completed_seeds:
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)

        if is_nominal:
            m_rob = load_model(os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth"))
            ap_paths = []
        else:
            summ_path = os.path.join(rho_dir, f"seed_{seed}_summary.json")
            if not os.path.exists(summ_path):
                print(f"  [MISSING] rho={rho:.4f} seed={seed}")
                continue
            with open(summ_path) as f:
                summ = json.load(f)
            fmp = summ.get("final_mp_iter", 0)
            ccg_iters = summ.get("ccg_iters", fmp)
            if fmp == 0:
                print(f"  [NO MP] rho={rho:.4f} seed={seed} — using nominal")
                m_rob = load_model(os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth"))
                ap_paths = []
            else:
                rob_ck = os.path.join(rho_dir, f"seed_{seed}_mp_iter{fmp}.pth")
                if not os.path.exists(rob_ck):
                    print(f"  [MISSING MODEL] rho={rho:.4f} seed={seed}")
                    continue
                m_rob    = load_model(rob_ck)
                ap_paths = [os.path.join(rho_dir, f"seed_{seed}_ap_iter{k}.json")
                            for k in range(1, ccg_iters + 1)
                            if os.path.exists(
                                os.path.join(rho_dir, f"seed_{seed}_ap_iter{k}.json"))]

        print(f"  rho={rho:.4f} seed={seed}: IS ({len(ap_paths)} AP cols) ...", flush=True)
        is_val = exact_mp_is(m_rob.state_dict(), ap_paths, xi1_tr, xi2_tr)
        rob_is.append(is_val)

        Q2_te, q10, q1r = m_rob.get_Q2(xi1_test_t)
        oos_val = mean_cost(q10, q1r, Q2_te, xi1_test, xi2_test)
        rob_oos.append(oos_val)
        n_ok += 1
        print(f"    IS={is_val:.4f}  OOS={oos_val:.4f}")

    sweep_res[str(rho)] = {"rob_is": rob_is, "rob_oos": rob_oos, "n_seeds": n_ok}
    print(f"rho={rho:.4f}: mean IS={np.mean(rob_is):.4f}  OOS={np.mean(rob_oos):.4f}  (n={n_ok})")

    with open(OUT_PATH, "w") as f:
        json.dump({"sweep": sweep_res, "benchmarks": benchmarks,
                   "meta": {"N_train": N_TRAIN, "N_test": N_TEST,
                             "seed_test": SEED_TEST, "seeds": SEEDS}}, f, indent=2)
    print(f"  [Saved] {OUT_PATH}")

print(f"\n[Done] {OUT_PATH}")
