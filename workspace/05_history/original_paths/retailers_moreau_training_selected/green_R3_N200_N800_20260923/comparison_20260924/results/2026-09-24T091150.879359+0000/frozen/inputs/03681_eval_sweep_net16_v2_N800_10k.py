"""
eval_sweep_net16_v2_N800_10k.py  --  Evaluate R=3, N=800 neural robust sweep on 10k test set.
==============================================================================================
Saves: sweep_eval_net16_v2_N800_10k.json
"""

import os, json, argparse
import time
from datetime import datetime, timezone
import numpy as np
from scipy.stats import truncnorm
import torch, torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="Recompute all sweep entries")
parser.add_argument("--force-rhos", nargs="*", type=float, default=[],
                    help="Recompute only the listed rho values")
args = parser.parse_args()
RUN_STARTED_UTC = datetime.now(timezone.utc).isoformat()
RUN_STARTED = time.perf_counter()
recomputed_rhos = []

NOMINAL_DIR  = "/data/home/aviyac/retailers_problem/R3/N800/saainit_net16_v2"
SWEEP_DIR    = "/data/home/aviyac/retailers_problem/R3/N800/rho_sweep_net16_v2"
SEEDS        = list(range(10))
N_TRAIN      = 800
N_TEST       = 10000
SEED_TEST    = 9999
R            = 3
c, h, b      = 0.25, 0.05, 0.5
KNN_K_GRID   = [k for k in (1, 3, 5, 7, 9, 11, 15, 20, 25, 35, 50, 75, 100, 150, 200, 300, 500, 800) if k < N_TRAIN]
if N_TRAIN - 1 not in KNN_K_GRID:
    KNN_K_GRID.append(N_TRAIN - 1)
KNN_SELECTION_VERSION = "per_seed_loo_grid_v3"

# ── data ──────────────────────────────────────────────────────────────────────
def gen(N, seed):
    np.random.seed(seed)
    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

# ── Gurobi projection ─────────────────────────────────────────────────────────
_grb_env = None
def _get_grb_env():
    global _grb_env
    if _grb_env is None:
        _grb_env = gp.Env(empty=True)
        _grb_env.setParam("OutputFlag", 0)
        _grb_env.start()
    return _grb_env

def _project_simplex_scaled(z, cap):
    """Project z onto {q >= 0, sum(q) <= cap} (Duchi et al. 2008 simplex projection).
    Works for a single row (R,) or a batch (N, R). Much faster than Gurobi for large N.
    """
    scalar = z.ndim == 1
    if scalar:
        z = z[None, :]            # (1, R)
    z_nn = np.maximum(z, 0.0)    # clip negatives first
    if np.all(z_nn.sum(axis=1) <= cap):
        return z_nn[0] if scalar else z_nn
    # Project onto scaled simplex {q >= 0, sum(q) = cap} using sort-based algorithm
    u = np.sort(z_nn, axis=1)[:, ::-1]   # descending
    cssv = np.cumsum(u, axis=1)
    rho = np.sum(u * np.arange(1, z.shape[1] + 1) > (cssv - cap), axis=1) - 1
    theta = (cssv[np.arange(len(z)), rho] - cap) / (rho + 1.0)
    q = np.maximum(z_nn - theta[:, None], 0.0)
    # Handle rows that don't need projection
    needs = z_nn.sum(axis=1) > cap
    q[~needs] = z_nn[~needs]
    return q[0] if scalar else q


def _gurobi_project(z_row, q10):
    return _project_simplex_scaled(np.asarray(z_row, dtype=float), q10)

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
                          torch.tensor(q1r).unsqueeze(0).expand(N, -1),
                          torch.full((N, 1), q10)], dim=1)
        z   = self.fc_layers(ni).numpy()
        Q2  = _project_simplex_scaled(z, q10)   # batch projection — O(N*R*log R)
        return Q2, q10, q1r

def load_model(path):
    m = Model()
    m.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"], strict=False)
    m.eval()
    return m

def scenario_costs(Q10, Q1r, Q2, xi1, xi2):
    s1 = c * (Q10 + Q1r.sum()) + h * Q10
    v  = np.maximum(b * (xi1 + xi2 - Q1r - Q2) - h * Q2,
          np.maximum(h * (Q1r - xi1 - xi2),
                     b * (xi1 - Q1r) - h * xi2))
    return s1 + v.sum(axis=1)

def mean_cost(Q10, Q1r, Q2, xi1, xi2):
    return float(scenario_costs(Q10, Q1r, Q2, xi1, xi2).mean())

def cvar_cost(Q10, Q1r, Q2, xi1, xi2, alpha=0.05):
    costs = scenario_costs(Q10, Q1r, Q2, xi1, xi2)
    cutoff = int(np.floor((1 - alpha) * len(costs)))
    return float(np.sort(costs)[cutoff:].mean())

# ── Exact IS: MP objective ────────────────────────────────────────────────────
def exact_mp_is(state_dict, ap_json_paths, xi1_np, xi2_np):
    m = Model()
    m.load_state_dict(state_dict, strict=False)
    m.eval()
    q10 = torch.nn.functional.softplus(m.Q10).item()
    q1r = torch.nn.functional.softplus(m.Q1r).detach().numpy()

    adv_xi1_by_ball, adv_xi2_by_ball = {}, {}
    for path in ap_json_paths:
        with open(path) as f:
            for s in json.load(f):
                bid = s["ball_id"]
                adv_xi1_by_ball.setdefault(bid, []).append(np.array(s["worst_xi1"]))
                adv_xi2_by_ball.setdefault(bid, []).append(np.array(s["worst_xi2"]))
    max_cols = max(len(v) for v in adv_xi1_by_ball.values())
    adv_xi1_cols = [xi1_np.copy() for _ in range(max_cols)]
    adv_xi2_cols = [xi2_np.copy() for _ in range(max_cols)]
    for bid in adv_xi1_by_ball:
        for k, (ax1, ax2) in enumerate(zip(adv_xi1_by_ball[bid], adv_xi2_by_ball[bid])):
            adv_xi1_cols[k][bid] = ax1
            adv_xi2_cols[k][bid] = ax2

    all_xi1 = [xi1_np] + adv_xi1_cols
    all_xi2 = [xi2_np] + adv_xi2_cols

    col_costs = []
    for cx1, cx2 in zip(all_xi1, all_xi2):
        cx1_t = torch.tensor(cx1, dtype=torch.float32)
        Q2_col, _, _ = m.get_Q2(cx1_t)
        col_costs.append(scenario_costs(q10, q1r, Q2_col, cx1, cx2))
    per_sample_max = np.stack(col_costs, axis=1).max(axis=1)
    return float(per_sample_max.mean())

# ── SAA helpers ───────────────────────────────────────────────────────────────
def saa_lp_Q2(Q10, Q1r, xi1, xi2):
    N = xi1.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m   = gp.Model(env=env)
    Q2v = m.addVars(N, R, lb=0.0)
    vv  = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i, r] for r in range(R)) <= Q10)
        for r in range(R):
            m.addConstr(vv[i, r] >= b * (xi1[i,r]+xi2[i,r]-Q2v[i,r]-Q1r[r]) - h*Q2v[i,r])
            m.addConstr(vv[i, r] >= h * (Q1r[r] - xi1[i,r] - xi2[i,r]))
            m.addConstr(vv[i, r] >= b * (xi1[i,r] - Q1r[r]) - h * xi2[i,r])
    m.setObjective(gp.quicksum(vv[i,r] for i in range(N) for r in range(R)), GRB.MINIMIZE)
    m.optimize()
    return np.array([[Q2v[i, r].X for r in range(R)] for i in range(N)])

def saa_lp_full(xi1, xi2):
    N = xi1.shape[0]
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model(env=env)
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
    return Q10_opt, Q1r_opt, Q2_opt

def knn_Q2(xi1_te, xi1_tr, Q2_tr, Q10, k=5):
    k = int(k)
    if k < 1 or k > xi1_tr.shape[0]:
        raise ValueError(f"k must be in [1, {xi1_tr.shape[0]}], got {k}")
    diff  = xi1_te[:, None, :] - xi1_tr[None, :, :]
    dists = np.sqrt((diff**2).sum(2))
    idx   = np.argpartition(dists, k - 1, axis=1)[:, :k]
    d     = np.take_along_axis(dists, idx, axis=1)
    w     = 1.0 / np.maximum(d, 1e-9)
    w    /= w.sum(1, keepdims=True)
    Q2    = (w[:, :, None] * Q2_tr[idx]).sum(1)
    Q2    = np.maximum(Q2, 0.0)
    s     = Q2.sum(1, keepdims=True)
    scale = np.ones_like(s)
    np.divide(Q10, s, out=scale, where=s > Q10)
    return Q2 * scale

def knn_Q2_loo(xi1_tr, Q2_tr, Q10, k):
    k = int(k)
    n_train = xi1_tr.shape[0]
    if k < 1 or k >= n_train:
        raise ValueError(f"k must be in [1, {n_train - 1}] for leave-one-out, got {k}")
    diff  = xi1_tr[:, None, :] - xi1_tr[None, :, :]
    dists = np.sqrt((diff**2).sum(2))
    np.fill_diagonal(dists, np.inf)
    idx   = np.argpartition(dists, k - 1, axis=1)[:, :k]
    d     = np.take_along_axis(dists, idx, axis=1)
    w     = 1.0 / np.maximum(d, 1e-9)
    w    /= w.sum(1, keepdims=True)
    Q2    = (w[:, :, None] * Q2_tr[idx]).sum(1)
    Q2    = np.maximum(Q2, 0.0)
    s     = Q2.sum(1, keepdims=True)
    scale = np.ones_like(s)
    np.divide(Q10, s, out=scale, where=s > Q10)
    return Q2 * scale

def summarize_knn_grid(knn_by_k):
    for vals in knn_by_k.values():
        vals["mean_is"] = float(np.mean(vals["is"]))
        vals["mean_val"] = float(np.mean(vals["val"]))
        vals["mean_oos"] = float(np.mean(vals["oos"]))

# ── generate test data ────────────────────────────────────────────────────────
print(f"Generating test data: N={N_TEST}, seed={SEED_TEST}")
xi1_test, xi2_test = gen(N_TEST, SEED_TEST)
xi1_test_t = torch.tensor(xi1_test, dtype=torch.float32)

# ── load existing cache ───────────────────────────────────────────────────────
out_path = os.path.join(SWEEP_DIR, "sweep_eval_net16_v2_N800_10k.json")
_cache = {}
if os.path.exists(out_path):
    with open(out_path) as f:
        _cache = json.load(f)
    print(f"[Cache] Loaded existing results from {out_path}")

# ── benchmarks ────────────────────────────────────────────────────────────────
cached_benchmarks = _cache.get("benchmarks", {})
benchmarks_current = (
    cached_benchmarks.get("_full_lp_10k")
    and cached_benchmarks.get("_knn_k_selection") == KNN_SELECTION_VERSION
    and "selected_k_by_seed" in cached_benchmarks.get("saa_knn", {})
)

if benchmarks_current:
    benchmarks = cached_benchmarks
    print("[Cache] Reusing benchmarks.")
else:
    reuse_saa_lb = cached_benchmarks.get("_full_lp_10k") and "saa_lb" in cached_benchmarks
    if reuse_saa_lb:
        print("\nSelecting kNN-SAA k; reusing cached SAA-LP lower bound ...")
        saa_lb_is = cached_benchmarks["saa_lb"]["is"]
        saa_lb_oos = cached_benchmarks["saa_lb"]["oos"]
        lb_oos_val = float(np.mean(saa_lb_oos))
        print(f"  SAA-LP true LB OOS={lb_oos_val:.4f}")
    else:
        print("\nComputing SAA-LP true lower bound on 10k test set ...")
        Q10_lb_te, Q1r_lb_te, Q2_lb_te_opt = saa_lp_full(xi1_test, xi2_test)
        lb_oos_val = mean_cost(Q10_lb_te, Q1r_lb_te, Q2_lb_te_opt, xi1_test, xi2_test)
        saa_lb_is = []
        print(f"  SAA-LP true LB OOS={lb_oos_val:.4f}")

    print("\nSelecting kNN k by leave-one-out training validation; evaluating on 10k test set ...")
    knn_by_k = {str(k): {"is": [], "val": [], "oos": []} for k in KNN_K_GRID}
    selected_k_by_seed, selected_is, selected_val, selected_oos = [], [], [], []
    for seed in SEEDS:
        nom_summ_path = os.path.join(NOMINAL_DIR, f"seed_{seed}_summary.json")
        if not os.path.exists(nom_summ_path):
            print(f"  [MISSING NOMINAL SUMMARY] seed={seed} — skipping benchmark")
            continue
        with open(nom_summ_path) as f:
            summ = json.load(f)
        Q10_s = summ["saa_Q10"]
        Q1r_s = np.array(summ["saa_Q1r"])
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)
        Q2_tr_opt = saa_lp_Q2(Q10_s, Q1r_s, xi1_tr, xi2_tr)

        if not reuse_saa_lb:
            Q10_lb_tr, Q1r_lb_tr, Q2_lb_tr = saa_lp_full(xi1_tr, xi2_tr)
            saa_lb_is.append(mean_cost(Q10_lb_tr, Q1r_lb_tr, Q2_lb_tr, xi1_tr, xi2_tr))

        for k in KNN_K_GRID:
            Q2_knn_tr = knn_Q2(xi1_tr, xi1_tr, Q2_tr_opt, Q10_s, k=k)
            knn_by_k[str(k)]["is"].append(mean_cost(Q10_s, Q1r_s, Q2_knn_tr, xi1_tr, xi2_tr))

            Q2_knn_val = knn_Q2_loo(xi1_tr, Q2_tr_opt, Q10_s, k=k)
            knn_by_k[str(k)]["val"].append(mean_cost(Q10_s, Q1r_s, Q2_knn_val, xi1_tr, xi2_tr))

            Q2_knn_te = knn_Q2(xi1_test, xi1_tr, Q2_tr_opt, Q10_s, k=k)
            knn_by_k[str(k)]["oos"].append(mean_cost(Q10_s, Q1r_s, Q2_knn_te, xi1_test, xi2_test))

        seed_best_k = min(KNN_K_GRID, key=lambda k: (knn_by_k[str(k)]["val"][-1], k))
        selected_k_by_seed.append(seed_best_k)
        selected_is.append(knn_by_k[str(seed_best_k)]["is"][-1])
        selected_val.append(knn_by_k[str(seed_best_k)]["val"][-1])
        selected_oos.append(knn_by_k[str(seed_best_k)]["oos"][-1])
        print(f"  seed {seed}: best k={seed_best_k}  LOO={knn_by_k[str(seed_best_k)]['val'][-1]:.4f}  10k OOS={knn_by_k[str(seed_best_k)]['oos'][-1]:.4f}")

    summarize_knn_grid(knn_by_k)
    if not reuse_saa_lb:
        saa_lb_oos = [lb_oos_val] * len(saa_lb_is)
    print("  kNN grid mean LOO: " + ", ".join(
        f"k={k}: {knn_by_k[str(k)]['mean_val']:.4f}" for k in KNN_K_GRID))
    print("  kNN grid mean 10k OOS: " + ", ".join(
        f"k={k}: {knn_by_k[str(k)]['mean_oos']:.4f}" for k in KNN_K_GRID))
    selected_k_counts = {str(k): selected_k_by_seed.count(k) for k in sorted(set(selected_k_by_seed))}
    selected_k = selected_k_by_seed[0] if len(selected_k_counts) == 1 else None
    print(f"  Selected k by seed: {selected_k_by_seed}")
    print(f"  Mean selected 10k OOS: {np.mean(selected_oos):.4f}")
    benchmarks = {
        "_full_lp_10k": True,
        "_knn_k_selection": KNN_SELECTION_VERSION,
        "knn_k_grid": KNN_K_GRID,
        "saa_lb":  {"is": saa_lb_is, "oos": saa_lb_oos},
        "saa_knn": {
            "is": selected_is,
            "val": selected_val,
            "oos": selected_oos,
            "selected_k": selected_k,
            "selected_k_by_seed": selected_k_by_seed,
            "selected_k_counts": selected_k_counts,
            "selection_metric": "per_seed_loo_validation_cost",
        },
        "saa_knn_by_k": knn_by_k,
    }

# ── discover rho directories ──────────────────────────────────────────────────
rho_sources = {}
for entry in sorted(os.listdir(SWEEP_DIR)):
    full = os.path.join(SWEEP_DIR, entry)
    if entry.startswith("rho_") and os.path.isdir(full):
        tag = entry[4:]
        rho = float(tag.replace("p", "."))
        rho_sources[rho] = ("sweep", full)
rho_sources[0.0] = ("nominal",)
all_rhos = sorted(rho_sources.keys())
print(f"\nFound {len(all_rhos)} RHO values: {all_rhos}")

# ── evaluate each rho ─────────────────────────────────────────────────────────
def _n_available(rho, src):
    if src[0] == "nominal":
        return len(SEEDS)
    rho_dir = src[1]
    return sum(1 for s in SEEDS
               if os.path.exists(os.path.join(rho_dir, f"seed_{s}_summary.json")))

sweep_res = dict(_cache.get("sweep", {}))

for rho in all_rhos:
    src = rho_sources[rho]
    n_avail = _n_available(rho, src)
    cached  = sweep_res.get(str(rho), {})
    force_this_rho = args.force or any(abs(rho - r) <= 1e-12 for r in args.force_rhos)
    if not force_this_rho and cached.get("n_seeds", 0) >= n_avail and n_avail > 0:
        print(f"[Cache] rho={rho:.3f}: {cached['n_seeds']} seeds already evaluated — skipping")
        continue
    rho_started = time.perf_counter()
    rob_is, rob_oos, rob_cvar = [], [], []
    nom_is, nom_oos = [], []
    n_ok = 0

    for seed in SEEDS:
        xi1_tr, xi2_tr = gen(N_TRAIN, seed)
        xi1_tr_t = torch.tensor(xi1_tr, dtype=torch.float32)

        nom_path = os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth")
        if not os.path.exists(nom_path):
            print(f"  [MISSING NOMINAL] seed={seed} — skipping")
            continue
        m_nom = load_model(nom_path)
        Q2_n_tr, q10n, q1rn = m_nom.get_Q2(xi1_tr_t)
        nom_is.append(mean_cost(q10n, q1rn, Q2_n_tr, xi1_tr, xi2_tr))
        Q2_n_te, _, _ = m_nom.get_Q2(xi1_test_t)
        nom_oos.append(mean_cost(q10n, q1rn, Q2_n_te, xi1_test, xi2_test))

        if src[0] == "nominal":
            rob_is.append(nom_is[-1])
            rob_oos.append(nom_oos[-1])
            Q2_n_te2, _, _ = m_nom.get_Q2(xi1_test_t)
            rob_cvar.append(cvar_cost(q10n, q1rn, Q2_n_te2, xi1_test, xi2_test))
            n_ok += 1
            continue

        rho_dir   = src[1]
        summ_file = os.path.join(rho_dir, f"seed_{seed}_summary.json")
        if not os.path.exists(summ_file):
            print(f"  [MISSING] rho={rho:.3f} seed={seed} — skipping")
            continue
        with open(summ_file) as f:
            summ = json.load(f)
        fmp = summ.get("final_mp_iter", 0)
        ccg_iters = summ.get("ccg_iters", fmp)
        if fmp == 0:
            rob_is.append(nom_is[-1])
            rob_oos.append(nom_oos[-1])
            Q2_n_te3, _, _ = m_nom.get_Q2(xi1_test_t)
            rob_cvar.append(cvar_cost(q10n, q1rn, Q2_n_te3, xi1_test, xi2_test))
            n_ok += 1
            continue

        rob_path = os.path.join(rho_dir, f"seed_{seed}_mp_iter{fmp}.pth")
        if not os.path.exists(rob_path):
            print(f"  [MISSING MODEL] rho={rho:.3f} seed={seed} — skipping")
            continue

        m_rob = load_model(rob_path)
        ap_paths = [os.path.join(rho_dir, f"seed_{seed}_ap_iter{k}.json")
                    for k in range(1, ccg_iters + 1)
                    if os.path.exists(os.path.join(rho_dir, f"seed_{seed}_ap_iter{k}.json"))]
        print(f"  exact IS: rho={rho:.3f} seed={seed} ({len(ap_paths)} AP cols) ...", flush=True)
        rob_is_val = exact_mp_is(m_rob.state_dict(), ap_paths, xi1_tr, xi2_tr)
        rob_is.append(rob_is_val)
        Q2_r_te, q10r, q1rr = m_rob.get_Q2(xi1_test_t)
        rob_oos.append(mean_cost(q10r, q1rr, Q2_r_te, xi1_test, xi2_test))
        rob_cvar.append(cvar_cost(q10r, q1rr, Q2_r_te, xi1_test, xi2_test))
        n_ok += 1

    sweep_res[str(rho)] = {
        "rob_is":   rob_is,
        "rob_oos":  rob_oos,
        "rob_cvar": rob_cvar,
        "nom_is":   nom_is,
        "nom_oos":  nom_oos,
        "n_seeds":  n_ok,
        "evaluation_time_s": time.perf_counter() - rho_started,
    }
    recomputed_rhos.append(rho)
    print(f"rho={rho:.3f}: rob IS={np.mean(rob_is):.4f}  OOS={np.mean(rob_oos):.4f}  "
          f"(n_seeds={n_ok})")
    out = {
        "rhos":       sorted(float(k) for k in sweep_res),
        "sweep":      sweep_res,
        "benchmarks": benchmarks,
        "meta": {"N_train": N_TRAIN, "N_test": N_TEST, "seed_test": SEED_TEST,
                 "seeds": SEEDS},
        "runtime": {"start_utc": RUN_STARTED_UTC,
                "current_invocation_elapsed_s": time.perf_counter() - RUN_STARTED,
                "status": "running", "force_all": args.force,
                "force_rhos": args.force_rhos,
                "recomputed_rhos": recomputed_rhos},
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  [Saved] {out_path}", flush=True)

out = {
    "rhos":       sorted(float(k) for k in sweep_res),
    "sweep":      sweep_res,
    "benchmarks": benchmarks,
    "meta": {"N_train": N_TRAIN, "N_test": N_TEST, "seed_test": SEED_TEST,
             "seeds": SEEDS},
    "runtime": {"start_utc": RUN_STARTED_UTC,
                "end_utc": datetime.now(timezone.utc).isoformat(),
                "current_invocation_elapsed_s": time.perf_counter() - RUN_STARTED,
                "status": "completed", "force_all": args.force,
                "force_rhos": args.force_rhos,
                "recomputed_rhos": recomputed_rhos},
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[Done] Results saved to {out_path}")
