"""
run_all_saainit_N800.py — Full CCG pipeline, N=800, SAA-init, 10 seeds, PARALLEL
==================================================================================
Seeds are independent after the LR search — each seed runs in its own process.

Per-seed files:
  seed_{k}_nominal.pth       — nominal checkpoint
  seed_{k}_ap_iter{j}.json   — adversarial scenarios for CCG iter j
  seed_{k}_mp_iter{j}.pth    — robust checkpoint after MP iter j
  seed_{k}_summary.json      — convergence results
  seed_{k}_log.txt           — full per-seed log
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
import multiprocessing as mp

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

# ============================================================
# CONFIGURATION
# ============================================================
class Cfg:
    R             = 3
    c             = 0.25
    h             = 0.05
    b             = 0.5
    N_SAMPLES     = 800

    LR_Q          = 0.01
    EPOCHS        = 3000
    EARLY_STOP    = 300
    SCHED_PATIENCE = 150
    SCHED_FACTOR  = 0.7

    SCS_EPS       = 1e-4
    SCS_MAX_ITERS = 200000

    RHO_1         = 0.3
    RHO_2         = 0.3
    CCG_MAX_ITER  = 8
    CCG_CONV_THRESH_PCT = 1.0

    SEEDS         = list(range(10))
    NET_SEED      = 0
    GRID_LR_NET   = [0.005, 0.01, 0.05]
    LR_SEARCH_SEED = 0
    BASE_DIR      = "/data/home/aviyac/retailers_problem/R3/N800/saainit"
    N_WORKERS     = 10      # server is free; 10 seeds in parallel

# ============================================================
# HELPERS
# ============================================================
def softplus_np(x):
    x = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def softplus_inverse(y):
    y = np.atleast_1d(np.array(y, dtype=np.float64))
    return np.where(y > 20, y, np.log(np.expm1(y)))

def generate_data(seed):
    cfg = Cfg()
    np.random.seed(seed)
    xi1 = np.maximum(np.random.normal(6.0, 2.5, (cfg.N_SAMPLES, cfg.R)), 0.0)
    xi2 = np.maximum(np.random.normal(2 * (6.0 - xi1)**2, 2.5), 0.0)
    return xi1, xi2

def solve_saa(xi1_data, xi2_data):
    cfg = Cfg()
    N, R = xi1_data.shape
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m = gp.Model(env=env)
    Q10_v = m.addVar(lb=0.0)
    Q1r_v = m.addVars(R, lb=0.0)
    Q2_v  = m.addVars(N, R, lb=0.0)
    vv    = m.addVars(N, R, lb=-GRB.INFINITY)
    for i in range(N):
        m.addConstr(gp.quicksum(Q2_v[i, r] for r in range(R)) <= Q10_v)
        for r in range(R):
            m.addConstr(vv[i,r] >= cfg.b*(xi1_data[i,r]+xi2_data[i,r]-Q2_v[i,r]-Q1r_v[r]) - cfg.h*Q2_v[i,r])
            m.addConstr(vv[i,r] >= cfg.h*(Q1r_v[r]-xi1_data[i,r]-xi2_data[i,r]))
            m.addConstr(vv[i,r] >= cfg.b*(xi1_data[i,r]-Q1r_v[r]) - cfg.h*xi2_data[i,r])
    obj = (cfg.c*(Q10_v + gp.quicksum(Q1r_v[r] for r in range(R))) + cfg.h*Q10_v
           + (1.0/N)*gp.quicksum(vv[i,r] for i in range(N) for r in range(R)))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    return Q10_v.X, np.array([Q1r_v[r].X for r in range(R)]), m.ObjVal

def to_serializable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, float)): return float(obj)
    return obj

def ckpt(seed, tag):
    return os.path.join(Cfg.BASE_DIR, f"seed_{seed}_{tag}.pth")

def ap_json(seed, k):
    return os.path.join(Cfg.BASE_DIR, f"seed_{seed}_ap_iter{k}.json")

def summary_json(seed):
    return os.path.join(Cfg.BASE_DIR, f"seed_{seed}_summary.json")

def seed_log(seed):
    return os.path.join(Cfg.BASE_DIR, f"seed_{seed}_log.txt")

def grid_ckpt(lr_net):
    lr_tag = f"{lr_net:.4f}".replace('.', 'p')
    return os.path.join(Cfg.BASE_DIR, f"lrsearch_lr{lr_tag}.pth")

# ============================================================
# MODEL
# ============================================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Cfg()
        self.fc_layers = nn.Sequential(
            nn.Linear(cfg.R * 2 + 1, 32), nn.ReLU(),
            nn.Linear(32, 32),             nn.ReLU(),
            nn.Linear(32, cfg.R)
        )
        self.Q10 = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.Q1r = nn.Parameter(torch.zeros(cfg.R, dtype=torch.float32))

        x       = cp.Variable(cfg.R)
        z_param = cp.Parameter(cfg.R)
        Q10_p   = cp.Parameter(1)
        cons    = [cp.sum(x) <= Q10_p + 1e-6, x >= 0]
        self.proj_layer = CvxpyLayer(
            cp.Problem(cp.Minimize(cp.sum_squares(x - z_param)), cons),
            parameters=[z_param, Q10_p], variables=[x]
        )

    def per_sample_cost(self, xi1, xi2):
        cfg = Cfg()
        N   = xi1.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        net_in = torch.cat([xi1, q1r.unsqueeze(0).expand(N, -1), q10.expand(N, 1)], dim=1)
        z = self.fc_layers(net_in)
        q2, = self.proj_layer(z, q10.expand(N, 1),
                              solver_args={"solve_method": "SCS",
                                           "eps": cfg.SCS_EPS,
                                           "max_iters": cfg.SCS_MAX_ITERS})
        stage1 = cfg.c * (q10 + torch.sum(q1r)) + cfg.h * q10
        t1 = cfg.b * (xi2 + xi1 - q2 - q1r) - cfg.h * q2
        t2 = cfg.h * (q1r - xi1 - xi2)
        t3 = cfg.b * (xi1 - q1r) - cfg.h * xi2
        return stage1 + torch.sum(torch.maximum(torch.maximum(t1, t2), t3), dim=1)

def make_model_saa_init(q10_saa, q1r_saa, net_seed=None):
    cfg = Cfg()
    torch.manual_seed(net_seed if net_seed is not None else cfg.NET_SEED)
    m = Model()
    with torch.no_grad():
        m.Q10.data = torch.tensor(softplus_inverse(q10_saa),  dtype=torch.float32)
        m.Q1r.data = torch.tensor(softplus_inverse(q1r_saa),  dtype=torch.float32)
    return m

# ============================================================
# TRAINING
# ============================================================
def train(model, loss_fn, label, lr_net, log_fn=print, epochs=None):
    cfg = Cfg()
    epochs = epochs or cfg.EPOCHS
    opt = torch.optim.Adam([
        {'params': model.fc_layers.parameters(), 'lr': lr_net},
        {'params': [model.Q10, model.Q1r],       'lr': cfg.LR_Q},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=cfg.SCHED_FACTOR, patience=cfg.SCHED_PATIENCE, threshold=1e-3
    )
    best_loss, best_state, no_imp = float('inf'), None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step(loss.item())
        v = loss.item()
        if v < best_loss - 1e-4:
            best_loss = v
            best_state = {k: w.clone() for k, w in model.state_dict().items() if not w.is_sparse}
            no_imp = 0
        else:
            no_imp += 1
        if ep % 50 == 0 or ep == 1:
            q10v = torch.nn.functional.softplus(model.Q10).item()
            q1rv = torch.nn.functional.softplus(model.Q1r).detach().numpy()
            lr_q = opt.param_groups[1]['lr']
            log_fn(f"  [{label}] ep {ep:4d}/{epochs} | loss={v:.4f} | "
                   f"Q10={q10v:.3f} | Q1r={np.round(q1rv,3)} | "
                   f"lr_Q={lr_q:.5f} | no_imp={no_imp}/{cfg.EARLY_STOP} | "
                   f"t={time.time()-t0:.0f}s")
        if no_imp >= cfg.EARLY_STOP:
            log_fn(f"  [{label}] Early stop ep={ep}, best={best_loss:.4f}")
            break
    model.load_state_dict(best_state, strict=False)
    return best_loss, best_state

# ============================================================
# ADVERSARIAL PROBLEM (AP)
# ============================================================
def load_weights_from_state(state_dict):
    cfg = Cfg()
    Q10_val = softplus_np(state_dict['Q10'].numpy())[0]
    Q1r_val = softplus_np(state_dict['Q1r'].numpy())
    W1_full = state_dict['fc_layers.0.weight'].numpy()
    b1_raw  = state_dict['fc_layers.0.bias'].numpy()
    return {
        'W1':  W1_full[:, :cfg.R],
        'b1':  b1_raw + W1_full[:, cfg.R:2*cfg.R] @ Q1r_val + W1_full[:, 2*cfg.R:] @ [Q10_val],
        'W2':  state_dict['fc_layers.2.weight'].numpy(),
        'b2':  state_dict['fc_layers.2.bias'].numpy(),
        'W3':  state_dict['fc_layers.4.weight'].numpy(),
        'b3':  state_dict['fc_layers.4.bias'].numpy(),
        'Q10': Q10_val,
        'Q1r': Q1r_val,
    }

def solve_ap_one(weights, xi1_hat, xi2_hat, idx):
    cfg = Cfg()
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']
    W3, b3 = weights['W3'], weights['b3']
    Q10_val, Q1r_val = weights['Q10'], weights['Q1r']

    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m   = gp.Model(f"AP_{idx}", env=env)

    xi1 = m.addVars(cfg.R, lb=np.maximum(0, xi1_hat - cfg.RHO_1), ub=xi1_hat + cfg.RHO_1)
    xi2 = m.addVars(cfg.R, lb=np.maximum(0, xi2_hat - cfg.RHO_2), ub=xi2_hat + cfg.RHO_2)

    def relu_layer(inp, W, b, tag):
        n = W.shape[0]
        xp = m.addVars(n, lb=0, name=f"xp{tag}")
        xm = m.addVars(n, lb=0, name=f"xm{tag}")
        z  = m.addVars(n, vtype=GRB.BINARY, name=f"z{tag}")
        for j in range(n):
            expr = gp.quicksum(W[j,k] * inp[k] for k in range(W.shape[1])) + b[j]
            m.addConstr(expr == xp[j] - xm[j])
            m.addGenConstrIndicator(z[j], 1, xp[j] == 0)
            m.addGenConstrIndicator(z[j], 0, xm[j] == 0)
        return xp

    l1 = relu_layer(xi1, W1, b1, "1")
    l2 = relu_layer(l1,  W2, b2, "2")

    R = cfg.R
    zs = m.addVars(R, lb=-GRB.INFINITY)
    for j in range(R):
        m.addConstr(zs[j] == gp.quicksum(W3[j,k]*l2[k] for k in range(W3.shape[1])) + b3[j])

    q2  = m.addVars(R, lb=0)
    lam = m.addVar(lb=0)
    mu  = m.addVars(R, lb=0)
    for r in range(R):
        m.addConstr(2.0*q2[r] - 2*zs[r] + lam - mu[r] == 0)
    m.addConstr(gp.quicksum(q2[r] for r in range(R)) <= Q10_val)
    u_lam = m.addVar(vtype=GRB.BINARY)
    m.addGenConstrIndicator(u_lam, 1, lam == 0)
    m.addGenConstrIndicator(u_lam, 0, Q10_val - gp.quicksum(q2[r] for r in range(R)) == 0)
    u_mu = m.addVars(R, vtype=GRB.BINARY)
    for r in range(R):
        m.addGenConstrIndicator(u_mu[r], 1, mu[r] == 0)
        m.addGenConstrIndicator(u_mu[r], 0, q2[r] == 0)

    v = m.addVars(R, lb=-GRB.INFINITY)
    for r in range(R):
        g1 = m.addVar(lb=-GRB.INFINITY)
        g2 = m.addVar(lb=-GRB.INFINITY)
        g3 = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(g1 == cfg.b*(xi1[r]+xi2[r]-Q1r_val[r]-q2[r]) - cfg.h*q2[r])
        m.addConstr(g2 == cfg.h*(Q1r_val[r]-xi1[r]-xi2[r]))
        m.addConstr(g3 == cfg.b*(xi1[r]-Q1r_val[r]) - cfg.h*xi2[r])
        m.addGenConstrMax(v[r], [g1, g2, g3])

    stage1 = cfg.c*(Q10_val + np.sum(Q1r_val)) + cfg.h*Q10_val
    m.setObjective(stage1 + gp.quicksum(v[r] for r in range(R)), GRB.MAXIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        return (np.array([xi1[r].X for r in range(R)]),
                np.array([xi2[r].X for r in range(R)]),
                m.ObjVal)
    return None, None, None

def run_ap(state_dict, seed, ccg_iter, xi1_np, xi2_np, log_fn):
    cfg = Cfg()
    weights = load_weights_from_state(state_dict)
    log_fn(f"  AP  Q10={weights['Q10']:.4f}, Q1r={np.round(weights['Q1r'],3)}")
    scenarios = []
    t0 = time.time()
    for i in range(cfg.N_SAMPLES):
        wx1, wx2, mc = solve_ap_one(weights, xi1_np[i], xi2_np[i], i)
        if wx1 is not None:
            scenarios.append({"ball_id": i, "xi1_hat": xi1_np[i], "xi2_hat": xi2_np[i],
                               "worst_xi1": wx1, "worst_xi2": wx2, "max_cost": mc})
    costs = [s["max_cost"] for s in scenarios]
    log_fn(f"  AP done: n={len(scenarios)}, mean={np.mean(costs):.4f}, "
           f"max={np.max(costs):.4f}, t={time.time()-t0:.0f}s")
    path = ap_json(seed, ccg_iter)
    with open(path, "w") as f:
        json.dump([{k: to_serializable(v) for k,v in s.items()} for s in scenarios], f)
    return np.mean(costs), path

# ============================================================
# MASTER PROBLEM (MP)
# ============================================================
def run_mp(start_state, scenario_paths, label, xi1_np, xi2_np, xi1_t, xi2_t, lr_net, log_fn):
    adv_xi1_by_ball, adv_xi2_by_ball = {}, {}
    for path in scenario_paths:
        with open(path) as f:
            for s in json.load(f):
                bid = s["ball_id"]
                adv_xi1_by_ball.setdefault(bid, []).append(s["worst_xi1"])
                adv_xi2_by_ball.setdefault(bid, []).append(s["worst_xi2"])

    max_cols = max(len(v) for v in adv_xi1_by_ball.values())
    adv_xi1_cols = [xi1_np.copy() for _ in range(max_cols)]
    adv_xi2_cols = [xi2_np.copy() for _ in range(max_cols)]
    for bid in adv_xi1_by_ball:
        for k, (ax1, ax2) in enumerate(zip(adv_xi1_by_ball[bid], adv_xi2_by_ball[bid])):
            adv_xi1_cols[k][bid] = ax1
            adv_xi2_cols[k][bid] = ax2
    adv_t1 = [torch.tensor(c, dtype=torch.float32) for c in adv_xi1_cols]
    adv_t2 = [torch.tensor(c, dtype=torch.float32) for c in adv_xi2_cols]

    model = Model()
    model.load_state_dict(start_state, strict=False)

    def loss_fn(m):
        cols = [m.per_sample_cost(xi1_t, xi2_t)]
        for t1, t2 in zip(adv_t1, adv_t2):
            cols.append(m.per_sample_cost(t1, t2))
        return torch.stack(cols, dim=1).max(dim=1).values.mean()

    best_loss, best_state = train(model, loss_fn, label, lr_net, log_fn=log_fn)
    return best_loss, best_state

# ============================================================
# PER-SEED PIPELINE (runs in a worker process)
# ============================================================
def run_seed(seed, best_lr_net):
    cfg = Cfg()
    log_path = seed_log(seed)
    log_f = open(log_path, "w", buffering=1)

    def log_fn(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        log_f.write(line + "\n")
        print(f"[seed={seed}] {line}", flush=True)

    t_seed = time.time()
    log_fn(f"{'='*60}")
    log_fn(f"SEED {seed}  (N={cfg.N_SAMPLES}, RHO={cfg.RHO_1}, lr_net={best_lr_net})")
    log_fn(f"{'='*60}")

    xi1_np, xi2_np = generate_data(seed)
    xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
    xi2_t = torch.tensor(xi2_np, dtype=torch.float32)
    log_fn(f"Data: xi1 mean={xi1_np.mean():.3f}, xi2 mean={xi2_np.mean():.3f}")

    saa_Q10, saa_Q1r, saa_obj = solve_saa(xi1_np, xi2_np)
    log_fn(f"SAA target: {saa_obj:.4f}  Q10={saa_Q10:.4f}  Q1r={np.round(saa_Q1r,4)}")

    # Check if already done
    summ_path = summary_json(seed)
    if os.path.exists(summ_path):
        with open(summ_path) as f:
            stored = json.load(f)
        if stored.get("converged", False):
            log_fn(f"[SKIP] Already converged at iter {stored['ccg_iters']}")
            log_f.close()
            return stored

    # ---- NOMINAL ----
    nom_ckpt = ckpt(seed, "nominal")
    if os.path.exists(nom_ckpt):
        saved     = torch.load(nom_ckpt, map_location="cpu")
        nom_state = saved["model_state_dict"]
        nom_loss  = saved["best_loss"]
        gap_pct   = (nom_loss - saa_obj) / abs(saa_obj) * 100
        log_fn(f"[SKIP nominal] loss={nom_loss:.4f}  gap={gap_pct:+.2f}%")
    else:
        log_fn(f"--- NOMINAL TRAINING ---")
        model = make_model_saa_init(saa_Q10, saa_Q1r)
        nom_loss, nom_state = train(model,
                                    lambda m: m.per_sample_cost(xi1_t, xi2_t).mean(),
                                    f"nom_s{seed}", best_lr_net, log_fn=log_fn)
        torch.save({"model_state_dict": nom_state, "best_loss": float(nom_loss),
                    "lr_net": best_lr_net, "seed": seed, "lr_Q": cfg.LR_Q,
                    "n_samples": cfg.N_SAMPLES, "saa_obj": float(saa_obj),
                    "saa_Q10": float(saa_Q10), "saa_Q1r": saa_Q1r.tolist()},
                   nom_ckpt)
        gap_pct = (nom_loss - saa_obj) / abs(saa_obj) * 100
        log_fn(f"Nominal done: loss={nom_loss:.4f}  SAA={saa_obj:.4f}  gap={gap_pct:+.2f}%")

    # ---- CCG ----
    cur_state      = nom_state
    scenario_paths = []
    ccg_results    = []
    mp_loss        = nom_loss
    converged      = False

    for it in range(1, cfg.CCG_MAX_ITER + 1):
        log_fn(f"=== CCG iter {it} ===")

        ap_path = ap_json(seed, it)
        if os.path.exists(ap_path):
            with open(ap_path) as f: sc = json.load(f)
            ap_mean = float(np.mean([s["max_cost"] for s in sc]))
            log_fn(f"[SKIP AP iter {it}] ap_mean={ap_mean:.4f}")
        else:
            ap_mean, ap_path = run_ap(cur_state, seed, it, xi1_np, xi2_np, log_fn)

        scenario_paths.append(ap_json(seed, it))
        delta     = ap_mean - mp_loss
        delta_pct = delta / mp_loss * 100
        log_fn(f"Δ = AP({ap_mean:.4f}) - MP({mp_loss:.4f}) = {delta:.4f}  ({delta_pct:.2f}%)")

        if it > 1 and delta_pct < cfg.CCG_CONV_THRESH_PCT:
            log_fn(f"*** CONVERGED at iter {it}: Δ={delta_pct:.2f}% < {cfg.CCG_CONV_THRESH_PCT}% ***")
            converged = True
            ccg_results.append({"iter": it, "ap_mean": ap_mean, "mp_loss": mp_loss,
                                 "delta": delta, "delta_pct": delta_pct, "converged": True})
            break

        mp_ckpt = ckpt(seed, f"mp_iter{it}")
        if os.path.exists(mp_ckpt):
            saved     = torch.load(mp_ckpt, map_location="cpu")
            mp_loss   = saved["best_loss"]
            cur_state = saved["model_state_dict"]
            log_fn(f"[SKIP MP iter {it}] mp_loss={mp_loss:.4f}")
        else:
            log_fn(f"--- MP iter {it} ({len(scenario_paths)} scenario file(s)) ---")
            mp_loss, cur_state = run_mp(cur_state, scenario_paths,
                                        f"mp_s{seed}_i{it}",
                                        xi1_np, xi2_np, xi1_t, xi2_t,
                                        best_lr_net, log_fn)
            torch.save({"model_state_dict": cur_state, "best_loss": float(mp_loss),
                        "seed": seed, "ccg_iter": it}, mp_ckpt)
            log_fn(f"MP iter {it} done: loss={mp_loss:.4f}")

        ccg_results.append({"iter": it, "ap_mean": ap_mean, "mp_loss": mp_loss,
                             "delta": delta, "delta_pct": delta_pct, "converged": False})

    final_model = Model()
    final_model.load_state_dict(cur_state, strict=False)
    final_Q10 = torch.nn.functional.softplus(final_model.Q10).item()
    final_Q1r = torch.nn.functional.softplus(final_model.Q1r).detach().numpy().tolist()

    seed_result = {
        "seed":          seed,
        "saa_obj":       float(saa_obj),
        "saa_Q10":       float(saa_Q10),
        "saa_Q1r":       saa_Q1r.tolist(),
        "lr_net_used":   best_lr_net,
        "nominal_loss":  float(nom_loss),
        "ccg_iters":     len(ccg_results),
        "final_mp_loss": float(mp_loss),
        "final_Q10":     final_Q10,
        "final_Q1r":     final_Q1r,
        "converged":     converged,
        "ccg_trajectory": ccg_results,
        "total_time_s":  time.time() - t_seed,
    }
    with open(summ_path, "w") as f:
        json.dump(seed_result, f, indent=2)

    log_fn(f"SEED {seed} DONE: nominal={nom_loss:.4f}, final_mp={mp_loss:.4f}, "
           f"Q10={final_Q10:.3f}, Q1r={np.round(final_Q1r,3)}, "
           f"converged={converged}, t={time.time()-t_seed:.0f}s")
    log_f.close()
    return seed_result


def run_seed_worker(args):
    """Top-level wrapper for multiprocessing (must be picklable)."""
    seed, best_lr_net = args
    return run_seed(seed, best_lr_net)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    cfg = Cfg()
    os.makedirs(cfg.BASE_DIR, exist_ok=True)

    # ---- LR SEARCH (sequential, on LR_SEARCH_SEED) ----
    print(f"\n{'='*70}")
    print(f"LR_NET SEARCH on data seed {cfg.LR_SEARCH_SEED}")
    print(f"Grid: {cfg.GRID_LR_NET}")
    print(f"{'='*70}")

    lr_xi1, lr_xi2 = generate_data(cfg.LR_SEARCH_SEED)
    lr_t1 = torch.tensor(lr_xi1, dtype=torch.float32)
    lr_t2 = torch.tensor(lr_xi2, dtype=torch.float32)
    lr_saa_Q10, lr_saa_Q1r, lr_saa_obj = solve_saa(lr_xi1, lr_xi2)
    print(f"SAA on seed {cfg.LR_SEARCH_SEED}: {lr_saa_obj:.4f}")

    best_lr_net  = cfg.GRID_LR_NET[0]
    best_lr_loss = float('inf')

    for lr_net in cfg.GRID_LR_NET:
        gpath = grid_ckpt(lr_net)
        if os.path.exists(gpath):
            g_loss = torch.load(gpath, map_location="cpu")["best_loss"]
            gap    = (g_loss - lr_saa_obj) / abs(lr_saa_obj) * 100
            print(f"  [SKIP] lr_net={lr_net} → loss={g_loss:.4f}  gap={gap:+.2f}%")
        else:
            model  = make_model_saa_init(lr_saa_Q10, lr_saa_Q1r)
            g_loss, g_state = train(
                model,
                lambda m: m.per_sample_cost(lr_t1, lr_t2).mean(),
                f"lrsearch_lr{lr_net}", lr_net
            )
            torch.save({"model_state_dict": g_state, "best_loss": float(g_loss),
                        "lr_net": lr_net}, gpath)
            gap = (g_loss - lr_saa_obj) / abs(lr_saa_obj) * 100
            print(f"  lr_net={lr_net} → loss={g_loss:.4f}  gap={gap:+.2f}%")
        g_loss = torch.load(gpath, map_location="cpu")["best_loss"]
        if g_loss < best_lr_loss:
            best_lr_loss = g_loss
            best_lr_net  = lr_net

    print(f"\nBest LR_NET = {best_lr_net}  (loss={best_lr_loss:.4f})")
    print(f"\nLaunching {len(cfg.SEEDS)} seeds in parallel ({cfg.N_WORKERS} workers)...")

    # ---- PARALLEL SEED RUNS ----
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=cfg.N_WORKERS) as pool:
        results_list = pool.map(
            run_seed_worker,
            [(s, best_lr_net) for s in cfg.SEEDS]
        )

    all_seed_results = {r["seed"]: r for r in results_list if r is not None}

    # ---- STATISTICS ----
    print(f"\n\n{'='*70}")
    print("STATISTICS ACROSS ALL SEEDS")
    print(f"N={cfg.N_SAMPLES}, RHO={cfg.RHO_1}, lr_Q={cfg.LR_Q}, lr_net={best_lr_net}")
    print(f"{'='*70}")

    nom_losses = [all_seed_results[s]["nominal_loss"]  for s in cfg.SEEDS if s in all_seed_results]
    mp_losses  = [all_seed_results[s]["final_mp_loss"] for s in cfg.SEEDS if s in all_seed_results]
    q10_vals   = [all_seed_results[s]["final_Q10"]     for s in cfg.SEEDS if s in all_seed_results]
    ccg_iters  = [all_seed_results[s]["ccg_iters"]     for s in cfg.SEEDS if s in all_seed_results]
    conv_count = sum(all_seed_results[s]["converged"]  for s in cfg.SEEDS if s in all_seed_results)

    print(f"\nNominal loss:  mean={np.mean(nom_losses):.4f}  std={np.std(nom_losses):.4f}")
    print(f"Final MP loss: mean={np.mean(mp_losses):.4f}  std={np.std(mp_losses):.4f}")
    print(f"Final Q10:     mean={np.mean(q10_vals):.4f}  std={np.std(q10_vals):.4f}")
    print(f"CCG iters:     mean={np.mean(ccg_iters):.1f}  min={np.min(ccg_iters)}  max={np.max(ccg_iters)}")
    print(f"Converged:     {conv_count}/{len(cfg.SEEDS)}")

    print(f"\n{'Seed':>5} | {'Nom.Loss':>9} | {'MP.Loss':>9} | {'Q10':>7} | {'Q1r':>22} | {'iters':>5} | Conv")
    print("-" * 80)
    for s in cfg.SEEDS:
        if s not in all_seed_results:
            print(f"  {s:3d} | {'—':>9}")
            continue
        r = all_seed_results[s]
        print(f"  {s:3d} | {r['nominal_loss']:9.4f} | {r['final_mp_loss']:9.4f} | "
              f"{r['final_Q10']:7.3f} | {str(np.round(r['final_Q1r'],3)):>22} | "
              f"{r['ccg_iters']:5d} | {'YES' if r['converged'] else 'NO'}")

    out_path = os.path.join(cfg.BASE_DIR, "results_N800_saainit.txt")
    with open(out_path, "w") as f:
        f.write(f"N={cfg.N_SAMPLES}, RHO={cfg.RHO_1}, lr_net={best_lr_net}, lr_Q={cfg.LR_Q}\n\n")
        for s in cfg.SEEDS:
            if s not in all_seed_results: continue
            r = all_seed_results[s]
            f.write(json.dumps(r, indent=2) + "\n\n")
    print(f"\nResults saved to {out_path}")
    print("[Done]")
