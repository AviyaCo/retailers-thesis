"""
run_cv_nominal.py — Per-seed 5-fold CV for LR selection, shipping problem N=200
================================================================================
For each seed 0-9:
    1. Solve SAA-MILP on all 200 points → Q10/Q1r initialization + benchmark
    2. 5-fold CV:
       for each fold k:
         for each LR in LR_GRID:
           train on 40 pts (no verbose), eval mean cost on 10 val pts
  3. Pick LR with lowest mean val cost across 5 folds
  4. Retrain final model on all 50 pts with chosen LR
  5. Save seed_{k}_nominal.pth  (skips if cv_lr_used key already present)

Outputs:
  seed_{k}_nominal.pth      checkpoint (includes cv_lr_used, cv_means)
  seed_{k}_summary.json     soft_loss / hard_loss / saa_milp / cv_lr_used
  cv_lr_results.json        summary across all seeds
"""

import argparse, os, json, time
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

# ============================================================
# CONFIGURATION
# ============================================================
class Cfg:
    R             = 3
    c             = 0.25
    h             = 0.05
    b             = 0.5
    f             = 0.1
    GAMMA         = 5.0

    N_SAMPLES     = 200
    N_FOLDS       = 5

    LR_Q          = 0.01
    EPOCHS        = 3000
    EARLY_STOP    = 300
    SCHED_PATIENCE = 150
    SCHED_FACTOR  = 0.7

    SCS_EPS       = 1e-3
    SCS_MAX_ITERS = 200000

    SEEDS         = list(range(10))
    NET_SEED      = 0
    LR_GRID       = [0.005, 0.01, 0.05]
    TRAIN_OPTIMIZER = "Adam"

    BASE_DIR = "/data/home/aviyac/retailers_with_shipping/N200"

cfg = Cfg()

FOLD_SIZE = cfg.N_SAMPLES // cfg.N_FOLDS   # 10
IDX       = np.arange(cfg.N_SAMPLES)
FOLD_VAL  = [IDX[k * FOLD_SIZE:(k + 1) * FOLD_SIZE] for k in range(cfg.N_FOLDS)]
FOLD_TR   = [np.concatenate([IDX[:k * FOLD_SIZE], IDX[(k + 1) * FOLD_SIZE:]])
             for k in range(cfg.N_FOLDS)]

def log(*a): print(*a, flush=True)

# ============================================================
# HELPERS
# ============================================================
def softplus_inverse(y):
    y = np.atleast_1d(np.array(y, dtype=np.float64))
    return np.where(y > 20, y, np.log(np.expm1(y)))

def generate_data(seed):
    np.random.seed(seed)
    a   = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a, np.inf, loc=6.0, scale=2.5, size=(cfg.N_SAMPLES, cfg.R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a2  = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

def solve_saa_with_shipping(xi1_data, xi2_data):
    """SAA MILP with fixed shipping cost.  Binary z[i,r]=1 iff Q2[i,r]>0."""
    N, R = xi1_data.shape
    BIG_M = float((xi1_data + xi2_data).max()) + 1.0   # tight: max single-retailer demand
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m   = gp.Model(env=env)
    Q10_v = m.addVar(lb=0.0, name="Q10")
    Q1r_v = m.addVars(R, lb=0.0, name="Q1r")
    Q2_v  = m.addVars(N, R, lb=0.0, name="Q2")
    z_v   = m.addVars(N, R, vtype=GRB.BINARY, name="z")
    vv    = m.addVars(N, R, lb=-GRB.INFINITY, name="v")
    for i in range(N):
        m.addConstr(gp.quicksum(Q2_v[i, r] for r in range(R)) <= Q10_v)
        for r in range(R):
            m.addConstr(vv[i,r] >= cfg.b*(xi1_data[i,r]+xi2_data[i,r]-Q2_v[i,r]-Q1r_v[r]) - cfg.h*Q2_v[i,r])
            m.addConstr(vv[i,r] >= cfg.h*(Q1r_v[r]-xi1_data[i,r]-xi2_data[i,r]))
            m.addConstr(vv[i,r] >= cfg.b*(xi1_data[i,r]-Q1r_v[r]) - cfg.h*xi2_data[i,r])
            m.addConstr(Q2_v[i, r] <= BIG_M * z_v[i, r])
    obj = (cfg.c*(Q10_v + gp.quicksum(Q1r_v[r] for r in range(R))) + cfg.h*Q10_v
           + (1.0/N)*gp.quicksum(
               vv[i,r] + cfg.f*z_v[i,r]
               for i in range(N) for r in range(R)))
    m.setObjective(obj, GRB.MINIMIZE)
    m.optimize()
    return Q10_v.X, np.array([Q1r_v[r].X for r in range(R)]), m.ObjVal

# ============================================================
# MODEL
# ============================================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(cfg.R * 2 + 1, 16), nn.ReLU(),
            nn.Linear(16, 16),             nn.ReLU(),
            nn.Linear(16, cfg.R)
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
        N   = xi1.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        net_in = torch.cat([xi1,
                            q1r.unsqueeze(0).expand(N, -1),
                            q10.expand(N, 1)], dim=1)
        z   = self.fc_layers(net_in)
        q2, = self.proj_layer(z, q10.expand(N, 1),
                              solver_args={"solve_method": "Clarabel"})
        q2 = torch.relu(q2)
        stage1    = cfg.c * (q10 + torch.sum(q1r)) + cfg.h * q10
        t1 = cfg.b * (xi2 + xi1 - q2 - q1r) - cfg.h * q2
        t2 = cfg.h * (q1r - xi1 - xi2)
        t3 = cfg.b * (xi1 - q1r) - cfg.h * xi2
        v  = torch.sum(torch.maximum(torch.maximum(t1, t2), t3), dim=1)
        z_soft    = 1.0 - torch.exp(-cfg.GAMMA * q2)
        ship_cost = cfg.f * torch.sum(z_soft, dim=1)
        return stage1 + v + ship_cost

    @torch.no_grad()
    def hard_shipping_cost(self, xi1, xi2):
        N   = xi1.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        net_in = torch.cat([xi1,
                            q1r.unsqueeze(0).expand(N, -1),
                            q10.expand(N, 1)], dim=1)
        z   = self.fc_layers(net_in)
        q2, = self.proj_layer(z, q10.expand(N, 1),
                              solver_args={"solve_method": "Clarabel"})
        stage1    = cfg.c * (q10 + torch.sum(q1r)) + cfg.h * q10
        t1 = cfg.b * (xi2 + xi1 - q2 - q1r) - cfg.h * q2
        t2 = cfg.h * (q1r - xi1 - xi2)
        t3 = cfg.b * (xi1 - q1r) - cfg.h * xi2
        v  = torch.sum(torch.maximum(torch.maximum(t1, t2), t3), dim=1)
        z_hard    = (q2 > 1e-6).float()
        ship_cost = cfg.f * torch.sum(z_hard, dim=1)
        return stage1 + v + ship_cost

def make_model(q10_saa, q1r_saa):
    torch.manual_seed(cfg.NET_SEED)
    m = Model()
    with torch.no_grad():
        m.Q10.data = torch.tensor(softplus_inverse(q10_saa), dtype=torch.float32)
        m.Q1r.data = torch.tensor(softplus_inverse(q1r_saa), dtype=torch.float32)
    return m

# ============================================================
# TRAINING
# ============================================================
def train(model, xi1_t, xi2_t, lr_net, verbose=False, label=""):
    n_train = xi1_t.shape[0]
    log(f"  [{label}] start training | optimizer={cfg.TRAIN_OPTIMIZER} | "
        f"N={n_train} | lr_net={lr_net} | lr_Q={cfg.LR_Q}")
    opt = torch.optim.Adam([
        {'params': model.fc_layers.parameters(), 'lr': lr_net},
        {'params': [model.Q10, model.Q1r],       'lr': cfg.LR_Q},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=cfg.SCHED_FACTOR,
        patience=cfg.SCHED_PATIENCE, threshold=1e-3
    )
    best_loss, best_state, no_imp = float('inf'), None, 0
    t0 = time.time()
    for ep in range(1, cfg.EPOCHS + 1):
        opt.zero_grad()
        loss = model.per_sample_cost(xi1_t, xi2_t).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        sched.step(loss.item())
        v = loss.item()
        if v < best_loss - 1e-4:
            best_loss = v
            best_state = {k: w.clone() for k, w in model.state_dict().items()
                          if not w.is_sparse}
            no_imp = 0
        else:
            no_imp += 1
        if verbose and (ep % 50 == 0 or ep == 1):
            q10v = torch.nn.functional.softplus(model.Q10).item()
            q1rv = torch.nn.functional.softplus(model.Q1r).detach().numpy()
            lr_q = opt.param_groups[1]['lr']
            log(f"  [{label}] ep {ep:4d}/{cfg.EPOCHS} | loss={v:.4f} | "
                f"Q10={q10v:.3f} | Q1r={np.round(q1rv,3)} | "
                f"lr_Q={lr_q:.5f} | no_imp={no_imp}/{cfg.EARLY_STOP} | "
                f"t={time.time()-t0:.0f}s")
        if no_imp >= cfg.EARLY_STOP:
            if verbose:
                log(f"  [{label}] Early stop ep={ep}, best={best_loss:.4f}, "
                    f"elapsed={time.time()-t0:.1f}s")
            break
    if best_state is None:
        return float('inf'), None, time.time() - t0
    model.load_state_dict(best_state, strict=False)
    log(f"  [{label}] training done | best={best_loss:.4f} | "
        f"epochs_run={ep} | elapsed={time.time()-t0:.1f}s")
    return best_loss, best_state, time.time() - t0

# ============================================================
# MAIN
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None,
                    help="Run only this seed (omit to run all)")
args = parser.parse_args()
seeds_to_run = [args.seed] if args.seed is not None else cfg.SEEDS

os.makedirs(cfg.BASE_DIR, exist_ok=True)
cv_results = {}

for seed in seeds_to_run:
    log(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
    t_seed = time.time()

    nom_ckpt_path = os.path.join(cfg.BASE_DIR, f"seed_{seed}_nominal.pth")
    summ_path     = os.path.join(cfg.BASE_DIR, f"seed_{seed}_summary.json")

    # Skip only if this exact script already produced this checkpoint
    if os.path.exists(nom_ckpt_path):
        saved = torch.load(nom_ckpt_path, map_location="cpu")
        if "cv_lr_used" in saved:
            log(f"  [SKIP] seed {seed}: cv_lr={saved['cv_lr_used']}  "
                f"soft={saved['best_loss']:.4f}  hard={saved.get('hard_loss', float('nan')):.4f}")
            cv_results[seed] = {
                "best_lr":  saved["cv_lr_used"],
                "cv_means": saved.get("cv_means", {}),
                "nom_loss": float(saved["best_loss"]),
                "hard_loss": float(saved.get("hard_loss", float("nan"))),
                "saa_obj":  float(saved.get("saa_milp_obj", 0)),
            }
            continue
        else:
            log(f"  [REDO] seed {seed}: existing checkpoint lacks cv_lr_used → rerunning with CV")

    xi1_np, xi2_np = generate_data(seed)
    xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
    xi2_t = torch.tensor(xi2_np, dtype=torch.float32)

    # SAA-MILP on full training set (init + benchmark)
    t_saa = time.time()
    saa_Q10, saa_Q1r, saa_obj = solve_saa_with_shipping(xi1_np, xi2_np)
    saa_time = time.time() - t_saa
    log(f"  SAA-MILP ref={saa_obj:.4f}  Q10={saa_Q10:.4f}  "
        f"Q1r={np.round(saa_Q1r,4)}  t={saa_time:.1f}s")

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    log(f"  {cfg.N_FOLDS}-fold CV  (train={cfg.N_SAMPLES-FOLD_SIZE}, val={FOLD_SIZE})  "
        f"LRs={cfg.LR_GRID}")
    cv_val = {lr: [] for lr in cfg.LR_GRID}

    for fold_k in range(cfg.N_FOLDS):
        t_fold = time.time()
        tr_idx  = FOLD_TR[fold_k]
        val_idx = FOLD_VAL[fold_k]
        xi1_tr_t  = torch.tensor(xi1_np[tr_idx],  dtype=torch.float32)
        xi2_tr_t  = torch.tensor(xi2_np[tr_idx],  dtype=torch.float32)
        xi1_val_t = torch.tensor(xi1_np[val_idx], dtype=torch.float32)
        xi2_val_t = torch.tensor(xi2_np[val_idx], dtype=torch.float32)

        for lr in cfg.LR_GRID:
            model = make_model(saa_Q10, saa_Q1r)
            train(model, xi1_tr_t, xi2_tr_t, lr_net=lr, verbose=False,
                  label=f"s{seed}_f{fold_k}_lr{lr}")
            with torch.no_grad():
                val_cost = model.per_sample_cost(xi1_val_t, xi2_val_t).mean().item()
            cv_val[lr].append(val_cost)

        fold_summary = "  ".join(
            f"lr={lr:.3f}→{cv_val[lr][-1]:.4f}" for lr in cfg.LR_GRID)
        log(f"    fold {fold_k}: {fold_summary} | t={time.time()-t_fold:.1f}s")

    cv_means = {lr: float(np.mean(cv_val[lr])) for lr in cfg.LR_GRID}
    best_lr  = min(cfg.LR_GRID, key=lambda lr: cv_means[lr])
    log("  CV means: " + "  ".join(
        f"lr={lr:.3f}→{cv_means[lr]:.4f}" for lr in cfg.LR_GRID))
    log(f"  >> Best LR for seed {seed}: {best_lr}  "
        f"(mean val={cv_means[best_lr]:.4f})")

    # ── Final training on all samples ──────────────────────────────────────────
    log(f"  Final training: lr={best_lr}, N={cfg.N_SAMPLES} ...")
    model = make_model(saa_Q10, saa_Q1r)
    nom_loss, nom_state, nom_time = train(
        model, xi1_t, xi2_t, lr_net=best_lr, verbose=True, label=f"nom_s{seed}")

    model.load_state_dict(nom_state, strict=False)
    hard_loss = model.hard_shipping_cost(xi1_t, xi2_t).mean().item()
    q10_val   = torch.nn.functional.softplus(model.Q10).item()
    q1r_val   = torch.nn.functional.softplus(model.Q1r).detach().numpy().tolist()

    gap_soft = (nom_loss  - saa_obj) / abs(saa_obj) * 100
    gap_hard = (hard_loss - saa_obj) / abs(saa_obj) * 100
    log(f"  Done: soft={nom_loss:.4f} ({gap_soft:+.1f}%)  "
        f"hard={hard_loss:.4f} ({gap_hard:+.1f}%)  "
        f"saa={saa_obj:.4f}  t={nom_time:.0f}s")
    seed_time = time.time() - t_seed
    log(f"  Seed {seed} total_time={seed_time:.1f}s ({seed_time/60:.2f} min)")

    torch.save({
        "model_state_dict": nom_state,
        "best_loss":    float(nom_loss),
        "hard_loss":    float(hard_loss),
        "cv_lr_used":   best_lr,
        "cv_means":     cv_means,
        "seed":         seed,
        "n_samples":    cfg.N_SAMPLES,
        "saa_Q10":      float(saa_Q10),
        "saa_Q1r":      saa_Q1r.tolist(),
        "saa_milp_obj": float(saa_obj),
        "Q10_trained":  q10_val,
        "Q1r_trained":  q1r_val,
        "optimizer":    cfg.TRAIN_OPTIMIZER,
        "seed_time_s":  float(seed_time),
    }, nom_ckpt_path)

    with open(summ_path, "w") as f:
        json.dump({"soft_loss": float(nom_loss), "hard_loss": float(hard_loss),
                   "saa_milp": float(saa_obj), "cv_lr_used": best_lr,
                   "cv_means": cv_means, "optimizer": cfg.TRAIN_OPTIMIZER,
                   "seed_time_s": float(seed_time)}, f, indent=2)

    cv_results[seed] = {
        "best_lr":  best_lr, "cv_means": cv_means,
        "nom_loss": float(nom_loss), "hard_loss": float(hard_loss),
        "saa_obj":  float(saa_obj),
    }

# ── Global summary (only when running all seeds) ──────────────────────────────
if args.seed is None:
    with open(os.path.join(cfg.BASE_DIR, "cv_lr_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2)

    log(f"\n{'='*70}\nCV LR Summary:")
    for seed in cfg.SEEDS:
        if seed not in cv_results:
            continue
        r = cv_results[seed]
        log(f"  seed {seed}: lr={r['best_lr']}  "
            f"soft={r['nom_loss']:.4f}  hard={r['hard_loss']:.4f}  "
            f"saa={r['saa_obj']:.4f}")
    log(f"\nSaved: {os.path.join(cfg.BASE_DIR, 'cv_lr_results.json')}")
