"""
run_rho_seed_net16_v2_N50_R10.py  --  single (rho, seed) CCG worker, R=10, N=50
=================================================================================
Usage:  python run_rho_seed_net16_v2_N50_R10.py --rho 0.30 --seed 5
"""

import argparse, os, json, time, sys
import numpy as np
from scipy.stats import truncnorm
import torch, torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB

parser = argparse.ArgumentParser()
parser.add_argument("--rho",      type=float, required=True)
parser.add_argument("--seed",     type=int,   required=True)
parser.add_argument("--max_iter", type=int,   default=40)
parser.add_argument("--scs_max_iters", type=int, default=200000)
args = parser.parse_args()

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

RHO  = args.rho
SEED = args.seed
_rho_str = f"{RHO:.6f}".rstrip('0')
_parts = _rho_str.split('.')
if _rho_str.endswith('.'):
    _rho_str += '000'
elif len(_parts) > 1 and len(_parts[1]) < 3:
    _rho_str += '0' * (3 - len(_parts[1]))
RHO_TAG = _rho_str.replace(".", "p")

NOMINAL_DIR = "/data/home/aviyac/retailers_problem/R10/N50/saainit_net16_v2"
SWEEP_DIR   = "/data/home/aviyac/retailers_problem/R10/N50/rho_sweep_net16_v2"
RHO_DIR     = os.path.join(SWEEP_DIR, f"rho_{RHO_TAG}")
os.makedirs(RHO_DIR, exist_ok=True)

R          = 10
c, h, b    = 0.25, 0.05, 0.5
N_SAMPLES  = 50
LR_NET     = 0.05       # will be overridden by winner from saainit LR search
LR_Q       = 0.01
EPOCHS     = 3000
EARLY_STOP = 300
SCHED_PAT  = 150
SCHED_FAC  = 0.7
SCS_EPS    = 1e-4
SCS_ITERS  = args.scs_max_iters
RHO_1 = RHO_2 = RHO
CCG_MAX    = args.max_iter
CCG_CONV   = 1.0

def log(*a): print(*a, flush=True)

def hms(secs):
    h = int(secs // 3600); m = int((secs % 3600) // 60); s = int(secs % 60)
    return f"{h}:{m:02d}:{s:02d}"

def softplus_np(x):
    return np.log1p(np.exp(x))

def generate_data(seed):
    np.random.seed(seed)
    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5, size=(N_SAMPLES, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

def serial(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, float)): return float(obj)
    return obj

# ── Model ─────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R * 2 + 1, 16), nn.ReLU(),
            nn.Linear(16, 16),         nn.ReLU(),
            nn.Linear(16, R))
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))
        x = cp.Variable(R); zp = cp.Parameter(R); Q10p = cp.Parameter(1)
        self.proj_layer = CvxpyLayer(
            cp.Problem(cp.Minimize(cp.sum_squares(x - zp)),
                       [cp.sum(x) <= Q10p + 1e-6, x >= 0]),
            parameters=[zp, Q10p], variables=[x])

    def per_sample_cost(self, xi1, xi2):
        N   = xi1.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        ni  = torch.cat([xi1, q1r.unsqueeze(0).expand(N, -1), q10.expand(N, 1)], dim=1)
        z   = self.fc_layers(ni)
        q2, = self.proj_layer(z, q10.expand(N, 1),
                              solver_args={"solve_method": "SCS",
                                           "eps": SCS_EPS, "max_iters": SCS_ITERS})
        s1  = c * (q10 + q1r.sum()) + h * q10
        t1  = b * (xi2 + xi1 - q2 - q1r) - h * q2
        t2  = h * (q1r - xi1 - xi2)
        t3  = b * (xi1 - q1r) - h * xi2
        return s1 + torch.maximum(torch.maximum(t1, t2), t3).sum(dim=1)

# ── Training ──────────────────────────────────────────────────────────────────
def train(model, loss_fn, label):
    opt = torch.optim.Adam([
        {'params': model.fc_layers.parameters(), 'lr': LR_NET},
        {'params': [model.Q10, model.Q1r],       'lr': LR_Q}])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'min', factor=SCHED_FAC, patience=SCHED_PAT, threshold=1e-3)
    best, bstate, noimpr = float('inf'), None, 0
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss.item())
        v = loss.item()
        if v < best - 1e-4:
            best   = v
            bstate = {k: w.clone() for k, w in model.state_dict().items() if not w.is_sparse}
            noimpr = 0
        else:
            noimpr += 1
        if ep % 100 == 0 or ep == 1:
            q10v = torch.nn.functional.softplus(model.Q10).item()
            log(f"  [{label}] ep {ep:4d} | loss={v:.4f} | Q10={q10v:.3f} | "
                f"ni={noimpr}/{EARLY_STOP} | t={hms(time.time()-t0)}")
        if noimpr >= EARLY_STOP:
            log(f"  [{label}] Early stop ep={ep}, best={best:.4f}, t={hms(time.time()-t0)}")
            break
    model.load_state_dict(bstate, strict=False)
    return best, bstate

# ── Adversarial Problem ───────────────────────────────────────────────────────
def load_weights(sd):
    Q10_v = softplus_np(sd['Q10'].numpy().astype(np.float64))[0]
    Q1r_v = softplus_np(sd['Q1r'].numpy().astype(np.float64))
    W1f   = sd['fc_layers.0.weight'].numpy()
    return {'W1':  W1f[:, :R],
            'b1':  sd['fc_layers.0.bias'].numpy()
                    + W1f[:, R:2*R] @ Q1r_v
                    + W1f[:, 2*R:]  @ [Q10_v],
            'W2':  sd['fc_layers.2.weight'].numpy(),
            'b2':  sd['fc_layers.2.bias'].numpy(),
            'W3':  sd['fc_layers.4.weight'].numpy(),
            'b3':  sd['fc_layers.4.bias'].numpy(),
            'Q10': Q10_v, 'Q1r': Q1r_v}

def solve_ap_one(w, xi1h, xi2h, idx):
    W1, b1 = w['W1'], w['b1']
    W2, b2 = w['W2'], w['b2']
    W3, b3 = w['W3'], w['b3']
    Q10v, Q1rv = w['Q10'], w['Q1r']
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", 1); env.start()
    m   = gp.Model(f"AP_{idx}", env=env)
    xi1 = m.addVars(R, lb=np.maximum(0, xi1h - RHO_1), ub=xi1h + RHO_1)
    xi2 = m.addVars(R, lb=np.maximum(0, xi2h - RHO_2), ub=xi2h + RHO_2)
    def relu_layer(inp, W, b, tag):
        n  = W.shape[0]
        xp = m.addVars(n, lb=0, name=f"xp{tag}")
        xm = m.addVars(n, lb=0, name=f"xm{tag}")
        z  = m.addVars(n, vtype=GRB.BINARY, name=f"z{tag}")
        for j in range(n):
            ex = gp.quicksum(W[j, k] * inp[k] for k in range(W.shape[1])) + b[j]
            m.addConstr(ex == xp[j] - xm[j])
            m.addGenConstrIndicator(z[j], 1, xp[j] == 0)
            m.addGenConstrIndicator(z[j], 0, xm[j] == 0)
        return xp
    l1 = relu_layer(xi1, W1, b1, "1")
    l2 = relu_layer(l1,  W2, b2, "2")
    zs = m.addVars(R, lb=-GRB.INFINITY)
    for j in range(R):
        m.addConstr(zs[j] == gp.quicksum(W3[j, k] * l2[k] for k in range(W3.shape[1])) + b3[j])
    q2  = m.addVars(R, lb=0)
    lam = m.addVar(lb=0)
    mu  = m.addVars(R, lb=0)
    for r in range(R):
        m.addConstr(2 * q2[r] - 2 * zs[r] + lam - mu[r] == 0)
    m.addConstr(gp.quicksum(q2[r] for r in range(R)) <= Q10v)
    ul = m.addVar(vtype=GRB.BINARY)
    m.addGenConstrIndicator(ul, 1, lam == 0)
    m.addGenConstrIndicator(ul, 0, Q10v - gp.quicksum(q2[r] for r in range(R)) == 0)
    um = m.addVars(R, vtype=GRB.BINARY)
    for r in range(R):
        m.addGenConstrIndicator(um[r], 1, mu[r] == 0)
        m.addGenConstrIndicator(um[r], 0, q2[r] == 0)
    v = m.addVars(R, lb=-GRB.INFINITY)
    for r in range(R):
        g1, g2, g3 = (m.addVar(lb=-GRB.INFINITY) for _ in range(3))
        m.addConstr(g1 == b * (xi1[r] + xi2[r] - Q1rv[r] - q2[r]) - h * q2[r])
        m.addConstr(g2 == h * (Q1rv[r] - xi1[r] - xi2[r]))
        m.addConstr(g3 == b * (xi1[r] - Q1rv[r]) - h * xi2[r])
        m.addGenConstrMax(v[r], [g1, g2, g3])
    stage1 = c * (Q10v + np.sum(Q1rv)) + h * Q10v
    m.setObjective(stage1 + gp.quicksum(v[r] for r in range(R)), GRB.MAXIMIZE)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        return (np.array([xi1[r].X for r in range(R)]),
                np.array([xi2[r].X for r in range(R)]),
                m.ObjVal)
    return None, None, None

def run_ap(sd, xi1_np, xi2_np, ap_path):
    w = load_weights(sd)
    log(f"    AP: Q10={w['Q10']:.4f}")
    sc = []; t0 = time.time()
    for i in range(N_SAMPLES):
        wx1, wx2, mc = solve_ap_one(w, xi1_np[i], xi2_np[i], i)
        if wx1 is not None:
            sc.append({"ball_id": i, "xi1_hat": xi1_np[i], "xi2_hat": xi2_np[i],
                       "worst_xi1": wx1, "worst_xi2": wx2, "max_cost": mc})
    costs = [s["max_cost"] for s in sc]
    log(f"    AP done: n={len(sc)}, mean={np.mean(costs):.4f}, t={hms(time.time()-t0)}")
    with open(ap_path, "w") as f:
        json.dump([{k: serial(v) for k, v in s.items()} for s in sc], f)
    return float(np.mean(costs))

def run_mp(sd, sp_list, label, xi1_t, xi2_t, xi1_np, xi2_np):
    ab1, ab2 = {}, {}
    for p in sp_list:
        with open(p) as f:
            for s in json.load(f):
                bid = s["ball_id"]
                ab1.setdefault(bid, []).append(s["worst_xi1"])
                ab2.setdefault(bid, []).append(s["worst_xi2"])
    mc   = max(len(v) for v in ab1.values())
    ac1  = [xi1_np.copy() for _ in range(mc)]
    ac2  = [xi2_np.copy() for _ in range(mc)]
    for bid in ab1:
        for k, (a1, a2) in enumerate(zip(ab1[bid], ab2[bid])):
            ac1[k][bid] = a1; ac2[k][bid] = a2
    at1  = [torch.tensor(c, dtype=torch.float32) for c in ac1]
    at2  = [torch.tensor(c, dtype=torch.float32) for c in ac2]
    model = Model(); model.load_state_dict(sd, strict=False)
    def lf(m):
        cols = [m.per_sample_cost(xi1_t, xi2_t)]
        for t1, t2 in zip(at1, at2):
            cols.append(m.per_sample_cost(t1, t2))
        return torch.stack(cols, dim=1).max(dim=1).values.mean()
    return train(model, lf, label)

# ── Main ──────────────────────────────────────────────────────────────────────
t_start = time.time()
log(f"\n{'='*60}")
log(f"R=10  rho={RHO}  seed={SEED}  dir={RHO_DIR}")
log(f"{'='*60}")

summ_path = os.path.join(RHO_DIR, f"seed_{SEED}_summary.json")
if os.path.exists(summ_path):
    with open(summ_path) as f:
        stored = json.load(f)
    if stored.get("completed", False):
        log(f"[SKIP] rho={RHO} seed={SEED} already completed.")
        sys.exit(0)

xi1_np, xi2_np = generate_data(SEED)
xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
xi2_t = torch.tensor(xi2_np, dtype=torch.float32)

nom_ck    = torch.load(os.path.join(NOMINAL_DIR, f"seed_{SEED}_nominal.pth"), map_location="cpu")
nom_state = nom_ck["model_state_dict"]
nom_loss  = float(nom_ck["best_loss"])
# Use the LR_NET that was best in saainit (stored in checkpoint if available)
if "lr_net" in nom_ck:
    LR_NET = nom_ck["lr_net"]
log(f"Loaded nominal: loss={nom_loss:.4f}  lr_net={LR_NET}")

cur_state  = nom_state
sp_list    = []
mp_loss    = nom_loss
converged  = False
ccg_res    = []
final_mp_iter = 0

for it in range(1, CCG_MAX + 1):
    log(f"\n  === CCG iter {it} (rho={RHO} seed={SEED}, t={hms(time.time()-t_start)}) ===")

    ap_path = os.path.join(RHO_DIR, f"seed_{SEED}_ap_iter{it}.json")
    if os.path.exists(ap_path):
        with open(ap_path) as f: sc = json.load(f)
        ap_mean = float(np.mean([s["max_cost"] for s in sc]))
        log(f"    [SKIP AP] mean={ap_mean:.4f}")
    else:
        t_ap = time.time()
        ap_mean = run_ap(cur_state, xi1_np, xi2_np, ap_path)
        log(f"    AP total t={hms(time.time()-t_ap)}")

    sp_list.append(ap_path)
    dpct = (ap_mean - mp_loss) / max(abs(mp_loss), 1e-9) * 100
    log(f"    Δ = {ap_mean-mp_loss:.4f} ({dpct:.2f}%)")

    if it > 1 and abs(dpct) < CCG_CONV:
        log(f"    *** CONVERGED at iter {it} ***")
        converged = True
        ccg_res.append({"iter": it, "ap_mean": ap_mean, "mp_loss": mp_loss,
                         "delta_pct": dpct, "converged": True})
        break

    mp_ck_path = os.path.join(RHO_DIR, f"seed_{SEED}_mp_iter{it}.pth")
    if os.path.exists(mp_ck_path):
        ck        = torch.load(mp_ck_path, map_location="cpu")
        mp_loss   = float(ck["best_loss"])
        cur_state = ck["model_state_dict"]
        log(f"    [SKIP MP] loss={mp_loss:.4f}")
    else:
        t_mp = time.time()
        log(f"    --- MP iter {it} ---")
        mp_loss, cur_state = run_mp(cur_state, sp_list,
                                    f"s{SEED}_r{RHO_TAG}_i{it}",
                                    xi1_t, xi2_t, xi1_np, xi2_np)
        torch.save({"model_state_dict": cur_state, "best_loss": float(mp_loss),
                    "seed": SEED, "rho": RHO, "ccg_iter": it}, mp_ck_path)
        log(f"    MP iter {it} done: loss={mp_loss:.4f}, t={hms(time.time()-t_mp)}")

    final_mp_iter = it
    ccg_res.append({"iter": it, "ap_mean": ap_mean, "mp_loss": mp_loss,
                     "delta_pct": dpct, "converged": False})

total_t = time.time() - t_start
# --- report the incumbent: policy with the lowest worst-case (AP) cost ---
# AP at iter `it` evaluates the policy from MP iter `it-1` (iter 1 -> nominal);
# the master objective rises by construction, so the genuine robust solution is
# the iterate with the smallest adversarial cost, not the last one.
_aps = [(e["iter"], e["ap_mean"]) for e in ccg_res if "ap_mean" in e]
_best_it = min(_aps, key=lambda t: t[1])[0] if _aps else 1
inc_mp_iter = _best_it - 1
inc_loss = nom_loss if inc_mp_iter <= 0 else next(
    (e["mp_loss"] for e in ccg_res if e["iter"] == inc_mp_iter), float(mp_loss))
result = {
    "seed": SEED, "rho": RHO,
    "nominal_loss": nom_loss, "final_mp_loss": float(inc_loss),
    "final_mp_iter": inc_mp_iter,
    "ccg_iters": inc_mp_iter, "converged": converged,
    "completed": True,
    "best_ap_value": (min(v for _, v in _aps) if _aps else None),
    "best_ap_iter": _best_it,
    "last_mp_loss": float(mp_loss), "last_mp_iter": final_mp_iter,
    "ccg_trajectory": ccg_res,
    "total_time_s": total_t
}

with open(summ_path, "w") as f:
    json.dump(result, f, indent=2)

log(f"\n[Done] R=10 rho={RHO} seed={SEED} | mp_loss={mp_loss:.4f} | "
    f"iter={final_mp_iter} | conv={converged} | t={hms(total_t)}")
