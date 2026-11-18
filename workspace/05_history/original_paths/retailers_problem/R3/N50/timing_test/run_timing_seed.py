"""
run_timing_seed.py  --  Full timing benchmark for one (rho, seed)
=================================================================
Runs EVERYTHING from scratch (no cached checkpoints):
  1. SAA LP solve
  2. Nominal neural network training (SAA-initialized)
  3. CCG iterations (AP + MP) for the given rho

All phase timings are recorded in the output JSON.

Usage:
    python run_timing_seed.py --rho 0.001 --seed 0
    python run_timing_seed.py --rho 0.300 --seed 1
    python run_timing_seed.py --rho 1.500 --seed 2

Output: timing_test/rho_{tag}/seed_{seed}_timing.json
"""

import argparse, os, json, time
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
parser.add_argument("--max_iter", type=int,   default=15)
args = parser.parse_args()
t_total_start = time.time()

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

RHO  = args.rho
SEED = args.seed

_s = f"{RHO:.6f}".rstrip("0")
_p = _s.split(".")
if len(_p) > 1 and len(_p[1]) < 3:
    _s += "0" * (3 - len(_p[1]))
RHO_TAG = _s.replace(".", "p")

BASE_DIR = "/data/home/aviyac/retailers_problem/R3/N50/timing_test"
RHO_DIR  = os.path.join(BASE_DIR, f"rho_{RHO_TAG}")
os.makedirs(RHO_DIR, exist_ok=True)

OUT_PATH = os.path.join(RHO_DIR, f"seed_{SEED}_timing.json")

# ── config ────────────────────────────────────────────────────────────────────
R          = 3
c, h, b    = 0.25, 0.05, 0.5
N          = 50
LR_NET     = 0.05
LR_Q       = 0.01
EPOCHS     = 3000
EARLY_STOP = 300
SCHED_PAT  = 150
SCHED_FAC  = 0.7
SCS_EPS    = 1e-4
SCS_ITERS  = 200000
CCG_CONV   = 1.0   # % gap threshold

def log(*a): print(*a, flush=True)

def softplus_np(x):
    return np.log1p(np.exp(x))

def softplus_inverse(y):
    y = np.atleast_1d(np.array(y, dtype=np.float64))
    return np.where(y > 20, y, np.log(np.expm1(y)))

def serial(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, float)): return float(obj)
    return obj

# ── data generation ───────────────────────────────────────────────────────────
def generate_data(seed):
    np.random.seed(seed)
    a = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    xi2 = truncnorm.rvs((0.0 - mu2) / 2.5, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

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
        x = cp.Variable(R); zp = cp.Parameter(R); Q10p = cp.Parameter(1)
        self.proj_layer = CvxpyLayer(
            cp.Problem(cp.Minimize(cp.sum_squares(x - zp)),
                       [cp.sum(x) <= Q10p + 1e-6, x >= 0]),
            parameters=[zp, Q10p], variables=[x])

    def per_sample_cost(self, xi1, xi2):
        Ns  = xi1.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        ni  = torch.cat([xi1, q1r.unsqueeze(0).expand(Ns,-1), q10.expand(Ns,1)], dim=1)
        z   = self.fc_layers(ni)
        q2, = self.proj_layer(z, q10.expand(Ns,1),
                              solver_args={"solve_method":"SCS","eps":SCS_EPS,"max_iters":SCS_ITERS})
        s1  = c*(q10 + q1r.sum()) + h*q10
        t1  = b*(xi2+xi1-q2-q1r) - h*q2
        t2  = h*(q1r-xi1-xi2)
        t3  = b*(xi1-q1r) - h*xi2
        return s1 + torch.maximum(torch.maximum(t1,t2),t3).sum(dim=1)

# ── training ──────────────────────────────────────────────────────────────────
def train(model, loss_fn, label):
    opt = torch.optim.Adam([
        {'params': model.fc_layers.parameters(), 'lr': LR_NET},
        {'params': [model.Q10, model.Q1r],       'lr': LR_Q}])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'min', factor=SCHED_FAC, patience=SCHED_PAT, threshold=1e-3)
    best, bstate, noimpr = float('inf'), None, 0
    t0 = time.time()
    for ep in range(1, EPOCHS+1):
        opt.zero_grad(); loss = loss_fn(model); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(loss.item())
        v = loss.item()
        if v < best - 1e-4:
            best = v; bstate = {k: w.clone() for k,w in model.state_dict().items() if not w.is_sparse}; noimpr = 0
        else:
            noimpr += 1
        if ep % 100 == 0 or ep == 1:
            log(f"  [{label}] ep {ep:4d} | loss={v:.4f} | ni={noimpr}/{EARLY_STOP} | t={time.time()-t0:.0f}s")
        if noimpr >= EARLY_STOP:
            log(f"  [{label}] Early stop ep={ep}, best={best:.4f}"); break
    model.load_state_dict(bstate, strict=False)
    return best, bstate, time.time() - t0

# ── SAA LP ────────────────────────────────────────────────────────────────────
def solve_saa(xi1_np, xi2_np):
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m   = gp.Model(env=env)
    Q10v = m.addVar(lb=0, name="Q10")
    Q1rv = m.addVars(R, lb=0, name="Q1r")
    Q2v  = m.addVars(N, R, lb=0, name="Q2")
    vv   = m.addVars(N, R, lb=-GRB.INFINITY, name="v")
    for i in range(N):
        m.addConstr(gp.quicksum(Q2v[i,r] for r in range(R)) <= Q10v)
        for r in range(R):
            m.addConstr(vv[i,r] >= b*(xi1_np[i,r]+xi2_np[i,r]-Q2v[i,r]-Q1rv[r]) - h*Q2v[i,r])
            m.addConstr(vv[i,r] >= h*(Q1rv[r]-xi1_np[i,r]-xi2_np[i,r]))
            m.addConstr(vv[i,r] >= b*(xi1_np[i,r]-Q1rv[r]) - h*xi2_np[i,r])
    m.setObjective(c*(Q10v+gp.quicksum(Q1rv[r] for r in range(R)))+h*Q10v
                   +(1.0/N)*gp.quicksum(vv[i,r] for i in range(N) for r in range(R)), GRB.MINIMIZE)
    m.optimize()
    return Q10v.X, np.array([Q1rv[r].X for r in range(R)]), m.ObjVal

# ── adversarial problem ───────────────────────────────────────────────────────
def load_weights(sd):
    Q10v = softplus_np(sd['Q10'].numpy())[0]
    Q1rv = softplus_np(sd['Q1r'].numpy())
    W1f  = sd['fc_layers.0.weight'].numpy()
    return {'W1': W1f[:,:R],
            'b1': sd['fc_layers.0.bias'].numpy() + W1f[:,R:2*R]@Q1rv + W1f[:,2*R:]@[Q10v],
            'W2': sd['fc_layers.2.weight'].numpy(), 'b2': sd['fc_layers.2.bias'].numpy(),
            'W3': sd['fc_layers.4.weight'].numpy(), 'b3': sd['fc_layers.4.bias'].numpy(),
            'Q10': Q10v, 'Q1r': Q1rv}

def solve_ap_one(w, xi1h, xi2h, idx):
    env = gp.Env(empty=True); env.setParam("OutputFlag",0); env.setParam("Threads",1); env.start()
    m   = gp.Model(f"AP_{idx}", env=env)
    xi1 = m.addVars(R, lb=np.maximum(0, xi1h-RHO), ub=xi1h+RHO)
    xi2 = m.addVars(R, lb=np.maximum(0, xi2h-RHO), ub=xi2h+RHO)
    def relu_layer(inp, W, b_vec, tag):
        n = W.shape[0]
        xp = m.addVars(n, lb=0, name=f"xp{tag}"); xm = m.addVars(n, lb=0, name=f"xm{tag}")
        z  = m.addVars(n, vtype=GRB.BINARY, name=f"z{tag}")
        for j in range(n):
            ex = gp.quicksum(W[j,k]*inp[k] for k in range(W.shape[1])) + b_vec[j]
            m.addConstr(ex == xp[j]-xm[j])
            m.addGenConstrIndicator(z[j],1,xp[j]==0); m.addGenConstrIndicator(z[j],0,xm[j]==0)
        return xp
    l1 = relu_layer(xi1, w['W1'], w['b1'], "1")
    l2 = relu_layer(l1,  w['W2'], w['b2'], "2")
    zs = m.addVars(R, lb=-GRB.INFINITY)
    for j in range(R):
        m.addConstr(zs[j]==gp.quicksum(w['W3'][j,k]*l2[k] for k in range(w['W3'].shape[1]))+w['b3'][j])
    q2=m.addVars(R,lb=0); lam=m.addVar(lb=0); mu=m.addVars(R,lb=0)
    for r in range(R):
        m.addConstr(2*q2[r]-2*zs[r]+lam-mu[r]==0)
    m.addConstr(gp.quicksum(q2[r] for r in range(R))<=w['Q10'])
    ul=m.addVar(vtype=GRB.BINARY)
    m.addGenConstrIndicator(ul,1,lam==0); m.addGenConstrIndicator(ul,0,w['Q10']-gp.quicksum(q2[r] for r in range(R))==0)
    um=m.addVars(R,vtype=GRB.BINARY)
    for r in range(R):
        m.addGenConstrIndicator(um[r],1,mu[r]==0); m.addGenConstrIndicator(um[r],0,q2[r]==0)
    v=m.addVars(R,lb=-GRB.INFINITY)
    for r in range(R):
        g1,g2,g3 = (m.addVar(lb=-GRB.INFINITY) for _ in range(3))
        m.addConstr(g1==b*(xi1[r]+xi2[r]-w['Q1r'][r]-q2[r])-h*q2[r])
        m.addConstr(g2==h*(w['Q1r'][r]-xi1[r]-xi2[r]))
        m.addConstr(g3==b*(xi1[r]-w['Q1r'][r])-h*xi2[r])
        m.addGenConstrMax(v[r],[g1,g2,g3])
    stage1 = c*(w['Q10']+np.sum(w['Q1r']))+h*w['Q10']
    m.setObjective(stage1+gp.quicksum(v[r] for r in range(R)),GRB.MAXIMIZE)
    m.optimize()
    if m.status==GRB.OPTIMAL:
        return np.array([xi1[r].X for r in range(R)]), np.array([xi2[r].X for r in range(R)]), m.ObjVal
    return None, None, None

def run_ap(sd, xi1_np, xi2_np):
    w = load_weights(sd)
    sc = []; t0 = time.time()
    for i in range(N):
        wx1, wx2, mc = solve_ap_one(w, xi1_np[i], xi2_np[i], i)
        if wx1 is not None:
            sc.append({"ball_id":i,"xi1_hat":xi1_np[i],"xi2_hat":xi2_np[i],
                       "worst_xi1":wx1,"worst_xi2":wx2,"max_cost":mc})
    costs = [s["max_cost"] for s in sc]
    ap_time = time.time() - t0
    log(f"    AP done: n={len(sc)}, mean={np.mean(costs):.4f}, t={ap_time:.0f}s")
    return float(np.mean(costs)), sc, ap_time

def run_mp(sd, all_scenarios, label, xi1_t, xi2_t, xi1_np, xi2_np):
    ab1, ab2 = {}, {}
    for sc in all_scenarios:
        for s in sc:
            bid = s["ball_id"]
            ab1.setdefault(bid,[]).append(s["worst_xi1"]); ab2.setdefault(bid,[]).append(s["worst_xi2"])
    mc = max(len(v) for v in ab1.values())
    ac1=[xi1_np.copy() for _ in range(mc)]; ac2=[xi2_np.copy() for _ in range(mc)]
    for bid in ab1:
        for k,(a1,a2) in enumerate(zip(ab1[bid],ab2[bid])):
            ac1[k][bid]=a1; ac2[k][bid]=a2
    at1=[torch.tensor(c,dtype=torch.float32) for c in ac1]
    at2=[torch.tensor(c,dtype=torch.float32) for c in ac2]
    model=Model(); model.load_state_dict(sd,strict=False)
    def lf(m):
        cols=[m.per_sample_cost(xi1_t,xi2_t)]
        for t1,t2 in zip(at1,at2): cols.append(m.per_sample_cost(t1,t2))
        return torch.stack(cols,dim=1).max(dim=1).values.mean()
    return train(model, lf, label)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
log(f"\n{'='*60}")
log(f"TIMING BENCHMARK  rho={RHO}  seed={SEED}")
log(f"{'='*60}")

xi1_np, xi2_np = generate_data(SEED)
xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
xi2_t = torch.tensor(xi2_np, dtype=torch.float32)

# 1. SAA LP
log("\n--- Phase 1: SAA LP ---")
t0 = time.time()
saa_Q10, saa_Q1r, saa_obj = solve_saa(xi1_np, xi2_np)
saa_time = time.time() - t0
log(f"  SAA done: Q10={saa_Q10:.4f}  obj={saa_obj:.4f}  t={saa_time:.1f}s")

# 2. Nominal training
log("\n--- Phase 2: Nominal training ---")
torch.manual_seed(0)
model = Model()
with torch.no_grad():
    model.Q10.data = torch.tensor(softplus_inverse(saa_Q10), dtype=torch.float32)
    model.Q1r.data = torch.tensor(softplus_inverse(saa_Q1r), dtype=torch.float32)
nom_loss, nom_state, nom_time = train(
    model, lambda m: m.per_sample_cost(xi1_t, xi2_t).mean(), f"nom_s{SEED}")
log(f"  Nominal done: loss={nom_loss:.4f}  t={nom_time:.0f}s")

# 3. CCG iterations
log(f"\n--- Phase 3: CCG  (rho={RHO}) ---")
cur_state = nom_state
all_scenarios = []
mp_loss   = nom_loss
converged = False
ccg_traj  = []

for it in range(1, args.max_iter + 1):
    log(f"\n  === CCG iter {it} ===")

    ap_mean, sc, ap_time = run_ap(cur_state, xi1_np, xi2_np)
    all_scenarios.append(sc)
    dpct = (ap_mean - mp_loss) / max(abs(mp_loss), 1e-9) * 100
    log(f"    Δ = {ap_mean-mp_loss:.4f} ({dpct:.2f}%)")

    if it > 1 and dpct < CCG_CONV:
        log(f"    *** CONVERGED at iter {it} ***")
        converged = True
        ccg_traj.append({"iter":it,"ap_mean":ap_mean,"mp_loss":mp_loss,
                          "delta_pct":dpct,"converged":True,
                          "ap_time_s":round(ap_time,2),"mp_time_s":0.0})
        break

    log(f"    --- MP iter {it} ---")
    mp_loss, cur_state, mp_time = run_mp(cur_state, all_scenarios,
                                          f"s{SEED}_r{RHO_TAG}_i{it}",
                                          xi1_t, xi2_t, xi1_np, xi2_np)
    log(f"    MP done: loss={mp_loss:.4f}  t={mp_time:.0f}s")
    ccg_traj.append({"iter":it,"ap_mean":ap_mean,"mp_loss":mp_loss,
                      "delta_pct":dpct,"converged":False,
                      "ap_time_s":round(ap_time,2),"mp_time_s":round(mp_time,2)})

total_time = time.time() - t_total_start
ccg_time   = sum(t.get("ap_time_s",0)+t.get("mp_time_s",0) for t in ccg_traj)

result = {
    "seed": SEED, "rho": RHO,
    "saa_time_s":     round(saa_time, 2),
    "nominal_time_s": round(nom_time, 2),
    "ccg_iters":      len(ccg_traj),
    "ccg_time_s":     round(ccg_time, 2),
    "total_time_s":   round(total_time, 2),
    "converged":      converged,
    "nom_loss":       float(nom_loss),
    "final_mp_loss":  float(mp_loss),
    "ccg_trajectory": ccg_traj,
}

with open(OUT_PATH, "w") as f:
    json.dump(result, f, indent=2)

log(f"\n[Done] rho={RHO}  seed={SEED}")
log(f"  SAA:     {saa_time:.1f}s")
log(f"  Nominal: {nom_time:.0f}s  ({nom_time/60:.1f}min)")
log(f"  CCG:     {ccg_time:.0f}s  ({ccg_time/60:.1f}min)  [{len(ccg_traj)} iters]")
log(f"  TOTAL:   {total_time:.0f}s  ({total_time/60:.1f}min)")
log(f"Saved: {OUT_PATH}")
