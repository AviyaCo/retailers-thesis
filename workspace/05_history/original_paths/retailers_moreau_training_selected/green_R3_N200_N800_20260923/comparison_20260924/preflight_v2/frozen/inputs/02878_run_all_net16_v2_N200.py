"""
run_all_saainit.py — Full CCG pipeline for N=50, SAA-initialized Q10/Q1r, 10 seeds
=====================================================================================
Key differences from run_N50_rho03:
  - Q10, Q1r initialized to per-seed SAA solution (solved fresh for each dataset)
  - lr_Q = 0.01  (small; warm start near SAA solution)
  - 10 seeds for DATA GENERATION (different N=50 datasets per seed)
  - Network weights initialized with a fixed torch seed (reproducible)
  - Fully automated: nominal training + CCG loop per seed

Directory: run_N50_saainit/
Per-seed files:
  seed_{k}_nominal.pth       — nominal checkpoint
  seed_{k}_ap_iter{j}.json   — adversarial scenarios for CCG iter j
  seed_{k}_mp_iter{j}.pth    — robust checkpoint after MP iter j
  seed_{k}_summary.json      — convergence results for seed k

Final output: results_saainit.txt (full log + statistics table)
"""

import os, sys, json, time
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
    N_SAMPLES     = 200

    # Training
    #LR_NET        = 0.1          # network starts random — needs larger steps
    LR_Q          = 0.01        # Q starts at SAA solution — keep it tight
    EPOCHS        = 3000
    EARLY_STOP    = 300
    SCHED_PATIENCE = 150 #how much too wait for any change
    SCHED_FACTOR  = 0.7 # how nuch to reduse lr

    SCS_EPS       = 1e-4 #the accuracy paramrter of the solver
    SCS_MAX_ITERS = 200000

    # CCG
    RHO_1         = 0.3 # the radius of the uncertainty set for xi1 (same for all dimensions)
    RHO_2         = 0.3
    CCG_MAX_ITER  = 40
    CCG_CONV_THRESH_PCT = 1.0    # declare convergence when (AP_mean - MP_loss) / MP_loss * 100 < this

    # Experiment
    SEEDS         = list(range(10))   # [0, 1, 2, ..., 9] — each is a data seed
    NET_SEED      = 0                 # fixed torch seed for network weight init
    # LR_NET search (run once on data seed 0, winner used for all seeds)
    GRID_LR_NET   = [0.005, 0.01, 0.05]
    LR_SEARCH_SEED = 0               # which data seed to use for the search
    BASE_DIR      = "/data/home/aviyac/retailers_problem/R3/N200/saainit_net16_v2"

cfg = Cfg()

# ============================================================
# HELPERS
# ============================================================
def softplus_np(x):
    return np.log1p(np.exp(x))

def softplus_inverse(y):
    """Inverse of softplus: log(exp(y) - 1). Safe for y >> 1."""
    y = np.atleast_1d(np.array(y, dtype=np.float64))
    return np.where(y > 20, y, np.log(np.expm1(y)))

def solve_saa(xi1_data, xi2_data):
    """Solve SAA LP with Gurobi. Returns (Q10, Q1r, obj_val)."""
    N, R = xi1_data.shape
    '''env fot guroby which we declare its params. 
    we set output flag to 0 to surpress the solver output.'''
    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start() 
    m = gp.Model(env=env) #the model
    Q10_v = m.addVar(lb=0.0, name="Q10") #first stage variable the wearhouse order, non negative.
    Q1r_v = m.addVars(R, lb=0.0, name="Q1r") #first stage variable for each r, same for all samples, non negative.
    Q2_v  = m.addVars(N, R, lb=0.0, name="Q2") #second stage variable for each sample and r. depends on the scenario, so separate variables per sample,non negative.
    vv    = m.addVars(N, R, lb=-GRB.INFINITY, name="v") #auxiliary variables for each sample and r, can be negative. need to br the maximum between the 3 terms in the cost function. so we bound it from below by each of the 3 terms, and minimize it in the objective.
    #for each sample, we have the same Q10 and Q1r, but different Q2. the constraints are per sample, and the objective is the average over samples.
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

def grid_ckpt(lr_net): #filename for a given lr_net in the grid search (used to save/load checkpoints)
    lr_tag = f"{lr_net:.4f}".replace('.', 'p') #to make it a valid filename (e.g., "0.0100" → "0p0100")
    return os.path.join(cfg.BASE_DIR, f"lrsearch_lr{lr_tag}.pth")

def ckpt(seed, tag):
    return os.path.join(cfg.BASE_DIR, f"seed_{seed}_{tag}.pth")

def ap_json(seed, k):
    return os.path.join(cfg.BASE_DIR, f"seed_{seed}_ap_iter{k}.json")

def summary_json(seed):
    return os.path.join(cfg.BASE_DIR, f"seed_{seed}_summary.json")

def to_serializable(obj): #convert numpy types to native Python types for JSON serialization - json dont like some staff
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, float)): return float(obj)
    return obj

def generate_data(seed): #generate dataset (xi1, xi2) for a given seed.
    np.random.seed(seed)
    a_xi1 = (0.0 - 6.0) / 2.5  # lower-clip for xi1 ~ TruncatedNormal(6, 2.5, 0, inf)
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5,
                        size=(cfg.N_SAMPLES, cfg.R))
    mu2 = 2 * (6.0 - xi1) ** 2  # shape (N, R)
    a_xi2 = (0.0 - mu2) / 2.5   # per-element lower-clip
    xi2 = truncnorm.rvs(a_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2

# Placeholders — overwritten per seed in the main loop
xi1_np, xi2_np = None, None
xi1_t, xi2_t   = None, None

# ============================================================
# MODEL
# ============================================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple 3-layer MLP with ReLU activations. Input is (xi1, Q1r, Q10), output is z for the projection layer.
        self.fc_layers = nn.Sequential(
            nn.Linear(cfg.R * 2 + 1, 16), nn.ReLU(),
            nn.Linear(16, 16),             nn.ReLU(),
            nn.Linear(16, cfg.R)
        )
        self.Q10 = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.Q1r = nn.Parameter(torch.zeros(cfg.R, dtype=torch.float32))
    # Proj layer for the Q2 optimization: min_x ||x - z||^2 s.t. sum(x) <= Q10, x >= 0 closest point to z in the feasible set of Q2.
    #  We use cvxpylayers to solve this projection as a differentiable layer in the network.
        x       = cp.Variable(cfg.R)
        z_param = cp.Parameter(cfg.R)
        Q10_p   = cp.Parameter(1)
        cons    = [cp.sum(x) <= Q10_p + 1e-6, x >= 0] # add small tolerance to the sum constraint to avoid numerical issues with SCS
        self.proj_layer = CvxpyLayer(
            cp.Problem(cp.Minimize(cp.sum_squares(x - z_param)), cons),
            parameters=[z_param, Q10_p], variables=[x]
        )

    def per_sample_cost(self, xi1, xi2): 
        N   = xi1.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        # we duplicate q10 and q1r for each sample to concatenate with xi1, 
        # then feed into the network to get z for the projection layer.
        #  the projection layer gives us q2, which we use to compute the cost terms.
        net_in = torch.cat([xi1,
                            q1r.unsqueeze(0).expand(N, -1),
                            q10.expand(N, 1)], dim=1)
        z = self.fc_layers(net_in) # shape (N, R)
        q2, = self.proj_layer(z, q10.expand(N, 1),
                              solver_args={"solve_method": "SCS",
                                           "eps": cfg.SCS_EPS,
                                           "max_iters": cfg.SCS_MAX_ITERS})
        stage1  = cfg.c * (q10 + torch.sum(q1r)) + cfg.h * q10
        t1 = cfg.b * (xi2 + xi1 - q2 - q1r) - cfg.h * q2
        t2 = cfg.h * (q1r - xi1 - xi2)
        t3 = cfg.b * (xi1 - q1r) - cfg.h * xi2
        return stage1 + torch.sum(torch.maximum(torch.maximum(t1, t2), t3), dim=1) 
    # shape (N,) the cost for each sample

def make_model_saa_init(q10_saa, q1r_saa, net_seed=None):
    """Create model with network weights from net_seed (default: cfg.NET_SEED), Q10/Q1r from SAA."""
    torch.manual_seed(net_seed if net_seed is not None else cfg.NET_SEED)
    m = Model()
    with torch.no_grad(): #dont compute gradients for this initialization step, we will finetune from here
        m.Q10.data = torch.tensor(softplus_inverse(q10_saa),  dtype=torch.float32)
        m.Q1r.data = torch.tensor(softplus_inverse(q1r_saa),  dtype=torch.float32)
    return m

# ============================================================
# TRAINING (nominal or MP — same logic, different loss fn)
# ============================================================
def train(model, loss_fn, label, epochs=None, log=True):
    """Train model using loss_fn(model). Returns best_loss, best_state_dict."""
    epochs = epochs or cfg.EPOCHS
    #optimizer is ADAM, with different learning rates for the network weights and the Q10/Q1r parameters (Q10/Q1r have smaller lr since they are initialized near the SAA solution)
    opt = torch.optim.Adam([
        {'params': model.fc_layers.parameters(), 'lr': cfg.LR_NET},
        {'params': [model.Q10, model.Q1r],       'lr': cfg.LR_Q},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=cfg.SCHED_FACTOR, patience=cfg.SCHED_PATIENCE, threshold=1e-3
    )
    best_loss, best_state, no_imp = float('inf'), None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad() #zero the gradients before each step, apparently they accumulate by default in pytorch and we dont want that
        loss = loss_fn(model) #compute the loss using the provided loss function (average on the balls and take the max in each ball).
        loss.backward() #backpropagate to compute gradients for all parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #clip gradients to prevent exploding gradients, which can happen with the projection layer and lead to numerical instability.
        opt.step() #update the parameters using the optimizer
        sched.step(loss.item()) #step the learning rate scheduler based on the current loss.
        v = loss.item() # get the scalar loss value as a Python float for logging and comparison
        if v < best_loss - 1e-4:
            best_loss = v
            best_state = {k: w.clone() for k, w in model.state_dict().items() if not w.is_sparse}
            no_imp = 0
        else:
            no_imp += 1
        if (ep % 50 == 0 or ep == 1) and log:
            q10v = torch.nn.functional.softplus(model.Q10).item()
            q1rv = torch.nn.functional.softplus(model.Q1r).detach().numpy()
            lr_q = opt.param_groups[1]['lr']
            print(f"  [{label}] ep {ep:4d}/{epochs} | loss={v:.4f} | "
                  f"Q10={q10v:.3f} | Q1r={np.round(q1rv,3)} | "
                  f"lr_Q={lr_q:.5f} | no_imp={no_imp}/{cfg.EARLY_STOP} | "
                  f"t={time.time()-t0:.0f}s")
        if no_imp >= cfg.EARLY_STOP:
            if log:
                print(f"  [{label}] Early stop ep={ep}, best={best_loss:.4f}")
            break
    model.load_state_dict(best_state, strict=False)
    return best_loss, best_state

# ============================================================
# ADVERSARIAL PROBLEM (AP)
# ============================================================
def load_weights_from_state(state_dict):
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
#we take the b1 to be the original bias plus the contributions from the Q1r and Q10 terms,
#  since in the AP we will feed in the xi1 values but not the Q1r/Q10 (they are fixed in the AP).
def solve_ap_one(weights, xi1_hat, xi2_hat, idx):
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']
    W3, b3 = weights['W3'], weights['b3']
    Q10_val, Q1r_val = weights['Q10'], weights['Q1r']

    env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.start()
    m   = gp.Model(f"AP_{idx}", env=env)

    xi1 = m.addVars(cfg.R, lb=np.maximum(0, xi1_hat - cfg.RHO_1), ub=xi1_hat + cfg.RHO_1) #uncertainty set for xi1 is a box around xi1_hat with radius RHO_1, truncated at 0 since xi1 cannot be negative.
    xi2 = m.addVars(cfg.R, lb=np.maximum(0, xi2_hat - cfg.RHO_2), ub=xi2_hat + cfg.RHO_2) #uncertainty set for xi2 is a box around xi2_hat with radius RHO_2, truncated at 0 since xi2 cannot be negative.

    def relu_layer(inp, W, b, tag):
        n = W.shape[0]
        xp = m.addVars(n, lb=0, name=f"xp{tag}") #positive part of the pre-activation. which we return
        xm = m.addVars(n, lb=0, name=f"xm{tag}") #negative part of the pre-activation.
        z  = m.addVars(n, vtype=GRB.BINARY, name=f"z{tag}") #binary variables to model the ReLU activation: z[j] = 1 if the pre-activation is <= 0 (i.e., xm[j] > 0), z[j] = 0 if the pre-activation is > 0 (i.e., xp[j] > 0)
        for j in range(n):
            expr = gp.quicksum(W[j,k] * inp[k] for k in range(W.shape[1])) + b[j] #the pre-activation for neuron j in this layer.
            m.addConstr(expr == xp[j] - xm[j]) #the pre-activation is the difference between the positive and negative parts. (if positive, then xp[j] > 0 and xm[j] = 0; if negative, then xm[j] > 0 and xp[j] = 0 )
            m.addGenConstrIndicator(z[j], 1, xp[j] == 0) # if z[j] = 1, then xp[j] must be 0 (pre-activation <= 0)
            m.addGenConstrIndicator(z[j], 0, xm[j] == 0) # if z[j] = 0, then xm[j] must be 0 (pre-activation > 0)
        return xp

    l1 = relu_layer(xi1, W1, b1, "1")
    l2 = relu_layer(l1,  W2, b2, "2")

    R = cfg.R
    zs = m.addVars(R, lb=-GRB.INFINITY, name="zs")
    for j in range(R):
        m.addConstr(zs[j] == gp.quicksum(W3[j,k]*l2[k] for k in range(W3.shape[1])) + b3[j])
    #no relu on the last layer. 
    #now we deal we the KKT conditions for the projection layer. the projection layer solves min_q2 ||q2 - zs||^2 s.t. sum(q2) <= Q10, q2 >= 0.
    q2  = m.addVars(R, lb=0)
    lam = m.addVar(lb=0)
    mu  = m.addVars(R, lb=0)
    for r in range(R):
        m.addConstr(2.0*q2[r] - 2*zs[r] + lam - mu[r] == 0) # stationarity
    m.addConstr(gp.quicksum(q2[r] for r in range(R)) <= Q10_val) # primal feasibility
    u_lam = m.addVar(vtype=GRB.BINARY)
    m.addGenConstrIndicator(u_lam, 1, lam == 0) # if u_lam = 1, then lam must be 0 (complementary slackness)
    m.addGenConstrIndicator(u_lam, 0, Q10_val - gp.quicksum(q2[r] for r in range(R)) == 0) # if u_lam = 0, then the constraint is tight (complementary slackness)
    u_mu = m.addVars(R, vtype=GRB.BINARY)
    for r in range(R):
        m.addGenConstrIndicator(u_mu[r], 1, mu[r]  == 0) # if u_mu[r] = 1, then mu[r] must be 0 (complementary slackness)
        m.addGenConstrIndicator(u_mu[r], 0, q2[r]  == 0) # if u_mu[r] = 0, then q2[r] must be 0 (complementary slackness)

    v = m.addVars(R, lb=-GRB.INFINITY) # auxiliary variables to model the max of the 3 cost terms for each r,
    #same as in the SAA LP. we will minimize this in the objective, 
    # we add constraints to ensure it is greater than or equal to each of the 3 terms.
    for r in range(R):
        g1 = m.addVar(lb=-GRB.INFINITY)
        g2 = m.addVar(lb=-GRB.INFINITY)
        g3 = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(g1 == cfg.b*(xi1[r]+xi2[r]-Q1r_val[r]-q2[r]) - cfg.h*q2[r])
        m.addConstr(g2 == cfg.h*(Q1r_val[r]-xi1[r]-xi2[r]))
        m.addConstr(g3 == cfg.b*(xi1[r]-Q1r_val[r]) - cfg.h*xi2[r])
        m.addGenConstrMax(v[r], [g1,g2,g3]) # v[r] is the maximum of the 3 terms for this r.

    stage1 = cfg.c*(Q10_val + np.sum(Q1r_val)) + cfg.h*Q10_val
    m.setObjective(stage1 + gp.quicksum(v[r] for r in range(R)), GRB.MAXIMIZE)
    m.optimize()

    if m.status == GRB.OPTIMAL:
        return (np.array([xi1[r].X for r in range(R)]),
                np.array([xi2[r].X for r in range(R)]),
                m.ObjVal)
    return None, None, None

def run_ap(state_dict, seed, ccg_iter, xi1_np, xi2_np):
    weights = load_weights_from_state(state_dict)
    print(f"  AP  Q10={weights['Q10']:.4f}, Q1r={np.round(weights['Q1r'],3)}")
    scenarios = []
    t0 = time.time()
    for i in range(cfg.N_SAMPLES):
        wx1, wx2, mc = solve_ap_one(weights, xi1_np[i], xi2_np[i], i)
        if wx1 is not None:
            scenarios.append({"ball_id": i, "xi1_hat": xi1_np[i], "xi2_hat": xi2_np[i],
                               "worst_xi1": wx1, "worst_xi2": wx2, "max_cost": mc})
    costs = [s["max_cost"] for s in scenarios]
    print(f"  AP done: n={len(scenarios)}, mean={np.mean(costs):.4f}, "
          f"max={np.max(costs):.4f}, t={time.time()-t0:.0f}s")
    # Save
    path = ap_json(seed, ccg_iter)
    with open(path, "w") as f:
        json.dump([{k: to_serializable(v) for k,v in s.items()} for s in scenarios], f)
    return np.mean(costs), path

# ============================================================
# MASTER PROBLEM (MP) — robust retraining
# ============================================================
def run_mp(start_state, scenario_paths, label, xi1_np, xi2_np, xi1_t, xi2_t):
    # Load all adversarial scenarios
    adv_xi1_by_ball, adv_xi2_by_ball = {}, {}
    for path in scenario_paths:
        with open(path) as f:
            for s in json.load(f):
                bid = s["ball_id"]
                adv_xi1_by_ball.setdefault(bid, []).append(s["worst_xi1"])
                adv_xi2_by_ball.setdefault(bid, []).append(s["worst_xi2"])
    #we wish to minimize the robust loss,so we need maximum
    #  cost across all scenarios (original + adversarial)
    #  for each sample.
    max_cols = max(len(v) for v in adv_xi1_by_ball.values())
    adv_xi1_cols = [xi1_np.copy() for _ in range(max_cols)]
    adv_xi2_cols = [xi2_np.copy() for _ in range(max_cols)]
    for bid in adv_xi1_by_ball:
        for k, (ax1, ax2) in enumerate(zip(adv_xi1_by_ball[bid], adv_xi2_by_ball[bid])):
            adv_xi1_cols[k][bid] = ax1
            adv_xi2_cols[k][bid] = ax2
    #make tensors so we can run in pytorch.
    adv_t1 = [torch.tensor(c, dtype=torch.float32) for c in adv_xi1_cols]
    adv_t2 = [torch.tensor(c, dtype=torch.float32) for c in adv_xi2_cols]

    model = Model()
    model.load_state_dict(start_state, strict=False)

    def loss_fn(m):
        cols = [m.per_sample_cost(xi1_t, xi2_t)] # cost on the original data
        for t1, t2 in zip(adv_t1, adv_t2): # costs on the adversarial scenarios (one column per scenario)
            cols.append(m.per_sample_cost(t1, t2))
        return torch.stack(cols, dim=1).max(dim=1).values.mean() # robust loss: for each sample,
        # take the max cost across all scenarios (original + adversarial), then average over samples

    best_loss, best_state = train(model, loss_fn, label)
    return best_loss, best_state

# ============================================================
# LR_NET SEARCH (once on LR_SEARCH_SEED, winner used for all seeds)
# ============================================================
lr_search_xi1, lr_search_xi2 = generate_data(cfg.LR_SEARCH_SEED)
lr_search_t1 = torch.tensor(lr_search_xi1, dtype=torch.float32)
lr_search_t2 = torch.tensor(lr_search_xi2, dtype=torch.float32)
_, _, lr_search_saa_obj = solve_saa(lr_search_xi1, lr_search_xi2)
lr_search_saa_Q10, lr_search_saa_Q1r, _ = solve_saa(lr_search_xi1, lr_search_xi2)

print(f"\n{'='*70}")
print(f"LR_NET SEARCH on data seed {cfg.LR_SEARCH_SEED} (SAA={lr_search_saa_obj:.4f})")
print(f"Grid: {cfg.GRID_LR_NET}")
print(f"{'='*70}")
best_lr_net = cfg.GRID_LR_NET[0]  # fallback (overwritten by search)
best_lr_loss = float('inf')
#search for best LR. 
for lr_net in cfg.GRID_LR_NET:
    gpath = grid_ckpt(lr_net)
    #if we already ran this LR_NET, load the results and skip the training.
    if os.path.exists(gpath):
        g = torch.load(gpath, map_location="cpu")
        g_loss = g["best_loss"]
        gap = (g_loss - lr_search_saa_obj) / abs(lr_search_saa_obj) * 100
        print(f"  [SKIP] lr_net={lr_net} → loss={g_loss:.4f}  gap={gap:+.2f}%")
    else:
        #train model.
        model = make_model_saa_init(lr_search_saa_Q10, lr_search_saa_Q1r)
        cfg.LR_NET = lr_net
        '''
        we train to minimize the average cost on the original SAA dataset.
        we need to sent to train the loss function - in this case we 
        dont have an adversarial points ao we just compute the cost on the original data. 
        '''
        g_loss, g_state = train(
            model,
            lambda m: m.per_sample_cost(lr_search_t1, lr_search_t2).mean(),
            f"lrsearch_lr{lr_net}"
        )
        #saving the results. 
        torch.save({"model_state_dict": g_state, "best_loss": float(g_loss),
                    "lr_net": lr_net}, gpath)
        gap = (g_loss - lr_search_saa_obj) / abs(lr_search_saa_obj) * 100
        print(f"  lr_net={lr_net} → loss={g_loss:.4f}  gap={gap:+.2f}%")
    g_loss = torch.load(gpath, map_location="cpu")["best_loss"]
    if g_loss < best_lr_loss:
        best_lr_loss = g_loss
        best_lr_net  = lr_net

cfg.LR_NET = best_lr_net
print(f"\nBest LR_NET = {best_lr_net}  (loss={best_lr_loss:.4f})  → using for all seeds\n")
print("LR search complete. Proceeding to full 10-seed pipeline...")


all_seed_results = {}

for seed in cfg.SEEDS:
    print(f"\n{'#'*70}")
    print(f"# SEED {seed}")
    print(f"{'#'*70}")
    t_seed = time.time()

    # Generate dataset for this seed
    xi1_np, xi2_np = generate_data(seed)
    xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
    xi2_t = torch.tensor(xi2_np, dtype=torch.float32)
    print(f"  Data: xi1 mean={xi1_np.mean():.3f}, xi2 mean={xi2_np.mean():.3f}")

    # Solve SAA for this seed's dataset
    t_saa = time.time()
    saa_Q10, saa_Q1r, saa_obj = solve_saa(xi1_np, xi2_np)
    print(f"  SAA target loss: {saa_obj:.4f}  Q10={saa_Q10:.4f}  "
          f"Q1r={np.round(saa_Q1r, 4)}  (t={time.time()-t_saa:.1f}s)")

    summ_path = summary_json(seed)
    if os.path.exists(summ_path):
        with open(summ_path) as f:
            stored = json.load(f)
        if stored.get("converged", False):
            print(f"  [SKIP] seed {seed} already completed (converged at iter {stored['ccg_iters']})")
            all_seed_results[seed] = stored
            continue

    # ---- NOMINAL TRAINING (uses best lr_net from search) ----
    nom_ckpt = ckpt(seed, "nominal")
    if os.path.exists(nom_ckpt):
        saved = torch.load(nom_ckpt, map_location="cpu")
        nom_state = saved["model_state_dict"]
        nom_loss  = saved["best_loss"]
        best_lr   = saved.get("lr_net", "?")
        gap_pct   = (nom_loss - saa_obj) / abs(saa_obj) * 100
        print(f"  [SKIP nominal] loss={nom_loss:.4f}  gap={gap_pct:+.2f}%  (lr_net={best_lr})")
    else:
        #if didnt train nominal yet:
        print(f"\n  --- NOMINAL TRAINING  data_seed={seed}, lr_net={cfg.LR_NET}, lr_Q={cfg.LR_Q} ---")
        model = make_model_saa_init(saa_Q10, saa_Q1r)
        nom_loss, nom_state = train(model,
                                    lambda m: m.per_sample_cost(xi1_t, xi2_t).mean(),
                                    f"nom_s{seed}")
        torch.save({"model_state_dict": nom_state, "best_loss": float(nom_loss),
                    "lr_net": cfg.LR_NET, "seed": seed, "lr_Q": cfg.LR_Q,
                    "n_samples": cfg.N_SAMPLES, "saa_obj": float(saa_obj),
                    "saa_Q10": float(saa_Q10), "saa_Q1r": saa_Q1r.tolist()},
                   nom_ckpt)
        gap_pct = (nom_loss - saa_obj) / abs(saa_obj) * 100
        best_lr = cfg.LR_NET
        print(f"  Nominal done: loss={nom_loss:.4f}  SAA target={saa_obj:.4f}  gap={gap_pct:+.2f}%  saved {nom_ckpt}")

    # ---- CCG ITERATIONS ----
    cur_state = nom_state
    scenario_paths = []
    ccg_results = []
    mp_loss = nom_loss
    converged = False

    for it in range(1, cfg.CCG_MAX_ITER + 1):
        print(f"\n  === CCG iter {it} ===")

        # AP step
        ap_path = ap_json(seed, it)
        if os.path.exists(ap_path):
            with open(ap_path) as f: sc = json.load(f)
            ap_mean = float(np.mean([s["max_cost"] for s in sc]))
            print(f"  [SKIP AP iter {it}] ap_mean={ap_mean:.4f}")
        else:
            ap_mean, ap_path = run_ap(cur_state, seed, it, xi1_np, xi2_np)

        scenario_paths.append(ap_json(seed, it))
        delta = ap_mean - mp_loss
        delta_pct = delta / mp_loss * 100
        print(f"  Δ = AP_mean({ap_mean:.4f}) - MP_loss({mp_loss:.4f}) = {delta:.4f}  ({delta_pct:.2f}%)")

        if it > 1 and delta_pct < cfg.CCG_CONV_THRESH_PCT:
            print(f"  *** CONVERGED at iter {it}: Δ={delta_pct:.2f}% < {cfg.CCG_CONV_THRESH_PCT}% ***")
            converged = True
            ccg_results.append({"iter": it, "ap_mean": ap_mean, "mp_loss": mp_loss, "delta": delta, "delta_pct": delta_pct, "converged": True})
            break

        # MP step
        mp_ckpt = ckpt(seed, f"mp_iter{it}")
        if os.path.exists(mp_ckpt):
            mp_loss  = torch.load(mp_ckpt, map_location="cpu")["best_loss"]
            cur_state = torch.load(mp_ckpt, map_location="cpu")["model_state_dict"]
            print(f"  [SKIP MP iter {it}] mp_loss={mp_loss:.4f}")
        else:
            print(f"  --- MP iter {it} (seed={seed}, {len(scenario_paths)} scenario file(s)) ---")
            mp_loss, cur_state = run_mp(cur_state, scenario_paths, f"mp_s{seed}_i{it}", xi1_np, xi2_np, xi1_t, xi2_t)
            torch.save({"model_state_dict": cur_state, "best_loss": float(mp_loss),
                        "seed": seed, "ccg_iter": it},
                       mp_ckpt)
            print(f"  MP iter {it} done: loss={mp_loss:.4f}, saved {mp_ckpt}")

        ccg_results.append({"iter": it, "ap_mean": ap_mean, "mp_loss": mp_loss,
                             "delta": delta, "delta_pct": delta_pct, "converged": False})

    # Final Q10, Q1r
    final_model = Model()
    final_model.load_state_dict(cur_state, strict=False)
    final_Q10 = torch.nn.functional.softplus(final_model.Q10).item()
    final_Q1r = torch.nn.functional.softplus(final_model.Q1r).detach().numpy().tolist()

    seed_result = {
        "seed":        seed,
        "saa_obj":     float(saa_obj),
        "saa_Q10":     float(saa_Q10),
        "saa_Q1r":     saa_Q1r.tolist(),
        "lr_net_used": best_lr,
        "nominal_loss": float(nom_loss),
        "ccg_iters":   len(ccg_results),
        "final_mp_loss": float(mp_loss),
        "final_Q10":   final_Q10,
        "final_Q1r":   final_Q1r,
        "converged":   converged,
        "ccg_trajectory": ccg_results,
        "total_time_s": time.time() - t_seed,
    }
    with open(summ_path, "w") as f:
        json.dump(seed_result, f, indent=2)

    all_seed_results[seed] = seed_result
    print(f"\n  SEED {seed} DONE: nominal={nom_loss:.4f}, "
          f"final_mp={mp_loss:.4f}, Q10={final_Q10:.3f}, "
          f"Q1r={np.round(final_Q1r,3)}, converged={converged}, "
          f"t={time.time()-t_seed:.0f}s")

# ============================================================
# STATISTICS ACROSS ALL SEEDS
# ============================================================
print(f"\n\n{'='*70}")
print("STATISTICS ACROSS ALL SEEDS")
print(f"lr_Q={cfg.LR_Q}, RHO_1={cfg.RHO_1}, RHO_2={cfg.RHO_2}, N={cfg.N_SAMPLES}")
print(f"Convergence threshold: {cfg.CCG_CONV_THRESH_PCT}% (percentage of MP loss)")
print(f"{'='*70}")

nom_losses  = [all_seed_results[s]["nominal_loss"]  for s in cfg.SEEDS if s in all_seed_results]
mp_losses   = [all_seed_results[s]["final_mp_loss"] for s in cfg.SEEDS if s in all_seed_results]
q10_vals    = [all_seed_results[s]["final_Q10"]     for s in cfg.SEEDS if s in all_seed_results]
ccg_iters   = [all_seed_results[s]["ccg_iters"]     for s in cfg.SEEDS if s in all_seed_results]
conv_count  = sum(all_seed_results[s]["converged"]  for s in cfg.SEEDS if s in all_seed_results)

print(f"\nNominal loss:   mean={np.mean(nom_losses):.4f}  std={np.std(nom_losses):.4f}  "
      f"min={np.min(nom_losses):.4f}  max={np.max(nom_losses):.4f}")
print(f"Final MP loss:  mean={np.mean(mp_losses):.4f}  std={np.std(mp_losses):.4f}  "
      f"min={np.min(mp_losses):.4f}  max={np.max(mp_losses):.4f}")
print(f"Final Q10:      mean={np.mean(q10_vals):.4f}  std={np.std(q10_vals):.4f}  "
      f"min={np.min(q10_vals):.4f}  max={np.max(q10_vals):.4f}")
print(f"CCG iters:      mean={np.mean(ccg_iters):.1f}  min={np.min(ccg_iters)}  max={np.max(ccg_iters)}")
print(f"Converged:      {conv_count}/{len(cfg.SEEDS)}")

print(f"\n{'Seed':>5} | {'Nom.Loss':>9} | {'MP.Loss':>9} | {'Q10':>7} | {'Q1r':>22} | {'CCG iters':>9} | Conv")
print("-" * 85)
for s in cfg.SEEDS:
    if s not in all_seed_results:
        print(f"  {s:3d} | {'—':>9}")
        continue
    r = all_seed_results[s]
    print(f"  {s:3d} | {r['nominal_loss']:9.4f} | {r['final_mp_loss']:9.4f} | "
          f"{r['final_Q10']:7.3f} | {str(np.round(r['final_Q1r'],3)):>22} | "
          f"{r['ccg_iters']:9d} | {'YES' if r['converged'] else 'NO'}")

print(f"\n[Done] run_all_net16.py complete.")
