"""
neural_policy_N50.py — Predict-and-Project Neural Policy (LR grid search)
==========================================================================
  - N_SAMPLES = 50, hidden = 32
  - Grid over lr_Q values x seeds, keeps the best
  - SAA LP comparison at the end (Gurobi)
"""

import time
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import os
import copy
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

print("=" * 60)
print("NEURAL POLICY N50 — LR grid search, N=50, hidden=32")
print("=" * 60)

device = torch.device("cpu")
print(f"Device: {device}")

# ==========================================
# CONFIGURATION
# ==========================================
class PaperConfig:
    R         = 3
    c         = 0.25
    h         = 0.05
    b         = 0.5
    N_SAMPLES = 50
    EPOCHS    = 3000

    SCS_EPS       = 1e-5
    SCS_MAX_ITERS = 100000

    EARLY_STOP_PATIENCE = 300

    CHECKPOINT_PATH = "/data/home/aviyac/retailers_problem/R3/N50/code_check/run_N50_rho03/thesis_checkpoint_N50.pth"

cfg = PaperConfig()

SEEDS = [42, 123, 7]
LR_Q_VALUES = [0.1, 0.2, 0.5]
LR_NET = 0.01

# ==========================================
# DATA GENERATION  (seed=42 for data, always the same)
# ==========================================
def generate_data(N, R):
    np.random.seed(42)
    xi1 = np.random.normal(loc=6.0, scale=2.5, size=(N, R))
    xi1 = np.maximum(xi1, 0.0)
    mean_xi2 = 2 * (6.0 - xi1)**2
    xi2 = np.random.normal(loc=mean_xi2, scale=2.5)
    xi2 = np.maximum(xi2, 0.0)
    return xi1, xi2

print(f"Generating data: N={cfg.N_SAMPLES}, R={cfg.R}")
xi1_np, xi2_np = generate_data(cfg.N_SAMPLES, cfg.R)

xi1_torch = torch.tensor(xi1_np, dtype=torch.float32).to(device)
xi2_torch = torch.tensor(xi2_np, dtype=torch.float32).to(device)

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
class PredictAndProjectModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.fc_layers = nn.Sequential(
            nn.Linear(config.R * 2 + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, config.R)
        )

        self.Q10 = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.Q1r = nn.Parameter(torch.zeros(config.R, dtype=torch.float32))

        x       = cp.Variable(config.R)
        z_param = cp.Parameter(config.R)
        Q10_p   = cp.Parameter(1)
        cons    = [cp.sum(x) <= Q10_p + 1e-6, x >= 0]
        self.projection_layer = CvxpyLayer(
            cp.Problem(cp.Minimize(cp.sum_squares(x - z_param)), cons),
            parameters=[z_param, Q10_p],
            variables=[x]
        )

    def forward(self, xi1, xi2):
        batch_size = xi1.shape[0]

        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)

        q1r_input = q1r.unsqueeze(0).expand(batch_size, -1)
        q10_input = q10.expand(batch_size, 1)
        net_input = torch.cat([xi1, q1r_input, q10_input], dim=1)
        z_suggestion = self.fc_layers(net_input)

        q2_shipments, = self.projection_layer(
            z_suggestion,
            q10.expand(batch_size, 1),
            solver_args={
                "solve_method": "SCS",
                "eps":           self.cfg.SCS_EPS,
                "max_iters":     self.cfg.SCS_MAX_ITERS,
            }
        )

        cost_s1 = self.cfg.c * (q10 + torch.sum(q1r))
        term1   = self.cfg.b * (xi2 + xi1 - q2_shipments - q1r) - self.cfg.h * q2_shipments
        term2   = self.cfg.h * (q1r - xi1 - xi2)
        term3   = self.cfg.b * (xi1 - q1r) - self.cfg.h * xi2
        v_nodes = torch.maximum(torch.maximum(term1, term2), term3)
        total_loss = cost_s1 + self.cfg.h * q10 + torch.mean(torch.sum(v_nodes, dim=1))

        return total_loss, q2_shipments


# ==========================================
# TRAINING FUNCTION (one seed + lr_Q)
# ==========================================
def train_one_seed(seed, lr_Q, cfg, xi1_torch, xi2_torch):
    print(f"\n{'='*60}")
    print(f"  SEED {seed}  |  lr_Q={lr_Q}  |  lr_net={LR_NET}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    model = PredictAndProjectModel(cfg).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.fc_layers.parameters(), 'lr': LR_NET},
        {'params': [model.Q10, model.Q1r],       'lr': lr_Q},
    ])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=150, threshold=1e-3
    )

    best_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    total_start = time.time()

    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start = time.time()

        optimizer.zero_grad()
        loss, _ = model(xi1_torch, xi2_torch)
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        epoch_time = time.time() - epoch_start
        current_loss = loss.item()

        if current_loss < best_loss - 1e-4:
            best_loss = current_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()
                          if not v.is_sparse}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_net    = optimizer.param_groups[0]['lr']
            lr_params = optimizer.param_groups[1]['lr']
            q10_v     = torch.nn.functional.softplus(model.Q10).item()
            q1r_v     = torch.nn.functional.softplus(model.Q1r).detach().numpy()
            elapsed   = time.time() - total_start
            print(f"Epoch {epoch:4d}/{cfg.EPOCHS} | Loss: {current_loss:.4f} | "
                  f"Q10: {q10_v:.3f} | Q1r: {np.round(q1r_v, 3)} | "
                  f"lr_net: {lr_net:.5f} | lr_Q: {lr_params:.5f} | "
                  f"epoch: {epoch_time:.2f}s | total: {elapsed:.1f}s | "
                  f"no_improv: {epochs_no_improve}/{cfg.EARLY_STOP_PATIENCE}")

        if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
            print(f"\n>> Early stopping at epoch {epoch}: "
                  f"no improvement for {cfg.EARLY_STOP_PATIENCE} epochs. "
                  f"Best loss: {best_loss:.4f}")
            break

    total_time = time.time() - total_start

    # Reload best weights
    model.load_state_dict(best_state, strict=False)
    with torch.no_grad():
        final_loss, _ = model(xi1_torch, xi2_torch)

    q10_v = torch.nn.functional.softplus(model.Q10).item()
    q1r_v = torch.nn.functional.softplus(model.Q1r).detach().numpy()
    print(f"\nSeed {seed}, lr_Q={lr_Q} best: Loss={final_loss.item():.4f}, Q10={q10_v:.3f}, "
          f"Q1r={np.round(q1r_v, 3)}, epochs={epoch}, time={total_time:.1f}s")

    return best_loss, best_state, epoch, total_time


# ==========================================
# RUN ALL (seed, lr_Q) COMBINATIONS
# ==========================================
configs = [(s, lq) for lq in LR_Q_VALUES for s in SEEDS]
print(f"\nRunning {len(configs)} configs: {len(SEEDS)} seeds x {len(LR_Q_VALUES)} lr_Q values")
print(f"lr_Q values: {LR_Q_VALUES}")
print(f"Seeds: {SEEDS}")
print(f"SCS: eps={cfg.SCS_EPS}, max_iters={cfg.SCS_MAX_ITERS}")
print(f"Scheduler patience: 150 | Early stop patience: {cfg.EARLY_STOP_PATIENCE}")

global_best_loss = float('inf')
global_best_state = None
global_best_config = None
all_results = []
grand_start = time.time()

for seed, lr_Q in configs:
    loss, state, ep, t = train_one_seed(seed, lr_Q, cfg, xi1_torch, xi2_torch)
    all_results.append((seed, lr_Q, loss, ep, t))
    if loss < global_best_loss:
        global_best_loss = loss
        global_best_state = state
        global_best_config = (seed, lr_Q)

grand_time = time.time() - grand_start

# ==========================================
# SUMMARY
# ==========================================
print("\n" + "=" * 60)
print("LR GRID SEARCH SUMMARY")
print("=" * 60)
for seed, lr_Q, loss, ep, t in all_results:
    marker = " <-- BEST" if (seed, lr_Q) == global_best_config else ""
    print(f"  Seed {seed:5d}, lr_Q={lr_Q:.3f}: Loss={loss:.4f}, epochs={ep}, time={t:.1f}s{marker}")
print(f"\nBest config: seed={global_best_config[0]}, lr_Q={global_best_config[1]} (loss={global_best_loss:.4f})")
print(f"Total time all configs: {grand_time:.1f}s")

# ==========================================
# LOAD BEST AND SAVE CHECKPOINT
# ==========================================
model = PredictAndProjectModel(cfg).to(device)
model.load_state_dict(global_best_state, strict=False)

print("\nSaving checkpoint (best config)...")
torch.save({
    'epoch':                None,
    'model_state_dict':     global_best_state,
    'best_loss':            global_best_loss,
    'best_seed':            global_best_config[0],
    'best_lr_Q':            global_best_config[1],
    'scs_eps':              cfg.SCS_EPS,
    'scs_max_iters':        cfg.SCS_MAX_ITERS,
    'n_samples':            cfg.N_SAMPLES,
}, cfg.CHECKPOINT_PATH)
print(f"Saved to {cfg.CHECKPOINT_PATH}")

# ==========================================
# FINAL RESULTS
# ==========================================
with torch.no_grad():
    final_loss, _ = model(xi1_torch, xi2_torch)

final_Q10 = torch.nn.functional.softplus(model.Q10).item()
final_Q1r = torch.nn.functional.softplus(model.Q1r).detach().cpu().numpy()

print("\n" + "=" * 60)
print("FINAL TRAINED MODEL PARAMETERS (N=50, hidden=32)")
print("=" * 60)
print(f"Final Loss:                      {final_loss.item():.4f}")
print(f"Learned Warehouse Capacity (Q10): {final_Q10:.4f}")
print(f"Learned Retailer Inventory (Q1r): {np.round(final_Q1r, 4)}")
print(f"Training samples (N):             {cfg.N_SAMPLES}")
print(f"Best config:                      seed={global_best_config[0]}, lr_Q={global_best_config[1]}")
print(f"Total training time (all configs): {grand_time:.1f}s")
print("=" * 60)

# ==========================================
# SAA LP COMPARISON  (Gurobi)
# ==========================================
print("\n" + "=" * 60)
print(f"SAA LP COMPARISON (N={cfg.N_SAMPLES}, same training data, Gurobi)")
print("=" * 60)

t_saa = time.time()
N = cfg.N_SAMPLES
R = cfg.R

env_saa = gp.Env(empty=True); env_saa.setParam("OutputFlag", 0); env_saa.start()
m_saa = gp.Model(env=env_saa)

Q10_v = m_saa.addVar(lb=0.0, name="Q10")
Q1r_v = m_saa.addVars(R, lb=0.0, name="Q1r")
Q2_v  = m_saa.addVars(N, R, lb=0.0, name="Q2")
vv    = m_saa.addVars(N, R, lb=-GRB.INFINITY, name="v")

for i in range(N):
    m_saa.addConstr(gp.quicksum(Q2_v[i, r] for r in range(R)) <= Q10_v)
    for r in range(R):
        m_saa.addConstr(vv[i,r] >= cfg.b*(xi1_np[i,r]+xi2_np[i,r]-Q2_v[i,r]-Q1r_v[r]) - cfg.h*Q2_v[i,r])
        m_saa.addConstr(vv[i,r] >= cfg.h*(Q1r_v[r]-xi1_np[i,r]-xi2_np[i,r]))
        m_saa.addConstr(vv[i,r] >= cfg.b*(xi1_np[i,r]-Q1r_v[r]) - cfg.h*xi2_np[i,r])

obj_saa = (cfg.c * (Q10_v + gp.quicksum(Q1r_v[r] for r in range(R))) + cfg.h * Q10_v
           + (1.0/N) * gp.quicksum(vv[i,r] for i in range(N) for r in range(R)))
m_saa.setObjective(obj_saa, GRB.MINIMIZE)
m_saa.optimize()

saa_Q10 = Q10_v.X
saa_Q1r = np.array([Q1r_v[r].X for r in range(R)])
saa_obj = m_saa.ObjVal
gap_pct = (final_loss.item() - saa_obj) / abs(saa_obj) * 100

print(f"SAA Objective (in-sample LB):    {saa_obj:.4f}")
print(f"SAA Q10:                         {saa_Q10:.4f}")
print(f"SAA Q1r:                         {np.round(saa_Q1r, 4)}")
print(f"Neural Loss:                     {final_loss.item():.4f}")
print(f"Neural Q10:                      {final_Q10:.4f}")
print(f"Neural Q1r:                      {np.round(final_Q1r, 4)}")
print(f"Gap (Neural vs SAA LB):          +{gap_pct:.2f}%")
print(f"SAA solve time:                  {time.time()-t_saa:.1f}s")
print("=" * 60)
