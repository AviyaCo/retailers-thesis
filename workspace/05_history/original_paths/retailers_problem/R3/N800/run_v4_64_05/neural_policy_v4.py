"""
neural_policy_v4.py — Predict-and-Project Neural Policy (clean start)
=====================================================================
Changes vs v2/v3:
  - N_SAMPLES = 800
  - Scheduler patience = 100
  - Early stopping after 200 epochs with no improvement
  - Per-epoch timing printed
  - No checkpoint loading — trains from scratch
  - SAA LP comparison at the end (Gurobi)
"""

import time
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

print("=" * 60)
print("NEURAL POLICY V4 — clean start, N=800")
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
    N_SAMPLES = 800          # increased from 200
    EPOCHS    = 2000

    SCS_EPS       = 1e-5
    SCS_MAX_ITERS = 100000

    EARLY_STOP_PATIENCE = 200  # stop if no improvement for 200 epochs

    CHECKPOINT_PATH = "/data/home/aviyac/retailers_problem/R3/N800/run_v4_64_05/thesis_checkpoint_v4.pth"

cfg = PaperConfig()

# ==========================================
# DATA GENERATION
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
            nn.Linear(config.R * 2 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, config.R)
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
# TRAINING SETUP
# ==========================================
model = PredictAndProjectModel(cfg).to(device)

optimizer = torch.optim.Adam([
    {'params': model.fc_layers.parameters(), 'lr': 0.01},
    {'params': [model.Q10, model.Q1r],       'lr': 0.05},
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=100, threshold=1e-3
)

print(f"\nSCS: eps={cfg.SCS_EPS}, max_iters={cfg.SCS_MAX_ITERS}")
print(f"Scheduler patience: 100 | Early stop patience: {cfg.EARLY_STOP_PATIENCE}")
print(f"Training from scratch for up to {cfg.EPOCHS} epochs...\n")

# ==========================================
# TRAINING LOOP  (with timing + early stopping)
# ==========================================
best_loss = float('inf')
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

    # --- early stopping tracking ---
    if current_loss < best_loss - 1e-4:
        best_loss = current_loss
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
print(f"\nTraining completed in {total_time:.1f}s ({epoch} epochs)")

# ==========================================
# SAVE CHECKPOINT
# ==========================================
print("\nSaving checkpoint...")
torch.save({
    'epoch':                epoch,
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_loss':            best_loss,
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
print("FINAL TRAINED MODEL PARAMETERS (v4 — N=800)")
print("=" * 60)
print(f"Final Loss:                      {final_loss.item():.4f}")
print(f"Learned Warehouse Capacity (Q10): {final_Q10:.4f}")
print(f"Learned Retailer Inventory (Q1r): {np.round(final_Q1r, 4)}")
print(f"Training samples (N):             {cfg.N_SAMPLES}")
print(f"Epochs completed:                 {epoch}")
print(f"Total training time:              {total_time:.1f}s")
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
