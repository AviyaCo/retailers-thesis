"""
master_problem_32_05_iter2.py — CCG Master Problem Iteration 2
============================================================
Retrains using adversarial scenarios from iter 1 AND 2.
    CHECKPOINT_IN  = thesis_checkpoint_mp_iter1_32_05.pth
    CHECKPOINT_OUT = thesis_checkpoint_mp_iter2_32_05.pth
    SCENARIOS_FILES = [iter1, iter2]
"""

import time
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import json
import os

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

# ==========================================
# CONFIGURATION
# ==========================================
class MPConfig:
    R         = 3
    c         = 0.25
    h         = 0.05
    b         = 0.5
    N_SAMPLES = 800
    EPOCHS    = 2500

    SCS_EPS       = 1e-5
    SCS_MAX_ITERS = 100000

    EARLY_STOP_PATIENCE = 100

    # --- CCG iteration settings ---
    CHECKPOINT_IN  = "/data/home/aviyac/retailers_problem/R3/N800/code_check/run_32_05/thesis_checkpoint_mp_iter1_32_05.pth"
    CHECKPOINT_OUT = "/data/home/aviyac/retailers_problem/R3/N800/code_check/run_32_05/thesis_checkpoint_mp_iter2_32_05.pth"

    SCENARIOS_FILES = [
        "/data/home/aviyac/retailers_problem/R3/N800/code_check/run_32_05/adversarial_scenarios_iter1_32_05.json",
        "/data/home/aviyac/retailers_problem/R3/N800/code_check/run_32_05/adversarial_scenarios_iter2_32_05.json",
    ]

    RHO_1 = 0.5
    RHO_2 = 0.5

cfg = MPConfig()

# ==========================================
# DATA GENERATION (same seed as neural_policy)
# ==========================================
def generate_data(N, R):
    np.random.seed(42)
    xi1 = np.random.normal(loc=6.0, scale=2.5, size=(N, R))
    xi1 = np.maximum(xi1, 0.0)
    mean_xi2 = 2 * (6.0 - xi1)**2
    xi2 = np.random.normal(loc=mean_xi2, scale=2.5)
    xi2 = np.maximum(xi2, 0.0)
    return xi1, xi2

# ==========================================
# MODEL (32-32-3 architecture)
# ==========================================
class PredictAndProjectModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.fc_layers = nn.Sequential(
            nn.Linear(config.R * 2 + 1, 32), nn.ReLU(),
            nn.Linear(32, 32),               nn.ReLU(),
            nn.Linear(32, config.R)
        )
        self.Q10 = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.Q1r = nn.Parameter(torch.zeros(config.R, dtype=torch.float32))

        x     = cp.Variable(config.R)
        z_p   = cp.Parameter(config.R)
        Q10_p = cp.Parameter(1)
        cons  = [cp.sum(x) <= Q10_p + 1e-6, x >= 0]
        self.projection_layer = CvxpyLayer(
            cp.Problem(cp.Minimize(cp.sum_squares(x - z_p)), cons),
            parameters=[z_p, Q10_p], variables=[x]
        )

    def per_sample_cost(self, xi1, xi2):
        batch_size = xi1.shape[0]
        # Apply softplus to ensure Q10 and Q1r are non-negative
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        # Prepare input for the projection layer.
        q1r_input = q1r.unsqueeze(0).expand(batch_size, -1)
        q10_input = q10.expand(batch_size, 1)
        net_input = torch.cat([xi1, q1r_input, q10_input], dim=1)
        z_suggestion = self.fc_layers(net_input)

        q2, = self.projection_layer(
            z_suggestion, q10.expand(batch_size, 1),
            solver_args={
                "solve_method": "SCS",
                "eps":           cfg.SCS_EPS,
                "max_iters":     cfg.SCS_MAX_ITERS,
            }
        )

    # --- Compute cost using the cost function from the paper.
        stage1 = self.cfg.c * (q10 + torch.sum(q1r)) + self.cfg.h * q10

        t1 = self.cfg.b * (xi2 + xi1 - q2 - q1r) - self.cfg.h * q2
        t2 = self.cfg.h * (q1r - xi1 - xi2)
        t3 = self.cfg.b * (xi1 - q1r) - self.cfg.h * xi2
        v  = torch.sum(torch.maximum(torch.maximum(t1, t2), t3), dim=1)

        return stage1 + v

# ==========================================
# LOAD ADVERSARIAL SCENARIOS
# ==========================================
print("=" * 60)
print("MASTER PROBLEM 32-05 — CCG Iteration 2")
print(f"RHO_1={cfg.RHO_1}, RHO_2={cfg.RHO_2}, N={cfg.N_SAMPLES}")
print("=" * 60)

print("\nLoading adversarial scenarios...")
adv_xi1_list = []
adv_xi2_list = []
ball_ids      = []

for path in cfg.SCENARIOS_FILES:
    with open(path) as f:
        scenarios = json.load(f)
    for s in scenarios:
        adv_xi1_list.append(s["worst_xi1"])
        adv_xi2_list.append(s["worst_xi2"])
        ball_ids.append(s["ball_id"])

adv_xi1_by_ball = {}
adv_xi2_by_ball = {}
for idx, bid in enumerate(ball_ids):
    adv_xi1_by_ball.setdefault(bid, []).append(adv_xi1_list[idx])
    adv_xi2_by_ball.setdefault(bid, []).append(adv_xi2_list[idx])

print(f"Loaded {len(ball_ids)} adversarial scenarios across {len(cfg.SCENARIOS_FILES)} file(s).")
print(f"Covering {len(adv_xi1_by_ball)} unique balls.")

# ==========================================
# TRAINING DATA
# ==========================================
print("\nGenerating original training data...")
xi1_np, xi2_np = generate_data(cfg.N_SAMPLES, cfg.R)
xi1_orig = torch.tensor(xi1_np, dtype=torch.float32)
xi2_orig = torch.tensor(xi2_np, dtype=torch.float32)

max_cols = max(len(v) for v in adv_xi1_by_ball.values())
adv_xi1_cols = [xi1_np.copy() for _ in range(max_cols)]
adv_xi2_cols = [xi2_np.copy() for _ in range(max_cols)]
for bid in adv_xi1_by_ball:
    for k, (ax1, ax2) in enumerate(zip(adv_xi1_by_ball[bid], adv_xi2_by_ball[bid])):
        adv_xi1_cols[k][bid] = ax1
        adv_xi2_cols[k][bid] = ax2

adv_xi1_tensors = [torch.tensor(c, dtype=torch.float32) for c in adv_xi1_cols]
adv_xi2_tensors = [torch.tensor(c, dtype=torch.float32) for c in adv_xi2_cols]
print(f"Built {max_cols} adversarial column(s), each shape (N={cfg.N_SAMPLES}, R={cfg.R})")

# ==========================================
# MODEL + OPTIMIZER
# ==========================================
model = PredictAndProjectModel(cfg)

optimizer = torch.optim.Adam([
    {'params': model.fc_layers.parameters(), 'lr': 0.01},
    {'params': [model.Q10, model.Q1r],       'lr': 0.05},
])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.7, patience=80, threshold=1e-3
)

# --- Load starting checkpoint ---
print(f"\nLoading starting checkpoint: {cfg.CHECKPOINT_IN}")
if os.path.exists(cfg.CHECKPOINT_IN):
    try:
        ckpt = torch.load(cfg.CHECKPOINT_IN, map_location='cpu')
        state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"  Loaded from epoch {ckpt.get('epoch', '?')}")
    except RuntimeError as e:
        print(f"  WARNING: checkpoint incompatible ({e}). Starting fresh.")
else:
    print(f"  WARNING: {cfg.CHECKPOINT_IN} not found. Starting from scratch.")

q10_init = torch.nn.functional.softplus(model.Q10).item()
q1r_init = torch.nn.functional.softplus(model.Q1r).detach().numpy()
print(f"  Q10={q10_init:.4f}, Q1r={np.round(q1r_init, 3)}")

print(f"\nSCS settings: eps={cfg.SCS_EPS}, max_iters={cfg.SCS_MAX_ITERS}")
print(f"Early stop patience: {cfg.EARLY_STOP_PATIENCE}")

# ==========================================
# TRAINING LOOP
# ==========================================
print(f"\nStarting Master Problem 32-05 iter 2 training for up to {cfg.EPOCHS} epochs...")
print("Loss = mean_i [ max(cost(xi_hat_i), cost(worst_xi_i_1), cost(worst_xi_i_2)) ]\n")

best_loss = float('inf')
epochs_no_improve = 0
t_start = time.time()

for epoch in range(1, cfg.EPOCHS + 1):
    t_ep = time.time()
    optimizer.zero_grad()

    orig_costs = model.per_sample_cost(xi1_orig, xi2_orig)

    cost_columns = [orig_costs]
    for t_xi1, t_xi2 in zip(adv_xi1_tensors, adv_xi2_tensors):
        cost_columns.append(model.per_sample_cost(t_xi1, t_xi2))
    #taking worst cost across all adversarial scenarios + original for each sample
    worst_costs = torch.stack(cost_columns, dim=1).max(dim=1).values
    #computing mean of worst costs. this is the loss we want to minimize
    loss = worst_costs.mean()
    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())

    current_loss = loss.item()
    if current_loss < best_loss - 1e-3:
        best_loss = current_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epoch % 10 == 0 or epoch == 1:
        lr_net = optimizer.param_groups[0]['lr']
        lr_q   = optimizer.param_groups[1]['lr']
        q10_v  = torch.nn.functional.softplus(model.Q10).item()
        q1r_v  = torch.nn.functional.softplus(model.Q1r).detach().numpy()
        elapsed = time.time() - t_start
        ep_time = time.time() - t_ep
        print(f"Epoch {epoch:4d}/{cfg.EPOCHS} | Loss: {current_loss:.4f} | "
              f"Q10: {q10_v:.3f} | Q1r: {np.round(q1r_v, 3)} | "
              f"lr_net: {lr_net:.5f} | lr_Q: {lr_q:.5f} | "
              f"ep: {ep_time:.2f}s | total: {elapsed:.1f}s | "
              f"no_improv: {epochs_no_improve}/{cfg.EARLY_STOP_PATIENCE}")

    if epoch % 100 == 0:
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss':            best_loss,
            'scs_eps':              cfg.SCS_EPS,
            'scs_max_iters':        cfg.SCS_MAX_ITERS,
            'n_samples':            cfg.N_SAMPLES,
        }, cfg.CHECKPOINT_OUT)
        print(f"  [checkpoint saved at epoch {epoch}]")

    if epochs_no_improve >= cfg.EARLY_STOP_PATIENCE:
        print(f"\n>> Early stopping at epoch {epoch}: "
              f"no improvement for {cfg.EARLY_STOP_PATIENCE} epochs. "
              f"Best loss: {best_loss:.4f}")
        break

total_time = time.time() - t_start
print(f"\nTraining completed in {total_time:.1f}s ({epoch} epochs)")

# ==========================================
# SAVE CHECKPOINT
# ==========================================
torch.save({
    'epoch':                epoch,
    'model_state_dict':     model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_loss':            best_loss,
    'scs_eps':              cfg.SCS_EPS,
    'scs_max_iters':        cfg.SCS_MAX_ITERS,
    'n_samples':            cfg.N_SAMPLES,
}, cfg.CHECKPOINT_OUT)
print(f"\n[+] Saved checkpoint to: {cfg.CHECKPOINT_OUT}")

# ==========================================
# FINAL SUMMARY
# ==========================================
with torch.no_grad():
    orig_costs_final = model.per_sample_cost(xi1_orig, xi2_orig)
    cost_cols_final  = [orig_costs_final]
    for t_xi1, t_xi2 in zip(adv_xi1_tensors, adv_xi2_tensors):
        cost_cols_final.append(model.per_sample_cost(t_xi1, t_xi2))
    worst_costs_final = torch.stack(cost_cols_final, dim=1).max(dim=1).values

q10_final = torch.nn.functional.softplus(model.Q10).item()
q1r_final = torch.nn.functional.softplus(model.Q1r).detach().numpy()

print("\n" + "=" * 60)
print("MASTER PROBLEM 32-05 — FINAL RESULTS (Iteration 2)")
print("=" * 60)
print(f"Q10:                     {q10_final:.4f}")
print(f"Q1r:                     {np.round(q1r_final, 4)}")
print(f"Mean original cost:      {orig_costs_final.mean().item():.4f}")
print(f"Mean worst-case cost:    {worst_costs_final.mean().item():.4f}")
print(f"Max  worst-case cost:    {worst_costs_final.max().item():.4f}")
print(f"Epochs completed:        {epoch}")
print(f"Training time:           {total_time:.1f}s")
print(f"\nCheckpoint saved to:     {cfg.CHECKPOINT_OUT}")
print("=" * 60)
