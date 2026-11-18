import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import json

# ==========================================
# CONFIGURATION
# ==========================================
class APConfig:
    R = 3
    c = 0.25
    h = 0.05
    b = 0.5

    # Uncertainty ball radii
    RHO_1 = 0.5   # radius for xi1
    RHO_2 = 0.5   # radius for xi2

cfg = APConfig()

def softplus_np(x):
    """Numpy softplus: log(1 + exp(x)). Matches torch.nn.functional.softplus used in forward()."""
    return np.log1p(np.exp(x))

def load_weights(checkpoint_path):
    """Load network weights and stage-1 parameters from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    Q10_val = softplus_np(state_dict['Q10'].numpy())[0]
    Q1r_val = softplus_np(state_dict['Q1r'].numpy())

    W1_full = state_dict['fc_layers.0.weight'].numpy()  # shape: (64, 2R+1)
    b1_raw  = state_dict['fc_layers.0.bias'].numpy()    # shape: (64,)

    W1_xi1 = W1_full[:, :cfg.R]
    b1_eff  = (b1_raw
               + W1_full[:, cfg.R:2*cfg.R] @ Q1r_val
               + W1_full[:, 2*cfg.R:]       @ [Q10_val])

    weights = {
        'W1':  W1_xi1,
        'b1':  b1_eff,
        'W2':  state_dict['fc_layers.2.weight'].numpy(),
        'b2':  state_dict['fc_layers.2.bias'].numpy(),
        'W3':  state_dict['fc_layers.4.weight'].numpy(),
        'b3':  state_dict['fc_layers.4.bias'].numpy(),
        'Q10': Q10_val,
        'Q1r': Q1r_val,
    }
    return weights

def solve_ap_for_sample(weights, xi1_hat, xi2_hat, sample_idx):
    """
    Solves the Adversarial Problem (AP) for a specific historical sample.
    Finds the worst-case xi within the uncertainty ball around (xi1_hat, xi2_hat).
    """
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']
    W3, b3 = weights['W3'], weights['b3']
    Q10_val = weights['Q10']
    Q1r_val = weights['Q1r']

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(f"AdversarialProblem_Sample_{sample_idx}", env=env)

    # Uncertainty ball
    xi1 = m.addVars(cfg.R, lb=np.maximum(0, xi1_hat - cfg.RHO_1), ub=xi1_hat + cfg.RHO_1, name="xi1")
    xi2 = m.addVars(cfg.R, lb=np.maximum(0, xi2_hat - cfg.RHO_2), ub=xi2_hat + cfg.RHO_2, name="xi2")

    # Neural Network MILP Formulation (Fischetti & Jo)
    def add_relu_layer(input_vars, W, b, layer_idx):
        out_dim = W.shape[0]
        x_plus  = m.addVars(out_dim, lb=0, name=f"x_plus_{layer_idx}")
        x_minus = m.addVars(out_dim, lb=0, name=f"x_minus_{layer_idx}")
        z       = m.addVars(out_dim, vtype=GRB.BINARY, name=f"z_{layer_idx}")
        for j in range(out_dim):
            expr = gp.quicksum(W[j, k] * input_vars[k] for k in range(W.shape[1])) + b[j]
            m.addConstr(expr == x_plus[j] - x_minus[j], name=f"relu_eq_{layer_idx}_{j}")
            m.addGenConstrIndicator(z[j], 1, x_plus[j]  == 0, name=f"ind_plus_{layer_idx}_{j}")
            m.addGenConstrIndicator(z[j], 0, x_minus[j] == 0, name=f"ind_minus_{layer_idx}_{j}")
        return x_plus

    layer1_out = add_relu_layer(xi1, W1, b1, 1)
    layer2_out = add_relu_layer(layer1_out, W2, b2, 2)

    z_suggest = m.addVars(cfg.R, lb=-GRB.INFINITY, name="z_suggest")
    for j in range(cfg.R):
        expr = gp.quicksum(W3[j, k] * layer2_out[k] for k in range(W3.shape[1])) + b3[j]
        m.addConstr(z_suggest[j] == expr, name=f"final_layer_{j}")

    # Projection Layer KKT Conditions
    Q10_eff = Q10_val
    q2  = m.addVars(cfg.R, lb=0, name="q2")
    lam = m.addVar(lb=0, name="lambda")
    mu  = m.addVars(cfg.R, lb=0, name="mu")

    for r in range(cfg.R):
        m.addConstr(2.0 * q2[r] - 2 * z_suggest[r] + lam - mu[r] == 0, name=f"stationarity_{r}")

    m.addConstr(gp.quicksum(q2[r] for r in range(cfg.R)) <= Q10_eff, name="capacity")

    u_lam = m.addVar(vtype=GRB.BINARY, name="u_lam")
    m.addGenConstrIndicator(u_lam, 1, lam == 0, name="comp_lam_1")
    m.addGenConstrIndicator(u_lam, 0, Q10_eff - gp.quicksum(q2[r] for r in range(cfg.R)) == 0, name="comp_lam_0")

    u_mu = m.addVars(cfg.R, vtype=GRB.BINARY, name="u_mu")
    for r in range(cfg.R):
        m.addGenConstrIndicator(u_mu[r], 1, mu[r]  == 0, name=f"comp_mu_1_{r}")
        m.addGenConstrIndicator(u_mu[r], 0, q2[r]  == 0, name=f"comp_mu_0_{r}")

    # Objective: maximise total cost
    v = m.addVars(cfg.R, lb=-GRB.INFINITY, name="v")
    for r in range(cfg.R):
        g1 = m.addVar(lb=-GRB.INFINITY, name=f"g1_{r}")
        g2 = m.addVar(lb=-GRB.INFINITY, name=f"g2_{r}")
        g3 = m.addVar(lb=-GRB.INFINITY, name=f"g3_{r}")
        m.addConstr(g1 == cfg.b * (xi1[r] + xi2[r] - Q1r_val[r] - q2[r]) - cfg.h * q2[r])
        m.addConstr(g2 == cfg.h * (Q1r_val[r] - xi1[r] - xi2[r]))
        m.addConstr(g3 == cfg.b * (xi1[r] - Q1r_val[r]) - cfg.h * xi2[r])
        m.addGenConstrMax(v[r], [g1, g2, g3], name=f"max_v_{r}")

    stage1_cost = cfg.c * (Q10_val + np.sum(Q1r_val)) + cfg.h * Q10_val
    m.setObjective(stage1_cost + gp.quicksum(v[r] for r in range(cfg.R)), GRB.MAXIMIZE)

    m.optimize()

    if m.status == GRB.OPTIMAL:
        worst_xi1 = np.array([xi1[r].X for r in range(cfg.R)])
        worst_xi2 = np.array([xi2[r].X for r in range(cfg.R)])
        max_cost  = m.ObjVal
        return worst_xi1, worst_xi2, max_cost
    else:
        return None, None, None

def generate_data(N, R):
    np.random.seed(42)
    xi1 = np.random.normal(loc=6.0, scale=2.5, size=(N, R))
    xi1 = np.maximum(xi1, 0.0)
    mean_xi2 = 2 * (6.0 - xi1) ** 2
    xi2 = np.random.normal(loc=mean_xi2, scale=2.5)
    xi2 = np.maximum(xi2, 0.0)
    return xi1, xi2

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    return obj

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 60)
    print("ADVERSARIAL PROBLEM V4 — CCG Iteration 1")
    print(f"RHO_1={cfg.RHO_1}, RHO_2={cfg.RHO_2}")
    print("=" * 60)

    N_SAMPLES = 800
    xi1_np, xi2_np = generate_data(N_SAMPLES, cfg.R)

    # --- Update checkpoint path each CCG iteration ---
    checkpoint_file = "/data/home/aviyac/retailers_problem/R3/N800/run_v4_64_05/thesis_checkpoint_v4.pth"  # iter1: nominal policy

    print(f"Loading checkpoint: {checkpoint_file}")
    weights = load_weights(checkpoint_file)
    print(f"Q10={weights['Q10']:.4f}, Q1r={np.round(weights['Q1r'], 3)}")

    adversarial_scenarios = []
    print(f"\nStarting AP loop for {N_SAMPLES} samples...")

    for i in range(N_SAMPLES):
        if i % 100 == 0:
            elapsed = time.time() - t_start
            print(f"Processing sample {i}/{N_SAMPLES}... ({elapsed:.0f}s elapsed)")
        xi1_hat = xi1_np[i]
        xi2_hat = xi2_np[i]
        worst_xi1, worst_xi2, max_cost = solve_ap_for_sample(weights, xi1_hat, xi2_hat, sample_idx=i)
        if worst_xi1 is not None:
            adversarial_scenarios.append({
                "ball_id":   i,
                "xi1_hat":   xi1_hat,
                "xi2_hat":   xi2_hat,
                "worst_xi1": worst_xi1,
                "worst_xi2": worst_xi2,
                "max_cost":  max_cost,
            })
        else:
            print(f"Warning: Failed to solve AP for sample {i}")

    serializable = [
        {
            "ball_id":   int(e["ball_id"]),
            "xi1_hat":   to_serializable(e["xi1_hat"]),
            "xi2_hat":   to_serializable(e["xi2_hat"]),
            "worst_xi1": to_serializable(e["worst_xi1"]),
            "worst_xi2": to_serializable(e["worst_xi2"]),
            "max_cost":  float(e["max_cost"]),
        }
        for e in adversarial_scenarios
    ]

    save_path = "/data/home/aviyac/retailers_problem/R3/N800/run_v4_64_05/adversarial_scenarios_iter1_v4.json"
    with open(save_path, "w") as f:
        json.dump(serializable, f)

    t_total = time.time() - t_start
    costs = [e["max_cost"] for e in adversarial_scenarios]
    print(f"\n{'=' * 60}")
    print(f"[+] Finished {N_SAMPLES} samples in {t_total:.0f}s ({t_total/60:.1f} min)")
    print(f"[+] Saved {len(adversarial_scenarios)} adversarial scenarios to {save_path}")
    print(f"Mean worst-case cost: {np.mean(costs):.4f}")
    print(f"Max  worst-case cost: {np.max(costs):.4f}")
    print(f"Min  worst-case cost: {np.min(costs):.4f}")
    print(f"Std  worst-case cost: {np.std(costs):.4f}")
    print(f"{'=' * 60}")
