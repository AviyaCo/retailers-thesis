"""
eval_on_cond_lb.py  --  Evaluate neural policy on the same 640k conditional scenarios
                         used by saa_cond_lb.py (N=800, M=800, seed=42).

This checks whether saa_cond_lb (19.749098) is truly a lower bound:
  neural_cost_on_cond_data  >=  19.749098  ← should hold if LB is valid
"""
import os, json
import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

R        = 3
c, h, b  = 0.25, 0.05, 0.5
N        = 800
M        = 800
SEED     = 42

NOMINAL_DIR = "/data/home/aviyac/retailers_problem/R3/N800/saainit_net16_v2"
SWEEP_DIR   = "/data/home/aviyac/retailers_problem/R3/N800/rho_sweep_net16_v2"
COND_LB_JSON = "/data/home/aviyac/retailers_problem/R3/saa_cond_lb.json"

# ── Generate the SAME conditional data as saa_cond_lb.py ─────────────────────
print(f"Generating conditional data (N={N}, M={M}, seed={SEED}) ...")
rng = np.random.default_rng(SEED)
a_xi1 = (0.0 - 6.0) / 2.5
xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5,
                     size=(N, R), random_state=rng)          # (N, R)
mu2 = 2.0 * (6.0 - xi1) ** 2                                # (N, R)
xi2 = np.empty((N, M, R), dtype=np.float64)
for r in range(R):
    mu2_r = mu2[:, r]
    a_r   = (0.0 - mu2_r) / 2.5
    xi2[:, :, r] = truncnorm.rvs(
        a_r[:, None], np.inf,
        loc=mu2_r[:, None], scale=2.5,
        size=(N, M), random_state=rng,
    )
print(f"  xi1: {xi1.shape}  xi2: {xi2.shape}")

# ── Model architecture (matches eval_sweep_net16_v2_N800_10k.py) ─────────────
def _project_simplex_scaled(z, cap):
    scalar = z.ndim == 1
    if scalar:
        z = z[None, :]
    z_nn = np.maximum(z, 0.0)
    if np.all(z_nn.sum(axis=1) <= cap):
        return z_nn[0] if scalar else z_nn
    u = np.sort(z_nn, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    rho = np.sum(u * np.arange(1, z.shape[1] + 1) > (cssv - cap), axis=1) - 1
    theta = (cssv[np.arange(len(z)), rho] - cap) / (rho + 1.0)
    q = np.maximum(z_nn - theta[:, None], 0.0)
    needs = z_nn.sum(axis=1) > cap
    q[~needs] = z_nn[~needs]
    return q[0] if scalar else q

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
    def get_Q2_batch(self, xi1_np):
        """xi1_np: (N,R) → Q2: (N,R), q10: float, q1r: (R,)"""
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
        ni = torch.cat([xi1_t,
                        torch.tensor(q1r).unsqueeze(0).expand(len(xi1_np), -1),
                        torch.full((len(xi1_np), 1), q10)], dim=1)
        z  = self.fc_layers(ni).numpy()
        Q2 = _project_simplex_scaled(z, q10)
        return Q2, q10, q1r

def load_model(path):
    m = Model()
    sd = torch.load(path, map_location="cpu")
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

# ── Cost on conditional data ──────────────────────────────────────────────────
def eval_on_cond(model):
    """
    For each xi1_i, compute Q2_i = network(xi1_i) [shape (R,)].
    Then average cost over all M xi2_{ij} values.
    Returns mean cost over all NM pairs (= good estimate of E_{joint}[cost]).
    """
    Q2_all, q10, q1r = model.get_Q2_batch(xi1)   # (N, R), float, (R,)

    # For each xi1_i, Q2_i is fixed. Average cost over M xi2 samples.
    # xi2: (N, M, R),  Q2_all: (N, R) → broadcast Q2 to (N, 1, R)
    Q2_exp = Q2_all[:, None, :]       # (N, 1, R)
    q1r_exp = q1r[None, None, :]      # (1, 1, R)

    # v[i,j,r] = max(term1, term2, term3)
    term1 = b*(xi1[:,None,:] + xi2 - q1r_exp - Q2_exp) - h*Q2_exp
    term2 = h*(q1r_exp - xi1[:,None,:] - xi2)
    term3 = b*(xi1[:,None,:] - q1r_exp) - h*xi2
    v = np.maximum(term1, np.maximum(term2, term3))  # (N, M, R)

    s1 = c * (q10 + q1r.sum()) + h * q10
    # per-pair total cost: s1 + sum_r v[i,j,r]
    total = s1 + v.sum(axis=2)      # (N, M)
    mean_cost = float(total.mean())
    return mean_cost, q10, q1r

# ── Read conditional LB ───────────────────────────────────────────────────────
with open(COND_LB_JSON) as f:
    cond_lb = float(json.load(f)["saa_cond_lb"])
print(f"\nConditional SAA LB (clairvoyant): {cond_lb:.6f}")
print("=" * 60)

# ── Evaluate all nominal seeds ────────────────────────────────────────────────
print("\nEvaluating nominal models (rho=0) on 640k conditional scenarios ...")
nom_results = []
for seed in range(10):
    path = os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth")
    if not os.path.exists(path):
        continue
    m = load_model(path)
    cost, q10, q1r = eval_on_cond(m)
    gap = cost - cond_lb
    print(f"  seed={seed}:  cost={cost:.6f}  gap_from_LB={gap:+.6f}  "
          f"Q10={q10:.4f}  Q1r={np.round(q1r,4)}")
    nom_results.append((seed, cost))

if nom_results:
    best_seed, best_cost = min(nom_results, key=lambda x: x[1])
    print(f"\n  Best nominal: seed={best_seed}, cost={best_cost:.6f}")

# ── Evaluate best robust model (rho=0.001) ────────────────────────────────────
print("\nEvaluating rho=0.001 models on 640k conditional scenarios ...")
rho_dir = os.path.join(SWEEP_DIR, "rho_0p001")
rob_results = []
for seed in range(10):
    path = os.path.join(rho_dir, f"seed_{seed}_summary.json")
    pth  = os.path.join(rho_dir, f"seed_{seed}_mp_final.pth")
    if not os.path.exists(pth):
        # try alternate naming
        pth = os.path.join(rho_dir, f"seed_{seed}.pth")
    if not os.path.exists(pth):
        continue
    m = load_model(pth)
    cost, q10, q1r = eval_on_cond(m)
    gap = cost - cond_lb
    print(f"  seed={seed}:  cost={cost:.6f}  gap_from_LB={gap:+.6f}")
    rob_results.append((seed, cost))

if rob_results:
    best_seed_r, best_cost_r = min(rob_results, key=lambda x: x[1])
    print(f"\n  Best rho=0.001: seed={best_seed_r}, cost={best_cost_r:.6f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print(f"  Conditional SAA LB (clairvoyant, 640k scenarios): {cond_lb:.6f}")
if nom_results:
    best = min(v for _,v in nom_results)
    print(f"  Best nominal neural OOS (640k cond scenarios):     {best:.6f}")
    print(f"  Gap (neural - LB):                                 {best-cond_lb:+.6f}")
    if best >= cond_lb:
        print("  ✓  LB is VALID (neural >= clairvoyant LB)")
    else:
        print("  ✗  LB VIOLATED on this sample (neural < clairvoyant LB)")
