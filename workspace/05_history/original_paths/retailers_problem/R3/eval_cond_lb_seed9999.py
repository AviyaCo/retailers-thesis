"""
eval_cond_lb_seed9999.py
========================
Run the clairvoyant conditional LP *and* evaluate the best neural policy
on the SAME 640k scenarios (N=800, M=800, seed=9999) — the same seed used
for the 10k OOS test set in the sweep eval.

Goal: check whether LP ≤ neural on fairly-matched data, and decide whether
the conditional LB is a good number to add to the plots.
"""
import os, json, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

R        = 3
c, h, b  = 0.25, 0.05, 0.5
N        = 800
M        = 800
SEED     = 9999          # ← same seed as the 10k OOS test set

NOMINAL_DIR = "/data/home/aviyac/retailers_problem/R3/N800/saainit_net16_v2"
OUT_JSON    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "saa_cond_lb_seed9999.json")

# ── Generate conditional data ─────────────────────────────────────────────────
print(f"Generating conditional data  (N={N}, M={M}, seed={SEED}) ...")
rng = np.random.default_rng(SEED)
a_xi1 = (0.0 - 6.0) / 2.5
xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5,
                     size=(N, R), random_state=rng)
mu2  = 2.0 * (6.0 - xi1) ** 2
xi2  = np.empty((N, M, R), dtype=np.float64)
for r in range(R):
    mu2_r = mu2[:, r]
    a_r   = (0.0 - mu2_r) / 2.5
    xi2[:, :, r] = truncnorm.rvs(
        a_r[:, None], np.inf, loc=mu2_r[:, None], scale=2.5,
        size=(N, M), random_state=rng,
    )
print(f"  xi1: {xi1.shape}  xi2: {xi2.shape}  "
      f"xi2 range [{xi2.min():.3f}, {xi2.max():.3f}]")

# ── Conditional clairvoyant LP (identical to saa_cond_lb.py) ─────────────────
def solve_cond_lp(xi1, xi2, verbose=True):
    N_loc, M_loc = xi1.shape[0], xi2.shape[1]
    NM = N_loc * M_loc;  NMR = NM * R
    I_Q10 = 0;  I_Q1R = 1;  I_Q2 = 1 + R;  I_V = 1 + R + NMR
    n_vars = I_V + NMR

    c_obj = np.zeros(n_vars)
    c_obj[I_Q10] = c + h;  c_obj[I_Q1R:I_Q2] = c;  c_obj[I_V:] = 1.0 / NM

    lb = np.full(n_vars, -GRB.INFINITY)
    lb[I_Q10] = 0.0;  lb[I_Q1R:I_Q2] = 0.0;  lb[I_Q2:I_V] = 0.0

    flat   = np.arange(NMR, dtype=np.int64)
    i_flat = flat // (M_loc * R)
    j_flat = (flat % (M_loc * R)) // R
    r_flat = flat % R
    q2_col  = I_Q2 + (i_flat * M_loc + j_flat) * R + r_flat
    v_col   = I_V  + (i_flat * M_loc + j_flat) * R + r_flat
    q1r_col = I_Q1R + r_flat
    xi1_ir  = xi1[i_flat, r_flat]
    xi2_ijr = xi2[i_flat, j_flat, r_flat]

    ij_flat = np.arange(NM, dtype=np.int64)
    ij_rep  = np.repeat(ij_flat, R)
    r_rep   = np.tile(np.arange(R, dtype=np.int64), NM)
    A_bud   = sp.csr_matrix(
        (np.concatenate([np.ones(NM), np.full(NM*R, -1.0)]),
         (np.concatenate([ij_flat, ij_rep]),
          np.concatenate([np.zeros(NM, dtype=np.int64), I_Q2 + ij_rep*R + r_rep]))),
        shape=(NM, n_vars), dtype=np.float64)
    bud_rhs = np.zeros(NM)

    r1_rows = np.tile(flat, 3)
    r1_cols = np.concatenate([v_col, q2_col, q1r_col])
    r1_data = np.concatenate([np.ones(NMR), np.full(NMR, b+h), np.full(NMR, b)])
    rhs1    = b * (xi1_ir + xi2_ijr)

    r2_rows = np.tile(NMR + flat, 2)
    r2_cols = np.concatenate([v_col, q1r_col])
    r2_data = np.concatenate([np.ones(NMR), np.full(NMR, -h)])
    rhs2    = -h * (xi1_ir + xi2_ijr)

    r3_rows = np.tile(2*NMR + flat, 2)
    r3_cols = np.concatenate([v_col, q1r_col])
    r3_data = np.concatenate([np.ones(NMR), np.full(NMR, b)])
    rhs3    = b * xi1_ir - h * xi2_ijr

    A_rec = sp.csr_matrix(
        (np.concatenate([r1_data, r2_data, r3_data]),
         (np.concatenate([r1_rows, r2_rows, r3_rows]),
          np.concatenate([r1_cols, r2_cols, r3_cols]))),
        shape=(3*NMR, n_vars), dtype=np.float64)
    rhs_rec = np.concatenate([rhs1, rhs2, rhs3])

    env = gp.Env(empty=True);  env.setParam("OutputFlag", 1 if verbose else 0);  env.start()
    model = gp.Model(env=env)
    model.setParam("Method", 2);  model.setParam("Crossover", 0)
    x = model.addMVar(n_vars, lb=lb, obj=c_obj, name="x")
    model.ModelSense = GRB.MINIMIZE
    model.addMConstr(A_bud, x, '>', bud_rhs,  name="budget")
    model.addMConstr(A_rec, x, '>', rhs_rec,  name="recourse")

    t0 = time.time()
    model.optimize()
    t_solve = time.time() - t0

    status_str = {GRB.OPTIMAL:"OPTIMAL",GRB.INFEASIBLE:"INFEASIBLE",
                  GRB.SUBOPTIMAL:"SUBOPTIMAL"}.get(model.Status, f"code={model.Status}")
    print(f"\nGurobi status: {status_str}")
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Not OPTIMAL: {status_str}")

    xv = x.X
    Q10 = xv[I_Q10];  Q1r = xv[I_Q1R:I_Q2]
    return model.ObjVal, Q10, Q1r, t_solve

# ── Neural policy helpers ─────────────────────────────────────────────────────
def _project_simplex_scaled(z, cap):
    scalar = z.ndim == 1
    if scalar: z = z[None, :]
    z_nn = np.maximum(z, 0.0)
    if np.all(z_nn.sum(axis=1) <= cap): return z_nn[0] if scalar else z_nn
    u = np.sort(z_nn, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    rho = np.sum(u * np.arange(1, z.shape[1]+1) > (cssv - cap), axis=1) - 1
    theta = (cssv[np.arange(len(z)), rho] - cap) / (rho + 1.0)
    q = np.maximum(z_nn - theta[:, None], 0.0)
    needs = z_nn.sum(axis=1) > cap;  q[~needs] = z_nn[~needs]
    return q[0] if scalar else q

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R*2+1, 16), nn.ReLU(), nn.Linear(16,16), nn.ReLU(), nn.Linear(16,R))
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))
    @torch.no_grad()
    def get_Q2_batch(self, xi1_np):
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
        ni = torch.cat([xi1_t,
                        torch.tensor(q1r).unsqueeze(0).expand(len(xi1_np),-1),
                        torch.full((len(xi1_np),1), q10)], dim=1)
        z  = self.fc_layers(ni).numpy()
        return _project_simplex_scaled(z, q10), q10, q1r

def load_model(path):
    m = Model()
    sd = torch.load(path, map_location="cpu")
    if "model_state_dict" in sd: sd = sd["model_state_dict"]
    m.load_state_dict(sd, strict=False);  m.eval()
    return m

def eval_neural_on_cond(model):
    Q2_all, q10, q1r = model.get_Q2_batch(xi1)     # (N,R)
    Q2_exp  = Q2_all[:, None, :]                    # (N,1,R)
    q1r_exp = q1r[None, None, :]                    # (1,1,R)
    term1 = b*(xi1[:,None,:] + xi2 - q1r_exp - Q2_exp) - h*Q2_exp
    term2 = h*(q1r_exp - xi1[:,None,:] - xi2)
    term3 = b*(xi1[:,None,:] - q1r_exp) - h*xi2
    v = np.maximum(term1, np.maximum(term2, term3))
    s1 = c*(q10 + q1r.sum()) + h*q10
    return float((s1 + v.sum(axis=2)).mean()), q10, q1r

# ── Run LP ────────────────────────────────────────────────────────────────────
t_total = time.time()
print(f"\nSolving clairvoyant LP  ({N}×{M}={N*M:,} scenarios, seed={SEED}) ...")
lp_val, Q10_lp, Q1r_lp, t_solve = solve_cond_lp(xi1, xi2)
t_total = time.time() - t_total

print(f"\n{'='*60}")
print(f"Clairvoyant LP (seed={SEED})  =  {lp_val:.6f}")
print(f"  Q10 = {Q10_lp:.4f}   Q1r = {np.round(Q1r_lp,4)}")
print(f"  solve={t_solve:.1f}s   total={t_total:.1f}s")

# ── Evaluate neural models on same data ───────────────────────────────────────
print(f"\nEvaluating nominal neural models on same {N*M:,} scenarios ...")
neural_results = []
for seed in range(10):
    path = os.path.join(NOMINAL_DIR, f"seed_{seed}_nominal.pth")
    if not os.path.exists(path): continue
    m = load_model(path)
    cost, q10_n, q1r_n = eval_neural_on_cond(m)
    gap = cost - lp_val
    print(f"  neural seed={seed}:  cost={cost:.6f}  gap={gap:+.6f}")
    neural_results.append((seed, cost))

best_seed, best_cost = min(neural_results, key=lambda x: x[1])

# ── Save & Summary ────────────────────────────────────────────────────────────
result = {"seed": SEED, "N": N, "M": M,
          "saa_cond_lb_seed9999": lp_val,
          "best_neural_cost_cond": best_cost,
          "gap": best_cost - lp_val}
with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n{'='*60}")
print(f"SUMMARY  (seed={SEED}, {N*M:,} conditional scenarios)")
print(f"  Clairvoyant LB  : {lp_val:.6f}")
print(f"  Best neural cost: {best_cost:.6f}  (seed={best_seed})")
print(f"  Gap (neural-LB) : {best_cost - lp_val:+.6f}")
if best_cost >= lp_val:
    print("  -> LB is VALID on this seed  ✓")
else:
    print("  -> LB VIOLATED on this seed  ✗")
print(f"\nSaved: {OUT_JSON}")
