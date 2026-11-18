"""
saa_cross_lb.py  --  Clairvoyant SAA LP on N x N cross-product scenario set.
Cross-product clairvoyant LP lower bound.
==============================================================================
Standard SAA LP but with 800 x 800 = 640 000 synthetic scenarios formed by
taking the Cartesian product of 800 xi1 samples x 800 xi2 samples.

For each cross-product scenario (xi1_i, xi2_j), Q2_{ij} is chosen optimally
after seeing both xi1 and xi2 (fully anticipative / clairvoyant).  The LP
objective is the mean cost over all 640 000 pairs:

  min  (c+h)*Q10 + c*sum_r Q1r_r
       + 1/(N*M) * sum_{i,j,r} v[i,j,r]

  s.t. Q10 - sum_r Q2[i,j,r] >= 0                                  for all (i,j)
       v[i,j,r] + (b+h)*Q2[i,j,r] + b*Q1r[r] >= b*(xi1[i,r]+xi2[j,r])
       v[i,j,r] - h*Q1r[r]                    >= -h*(xi1[i,r]+xi2[j,r])
       v[i,j,r] + b*Q1r[r]                    >=  b*xi1[i,r] - h*xi2[j,r]
       Q10, Q1r[r], Q2[i,j,r] >= 0;  v[i,j,r] free

This is a valid lower bound on any policy's expected cost (including the
neural policy that only sees xi1), because it gives Q2 full information
and uses a dense uniform distribution over the xi1 x xi2 cross-product.

LP size (N = M = 800, R = 3):
  Variables  : 1 + R + 2*N*M*R = 3,840,004
  Constraints: N*M + 3*N*M*R   = 6,400,000

Uses numpy vectorisation to build the sparse constraint matrix (no Python
loops over scenarios).  Uses Gurobi barrier without crossover.
Estimated runtime: 20-90 min depending on hardware.
"""

import os, json, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import truncnorm
import gurobipy as gp
from gurobipy import GRB

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

# ── Configuration ──────────────────────────────────────────────────────────────
R        = 3
c, h, b  = 0.25, 0.05, 0.5
N        = 800   # xi1 samples
M        = 800   # xi2 samples (drawn together with xi1, same seed)
SEED     = 42
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saa_cross_lb.json")


# ── Data generation (same distribution as all other scripts) ──────────────────
def gen(N, seed):
    np.random.seed(seed)
    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5, size=(N, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2    # both (N, R)


# ── Cross-product SAA LP ───────────────────────────────────────────────────────
def solve_cross_lp(xi1, xi2, verbose=True):
    """
    xi1 : (N, R)  first-stage demand scenarios
    xi2 : (M, R)  second-stage demand scenarios (used as the xi2 pool)

    Variable layout
    ---------------
    x[0]              : Q10
    x[1 : 1+R]        : Q1r[r]
    x[1+R : 1+R+NMR]  : Q2[i,j,r]  = x[1+R  + (i*M+j)*R + r]   (anticipative)
    x[1+R+NMR : end]  : v[i,j,r]   = x[1+R+NMR + (i*M+j)*R + r] (free)
    """
    N_loc = xi1.shape[0]
    M_loc = xi2.shape[0]
    NM    = N_loc * M_loc
    NMR   = NM * R

    # ── Variable index offsets ─────────────────────────────────────────────
    I_Q10 = 0
    I_Q1R = 1
    I_Q2  = 1 + R
    I_V   = 1 + R + NMR
    n_vars = I_V + NMR

    # ── Objective ─────────────────────────────────────────────────────────
    c_obj              = np.zeros(n_vars)
    c_obj[I_Q10]       = c + h
    c_obj[I_Q1R:I_Q2]  = c
    c_obj[I_V:]        = 1.0 / NM

    # ── Variable lower bounds (v is free) ─────────────────────────────────
    lb             = np.full(n_vars, -GRB.INFINITY)
    lb[I_Q10]      = 0.0
    lb[I_Q1R:I_Q2] = 0.0
    lb[I_Q2:I_V]   = 0.0

    # ── Flat index arrays (no Python loops) ───────────────────────────────
    flat   = np.arange(NMR, dtype=np.int64)
    i_flat = flat // (M_loc * R)           # xi1 scenario index
    j_flat = (flat % (M_loc * R)) // R     # xi2 scenario index
    r_flat = flat % R                       # retailer index

    q2_col  = I_Q2 + (i_flat * M_loc + j_flat) * R + r_flat
    v_col   = I_V  + (i_flat * M_loc + j_flat) * R + r_flat
    q1r_col = I_Q1R + r_flat

    xi1_ir = xi1[i_flat, r_flat]
    xi2_jr = xi2[j_flat, r_flat]

    # ── Budget constraints: Q10 - sum_r Q2[i,j,r] >= 0  for all (i,j) ───
    ij_flat = np.arange(NM, dtype=np.int64)          # row index per (i,j) pair
    ij_rep  = np.repeat(ij_flat, R)
    r_rep   = np.tile(np.arange(R, dtype=np.int64), NM)

    bud_rows = np.concatenate([ij_flat, ij_rep])
    bud_cols = np.concatenate([np.zeros(NM, dtype=np.int64),
                                I_Q2 + ij_rep * R + r_rep])
    bud_data = np.concatenate([np.ones(NM), np.full(NM * R, -1.0)])
    A_bud    = sp.csr_matrix((bud_data, (bud_rows, bud_cols)),
                              shape=(NM, n_vars), dtype=np.float64)
    bud_rhs  = np.zeros(NM)

    # ── Recourse constraints ──────────────────────────────────────────────
    # Type 1: v + (b+h)*Q2 + b*Q1r >= b*(xi1+xi2)  [3 nonzeros / row]
    r1_rows = np.tile(flat, 3)
    r1_cols = np.concatenate([v_col, q2_col, q1r_col])
    r1_data = np.concatenate([np.ones(NMR), np.full(NMR, b + h), np.full(NMR, b)])
    rhs1    = b * (xi1_ir + xi2_jr)

    # Type 2: v - h*Q1r >= -h*(xi1+xi2)  [2 nonzeros / row]
    r2_rows = np.tile(NMR + flat, 2)
    r2_cols = np.concatenate([v_col, q1r_col])
    r2_data = np.concatenate([np.ones(NMR), np.full(NMR, -h)])
    rhs2    = -h * (xi1_ir + xi2_jr)

    # Type 3: v + b*Q1r >= b*xi1 - h*xi2  [2 nonzeros / row]
    r3_rows = np.tile(2 * NMR + flat, 2)
    r3_cols = np.concatenate([v_col, q1r_col])
    r3_data = np.concatenate([np.ones(NMR), np.full(NMR, b)])
    rhs3    = b * xi1_ir - h * xi2_jr

    A_rec   = sp.csr_matrix(
        (np.concatenate([r1_data, r2_data, r3_data]),
         (np.concatenate([r1_rows, r2_rows, r3_rows]),
          np.concatenate([r1_cols, r2_cols, r3_cols]))),
        shape=(3 * NMR, n_vars), dtype=np.float64)
    rhs_rec = np.concatenate([rhs1, rhs2, rhs3])

    # ── Build and solve Gurobi model ──────────────────────────────────────
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam("Method", 2)     # barrier — fastest for large LPs
    model.setParam("Crossover", 0)  # skip crossover; only need obj value

    x = model.addMVar(n_vars, lb=lb, obj=c_obj, name="x")
    model.ModelSense = GRB.MINIMIZE

    model.addMConstr(A_bud, x, '>', bud_rhs,  name="budget")
    model.addMConstr(A_rec, x, '>', rhs_rec,  name="recourse")

    t_solve = time.time()
    model.optimize()
    t_solve = time.time() - t_solve

    STATUS_NAMES = {
        GRB.OPTIMAL:    "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED:  "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }
    status_str = STATUS_NAMES.get(model.Status, f"UNKNOWN({model.Status})")
    print(f"\nGurobi status: {status_str} (code={model.Status})")

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Gurobi did not reach OPTIMAL (status={status_str}). "
            "Solution cannot be used as a valid lower bound."
        )

    xv      = x.X
    obj_val = model.ObjVal
    Q10_val = float(xv[I_Q10])
    Q1r_val = xv[I_Q1R:I_Q2].copy()

    return obj_val, Q10_val, Q1r_val, t_solve


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"SAA Cross-Product LB  (R={R}, N={N}, M={M}, seed={SEED})")
    print(f"  cost params: c={c}, h={h}, b={b}")

    t0 = time.time()
    print(f"\nGenerating {N} scenarios ...")
    xi1, xi2 = gen(N, SEED)
    print(f"  xi1: {xi1.shape}  [{xi1.min():.3f}, {xi1.max():.3f}]")
    print(f"  xi2: {xi2.shape}  [{xi2.min():.3f}, {xi2.max():.3f}]")

    NM           = N * M
    NMR          = NM * R
    n_vars_total = 1 + R + NMR + NMR
    n_cons_total = NM + 3 * NMR
    print(f"\nLP size:  {n_vars_total:,} variables,  {n_cons_total:,} constraints")
    print("Building sparse constraint matrix ...")

    lb_val, Q10_sol, Q1r_sol, t_solve = solve_cross_lp(xi1, xi2)

    t_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SAA Cross-Product LB  =  {lb_val:.6f}")
    print(f"  Q10      = {Q10_sol:.4f}")
    print(f"  Q1r      = {np.round(Q1r_sol, 4)}")
    print(f"  sum Q1r  = {Q1r_sol.sum():.4f}")
    print(f"  solve    = {t_solve:.1f}s  ({t_solve/60:.1f} min)")
    print(f"  total    = {t_total:.1f}s  ({t_total/60:.1f} min)")

    result = {
        "saa_cross_lb":  lb_val,
        "Q10":           Q10_sol,
        "Q1r":           Q1r_sol.tolist(),
        "N":             N,
        "M":             M,
        "seed":          SEED,
        "R":             R,
        "c": c, "h": h, "b": b,
        "n_vars":        n_vars_total,
        "n_constraints": n_cons_total,
        "solve_sec":     t_solve,
        "total_sec":     t_total,
        "note": (
            "Clairvoyant SAA LP on N*M=640000 cross-product scenarios. "
            "Q2[i,j] is chosen optimally given (xi1_i, xi2_j) — fully anticipative. "
            "Valid lower bound on any policy's expected cost."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
