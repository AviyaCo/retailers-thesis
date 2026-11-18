"""
saa_cond_lb.py  --  Clairvoyant SAA LP on N x M *conditional* scenario set.
============================================================================
For each of N xi1 samples, draw M fresh xi2 samples from the TRUE conditional
distribution  xi2 | xi1_i ~ TruncNorm(mu2_i, 2.5, lb=0),  mu2_i = 2*(6-xi1_i)^2.
This gives NM = 640 000 scenario pairs (xi1_i, xi2_{ij}) drawn from the
TRUE joint distribution — i.e. the cross-product respects the stage-1 → stage-2
dependence.

Why this is a valid lower bound
---------------------------------
The clairvoyant Q2_{ij} is chosen after seeing BOTH xi1_i and xi2_{ij}.  As N,M→∞
the LP value converges to E[optimal_clairvoyant_cost] under the true distribution,
which lower-bounds any implementable policy's expected cost.

Contrast with saa_cross_lb.py
-------------------------------
saa_cross_lb.py pools xi2 samples independently and forms the *product of marginals*
distribution, which is NOT the true joint distribution.  Its LP value can be
higher (or lower) than the true clairvoyant bound — it is NOT a valid LB.
This script fixes that.

LP size (N = M = 800, R = 3):
  Variables  : 1 + R + 2*N*M*R = 3 840 004
  Constraints: N*M + 3*N*M*R   = 6 400 000

Saves result to  R3/saa_cond_lb.json  (key: "saa_cond_lb").
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
M        = 800   # conditional xi2 samples per xi1
SEED     = 42
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saa_cond_lb.json")


# ── Conditional data generation ────────────────────────────────────────────────
def gen_conditional(N, M, seed):
    """
    Returns:
        xi1  : (N, R)    — first-stage demands
        xi2  : (N, M, R) — xi2[i, j, r] ~ TruncNorm(2*(6-xi1[i,r])^2, 2.5, lb=0)
                           i.e. M fresh conditional samples per (xi1_i, retailer r)
    """
    rng = np.random.default_rng(seed)

    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5,
                         size=(N, R), random_state=rng)    # (N, R)

    mu2 = 2.0 * (6.0 - xi1) ** 2                          # (N, R)

    # For each retailer r, draw M conditional xi2 samples for all N xi1_i at once.
    # truncnorm.rvs accepts array-valued (a, loc) when size is specified.
    xi2 = np.empty((N, M, R), dtype=np.float64)
    for r in range(R):
        mu2_r  = mu2[:, r]                        # (N,)
        a_r    = (0.0 - mu2_r) / 2.5             # (N,) lower-truncation standard
        # Draw (N, M) samples: for xi1_i draw M values from xi2|xi1_i,r
        xi2[:, :, r] = truncnorm.rvs(
            a_r[:, None], np.inf,
            loc=mu2_r[:, None], scale=2.5,
            size=(N, M), random_state=rng,
        )
    return xi1, xi2


# ── Conditional cross-product SAA LP ──────────────────────────────────────────
def solve_cond_lp(xi1, xi2, verbose=True):
    """
    xi1 : (N, R)
    xi2 : (N, M, R)   xi2[i, j, r] = j-th conditional sample for xi1_i

    Variable layout (identical to saa_cross_lb.py):
        x[0]              : Q10
        x[1 : 1+R]        : Q1r[r]
        x[1+R : 1+R+NMR]  : Q2[i,j,r]  (anticipative)
        x[1+R+NMR : end]  : v[i,j,r]   (free)
    """
    N_loc = xi1.shape[0]
    M_loc = xi2.shape[1]
    NM    = N_loc * M_loc
    NMR   = NM * R

    I_Q10 = 0
    I_Q1R = 1
    I_Q2  = 1 + R
    I_V   = 1 + R + NMR
    n_vars = I_V + NMR

    # ── Objective ──────────────────────────────────────────────────────────
    c_obj              = np.zeros(n_vars)
    c_obj[I_Q10]       = c + h
    c_obj[I_Q1R:I_Q2]  = c
    c_obj[I_V:]        = 1.0 / NM

    # ── Lower bounds (v is free) ────────────────────────────────────────────
    lb             = np.full(n_vars, -GRB.INFINITY)
    lb[I_Q10]      = 0.0
    lb[I_Q1R:I_Q2] = 0.0
    lb[I_Q2:I_V]   = 0.0

    # ── Flat index arrays (no Python loops over scenarios) ─────────────────
    flat   = np.arange(NMR, dtype=np.int64)
    i_flat = flat // (M_loc * R)           # xi1 scenario index
    j_flat = (flat % (M_loc * R)) // R     # conditional xi2 sample index
    r_flat = flat % R                       # retailer index

    q2_col  = I_Q2 + (i_flat * M_loc + j_flat) * R + r_flat
    v_col   = I_V  + (i_flat * M_loc + j_flat) * R + r_flat
    q1r_col = I_Q1R + r_flat

    xi1_ir  = xi1[i_flat, r_flat]
    # KEY DIFFERENCE from saa_cross_lb: xi2 is conditional on xi1_i
    xi2_ijr = xi2[i_flat, j_flat, r_flat]

    # ── Budget constraints: Q10 - sum_r Q2[i,j,r] >= 0  for all (i,j) ────
    ij_flat = np.arange(NM, dtype=np.int64)
    ij_rep  = np.repeat(ij_flat, R)
    r_rep   = np.tile(np.arange(R, dtype=np.int64), NM)

    bud_rows = np.concatenate([ij_flat, ij_rep])
    bud_cols = np.concatenate([np.zeros(NM, dtype=np.int64),
                                I_Q2 + ij_rep * R + r_rep])
    bud_data = np.concatenate([np.ones(NM), np.full(NM * R, -1.0)])
    A_bud    = sp.csr_matrix((bud_data, (bud_rows, bud_cols)),
                              shape=(NM, n_vars), dtype=np.float64)
    bud_rhs  = np.zeros(NM)

    # ── Recourse constraints (same three types as saa_cross_lb) ───────────
    # Type 1: v + (b+h)*Q2 + b*Q1r >= b*(xi1+xi2)
    r1_rows = np.tile(flat, 3)
    r1_cols = np.concatenate([v_col, q2_col, q1r_col])
    r1_data = np.concatenate([np.ones(NMR), np.full(NMR, b + h), np.full(NMR, b)])
    rhs1    = b * (xi1_ir + xi2_ijr)

    # Type 2: v - h*Q1r >= -h*(xi1+xi2)
    r2_rows = np.tile(NMR + flat, 2)
    r2_cols = np.concatenate([v_col, q1r_col])
    r2_data = np.concatenate([np.ones(NMR), np.full(NMR, -h)])
    rhs2    = -h * (xi1_ir + xi2_ijr)

    # Type 3: v + b*Q1r >= b*xi1 - h*xi2
    r3_rows = np.tile(2 * NMR + flat, 2)
    r3_cols = np.concatenate([v_col, q1r_col])
    r3_data = np.concatenate([np.ones(NMR), np.full(NMR, b)])
    rhs3    = b * xi1_ir - h * xi2_ijr

    A_rec   = sp.csr_matrix(
        (np.concatenate([r1_data, r2_data, r3_data]),
         (np.concatenate([r1_rows, r2_rows, r3_rows]),
          np.concatenate([r1_cols, r2_cols, r3_cols]))),
        shape=(3 * NMR, n_vars), dtype=np.float64)
    rhs_rec = np.concatenate([rhs1, rhs2, rhs3])

    # ── Gurobi barrier (no crossover) ─────────────────────────────────────
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1 if verbose else 0)
    env.start()
    model = gp.Model(env=env)
    model.setParam("Method", 2)
    model.setParam("Crossover", 0)

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
    print(f"SAA Conditional LB  (R={R}, N={N}, M={M}, seed={SEED})")
    print(f"  cost params: c={c}, h={h}, b={b}")
    print(f"  xi2 | xi1_i ~ TruncNorm(2*(6-xi1_i)^2, 2.5, lb=0)  [true conditional]")

    t0 = time.time()
    print(f"\nGenerating {N} xi1 samples + {M} conditional xi2 per xi1 ...")
    xi1, xi2 = gen_conditional(N, M, SEED)
    print(f"  xi1:  {xi1.shape}  [{xi1.min():.3f}, {xi1.max():.3f}]")
    print(f"  xi2:  {xi2.shape}  [{xi2.min():.3f}, {xi2.max():.3f}]")
    print(f"  (xi2 is correlated with xi1 — uses true joint distribution)")

    NM           = N * M
    NMR          = NM * R
    n_vars_total = 1 + R + NMR + NMR
    n_cons_total = NM + 3 * NMR
    print(f"\nLP size:  {n_vars_total:,} variables,  {n_cons_total:,} constraints")
    print("Building sparse constraint matrix ...")

    lb_val, Q10_sol, Q1r_sol, t_solve = solve_cond_lp(xi1, xi2)

    t_total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"SAA Conditional LB  =  {lb_val:.6f}")
    print(f"  Q10      = {Q10_sol:.4f}")
    print(f"  Q1r      = {np.round(Q1r_sol, 4)}")
    print(f"  sum Q1r  = {Q1r_sol.sum():.4f}")
    print(f"  solve    = {t_solve:.1f}s  ({t_solve/60:.1f} min)")
    print(f"  total    = {t_total:.1f}s  ({t_total/60:.1f} min)")

    # Compare with cross-product LB (if available)
    cross_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saa_cross_lb.json")
    if os.path.exists(cross_json):
        cross_lb = json.load(open(cross_json))["saa_cross_lb"]
        print(f"\n  Cross-product LB (product distribution): {cross_lb:.6f}")
        print(f"  Conditional LB   (true joint distribution): {lb_val:.6f}")
        print(f"  Difference: {lb_val - cross_lb:+.6f}")

    result = {
        "saa_cond_lb":   lb_val,
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
            "Clairvoyant SAA LP on N*M conditional scenarios. "
            "For each xi1_i, M fresh xi2 samples are drawn from the TRUE conditional "
            "distribution xi2|xi1_i ~ TruncNorm(2*(6-xi1_i)^2, 2.5, lb=0). "
            "This is a VALID lower bound on any policy's expected cost under the "
            "true joint distribution. "
            "Contrast with saa_cross_lb.py which uses the product of marginals "
            "(NOT the true joint) and is therefore NOT a valid lower bound."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")
