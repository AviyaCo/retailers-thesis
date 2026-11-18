"""
saa_cond_mip_1m_shipping_exact.py
=================================
Conditional clairvoyant lower bound for the shipping problem with
N=5000 xi1 samples x M=200 conditional xi2 samples = 1,000,000 pairs.

This is the exact fixed-charge shipping MIP: shipment indicators z are binary.
At this scale the model has 3,000,000 binary variables, so it may take a long
time and substantial memory.
"""
import json
import os
import time
import argparse
from datetime import datetime, timedelta

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
from scipy.stats import truncnorm
import torch
import torch.nn as nn


os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

R = 3
c, h, b = 0.25, 0.05, 0.5
f_ship = 0.1
N = 5000
M = 200
SEED = 42

BASE_DIR = "/data/home/aviyac/retailers_with_shipping/N200"
OUT_JSON = os.path.join(BASE_DIR, "saa_cond_mip_1m_shipping_exact.json")
OUT_PARTIAL_JSON = os.path.join(BASE_DIR, "saa_cond_mip_1m_shipping_exact.partial.json")


parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=32)
parser.add_argument("--mip-gap", type=float, default=0.003,
                    help="Target relative MIP gap (default: 0.003 = 0.3%%)")
parser.add_argument("--time-limit-s", type=float, default=None,
                    help="Optional explicit time limit in seconds")
parser.add_argument("--heartbeat-s", type=int, default=600,
                    help="Progress heartbeat interval in seconds")
parser.add_argument("--disable-friday-limit", action="store_true",
                    help="Disable automatic limit until Friday 23:59 local time")
args = parser.parse_args()


def _seconds_until_friday_2359():
    now = datetime.now()
    days_ahead = (4 - now.weekday()) % 7  # Friday=4
    target = (now + timedelta(days=days_ahead)).replace(
        hour=23, minute=59, second=0, microsecond=0
    )
    # If already past Friday 23:59 this week, use next Friday.
    if target <= now:
        target = target + timedelta(days=7)
    return max(60.0, (target - now).total_seconds())


def _safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x) or abs(x) >= 1e99:
        return None
    return x


def _status_name(code):
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }
    return names.get(code, str(code))


def generate_conditional_data():
    print(f"Generating {N} xi1 samples and {M} conditional xi2 samples per xi1 ...")
    rng = np.random.default_rng(SEED)
    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(
        a_xi1, np.inf, loc=6.0, scale=2.5, size=(N, R), random_state=rng
    )
    mu2 = 2.0 * (6.0 - xi1) ** 2
    xi2 = np.empty((N, M, R), dtype=np.float64)
    for r in range(R):
        mu2_r = mu2[:, r]
        a_r = (0.0 - mu2_r) / 2.5
        xi2[:, :, r] = truncnorm.rvs(
            a_r[:, None],
            np.inf,
            loc=mu2_r[:, None],
            scale=2.5,
            size=(N, M),
            random_state=rng,
        )
    print(f"  xi1: {xi1.shape}, range [{xi1.min():.3f}, {xi1.max():.3f}]")
    print(f"  xi2: {xi2.shape}, range [{xi2.min():.3f}, {xi2.max():.3f}]")
    return xi1, xi2


def solve_exact_shipping_mip(xi1, xi2):
    n_loc, m_loc = xi1.shape[0], xi2.shape[1]
    nm = n_loc * m_loc
    nmr = nm * R

    i_q10 = 0
    i_q1r = 1
    i_q2 = 1 + R
    i_z = i_q2 + nmr
    i_v = i_z + nmr
    n_vars = i_v + nmr

    print(
        "Exact shipping MIP size: "
        f"{n_vars:,} vars ({nmr:,} binaries), "
        f"{nm + nmr + 3 * nmr:,} constraints"
    )

    obj = np.zeros(n_vars)
    obj[i_q10] = c + h
    obj[i_q1r:i_q2] = c
    obj[i_z:i_v] = f_ship / nm
    obj[i_v:] = 1.0 / nm

    lb = np.full(n_vars, -GRB.INFINITY)
    ub = np.full(n_vars, GRB.INFINITY)
    lb[i_q10] = 0.0
    lb[i_q1r:i_q2] = 0.0
    lb[i_q2:i_z] = 0.0
    lb[i_z:i_v] = 0.0
    ub[i_z:i_v] = 1.0

    flat = np.arange(nmr, dtype=np.int64)
    i_flat = flat // (m_loc * R)
    j_flat = (flat % (m_loc * R)) // R
    r_flat = flat % R
    q2_col = i_q2 + (i_flat * m_loc + j_flat) * R + r_flat
    z_col = i_z + (i_flat * m_loc + j_flat) * R + r_flat
    v_col = i_v + (i_flat * m_loc + j_flat) * R + r_flat
    q1r_col = i_q1r + r_flat
    xi1_ir = xi1[i_flat, r_flat]
    xi2_ijr = xi2[i_flat, j_flat, r_flat]

    print("  Building budget constraints ...")
    ij_flat = np.arange(nm, dtype=np.int64)
    ij_rep = np.repeat(ij_flat, R)
    r_rep = np.tile(np.arange(R, dtype=np.int64), nm)
    a_budget = sp.csr_matrix(
        (
            np.concatenate([np.ones(nm), np.full(nm * R, -1.0)]),
            (
                np.concatenate([ij_flat, ij_rep]),
                np.concatenate(
                    [
                        np.zeros(nm, dtype=np.int64),
                        i_q2 + ij_rep * R + r_rep,
                    ]
                ),
            ),
        ),
        shape=(nm, n_vars),
        dtype=np.float64,
    )

    print("  Building shipment-link constraints ...")
    big_m = float((xi1 + xi2.max(axis=1)).max()) + 1.0
    a_link = sp.csr_matrix(
        (
            np.concatenate([np.ones(nmr), np.full(nmr, -big_m)]),
            (np.concatenate([flat, flat]), np.concatenate([q2_col, z_col])),
        ),
        shape=(nmr, n_vars),
        dtype=np.float64,
    )

    print("  Building recourse constraints ...")
    r1_rows = np.tile(flat, 3)
    r1_cols = np.concatenate([v_col, q2_col, q1r_col])
    r1_data = np.concatenate([np.ones(nmr), np.full(nmr, b + h), np.full(nmr, b)])
    rhs1 = b * (xi1_ir + xi2_ijr)

    r2_rows = np.tile(nmr + flat, 2)
    r2_cols = np.concatenate([v_col, q1r_col])
    r2_data = np.concatenate([np.ones(nmr), np.full(nmr, -h)])
    rhs2 = -h * (xi1_ir + xi2_ijr)

    r3_rows = np.tile(2 * nmr + flat, 2)
    r3_cols = np.concatenate([v_col, q1r_col])
    r3_data = np.concatenate([np.ones(nmr), np.full(nmr, b)])
    rhs3 = b * xi1_ir - h * xi2_ijr

    a_rec = sp.csr_matrix(
        (
            np.concatenate([r1_data, r2_data, r3_data]),
            (
                np.concatenate([r1_rows, r2_rows, r3_rows]),
                np.concatenate([r1_cols, r2_cols, r3_cols]),
            ),
        ),
        shape=(3 * nmr, n_vars),
        dtype=np.float64,
    )

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 1)
    env.start()
    model = gp.Model(env=env)
    model.setParam("Threads", args.threads)
    model.setParam("MIPGap", args.mip_gap)

    if args.time_limit_s is not None:
        time_limit_s = float(args.time_limit_s)
    elif args.disable_friday_limit:
        time_limit_s = None
    else:
        time_limit_s = _seconds_until_friday_2359()

    if time_limit_s is not None:
        model.setParam("TimeLimit", time_limit_s)
        print(f"Using TimeLimit={time_limit_s:.0f}s (until Friday cutoff unless overridden).")
    print(f"Using MIPGap={args.mip_gap:.6f} and Threads={args.threads}.")

    x = model.addMVar(n_vars, lb=lb, ub=ub, obj=obj, name="x")
    x[i_z:i_v].VType = GRB.BINARY
    model.ModelSense = GRB.MINIMIZE
    model.addMConstr(a_budget, x, ">", np.zeros(nm), name="budget")
    model.addMConstr(a_link, x, "<", np.zeros(nmr), name="ship_link")
    model.addMConstr(
        a_rec, x, ">", np.concatenate([rhs1, rhs2, rhs3]), name="recourse"
    )

    hb = {"last": -1e18}

    def _heartbeat_callback(cb_model, where):
        if where != GRB.Callback.MIP:
            return
        runtime = cb_model.cbGet(GRB.Callback.RUNTIME)
        if runtime - hb["last"] < args.heartbeat_s:
            return

        best_obj = _safe_float(cb_model.cbGet(GRB.Callback.MIP_OBJBST))
        best_bd = _safe_float(cb_model.cbGet(GRB.Callback.MIP_OBJBND))
        solcnt = int(cb_model.cbGet(GRB.Callback.MIP_SOLCNT))
        nodecnt = float(cb_model.cbGet(GRB.Callback.MIP_NODCNT))
        gap = None
        if best_obj is not None and best_bd is not None and abs(best_obj) > 1e-12:
            gap = abs(best_obj - best_bd) / abs(best_obj)

        msg_gap = f"{100.0 * gap:.4f}%" if gap is not None else "n/a"
        print(
            f"[HEARTBEAT] t={runtime:.0f}s nodes={nodecnt:.0f} "
            f"sols={solcnt} best={best_obj} bound={best_bd} gap={msg_gap}",
            flush=True,
        )

        partial = {
            "N": N,
            "M": M,
            "seed": SEED,
            "scenarios": N * M,
            "exact_mip": True,
            "status": "RUNNING",
            "runtime_s": runtime,
            "best_obj": best_obj,
            "best_bound": best_bd,
            "mip_gap": gap,
            "sol_count": solcnt,
            "node_count": nodecnt,
            "time_limit_s": time_limit_s,
            "mip_gap_target": args.mip_gap,
        }
        with open(OUT_PARTIAL_JSON, "w") as f:
            json.dump(partial, f, indent=2)

        hb["last"] = runtime

    t0 = time.time()
    model.optimize(_heartbeat_callback)
    solve_time = time.time() - t0

    status_code = model.Status
    status_name = _status_name(status_code)
    if model.SolCount <= 0:
        raise RuntimeError(
            f"Gurobi finished without an incumbent solution: status={status_name} ({status_code})"
        )

    xv = x.X
    result = {
        "N": N,
        "M": M,
        "seed": SEED,
        "scenarios": N * M,
        "relaxed": False,
        "exact_mip": True,
        "shipping_binary_note": "z is binary, exact fixed-charge shipping clairvoyant MIP.",
        "saa_cond_mip_1m_shipping_exact": float(model.ObjVal),
        "status": status_name,
        "proven_optimal": bool(status_code == GRB.OPTIMAL),
        "best_bound": _safe_float(model.ObjBound),
        "mip_gap": _safe_float(model.MIPGap),
        "sol_count": int(model.SolCount),
        "node_count": float(model.NodeCount),
        "time_limit_s": time_limit_s,
        "mip_gap_target": args.mip_gap,
        "Q10": float(xv[i_q10]),
        "Q1r": xv[i_q1r:i_q2].tolist(),
        "solve_time_s": solve_time,
    }

    if status_code != GRB.OPTIMAL:
        print(
            f"[INFO] Non-optimal terminate ({status_name}); saving incumbent solution.",
            flush=True,
        )

    model.dispose()
    env.dispose()
    return result


def _project_simplex_scaled(z, cap):
    scalar = z.ndim == 1
    if scalar:
        z = z[None, :]
    z_nn = np.maximum(z, 0.0)
    sums = z_nn.sum(axis=1)
    if np.all(sums <= cap):
        return z_nn[0] if scalar else z_nn
    u = np.sort(z_nn, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    rho = np.sum(u * np.arange(1, z.shape[1] + 1) > (cssv - cap), axis=1) - 1
    theta = (cssv[np.arange(len(z)), rho] - cap) / (rho + 1.0)
    q = np.maximum(z_nn - theta[:, None], 0.0)
    q[sums <= cap] = z_nn[sums <= cap]
    return q[0] if scalar else q


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R * 2 + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, R),
        )
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))

    @torch.no_grad()
    def get_Q2_batch(self, xi1_np):
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        xi1_t = torch.tensor(xi1_np, dtype=torch.float32)
        q1r_t = torch.tensor(q1r, dtype=torch.float32).unsqueeze(0).expand(len(xi1_np), -1)
        nn_in = torch.cat([xi1_t, q1r_t, torch.full((len(xi1_np), 1), q10)], dim=1)
        return _project_simplex_scaled(self.fc_layers(nn_in).numpy(), q10), q10, q1r


def load_model(path):
    m = Model()
    ck = torch.load(path, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"], strict=False)
    m.eval()
    return m


def eval_neural_on_cond(model, xi1, xi2):
    chunk = 500
    total = 0.0
    q10, q1r = None, None
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        xi1_chunk = xi1[i0:i1]
        xi2_chunk = xi2[i0:i1]
        q2_chunk, q10, q1r = model.get_Q2_batch(xi1_chunk)
        q2_exp = q2_chunk[:, None, :]
        q1r_exp = q1r[None, None, :]
        ship = f_ship * (q2_exp > 1e-6).sum(axis=2)
        term1 = b * (xi1_chunk[:, None, :] + xi2_chunk - q1r_exp - q2_exp) - h * q2_exp
        term2 = h * (q1r_exp - xi1_chunk[:, None, :] - xi2_chunk)
        term3 = b * (xi1_chunk[:, None, :] - q1r_exp) - h * xi2_chunk
        v = np.maximum(term1, np.maximum(term2, term3))
        first_stage = c * (q10 + q1r.sum()) + h * q10
        total += (first_stage + ship + v.sum(axis=2)).sum()
    return float(total / (N * M))


def main():
    print(
        f"Shipping conditional clairvoyant exact MIP "
        f"(R={R}, N={N}, M={M}, seed={SEED})"
    )
    t_wall = time.time()
    xi1, xi2 = generate_conditional_data()
    result = solve_exact_shipping_mip(xi1, xi2)

    print(f"\nExact clairvoyant MIP: {result['saa_cond_mip_1m_shipping_exact']:.6f}")
    print(f"Q10={result['Q10']:.4f}, Q1r={np.round(result['Q1r'], 4)}")

    neural_costs = {}
    for seed in range(10):
        path = os.path.join(BASE_DIR, f"seed_{seed}_nominal.pth")
        if not os.path.exists(path):
            continue
        cost = eval_neural_on_cond(load_model(path), xi1, xi2)
        neural_costs[str(seed)] = float(cost)
        print(f"  nominal seed={seed}: hard-shipping cost={cost:.6f}")

    if neural_costs:
        best_seed = min(neural_costs, key=neural_costs.get)
        result["best_nominal_neural_seed"] = int(best_seed)
        result["best_nominal_neural_cost"] = neural_costs[best_seed]
        result["gap_best_nominal_neural_minus_exact_mip"] = (
            neural_costs[best_seed] - result["saa_cond_mip_1m_shipping_exact"]
        )
        result["nominal_neural_costs"] = neural_costs

    result["total_time_s"] = time.time() - t_wall
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
