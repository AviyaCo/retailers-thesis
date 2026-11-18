"""SRO-FA sweep for the R=10 three-stage retailers problem.

This is the R=10 counterpart of the existing R=3 SRO-FA baseline. Results are
saved incrementally, so interrupted runs can be resumed without recomputation.
"""

import argparse
import json
import os
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.stats import truncnorm


R = 10
N_TEST = 10000
SEED_TEST = 9999
C, H, B = 0.25, 0.05, 0.5
DEFAULT_RHOS = [
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2,
    0.3, 0.4, 0.45, 0.5, 0.6, 0.75, 0.8, 1.0, 1.5,
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, choices=(50, 200, 800), default=50)
    parser.add_argument("--rhos", nargs="*", type=float, default=DEFAULT_RHOS)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    return parser.parse_args()


def generate_data(n_samples, seed):
    np.random.seed(seed)
    lower_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(lower_xi1, np.inf, loc=6.0, scale=2.5,
                        size=(n_samples, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    lower_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(lower_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2


def build_partitions(xi1_train, delta):
    n_train = xi1_train.shape[0]
    balls_lo = np.maximum(0.0, xi1_train - delta)
    balls_hi = xi1_train + delta
    partitions = [(np.zeros(R), np.full(R, np.inf))]

    repeat = 0
    while len(partitions) < n_train and repeat < 2:
        for retailer in range(R):
            exponent = 1.0 / (R - retailer)
            new_partitions = list(partitions)
            for _ in range(len(partitions)):
                part_lo, part_hi = new_partitions.pop(0)
                inside = (np.all(xi1_train >= part_lo, axis=1)
                          & np.all(xi1_train <= part_hi, axis=1))
                values = sorted(xi1_train[inside, retailer])
                count = len(values)
                if count == 0:
                    continue

                cuts = max(int(count ** exponent), 1)
                points_per_cut = max(int(count / cuts), 1)
                if retailer == R - 1 or repeat > 0:
                    points_per_cut = 1
                new_cuts = int(np.ceil(count / points_per_cut))

                last_hi = part_lo[retailer]
                for cut in range(new_cuts):
                    split_lo = part_lo.copy()
                    split_hi = part_hi.copy()
                    if cut > 0:
                        split_lo[retailer] = last_hi
                    end = points_per_cut * (cut + 1)
                    if end < count:
                        split_hi[retailer] = (values[end - 1] + values[end]) / 2.0
                    last_hi = split_hi[retailer]
                    if split_hi[retailer] > split_lo[retailer]:
                        new_partitions.append((split_lo, split_hi))
            partitions = new_partitions
        repeat += 1

    if len(partitions) > n_train:
        raise RuntimeError(f"Too many partitions: K={len(partitions)} > N={n_train}")

    part_lo = np.asarray([part[0] for part in partitions])
    part_hi = np.asarray([part[1] for part in partitions])
    partition_map = []
    for sample in range(n_train):
        intersects = (np.all(part_lo <= balls_hi[sample], axis=1)
                      & np.all(part_hi >= balls_lo[sample], axis=1))
        partition_map.append(list(np.flatnonzero(intersects)))
    return partitions, partition_map


def solve_sro_fa(xi1_train, xi2_train, delta, partitions, partition_map):
    n_train = xi1_train.shape[0]
    partition_count = len(partitions)
    xi1_lo = np.maximum(0.0, xi1_train - delta)
    xi1_hi = xi1_train + delta
    xi2_lo = np.maximum(0.0, xi2_train - delta)
    xi2_hi = xi2_train + delta
    part_lo = np.asarray([part[0] for part in partitions])
    part_hi = np.asarray([part[1] for part in partitions])
    part_hi = np.where(np.isinf(part_hi), 1e9, part_hi)

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("Threads", 1)
    env.start()
    model = gp.Model(env=env)

    q10 = model.addVar(lb=0.0, name="Q10")
    q1r = model.addVars(R, lb=0.0, name="Q1r")
    q2 = model.addVars(R, partition_count, lb=0.0, name="Q2")
    robust_cost = model.addVars(n_train, lb=-GRB.INFINITY, name="obj")

    for partition in range(partition_count):
        model.addConstr(gp.quicksum(q2[retailer, partition]
                                    for retailer in range(R)) <= q10)

    for sample in range(n_train):
        for partition in partition_map[sample]:
            cell_lo = np.maximum(xi1_lo[sample], part_lo[partition])
            cell_hi = np.minimum(xi1_hi[sample], part_hi[partition])
            retailer_costs = []
            for retailer in range(R):
                cost = model.addVar(lb=-GRB.INFINITY)
                model.addConstr(
                    cost >= B * (xi2_hi[sample, retailer] + cell_hi[retailer]
                                 - q2[retailer, partition] - q1r[retailer])
                    - H * q2[retailer, partition])
                model.addConstr(
                    cost >= H * (q1r[retailer] - xi2_lo[sample, retailer]
                                 - cell_lo[retailer]))
                model.addConstr(
                    cost >= B * (cell_hi[retailer] - q1r[retailer])
                    - H * xi2_lo[sample, retailer])
                retailer_costs.append(cost)
            model.addConstr(robust_cost[sample] >= gp.quicksum(retailer_costs))

    model.setObjective(
        C * (q10 + gp.quicksum(q1r[retailer] for retailer in range(R)))
        + H * q10
        + gp.quicksum(robust_cost[sample] for sample in range(n_train)) / n_train,
        GRB.MINIMIZE,
    )

    solve_start = time.time()
    model.optimize()
    solve_time = time.time() - solve_start
    if model.status != GRB.OPTIMAL:
        status = model.status
        model.dispose()
        env.dispose()
        raise RuntimeError(f"Gurobi status {status}")

    q10_value = q10.X
    q1r_value = np.asarray([q1r[retailer].X for retailer in range(R)])
    q2_value = np.asarray([
        [q2[retailer, partition].X for partition in range(partition_count)]
        for retailer in range(R)
    ])
    objective = model.ObjVal
    model.dispose()
    env.dispose()
    return q10_value, q1r_value, q2_value, objective, solve_time


def evaluate(xi1_test, xi2_test, q10, q1r, q2, partitions):
    part_lo = np.asarray([part[0] for part in partitions])
    part_hi = np.asarray([part[1] for part in partitions])
    membership = np.ones((len(xi1_test), len(partitions)), dtype=bool)
    for retailer in range(R):
        membership &= (xi1_test[:, retailer, None] >= part_lo[None, :, retailer])
        membership &= (xi1_test[:, retailer, None] <= part_hi[None, :, retailer])
    if not np.all(membership.any(axis=1)):
        raise RuntimeError("At least one test point was not assigned to a partition")
    assigned = np.argmax(membership, axis=1)
    q2_test = q2[:, assigned].T

    stage1 = C * (q10 + q1r.sum()) + H * q10
    cost = np.maximum(
        B * (xi1_test + xi2_test - q1r - q2_test) - H * q2_test,
        np.maximum(
            H * (q1r - xi1_test - xi2_test),
            B * (xi1_test - q1r) - H * xi2_test,
        ),
    )
    return float(stage1 + cost.sum(axis=1).mean())


def main():
    args = parse_args()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_json = os.path.join(out_dir, f"sro_fa_results_N{args.n_train}_R10.json")
    os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"

    print(f"SRO-FA sweep: R={R}, N_train={args.n_train}, N_test={N_TEST}")
    print(f"Rho values ({len(args.rhos)}): {args.rhos}")
    print(f"Seeds ({len(args.seeds)}): {args.seeds}")
    print(f"Output: {out_json}", flush=True)

    xi1_test, xi2_test = generate_data(N_TEST, SEED_TEST)
    results = {}
    if os.path.exists(out_json):
        with open(out_json) as handle:
            results = json.load(handle)

    for rho in args.rhos:
        rho_key = str(rho)
        results.setdefault(rho_key, {})
        for seed in args.seeds:
            seed_key = str(seed)
            if seed_key in results[rho_key]:
                print(f"[Cache] rho={rho:g} seed={seed}")
                continue

            started = time.time()
            xi1_train, xi2_train = generate_data(args.n_train, seed)
            partition_started = time.time()
            partitions, partition_map = build_partitions(xi1_train, rho)
            partition_time = time.time() - partition_started
            q10, q1r, q2, objective, solve_time = solve_sro_fa(
                xi1_train, xi2_train, rho, partitions, partition_map)
            evaluation_started = time.time()
            oos = evaluate(xi1_test, xi2_test, q10, q1r, q2, partitions)
            evaluation_time = time.time() - evaluation_started
            total_time = time.time() - started

            print(
                f"rho={rho:g} seed={seed}: K={len(partitions)} "
                f"IS={objective:.4f} OOS={oos:.4f} "
                f"partition={partition_time:.2f}s LP={solve_time:.2f}s "
                f"eval={evaluation_time:.2f}s total={total_time:.2f}s",
                flush=True,
            )
            results[rho_key][seed_key] = {
                "is": float(objective),
                "oos": oos,
                "K": len(partitions),
                "Q10": float(q10),
                "Q1r": q1r.tolist(),
                "time_partition": partition_time,
                "time_lp": solve_time,
                "time_eval": evaluation_time,
                "time_total": total_time,
            }
            with open(out_json, "w") as handle:
                json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()