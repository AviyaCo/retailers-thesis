"""Select SRO-FA radii on independent validation draws.

The retained SRO-FA JSON files omit partition boundaries and partition-specific
recourse decisions. This script therefore reconstructs each policy from its
training sample, solves it again, and evaluates it on the same seed-specific
validation draws used by the Neural CCG selector. The seed-9999 test draw is
generated only after all ten radius choices for a configuration are frozen.
"""

import argparse
import json
import os
import platform
import socket
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.stats import truncnorm


ROOT = Path("/data/home/aviyac")
SEEDS = list(range(10))
VALIDATION_FRACTION = 0.25
VALIDATION_SEED_BASE = 88880
TEST_SIZE = 10000
TEST_SEED = 9999
C, H, B = 0.25, 0.05, 0.5
SHIPPING_CHARGE = 0.1
SHIPPING_THRESHOLD = 1e-10
SHIPPING_TIME_LIMIT_S = 10_800
POLICY_MATCH_TOLERANCE = 1e-5

EXPERIMENTS = {
    "R3": {
        "R": 3,
        "shipping": False,
        "neural_selection": ROOT / "retailers_problem/R3/rho_validation_results_net16_v2.json",
        "output": ROOT / "retailers_problem/R3/sro_fa_rho_validation_results.json",
        "sro_results": {
            n: ROOT / f"retailers_problem/R3/N{n}/sro_fa/sro_fa_results_N{n}_R3.json"
            for n in (50, 200, 800)
        },
    },
    "R10": {
        "R": 10,
        "shipping": False,
        "neural_selection": ROOT / "retailers_problem/R10/rho_validation_results_net16_v2.json",
        "output": ROOT / "retailers_problem/R10/sro_fa_rho_validation_results.json",
        "sro_results": {
            n: ROOT / f"retailers_problem/R10/sro_fa/sro_fa_results_N{n}_R10.json"
            for n in (50, 200, 800)
        },
    },
    "shipping": {
        "R": 3,
        "shipping": True,
        "neural_selection": ROOT / "retailers_with_shipping/rho_validation_results_net16_v2.json",
        "output": ROOT / "retailers_with_shipping/sro_fa_rho_validation_results.json",
        "sro_results": {
            n: ROOT / f"retailers_with_shipping/N{n}/sro_fa/sro_fa_shipping_results_N{n}_R3.json"
            for n in (50, 200, 800)
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(EXPERIMENTS),
        default=list(EXPERIMENTS),
    )
    parser.add_argument(
        "--n-train", nargs="+", type=int, choices=(50, 200, 800), default=[50, 200, 800]
    )
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json_dump(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary, path)


def generate_data(n_samples, seed, n_retailers):
    random_state = np.random.RandomState(seed)
    lower_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(
        lower_xi1,
        np.inf,
        loc=6.0,
        scale=2.5,
        size=(n_samples, n_retailers),
        random_state=random_state,
    )
    mu2 = 2 * (6.0 - xi1) ** 2
    lower_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(
        lower_xi2,
        np.inf,
        loc=mu2,
        scale=2.5,
        random_state=random_state,
    )
    return xi1, xi2


def build_partitions(xi1_train, delta):
    n_train, n_retailers = xi1_train.shape
    balls_lo = np.maximum(0.0, xi1_train - delta)
    balls_hi = xi1_train + delta
    partitions = [(np.zeros(n_retailers), np.full(n_retailers, np.inf))]

    repeat = 0
    while len(partitions) < n_train and repeat < 2:
        for retailer in range(n_retailers):
            exponent = 1.0 / (n_retailers - retailer)
            new_partitions = list(partitions)
            for _ in range(len(partitions)):
                part_lo, part_hi = new_partitions.pop(0)
                inside = np.all(xi1_train >= part_lo, axis=1) & np.all(
                    xi1_train <= part_hi, axis=1
                )
                values = sorted(xi1_train[inside, retailer])
                count = len(values)
                if count == 0:
                    continue

                cuts = max(int(count**exponent), 1)
                points_per_cut = max(int(count / cuts), 1)
                if retailer == n_retailers - 1 or repeat > 0:
                    points_per_cut = 1
                new_cut_count = int(np.ceil(count / points_per_cut))

                last_hi = part_lo[retailer]
                for cut in range(new_cut_count):
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
        intersects = np.all(part_lo <= balls_hi[sample], axis=1) & np.all(
            part_hi >= balls_lo[sample], axis=1
        )
        partition_map.append(list(np.flatnonzero(intersects)))
    return part_lo, part_hi, partition_map


def solve_sro_fa(xi1_train, xi2_train, delta, part_lo, part_hi, partition_map, shipping):
    n_train, n_retailers = xi1_train.shape
    partition_count = len(part_lo)
    xi1_lo = np.maximum(0.0, xi1_train - delta)
    xi1_hi = xi1_train + delta
    xi2_lo = np.maximum(0.0, xi2_train - delta)
    xi2_hi = xi2_train + delta
    finite_part_hi = np.where(np.isinf(part_hi), 1e9, part_hi)

    environment = gp.Env(empty=True)
    environment.setParam("OutputFlag", 0)
    environment.start()
    model = gp.Model(env=environment)
    model.setParam("Threads", 1)
    if shipping:
        model.setParam("MIPGap", 0.01)
        model.setParam("TimeLimit", SHIPPING_TIME_LIMIT_S)
        model.setParam("Presolve", 1)
        model.setParam("Method", 3)
        model.setParam("Crossover", 0)

    q10 = model.addVar(lb=0.0, name="Q10")
    q1r = model.addVars(n_retailers, lb=0.0, name="Q1r")
    q2 = model.addVars(n_retailers, partition_count, lb=0.0, name="Q2")
    robust_cost = model.addVars(n_train, lb=-GRB.INFINITY, name="obj")
    activation = None
    if shipping:
        activation = model.addVars(
            n_retailers, partition_count, vtype=GRB.BINARY, name="z_ship"
        )

    for partition in range(partition_count):
        model.addConstr(
            gp.quicksum(q2[retailer, partition] for retailer in range(n_retailers)) <= q10
        )
        if shipping:
            for retailer in range(n_retailers):
                model.addConstr(q2[retailer, partition] <= 1000.0 * activation[retailer, partition])

    for sample in range(n_train):
        for partition in partition_map[sample]:
            cell_lo = np.maximum(xi1_lo[sample], part_lo[partition])
            cell_hi = np.minimum(xi1_hi[sample], finite_part_hi[partition])
            retailer_costs = []
            for retailer in range(n_retailers):
                cost = model.addVar(lb=-GRB.INFINITY)
                model.addConstr(
                    cost
                    >= B
                    * (
                        xi2_hi[sample, retailer]
                        + cell_hi[retailer]
                        - q2[retailer, partition]
                        - q1r[retailer]
                    )
                    - H * q2[retailer, partition]
                )
                model.addConstr(
                    cost
                    >= H
                    * (q1r[retailer] - xi2_lo[sample, retailer] - cell_lo[retailer])
                )
                model.addConstr(
                    cost
                    >= B * (cell_hi[retailer] - q1r[retailer])
                    - H * xi2_lo[sample, retailer]
                )
                retailer_costs.append(cost)
            fixed_charge = 0.0
            if shipping:
                fixed_charge = SHIPPING_CHARGE * gp.quicksum(
                    activation[retailer, partition] for retailer in range(n_retailers)
                )
            model.addConstr(robust_cost[sample] >= gp.quicksum(retailer_costs) + fixed_charge)

    model.setObjective(
        C * (q10 + gp.quicksum(q1r[retailer] for retailer in range(n_retailers)))
        + H * q10
        + gp.quicksum(robust_cost[sample] for sample in range(n_train)) / n_train,
        GRB.MINIMIZE,
    )

    solve_started = time.perf_counter()
    model.optimize()
    solve_time = time.perf_counter() - solve_started
    status = int(model.Status)
    status_name = "OPTIMAL" if status == GRB.OPTIMAL else f"STATUS_{status}"
    solution_count = int(model.SolCount)
    if status != GRB.OPTIMAL or solution_count <= 0:
        model.dispose()
        environment.dispose()
        raise RuntimeError(
            f"SRO-FA solve failed: status={status_name}, sol_count={solution_count}"
        )

    q10_value = float(q10.X)
    q1r_value = np.asarray([q1r[retailer].X for retailer in range(n_retailers)])
    q2_value = np.asarray(
        [
            [q2[retailer, partition].X for partition in range(partition_count)]
            for retailer in range(n_retailers)
        ]
    )
    solve_record = {
        "status": status,
        "status_name": status_name,
        "objective": float(model.ObjVal),
        "objective_bound": float(model.ObjBound),
        "relative_gap": float(model.MIPGap) if shipping else 0.0,
        "solution_count": solution_count,
        "node_count": float(model.NodeCount),
        "gurobi_runtime_s": float(model.Runtime),
        "measured_solve_time_s": solve_time,
        "threads": 1,
    }
    model.dispose()
    environment.dispose()
    return q10_value, q1r_value, q2_value, solve_record


def evaluate_policy(xi1, xi2, q10, q1r, q2, part_lo, part_hi, shipping):
    membership = np.all(
        (xi1[:, None, :] >= part_lo[None, :, :])
        & (xi1[:, None, :] <= part_hi[None, :, :]),
        axis=2,
    )
    if not np.all(membership.any(axis=1)):
        raise RuntimeError("At least one evaluation point was not assigned to a partition")
    assigned = np.argmax(membership, axis=1)
    q2_scenarios = q2[:, assigned].T
    first_stage = C * (q10 + q1r.sum()) + H * q10
    recourse = np.maximum(
        B * (xi1 + xi2 - q1r - q2_scenarios) - H * q2_scenarios,
        np.maximum(
            H * (q1r - xi1 - xi2),
            B * (xi1 - q1r) - H * xi2,
        ),
    )
    costs = first_stage + recourse.sum(axis=1)
    if shipping:
        costs += SHIPPING_CHARGE * (q2_scenarios > SHIPPING_THRESHOLD).sum(axis=1)
    return float(costs.mean())


def save_policy(path, rho, q10, q1r, q2, part_lo, part_hi):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as stream:
        np.savez_compressed(
            stream,
            rho=np.asarray(rho),
            q10=np.asarray(q10),
            q1r=q1r,
            q2=q2,
            part_lo=part_lo,
            part_hi=part_hi,
        )
    os.replace(temporary, path)


def load_policy(path):
    with np.load(path) as policy:
        return {key: policy[key] for key in policy.files}


def initial_results(experiment_name, experiment):
    return {
        "meta": {
            "experiment": experiment_name,
            "protocol": "per_seed_independent_validation_sro_fa_rho_selection_v1",
            "validation_fraction": VALIDATION_FRACTION,
            "validation_size_rule": "ceil(validation_fraction * N_train)",
            "validation_seed_base": VALIDATION_SEED_BASE,
            "training_seeds": SEEDS,
            "candidate_rule": "same common positive rho candidates as Neural CCG, rho <= 1.5",
            "test_protocol": "seed-9999 10k draw generated only after all ten choices for N are frozen",
            "retraining_after_selection": False,
            "policy_reconstruction": "re-solve from original seed-specific training sample because historical JSON omits Q2 and partitions",
            "shipping_activation_threshold": SHIPPING_THRESHOLD if experiment["shipping"] else None,
            "solver": "Gurobi",
            "gurobi_version": list(gp.gurobi.version()),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "by_N": {},
        "runtime": {"status": "not_started", "invocations": []},
    }


def load_or_initialize(experiment_name, experiment):
    output = experiment["output"]
    if not output.exists():
        return initial_results(experiment_name, experiment)
    with open(output) as stream:
        results = json.load(stream)
    expected = "per_seed_independent_validation_sro_fa_rho_selection_v1"
    if results.get("meta", {}).get("protocol") != expected:
        raise RuntimeError(f"Refusing to resume incompatible artifact {output}")
    return results


def candidate_rhos(experiment, n_train, historical):
    with open(experiment["neural_selection"]) as stream:
        neural_selection = json.load(stream)
    candidates = [float(rho) for rho in neural_selection["by_N"][str(n_train)]["candidate_rhos"]]
    if experiment["shipping"]:
        return candidates
    complete = []
    for rho in candidates:
        seed_results = historical.get(str(float(rho)), {})
        if all(str(seed) in seed_results for seed in SEEDS):
            complete.append(rho)
    if complete != candidates:
        missing = sorted(set(candidates) - set(complete))
        raise RuntimeError(f"Historical SRO artifact lacks complete candidates: {missing}")
    return candidates


def compare_historical_policy(historical_entry, q10, q1r, objective, shipping):
    if historical_entry is None:
        return {
            "available": False,
            "reason": "candidate absent from historical SRO-FA sweep",
        }
    comparison = {
        "available": True,
        "q10_absolute_difference": abs(q10 - float(historical_entry["Q10"])),
        "q1r_max_absolute_difference": float(
            np.max(np.abs(q1r - np.asarray(historical_entry["Q1r"])))
        ),
        "objective_absolute_difference": abs(objective - float(historical_entry["is"])),
    }
    if not shipping and max(comparison.values()) > POLICY_MATCH_TOLERANCE:
        raise RuntimeError(f"Reconstructed LP does not match historical artifact: {comparison}")
    return comparison


def run_training_size(experiment_name, experiment, n_train, results):
    with open(experiment["sro_results"][n_train]) as stream:
        historical = json.load(stream)
    candidates = candidate_rhos(experiment, n_train, historical)
    n_validation = int(np.ceil(VALIDATION_FRACTION * n_train))
    n_key = str(n_train)
    n_results = results["by_N"].setdefault(
        n_key,
        {
            "candidate_rhos": candidates,
            "N_val": n_validation,
            "selection_status": "running",
            "per_seed": {},
        },
    )
    if n_results["candidate_rhos"] != candidates:
        raise RuntimeError(f"Candidate grid changed for {experiment_name}, N={n_train}")

    policies_dir = experiment["output"].parent / "sro_fa_validation_policies" / f"N{n_train}"
    validation_phase_started = time.perf_counter()
    for seed in SEEDS:
        seed_key = str(seed)
        seed_results = n_results["per_seed"].setdefault(
            seed_key,
            {
                "validation_seed": VALIDATION_SEED_BASE + seed,
                "validation_candidates": {},
            },
        )
        xi1_train, xi2_train = generate_data(n_train, seed, experiment["R"])
        xi1_validation, xi2_validation = generate_data(
            n_validation, VALIDATION_SEED_BASE + seed, experiment["R"]
        )
        policy_path = policies_dir / f"seed_{seed}_selected_policy.npz"

        for rho in candidates:
            rho_key = str(float(rho))
            if rho_key in seed_results["validation_candidates"]:
                continue
            candidate_started = time.perf_counter()
            partition_started = time.perf_counter()
            part_lo, part_hi, partition_map = build_partitions(xi1_train, rho)
            partition_time = time.perf_counter() - partition_started
            q10, q1r, q2, solve_record = solve_sro_fa(
                xi1_train,
                xi2_train,
                rho,
                part_lo,
                part_hi,
                partition_map,
                experiment["shipping"],
            )
            validation_started = time.perf_counter()
            validation_cost = evaluate_policy(
                xi1_validation,
                xi2_validation,
                q10,
                q1r,
                q2,
                part_lo,
                part_hi,
                experiment["shipping"],
            )
            validation_time = time.perf_counter() - validation_started
            historical_entry = historical.get(rho_key, {}).get(seed_key)
            comparison = compare_historical_policy(
                historical_entry,
                q10,
                q1r,
                solve_record["objective"],
                experiment["shipping"],
            )

            existing_costs = {
                float(existing_rho): entry["validation_cost"]
                for existing_rho, entry in seed_results["validation_candidates"].items()
            }
            if not existing_costs or (validation_cost, rho) < min(
                (cost, existing_rho) for existing_rho, cost in existing_costs.items()
            ):
                save_policy(policy_path, rho, q10, q1r, q2, part_lo, part_hi)

            seed_results["validation_candidates"][rho_key] = {
                "validation_cost": validation_cost,
                "partition_count": len(part_lo),
                "solve": solve_record,
                "historical_comparison": comparison,
                "runtime": {
                    "partition_construction_s": partition_time,
                    "validation_evaluation_s": validation_time,
                    "candidate_total_s": time.perf_counter() - candidate_started,
                },
            }
            atomic_json_dump(experiment["output"], results)
            print(
                f"{experiment_name} N={n_train} seed={seed} rho={rho:g}: "
                f"validation={validation_cost:.6f} solve={solve_record['measured_solve_time_s']:.3f}s",
                flush=True,
            )

        validation_costs = {
            float(rho): entry["validation_cost"]
            for rho, entry in seed_results["validation_candidates"].items()
        }
        selected_rho = min(validation_costs, key=lambda rho: (validation_costs[rho], rho))
        policy = load_policy(policy_path)
        if not np.isclose(float(policy["rho"]), selected_rho):
            raise RuntimeError(f"Selected policy file is stale for {experiment_name}, N={n_train}, seed={seed}")
        seed_results["selected_rho"] = selected_rho
        seed_results["selected_validation_cost"] = validation_costs[selected_rho]
        seed_results["selected_policy_path"] = str(policy_path.relative_to(ROOT))
        atomic_json_dump(experiment["output"], results)

    n_results["validation_selection_time_current_invocation_s"] = (
        time.perf_counter() - validation_phase_started
    )
    n_results["selection_status"] = "frozen_before_test"
    atomic_json_dump(experiment["output"], results)

    test_generation_started = time.perf_counter()
    xi1_test, xi2_test = generate_data(TEST_SIZE, TEST_SEED, experiment["R"])
    n_results["test_generation_time_s"] = time.perf_counter() - test_generation_started
    test_phase_started = time.perf_counter()
    selected_oos = []
    selected_rhos = []
    for seed in SEEDS:
        seed_results = n_results["per_seed"][str(seed)]
        selected_rho = float(seed_results["selected_rho"])
        policy = load_policy(ROOT / seed_results["selected_policy_path"])
        evaluation_started = time.perf_counter()
        test_cost = evaluate_policy(
            xi1_test,
            xi2_test,
            float(policy["q10"]),
            policy["q1r"],
            policy["q2"],
            policy["part_lo"],
            policy["part_hi"],
            experiment["shipping"],
        )
        seed_results["selected_10k_oos"] = test_cost
        seed_results["test_evaluation_time_s"] = time.perf_counter() - evaluation_started
        selected_oos.append(test_cost)
        selected_rhos.append(selected_rho)

    counts = Counter(selected_rhos)
    n_results["selected_rhos"] = selected_rhos
    n_results["selected_rho_counts"] = {
        str(rho): counts[rho] for rho in sorted(counts)
    }
    n_results["selected_10k_oos_mean"] = float(np.mean(selected_oos))
    n_results["selected_10k_oos_std"] = float(np.std(selected_oos, ddof=1))
    n_results["test_evaluation_time_s"] = time.perf_counter() - test_phase_started
    n_results["selection_status"] = "completed"
    atomic_json_dump(experiment["output"], results)


def main():
    args = parse_args()
    os.environ["GRB_LICENSE_FILE"] = str(ROOT / "gurobi.lic")
    for experiment_name in args.experiments:
        experiment = EXPERIMENTS[experiment_name]
        results = load_or_initialize(experiment_name, experiment)
        invocation = {
            "start_utc": utc_now(),
            "end_utc": None,
            "status": "running",
            "n_train": args.n_train,
        }
        results["runtime"]["status"] = "running"
        results["runtime"]["invocations"].append(invocation)
        atomic_json_dump(experiment["output"], results)
        invocation_started = time.perf_counter()
        try:
            for n_train in args.n_train:
                run_training_size(experiment_name, experiment, n_train, results)
        except BaseException:
            invocation["end_utc"] = utc_now()
            invocation["status"] = "failed_or_interrupted"
            invocation["elapsed_s"] = time.perf_counter() - invocation_started
            results["runtime"]["status"] = "incomplete"
            atomic_json_dump(experiment["output"], results)
            raise
        invocation["end_utc"] = utc_now()
        invocation["status"] = "completed"
        invocation["elapsed_s"] = time.perf_counter() - invocation_started
        results["runtime"]["status"] = "completed"
        atomic_json_dump(experiment["output"], results)
        print(f"Saved {experiment['output']}", flush=True)


if __name__ == "__main__":
    main()