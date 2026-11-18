"""Evaluate the corrected fixed-shipping N=800 sweep on the canonical test set."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import truncnorm


BASE_DIR = Path("/data/home/aviyac/retailers_with_shipping/N800")
SWEEP_DIR = BASE_DIR / "rho_sweep_minibatch"
LEGACY_EVAL_PATH = BASE_DIR / "rho_sweep" / "sweep_eval_shipping_10k.json"
OUTPUT_PATH = SWEEP_DIR / "sweep_eval_shipping_10k.json"

R = 3
N_TRAIN = 800
N_TEST = 10_000
TEST_SEED = 9999
SEEDS = list(range(10))
COST_C = 0.25
COST_H = 0.05
COST_B = 0.5
SHIPPING_COST = 0.1
SHIPPING_THRESHOLD = 1e-6
EVAL_CHUNK_SIZE = 2_000
PROJECTION_METHOD = "gurobi_qp"
EVALUATOR_VERSION = "gurobi_qp_final_mp_is_v2"
KNN_SELECTION_VERSION = "shipping_per_seed_loo_grid_v1"

torch.set_num_threads(1)


def generate_data(n_samples, seed):
    np.random.seed(seed)
    lower_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(
        lower_xi1, np.inf, loc=6.0, scale=2.5, size=(n_samples, R)
    )
    mean_xi2 = 2 * (6.0 - xi1) ** 2
    lower_xi2 = (0.0 - mean_xi2) / 2.5
    xi2 = truncnorm.rvs(lower_xi2, np.inf, loc=mean_xi2, scale=2.5)
    return xi1, xi2


_gurobi_environment = None


def get_gurobi_environment():
    global _gurobi_environment
    if _gurobi_environment is None:
        _gurobi_environment = gp.Env(empty=True)
        _gurobi_environment.setParam("OutputFlag", 0)
        _gurobi_environment.start()
    return _gurobi_environment


def project_with_gurobi(raw_values, capacity):
    projected = []
    for raw_row in np.asarray(raw_values, dtype=float):
        model = gp.Model(env=get_gurobi_environment())
        model.setParam("OutputFlag", 0)
        q2 = model.addVars(R, lb=0.0)
        model.addConstr(gp.quicksum(q2[index] for index in range(R)) <= capacity)
        model.setObjective(
            gp.quicksum(
                (q2[index] - float(raw_row[index])) ** 2 for index in range(R)
            ),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.status != GRB.OPTIMAL:
            status = model.status
            model.dispose()
            raise RuntimeError(f"Gurobi projection failed with status {status}")
        projected.append([q2[index].X for index in range(R)])
        model.dispose()
    return np.asarray(projected)


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
    def decisions(self, xi1):
        xi1_tensor = torch.as_tensor(xi1, dtype=torch.float32)
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).cpu().numpy()
        network_input = torch.cat(
            [
                xi1_tensor,
                torch.as_tensor(q1r, dtype=torch.float32)
                .unsqueeze(0)
                .expand(len(xi1_tensor), -1),
                torch.full((len(xi1_tensor), 1), q10),
            ],
            dim=1,
        )
        raw_q2 = self.fc_layers(network_input).cpu().numpy()
        return project_with_gurobi(raw_q2, q10), q10, q1r


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Model()
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_unexpected = {"proj_layer.q", "proj_layer.A"}
    if incompatible.missing_keys or not set(incompatible.unexpected_keys).issubset(
        allowed_unexpected
    ):
        raise RuntimeError(
            f"Incompatible checkpoint {checkpoint_path}: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model


def scenario_costs(q10, q1r, q2, xi1, xi2):
    shipping = SHIPPING_COST * (q2 > SHIPPING_THRESHOLD).sum(axis=1)
    recourse = np.maximum(
        COST_B * (xi1 + xi2 - q1r - q2) - COST_H * q2,
        np.maximum(
            COST_H * (q1r - xi1 - xi2),
            COST_B * (xi1 - q1r) - COST_H * xi2,
        ),
    )
    stage_one = COST_C * (q10 + q1r.sum()) + COST_H * q10
    return stage_one + shipping + recourse.sum(axis=1)


def evaluate_mean(model, xi1, xi2):
    total = 0.0
    count = 0
    for start in range(0, len(xi1), EVAL_CHUNK_SIZE):
        stop = min(start + EVAL_CHUNK_SIZE, len(xi1))
        q2, q10, q1r = model.decisions(xi1[start:stop])
        costs = scenario_costs(q10, q1r, q2, xi1[start:stop], xi2[start:stop])
        total += float(costs.sum())
        count += len(costs)
    return total / count


def exact_augmented_is(model, summary_path, summary, xi1, xi2):
    adversarial_xi1 = {}
    adversarial_xi2 = {}
    incumbent_iteration = int(summary.get("final_mp_iter", 0))
    for iteration in range(1, incumbent_iteration + 1):
        ap_path = summary_path.parent / f"seed_{summary['seed']}_ap_iter{iteration}.json"
        with ap_path.open() as stream:
            scenarios = json.load(stream)
        ball_ids = {entry["ball_id"] for entry in scenarios}
        if len(scenarios) != N_TRAIN or ball_ids != set(range(N_TRAIN)):
            raise RuntimeError(f"Incomplete AP file: {ap_path}")
        for entry in scenarios:
            ball_id = entry["ball_id"]
            adversarial_xi1.setdefault(ball_id, []).append(entry["worst_xi1"])
            adversarial_xi2.setdefault(ball_id, []).append(entry["worst_xi2"])

    columns_xi1 = [xi1]
    columns_xi2 = [xi2]
    if adversarial_xi1:
        column_count = max(len(values) for values in adversarial_xi1.values())
        columns_xi1.extend(xi1.copy() for _ in range(column_count))
        columns_xi2.extend(xi2.copy() for _ in range(column_count))
        for ball_id, values in adversarial_xi1.items():
            for column, value in enumerate(values, start=1):
                columns_xi1[column][ball_id] = value
                columns_xi2[column][ball_id] = adversarial_xi2[ball_id][column - 1]

    per_column_costs = []
    for column_xi1, column_xi2 in zip(columns_xi1, columns_xi2):
        q2, q10, q1r = model.decisions(column_xi1)
        per_column_costs.append(
            scenario_costs(q10, q1r, q2, column_xi1, column_xi2)
        )
    return float(np.stack(per_column_costs, axis=1).max(axis=1).mean())


def checkpoint_for_summary(summary_path, summary):
    iteration = int(summary.get("final_mp_iter", 0))
    if iteration <= 0:
        return BASE_DIR / f"seed_{summary['seed']}_nominal.pth"
    return summary_path.parent / f"seed_{summary['seed']}_mp_iter{iteration}.pth"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with LEGACY_EVAL_PATH.open() as stream:
        legacy_evaluation = json.load(stream)
    benchmarks = dict(legacy_evaluation["benchmarks"])
    if (
        legacy_evaluation.get("meta", {}).get("N_test") != N_TEST
        or legacy_evaluation.get("meta", {}).get("seed_test") != TEST_SEED
        or benchmarks.get("_knn_k_selection") != KNN_SELECTION_VERSION
        or len(benchmarks.get("saa_knn", {}).get("selected_k_by_seed", [])) != len(SEEDS)
        or not all(
            len(benchmarks.get("saa_knn", {}).get(field, [])) == len(SEEDS)
            and np.isfinite(benchmarks["saa_knn"][field]).all()
            for field in ("is", "val", "oos")
        )
        or len(benchmarks.get("saa_milp", {}).get("is", [])) != len(SEEDS)
        or not np.isfinite(benchmarks["saa_milp"]["is"]).all()
    ):
        raise RuntimeError(f"Incomplete or stale benchmark artifact: {LEGACY_EVAL_PATH}")

    xi1_test, xi2_test = generate_data(N_TEST, TEST_SEED)
    saa_nn_is = []
    saa_nn_oos = []
    benchmark_started = time.perf_counter()
    for seed in SEEDS:
        model = load_model(BASE_DIR / f"seed_{seed}_nominal.pth")
        xi1_train, xi2_train = generate_data(N_TRAIN, seed)
        saa_nn_is.append(evaluate_mean(model, xi1_train, xi2_train))
        saa_nn_oos.append(evaluate_mean(model, xi1_test, xi2_test))
    benchmarks["saa_nn"] = {"is": saa_nn_is, "oos": saa_nn_oos}
    benchmarks["_saa_nn_projection"] = PROJECTION_METHOD
    benchmark_evaluation_time_s = time.perf_counter() - benchmark_started
    sweep = {}
    checkpoint_provenance = {}
    per_rho_time = {}

    rho_directories = sorted(
        (path for path in SWEEP_DIR.glob("rho_*") if path.is_dir()),
        key=lambda path: float(path.name[4:].replace("p", ".")),
    )
    for rho_directory in rho_directories:
        rho_started = time.perf_counter()
        rho = float(rho_directory.name[4:].replace("p", "."))
        robust_is = []
        robust_oos = []
        rho_checkpoints = {}
        for seed in SEEDS:
            summary_path = rho_directory / f"seed_{seed}_summary.json"
            with summary_path.open() as stream:
                summary = json.load(stream)
            if summary.get("completed") is not True:
                raise RuntimeError(f"Incomplete summary: {summary_path}")

            checkpoint_path = checkpoint_for_summary(summary_path, summary)
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            model = load_model(checkpoint_path)
            xi1_train, xi2_train = generate_data(N_TRAIN, seed)
            robust_is.append(
                exact_augmented_is(model, summary_path, summary, xi1_train, xi2_train)
            )
            robust_oos.append(evaluate_mean(model, xi1_test, xi2_test))
            rho_checkpoints[str(seed)] = {
                "final_mp_iter": int(summary.get("final_mp_iter", 0)),
                "best_ap_iter": summary.get("best_ap_iter"),
                "checkpoint": os.path.relpath(checkpoint_path, BASE_DIR.parent),
            }
            print(
                f"rho={rho:g} seed={seed}: IS={robust_is[-1]:.6f} "
                f"OOS={robust_oos[-1]:.6f}",
                flush=True,
            )

        sweep[str(rho)] = {
            "rob_is": robust_is,
            "rob_oos": robust_oos,
            "n_seeds": len(SEEDS),
            "projection_method": PROJECTION_METHOD,
            "evaluator_version": EVALUATOR_VERSION,
            "evaluation_time_s": time.perf_counter() - rho_started,
        }
        checkpoint_provenance[str(rho)] = rho_checkpoints
        per_rho_time[str(rho)] = time.perf_counter() - rho_started

    result = {
        "sweep": sweep,
        "benchmarks": benchmarks,
        "meta": {
            "experiment": "fixed_shipping_N800_corrected_minibatch",
            "N_train": N_TRAIN,
            "N_test": N_TEST,
            "seed_test": TEST_SEED,
            "seeds": SEEDS,
            "shipping_evaluation": "hard charge 0.1 when Q2 > 1e-6",
            "incumbent_rule": "minimum AP mean; AP iteration t evaluates MP iteration t-1",
            "sweep_source": os.path.relpath(SWEEP_DIR, BASE_DIR.parent),
            "benchmark_source": os.path.relpath(LEGACY_EVAL_PATH, BASE_DIR.parent),
            "benchmark_source_sha256": file_sha256(LEGACY_EVAL_PATH),
            "benchmark_reuse_reason": "projection-independent SAA-kNN and SAA-MILP values are reused; SAA-NN is recomputed with Gurobi",
            "projection_method": PROJECTION_METHOD,
            "evaluator_version": EVALUATOR_VERSION,
        },
        "checkpoint_provenance": checkpoint_provenance,
        "runtime": {
            "started_at_utc": started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "per_rho_time_s": per_rho_time,
            "saa_nn_benchmark_evaluation_time_s": benchmark_evaluation_time_s,
            "total_time_s": time.perf_counter() - started,
            "status": "completed",
        },
    }
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w") as stream:
        json.dump(result, stream, indent=2)
    temporary_path.replace(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()