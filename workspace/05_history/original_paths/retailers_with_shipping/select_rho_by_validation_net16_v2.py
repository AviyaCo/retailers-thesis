"""Select fixed-shipping net16 radii by independent validation."""

import json
import sys as _sys
_sys.path.insert(0, '/data/home/aviyac/retailers_experiment')
from src.config import RADIUS_SELECTION_RULE, select_radius  # shared radius rule, 2026-11-18
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import gurobipy as gp
import torch
import torch.nn as nn
from gurobipy import GRB
from scipy.stats import truncnorm


BASE_DIR = Path("/data/home/aviyac/retailers_with_shipping")
OUTPUT_PATH = BASE_DIR / "rho_validation_results_net16_v2.json"
SEEDS = list(range(10))
VALIDATION_FRACTION = 0.25
VALIDATION_SEED_BASE = 88880
R = 3
COST_C = 0.25
COST_H = 0.05
COST_B = 0.5
SHIPPING_COST = 0.1
SHIPPING_THRESHOLD = 1e-6
PROJECTION_METHOD = "gurobi_qp"
EVALUATOR_VERSION = "gurobi_qp_final_mp_is_v2"

CONFIGS = {
    50: {
        "nominal_dir": BASE_DIR / "N50",
        "sweep_dir": BASE_DIR / "N50" / "rho_sweep",
        "eval_json": BASE_DIR / "N50" / "rho_sweep" / "sweep_eval_shipping_10k.json",
        "training_protocol": "historical full-batch MP",
    },
    200: {
        "nominal_dir": BASE_DIR / "N200",
        "sweep_dir": BASE_DIR / "N200" / "rho_sweep",
        "eval_json": BASE_DIR / "N200" / "rho_sweep" / "sweep_eval_shipping_10k.json",
        "training_protocol": "historical full-batch MP",
    },
    800: {
        "nominal_dir": BASE_DIR / "N800",
        "sweep_dir": BASE_DIR / "N800" / "rho_sweep_minibatch",
        "eval_json": BASE_DIR / "N800" / "rho_sweep_minibatch" / "sweep_eval_shipping_10k.json",
        "training_protocol": "corrected deterministic minibatch-200 MP",
    },
}

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


def validation_size(n_train):
    return int(np.ceil(VALIDATION_FRACTION * n_train))


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
        raise RuntimeError(f"Incompatible checkpoint: {checkpoint_path}")
    model.eval()
    return model


def mean_cost(model, xi1, xi2):
    q2, q10, q1r = model.decisions(xi1)
    shipping = SHIPPING_COST * (q2 > SHIPPING_THRESHOLD).sum(axis=1)
    recourse = np.maximum(
        COST_B * (xi1 + xi2 - q1r - q2) - COST_H * q2,
        np.maximum(
            COST_H * (q1r - xi1 - xi2),
            COST_B * (xi1 - q1r) - COST_H * xi2,
        ),
    )
    stage_one = COST_C * (q10 + q1r.sum()) + COST_H * q10
    return float((stage_one + shipping + recourse.sum(axis=1)).mean())


def checkpoint_for_seed(config, rho, seed):
    rho_directory = next(
        path
        for path in config["sweep_dir"].glob("rho_*")
        if abs(float(path.name[4:].replace("p", ".")) - rho) < 1e-12
    )
    summary_path = rho_directory / f"seed_{seed}_summary.json"
    with summary_path.open() as stream:
        summary = json.load(stream)
    if summary.get("completed") is not True:
        raise RuntimeError(f"Incomplete summary: {summary_path}")
    iteration = int(summary.get("final_mp_iter", 0))
    if iteration <= 0:
        return config["nominal_dir"] / f"seed_{seed}_nominal.pth"
    return rho_directory / f"seed_{seed}_mp_iter{iteration}.pth"


def candidate_rhos_from_training(config):
    candidates = []
    for directory in config["sweep_dir"].glob("rho_*"):
        rho = float(directory.name[4:].replace("p", "."))
        if not (0.0 < rho <= 1.5):
            continue
        checkpoints = [checkpoint_for_seed(config, rho, seed) for seed in SEEDS]
        if not all(path.exists() for path in checkpoints):
            raise RuntimeError(f"Incomplete training radius rho={rho:g}")
        candidates.append(rho)
    return sorted(candidates)


def validate_evaluation(config, evaluation, candidate_rhos):
    if evaluation.get("runtime", {}).get("status") != "completed":
        raise RuntimeError(f"Incomplete evaluation artifact: {config['eval_json']}")
    metadata = evaluation.get("meta", {})
    if (
        metadata.get("projection_method") != PROJECTION_METHOD
        or metadata.get("N_test") != 10_000
        or metadata.get("seed_test") != 9999
    ):
        raise RuntimeError(f"Noncanonical evaluation metadata: {config['eval_json']}")
    evaluated_rhos = sorted(
        float(key)
        for key in evaluation.get("sweep", {})
        if 0.0 < float(key) <= 1.5
    )
    if evaluated_rhos != candidate_rhos:
        raise RuntimeError(f"Training/evaluation radius mismatch: {config['eval_json']}")
    for rho in candidate_rhos:
        entry = evaluation["sweep"][str(float(rho))]
        if (
            entry.get("projection_method") != PROJECTION_METHOD
            or entry.get("evaluator_version") != EVALUATOR_VERSION
            or entry.get("n_seeds") != len(SEEDS)
            or len(entry.get("rob_is", [])) != len(SEEDS)
            or len(entry.get("rob_oos", [])) != len(SEEDS)
            or not np.isfinite(entry["rob_is"]).all()
            or not np.isfinite(entry["rob_oos"]).all()
        ):
            raise RuntimeError(
                f"Incomplete or stale rho={rho:g} in {config['eval_json']}"
            )


def select_size(n_train, config):
    started = time.perf_counter()
    candidate_rhos = candidate_rhos_from_training(config)
    if not candidate_rhos:
        raise RuntimeError(f"No canonical candidate radii in {config['eval_json']}")
    n_validation = validation_size(n_train)
    per_seed = {}
    selected_rhos = []
    for seed in SEEDS:
        xi1_validation, xi2_validation = generate_data(
            n_validation, VALIDATION_SEED_BASE + seed
        )
        validation_costs = {}
        for rho in candidate_rhos:
            model = load_model(checkpoint_for_seed(config, rho, seed))
            validation_costs[rho] = mean_cost(
                model, xi1_validation, xi2_validation
            )
        selected_rho = select_radius(validation_costs)
        selected_rhos.append(selected_rho)
        per_seed[str(seed)] = {
            "N_val": n_validation,
            "validation_seed": VALIDATION_SEED_BASE + seed,
            "selected_rho": selected_rho,
            "selected_validation_cost": validation_costs[selected_rho],
            "validation_costs": {
                str(rho): validation_costs[rho] for rho in candidate_rhos
            },
        }
        print(
            f"N={n_train} seed={seed}: rho={selected_rho:g} "
            f"validation={validation_costs[selected_rho]:.6f}",
            flush=True,
        )

    counts = Counter(selected_rhos)
    return {
        "candidate_rhos": candidate_rhos,
        "validation_time_s": time.perf_counter() - started,
        "selected_rhos": selected_rhos,
        "selected_rho_counts": {
            str(rho): counts[rho] for rho in sorted(counts)
        },
        "per_seed": per_seed,
        "source": {
            "sweep_dir": str(config["sweep_dir"].relative_to(BASE_DIR)),
            "eval_json": str(config["eval_json"].relative_to(BASE_DIR)),
            "training_protocol": config["training_protocol"],
        },
    }


def attach_test_results(n_train, config, selection):
    started = time.perf_counter()
    with config["eval_json"].open() as stream:
        evaluation = json.load(stream)
    validate_evaluation(config, evaluation, selection["candidate_rhos"])
    selected_oos = []
    for seed, selected_rho in enumerate(selection["selected_rhos"]):
        selected_value = float(
            evaluation["sweep"][str(float(selected_rho))]["rob_oos"][seed]
        )
        selected_oos.append(selected_value)
        selection["per_seed"][str(seed)]["selected_10k_oos"] = selected_value

    selection["selected_10k_oos_mean"] = float(np.mean(selected_oos))
    selection["selected_10k_oos_std"] = float(np.std(selected_oos, ddof=1))
    selection["cached_test_lookup_time_s"] = time.perf_counter() - started


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    result = {
        "meta": {
            "experiment": "shipping",
            "protocol": "per_seed_independent_validation_rho_selection",
            "validation_fraction": VALIDATION_FRACTION,
            "validation_size_rule": "ceil(validation_fraction * N_train)",
            "N_val_by_N": {
                str(n_train): validation_size(n_train) for n_train in CONFIGS
            },
            "seed_val_base": VALIDATION_SEED_BASE,
            "seeds": SEEDS,
            "rho_candidates": "complete positive radii with rho <= 1.5",
            "tie_break": RADIUS_SELECTION_RULE,
            "test_set": "10k seed-9999 artifact opened only after all validation selections",
            "retraining_after_selection": False,
            "shipping_cost": "hard charge 0.1 per retailer when Q2 > 1e-6",
            "N800_provenance": "corrected deterministic minibatch-200 sweep",
            "projection_method": PROJECTION_METHOD,
            "evaluator_version": EVALUATOR_VERSION,
        },
        "by_N": {},
    }
    for n_train, config in CONFIGS.items():
        result["by_N"][str(n_train)] = select_size(n_train, config)
    for n_train, config in CONFIGS.items():
        attach_test_results(n_train, config, result["by_N"][str(n_train)])
    result["runtime"] = {
        "start_utc": started_at,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "total_time_s": time.perf_counter() - started,
        "status": "completed",
    }
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w") as stream:
        json.dump(result, stream, indent=2)
    temporary_path.replace(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()