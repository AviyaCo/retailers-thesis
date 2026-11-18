"""Select Neural CCG radii for the R10 and fixed-shipping extensions.

For each experiment, training size, and seed, this script evaluates existing
rho-specific incumbent checkpoints on an independent validation draw. It chooses
rho using validation costs only, then reads the selected policy's saved 10k test
cost. No model is retrained.
"""

import json
import sys as _sys
_sys.path.insert(0, '/data/home/aviyac/retailers_experiment')
from src.config import RADIUS_SELECTION_RULE, select_radius  # shared radius rule, 2026-11-18
import os
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn


ROOT = "/data/home/aviyac"
SEEDS = list(range(10))
VALIDATION_FRACTION = 0.25
SEED_VAL_BASE = 88880
MAX_RHO = 1.5
C, H, B = 0.25, 0.05, 0.5
SHIPPING_CHARGE = 0.1
SHIPPING_THRESHOLD = 1e-6

EXPERIMENTS = {
    "R10": {
        "R": 10,
        "shipping": False,
        "output": os.path.join(
            ROOT, "retailers_problem/R10/rho_validation_results_net16_v2.json"
        ),
        "configs": {
            50: {
                "nominal_dir": "retailers_problem/R10/N50/saainit_net16_v2",
                "sweep_dir": "retailers_problem/R10/N50/rho_sweep_net16_v2",
                "eval_json": "retailers_problem/R10/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_N50_R10.json",
            },
            200: {
                "nominal_dir": "retailers_problem/R10/N200/saainit_net16_v2",
                "sweep_dir": "retailers_problem/R10/N200/rho_sweep_net16_v2",
                "eval_json": "retailers_problem/R10/N200/rho_sweep_net16_v2/sweep_eval_net16_v2_N200_R10.json",
            },
            800: {
                "nominal_dir": "retailers_problem/R10/N800/saainit_net16_v2",
                "sweep_dir": "retailers_problem/R10/N800/rho_sweep_net16_v2",
                "eval_json": "retailers_problem/R10/N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_R10_10k.json",
            },
        },
    },
    "shipping": {
        "R": 3,
        "shipping": True,
        "output": os.path.join(
            ROOT, "retailers_with_shipping/rho_validation_results_net16_v2.json"
        ),
        "configs": {
            n_train: {
                "nominal_dir": f"retailers_with_shipping/N{n_train}",
                "sweep_dir": f"retailers_with_shipping/N{n_train}/rho_sweep",
                "eval_json": f"retailers_with_shipping/N{n_train}/rho_sweep/sweep_eval_shipping_10k.json",
            }
            for n_train in (50, 200, 800)
        },
    },
}


def absolute_paths(config):
    return {key: os.path.join(ROOT, value) for key, value in config.items()}


def validation_size(n_train):
    return int(np.ceil(VALIDATION_FRACTION * n_train))


def generate_data(n_samples, seed, n_retailers):
    rng_state = np.random.get_state()
    try:
        np.random.seed(seed)
        lower_xi1 = (0.0 - 6.0) / 2.5
        xi1 = truncnorm.rvs(
            lower_xi1,
            np.inf,
            loc=6.0,
            scale=2.5,
            size=(n_samples, n_retailers),
        )
        mu2 = 2 * (6.0 - xi1) ** 2
        lower_xi2 = (0.0 - mu2) / 2.5
        xi2 = truncnorm.rvs(lower_xi2, np.inf, loc=mu2, scale=2.5)
    finally:
        np.random.set_state(rng_state)
    return xi1, xi2


def project_simplex_scaled(z, capacity):
    nonnegative = np.maximum(z, 0.0)
    needs_projection = nonnegative.sum(axis=1) > capacity
    if not np.any(needs_projection):
        return nonnegative

    projected = nonnegative.copy()
    active = nonnegative[needs_projection]
    sorted_active = np.sort(active, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_active, axis=1)
    indices = np.arange(1, z.shape[1] + 1)
    rho = np.sum(sorted_active * indices > cumulative - capacity, axis=1) - 1
    theta = (cumulative[np.arange(len(active)), rho] - capacity) / (rho + 1.0)
    projected[needs_projection] = np.maximum(active - theta[:, None], 0.0)
    return projected


class Model(nn.Module):
    def __init__(self, n_retailers):
        super().__init__()
        self.n_retailers = n_retailers
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * n_retailers + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, n_retailers),
        )
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(n_retailers))

    @torch.no_grad()
    def decisions(self, xi1):
        n_samples = xi1.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        inputs = torch.cat(
            [
                xi1,
                torch.tensor(q1r).unsqueeze(0).expand(n_samples, -1),
                torch.full((n_samples, 1), q10),
            ],
            dim=1,
        )
        raw_q2 = self.fc_layers(inputs).numpy()
        return project_simplex_scaled(raw_q2, q10), q10, q1r


def load_model(path, n_retailers):
    model = Model(n_retailers)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def mean_cost(q10, q1r, q2, xi1, xi2, shipping):
    first_stage = C * (q10 + q1r.sum()) + H * q10
    recourse = np.maximum(
        B * (xi1 + xi2 - q1r - q2) - H * q2,
        np.maximum(H * (q1r - xi1 - xi2), B * (xi1 - q1r) - H * xi2),
    )
    costs = first_stage + recourse.sum(axis=1)
    if shipping:
        costs = costs + SHIPPING_CHARGE * (q2 > SHIPPING_THRESHOLD).sum(axis=1)
    return float(costs.mean())


def incumbent_path(config, rho, seed):
    summary_path = os.path.join(
        config["sweep_dir"], f"rho_{rho:.3f}".replace(".", "p"), f"seed_{seed}_summary.json"
    )
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as summary_file:
        summary = json.load(summary_file)
    if summary.get("completed") is not True:
        return None

    final_mp_iter = int(summary.get("final_mp_iter", 0))
    if final_mp_iter == 0:
        path = os.path.join(config["nominal_dir"], f"seed_{seed}_nominal.pth")
    else:
        path = os.path.join(
            os.path.dirname(summary_path), f"seed_{seed}_mp_iter{final_mp_iter}.pth"
        )
    return path if os.path.exists(path) else None


def common_complete_rhos(config, eval_data):
    candidates = []
    for rho_text, entry in eval_data["sweep"].items():
        rho = float(rho_text)
        if not (0.0 < rho <= MAX_RHO) or entry.get("n_seeds") != len(SEEDS):
            continue
        if all(incumbent_path(config, rho, seed) is not None for seed in SEEDS):
            candidates.append(rho)
    return sorted(candidates)


def selected_test_cost(eval_data, rho, seed):
    values = eval_data["sweep"][str(float(rho))]["rob_oos"]
    if len(values) != len(SEEDS):
        raise RuntimeError(f"Expected {len(SEEDS)} test costs for rho={rho}, found {len(values)}")
    return float(values[seed])


def evaluate_training_size(n_train, experiment, raw_config):
    config = absolute_paths(raw_config)
    with open(config["eval_json"]) as eval_file:
        eval_data = json.load(eval_file)

    candidates = common_complete_rhos(config, eval_data)
    if not candidates:
        raise RuntimeError(f"No common complete radius candidates for N={n_train}")

    n_validation = validation_size(n_train)
    per_seed = {}
    selected_rhos = []
    validation_started = time.perf_counter()

    for seed in SEEDS:
        seed_started = time.perf_counter()
        xi1, xi2 = generate_data(
            n_validation, SEED_VAL_BASE + seed, experiment["R"]
        )
        xi1_tensor = torch.tensor(xi1, dtype=torch.float32)
        validation_costs = {}

        for rho in candidates:
            model = load_model(incumbent_path(config, rho, seed), experiment["R"])
            q2, q10, q1r = model.decisions(xi1_tensor)
            validation_costs[rho] = mean_cost(
                q10, q1r, q2, xi1, xi2, experiment["shipping"]
            )

        selected_rho = select_radius(validation_costs)
        selected_rhos.append(selected_rho)
        per_seed[str(seed)] = {
            "N_val": n_validation,
            "validation_seed": SEED_VAL_BASE + seed,
            "selected_rho": selected_rho,
            "selected_validation_cost": validation_costs[selected_rho],
            "validation_costs": {
                str(rho): validation_costs[rho] for rho in candidates
            },
            "validation_time_s": time.perf_counter() - seed_started,
        }
        print(
            f"N={n_train:>3} seed={seed}: rho={selected_rho:g} "
            f"validation={validation_costs[selected_rho]:.4f}",
            flush=True,
        )

    selection_time = time.perf_counter() - validation_started

    # Test costs are accessed only after every seed's radius has been selected.
    test_lookup_started = time.perf_counter()
    selected_test_costs = []
    for seed, rho in enumerate(selected_rhos):
        test_cost = selected_test_cost(eval_data, rho, seed)
        selected_test_costs.append(test_cost)
        per_seed[str(seed)]["selected_10k_oos"] = test_cost
    test_lookup_time = time.perf_counter() - test_lookup_started

    counts = Counter(selected_rhos)
    return {
        "candidate_rhos": candidates,
        "selected_10k_oos_mean": float(np.mean(selected_test_costs)),
        "selected_10k_oos_std": float(np.std(selected_test_costs, ddof=1)),
        "selected_rhos": selected_rhos,
        "selected_rho_counts": {
            str(rho): counts[rho] for rho in sorted(counts)
        },
        "per_seed": per_seed,
        "runtime": {
            "validation_selection_time_s": selection_time,
            "cached_test_lookup_time_s": test_lookup_time,
            "total_time_s": selection_time + test_lookup_time,
        },
    }


def run_experiment(name, experiment):
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    result = {
        "meta": {
            "experiment": name,
            "protocol": "per_seed_independent_validation_rho_selection",
            "validation_fraction": VALIDATION_FRACTION,
            "validation_size_rule": "ceil(validation_fraction * N_train)",
            "N_val_by_N": {
                str(n_train): validation_size(n_train)
                for n_train in experiment["configs"]
            },
            "seed_val_base": SEED_VAL_BASE,
            "seeds": SEEDS,
            "rho_candidates": "common completed positive radii with rho <= 1.5",
            "tie_break": RADIUS_SELECTION_RULE,
            "test_set": "saved 10k OOS from seed 9999; read only after all rho selections",
            "retraining_after_selection": False,
            "shipping_cost": (
                "hard charge 0.1 per retailer when Q2 > 1e-6"
                if experiment["shipping"]
                else None
            ),
        },
        "by_N": {},
    }

    for n_train, config in experiment["configs"].items():
        print(f"\n{name}: selecting rho for N={n_train}")
        result["by_N"][str(n_train)] = evaluate_training_size(
            n_train, experiment, config
        )

    result["runtime"] = {
        "start_utc": started_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "total_time_s": time.perf_counter() - started,
        "test_evaluation_time_s": None,
        "test_evaluation_time_note": "Test costs were reused from evaluation caches; original evaluation time is not attributed to this selection run.",
        "parallel_workers": 1,
        "torch_threads": torch.get_num_threads(),
    }
    with open(experiment["output"], "w") as output_file:
        json.dump(result, output_file, indent=2)
    print(f"Saved {experiment['output']}")
    return result


def main():
    overall_started = time.perf_counter()
    for name, experiment in EXPERIMENTS.items():
        result = run_experiment(name, experiment)
        for n_train, entry in result["by_N"].items():
            print(
                f"{name} N={n_train}: selected 10k OOS="
                f"{entry['selected_10k_oos_mean']:.6f}, "
                f"rho_counts={entry['selected_rho_counts']}"
            )
    print(f"Overall selection elapsed: {time.perf_counter() - overall_started:.2f}s")


if __name__ == "__main__":
    main()
