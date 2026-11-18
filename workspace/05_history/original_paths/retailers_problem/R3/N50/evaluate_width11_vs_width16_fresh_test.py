"""Final fresh-test comparison of validation-selected widths 11 and 16."""

import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from evaluate_plot_net8_vs_net16 import generate_data, load_model


BASE = Path(__file__).resolve().parent
WIDTH11_VALIDATION = BASE / "width11_interim_validation.json"
WIDTH16_REFERENCE = BASE / "net8_vs_net16_all_radii_10k.json"
OUTPUT = BASE / "width11_vs_width16_fresh_seed10000.json"
SEEDS = list(range(10))
N_TEST = 10_000
TEST_SEED = 10_000


def exact_two_sided_sign_flip_p_value(differences):
    observed = abs(float(np.mean(differences)))
    null_means = np.asarray(
        [
            np.mean(differences * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    return float(np.mean(np.abs(null_means) >= observed - 1e-14)), len(null_means)


def selected_width11(validation):
    radii = [float(rho) for rho in validation["radii"]]
    selected_rhos = []
    checkpoints = []
    validation_costs = []
    for seed in SEEDS:
        costs = {
            rho: float(validation["validation"][str(rho)]["values"][seed])
            for rho in radii
        }
        selected_rho = min(costs, key=lambda rho: (costs[rho], rho))
        selected_rhos.append(selected_rho)
        validation_costs.append(costs[selected_rho])
        checkpoints.append(
            validation["checkpoint_provenance"][str(selected_rho)][str(seed)][
                "checkpoint"
            ]
        )
    return selected_rhos, validation_costs, checkpoints


def selected_width16(reference):
    selection = reference["validation_selection"]["net16"]
    selected_rhos = [float(rho) for rho in selection["selected_rhos"]]
    validation_costs = [
        float(selection["per_seed"][str(seed)]["selected_validation_cost"])
        for seed in SEEDS
    ]
    checkpoints = [
        reference["checkpoint_provenance"]["net16"][str(rho)][str(seed)][
            "checkpoint"
        ]
        for seed, rho in enumerate(selected_rhos)
    ]
    return selected_rhos, validation_costs, checkpoints


def evaluate(checkpoints, xi1_test, xi2_test):
    costs = []
    times = []
    for checkpoint in checkpoints:
        started = time.perf_counter()
        model = load_model(checkpoint)
        costs.append(model.mean_cost(xi1_test, xi2_test))
        times.append(time.perf_counter() - started)
    return costs, times


def main():
    if OUTPUT.exists():
        raise RuntimeError(f"Refusing to overwrite fresh-test artifact: {OUTPUT}")

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with open(WIDTH11_VALIDATION) as stream:
        width11_validation = json.load(stream)
    with open(WIDTH16_REFERENCE) as stream:
        width16_reference = json.load(stream)

    width11_rhos, width11_validation_costs, width11_checkpoints = selected_width11(
        width11_validation
    )
    width16_rhos, width16_validation_costs, width16_checkpoints = selected_width16(
        width16_reference
    )
    for checkpoint in width11_checkpoints + width16_checkpoints:
        if not Path(checkpoint).exists():
            raise RuntimeError(f"Missing selected checkpoint: {checkpoint}")

    selection_frozen_utc = datetime.now(timezone.utc).isoformat()
    test_generation_started = time.perf_counter()
    xi1_test, xi2_test = generate_data(N_TEST, TEST_SEED)
    test_generation_time = time.perf_counter() - test_generation_started

    width11_costs, width11_times = evaluate(
        width11_checkpoints, xi1_test, xi2_test
    )
    width16_costs, width16_times = evaluate(
        width16_checkpoints, xi1_test, xi2_test
    )
    differences = np.asarray(width11_costs) - np.asarray(width16_costs)
    p_value, permutation_count = exact_two_sided_sign_flip_p_value(differences)

    payload = {
        "meta": {
            "experiment": "R3_N50_width11_vs_width16_fresh_test",
            "status": "completed",
            "n_train": 50,
            "n_test": N_TEST,
            "test_seed": TEST_SEED,
            "training_seeds": SEEDS,
            "selection_frozen_before_test_generation": True,
            "selection_frozen_utc": selection_frozen_utc,
            "selection_rule": "seed-specific minimum independent-validation cost; smaller rho breaks exact ties",
            "projection_method": "gurobi_qp",
            "primary_comparison": "width 11 minus width 16",
            "test": "two-sided exact paired permutation test on mean differences",
            "inference_unit": "training seed",
            "assumption": "under the null, width labels are exchangeable within each seed pair",
        },
        "width11": {
            "selected_rhos": width11_rhos,
            "selected_validation_costs": width11_validation_costs,
            "selected_checkpoints": width11_checkpoints,
            "fresh_test_costs": width11_costs,
            "fresh_test_mean": float(np.mean(width11_costs)),
            "fresh_test_std": float(np.std(width11_costs, ddof=1)),
        },
        "width16": {
            "selected_rhos": width16_rhos,
            "selected_validation_costs": width16_validation_costs,
            "selected_checkpoints": width16_checkpoints,
            "fresh_test_costs": width16_costs,
            "fresh_test_mean": float(np.mean(width16_costs)),
            "fresh_test_std": float(np.std(width16_costs, ddof=1)),
        },
        "paired_test": {
            "differences_width11_minus_width16": differences.tolist(),
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "width11_lower_seed_count": int(np.sum(differences < 0.0)),
            "ties": int(np.sum(differences == 0.0)),
            "exact_two_sided_p_value": p_value,
            "permutation_count": permutation_count,
            "multiplicity_adjustment": None,
        },
        "runtime": {
            "started_utc": started_utc,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "test_generation_time_s": test_generation_time,
            "width11_evaluation_time_by_seed_s": width11_times,
            "width11_evaluation_total_s": float(sum(width11_times)),
            "width16_evaluation_time_by_seed_s": width16_times,
            "width16_evaluation_total_s": float(sum(width16_times)),
            "total_time_s": time.perf_counter() - started,
            "parallel_workers": 1,
        },
        "sources": {
            "width11_validation": str(WIDTH11_VALIDATION),
            "width16_validation_and_provenance": str(WIDTH16_REFERENCE),
        },
    }
    temporary = OUTPUT.with_suffix(".json.tmp")
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
    temporary.replace(OUTPUT)
    print(f"width11 mean={payload['width11']['fresh_test_mean']:.6f}")
    print(f"width16 mean={payload['width16']['fresh_test_mean']:.6f}")
    print(
        f"difference={payload['paired_test']['mean_difference']:.6f}, "
        f"width11 lower={payload['paired_test']['width11_lower_seed_count']}/10, "
        f"p={p_value:.6f}"
    )
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()