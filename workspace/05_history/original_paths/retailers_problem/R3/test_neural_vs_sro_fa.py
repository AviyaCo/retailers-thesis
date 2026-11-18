"""Exact paired tests of Neural CCG against validation-selected SRO-FA."""

import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BASE = Path("/data/home/aviyac/retailers_problem/R3")
NEURAL_PATH = BASE / "rho_validation_results_net16_v2.json"
SRO_PATH = BASE / "sro_fa_rho_validation_results.json"
OUTPUT_PATH = BASE / "neural_vs_sro_fa_paired_tests.json"
TRAINING_SIZES = (50, 200, 800)
SEEDS = tuple(range(10))


def exact_two_sided_sign_flip_p_value(differences):
    observed = abs(float(np.mean(differences)))
    null_means = np.asarray(
        [
            np.mean(differences * np.asarray(signs))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    return float(np.mean(np.abs(null_means) >= observed - 1e-14)), len(null_means)


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def main():
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    with open(NEURAL_PATH) as stream:
        neural = json.load(stream)
    with open(SRO_PATH) as stream:
        sro = json.load(stream)

    comparisons = []
    for n_train in TRAINING_SIZES:
        neural_row = neural["by_N"][str(n_train)]
        sro_row = sro["by_N"][str(n_train)]
        if sro_row.get("selection_status") != "completed":
            raise RuntimeError(f"SRO-FA selection is incomplete for N={n_train}")
        neural_costs = np.asarray(
            [neural_row["per_seed"][str(seed)]["selected_10k_oos"] for seed in SEEDS]
        )
        sro_costs = np.asarray(
            [sro_row["per_seed"][str(seed)]["selected_10k_oos"] for seed in SEEDS]
        )
        differences = neural_costs - sro_costs
        p_value, permutation_count = exact_two_sided_sign_flip_p_value(differences)
        comparisons.append(
            {
                "N": n_train,
                "seeds": list(SEEDS),
                "neural_selected_rhos": neural_row["selected_rhos"],
                "sro_fa_selected_rhos": sro_row["selected_rhos"],
                "neural_cost_by_seed": neural_costs.tolist(),
                "sro_fa_cost_by_seed": sro_costs.tolist(),
                "paired_difference_neural_minus_sro_fa": differences.tolist(),
                "neural_mean": float(np.mean(neural_costs)),
                "sro_fa_mean": float(np.mean(sro_costs)),
                "mean_paired_difference": float(np.mean(differences)),
                "median_paired_difference": float(np.median(differences)),
                "neural_lower_seed_count": int(np.sum(differences < 0.0)),
                "ties": int(np.sum(differences == 0.0)),
                "exact_two_sided_p_value": p_value,
                "permutation_count": permutation_count,
            }
        )

    adjusted = holm_adjust(
        [comparison["exact_two_sided_p_value"] for comparison in comparisons]
    )
    for comparison, adjusted_p_value in zip(comparisons, adjusted):
        comparison["holm_adjusted_p_value_across_3"] = float(adjusted_p_value)
        comparison["reject_at_familywise_alpha_0.05"] = bool(adjusted_p_value <= 0.05)

    payload = {
        "meta": {
            "experiment": "R=3, f=0",
            "test": "two-sided exact paired permutation test on mean differences",
            "difference_direction": "Neural CCG minus SRO-FA; negative favors Neural CCG",
            "inference_unit": "training seed",
            "pairing": "common training seed, validation seed, candidate grid, and seed-9999 test scenarios",
            "multiplicity": "Holm familywise correction across N=50, 200, and 800",
            "familywise_alpha": 0.05,
            "assumption": "under the null, method labels are exchangeable within each seed pair",
            "neural_artifact": str(NEURAL_PATH),
            "sro_fa_artifact": str(SRO_PATH),
        },
        "comparisons": comparisons,
        "runtime": {
            "start_utc": started_utc,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "total_time_s": time.perf_counter() - started,
        },
    }
    with open(OUTPUT_PATH, "w") as stream:
        json.dump(payload, stream, indent=2)

    print(f"Saved {OUTPUT_PATH}")
    for comparison in comparisons:
        print(
            f"N={comparison['N']}: difference={comparison['mean_paired_difference']:.6f}, "
            f"Neural lower={comparison['neural_lower_seed_count']}/10, "
            f"p={comparison['exact_two_sided_p_value']:.6f}, "
            f"Holm p={comparison['holm_adjusted_p_value_across_3']:.6f}"
        )


if __name__ == "__main__":
    main()