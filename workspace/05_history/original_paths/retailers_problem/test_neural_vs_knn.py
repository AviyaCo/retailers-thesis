"""Exact paired tests of validation-selected Neural CCG against SAA-kNN."""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import permutation_test


ROOT = Path("/data/home/aviyac")
OUTPUT = ROOT / "retailers_problem/neural_vs_knn_paired_tests.json"
SEEDS = list(range(10))

CONFIGURATIONS = [
    (
        "R3_f0",
        n_train,
        ROOT / "retailers_problem/R3/rho_validation_results_net16_v2.json",
        ROOT / eval_path,
    )
    for n_train, eval_path in (
        (50, "retailers_problem/R3/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json"),
        (200, "retailers_problem/R3/N200/rho_sweep_net16_v2/sweep_eval_net16_v2_N200_10k.json"),
        (800, "retailers_problem/R3/N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_10k.json"),
    )
] + [
    (
        "R10_f0",
        n_train,
        ROOT / "retailers_problem/R10/rho_validation_results_net16_v2.json",
        ROOT / eval_path,
    )
    for n_train, eval_path in (
        (50, "retailers_problem/R10/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_N50_R10.json"),
        (200, "retailers_problem/R10/N200/rho_sweep_net16_v2/sweep_eval_net16_v2_N200_R10.json"),
        (800, "retailers_problem/R10/N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_R10_10k.json"),
    )
] + [
    (
        "R3_f0.1",
        n_train,
        ROOT / "retailers_with_shipping/rho_validation_results_net16_v2.json",
        ROOT / eval_path,
    )
    for n_train, eval_path in (
        (50, "retailers_with_shipping/N50/rho_sweep/sweep_eval_shipping_10k.json"),
        (200, "retailers_with_shipping/N200/rho_sweep/sweep_eval_shipping_10k.json"),
        (800, "retailers_with_shipping/N800/rho_sweep_minibatch/sweep_eval_shipping_10k.json"),
    )
]


def paired_mean_difference(neural_costs, knn_costs, axis=-1):
    return np.mean(neural_costs - knn_costs, axis=axis)


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def load_costs(selection_path, evaluation_path, n_train):
    with open(selection_path) as stream:
        neural_selection = json.load(stream)
    with open(evaluation_path) as stream:
        evaluation = json.load(stream)

    neural_by_seed = neural_selection["by_N"][str(n_train)]["per_seed"]
    neural_costs = np.asarray(
        [neural_by_seed[str(seed)]["selected_10k_oos"] for seed in SEEDS],
        dtype=float,
    )
    knn = evaluation["benchmarks"]["saa_knn"]
    knn_costs = np.asarray(knn["oos"], dtype=float)
    selected_k = [int(value) for value in knn["selected_k_by_seed"]]

    if len(knn_costs) != len(SEEDS) or len(selected_k) != len(SEEDS):
        raise RuntimeError(f"Expected ten paired observations for N={n_train}")
    if knn.get("selection_metric") != "per_seed_loo_validation_cost":
        raise RuntimeError(f"Unexpected kNN selection protocol in {evaluation_path}")
    if not np.all(np.isfinite(neural_costs)) or not np.all(np.isfinite(knn_costs)):
        raise RuntimeError(f"Non-finite selected cost for N={n_train}")
    return neural_costs, knn_costs, selected_k


def main():
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    comparisons = []

    for experiment, n_train, selection_path, evaluation_path in CONFIGURATIONS:
        neural_costs, knn_costs, selected_k = load_costs(
            selection_path, evaluation_path, n_train
        )
        differences = neural_costs - knn_costs
        test = permutation_test(
            (neural_costs, knn_costs),
            paired_mean_difference,
            permutation_type="samples",
            vectorized=True,
            n_resamples=np.inf,
            alternative="two-sided",
        )
        comparisons.append(
            {
                "experiment": experiment,
                "N": n_train,
                "seeds": SEEDS,
                "selected_k_by_seed": selected_k,
                "neural_cost_by_seed": neural_costs.tolist(),
                "knn_cost_by_seed": knn_costs.tolist(),
                "paired_difference_neural_minus_knn": differences.tolist(),
                "neural_mean": float(np.mean(neural_costs)),
                "knn_mean": float(np.mean(knn_costs)),
                "mean_paired_difference": float(np.mean(differences)),
                "median_paired_difference": float(np.median(differences)),
                "neural_lower_seed_count": int(np.sum(differences < 0.0)),
                "ties": int(np.sum(differences == 0.0)),
                "exact_two_sided_p_value": float(test.pvalue),
                "permutation_count": int(len(test.null_distribution)),
                "selection_artifact": str(selection_path.relative_to(ROOT)),
                "knn_artifact": str(evaluation_path.relative_to(ROOT)),
            }
        )

    adjusted = holm_adjust(
        [comparison["exact_two_sided_p_value"] for comparison in comparisons]
    )
    for comparison, adjusted_p_value in zip(comparisons, adjusted):
        comparison["holm_adjusted_p_value_across_9"] = float(adjusted_p_value)
        comparison["reject_at_familywise_alpha_0.05"] = bool(adjusted_p_value <= 0.05)

    payload = {
        "meta": {
            "analysis": "Neural CCG versus SAA-kNN seed-level paired comparison",
            "test": "two-sided exact paired permutation test on mean differences",
            "difference_direction": "Neural CCG minus SAA-kNN; negative favors Neural CCG",
            "permutations_per_comparison": 2 ** len(SEEDS),
            "multiplicity": "Holm familywise correction across all nine experiment-size comparisons",
            "familywise_alpha": 0.05,
            "pairing": "common training seed and common seed-9999 10k test scenarios",
            "inference_unit": "training seed",
            "test_set_used_for_selection": False,
            "assumption": "under the null, method labels are exchangeable within each seed pair",
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "comparisons": comparisons,
        "runtime": {
            "start_utc": started_utc,
            "end_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "total_time_s": time.perf_counter() - started,
        },
    }
    with open(OUTPUT, "w") as stream:
        json.dump(payload, stream, indent=2)
    print(f"Saved {OUTPUT}")
    for comparison in comparisons:
        print(
            f"{comparison['experiment']} N={comparison['N']}: "
            f"mean difference={comparison['mean_paired_difference']:.6f}, "
            f"p={comparison['exact_two_sided_p_value']:.6f}, "
            f"Holm p={comparison['holm_adjusted_p_value_across_9']:.6f}, "
            f"Neural lower={comparison['neural_lower_seed_count']}/10"
        )


if __name__ == "__main__":
    main()