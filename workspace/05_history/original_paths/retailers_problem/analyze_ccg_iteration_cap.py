"""Audit AP-min incumbent selection under a common CCG iteration cap."""

import argparse
import json
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm


WORKSPACE = Path("/data/home/aviyac")
DEFAULT_OUTPUT = WORKSPACE / "retailers_problem" / "ccg_iteration_cap8_audit.json"
N_TEST = 10_000
TEST_SEED = 9999
COST_C = 0.25
COST_H = 0.05
COST_B = 0.5
SHIPPING_COST = 0.1
SHIPPING_THRESHOLD = 1e-6
EVAL_CHUNK_SIZE = 2_000
OOS_MATCH_TOLERANCE = 1e-4

torch.set_num_threads(1)

EXPERIMENTS = [
    ("R3", 50, "retailers_problem/R3/N50/rho_sweep_net16_v2"),
    ("R3", 200, "retailers_problem/R3/N200/rho_sweep_net16_v2"),
    ("R3", 800, "retailers_problem/R3/N800/rho_sweep_net16_v2"),
    ("R10", 50, "retailers_problem/R10/N50/rho_sweep_net16_v2"),
    ("R10", 200, "retailers_problem/R10/N200/rho_sweep_net16_v2"),
    ("R10", 800, "retailers_problem/R10/N800/rho_sweep_net16_v2"),
    ("shipping_R3", 50, "retailers_with_shipping/N50/rho_sweep"),
    ("shipping_R3", 200, "retailers_with_shipping/N200/rho_sweep"),
    ("shipping_R3", 800, "retailers_with_shipping/N800/rho_sweep_minibatch"),
]

EVALUATION_PATHS = {
    ("R3", 50): "retailers_problem/R3/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json",
    ("R3", 200): "retailers_problem/R3/N200/rho_sweep_net16_v2/sweep_eval_net16_v2_N200_10k.json",
    ("R3", 800): "retailers_problem/R3/N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_10k.json",
    ("R10", 50): "retailers_problem/R10/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_N50_R10.json",
    ("R10", 200): "retailers_problem/R10/N200/rho_sweep_net16_v2/sweep_eval_net16_v2_N200_R10.json",
    ("R10", 800): "retailers_problem/R10/N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_R10_10k.json",
    ("shipping_R3", 50): "retailers_with_shipping/N50/rho_sweep/sweep_eval_shipping_10k.json",
    ("shipping_R3", 200): "retailers_with_shipping/N200/rho_sweep/sweep_eval_shipping_10k.json",
    ("shipping_R3", 800): "retailers_with_shipping/N800/rho_sweep_minibatch/sweep_eval_shipping_10k.json",
}


class Policy(nn.Module):
    def __init__(self, retailer_count, hidden_widths):
        super().__init__()
        first_width, second_width = hidden_widths
        self.fc_layers = nn.Sequential(
            nn.Linear(retailer_count * 2 + 1, first_width),
            nn.ReLU(),
            nn.Linear(first_width, second_width),
            nn.ReLU(),
            nn.Linear(second_width, retailer_count),
        )
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(retailer_count))


def load_policy(checkpoint_path, retailer_count):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    hidden_widths = (
        int(state["fc_layers.0.weight"].shape[0]),
        int(state["fc_layers.2.weight"].shape[0]),
    )
    model = Policy(retailer_count, hidden_widths)
    incompatible = model.load_state_dict(state, strict=False)
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


def generate_test_data(retailer_count):
    np.random.seed(TEST_SEED)
    lower_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(
        lower_xi1,
        np.inf,
        loc=6.0,
        scale=2.5,
        size=(N_TEST, retailer_count),
    )
    mean_xi2 = 2 * (6.0 - xi1) ** 2
    lower_xi2 = (0.0 - mean_xi2) / 2.5
    xi2 = truncnorm.rvs(lower_xi2, np.inf, loc=mean_xi2, scale=2.5)
    return xi1, xi2


def project_simplex_cap(raw_values, capacity):
    nonnegative = np.maximum(np.asarray(raw_values, dtype=float), 0.0)
    needs_projection = nonnegative.sum(axis=1) > capacity
    if not np.any(needs_projection):
        return nonnegative

    selected = nonnegative[needs_projection]
    sorted_values = np.sort(selected, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_values, axis=1)
    positions = np.arange(1, selected.shape[1] + 1)
    active_count = np.sum(
        sorted_values * positions > cumulative - capacity, axis=1
    )
    threshold = (
        cumulative[np.arange(len(selected)), active_count - 1] - capacity
    ) / active_count
    nonnegative[needs_projection] = np.maximum(
        selected - threshold[:, None], 0.0
    )
    return nonnegative


def evaluate_checkpoint(checkpoint_path, retailer_count, shipping, xi1, xi2):
    started = time.perf_counter()
    model = load_policy(checkpoint_path, retailer_count)
    q10 = torch.nn.functional.softplus(model.Q10).item()
    q1r = torch.nn.functional.softplus(model.Q1r).detach().numpy()
    total_cost = 0.0
    with torch.no_grad():
        for start in range(0, len(xi1), EVAL_CHUNK_SIZE):
            stop = min(start + EVAL_CHUNK_SIZE, len(xi1))
            xi1_chunk = xi1[start:stop]
            xi2_chunk = xi2[start:stop]
            network_input = torch.cat(
                [
                    torch.as_tensor(xi1_chunk, dtype=torch.float32),
                    torch.as_tensor(q1r, dtype=torch.float32)
                    .unsqueeze(0)
                    .expand(len(xi1_chunk), -1),
                    torch.full((len(xi1_chunk), 1), q10),
                ],
                dim=1,
            )
            raw_q2 = model.fc_layers(network_input).numpy()
            q2 = project_simplex_cap(raw_q2, q10)
            recourse = np.maximum(
                COST_B * (xi1_chunk + xi2_chunk - q1r - q2) - COST_H * q2,
                np.maximum(
                    COST_H * (q1r - xi1_chunk - xi2_chunk),
                    COST_B * (xi1_chunk - q1r) - COST_H * xi2_chunk,
                ),
            )
            costs = (
                COST_C * (q10 + q1r.sum())
                + COST_H * q10
                + recourse.sum(axis=1)
            )
            if shipping:
                costs += SHIPPING_COST * (q2 > SHIPPING_THRESHOLD).sum(axis=1)
            total_cost += float(costs.sum())
    return total_cost / len(xi1), time.perf_counter() - started


def evaluation_entry(sweep, rho):
    matching_keys = [key for key in sweep if abs(float(key) - rho) < 1e-12]
    if len(matching_keys) != 1:
        raise KeyError(f"Expected one evaluation entry for rho={rho}, got {matching_keys}")
    return sweep[matching_keys[0]]


def rho_from_directory(path):
    return float(path.name.removeprefix("rho_").replace("p", "."))


def nominal_checkpoint(family, n_train, seed):
    if family == "shipping_R3":
        return WORKSPACE / "retailers_with_shipping" / f"N{n_train}" / f"seed_{seed}_nominal.pth"

    base = WORKSPACE / "retailers_problem" / family / f"N{n_train}" / "saainit_net16_v2"
    candidates = [
        base / f"seed_{seed}_nominal.pth",
        base / f"seed_{seed}_final.pth",
        base / f"seed_{seed}.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def checkpoint_for_ap(family, n_train, rho_directory, seed, ap_iteration):
    mp_iteration = ap_iteration - 1
    if mp_iteration == 0:
        return nominal_checkpoint(family, n_train, seed), mp_iteration
    return rho_directory / f"seed_{seed}_mp_iter{mp_iteration}.pth", mp_iteration


def analyze_experiment(family, n_train, relative_root, cap):
    root = WORKSPACE / relative_root
    rows = []
    for summary_path in sorted(root.glob("rho_*/seed_*_summary.json")):
        with summary_path.open() as stream:
            summary = json.load(stream)
        trajectory = [
            (int(entry["iter"]), float(entry["ap_mean"]))
            for entry in summary.get("ccg_trajectory", [])
            if "iter" in entry and "ap_mean" in entry
        ]
        if not trajectory:
            continue

        eligible = [entry for entry in trajectory if entry[0] <= cap]
        if not eligible:
            raise RuntimeError(f"No AP iteration at or below cap {cap}: {summary_path}")

        full_ap_iteration, full_ap_mean = min(trajectory, key=lambda entry: entry[1])
        capped_ap_iteration, capped_ap_mean = min(eligible, key=lambda entry: entry[1])
        seed = int(summary.get("seed", summary_path.stem.split("_")[1]))
        capped_checkpoint, capped_mp_iteration = checkpoint_for_ap(
            family, n_train, summary_path.parent, seed, capped_ap_iteration
        )
        full_checkpoint, full_mp_iteration = checkpoint_for_ap(
            family, n_train, summary_path.parent, seed, full_ap_iteration
        )
        rows.append(
            {
                "rho": rho_from_directory(summary_path.parent),
                "seed": seed,
                "actual_last_ap_iteration": max(entry[0] for entry in trajectory),
                "full_ap_min_iteration": full_ap_iteration,
                "full_mp_iteration": full_mp_iteration,
                "full_ap_mean": full_ap_mean,
                "capped_ap_min_iteration": capped_ap_iteration,
                "capped_mp_iteration": capped_mp_iteration,
                "capped_ap_mean": capped_ap_mean,
                "selection_changed": capped_ap_iteration != full_ap_iteration,
                "ap_mean_increase": capped_ap_mean - full_ap_mean,
                "stored_best_ap_iteration": summary.get("best_ap_iter"),
                "stored_final_mp_iteration": summary.get("final_mp_iter"),
                "full_selection_matches_stored": (
                    summary.get("final_mp_iter") is None
                    or int(summary["final_mp_iter"]) == full_mp_iteration
                ),
                "full_checkpoint": str(full_checkpoint.relative_to(WORKSPACE)),
                "capped_checkpoint": str(capped_checkpoint.relative_to(WORKSPACE)),
                "full_checkpoint_exists": full_checkpoint.exists(),
                "capped_checkpoint_exists": capped_checkpoint.exists(),
                "summary": str(summary_path.relative_to(WORKSPACE)),
            }
        )

    if not rows:
        raise RuntimeError(f"No canonical summaries found under {root}")

    trajectory_distribution = Counter(row["actual_last_ap_iteration"] for row in rows)
    changed_rows = [row for row in rows if row["selection_changed"]]
    return {
        "family": family,
        "n_train": n_train,
        "source_directory": relative_root,
        "run_count": len(rows),
        "actual_iteration_min": min(row["actual_last_ap_iteration"] for row in rows),
        "actual_iteration_max": max(row["actual_last_ap_iteration"] for row in rows),
        "trajectory_length_distribution": {
            str(iteration): count
            for iteration, count in sorted(trajectory_distribution.items())
        },
        "runs_reaching_beyond_cap": sum(
            row["actual_last_ap_iteration"] > cap for row in rows
        ),
        "selection_change_count": len(changed_rows),
        "missing_full_checkpoint_count": sum(
            not row["full_checkpoint_exists"] for row in rows
        ),
        "missing_capped_checkpoint_count": sum(
            not row["capped_checkpoint_exists"] for row in rows
        ),
        "stored_selection_mismatch_count": sum(
            not row["full_selection_matches_stored"] for row in rows
        ),
        "changed_runs": changed_rows,
    }


def evaluate_oos_changes(experiment):
    family = experiment["family"]
    n_train = experiment["n_train"]
    retailer_count = 10 if family == "R10" else 3
    shipping = family == "shipping_R3"
    evaluation_path = WORKSPACE / EVALUATION_PATHS[(family, n_train)]
    with evaluation_path.open() as stream:
        canonical_evaluation = json.load(stream)
    sweep = canonical_evaluation["sweep"]
    xi1, xi2 = generate_test_data(retailer_count)

    evaluation_started = time.perf_counter()
    per_rho = {}
    full_evaluation_time = 0.0
    capped_evaluation_time = 0.0
    max_canonical_error = 0.0
    canonical_mismatch_count = 0
    for row in experiment["changed_runs"]:
        canonical_entry = evaluation_entry(sweep, row["rho"])
        canonical_values = canonical_entry["rob_oos"]
        if len(canonical_values) != 10:
            raise RuntimeError(
                f"Expected 10 canonical OOS values for rho={row['rho']} in {evaluation_path}"
            )
        canonical_oos = float(canonical_values[row["seed"]])
        full_oos, full_time = evaluate_checkpoint(
            WORKSPACE / row["full_checkpoint"],
            retailer_count,
            shipping,
            xi1,
            xi2,
        )
        capped_oos, capped_time = evaluate_checkpoint(
            WORKSPACE / row["capped_checkpoint"],
            retailer_count,
            shipping,
            xi1,
            xi2,
        )
        canonical_error = abs(full_oos - canonical_oos)
        max_canonical_error = max(max_canonical_error, canonical_error)
        canonical_matches_full = canonical_error <= OOS_MATCH_TOLERANCE
        canonical_mismatch_count += not canonical_matches_full
        full_evaluation_time += full_time
        capped_evaluation_time += capped_time
        row.update(
            {
                "canonical_full_oos": canonical_oos,
                "recomputed_full_oos": full_oos,
                "capped_oos": capped_oos,
                "capped_minus_full_oos": capped_oos - full_oos,
                "canonical_full_oos_abs_error": canonical_error,
                "canonical_matches_recomputed_full_oos": canonical_matches_full,
                "full_oos_evaluation_time_s": full_time,
                "capped_oos_evaluation_time_s": capped_time,
            }
        )

        rho_key = str(row["rho"])
        if rho_key not in per_rho:
            per_rho[rho_key] = {
                "rho": row["rho"],
                "full_oos_by_seed": [float(value) for value in canonical_values],
                "capped_oos_by_seed": [float(value) for value in canonical_values],
                "changed_seeds": [],
            }
        per_rho[rho_key]["full_oos_by_seed"][row["seed"]] = full_oos
        per_rho[rho_key]["capped_oos_by_seed"][row["seed"]] = capped_oos
        per_rho[rho_key]["changed_seeds"].append(row["seed"])

    for values in per_rho.values():
        full_mean = float(np.mean(values["full_oos_by_seed"]))
        capped_mean = float(np.mean(values["capped_oos_by_seed"]))
        values.update(
            {
                "full_oos_mean": full_mean,
                "capped_oos_mean": capped_mean,
                "capped_minus_full_oos_mean": capped_mean - full_mean,
            }
        )
        values["changed_seeds"].sort()

    rho_deltas = [
        values["capped_minus_full_oos_mean"] for values in per_rho.values()
    ]
    experiment["oos_sensitivity"] = {
        "status": "completed" if experiment["changed_runs"] else "not_needed",
        "canonical_evaluation": str(evaluation_path.relative_to(WORKSPACE)),
        "n_test": N_TEST,
        "test_seed": TEST_SEED,
        "changed_checkpoint_pairs_evaluated": len(experiment["changed_runs"]),
        "full_checkpoint_evaluation_time_s": full_evaluation_time,
        "capped_checkpoint_evaluation_time_s": capped_evaluation_time,
        "total_evaluation_time_s": time.perf_counter() - evaluation_started,
        "max_canonical_full_oos_abs_error": max_canonical_error,
        "canonical_full_oos_mismatch_count": canonical_mismatch_count,
        "canonical_match_tolerance": OOS_MATCH_TOLERANCE,
        "max_abs_rho_mean_change": max(map(abs, rho_deltas), default=0.0),
        "per_affected_rho": dict(
            sorted(per_rho.items(), key=lambda item: float(item[0]))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--evaluate-oos",
        action="store_true",
        help="Reevaluate changed full and capped incumbents on the canonical 10k test set",
    )
    args = parser.parse_args()
    if args.cap < 1:
        raise ValueError("--cap must be at least 1")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    experiments = [
        analyze_experiment(family, n_train, root, args.cap)
        for family, n_train, root in EXPERIMENTS
    ]
    missing = sum(
        experiment["missing_full_checkpoint_count"]
        + experiment["missing_capped_checkpoint_count"]
        for experiment in experiments
    )
    if missing:
        raise FileNotFoundError(f"The audit found {missing} missing selected checkpoints")
    stored_mismatches = sum(
        experiment["stored_selection_mismatch_count"] for experiment in experiments
    )
    if stored_mismatches:
        raise RuntimeError(
            f"The audit found {stored_mismatches} stored final_mp_iter values that "
            "do not match full-trajectory AP-min selection"
        )
    if args.evaluate_oos:
        for experiment in experiments:
            evaluate_oos_changes(experiment)

    output = {
        "experiment": "post_hoc_ccg_iteration_cap_sensitivity",
        "method": "AP-min incumbent selected using only AP iterations at or below cap",
        "cap": args.cap,
        "test_data_used_for_selection": False,
        "oos_evaluation_requested": args.evaluate_oos,
        "ap_to_mp_mapping": "AP iteration t evaluates MP checkpoint t-1",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "experiments": experiments,
        "totals": {
            "run_count": sum(experiment["run_count"] for experiment in experiments),
            "selection_change_count": sum(
                experiment["selection_change_count"] for experiment in experiments
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        json.dump(output, stream, indent=2)
        stream.write("\n")
    print(json.dumps(output["totals"], indent=2))
    for experiment in experiments:
        print(
            f"{experiment['family']} N={experiment['n_train']}: "
            f"runs={experiment['run_count']}, "
            f"actual_max={experiment['actual_iteration_max']}, "
            f"changed={experiment['selection_change_count']}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()