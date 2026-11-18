"""Evaluate the complete width-11 sweep on the historical seed-9999 OOS draw."""

import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from evaluate_plot_net8_vs_net16 import generate_data, load_model
from plot_width11_interim_validation import (
    completed_radii,
    incumbent_checkpoint,
    statistics,
)


BASE_DIR = Path(__file__).resolve().parent
NOMINAL_DIR = BASE_DIR / "saainit_net11_v2"
REFERENCE_PATH = BASE_DIR / "net8_vs_net16_all_radii_10k.json"
OUTPUT_JSON = BASE_DIR / "width11_oos_10k.json"
OUTPUT_PNG = BASE_DIR / "width11_oos_10k.png"
OUTPUT_PDF = BASE_DIR / "width11_oos_10k.pdf"

SEEDS = list(range(10))
N_TEST = 10_000
TEST_SEED = 9999
FRESH_FINAL_TEST_SEED = 10_000


def write_result(result):
    temporary_path = OUTPUT_JSON.with_suffix(".json.tmp")
    with temporary_path.open("w") as stream:
        json.dump(result, stream, indent=2)
    temporary_path.replace(OUTPUT_JSON)


def plot(result):
    radii = np.asarray(result["radii"], dtype=float)
    curves = {
        "net11": result["width11_oos"],
        "net13": result["reference_oos"]["net13"],
        "net16": result["reference_oos"]["net16"],
    }
    styles = {
        "net11": ("#1f6f5f", "o", "Width 11 OOS (post-hoc)"),
        "net13": ("#7b3294", "^", "Width 13 OOS"),
        "net16": ("#2166ac", "s", "Width 16 OOS"),
    }

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for width in ("net13", "net16", "net11"):
        color, marker, label = styles[width]
        means = np.asarray([curves[width][str(rho)]["mean"] for rho in radii])
        low = np.asarray(
            [curves[width][str(rho)]["percentile_20"] for rho in radii]
        )
        high = np.asarray(
            [curves[width][str(rho)]["percentile_80"] for rho in radii]
        )
        axis.plot(
            radii,
            means,
            marker=marker,
            color=color,
            lw=2.1 if width == "net11" else 1.8,
            ms=4.2 if width == "net11" else 3.7,
            label=label,
        )
        axis.fill_between(radii, low, high, color=color, alpha=0.1)

    nominal_mean = result["width11_nominal_oos"]["mean"]
    axis.axhline(
        nominal_mean,
        color="#b13c2e",
        ls="--",
        lw=1.7,
        label=f"Width 11 nominal OOS ({nominal_mean:.3f})",
    )
    observed_rho = result["observed_width11_minimum"]["rho"]
    observed_mean = result["observed_width11_minimum"]["mean_oos_cost"]
    axis.scatter(
        observed_rho,
        observed_mean,
        s=70,
        facecolor="#f2c14e",
        edgecolor="#333333",
        zorder=5,
        label=f"Observed width 11 minimum: $\\rho={observed_rho:g}$ ({observed_mean:.3f})",
    )

    display_ticks = [
        rho
        for rho in (
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.65,
        )
        if rho in radii
    ]
    axis.set_xscale("log")
    axis.set_xlim(radii.min() / 1.15, radii.max() * 1.15)
    axis.set_xticks(display_ticks)
    axis.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axis.tick_params(axis="x", labelsize=8, rotation=45)
    axis.set_xlabel("Training radius $\\rho$")
    axis.set_ylabel("Mean OOS cost")
    axis.set_title("Width comparison on common OOS scenarios")
    fig.suptitle(
        "Retailers R=3, N=50 | 10,000 common scenarios, seed 9999\n"
        "Complete 23-radius width-11 sweep; post-hoc and selection-ineligible",
        fontsize=13,
    )
    axis.grid(axis="y", color="#cccccc", alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best", fontsize=8.3)
    fig.subplots_adjust(bottom=0.2, top=0.82)
    fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    complete = completed_radii()
    if len(complete) != 23:
        raise RuntimeError(
            f"Expected 23 complete width-11 radii, found {len(complete)}"
        )
    radii = [rho for rho, _, _ in complete]

    with REFERENCE_PATH.open() as stream:
        reference = json.load(stream)
    reference_meta = reference.get("meta", {})
    if (
        reference_meta.get("N_test") != N_TEST
        or reference_meta.get("test_seed") != TEST_SEED
        or reference_meta.get("projection_method") != "gurobi_qp"
        or reference_meta.get("training_seeds") != SEEDS
    ):
        raise RuntimeError("Unexpected width-13/16 OOS reference metadata")
    reference_oos = {"net13": {}, "net16": {}}
    for rho in radii:
        for width in reference_oos:
            entry = reference.get(width, {}).get(str(rho))
            if entry is None or len(entry.get("values", [])) != len(SEEDS):
                raise RuntimeError(f"Incomplete {width} OOS reference at rho={rho:g}")
            reference_oos[width][str(rho)] = entry

    result = {
        "meta": {
            "experiment": "R3_N50_width11_complete_seed9999_oos",
            "status": "running",
            "interim": False,
            "sweep_status": "completed",
            "post_hoc_exploratory": True,
            "eligible_for_model_or_radius_selection": False,
            "hidden_widths": [11, 11],
            "n_train": 50,
            "n_test": N_TEST,
            "test_seed": TEST_SEED,
            "fresh_final_test_seed_not_accessed": FRESH_FINAL_TEST_SEED,
            "training_seeds": SEEDS,
            "radius_inclusion_rule": "all 23 preregistered radii completed for all ten training seeds",
            "projection_method": "gurobi_qp",
            "evaluator_version": "gurobi_qp_complete_oos_v1",
            "incumbent_rule": "minimum AP mean; AP iteration t evaluates MP iteration t-1",
            "interval": "20th-80th percentile across training seeds",
        },
        "radii": radii,
        "width11_nominal_oos": None,
        "width11_oos": {},
        "reference_oos": reference_oos,
        "reference_source": str(REFERENCE_PATH),
        "checkpoint_provenance": {},
        "runtime": {
            "started_at_utc": started_at,
            "test_generation_time_s": None,
            "nominal_evaluation": {},
            "radius_evaluation": {},
            "plot_time_s": None,
            "total_time_s": None,
            "parallel_workers": 1,
            "thread_limit": 1,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    write_result(result)

    try:
        test_started = time.perf_counter()
        xi1_test, xi2_test = generate_data(N_TEST, TEST_SEED)
        result["runtime"]["test_generation_time_s"] = time.perf_counter() - test_started

        nominal_values = []
        nominal_started = time.perf_counter()
        for seed in SEEDS:
            seed_started = time.perf_counter()
            model = load_model(NOMINAL_DIR / f"seed_{seed}_nominal.pth")
            nominal_values.append(model.mean_cost(xi1_test, xi2_test))
            result["runtime"]["nominal_evaluation"][str(seed)] = {
                "wall_time_s": time.perf_counter() - seed_started,
                "status": "completed",
            }
        result["runtime"]["nominal_evaluation"]["total_wall_time_s"] = (
            time.perf_counter() - nominal_started
        )
        result["width11_nominal_oos"] = statistics(nominal_values)
        write_result(result)

        for rho, directory, summaries in complete:
            radius_started = time.perf_counter()
            values = []
            result["checkpoint_provenance"][str(rho)] = {}
            result["runtime"]["radius_evaluation"][str(rho)] = {
                "per_seed": {},
                "status": "running",
            }
            for seed, summary in zip(SEEDS, summaries):
                seed_started = time.perf_counter()
                checkpoint = incumbent_checkpoint(directory, summary, seed)
                model = load_model(checkpoint)
                values.append(model.mean_cost(xi1_test, xi2_test))
                result["runtime"]["radius_evaluation"][str(rho)]["per_seed"][
                    str(seed)
                ] = {
                    "wall_time_s": time.perf_counter() - seed_started,
                    "status": "completed",
                }
                result["checkpoint_provenance"][str(rho)][str(seed)] = {
                    "checkpoint": str(checkpoint),
                    "incumbent_mp_iteration": int(summary["final_mp_iter"]),
                    "best_ap_iteration": int(summary["best_ap_iter"]),
                }
            result["width11_oos"][str(rho)] = statistics(values)
            radius_runtime = result["runtime"]["radius_evaluation"][str(rho)]
            radius_runtime["total_wall_time_s"] = time.perf_counter() - radius_started
            radius_runtime["status"] = "completed"
            write_result(result)
            print(
                f"rho={rho:g}: width11 OOS mean={result['width11_oos'][str(rho)]['mean']:.6f} "
                f"time={radius_runtime['total_wall_time_s']:.3f}s",
                flush=True,
            )

        means = {
            rho: result["width11_oos"][str(rho)]["mean"] for rho in radii
        }
        observed_rho = min(means, key=lambda rho: (means[rho], rho))
        result["observed_width11_minimum"] = {
            "rho": observed_rho,
            "mean_oos_cost": means[observed_rho],
            "selection_eligible": False,
        }
        plot_started = time.perf_counter()
        plot(result)
        result["runtime"]["plot_time_s"] = time.perf_counter() - plot_started
        result["meta"]["status"] = "completed"
        result["runtime"]["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["runtime"]["total_time_s"] = time.perf_counter() - started
        write_result(result)
        print(f"Observed width-11 minimum: rho={observed_rho:g}, mean={means[observed_rho]:.6f}")
        print(f"Saved {OUTPUT_JSON}")
        print(f"Saved {OUTPUT_PNG}")
        print(f"Saved {OUTPUT_PDF}")
    except Exception as error:
        result["meta"]["status"] = "failed"
        result["runtime"]["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["runtime"]["total_time_s"] = time.perf_counter() - started
        result["error"] = f"{type(error).__name__}: {error}"
        write_result(result)
        raise


if __name__ == "__main__":
    main()