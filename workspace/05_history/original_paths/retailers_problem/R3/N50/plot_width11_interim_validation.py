"""Plot the partial width-11 validation trend with historical OOS references."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from evaluate_plot_net8_vs_net16 import generate_data, load_model


BASE_DIR = Path(__file__).resolve().parent
SWEEP_DIR = BASE_DIR / "rho_sweep_net11_v2"
NOMINAL_DIR = BASE_DIR / "saainit_net11_v2"
REFERENCE_PATH = BASE_DIR / "net8_vs_net16_all_radii_10k.json"
OUTPUT_JSON = BASE_DIR / "width11_interim_validation.json"
OUTPUT_PNG = BASE_DIR / "width11_interim_validation.png"
OUTPUT_PDF = BASE_DIR / "width11_interim_validation.pdf"

SEEDS = list(range(10))
VALIDATION_SIZE = 13
VALIDATION_SEED_BASE = 88880


def completed_radii():
    radii = []
    for directory in SWEEP_DIR.glob("rho_*"):
        rho = float(directory.name[4:].replace("p", "."))
        summaries = []
        for seed in SEEDS:
            path = directory / f"seed_{seed}_summary.json"
            if not path.exists():
                break
            with path.open() as stream:
                summary = json.load(stream)
            if summary.get("completed") is not True:
                break
            summaries.append(summary)
        if len(summaries) == len(SEEDS):
            radii.append((rho, directory, summaries))
    return sorted(radii)


def incumbent_checkpoint(directory, summary, seed):
    mp_iteration = int(summary["final_mp_iter"])
    if mp_iteration <= 0:
        path = NOMINAL_DIR / f"seed_{seed}_nominal.pth"
    else:
        path = directory / f"seed_{seed}_mp_iter{mp_iteration}.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def statistics(values):
    array = np.asarray(values, dtype=float)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "percentile_20": float(np.percentile(array, 20)),
        "percentile_80": float(np.percentile(array, 80)),
    }


def plot(result):
    radii = np.asarray(result["radii"], dtype=float)
    means = np.asarray([result["validation"][str(rho)]["mean"] for rho in radii])
    low = np.asarray(
        [result["validation"][str(rho)]["percentile_20"] for rho in radii]
    )
    high = np.asarray(
        [result["validation"][str(rho)]["percentile_80"] for rho in radii]
    )
    nominal_mean = result["nominal_validation"]["mean"]
    net13_means = np.asarray(
        [result["reference_oos"]["net13"][str(rho)]["mean"] for rho in radii]
    )
    net16_means = np.asarray(
        [result["reference_oos"]["net16"][str(rho)]["mean"] for rho in radii]
    )
    best_index = int(np.argmin(means))

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    axis.plot(
        radii,
        net13_means,
        "^-",
        color="#7b3294",
        lw=1.8,
        ms=3.8,
        label="Width 13 OOS reference (seed 9999)",
    )
    axis.plot(
        radii,
        net16_means,
        "s-",
        color="#2166ac",
        lw=1.8,
        ms=3.5,
        label="Width 16 OOS reference (seed 9999)",
    )
    axis.plot(
        radii,
        means,
        "o-",
        color="#1f6f5f",
        lw=2.2,
        ms=4.2,
        label="Width 11 robust validation mean",
    )
    axis.fill_between(
        radii,
        low,
        high,
        color="#1f6f5f",
        alpha=0.16,
        label="20th--80th percentile across seeds",
    )
    axis.axhline(
        nominal_mean,
        color="#b13c2e",
        ls="--",
        lw=1.8,
        label=f"Width 11 nominal validation mean ({nominal_mean:.3f})",
    )
    axis.scatter(
        radii[best_index],
        means[best_index],
        s=70,
        facecolor="#f2c14e",
        edgecolor="#333333",
        zorder=5,
        label=f"Current minimum: $\\rho={radii[best_index]:g}$ ({means[best_index]:.3f})",
    )

    axis.set_xscale("log")
    axis.set_xlim(radii.min() / 1.15, radii.max() * 1.15)
    display_ticks = [
        rho
        for rho in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65)
        if rho in radii
    ]
    axis.set_xticks(display_ticks)
    axis.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axis.tick_params(axis="x", labelsize=8, rotation=45)
    axis.set_xlabel("Training radius $\\rho$")
    axis.set_ylabel("Mean cost")
    axis.set_title("Width 11 interim trend with OOS references")
    fig.suptitle(
        "Retailers R=3, N=50 | complete ten-seed radii only\n"
        "Width 11: independent validation | Widths 13/16: 10,000 OOS scenarios",
        fontsize=13,
    )
    axis.grid(axis="y", color="#cccccc", alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="best", fontsize=8.5)
    fig.subplots_adjust(bottom=0.2, top=0.82)
    fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    complete = completed_radii()
    if not complete:
        raise RuntimeError("No width-11 radius has complete ten-seed coverage")

    with REFERENCE_PATH.open() as stream:
        reference = json.load(stream)
    reference_meta = reference.get("meta", {})
    if (
        reference_meta.get("N_test") != 10_000
        or reference_meta.get("test_seed") != 9999
        or reference_meta.get("projection_method") != "gurobi_qp"
        or reference_meta.get("training_seeds") != SEEDS
    ):
        raise RuntimeError("Unexpected width-13/16 OOS reference metadata")
    reference_oos = {"net13": {}, "net16": {}}
    for rho, _, _ in complete:
        for width in reference_oos:
            entry = reference.get(width, {}).get(str(rho))
            if entry is None or len(entry.get("values", [])) != len(SEEDS):
                raise RuntimeError(f"Incomplete {width} OOS reference at rho={rho:g}")
            reference_oos[width][str(rho)] = entry

    validation_data = {
        seed: generate_data(VALIDATION_SIZE, VALIDATION_SEED_BASE + seed)
        for seed in SEEDS
    }

    nominal_values = []
    for seed in SEEDS:
        model = load_model(NOMINAL_DIR / f"seed_{seed}_nominal.pth")
        xi1, xi2 = validation_data[seed]
        nominal_values.append(model.mean_cost(xi1, xi2))

    validation = {}
    checkpoint_provenance = {}
    for rho, directory, summaries in complete:
        values = []
        checkpoint_provenance[str(rho)] = {}
        for seed, summary in zip(SEEDS, summaries):
            checkpoint = incumbent_checkpoint(directory, summary, seed)
            model = load_model(checkpoint)
            xi1, xi2 = validation_data[seed]
            values.append(model.mean_cost(xi1, xi2))
            checkpoint_provenance[str(rho)][str(seed)] = {
                "checkpoint": str(checkpoint),
                "incumbent_mp_iteration": int(summary["final_mp_iter"]),
                "best_ap_iteration": int(summary["best_ap_iter"]),
            }
        validation[str(rho)] = statistics(values)
        print(f"rho={rho:g}: validation mean={validation[str(rho)]['mean']:.6f}")

    means = {rho: values["mean"] for rho, values in validation.items()}
    selected_rho = min(means, key=lambda rho: (means[rho], float(rho)))
    result = {
        "meta": {
            "experiment": "R3_N50_width11_interim_validation_trend",
            "status": "completed",
            "interim": True,
            "hidden_widths": [11, 11],
            "n_train": 50,
            "training_seeds": SEEDS,
            "validation_size": VALIDATION_SIZE,
            "validation_seed_base": VALIDATION_SEED_BASE,
            "radius_inclusion_rule": "completed=True for all ten training seeds",
            "width11_test_data_accessed": False,
            "historical_reference_test_artifact_accessed": True,
            "historical_reference_test_seed": 9999,
            "historical_reference_test_size": 10000,
            "fresh_final_test_seed_not_accessed": 10000,
            "metric_boundary": "width 11 uses independent validation; widths 13 and 16 use historical seed-9999 OOS only as visual references",
            "projection_method": "gurobi_qp",
            "incumbent_rule": "minimum AP mean; AP iteration t evaluates MP iteration t-1",
            "interval": "20th-80th percentile across training seeds",
        },
        "radii": [rho for rho, _, _ in complete],
        "nominal_validation": statistics(nominal_values),
        "validation": validation,
        "reference_oos": reference_oos,
        "reference_source": str(REFERENCE_PATH),
        "current_minimum": {
            "rho": float(selected_rho),
            "mean_validation_cost": float(means[selected_rho]),
        },
        "checkpoint_provenance": checkpoint_provenance,
        "runtime": {
            "started_at_utc": started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_and_plot_time_s": time.perf_counter() - started,
        },
    }
    temporary_path = OUTPUT_JSON.with_suffix(".json.tmp")
    with temporary_path.open("w") as stream:
        json.dump(result, stream, indent=2)
    temporary_path.replace(OUTPUT_JSON)
    plot(result)
    print(f"Current minimum: rho={selected_rho}, mean={means[selected_rho]:.6f}")
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()