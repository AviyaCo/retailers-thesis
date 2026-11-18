"""
Side-by-side fixed-shipping R=3 sweep figures.

Reads the existing 10k evaluation JSONs and creates comparison figures using
the same plotting conventions as the standalone shipping sweep plots. Pending
panels are left as placeholders until their 10k evaluation JSONs exist.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


BASE_DIR = Path("/data/home/aviyac/retailers_with_shipping")
OUT_PNG = BASE_DIR / "figure_shipping_R3_N50_N200_N800_side_by_side.png"
OUT_PDF = BASE_DIR / "figure_shipping_R3_N50_N200_N800_side_by_side.pdf"
OUT_LEGACY_PNG = BASE_DIR / "figure_shipping_R3_N50_N200_side_by_side.png"
OUT_LEGACY_PDF = BASE_DIR / "figure_shipping_R3_N50_N200_side_by_side.pdf"

C_ROB = "#2166ac"
C_ROB_IS = "#6baed6"
C_KNN = "#e6a817"
C_COND = "#1a9641"
C_SRO = "#d62728"

CONFIGS = [
    {
        "N": 50,
        "eval_json": BASE_DIR / "N50" / "rho_sweep" / "sweep_eval_shipping_10k.json",
        "sro_json": BASE_DIR / "N50" / "sro_fa" / "sro_fa_shipping_results_N50_R3.json",
        "cond_saa_json": BASE_DIR / "N50" / "saa_cond_mip_1m_shipping_exact.json",
    },
    {
        "N": 200,
        "eval_json": BASE_DIR / "N200" / "rho_sweep" / "sweep_eval_shipping_10k.json",
        "sro_json": BASE_DIR / "N200" / "sro_fa" / "sro_fa_shipping_results_N200_R3.json",
        "cond_saa_json": BASE_DIR / "N200" / "saa_cond_mip_1m_shipping_exact.json",
    },
    {
        "N": 800,
        "eval_json": BASE_DIR / "N800" / "rho_sweep" / "sweep_eval_shipping_10k.json",
        "sro_json": BASE_DIR / "N800" / "sro_fa" / "sro_fa_shipping_results_N800_R3.json",
        "cond_saa_json": BASE_DIR / "N800" / "saa_cond_mip_1m_shipping_exact.json",
    },
]


def load_eval(eval_path):
    with eval_path.open() as eval_file:
        data = json.load(eval_file)

    benchmarks = data["benchmarks"]
    knn = benchmarks["saa_knn"]
    selected_by_seed = knn.get("selected_k_by_seed", [])
    if len(selected_by_seed) != 10 or len(knn.get("oos", [])) != 10:
        raise ValueError(f"Incomplete seed-wise kNN CV benchmark in {eval_path}")

    rhos = []
    rob_oos_mean = []
    rob_oos_low = []
    rob_oos_high = []
    rob_is_mean = []
    seed_counts = []

    for rho_key, entry in sorted(data["sweep"].items(), key=lambda item: float(item[0])):
        rho = float(rho_key)
        if rho <= 0 or rho > 1.5:
            continue

        rob_oos = np.array(entry.get("rob_oos", []), dtype=float)
        rob_is = np.array(entry.get("rob_is", []), dtype=float)
        if entry.get("n_seeds") != 10 or rob_oos.size != 10 or rob_is.size != 10:
            continue

        rhos.append(rho)
        rob_oos_mean.append(float(rob_oos.mean()))
        rob_oos_low.append(float(np.percentile(rob_oos, 20)) if rob_oos.size >= 2 else float(rob_oos.mean()))
        rob_oos_high.append(float(np.percentile(rob_oos, 80)) if rob_oos.size >= 2 else float(rob_oos.mean()))
        rob_is_mean.append(float(rob_is.mean()))
        seed_counts.append(int(entry.get("n_seeds", rob_oos.size)))

    return {
        "rhos": np.array(rhos, dtype=float),
        "rob_oos_mean": np.array(rob_oos_mean, dtype=float),
        "rob_oos_low": np.array(rob_oos_low, dtype=float),
        "rob_oos_high": np.array(rob_oos_high, dtype=float),
        "rob_is_mean": np.array(rob_is_mean, dtype=float),
        "seed_counts": np.array(seed_counts, dtype=int),
        "knn_oos": float(np.mean(knn["oos"])),
        "knn_selected_k": knn.get("selected_k"),
        "knn_selected_by_seed": knn.get("selected_k_by_seed"),
    }


def load_sro(sro_path):
    if not sro_path.exists():
        return (np.array([]),) * 4

    with sro_path.open() as sro_file:
        data = json.load(sro_file)

    rhos = []
    means = []
    lows = []
    highs = []
    for rho_key in sorted(data, key=float):
        rho = float(rho_key)
        seeds = data[rho_key]
        if rho <= 0 or rho > 1.5 or len(seeds) != 10:
            continue
        values = np.array([seeds[seed]["oos"] for seed in seeds], dtype=float)
        rhos.append(rho)
        means.append(float(values.mean()))
        lows.append(float(np.percentile(values, 20)) if values.size >= 2 else float(values.mean()))
        highs.append(float(np.percentile(values, 80)) if values.size >= 2 else float(values.mean()))

    return np.array(rhos), np.array(means), np.array(lows), np.array(highs)


def load_conditional_saa(cond_saa_path):
    if not cond_saa_path.exists():
        return None
    with cond_saa_path.open() as cond_saa_file:
        data = json.load(cond_saa_file)
    value = data.get("saa_cond_mip_1m_shipping_exact")
    return float(value) if value is not None else None


def sparse_xticks(rhos):
    ticks = []
    previous = None
    for rho in sorted(rhos):
        if previous is None or rho / previous > 1.25:
            ticks.append(rho)
            previous = rho
    return ticks


def knn_label(stats):
    selected_by_seed = stats["knn_selected_by_seed"]
    selected_k = stats["knn_selected_k"]
    if selected_by_seed and len(set(selected_by_seed)) > 1:
        return "SAA-kNN (seed-wise k) OOS"
    if selected_by_seed:
        return f"SAA-kNN (k={selected_by_seed[0]}) OOS"
    if selected_k:
        return f"SAA-kNN (k={selected_k}) OOS"
    return "SAA-kNN OOS"


def plot_panel(axis, config, show_legend, stats, fallback_rhos):
    sro_rhos, sro_mean, sro_low, sro_high = load_sro(config["sro_json"])
    cond_saa = load_conditional_saa(config["cond_saa_json"])

    if stats is None:
        rhos = np.array(fallback_rhos, dtype=float)
        if rhos.size == 0:
            rhos = sro_rhos[sro_rhos > 0]
    else:
        full_mask = stats["seed_counts"] == 10
        rhos = stats["rhos"]

        if full_mask.any():
            axis.plot(
                rhos[full_mask],
                stats["rob_oos_mean"][full_mask],
                "o-",
                color=C_ROB,
                lw=2.0,
                ms=3.5,
                zorder=4,
                label="Neural robust OOS" if show_legend else None,
            )
            axis.fill_between(
                rhos[full_mask],
                stats["rob_oos_low"][full_mask],
                stats["rob_oos_high"][full_mask],
                alpha=0.18,
                color=C_ROB,
                linewidth=0,
            )
            axis.plot(
                rhos[full_mask],
                stats["rob_is_mean"][full_mask],
                "^-",
                color=C_ROB_IS,
                lw=1.5,
                ms=3.0,
                zorder=4,
                label="Neural robust IS" if show_legend else None,
            )

        axis.axhline(
            stats["knn_oos"],
            color=C_KNN,
            ls="-.",
            lw=1.7,
            label=knn_label(stats) if show_legend else None,
        )
    if stats is None:
        axis.text(
            0.5,
            0.56,
            "10k eval pending",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )

    if cond_saa is not None:
        axis.axhline(
            cond_saa,
            color=C_COND,
            ls=(0, (7, 3)),
            lw=2.0,
            label="Conditional clairvoyant LB" if show_legend else None,
        )

    if sro_rhos.size:
        positive_mask = sro_rhos > 0
        axis.plot(
            sro_rhos[positive_mask],
            sro_mean[positive_mask],
            "s--",
            color=C_SRO,
            lw=1.7,
            ms=3.5,
            zorder=3,
            label="SRO-FA OOS" if show_legend else None,
        )
        axis.fill_between(
            sro_rhos[positive_mask],
            sro_low[positive_mask],
            sro_high[positive_mask],
            alpha=0.14,
            color=C_SRO,
            linewidth=0,
        )

    axis.set_title(f"$N={config['N']}$", fontsize=12, pad=8)
    axis.set_xscale("log")
    for rho in rhos:
        axis.axvline(rho, color="#e8e8e8", lw=0.5, zorder=0)
    if rhos.size:
        axis.set_xlim(float(rhos.min()) / 1.15, float(rhos.max()) * 1.15)
        axis.set_xticks(sparse_xticks(rhos))
    else:
        axis.set_xlim(0.01, 1.0)
        axis.set_xticks([0.01, 0.1, 1.0])
    axis.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    axis.tick_params(axis="x", which="major", labelsize=8, rotation=45)
    axis.tick_params(axis="y", which="major", labelsize=9)
    axis.set_xlabel("Training radius $\\rho$", fontsize=10)
    axis.grid(True, which="major", axis="y", alpha=0.45, lw=0.8, color="#bbbbbb")

    if stats is None:
        return {
            "N": config["N"],
            "eval_ready": False,
            "cond_saa": cond_saa,
            "sro_points": int(np.sum(sro_rhos > 0)),
        }

    best_index = int(np.argmin(stats["rob_oos_mean"]))
    return {
        "N": config["N"],
        "eval_ready": True,
        "best_rho": float(rhos[best_index]),
        "best_oos": float(stats["rob_oos_mean"][best_index]),
        "knn_oos": float(stats["knn_oos"]),
        "cond_saa": cond_saa,
        "n_seed_min": int(stats["seed_counts"].min()),
        "n_seed_max": int(stats["seed_counts"].max()),
    }


def render_figure(configs, out_png, out_pdf, stats_by_n, fallback_rhos):
    width = 4.7 if len(configs) == 2 else 4.35
    fig, axes = plt.subplots(1, len(configs), figsize=(width * len(configs), 3.8), sharey=True)
    axes = np.atleast_1d(axes)

    summaries = []
    for panel_index, (axis, config) in enumerate(zip(axes, configs)):
        summaries.append(
            plot_panel(
                axis,
                config,
                show_legend=(panel_index == 0),
                stats=stats_by_n.get(config["N"]),
                fallback_rhos=fallback_rhos,
            )
        )
        if panel_index == 0:
            axis.set_ylabel("Mean cost", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(handles)),
            frameon=True,
            framealpha=0.94,
            fontsize=8.3,
            bbox_to_anchor=(0.5, -0.015),
        )

    fig.suptitle("Fixed-shipping retailers, R=3", fontsize=13, fontweight="bold", y=0.98)
    left = 0.075 if len(configs) == 2 else 0.055
    fig.subplots_adjust(left=left, right=0.99, top=0.82, bottom=0.27, wspace=0.08)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    return summaries


def main():
    stats_by_n = {
        config["N"]: load_eval(config["eval_json"]) if config["eval_json"].exists() else None
        for config in CONFIGS
    }
    rho_chunks = [stats["rhos"] for stats in stats_by_n.values() if stats is not None and stats["rhos"].size]
    fallback_rhos = np.unique(np.concatenate(rho_chunks)) if rho_chunks else np.array([0.01, 0.1, 1.0])

    summaries = render_figure(CONFIGS, OUT_PNG, OUT_PDF, stats_by_n, fallback_rhos)
    render_figure(CONFIGS[:2], OUT_LEGACY_PNG, OUT_LEGACY_PDF, stats_by_n, fallback_rhos)

    for summary in summaries:
        if summary["eval_ready"]:
            print(
                f"N={summary['N']}: best rho={summary['best_rho']:g}, "
                f"best OOS={summary['best_oos']:.4f}, "
                f"SAA-kNN={summary['knn_oos']:.4f}, "
                f"cond LB={summary['cond_saa']:.4f}, "
                f"seeds={summary['n_seed_min']}-{summary['n_seed_max']}"
            )
        else:
            cond_text = f"cond LB={summary['cond_saa']:.4f}" if summary["cond_saa"] is not None else "cond LB missing"
            print(
                f"N={summary['N']}: 10k eval pending, {cond_text}, "
                f"SRO points={summary['sro_points']}"
            )


if __name__ == "__main__":
    main()