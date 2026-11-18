"""
Side-by-side thesis figure for R=3, N in {50, 200, 800}.

Uses the 10k evaluation JSONs and hides the SAA-LP clairvoyant curve.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


BASE = "/data/home/aviyac/retailers_problem/R3"
OUT_PNG = os.path.join(BASE, "figure_R3_N50_N200_N800_side_by_side.png")
OUT_PDF = os.path.join(BASE, "figure_R3_N50_N200_N800_side_by_side.pdf")
OUT_LINEAR_PNG = os.path.join(
    BASE, "figure_R3_N50_N200_N800_side_by_side_linear.png"
)
OUT_LINEAR_PDF = os.path.join(
    BASE, "figure_R3_N50_N200_N800_side_by_side_linear.pdf"
)

COND_CLAIRVOYANT_LB = 19.395

C_ROB = "#2166ac"
C_ROB_IS = "#6baed6"
C_KNN = "#e6a817"
C_SAA = "#1a9641"
C_SRO = "#d62728"

CONFIGS = [
    {
        "N": 50,
        "eval_json": os.path.join(BASE, "N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json"),
        "sro_json": os.path.join(BASE, "N50/sro_fa/sro_fa_results_N50_R3.json"),
    },
    {
        "N": 200,
        "eval_json": os.path.join(BASE, "N200/rho_sweep_net16_v2/sweep_eval_net16_v2_N200_10k.json"),
        "sro_json": os.path.join(BASE, "N200/sro_fa/sro_fa_results_N200_R3.json"),
    },
    {
        "N": 800,
        "eval_json": os.path.join(BASE, "N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_10k.json"),
        "sro_json": os.path.join(BASE, "N800/sro_fa/sro_fa_results_N800_R3.json"),
    },
]


def load_sro(path):
    if not os.path.exists(path):
        return (np.array([]),) * 4

    with open(path) as f:
        data = json.load(f)

    rhos, means, lows, highs = [], [], [], []
    for rho_key in sorted(data, key=float):
        seeds = data[rho_key]
        if len(seeds) != 10:
            continue
        oos = np.array([seeds[s]["oos"] for s in seeds], dtype=float)
        rhos.append(float(rho_key))
        means.append(float(oos.mean()))
        lows.append(float(np.percentile(oos, 20)))
        highs.append(float(np.percentile(oos, 80)))

    return np.array(rhos), np.array(means), np.array(lows), np.array(highs)


def load_eval(path):
    with open(path) as f:
        data = json.load(f)

    benchmarks = data["benchmarks"]
    knn_oos = float(np.mean(benchmarks["saa_knn"]["oos"]))
    knn_selected_k = benchmarks["saa_knn"].get("selected_k")
    knn_selected_by_seed = benchmarks["saa_knn"].get("selected_k_by_seed")
    if not knn_selected_by_seed or len(knn_selected_by_seed) != 10:
        raise ValueError(f"Incomplete seed-wise kNN CV benchmark in {path}")
    rhos = np.array([rho for rho in data["rhos"] if 0 < rho <= 1.5], dtype=float)

    rob_oos_mean, rob_oos_lo, rob_oos_hi = [], [], []
    rob_is_mean, n_seeds = [], []
    for rho in rhos:
        entry = data["sweep"][str(float(rho))]
        oos = np.array(entry["rob_oos"], dtype=float)
        ins = np.array(entry["rob_is"], dtype=float)
        rob_oos_mean.append(float(oos.mean()))
        rob_oos_lo.append(float(np.percentile(oos, 20)) if len(oos) >= 2 else float(oos.mean()))
        rob_oos_hi.append(float(np.percentile(oos, 80)) if len(oos) >= 2 else float(oos.mean()))
        rob_is_mean.append(float(ins.mean()))
        n_seeds.append(int(entry["n_seeds"]))

    return {
        "rhos": rhos,
        "knn_oos": knn_oos,
        "knn_selected_k": knn_selected_k,
        "knn_selected_by_seed": knn_selected_by_seed,
        "rob_oos_mean": np.array(rob_oos_mean),
        "rob_oos_lo": np.array(rob_oos_lo),
        "rob_oos_hi": np.array(rob_oos_hi),
        "rob_is_mean": np.array(rob_is_mean),
        "n_seeds": np.array(n_seeds),
    }


def sparse_xticks(rhos):
    ticks, prev = [], None
    for rho in sorted(rhos):
        if prev is None or rho / prev > 1.25:
            ticks.append(rho)
            prev = rho
    return ticks


def plot_neural(ax, stats, label_oos, label_is):
    rhos = stats["rhos"]
    full_mask = stats["n_seeds"] == 10
    partial_mask = ~full_mask

    if full_mask.any():
        ax.plot(
            rhos[full_mask],
            stats["rob_oos_mean"][full_mask],
            "o-",
            color=C_ROB,
            lw=2.0,
            ms=3.5,
            zorder=4,
            label=label_oos,
        )
        ax.fill_between(
            rhos[full_mask],
            stats["rob_oos_lo"][full_mask],
            stats["rob_oos_hi"][full_mask],
            alpha=0.18,
            color=C_ROB,
            linewidth=0,
        )
        ax.plot(
            rhos[full_mask],
            stats["rob_is_mean"][full_mask],
            "^-",
            color=C_ROB_IS,
            lw=1.5,
            ms=3.0,
            zorder=4,
            label=label_is,
        )

    if partial_mask.any():
        if full_mask.any():
            last = np.where(full_mask)[0][-1]
            r_curve = np.concatenate([[rhos[last]], rhos[partial_mask]])
            oos_curve = np.concatenate([[stats["rob_oos_mean"][last]], stats["rob_oos_mean"][partial_mask]])
            is_curve = np.concatenate([[stats["rob_is_mean"][last]], stats["rob_is_mean"][partial_mask]])
        else:
            r_curve = rhos[partial_mask]
            oos_curve = stats["rob_oos_mean"][partial_mask]
            is_curve = stats["rob_is_mean"][partial_mask]

        ax.plot(r_curve, oos_curve, "o--", color=C_ROB, lw=1.2, ms=3.0, alpha=0.45, mfc="none")
        ax.plot(r_curve, is_curve, "^--", color=C_ROB_IS, lw=1.0, ms=3.0, alpha=0.45, mfc="none")


def plot_panel(ax, config, show_legend_labels=False, xscale="log"):
    stats = load_eval(config["eval_json"])
    sro_rhos, sro_mean, sro_lo, sro_hi = load_sro(config["sro_json"])

    label_oos = "Neural robust OOS" if show_legend_labels else None
    label_is = "Neural robust IS" if show_legend_labels else None
    selected_by_seed = stats["knn_selected_by_seed"]
    if selected_by_seed and len(set(selected_by_seed)) > 1:
        knn_label = "SAA-kNN (seed-wise k) OOS"
    elif selected_by_seed:
        knn_label = f"SAA-kNN (k={selected_by_seed[0]}) OOS"
    elif stats["knn_selected_k"]:
        knn_label = f"SAA-kNN (best k={stats['knn_selected_k']}) OOS"
    else:
        knn_label = "SAA-kNN OOS"
    plot_neural(ax, stats, label_oos, label_is)

    ax.axhline(
        stats["knn_oos"],
        color=C_KNN,
        ls="-.",
        lw=1.7,
        label=knn_label if show_legend_labels else None,
    )
    ax.axhline(
        COND_CLAIRVOYANT_LB,
        color=C_SAA,
        ls=(0, (7, 3)),
        lw=2.0,
        label="Conditional clairvoyant LB" if show_legend_labels else None,
    )

    if sro_rhos.size:
        mask = (sro_rhos > 0) & (sro_rhos <= 1.5)
        ax.plot(
            sro_rhos[mask],
            sro_mean[mask],
            "s--",
            color=C_SRO,
            lw=1.7,
            ms=3.5,
            zorder=3,
            label="SRO-FA OOS" if show_legend_labels else None,
        )
        ax.fill_between(
            sro_rhos[mask],
            sro_lo[mask],
            sro_hi[mask],
            alpha=0.18,
            color=C_SRO,
            linewidth=0,
        )

    ax.set_title(f"$N={config['N']}$", fontsize=12, pad=8)
    ax.set_xscale(xscale)
    for rho in stats["rhos"]:
        ax.axvline(rho, color="#e8e8e8", lw=0.5, zorder=0)

    if xscale == "log":
        ax.set_xticks(sparse_xticks(stats["rhos"]))
    else:
        ax.set_xlim(0.0, 1.5)
        ax.set_xticks(np.arange(0.0, 1.51, 0.25))
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=45)
    ax.tick_params(axis="y", which="major", labelsize=9)
    ax.set_xlabel("Training radius $\\rho$", fontsize=10)
    ax.grid(True, which="major", axis="y", alpha=0.45, lw=0.8, color="#bbbbbb")

    best_idx = int(np.argmin(stats["rob_oos_mean"]))
    return {
        "N": config["N"],
        "best_rho": float(stats["rhos"][best_idx]),
        "best_oos": float(stats["rob_oos_mean"][best_idx]),
        "knn_oos": float(stats["knn_oos"]),
        "knn_selected_k": stats["knn_selected_k"],
        "knn_selected_by_seed": stats["knn_selected_by_seed"],
        "n_seed_min": int(stats["n_seeds"].min()),
        "n_seed_max": int(stats["n_seeds"].max()),
    }


def main(xscale="log"):
    if xscale not in {"log", "linear"}:
        raise ValueError(f"Unsupported x scale: {xscale}")
    out_png = OUT_LINEAR_PNG if xscale == "linear" else OUT_PNG
    out_pdf = OUT_LINEAR_PDF if xscale == "linear" else OUT_PDF
    fig, axes = plt.subplots(1, 3, figsize=(13.05, 3.8), sharey=True)
    fig.suptitle("Retailers, R=3", fontsize=13, fontweight="bold", y=0.98)
    summaries = []

    for i, (ax, config) in enumerate(zip(axes, CONFIGS)):
        summaries.append(
            plot_panel(
                ax,
                config,
                show_legend_labels=(i == 0),
                xscale=xscale,
            )
        )
        if i == 0:
            ax.set_ylabel("Mean cost", fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=True,
        framealpha=0.94,
        fontsize=8.3,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.27, wspace=0.08)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")
    for summary in summaries:
        selected_by_seed = summary.get("knn_selected_by_seed")
        if selected_by_seed:
            k_summary = "[" + ",".join(str(k) for k in selected_by_seed) + "]"
        else:
            k_summary = str(summary.get("knn_selected_k"))
        print(
            f"N={summary['N']}: best rho={summary['best_rho']:g}, "
            f"best OOS={summary['best_oos']:.4f}, "
            f"SAA-kNN={summary['knn_oos']:.4f} (k_by_seed={k_summary}), "
            f"seeds={summary['n_seed_min']}-{summary['n_seed_max']}"
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["--linear"]:
        main(xscale="linear")
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit(
            "Usage: plot_R3_N50_N200_N800_side_by_side.py [--linear]"
        )
