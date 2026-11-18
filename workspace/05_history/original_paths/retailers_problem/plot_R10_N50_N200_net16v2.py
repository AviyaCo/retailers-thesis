"""
plot_R10_N50_N200_net16v2.py -- Combined R10 net16 v2 sweep figure.

Creates a single paper-ready figure with three panels:
  - left:  R=10, N=50
    - center: R=10, N=200
    - right: R=10, N=800

Each panel shows neural robust OOS/IS, SAA-kNN, and the conditional lower
bound. The SAA-LP lower bound stays hidden, matching the standalone plots.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = "/data/home/aviyac/retailers_problem"
N50_JSON = os.path.join(BASE, "R10", "N50", "rho_sweep_net16_v2", "sweep_eval_net16_v2_N50_R10.json")
N200_JSON = os.path.join(BASE, "R10", "N200", "rho_sweep_net16_v2", "sweep_eval_net16_v2_N200_R10.json")
N800_JSON = os.path.join(BASE, "R10", "N800", "rho_sweep_net16_v2", "sweep_eval_net16_v2_N800_R10_10k.json")
OUT = os.path.join(BASE, "R10", "R10_N50_N200_N800_net16v2_side_by_side.png")
OUT_PDF = os.path.join(BASE, "R10", "R10_N50_N200_N800_net16v2_side_by_side.pdf")

COND_LB_JSONS = [
    (os.path.join(BASE, "R10", "saa_cond_lb_large.json"), "saa_cond_lb_large"),
    (os.path.join(BASE, "R10", "saa_cond_lb_1m.json"), "saa_cond_lb_1m"),
]

C_ROB = "#2166ac"
C_ROB_IS = "#6baed6"
C_SAA = "#1a9641"
C_KNN = "#e6a817"
C_SRO = "#d62728"

SRO_JSONS = {
    50: os.path.join(BASE, "R10", "sro_fa", "sro_fa_results_N50_R10.json"),
    200: os.path.join(BASE, "R10", "sro_fa", "sro_fa_results_N200_R10.json"),
    800: os.path.join(BASE, "R10", "sro_fa", "sro_fa_results_N800_R10.json"),
}


def load_cond_lb():
    for cond_lb_json, key in COND_LB_JSONS:
        if not os.path.exists(cond_lb_json):
            continue
        with open(cond_lb_json) as f:
            data = json.load(f)
        if key in data:
            return float(data[key])
    return None


def load_eval(eval_path):
    with open(eval_path) as f:
        return json.load(f)


def load_sro(path, rho_max=1.5):
    with open(path) as f:
        data = json.load(f)
    rhos, means, lows, highs = [], [], [], []
    for rho_key in sorted(data, key=float):
        rho = float(rho_key)
        seeds = data[rho_key]
        if rho <= 0 or rho > rho_max or len(seeds) != 10:
            continue
        oos = np.array([seeds[seed]["oos"] for seed in seeds], dtype=float)
        rhos.append(rho)
        means.append(float(oos.mean()))
        lows.append(float(np.percentile(oos, 20)))
        highs.append(float(np.percentile(oos, 80)))
    return tuple(np.array(values) for values in (rhos, means, lows, highs))


def knn_label(knn):
    selected_by_seed = knn.get("selected_k_by_seed", [])
    if len(selected_by_seed) != 10:
        raise ValueError("R10 figure requires 10 seed-wise cross-validated kNN choices")
    if len(set(selected_by_seed)) == 1:
        return f"SAA-kNN (k={selected_by_seed[0]}) OOS"
    return "SAA-kNN (seed-wise k) OOS"


def build_panel(ax, data, sro_path, title, cond_lb, rho_max=1.5):
    bm = data["benchmarks"]
    saa_lb = float(np.mean(bm["saa_lb"]["oos"]))
    knn = bm["saa_knn"]
    knn_oos = float(np.mean(knn["oos"]))

    rhos_pos = [rho for rho in data["rhos"] if 0 < rho <= rho_max]
    rob_oos_mean, rob_oos_lo, rob_oos_hi = [], [], []
    rob_is_mean, n_seeds_list = [], []
    for rho in rhos_pos:
        entry = data["sweep"][str(rho)]
        oos = np.array(entry["rob_oos"], dtype=float)
        ins = np.array(entry["rob_is"], dtype=float)
        rob_oos_mean.append(float(oos.mean()))
        rob_oos_lo.append(float(np.percentile(oos, 20)) if len(oos) >= 2 else float(oos.mean()))
        rob_oos_hi.append(float(np.percentile(oos, 80)) if len(oos) >= 2 else float(oos.mean()))
        rob_is_mean.append(float(ins.mean()))
        n_seeds_list.append(entry["n_seeds"])

    rob_oos_mean = np.array(rob_oos_mean)
    rob_oos_lo = np.array(rob_oos_lo)
    rob_oos_hi = np.array(rob_oos_hi)
    rob_is_mean = np.array(rob_is_mean)
    rhos_arr = np.array(rhos_pos)
    full_mask = np.array(n_seeds_list) == 10
    partial_mask = ~full_mask
    best_oos = float(rob_oos_mean.min()) if rob_oos_mean.size else float("nan")
    best_rho = float(rhos_arr[np.argmin(rob_oos_mean)]) if rob_oos_mean.size else float("nan")

    if full_mask.any():
        ax.plot(
            rhos_arr[full_mask],
            rob_oos_mean[full_mask],
            "o-",
            color=C_ROB,
            lw=2,
            ms=3.5,
            zorder=4,
            label="Neural robust OOS",
        )
        ax.fill_between(
            rhos_arr[full_mask],
            rob_oos_lo[full_mask],
            rob_oos_hi[full_mask],
            alpha=0.18,
            color=C_ROB,
        )
    if partial_mask.any():
        if full_mask.any():
            last = np.where(full_mask)[0][-1]
            r_curve = np.concatenate([[rhos_arr[last]], rhos_arr[partial_mask]])
            m_curve = np.concatenate([[rob_oos_mean[last]], rob_oos_mean[partial_mask]])
        else:
            r_curve, m_curve = rhos_arr[partial_mask], rob_oos_mean[partial_mask]
        ax.plot(r_curve, m_curve, "o--", color=C_ROB, lw=1.2, ms=3, alpha=0.45, mfc="none")

    if full_mask.any():
        ax.plot(
            rhos_arr[full_mask],
            rob_is_mean[full_mask],
            "^-",
            color=C_ROB_IS,
            lw=1.5,
            ms=3,
            zorder=4,
            label="Neural robust IS",
        )
    if partial_mask.any():
        if full_mask.any():
            last = np.where(full_mask)[0][-1]
            r_curve = np.concatenate([[rhos_arr[last]], rhos_arr[partial_mask]])
            m_curve = np.concatenate([[rob_is_mean[last]], rob_is_mean[partial_mask]])
        else:
            r_curve, m_curve = rhos_arr[partial_mask], rob_is_mean[partial_mask]
        ax.plot(r_curve, m_curve, "^--", color=C_ROB_IS, lw=1.0, ms=3, alpha=0.45, mfc="none")

    ax.axhline(knn_oos, color=C_KNN, ls="-.", lw=1.7, label=knn_label(knn))
    if cond_lb is not None:
        ax.axhline(
            cond_lb,
            color=C_SAA,
            ls=(0, (7, 3)),
            lw=2.0,
            label="Conditional clairvoyant LB",
        )

    sro_rhos, sro_mean, sro_low, sro_high = load_sro(sro_path, rho_max)
    if sro_rhos.size:
        ax.plot(sro_rhos, sro_mean, "s--", color=C_SRO, lw=1.7, ms=3.5,
                zorder=3, label="SRO-FA OOS")
        ax.fill_between(sro_rhos, sro_low, sro_high, alpha=0.18,
                        color=C_SRO, linewidth=0)

    ax.set_xscale("log")
    for rho in rhos_pos:
        ax.axvline(rho, color="#e8e8e8", lw=0.5, zorder=0)

    labeled, prev = [], None
    for rho in sorted(rhos_pos):
        if prev is None or rho / prev > 1.25:
            labeled.append(rho)
            prev = rho
    ax.set_xticks(labeled)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=45)
    ax.set_title(title, fontsize=12, pad=8)
    ax.tick_params(axis="y", which="major", labelsize=9)
    ax.grid(True, which="major", axis="y", alpha=0.45, lw=0.8, color="#bbbbbb")

    return {
        "best_rho": best_rho,
        "best_oos": best_oos,
        "knn_oos": knn_oos,
        "cond_lb": cond_lb,
        "saa_lb": saa_lb,
        "knn_selected_by_seed": knn["selected_k_by_seed"],
    }


def main():
    if not os.path.exists(N50_JSON):
        print(f"[SKIP] {N50_JSON} not found.")
        raise SystemExit(0)
    if not os.path.exists(N200_JSON):
        print(f"[SKIP] {N200_JSON} not found.")
        raise SystemExit(0)
    if not os.path.exists(N800_JSON):
        print(f"[SKIP] {N800_JSON} not found.")
        raise SystemExit(0)

    cond_lb = load_cond_lb()
    d50 = load_eval(N50_JSON)
    d200 = load_eval(N200_JSON)
    d800 = load_eval(N800_JSON)

    fig, axes = plt.subplots(1, 3, figsize=(13.05, 3.8), sharey=True)
    fig.suptitle("Retailers, R=10", fontsize=13, fontweight="bold", y=0.98)

    stats_50 = build_panel(axes[0], d50, SRO_JSONS[50], "$N=50$", cond_lb)
    stats_200 = build_panel(axes[1], d200, SRO_JSONS[200], "$N=200$", cond_lb)
    stats_800 = build_panel(axes[2], d800, SRO_JSONS[800], "$N=800$", cond_lb)

    axes[0].set_ylabel("Mean cost", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Training radius $\\rho$", fontsize=10)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        legend_handles,
        legend_labels,
        fontsize=8.3,
        loc="lower center",
        ncol=5,
        frameon=True,
        framealpha=0.92,
        bbox_to_anchor=(0.5, -0.015),
    )

    fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=0.27, wspace=0.08)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUT}")
    print(f"Saved {OUT_PDF}")
    for n, stats in ((50, stats_50), (200, stats_200), (800, stats_800)):
        print(f"N={n}: best rho={stats['best_rho']:g}, best OOS={stats['best_oos']:.4f}, "
              f"SAA-kNN={stats['knn_oos']:.4f} (k_by_seed={stats['knn_selected_by_seed']})")
    if cond_lb is not None:
        print(f"Cond. LB: {cond_lb:.4f}")


if __name__ == "__main__":
    main()