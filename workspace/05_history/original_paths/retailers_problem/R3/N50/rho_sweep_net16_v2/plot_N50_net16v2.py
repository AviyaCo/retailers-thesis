"""
plot_N50_net16v2.py -- IS+OOS figures for R=3, N=50 (net16 v2, truncnorm).

Matches the N200 plotting style:
  - Neural robust OOS with 20/80 percentile band
  - Neural robust IS
  - SAA-kNN OOS
  - Conditional clairvoyant lower bound
  - SRO-FA OOS with 20/80 percentile band, when available
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = "/data/home/aviyac/retailers_problem"
SWEEP_DIR = f"{BASE}/R3/N50/rho_sweep_net16_v2"
SRO_JSON = f"{BASE}/R3/N50/sro_fa/sro_fa_results_N50_R3.json"

EVAL_10K = os.path.join(SWEEP_DIR, "sweep_eval_net16_v2_10k.json")
OUT_10K = os.path.join(SWEEP_DIR, "figure_N50_net16v2_10k.png")

COND_CLAIRVOYANT_LB = 19.395

C_ROB = "#2166ac"
C_ROB_IS = "#6baed6"
C_SAA = "#1a9641"
C_KNN = "#e6a817"
C_SRO = "#d62728"


def load_sro_fa():
    if not os.path.exists(SRO_JSON):
        return (np.array([]),) * 4

    with open(SRO_JSON) as f:
        sro_data = json.load(f)

    rhos, means, lows, highs = [], [], [], []
    for rho_key in sorted(sro_data, key=float):
        seeds = sro_data[rho_key]
        oos = np.array([seeds[s]["oos"] for s in seeds], dtype=float)
        rhos.append(float(rho_key))
        means.append(float(oos.mean()))
        lows.append(float(np.percentile(oos, 20)))
        highs.append(float(np.percentile(oos, 80)))

    return np.array(rhos), np.array(means), np.array(lows), np.array(highs)


SRO_RHOS, SRO_OOS_MEAN, SRO_OOS_LO, SRO_OOS_HI = load_sro_fa()
SRO_FA_BEST = float(SRO_OOS_MEAN.min()) if SRO_OOS_MEAN.size else float("nan")


def make_figure(eval_json_path, out_path, n_test_label):
    if not os.path.exists(eval_json_path):
        print(f"[SKIP] {eval_json_path} not found.")
        return

    with open(eval_json_path) as f:
        data = json.load(f)

    benchmarks = data["benchmarks"]
    saa_lb = float(np.mean(benchmarks["saa_lb"]["oos"]))
    knn_oos = float(np.mean(benchmarks["saa_knn"]["oos"]))
    knn_selected_by_seed = benchmarks["saa_knn"].get("selected_k_by_seed")
    knn_selected_k = benchmarks["saa_knn"].get("selected_k")
    if knn_selected_by_seed and len(set(knn_selected_by_seed)) > 1:
        knn_label = "SAA-kNN (seed-wise k) OOS"
    elif knn_selected_by_seed:
        knn_label = f"SAA-kNN (k={knn_selected_by_seed[0]}) OOS"
    elif knn_selected_k:
        knn_label = f"SAA-kNN (best k={knn_selected_k}) OOS"
    else:
        knn_label = "SAA-kNN (k=5) OOS"

    rhos_pos = [rho for rho in data["rhos"] if 0 < rho <= 1.5]

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

    rhos_arr = np.array(rhos_pos)
    rob_oos_mean = np.array(rob_oos_mean)
    rob_oos_lo = np.array(rob_oos_lo)
    rob_oos_hi = np.array(rob_oos_hi)
    rob_is_mean = np.array(rob_is_mean)
    full_mask = np.array(n_seeds_list) == 10
    partial_mask = ~full_mask
    best_oos = float(rob_oos_mean.min()) if rob_oos_mean.size else float("nan")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        f"Three-stage inventory: R=3, N=50  (net16 v2, truncnorm)  --  "
        f"$N_{{test}}$={n_test_label}\n"
        "In-sample vs out-of-sample cost vs training radius",
        fontsize=11,
        fontweight="bold",
    )

    if full_mask.any():
        ax.plot(
            rhos_arr[full_mask],
            rob_oos_mean[full_mask],
            "o-",
            color=C_ROB,
            lw=2,
            ms=3.5,
            zorder=4,
            label=f"Neural robust OOS (mean +/- 20/80 pct, best={best_oos:.2f})",
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

    ax.axhline(knn_oos, color=C_KNN, ls="-.", lw=1.7, label=f"{knn_label}  ({knn_oos:.2f})")
    ax.axhline(
        COND_CLAIRVOYANT_LB,
        color=C_SAA,
        ls=(0, (7, 3)),
        lw=2.0,
        label=f"Cond. clairvoyant LB, seed=42 ({COND_CLAIRVOYANT_LB:.3f})",
    )

    if SRO_RHOS.size:
        plot_mask = (SRO_RHOS > 0) & (SRO_RHOS <= 1.5)
        ax.plot(
            SRO_RHOS[plot_mask],
            SRO_OOS_MEAN[plot_mask],
            "s--",
            color=C_SRO,
            lw=1.7,
            ms=3.5,
            zorder=3,
            label=f"SRO-FA OOS (mean +/- 20/80 pct, best={SRO_FA_BEST:.2f})",
        )
        ax.fill_between(
            SRO_RHOS[plot_mask],
            SRO_OOS_LO[plot_mask],
            SRO_OOS_HI[plot_mask],
            alpha=0.18,
            color=C_SRO,
        )

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

    ax.set_xlabel("Training radius  $\\rho$", fontsize=11)
    ax.set_ylabel("Mean cost", fontsize=11)
    ax.grid(True, which="major", axis="y", alpha=0.45, lw=0.8, color="#bbbbbb")
    ax.legend(
        fontsize=8.2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=2,
        frameon=True,
        framealpha=0.92,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    print(f"  Best OOS: rho={rhos_arr[int(np.argmin(rob_oos_mean))]:g}  OOS={best_oos:.4f}")
    if SRO_OOS_MEAN.size:
        print(f"  vs SRO-FA Best OOS: {SRO_FA_BEST:.4f}  -> improvement = {SRO_FA_BEST - best_oos:.4f}")
    print(f"  kNN OOS: {knn_oos:.4f}  SAA-LP LB hidden: {saa_lb:.4f}")


make_figure(EVAL_10K, OUT_10K, "10 000")
