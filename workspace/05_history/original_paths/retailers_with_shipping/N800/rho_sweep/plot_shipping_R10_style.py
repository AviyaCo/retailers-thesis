"""
R10-style sweep plot for shipping N800, plus a convergence-gap diagnostic plot.

Reads:
  - sweep_eval_shipping_10k.json
  - rho_0p500/seed_9_summary.json (default diagnostic target)

Saves:
    - figure_shipping_N800_R10_style.png/.pdf
  - convergence_gap_rho0p500_seed9.png
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SWEEP_DIR = Path("/data/home/aviyac/retailers_with_shipping/N800/rho_sweep")
BASE_DIR = Path("/data/home/aviyac/retailers_with_shipping/N800")
EVAL_JSON = SWEEP_DIR / "sweep_eval_shipping_10k.json"
SUMMARY_JSON = SWEEP_DIR / "rho_0p500" / "seed_9_summary.json"
SRO_JSON = BASE_DIR / "sro_fa" / "sro_fa_shipping_results_N800_R3.json"
COND_SAA_JSON = BASE_DIR / "saa_cond_mip_1m_shipping_exact.json"
OUT_MAIN_PNG = SWEEP_DIR / "figure_shipping_N800_R10_style.png"
OUT_MAIN_PDF = SWEEP_DIR / "figure_shipping_N800_R10_style.pdf"
OUT_DIAG_PNG = SWEEP_DIR / "convergence_gap_rho0p500_seed9.png"

# Palette aligned with the R10 figure style.
C_ROB = "#2166ac"
C_ROB_IS = "#6baed6"
C_KNN = "#e6a817"
C_SRO = "#d62728"


def _extract(data, key):
    items = sorted(
        [(float(r), v) for r, v in data["sweep"].items() if float(r) > 0],
        key=lambda x: x[0],
    )

    rhos, means, los, his = [], [], [], []
    for r, v in items:
        arr = np.array(v.get(key, []), dtype=float)
        if v.get("n_seeds") != 10 or arr.size != 10:
            continue
        rhos.append(r)
        means.append(float(arr.mean()))
        los.append(float(np.percentile(arr, 20)) if arr.size >= 2 else float(arr.mean()))
        his.append(float(np.percentile(arr, 80)) if arr.size >= 2 else float(arr.mean()))

    return np.array(rhos), np.array(means), np.array(los), np.array(his)


def _load_sro_fa_oos():
    if not SRO_JSON.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])

    with SRO_JSON.open() as f:
        sro_data = json.load(f)

    rhos, means, lows, highs = [], [], [], []
    for rho_key in sorted(sro_data, key=float):
        seeds = sro_data[rho_key]
        if len(seeds) != 10:
            continue
        oos = np.array([seeds[s]["oos"] for s in seeds], dtype=float)
        rhos.append(float(rho_key))
        means.append(float(oos.mean()))
        lows.append(float(np.percentile(oos, 20)) if len(oos) >= 2 else float(oos.mean()))
        highs.append(float(np.percentile(oos, 80)) if len(oos) >= 2 else float(oos.mean()))

    return np.array(rhos), np.array(means), np.array(lows), np.array(highs)


def _load_conditional_saa_value():
    if not COND_SAA_JSON.exists():
        return None
    with COND_SAA_JSON.open() as f:
        d = json.load(f)
    val = d.get("saa_cond_mip_1m_shipping_exact")
    return float(val) if val is not None else None


def _knn_label(knn):
    selected_by_seed = knn.get("selected_k_by_seed")
    selected_k = knn.get("selected_k")
    if selected_by_seed and len(set(selected_by_seed)) > 1:
        return "SAA-kNN (seed-wise k) OOS"
    if selected_by_seed:
        return f"SAA-kNN (k={selected_by_seed[0]}) OOS"
    if selected_k:
        return f"SAA-kNN (k={selected_k}) OOS"
    return "SAA-kNN OOS"


def _plot_sweep_figure(data):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle("N = 800 training scenarios  (shipping, R = 3)", fontsize=13, fontweight="bold", y=0.99)

    r_oos, m_oos, lo_oos, hi_oos = _extract(data, "rob_oos")
    r_is, m_is, lo_is, hi_is = _extract(data, "rob_is")

    ax.plot(r_oos, m_oos, "o-", color=C_ROB, lw=2.0, ms=3,
            label="Neural robust OOS (mean +/- 20/80 pct)", zorder=4)
    ax.fill_between(r_oos, lo_oos, hi_oos, alpha=0.18, color=C_ROB)

    ax.plot(r_is, m_is, "^-", color=C_ROB_IS, lw=1.5, ms=3,
            label="Neural robust IS", zorder=3)
    ax.fill_between(r_is, lo_is, hi_is, alpha=0.12, color=C_ROB_IS)

    bm = data["benchmarks"]
    knn = bm["saa_knn"]
    knn_oos = float(np.mean(knn["oos"]))
    sro_rhos, sro_mean, sro_lo, sro_hi = _load_sro_fa_oos()
    cond_saa = _load_conditional_saa_value()

    ax.axhline(knn_oos, color=C_KNN, ls="-.", lw=1.7,
               label=f"{_knn_label(knn)}  ({knn_oos:.2f})")
    if cond_saa is not None:
        ax.axhline(cond_saa, color="#1a9641", ls=(0, (7, 3)), lw=2.0,
                   label=f"Cond. clairvoyant LB, seed=42 ({cond_saa:.3f})")

    if sro_rhos.size:
        mask = sro_rhos > 0
        sro_best = float(sro_mean[mask].min()) if np.any(mask) else float(sro_mean.min())
        ax.plot(sro_rhos[mask], sro_mean[mask], "s--", color=C_SRO, lw=1.7, ms=3.5,
                label=f"SRO-FA OOS (mean +/- 20/80 pct, best={sro_best:.2f})", zorder=3)
        ax.fill_between(sro_rhos[mask], sro_lo[mask], sro_hi[mask], alpha=0.14, color=C_SRO)

    all_r = np.unique(np.concatenate([r_oos, r_is]))
    all_r = all_r[all_r > 0]

    ax.set_xscale("log")
    for r in all_r:
        ax.axvline(r, color="#e0e0e0", lw=0.6, zorder=0)
    prev, ticks = None, []
    for r in sorted(all_r):
        if prev is None or r / prev > 1.25:
            ticks.append(r)
            prev = r
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=45)
    ax.grid(True, which="major", axis="y", alpha=0.45, lw=0.8, color="#bbbbbb")
    ax.set_xlabel("Training radius  $\\rho$", fontsize=11)
    ax.set_ylabel("Mean cost", fontsize=11)
    ax.set_title("In-sample vs out-of-sample cost vs training radius", fontsize=10)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=2, frameon=True, framealpha=0.92)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    fig.savefig(OUT_MAIN_PNG, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_MAIN_PDF, bbox_inches="tight")
    plt.close(fig)


def _plot_convergence_gap(summary):
    traj = summary["ccg_trajectory"]
    iters = np.array([r["iter"] for r in traj], dtype=int)
    delta = np.array([r["delta_pct"] for r in traj], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, delta, "o-", color="#b2182b", ms=3, lw=1.5)
    ax.axhline(1.0, color="#333333", ls="--", lw=1.2, label="Convergence threshold = 1%")

    # Highlight best/closest point to the threshold.
    idx = int(np.argmin(delta))
    ax.scatter([iters[idx]], [delta[idx]], color="#2166ac", zorder=5, s=40,
               label=f"Best gap: {delta[idx] - 1.0:.4f}% at iter {iters[idx]}")

    ax.set_xlabel("CCG iteration")
    ax.set_ylabel("Delta (%) = (AP mean - MP loss) / |MP loss| * 100")
    ax.set_title("Convergence Gap: rho=0.5, seed=9")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIAG_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    if not EVAL_JSON.exists():
        raise FileNotFoundError(f"Missing eval file: {EVAL_JSON}")
    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(f"Missing summary file: {SUMMARY_JSON}")

    with EVAL_JSON.open() as f:
        data = json.load(f)
    with SUMMARY_JSON.open() as f:
        summary = json.load(f)

    _plot_sweep_figure(data)
    _plot_convergence_gap(summary)

    traj = summary["ccg_trajectory"]
    best = min(traj, key=lambda x: x["delta_pct"])
    last = traj[-1]

    print(f"Saved {OUT_MAIN_PNG}")
    print(f"Saved {OUT_MAIN_PDF}")
    print(f"Saved {OUT_DIAG_PNG}")
    print(f"Best delta_pct = {best['delta_pct']:.6f}% at iter {best['iter']} (gap to 1%: {best['delta_pct'] - 1.0:.6f}%)")
    print(f"Last delta_pct = {last['delta_pct']:.6f}% at iter {last['iter']} (gap to 1%: {last['delta_pct'] - 1.0:.6f}%)")


if __name__ == "__main__":
    main()
