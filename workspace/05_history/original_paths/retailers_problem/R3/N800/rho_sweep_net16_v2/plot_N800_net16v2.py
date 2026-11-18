"""
plot_N800_net16v2.py  —  IS+OOS figure for R=3, N=800 (net16 v2, truncnorm).
Produces one figure using the 10k test-set evaluation JSON.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE      = "/data/home/aviyac/retailers_problem"
SWEEP_DIR = f"{BASE}/R3/N800/rho_sweep_net16_v2"
SRO_JSON  = f"{BASE}/R3/N800/sro_fa/sro_fa_results_N800_R3.json"

EVAL_10K  = os.path.join(SWEEP_DIR, "sweep_eval_net16_v2_N800_10k.json")
OUT_10K   = os.path.join(SWEEP_DIR, "figure_N800_net16v2_10k.png")
OUT_COMBINED_10K = f"{BASE}/R3/N800/figure_combined_N800_R3.png"

# Load SRO-FA sweep results
sro_rhos, sro_oos_mean, sro_oos_lo, sro_oos_hi = [], [], [], []
if os.path.exists(SRO_JSON):
    with open(SRO_JSON) as f:
        sro_data = json.load(f)
    for rho_key in sorted(sro_data, key=float):
        seeds = sro_data[rho_key]
        oos_l = np.array([seeds[s]["oos"] for s in seeds])
        sro_rhos.append(float(rho_key))
        sro_oos_mean.append(float(oos_l.mean()))
        sro_oos_lo.append(float(np.percentile(oos_l, 20)))
        sro_oos_hi.append(float(np.percentile(oos_l, 80)))
    sro_rhos     = np.array(sro_rhos)
    sro_oos_mean = np.array(sro_oos_mean)
    sro_oos_lo   = np.array(sro_oos_lo)
    sro_oos_hi   = np.array(sro_oos_hi)
SRO_FA_BEST_RHO = float(min(sro_oos_mean)) if len(sro_oos_mean) else float('nan')
COND_CLAIRVOYANT_LB = 19.395

C_ROB    = "#2166ac"
C_ROB_IS = "#6baed6"
C_NOM    = "#555555"
C_SAA    = "#1a9641"
C_KNN    = "#e6a817"
C_SRO    = "#d62728"


def make_figure(eval_json_path, out_path, n_test_label, extra_out_path=None):
    if not os.path.exists(eval_json_path):
        print(f"[SKIP] {eval_json_path} not found.")
        return

    with open(eval_json_path) as f:
        d = json.load(f)

    bm        = d["benchmarks"]
    saa_lb    = float(np.mean(bm["saa_lb"]["oos"]))
    nom_oos   = float(np.mean(d["sweep"]["0.0"]["nom_oos"]))
    nom_is    = float(np.mean(d["sweep"]["0.0"]["nom_is"]))
    knn_oos   = float(np.mean(bm["saa_knn"]["oos"]))
    knn_selected_by_seed = bm["saa_knn"].get("selected_k_by_seed")
    knn_selected_k = bm["saa_knn"].get("selected_k")
    if knn_selected_by_seed and len(set(knn_selected_by_seed)) > 1:
        knn_label = "SAA-kNN (seed-wise k) OOS"
    elif knn_selected_by_seed:
        knn_label = f"SAA-kNN (k={knn_selected_by_seed[0]}) OOS"
    elif knn_selected_k:
        knn_label = f"SAA-kNN (best k={knn_selected_k}) OOS"
    else:
        knn_label = "SAA-kNN (k=5) OOS"

    # ── per-rho stats ──────────────────────────────────────────────────────────
    rhos_pos = [r for r in d["rhos"] if 0 < r <= 1.5]

    rob_oos_mean, rob_oos_lo, rob_oos_hi = [], [], []
    rob_is_mean, n_seeds_list = [], []
    for rho in rhos_pos:
        e   = d["sweep"][str(rho)]
        oos = np.array(e["rob_oos"], dtype=float)
        ris = np.array(e["rob_is"],  dtype=float)
        rob_oos_mean.append(oos.mean())
        rob_oos_lo.append(np.percentile(oos, 20) if len(oos) >= 2 else oos.mean())
        rob_oos_hi.append(np.percentile(oos, 80) if len(oos) >= 2 else oos.mean())
        rob_is_mean.append(ris.mean())
        n_seeds_list.append(e["n_seeds"])

    rob_oos_mean = np.array(rob_oos_mean)
    rob_oos_lo   = np.array(rob_oos_lo)
    rob_oos_hi   = np.array(rob_oos_hi)
    rob_is_mean  = np.array(rob_is_mean)
    rhos_arr     = np.array(rhos_pos)
    full_mask    = np.array(n_seeds_list) == 10
    partial_mask = ~full_mask
    best_oos = float(rob_oos_mean.min()) if rob_oos_mean.size > 0 else float('nan')

    # ── figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.suptitle(
        f"Three-stage inventory: R=3, N=800  (net16 v2, truncnorm)  —  $N_{{test}}$={n_test_label}\n"
        "In-sample vs out-of-sample cost vs training radius",
        fontsize=11, fontweight="bold")

    # Robust OOS
    if full_mask.any():
        ax.plot(rhos_arr[full_mask], rob_oos_mean[full_mask],
                "o-", color=C_ROB, lw=2, ms=3.5, zorder=4,
                label=f"Neural robust OOS (mean \u00b1 20/80 pct, best={best_oos:.2f})")
        ax.fill_between(rhos_arr[full_mask], rob_oos_lo[full_mask], rob_oos_hi[full_mask],
                        alpha=0.18, color=C_ROB)
    if partial_mask.any():
        if full_mask.any():
            last = np.where(full_mask)[0][-1]
            r_c = np.concatenate([[rhos_arr[last]], rhos_arr[partial_mask]])
            m_c = np.concatenate([[rob_oos_mean[last]], rob_oos_mean[partial_mask]])
            ax.plot(r_c, m_c, "o--", color=C_ROB, lw=1.2, ms=3, alpha=0.45, mfc="none")
        else:
            r_c, m_c = rhos_arr[partial_mask], rob_oos_mean[partial_mask]
            ax.plot(r_c, m_c, "o--", color=C_ROB, lw=1.2, ms=3, alpha=0.45, mfc="none",
                    label=f"Neural robust OOS (partial, best={best_oos:.2f})")

    # Robust IS
    if full_mask.any():
        ax.plot(rhos_arr[full_mask], rob_is_mean[full_mask],
                "^-", color=C_ROB_IS, lw=1.5, ms=3, zorder=4,
                label="Neural robust IS")
    if partial_mask.any():
        if full_mask.any():
            last = np.where(full_mask)[0][-1]
            r_c = np.concatenate([[rhos_arr[last]], rhos_arr[partial_mask]])
            m_c = np.concatenate([[rob_is_mean[last]], rob_is_mean[partial_mask]])
            ax.plot(r_c, m_c, "^--", color=C_ROB_IS, lw=1.0, ms=3, alpha=0.45, mfc="none")
        else:
            r_c, m_c = rhos_arr[partial_mask], rob_is_mean[partial_mask]
            ax.plot(r_c, m_c, "^--", color=C_ROB_IS, lw=1.0, ms=3, alpha=0.45, mfc="none",
                    label="Neural robust IS (partial)")

    # Benchmarks
    ax.axhline(knn_oos, color=C_KNN, ls="-.", lw=1.7,
               label=f"{knn_label}  ({knn_oos:.2f})")
    ax.axhline(COND_CLAIRVOYANT_LB, color=C_SAA, ls=(0,(7,3)), lw=2.0,
               label=f"Cond. clairvoyant LB, seed=42 ({COND_CLAIRVOYANT_LB:.3f})")

    # SRO-FA sweep curve
    if len(sro_rhos) > 0:
        plot_mask = (sro_rhos > 0) & (sro_rhos <= 1.5)
        ax.plot(sro_rhos[plot_mask], sro_oos_mean[plot_mask],
                "s--", color=C_SRO, lw=1.7, ms=3.5, zorder=3,
                label=f"SRO-FA OOS (mean \u00b1 20/80 pct, best={SRO_FA_BEST_RHO:.2f})")
        ax.fill_between(sro_rhos[plot_mask], sro_oos_lo[plot_mask], sro_oos_hi[plot_mask],
                        alpha=0.18, color=C_SRO)

    best_idx = int(np.argmin(rob_oos_mean)) if rob_oos_mean.size > 0 else 0
    best_rho = rhos_arr[best_idx] if rob_oos_mean.size > 0 else 0.0

    # x-axis
    ax.set_xscale("log")
    for r in rhos_pos:
        ax.axvline(r, color="#e8e8e8", lw=0.5, zorder=0)
    labeled, prev = [], None
    for r in sorted(rhos_pos):
        if prev is None or r / prev > 1.25:
            labeled.append(r)
            prev = r
    ax.set_xticks(labeled)
    ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=45)

    ax.set_xlabel("Training radius  $\\rho$", fontsize=11)
    ax.set_ylabel("Mean cost", fontsize=11)
    ax.grid(True, which="major", axis="y", alpha=0.45, lw=0.8, color="#bbbbbb")
    ax.legend(fontsize=8.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.3), ncol=2,
              frameon=True, framealpha=0.92)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if extra_out_path is not None:
        plt.savefig(extra_out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    if extra_out_path is not None:
        print(f"Saved {extra_out_path}")

    if rob_oos_mean.size > 0:
        best_idx = int(np.argmin(rob_oos_mean))
        best_rho = rhos_arr[best_idx]
        best_oos = rob_oos_mean[best_idx]
        print(f"  Best OOS: rho={best_rho:g}  OOS={best_oos:.4f}")
        if not np.isnan(SRO_FA_BEST_RHO):
            print(f"  vs SRO-FA Best OOS: {SRO_FA_BEST_RHO:.4f}  -> improvement = {SRO_FA_BEST_RHO - best_oos:.4f}")
        print(f"  kNN OOS: {knn_oos:.4f}  SAA-LP LB hidden: {saa_lb:.4f}")


make_figure(EVAL_10K, OUT_10K, "10 000", extra_out_path=OUT_COMBINED_10K)
