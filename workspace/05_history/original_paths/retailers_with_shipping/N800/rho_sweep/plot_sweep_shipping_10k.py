"""
plot_sweep_shipping_10k.py  --  IS + OOS sweep figure for shipping problem, N=800.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SWEEP_DIR = "/data/home/aviyac/retailers_with_shipping/N800/rho_sweep_minibatch"
BASE_DIR  = "/data/home/aviyac/retailers_with_shipping/N800"
EVAL_JSON = os.path.join(SWEEP_DIR, "sweep_eval_shipping_10k.json")
SRO_JSON  = os.path.join(BASE_DIR, "sro_fa", "sro_fa_shipping_results_N800_R3.json")
OUT       = os.path.join(SWEEP_DIR, "figure_shipping_N800_10k.png")
OUT_PDF   = os.path.join(SWEEP_DIR, "figure_shipping_N800_10k.pdf")

if not os.path.exists(EVAL_JSON):
    print(f"[ERROR] {EVAL_JSON} not found.")
    exit(1)

with open(EVAL_JSON) as f:
    d = json.load(f)

bm         = d["benchmarks"]
saa_knn_oos = float(np.mean(bm["saa_knn"]["oos"])) if "saa_knn" in bm else None
knn_selected_k = bm.get("saa_knn", {}).get("selected_k")
knn_selected_by_seed = bm.get("saa_knn", {}).get("selected_k_by_seed")
saa_milp   = float(np.mean(bm["saa_milp"]["is"]))

C_OOS    = "#2166ac"
C_IS     = "#6baed6"
C_KNN    = "#e6a817"
C_SAA_LB = "#1a9641"
C_SRO    = "#d62728"

def load_sro_fa():
    if not os.path.exists(SRO_JSON):
        return (np.array([]),) * 4
    with open(SRO_JSON) as f:
        sro_data = json.load(f)
    rhos, means, lows, highs = [], [], [], []
    for rho_key in sorted(sro_data, key=float):
        seeds = sro_data[rho_key]
        if not seeds:
            continue
        oos = np.array([seeds[s]["oos"] for s in seeds], dtype=float)
        rhos.append(float(rho_key))
        means.append(float(oos.mean()))
        lows.append(float(np.percentile(oos, 20)) if len(oos) >= 2 else float(oos.mean()))
        highs.append(float(np.percentile(oos, 80)) if len(oos) >= 2 else float(oos.mean()))
    return np.array(rhos), np.array(means), np.array(lows), np.array(highs)

# ── per-rho stats ─────────────────────────────────────────────────────────────
rhos_pos = sorted(r for r in (float(k) for k in d["sweep"]) if 0 < r <= 2.0)

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
best_oos = float(rob_oos_mean.min())
sro_rhos, sro_mean, sro_lo, sro_hi = load_sro_fa()
sro_best = float(sro_mean.min()) if sro_mean.size else None

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.suptitle(
    "Inventory with shipping: R=3, N=800  (net16, truncnorm)  --  $N_{test}=10\\,000$\n"
    "In-sample vs out-of-sample cost vs training radius",
    fontsize=11, fontweight="bold")

# Robust OOS
if full_mask.any():
    ax.plot(rhos_arr[full_mask], rob_oos_mean[full_mask],
            "o-", color=C_OOS, lw=2, ms=3.5, zorder=4,
            label=f"Neural robust OOS (mean +/- 20/80 pct, best={best_oos:.2f})")
    ax.fill_between(rhos_arr[full_mask], rob_oos_lo[full_mask], rob_oos_hi[full_mask],
                    alpha=0.18, color=C_OOS)
if partial_mask.any():
    if full_mask.any():
        last = np.where(full_mask)[0][-1]
        r_c = np.concatenate([[rhos_arr[last]], rhos_arr[partial_mask]])
        m_c = np.concatenate([[rob_oos_mean[last]], rob_oos_mean[partial_mask]])
    else:
        r_c, m_c = rhos_arr[partial_mask], rob_oos_mean[partial_mask]
    ax.plot(r_c, m_c, "o--", color=C_OOS, lw=1.2, ms=3, alpha=0.45, mfc="none")

# Robust IS
if full_mask.any():
    ax.plot(rhos_arr[full_mask], rob_is_mean[full_mask],
            "^-", color=C_IS, lw=1.5, ms=3.0, zorder=3,
            label="Neural robust IS")
if partial_mask.any():
    rp = rhos_arr[partial_mask]; mp = rob_is_mean[partial_mask]
    if full_mask.any():
        last = np.where(full_mask)[0][-1]
        rp = np.concatenate([[rhos_arr[last]], rp])
        mp = np.concatenate([[rob_is_mean[last]], mp])
    ax.plot(rp, mp, "^--", color=C_IS, lw=1.0, ms=3.0, alpha=0.45, mfc="none")

# SRO-FA
if sro_rhos.size:
    sro_mask = (sro_rhos > 0) & (sro_rhos <= 2.0)
    ax.plot(
        sro_rhos[sro_mask],
        sro_mean[sro_mask],
        "s--",
        color=C_SRO,
        lw=1.7,
        ms=3.5,
        zorder=3,
        label=f"SRO-FA OOS (mean +/- 20/80 pct, best={sro_best:.2f})",
    )
    ax.fill_between(
        sro_rhos[sro_mask],
        sro_lo[sro_mask],
        sro_hi[sro_mask],
        alpha=0.18,
        color=C_SRO,
        linewidth=0,
    )

# Benchmark flat lines
if saa_knn_oos is not None:
    if knn_selected_by_seed and len(set(knn_selected_by_seed)) > 1:
        knn_label = "SAA-kNN (seed-wise k) OOS"
    elif knn_selected_by_seed:
        knn_label = f"SAA-kNN (k={knn_selected_by_seed[0]}) OOS"
    elif knn_selected_k:
        knn_label = f"SAA-kNN (k={knn_selected_k}) OOS"
    else:
        knn_label = "SAA-kNN OOS"
    ax.axhline(saa_knn_oos, color=C_KNN, lw=1.7, ls="-.",
               label=f"{knn_label}  ({saa_knn_oos:.2f})")
ax.axhline(saa_milp, color=C_SAA_LB, lw=2.0, ls=(0, (7, 3)),
           label=f"SAA-MILP IS  ({saa_milp:.2f})")

# Axes
ax.set_xscale("log")
for rho in rhos_pos:
    ax.axvline(rho, color="#e8e8e8", lw=0.5, zorder=0)
labeled, prev = [], None
for rho in rhos_pos:
    if prev is None or rho / prev > 1.25:
        labeled.append(rho)
        prev = rho
ax.set_xticks(labeled)
ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
ax.tick_params(axis="x", which="major", labelsize=8, rotation=45)
ax.set_xlabel("Training radius  $\\rho$", fontsize=11)
ax.set_ylabel("Mean cost", fontsize=11)
ax.grid(True, which="major", axis="y", alpha=0.45, lw=0.8, color="#bbbbbb")
ax.legend(fontsize=8.4, loc="upper center",
          bbox_to_anchor=(0.5, -0.3), ncol=2,
          frameon=True, framealpha=0.92)

fig.tight_layout()
fig.subplots_adjust(bottom=0.38)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"[Saved] {OUT}")
print(f"[Saved] {OUT_PDF}")
if saa_knn_oos is not None:
    print(f"  SAA-kNN OOS = {saa_knn_oos:.4f}")
if sro_best is not None:
    print(f"  SRO-FA best OOS = {sro_best:.4f}")
print(f"  SAA-MILP   = {saa_milp:.4f}")
print(f"  Best rob OOS = {best_oos:.4f}  at rho={rhos_arr[int(np.argmin(rob_oos_mean))]:.4f}")
