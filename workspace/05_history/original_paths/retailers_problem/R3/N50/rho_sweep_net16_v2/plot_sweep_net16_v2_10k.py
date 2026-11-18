"""
plot_sweep_net16_v2_10k.py  —  Same as plot_sweep_net16_v2.py but reads 10k eval JSON.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EVAL_JSON = "/data/home/aviyac/retailers_problem/R3/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json"
OUT       = "/data/home/aviyac/retailers_problem/R3/N50/rho_sweep_net16_v2/sweep_plot_net16_v2_10k.png"

with open(EVAL_JSON) as f:
    d = json.load(f)

bm        = d["benchmarks"]
saa_lb    = float(np.mean(bm["saa_lb"]["oos"]))
saa_lb_is = float(np.mean(bm["saa_lb"]["is"]))
nom_oos   = float(np.mean(d["sweep"]["0.0"]["nom_oos"]))
nom_is    = float(np.mean(d["sweep"]["0.0"]["nom_is"]))
knn_oos   = float(np.mean(bm["saa_knn"]["oos"]))
knn_is    = float(np.mean(bm["saa_knn"]["is"]))
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

# From paper Figure 2, R=3, N=50  (read off chart)
SRO_FA_BEST_CV  = 21.3   # best rho selected by cross-validation
SRO_FA_BEST_RHO = 21.5   # best rho (oracle selection)

C_ROB    = "#2166ac"
C_ROB_IS = "#6baed6"
C_NOM    = "#555555"
C_SAA    = "#1a9641"
C_KNN    = "#e6a817"
C_SRO    = "#d62728"

# ── per-rho stats ────────────────────────────────────────────────────────────
rhos_pos = [r for r in d["rhos"] if r > 0]

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
best_oos     = float(rob_oos_mean.min()) if rob_oos_mean.size > 0 else float('nan')

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.suptitle("Three-stage inventory: R=3, N=50  (net16 v2, truncnorm)  —  N_test=10 000\n"
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
    else:
        r_c, m_c = rhos_arr[partial_mask], rob_oos_mean[partial_mask]
    ax.plot(r_c, m_c, "o--", color=C_ROB, lw=1.2, ms=3, alpha=0.45, mfc="none")

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
    else:
        r_c, m_c = rhos_arr[partial_mask], rob_is_mean[partial_mask]
    ax.plot(r_c, m_c, "^--", color=C_ROB_IS, lw=1.0, ms=3, alpha=0.45, mfc="none")

# Our benchmarks
ax.axhline(nom_oos,   color=C_NOM, ls=":",           lw=1.6,
           label=f"Neural nominal OOS  ({nom_oos:.2f})")
ax.axhline(nom_is,    color=C_NOM, ls=(0,(1,1)),     lw=1.2, alpha=0.6,
           label=f"Neural nominal IS  ({nom_is:.2f})")
ax.axhline(knn_oos,   color=C_KNN, ls="-.",           lw=1.7,
           label=f"{knn_label}  ({knn_oos:.2f})")
ax.axhline(saa_lb,    color=C_SAA, ls="--",           lw=1.7,
           label=f"SAA-LP clairvoyant LB  ({saa_lb:.2f})")

# Paper reference lines
ax.axhline(SRO_FA_BEST_CV,  color=C_SRO, ls=(0,(4,2)), lw=1.8,
           label=f"Paper: SRO-FA Best CV N=50  ({SRO_FA_BEST_CV:.2f})")
ax.axhline(SRO_FA_BEST_RHO, color=C_SRO, ls=(0,(2,2)), lw=1.3, alpha=0.6,
           label=f"Paper: SRO-FA Best \u03c1 N=50  ({SRO_FA_BEST_RHO:.2f})")

# Best point annotation
best_idx = int(np.argmin(rob_oos_mean))
best_rho = rhos_arr[best_idx]
ax.annotate(f"min OOS={best_oos:.3f}\nρ={best_rho:g}",
            xy=(best_rho, best_oos),
            xytext=(best_rho * 2.2, best_oos + 0.08),
            fontsize=8, color=C_ROB,
            arrowprops=dict(arrowstyle="->", color=C_ROB, lw=0.9))

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
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"\nBest OOS: rho={best_rho:g}  OOS={best_oos:.4f}")
print(f"vs SRO-FA Best CV: {SRO_FA_BEST_CV}  →  improvement = {SRO_FA_BEST_CV - best_oos:.3f}")
