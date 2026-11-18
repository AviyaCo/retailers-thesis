"""
plot_combined_N800_R3.py  —  IS+OOS sweep: all models, R=3 N=800
================================================================
Same style as plot_N50_net16v2.py:
  • Neural robust OOS (mean ± 20/80 pct)  — blue solid
  • Neural robust IS                       — light blue
  • SRO-FA OOS (mean ± 20/80 pct)         — red dashed
    • SAA-kNN (selected k)                   — gold dash-dot
  • SAA-LP clairvoyant LB                  — green dashed
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE        = "/data/home/aviyac/retailers_problem"
EVAL_JSON   = f"{BASE}/R3/N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_10k.json"
SRO_JSON    = f"{BASE}/R3/N800/sro_fa/sro_fa_results_N800_R3.json"
OUT         = f"{BASE}/R3/N800/figure_combined_N800_R3.png"
# Conditional clairvoyant LB  (retailers_problem/R3/saa_cond_lb_large.py)
#   SAA LP over N=5000 xi1 samples x M=200 xi2|xi1 samples -> 1 million (xi1,xi2) scenarios.
#   Sampling seed = 42.  Value = 19.3948.
#   Neural OOS costs on the same 1M data (training seeds 0-9):
#     best = 19.563 (seed 1), all seeds >= 19.563 > LB -> LB validity confirmed.
COND_LARGE_JSON = f"{BASE}/R3/saa_cond_lb_large.json"

os.makedirs(os.path.dirname(OUT), exist_ok=True)

C_ROB    = "#2166ac"
C_ROB_IS = "#6baed6"
C_SRO    = "#d62728"
C_SAA    = "#1a9641"
C_KNN    = "#e6a817"
C_COND   = "#1b7837"   # dark green — conditional clairvoyant LB (N=5000, seed=42)

# ── load canonical strict-Gurobi evaluation ──────────────────────────────────
with open(EVAL_JSON) as f:
    d = json.load(f)

bm        = d["benchmarks"]
saa_lb    = float(np.mean(bm["saa_lb"]["oos"]))
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
cond_lb_large_val = float(json.load(open(COND_LARGE_JSON))["saa_cond_lb_large"]) if os.path.exists(COND_LARGE_JSON) else None

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
best_oos     = float(rob_oos_mean.min())

# ── load SRO-FA ───────────────────────────────────────────────────────────────
sro_rhos, sro_oos_mean, sro_oos_lo, sro_oos_hi = [], [], [], []
if os.path.exists(SRO_JSON):
    with open(SRO_JSON) as f:
        sro_data = json.load(f)
    for rho_key in sorted(sro_data, key=float):
        rv = float(rho_key)
        if rv <= 0 or rv > 1.5:
            continue
        seeds = sro_data[rho_key]
        oos_l = np.array([seeds[s]["oos"] for s in seeds
                          if isinstance(seeds[s], dict) and "oos" in seeds[s]])
        sro_rhos.append(rv); sro_oos_mean.append(oos_l.mean())
        sro_oos_lo.append(np.percentile(oos_l, 20))
        sro_oos_hi.append(np.percentile(oos_l, 80))
    sro_rhos    = np.array(sro_rhos); sro_oos_mean = np.array(sro_oos_mean)
    sro_oos_lo  = np.array(sro_oos_lo); sro_oos_hi = np.array(sro_oos_hi)
best_sro = float(sro_oos_mean.min()) if len(sro_oos_mean) else None

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.suptitle(
    "Three-stage inventory: R=3, N=800  (net16 v2, truncnorm)  —  $N_{test}$=10 000\n"
    "In-sample vs out-of-sample cost vs training radius  |  Cond. LB: seed=42, $N_{\\xi_1}$=5000, M=200",
    fontsize=11, fontweight="bold")

def _plot_with_partial(ax, rhos, mean_, lo_, hi_, full_mask,
                       color, marker, label, zorder=4, fill_alpha=0.18):
    if full_mask.any():
        ax.plot(rhos[full_mask], mean_[full_mask],
                f"{marker}-", color=color, lw=2, ms=3.5, zorder=zorder, label=label)
        ax.fill_between(rhos[full_mask], lo_[full_mask], hi_[full_mask],
                        alpha=fill_alpha, color=color)
    part = ~full_mask
    if part.any():
        if full_mask.any():
            last = np.where(full_mask)[0][-1]
            r_c = np.concatenate([[rhos[last]], rhos[part]])
            m_c = np.concatenate([[mean_[last]], mean_[part]])
            l_c = np.concatenate([[lo_[last]],   lo_[part]])
            h_c = np.concatenate([[hi_[last]],   hi_[part]])
        else:
            r_c, m_c, l_c, h_c = rhos[part], mean_[part], lo_[part], hi_[part]
        lbl = label if not full_mask.any() else None
        ax.plot(r_c, m_c, f"{marker}--", color=color,
                lw=1.5, ms=3.5, alpha=0.7, mfc="none", label=lbl, zorder=zorder)
        ax.fill_between(r_c, l_c, h_c, alpha=fill_alpha * 0.6, color=color)

# Neural robust OOS
_plot_with_partial(ax, rhos_arr, rob_oos_mean, rob_oos_lo, rob_oos_hi, full_mask,
                   C_ROB, "o",
                   label=f"Neural robust OOS (mean \u00b1 20/80 pct, best={best_oos:.2f})")

# Neural robust IS
_is_lo = rob_is_mean; _is_hi = rob_is_mean  # no band for IS
if full_mask.any():
    ax.plot(rhos_arr[full_mask], rob_is_mean[full_mask],
            "^-", color=C_ROB_IS, lw=1.5, ms=3, zorder=4, label="Neural robust IS")
if partial_mask.any():
    if full_mask.any():
        last = np.where(full_mask)[0][-1]
        r_c = np.concatenate([[rhos_arr[last]], rhos_arr[partial_mask]])
        m_c = np.concatenate([[rob_is_mean[last]], rob_is_mean[partial_mask]])
    else:
        r_c, m_c = rhos_arr[partial_mask], rob_is_mean[partial_mask]
    lbl = "Neural robust IS" if not full_mask.any() else None
    ax.plot(r_c, m_c, "^--", color=C_ROB_IS, lw=1.0, ms=3,
            alpha=0.45, mfc="none", label=lbl)

# SRO-FA
if len(sro_rhos) > 0:
    pm = (sro_rhos > 0) & (sro_rhos <= 1.5)
    lbl = f"SRO-FA OOS (mean \u00b1 20/80 pct, best={best_sro:.2f})" if best_sro else "SRO-FA OOS"
    ax.plot(sro_rhos[pm], sro_oos_mean[pm],
            "s--", color=C_SRO, lw=1.7, ms=3.5, zorder=3, label=lbl)
    ax.fill_between(sro_rhos[pm], sro_oos_lo[pm], sro_oos_hi[pm],
                    alpha=0.18, color=C_SRO)

# Benchmarks
ax.axhline(knn_oos, color=C_KNN, ls="-.", lw=1.7,
           label=f"{knn_label}  ({knn_oos:.2f})")
ax.axhline(saa_lb,  color=C_SAA, ls="--", lw=1.7,
           label=f"SAA-LP clairvoyant LB  ({saa_lb:.3f})")
if cond_lb_large_val is not None:
    ax.axhline(cond_lb_large_val, color=C_COND, ls="-.", lw=2.2,
               label=f"Cond. clairvoyant LB, seed=42 ({cond_lb_large_val:.3f})")

# ── axis ──────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
for r in rhos_pos:
    ax.axvline(r, color="#e8e8e8", lw=0.5, zorder=0)
labeled, prev = [], None
for r in sorted(rhos_pos):
    if prev is None or r / prev > 1.25:
        labeled.append(r); prev = r
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
print(f"Saved  {OUT}")

print(f"\nBest OOS  neural: {best_oos:.4f}")
if best_sro:    print(f"Best OOS  SRO-FA: {best_sro:.4f}")
print(f"SAA-kNN:          {knn_oos:.4f}")
print(f"SAA LB:           {saa_lb:.4f}")
if cond_lb_large_val is not None: print(f"Cond. LB (large): {cond_lb_large_val:.4f}")
