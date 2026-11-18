"""
plot_combined_N50_R3.py  —  Combined OOS sweep: all models, R=3 N=50
=====================================================================
Shows on one panel:
    • Neural robust (strict Gurobi projection)         — blue
  • SRO-FA                                          — red dashed
    • SAA kNN (selected k)                            — gold dash-dot
  • SAA-LP clairvoyant LB                           — green dashed
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE     = "/data/home/aviyac/retailers_problem"
EVAL_JSON   = f"{BASE}/R3/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json"
SRO_JSON    = f"{BASE}/R3/N50/sro_fa/sro_fa_results_N50_R3.json"
OUT         = f"{BASE}/R3/N50/figure_combined_N50_R3.png"
# Conditional clairvoyant LB  (retailers_problem/R3/saa_cond_lb_large.py)
#   SAA LP over N=5000 xi1 samples x M=200 xi2|xi1 samples -> 1 million (xi1,xi2) scenarios.
#   Sampling seed = 42.  Value = 19.3948.
#   Neural OOS costs on the same 1M data (training seeds 0-9):
#     best = 19.563 (seed 1), all seeds >= 19.563 > LB -> LB validity confirmed.
COND_LARGE_JSON = f"{BASE}/R3/saa_cond_lb_large.json"

# ── colours ───────────────────────────────────────────────────────────────────
C_ROB    = "#2166ac"   # blue  — strict-Gurobi neural model
C_SRO    = "#d62728"   # red   — SRO-FA
C_SAA    = "#1a9641"   # green — SAA LB
C_KNN    = "#e6a817"   # gold  — kNN
C_COND   = "#1b7837"   # dark green — conditional clairvoyant LB (N=5000, seed=42)

def load_sweep(path):
    with open(path) as f:
        return json.load(f)

def extract_curve(d, rho_range=(0, 1.5)):
    rhos, mean_, lo_, hi_, ns_ = [], [], [], [], []
    for rho_key in sorted(d["sweep"], key=float):
        rv = float(rho_key)
        if rv <= 0 or rv > rho_range[1]:
            continue
        e   = d["sweep"][rho_key]
        oos = np.array(e["rob_oos"], dtype=float)
        if len(oos) == 0:
            continue
        rhos.append(rv)
        mean_.append(oos.mean())
        lo_.append(np.percentile(oos, 20) if len(oos) >= 2 else oos.mean())
        hi_.append(np.percentile(oos, 80) if len(oos) >= 2 else oos.mean())
        ns_.append(e["n_seeds"])
    return (np.array(rhos), np.array(mean_), np.array(lo_),
            np.array(hi_), np.array(ns_))

# ── load ──────────────────────────────────────────────────────────────────────
evaluation_ok = os.path.exists(EVAL_JSON)
sro_ok    = os.path.exists(SRO_JSON)

if evaluation_ok:
    dc = load_sweep(EVAL_JSON)
    rc, mc, lc, hc, nc = extract_curve(dc)
    bm         = dc["benchmarks"]
    saa_lb_oos = float(np.mean(bm["saa_lb"]["oos"]))    if "saa_lb"  in bm else None
    knn_oos    = float(np.mean(bm["saa_knn"]["oos"]))   if "saa_knn" in bm else None
    knn_selected_k = bm.get("saa_knn", {}).get("selected_k")
    knn_selected_by_seed = bm.get("saa_knn", {}).get("selected_k_by_seed")
    full_c     = nc == 10
    best_neural = float(mc[full_c].min()) if full_c.any() else float(mc.min())
else:
    rc = mc = lc = hc = nc = full_c = np.array([])
    saa_lb_oos = knn_oos = knn_selected_k = knn_selected_by_seed = best_neural = None
    print("WARNING: canonical Gurobi eval JSON not found")

# SRO-FA
sro_rhos, sro_mean, sro_lo, sro_hi = [], [], [], []
if sro_ok:
    with open(SRO_JSON) as f:
        sro_data = json.load(f)
    for rho_key in sorted(sro_data, key=float):
        rv = float(rho_key)
        if rv <= 0 or rv > 1.5:
            continue
        seeds = sro_data[rho_key]
        oos   = np.array([seeds[s]["oos"] for s in seeds])
        sro_rhos.append(rv); sro_mean.append(oos.mean())
        sro_lo.append(np.percentile(oos, 20)); sro_hi.append(np.percentile(oos, 80))
    sro_rhos = np.array(sro_rhos); sro_mean = np.array(sro_mean)
    sro_lo = np.array(sro_lo);     sro_hi = np.array(sro_hi)
    best_sro = float(sro_mean.min()) if len(sro_mean) else None
else:
    best_sro = None

# ── cross-product LB ──────────────────────────────────────────────────────────

cond_lb_large_val = float(json.load(open(COND_LARGE_JSON))["saa_cond_lb_large"]) if os.path.exists(COND_LARGE_JSON) else None
# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 5.5))
fig.suptitle(
    "Three-stage inventory: R=3, N=50  —  Out-of-sample cost vs training radius $\\rho$\n"
    r"$N_{\rm test}=10\,000$  |  Cond. LB: seed=42, $N_{\xi_1}$=5000, M=200",
    fontsize=11, fontweight="bold")

def plot_curve(ax, rhos, mean_, lo_, hi_, full_mask, color, marker,
               ls_full="-", ls_part="--", label=None, zorder=4):
    if full_mask.any():
        ax.plot(rhos[full_mask], mean_[full_mask],
                f"{marker}{ls_full}", color=color, lw=2, ms=4,
                zorder=zorder, label=label)
        ax.fill_between(rhos[full_mask], lo_[full_mask], hi_[full_mask],
                        alpha=0.16, color=color)
    part = ~full_mask
    if part.any():
        # connect last full point to first partial
        if full_mask.any():
            last = np.where(full_mask)[0][-1]
            r_c = np.concatenate([[rhos[last]], rhos[part]])
            m_c = np.concatenate([[mean_[last]], mean_[part]])
        else:
            r_c, m_c = rhos[part], mean_[part]
        lbl = label if not full_mask.any() else None
        ax.plot(r_c, m_c, f"{marker}{ls_part}", color=color,
                lw=1.2, ms=3, alpha=0.5, mfc="none", label=lbl, zorder=zorder)

# Strict-Gurobi neural model
if evaluation_ok and len(rc) > 0:
    plot_curve(ax, rc, mc, lc, hc, full_c, C_ROB, "o",
               label=f"Neural robust OOS (best={best_neural:.2f})")

# SRO-FA
if sro_ok and len(sro_rhos) > 0:
    ax.plot(sro_rhos, sro_mean, "s--", color=C_SRO, lw=1.7, ms=3.5, zorder=3,
            label=f"SRO-FA (best={best_sro:.2f})" if best_sro else "SRO-FA")
    ax.fill_between(sro_rhos, sro_lo, sro_hi, alpha=0.14, color=C_SRO)

# benchmarks
if knn_oos is not None:
    if knn_selected_by_seed and len(set(knn_selected_by_seed)) > 1:
        knn_label = "SAA-kNN (seed-wise k) OOS"
    elif knn_selected_by_seed:
        knn_label = f"SAA-kNN (k={knn_selected_by_seed[0]}) OOS"
    elif knn_selected_k:
        knn_label = f"SAA-kNN (best k={knn_selected_k}) OOS"
    else:
        knn_label = "SAA-kNN (k=5) OOS"
    ax.axhline(knn_oos, color=C_KNN, ls="-.", lw=1.7,
               label=f"{knn_label} ({knn_oos:.2f})")
if saa_lb_oos is not None:
    ax.axhline(saa_lb_oos, color=C_SAA, ls="--", lw=1.7,
               label=f"SAA-LP clairvoyant LB ({saa_lb_oos:.3f})")
if cond_lb_large_val is not None:
    ax.axhline(cond_lb_large_val, color=C_COND, ls="-.", lw=2.2,
               label=f"Cond. clairvoyant LB, seed=42 ({cond_lb_large_val:.3f})")

# ── axis formatting ───────────────────────────────────────────────────────────
all_rhos = np.concatenate([r for r in [rc, sro_rhos] if len(r) > 0])
if len(all_rhos) > 0:
    ax.set_xscale("log")
    for r in np.unique(all_rhos):
        ax.axvline(r, color="#ebebeb", lw=0.5, zorder=0)
    labeled, prev = [], None
    for r in sorted(np.unique(all_rhos)):
        if prev is None or r / prev > 1.25:
            labeled.append(r); prev = r
    ax.set_xticks(labeled)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="x", which="major", labelsize=8, rotation=45)

ax.set_xlabel("Training radius  $\\rho$", fontsize=11)
ax.set_ylabel("Mean OOS cost", fontsize=11)
ax.grid(True, which="major", axis="y", alpha=0.4, lw=0.8, color="#bbbbbb")
ax.legend(fontsize=8.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.32), ncol=2,
          frameon=True, framealpha=0.93)

plt.tight_layout()
plt.subplots_adjust(bottom=0.36)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved  {OUT}")

# ── console summary ───────────────────────────────────────────────────────────
if best_neural is not None: print(f"Best OOS  neural:     {best_neural:.4f}")
if best_sro    is not None: print(f"Best OOS  SRO-FA:     {best_sro:.4f}")
if saa_lb_oos  is not None: print(f"SAA LB:               {saa_lb_oos:.4f}")
if cond_lb_large_val is not None: print(f"Cond. LB (large):     {cond_lb_large_val:.4f}")
