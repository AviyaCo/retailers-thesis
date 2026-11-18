"""
plot_sweep_shipping.py  --  Plot IS/OOS cost vs rho for shipping problem
=========================================================================
Reads sweep_eval_shipping.json and produces a plot with:
  - OOS cost (mean ± std over seeds)
  - IS cost  (mean ± std over seeds)
  - SAA MILP lower bound (flat, dashed)
  - kNN-SAA OOS (flat, dashed)
  - Nominal (rho→0) marked

Usage:
    python3 plot_sweep_shipping.py [--no-show]
"""

import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--no-show", action="store_true")
parser.add_argument("--input",   default=os.path.join(os.path.dirname(__file__),
                                                       "sweep_eval_shipping.json"))
parser.add_argument("--output",  default=os.path.join(os.path.dirname(__file__),
                                                       "plot_sweep_shipping.png"))
args = parser.parse_args()

with open(args.input) as f:
    data = json.load(f)

sweep      = data["sweep"]
benchmarks = data["benchmarks"]

# ── collect rho-indexed arrays ────────────────────────────────────────────────
rhos, oos_mean, oos_std, is_mean, is_std, n_seeds = [], [], [], [], [], []

for rho in sorted(float(k) for k in sweep):
    v = sweep[str(rho)]
    rob_oos = [x for x in v["rob_oos"] if x is not None]
    rob_is  = [x for x in v["rob_is"]  if x is not None]
    if not rob_oos:
        continue
    rhos.append(rho)
    oos_mean.append(np.mean(rob_oos))
    oos_std .append(np.std(rob_oos))
    is_mean .append(np.mean(rob_is))
    is_std  .append(np.std(rob_is))
    n_seeds .append(v["n_seeds"])

rhos     = np.array(rhos)
oos_mean = np.array(oos_mean)
oos_std  = np.array(oos_std)
is_mean  = np.array(is_mean)
is_std   = np.array(is_std)

# ── benchmarks ────────────────────────────────────────────────────────────────
saa_lb_oos  = [x for x in benchmarks["saa_lb"]["oos"]  if x is not None]
saa_knn_oos = [x for x in benchmarks["saa_knn"]["oos"] if x is not None]
lb_val      = np.mean(saa_lb_oos)  if saa_lb_oos  else None
knn_val     = np.mean(saa_knn_oos) if saa_knn_oos else None

# nominal OOS from rho~0 or from sweep at smallest rho
if len(rhos):
    nom_vals = [x for x in sweep[str(rhos[0])]["nom_oos"] if x is not None]
    nom_val  = np.mean(nom_vals) if nom_vals else None
else:
    nom_val = None

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

C_OOS = "#2166ac"
C_IS  = "#d6604d"
C_LB  = "#4dac26"
C_KNN = "#7b2d8b"
C_NOM = "#888888"

ax.fill_between(rhos, oos_mean - oos_std, oos_mean + oos_std,
                alpha=0.18, color=C_OOS)
ax.plot(rhos, oos_mean, "-o", color=C_OOS, linewidth=2,
        markersize=5, label="Robust OOS (mean±std)")

ax.fill_between(rhos, is_mean - is_std, is_mean + is_std,
                alpha=0.18, color=C_IS)
ax.plot(rhos, is_mean, "--s", color=C_IS, linewidth=1.5,
        markersize=4, label="Robust IS (mean±std)")

if lb_val is not None:
    ax.axhline(lb_val,  color=C_LB,  linestyle=":",  linewidth=1.8,
               label=f"SAA MILP LB (IS mean)  ({lb_val:.3f})")
if knn_val is not None:
    ax.axhline(knn_val, color=C_KNN, linestyle="-.", linewidth=1.8,
               label=f"kNN-SAA OOS  ({knn_val:.3f})")
if nom_val is not None:
    ax.axhline(nom_val, color=C_NOM, linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"Nominal OOS  ({nom_val:.3f})")

ax.set_xscale("log")
ax.set_xlabel("ρ (robustness radius)", fontsize=12)
ax.set_ylabel("Mean cost", fontsize=12)
ax.set_title("Shipping problem — OOS cost vs ρ  (N=200 train, N=2000 test)",
             fontsize=12)
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, which="both", alpha=0.3)

# annotate seed count per rho
for x, y, n in zip(rhos, oos_mean, n_seeds):
    ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=6, color=C_OOS)

plt.tight_layout()
plt.savefig(args.output, dpi=150)
print(f"Saved: {args.output}")
if not args.no_show:
    plt.show()
