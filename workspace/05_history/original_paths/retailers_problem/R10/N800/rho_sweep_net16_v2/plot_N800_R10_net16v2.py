"""
plot_N800_R10_net16v2.py -- IS+OOS sweep figure for R=10, N=800 (net16 v2).

Uses the cleaned plot style from the R=3 figures: neural OOS/IS plus SAA-kNN,
with SAA-LP hidden. If the optional R10 conditional lower-bound JSON exists,
it is shown as a reference line.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

BASE = "/data/home/aviyac/retailers_problem"
SWEEP_DIR = f"{BASE}/R10/N800/rho_sweep_net16_v2"
EVAL_JSON = os.path.join(SWEEP_DIR, "sweep_eval_net16_v2_N800_R10_10k.json")
COND_LB_JSONS = [
    (f"{BASE}/R10/saa_cond_lb_large.json", "saa_cond_lb_large"),
    (f"{BASE}/R10/saa_cond_lb_1m.json", "saa_cond_lb_1m"),
]
OUT = os.path.join(SWEEP_DIR, "figure_N800_R10_net16v2_10k.png")

C_ROB = "#2166ac"
C_ROB_IS = "#6baed6"
C_SAA = "#1a9641"
C_KNN = "#e6a817"


def load_cond_lb():
    for cond_lb_json, key in COND_LB_JSONS:
        if not os.path.exists(cond_lb_json):
            continue
        with open(cond_lb_json) as f:
            data = json.load(f)
        if key in data:
            return float(data[key])
    return None


if not os.path.exists(EVAL_JSON):
    print(f"[SKIP] {EVAL_JSON} not found.")
    raise SystemExit(0)

with open(EVAL_JSON) as f:
    d = json.load(f)

bm = d["benchmarks"]
saa_lb = float(np.mean(bm["saa_lb"]["oos"]))
knn_oos = float(np.mean(bm["saa_knn"]["oos"]))
cond_lb = load_cond_lb()

rhos_pos = [rho for rho in d["rhos"] if 0 < rho <= 1.5]

rob_oos_mean, rob_oos_lo, rob_oos_hi = [], [], []
rob_is_mean, n_seeds_list = [], []
for rho in rhos_pos:
    entry = d["sweep"][str(rho)]
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

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.suptitle(
    "Three-stage inventory: R=10, N=800  (net16 v2, truncnorm)  --  "
    "$N_{test}$=10 000\n"
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

ax.axhline(knn_oos, color=C_KNN, ls="-.", lw=1.7, label=f"SAA-kNN (k=5) OOS  ({knn_oos:.2f})")
if cond_lb is not None:
    ax.axhline(
        cond_lb,
        color=C_SAA,
        ls=(0, (7, 3)),
        lw=2.0,
        label=f"Cond. clairvoyant LB, seed=42 ({cond_lb:.3f})",
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
plt.subplots_adjust(bottom=0.35)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved {OUT}")

if rob_oos_mean.size:
    best_idx = int(np.argmin(rob_oos_mean))
    print(f"\nBest OOS: rho={rhos_arr[best_idx]:g}  OOS={rob_oos_mean[best_idx]:.4f}")
    print(f"kNN OOS:     {knn_oos:.4f}")
    if cond_lb is not None:
        print(f"Cond. LB:    {cond_lb:.4f}")
    print(f"SAA-LP LB hidden: {saa_lb:.4f}")
