"""
plot_combined_R3_net16v2.py
===========================
3-panel figure (N=50 | N=200 | N=800), R=3, no fixed cost (f=0).
Uses canonical net16-v2 strict-Gurobi evaluation JSONs (N_test = 10 000).

Each panel shows:
  • OOS cost  (NN-SRO)   — blue solid  + 20/80-pct shading
  • IS  cost  (NN-SRO)   — red  dashed
  • OOS cost  (SRO-FA)   — orange solid + 20/80-pct shading
  • SAA-LP clairvoyant LB — green flat line

Output: retailers_problem/R3/results/figure_combined_R3_net16v2.png

Use --green-comparison to render the completed, frozen September24 comparison
(green/SCS OOS and IS, SAA-kNN, SRO-FA) instead of this historical-only figure.
It never evaluates or selects a policy and refuses incomplete comparison runs.
"""

import sys

if __name__ == '__main__' and '--green-comparison' in sys.argv:
    import hashlib
    import json
    from pathlib import Path
    import importlib.util

    comparison_root = Path('/data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924')
    if sys.argv[1:] != ['--green-comparison']:
        raise ValueError('Use --green-comparison without additional arguments')
    launch = json.loads((comparison_root / 'launch.json').read_text())
    result_dir = Path(launch['output'])
    summary = json.loads((result_dir / 'summary.json').read_text())
    if summary['status'] != 'completed' or not summary['source_hash_final']:
        raise ValueError('Comparison incomplete; no partial figure may be rendered')
    manifest_path = Path(summary['manifest'])
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != summary['manifest_sha256']:
        raise ValueError('Comparison manifest changed')
    manifest = json.loads(manifest_path.read_text())
    renderer = comparison_root / 'plot_comparison.py'
    expected = manifest['files'][str(renderer)]['sha256']
    if hashlib.sha256(renderer.read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen renderer changed')
    loader = importlib.util.spec_from_file_location('fixed_R3_plot', renderer)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    data = json.loads(Path(summary['plot_data']).read_text())
    module.render(data, result_dir)
    print(f'Saved comparison: {result_dir / "comparison_R3.png"}')
    raise SystemExit(0)

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = "/data/home/aviyac/retailers_problem"

# ── data paths ────────────────────────────────────────────────────────────────
CONFIGS = {
    "N=50":  {
        "eval":   f"{BASE}/R3/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json",
        "sro_fa": f"{BASE}/R3/N50/sro_fa/sro_fa_results_N50_R3.json",
    },
    "N=200": {
        "eval":   f"{BASE}/R3/N200/rho_sweep_net16_v2/sweep_eval_net16_v2_N200_10k.json",
        "sro_fa": f"{BASE}/R3/N200/sro_fa/sro_fa_results_N200_R3.json",
    },
    "N=800": {
        "eval":   f"{BASE}/R3/N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_10k.json",
        "sro_fa": f"{BASE}/R3/N800/sro_fa/sro_fa_results_N800_R3.json",
    },
}

OUT = f"{BASE}/R3/results/figure_combined_R3_net16v2.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ── colours ───────────────────────────────────────────────────────────────────
C_OOS   = "#2166ac"   # blue  — NN-SRO OOS
C_IS    = "#d6604d"   # red   — NN-SRO IS
C_SRO   = "#f97b0a"   # orange — SRO-FA OOS
C_SAA   = "#1a9641"   # green — SAA-LP LB
C_COND  = "#1b7837"   # dark green — conditional clairvoyant LB (N=5000, seed=42)
PROJECTION_METHOD = "gurobi_qp"
EVALUATOR_VERSION = "gurobi_qp_final_mp_is_v2"


# ── helpers ───────────────────────────────────────────────────────────────────
def load_eval(path):
    with open(path) as stream:
        data = json.load(stream)
    metadata = data.get("meta", {})
    if (
        data.get("runtime", {}).get("status") != "completed"
        or metadata.get("projection_method") != PROJECTION_METHOD
        or metadata.get("N_test") != 10_000
        or metadata.get("seed_test") != 9999
    ):
        raise ValueError(f"Noncanonical or incomplete evaluator artifact: {path}")
    for rho_key, entry in data.get("sweep", {}).items():
        oos = np.asarray(entry.get("rob_oos", []), dtype=float)
        robust_is = np.asarray(entry.get("rob_is", []), dtype=float)
        if (
            entry.get("projection_method") != PROJECTION_METHOD
            or entry.get("evaluator_version") != EVALUATOR_VERSION
            or entry.get("n_seeds") != 10
            or oos.size != 10
            or robust_is.size != 10
            or not np.isfinite(oos).all()
            or not np.isfinite(robust_is).all()
        ):
            raise ValueError(f"Incomplete or stale rho={rho_key} in {path}")
    saa_lb = data.get("benchmarks", {}).get("saa_lb", {})
    if len(saa_lb.get("oos", [])) != 10 or not np.isfinite(saa_lb["oos"]).all():
        raise ValueError(f"Incomplete or stale SAA-LP benchmark block: {path}")
    return data


def nn_sro_curves(d):
    """Return (rhos, oos_mean, oos_lo, oos_hi, is_mean) for all rho > 0."""
    rhos, oos_m, oos_lo, oos_hi, is_m = [], [], [], [], []
    for rho_key in sorted(d["sweep"], key=float):
        rv = float(rho_key)
        if rv <= 0:
            continue
        e = d["sweep"][rho_key]
        oos = np.array(e["rob_oos"], dtype=float)
        ris = np.array(e["rob_is"],  dtype=float)
        if len(oos) == 0:
            continue
        rhos.append(rv)
        oos_m.append(oos.mean())
        oos_lo.append(np.percentile(oos, 20) if len(oos) >= 2 else oos.mean())
        oos_hi.append(np.percentile(oos, 80) if len(oos) >= 2 else oos.mean())
        is_m.append(ris.mean() if len(ris) else float("nan"))
    return (np.array(rhos), np.array(oos_m), np.array(oos_lo),
            np.array(oos_hi), np.array(is_m))


def sro_fa_curves(sro_data):
    """Return (rhos, oos_mean, oos_lo, oos_hi) for all rho > 0."""
    rhos, oos_m, oos_lo, oos_hi = [], [], [], []
    for rho_key in sorted(sro_data, key=float):
        rv = float(rho_key)
        if rv <= 0:
            continue
        seeds = sro_data[rho_key]
        oos_l = np.array([seeds[s]["oos"] for s in seeds], dtype=float)
        if len(oos_l) == 0:
            continue
        rhos.append(rv)
        oos_m.append(oos_l.mean())
        oos_lo.append(np.percentile(oos_l, 20) if len(oos_l) >= 2 else oos_l.mean())
        oos_hi.append(np.percentile(oos_l, 80) if len(oos_l) >= 2 else oos_l.mean())
    return (np.array(rhos), np.array(oos_m),
            np.array(oos_lo), np.array(oos_hi))


def format_ax(ax, all_rhos, ylabel=None, xlabel=True):
    ax.set_xscale("log")
    for r in all_rhos:
        ax.axvline(r, color="#e0e0e0", lw=0.5, zorder=0)
    # readable subset of x-tick labels
    labeled, prev = [], None
    for r in sorted(all_rhos):
        if prev is None or r / prev >= 1.8:
            labeled.append(r)
            prev = r
    ax.set_xticks(labeled)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.tick_params(axis="x", labelsize=7.5, rotation=45)
    ax.grid(True, axis="y", alpha=0.40, lw=0.7, color="#bbbbbb")
    if xlabel:
        ax.set_xlabel(r"$\varepsilon_N$", fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)


# ── cross-product LB ──────────────────────────────────────────────────────────

# Conditional clairvoyant LB  (retailers_problem/R3/saa_cond_lb_large.py)
#   SAA LP over N=5000 xi1 samples x M=200 xi2|xi1 samples -> 1 million (xi1,xi2) scenarios.
#   Sampling seed = 42.  Value = 19.3948.
#   Neural OOS costs on the same 1M data (training seeds 0-9):
#     best = 19.563 (seed 1), all seeds >= 19.563 > LB -> LB validity confirmed.
_cond_large_json = os.path.join(BASE, "R3", "saa_cond_lb_large.json")
cond_lb_large_val = float(json.load(open(_cond_large_json))["saa_cond_lb_large"]) if os.path.exists(_cond_large_json) else None

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle(
    "Impact of robustness parameter on SRO for R = 3,  no fixed cost  ($f = 0$)\n"
    "dist: truncated normal  |  NN width = 16  |  Cond. LB: seed=42, $N_{\\xi_1}$=5000, M=200",
    fontsize=12, fontweight="bold",
)

handles_for_legend = None   # collect from first panel

for ax, (label, cfg) in zip(axes, CONFIGS.items()):
    d_eval = load_eval(cfg["eval"])
    d_sro  = json.load(open(cfg["sro_fa"])) if os.path.exists(cfg["sro_fa"]) else {}

    # NN-SRO curves
    rhos_nn, oos_m, oos_lo, oos_hi, is_m = nn_sro_curves(d_eval)

    h_oos, = ax.plot(rhos_nn, oos_m, "-", color=C_OOS, lw=2.2, zorder=4,
                     label="OOS cost (NN-SRO, width=16)")
    ax.fill_between(rhos_nn, oos_lo, oos_hi, alpha=0.18, color=C_OOS)

    h_is, = ax.plot(rhos_nn, is_m, "--", color=C_IS, lw=1.8, zorder=3,
                    label="IS cost (NN-SRO, width=16)")

    # SRO-FA curve
    h_sro = None
    if d_sro:
        rhos_sro, sro_m, sro_lo, sro_hi = sro_fa_curves(d_sro)
        h_sro, = ax.plot(rhos_sro, sro_m, "-", color=C_SRO, lw=2.2,
                         zorder=4, label="OOS cost (SRO-FA)")
        ax.fill_between(rhos_sro, sro_lo, sro_hi, alpha=0.18, color=C_SRO)

    # SAA-LP benchmark
    saa_oos = float(np.mean(d_eval["benchmarks"]["saa_lb"]["oos"]))
    h_saa = ax.axhline(saa_oos, color=C_SAA, lw=2.0, zorder=5,
                       label=f"SAA-LP benchmark  ({saa_oos:.3f})")
    h_cond = None
    if cond_lb_large_val is not None:
        h_cond = ax.axhline(cond_lb_large_val, color=C_COND, ls="-.", lw=2.2, zorder=5,
                            label=f"Cond. clairvoyant LB, seed=42 ({cond_lb_large_val:.3f})")

    # axes formatting
    all_rhos = np.unique(np.concatenate([rhos_nn,
                                         rhos_sro if d_sro else np.array([])]))
    format_ax(ax, all_rhos[all_rhos > 0],
              ylabel="Cost" if ax is axes[0] else None)
    ax.set_title(label, fontsize=12, fontweight="bold")

    if handles_for_legend is None:
        handles_for_legend = [h for h in [h_oos, h_is, h_sro, h_saa, h_cond] if h is not None]

# shared legend below the panels
fig.legend(
    handles=handles_for_legend,
    loc="lower center",
    ncol=4,
    fontsize=10,
    bbox_to_anchor=(0.5, -0.06),
    frameon=True, framealpha=0.93,
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
