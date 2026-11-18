"""Paired statistical comparison of the final NN policy and SRO-FA, R=3 and R=10 (2026-11-02).

Test A (between seeds): for each data seed s = 0..9 both methods are trained on the SAME N training
samples (`np.random.seed(s)`, the same truncated-normal generator; verified by hash below) and evaluated on
the SAME 10,000 test scenarios (seed 9999). The unit of analysis is the per-seed out-of-sample mean
cost; the paired difference is d_s = NN_s - SRO-FA_s (negative = NN cheaper). Reported per comparison:
mean and SE of d, wins, a two-sided paired t-test and an exact two-sided Wilcoxon signed-rank test (n=10),
and a Holm correction within each family of comparisons.

Comparisons: (1) matched radius, for every radius shared by both grids; (2) each method at its own
best fixed radius on this test (hindsight; the same rule for both methods, so descriptive only). No
radius is selected on validation data in either saved campaign.

Sources (read-only): final full-N NN plot data (per-seed OOS means, SHA-256 pinned in render/full_n.py)
and the historical SRO-FA result JSONs (per-seed OOS means). No training or evaluation is performed.
The test set has been used before, so these are descriptive statistics on a known test set, not a fresh
holdout.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import truncnorm

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1] / "workspace"
FULL_N = WORKSPACE / "05_history/original_paths/retailers_moreau_training_selected/green_fullN_cv_20261011/runs/production_v1"
SRO = WORKSPACE / "02_results/SRO_FA/no_shipping"
SRO_FILES = {(3, n): SRO / f"R3/N{n}/sro_fa_results_N{n}_R3.json" for n in (50, 200, 800)}
SRO_FILES.update({(10, n): SRO / f"R10/sro_fa_results_N{n}_R10.json" for n in (50, 200, 800)})
NN_HASHES = {3: "e9829c1033314f68f4f5df9fc1c7d1a6b7ce12963404f75e7711369d5444df2c",
             10: "f3df3c08f2dab2b553b2da35887e7018828ed66f7e2cb5012c34f9a994075e07"}
SEEDS = list(range(10))


def sro_generator(n, seed, r):
    """The SRO-FA scripts' generator (identical calls to the NN campaign's `split_data`)."""
    np.random.seed(seed)
    xi1 = truncnorm.rvs((0. - 6.) / 2.5, np.inf, loc=6., scale=2.5, size=(n, r))
    mu2 = 2 * (6. - xi1) ** 2
    xi2 = truncnorm.rvs((0. - mu2) / 2.5, np.inf, loc=mu2, scale=2.5)
    h = hashlib.sha256()
    for value in (xi1, xi2):
        h.update(np.asarray(value, dtype="<f8").tobytes())
    return h.hexdigest()


def verify_same_training_data(r):
    """NN summaries record data_sha256 of the full generated sample; compare with the SRO-FA generator."""
    seal = json.loads((FULL_N / "training_seal.json").read_text())
    checked = 0
    for c in seal["configurations"]:
        if c["kind"] != f"R{r}" or c["radius"] != 0.:
            continue
        path = WORKSPACE / "05_history/original_paths" / Path(c["summary_path"]).relative_to("/data/home/aviyac")
        summary = json.loads(path.read_text())
        if sro_generator(c["Ntotal"], c["seed"], r) != summary["data_sha256"]:
            raise RuntimeError(f"training data differ: R{r} N{c['Ntotal']} seed {c['seed']}")
        checked += 1
    return checked


def load_nn(r):
    path = FULL_N / f"published/R{r}/plot_data.json"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NN_HASHES[r]:
        raise RuntimeError(f"NN plot data checksum mismatch: {path}")
    data = json.loads(raw)
    out = {}
    for panel in data["panels"]:
        g = panel["green"]
        if g["seeds"] != SEEDS:
            raise RuntimeError("unexpected NN seeds")
        out[panel["Ntotal"]] = {float(rad): np.array(row, dtype=float) for rad, row in zip(g["radius"], g["oos"]["per_seed"])}
    return out, data["test"]


def load_sro(r, n):
    data = json.loads(SRO_FILES[(r, n)].read_text())
    out = {}
    for rad, per_seed in data.items():
        if set(per_seed) != {str(s) for s in SEEDS}:
            raise RuntimeError(f"SRO-FA R{r} N{n} rho {rad}: incomplete seeds")
        out[float(rad)] = np.array([per_seed[str(s)]["oos"] for s in SEEDS], dtype=float)
    return out


def paired(nn, sro):
    d = nn - sro
    t = stats.ttest_rel(nn, sro)
    w = stats.wilcoxon(nn, sro, alternative="two-sided", method="exact") if np.any(d != 0) else None
    return dict(nn_mean=float(nn.mean()), sro_mean=float(sro.mean()), diff_mean=float(d.mean()),
                diff_se=float(d.std(ddof=1) / np.sqrt(len(d))), nn_better_seeds=int((d < 0).sum()),
                sro_better_seeds=int((d > 0).sum()), t_stat=float(t.statistic), t_p=float(t.pvalue),
                wilcoxon_stat=float(w.statistic) if w else None, wilcoxon_p=float(w.pvalue) if w else 1.,
                per_seed_diff=d.tolist())


def holm(rows, key):
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    m, running = len(rows), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (m - rank) * rows[i][key]))
        rows[i][key + "_holm"] = running


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    record = dict(role=__doc__, comparisons={})
    lines = ["# Paired statistical tests: final NN policy vs SRO-FA", "", __doc__.strip(), ""]
    for r in (3, 10):
        record[f"R{r}_training_data_verified_seeds"] = verify_same_training_data(r)
        nn_all, test = load_nn(r)
        record[f"R{r}_test"] = dict(n_test=test["n_test"], seed=test["seed"])
        for n in (50, 200, 800):
            nn, sro = nn_all[n], load_sro(r, n)
            common = sorted(set(nn) & set(sro))
            matched = []
            for rad in common:
                row = dict(radius=rad, **paired(nn[rad], sro[rad]))
                matched.append(row)
            holm(matched, "t_p"); holm(matched, "wilcoxon_p")
            best_nn = min(nn, key=lambda x: (nn[x].mean(), x))
            best_sro = min(sro, key=lambda x: (sro[x].mean(), x))
            best = dict(nn_radius=best_nn, sro_radius=best_sro, **paired(nn[best_nn], sro[best_sro]))
            record["comparisons"][f"R{r}_N{n}"] = dict(matched_radius=matched, best_vs_best=best,
                                                      nn_radii=sorted(nn), sro_radii=sorted(sro))
            lines += [f"## R={r}, N={n}", "",
                      f"Best fixed radius on this test (hindsight, same rule for both): NN rho={best_nn:g} "
                      f"{best['nn_mean']:.4f} vs SRO-FA rho={best_sro:g} {best['sro_mean']:.4f}; "
                      f"NN - SRO-FA = {best['diff_mean']:.4f} (SE {best['diff_se']:.4f}); NN better in "
                      f"{best['nn_better_seeds']}/10 seeds; paired t p={best['t_p']:.2g}; Wilcoxon exact p={best['wilcoxon_p']:.2g}.",
                      "", "| radius | NN mean | SRO-FA mean | NN - SRO-FA (SE) | NN better (seeds) | t-test p (Holm) | Wilcoxon p (Holm) |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for row in matched:
                lines.append(f"| {row['radius']:g} | {row['nn_mean']:.4f} | {row['sro_mean']:.4f} | "
                             f"{row['diff_mean']:.4f} ({row['diff_se']:.4f}) | {row['nn_better_seeds']}/10 | "
                             f"{row['t_p']:.2g} ({row['t_p_holm']:.2g}) | {row['wilcoxon_p']:.2g} ({row['wilcoxon_p_holm']:.2g}) |")
            lines.append("")
    lines += ["Note: with n=10 the smallest possible two-sided exact Wilcoxon p-value is 2/1024 = 0.00195.",
              "Holm correction is applied within each R/N table (matched radii)."]
    (output / "paired_tests.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "statistical_tests_20261102")
    main(parser.parse_args().output)
