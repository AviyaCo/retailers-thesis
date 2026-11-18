"""Best-radius-in-hindsight NN vs SRO-FA on the final shipping (R=3) graph data.

Same analysis as `statistical_tests_20261102/best_radius_hindsight.json` (no-shipping), applied to
`load_full_n_data('shipping_R3')`. Read-only: no training or evaluation.
Run from the experiment root: python -m comparison.paper_figures.best_radius_shipping
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from .full_n import HISTORY, RUN, load_full_n_data
from .paired_tests import sro_generator

OUT = Path(__file__).resolve().parent / "statistical_tests_20261102/best_radius_hindsight_shipping.json"
SEEDS = list(range(10))


def verify_same_training_data():
    seal = json.loads((RUN / "training_seal.json").read_text())
    checked = 0
    for c in seal["configurations"]:
        if c["kind"] != "shipping_R3" or c["radius"] != 0.:
            continue
        path = HISTORY / Path(c["summary_path"]).relative_to("/data/home/aviyac")
        if sro_generator(c["Ntotal"], c["seed"], 3) != json.loads(path.read_text())["data_sha256"]:
            raise RuntimeError(f"training data differ: shipping N{c['Ntotal']} seed {c['seed']}")
        checked += 1
    return checked


def per_seed(method):
    if method["seeds"] != SEEDS:
        raise RuntimeError("unexpected seeds")
    arr = np.array(method["oos"]["per_seed"], dtype=float)
    if arr.shape != (len(method["radius"]), len(SEEDS)):
        raise RuntimeError(f"unexpected per-seed shape {arr.shape}")
    if not np.allclose(arr.mean(axis=1), method["oos"]["mean"]):
        raise RuntimeError("per-seed rows do not reproduce plotted means")
    return arr


def paired(nn, sro):
    d = nn - sro
    w = stats.wilcoxon(nn, sro, alternative="two-sided", method="exact")
    return dict(nn=float(nn.mean()), sro=float(sro.mean()), diff=float(d.mean()),
                se=float(d.std(ddof=1) / np.sqrt(len(d))), pct=float(100 * d.mean() / sro.mean()),
                nn_better=int((d < 0).sum()), t_p=float(stats.ttest_rel(nn, sro).pvalue),
                wilcoxon_p=float(w.pvalue))


def main():
    checked = verify_same_training_data()
    data = load_full_n_data("shipping_R3")
    out = dict(source="load_full_n_data('shipping_R3')", training_data_hash_checks=checked, panels={})
    print(f"training data identical (hash) for {checked} shipping (N, seed) pairs")
    for panel in data["panels"]:
        n = panel["Ntotal"]
        nn, sro = per_seed(panel["green"]), per_seed(panel["sro_fa"])
        saa = np.array(panel["saa_knn"]["oos"]["per_seed"], dtype=float).ravel()
        i, j = nn.mean(axis=1).argmin(), sro.mean(axis=1).argmin()
        a = paired(nn[i], sro[j])
        a.update(nn_rho=panel["green"]["radius"][i], sro_rho=panel["sro_fa"]["radius"][j],
                 saa_knn=float(saa.mean()))
        b = paired(nn.min(axis=0), sro.min(axis=0))
        out["panels"][str(n)] = dict(best_by_mean=a, best_per_seed=b)
        print(f"N{n} A: NN {a['nn_rho']}: {a['nn']:.3f} | SRO-FA {a['sro_rho']}: {a['sro']:.3f} | "
              f"{a['diff']:+.3f} ({a['se']:.3f}) {a['pct']:+.1f}% | {a['nn_better']}/10 | "
              f"t {a['t_p']:.2g} | W {a['wilcoxon_p']:.3f} | SAA-kNN {a['saa_knn']:.3f}")
        print(f"N{n} B: NN {b['nn']:.3f} | SRO-FA {b['sro']:.3f} | {b['diff']:+.3f} ({b['se']:.3f}) "
              f"{b['pct']:+.1f}% | {b['nn_better']}/10 | t {b['t_p']:.2g} | W {b['wilcoxon_p']:.3f}")
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
