"""Paired NN vs SRO-FA tests at validation-selected radii (Protocol A), 3 November 2026.

Read-only: uses saved per-seed test means; no training, selection or evaluation.
Selections (2026-11-18): radii chosen by the shared rule "largest radius within 1e-6 of the minimum validation
score", read from radius_selection_20261118/selections_new_rule.json (which reads the stored validation scores and
stored test evaluations of the original runs; the previous rule was "exact tie -> smaller radius").
Both methods train on the same 80% split rows (40/160/640) and select the radius
per seed on the same 20% validation rows (10/40/160); test = seed 9999, 10,000 rows.
Run: ../ap_env/bin/python comparison/paper_figures/statistical_tests_validation_20261103/paired_tests_validation.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
SOURCES = dict(selections=HERE.parent / 'radius_selection_20261118/selections_new_rule.json')
CASES = (('R3', 'R=3'), ('R10', 'R=10'), ('shipping_R3', 'R=3 shipping'))
SIZES = (50, 200, 800)
SEEDS = range(10)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    nn, fa = {}, {}
    case = {'R3': 'R3', 'R10': 'R10', 'R3_shipping': 'shipping_R3'}
    for row in json.loads(SOURCES['selections'].read_text())['selections']:
        target = {'neural_ccg': nn, 'sro_fa': fa}.get(row['source'])
        if target is not None:
            target[(case[row['setting']], row['N'], row['seed'])] = (row['new_radius'], row['new_test_mean'])
    expected = {(c, n, s) for c, _ in CASES for n in SIZES for s in SEEDS}
    if set(nn) != expected or set(fa) != expected:
        raise ValueError('Incomplete or unexpected per-seed coverage')
    return nn, fa


def main():
    nn, fa = load()
    rows = []
    for case, label in CASES:
        for n in SIZES:
            a = np.array([nn[(case, n, s)][1] for s in SEEDS])
            b = np.array([fa[(case, n, s)][1] for s in SEEDS])
            d = a - b
            rows.append(dict(
                case=case, label=label, Ntotal=n, Ntrain=4 * n // 5, Nval=n // 5,
                nn_mean=a.mean(), sro_fa_mean=b.mean(), diff_mean=d.mean(),
                diff_se=d.std(ddof=1) / np.sqrt(len(d)), diff_pct=100 * d.mean() / b.mean(),
                nn_cheaper=int((d < 0).sum()), seeds=len(d),
                t_p=stats.ttest_rel(a, b).pvalue,
                wilcoxon_exact_p=stats.wilcoxon(a, b, method='exact').pvalue,
                nn_selected_radius=[nn[(case, n, s)][0] for s in SEEDS],
                sro_fa_selected_radius=[fa[(case, n, s)][0] for s in SEEDS],
                nn_per_seed=a.tolist(), sro_fa_per_seed=b.tolist()))
    rows = [{k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()} for r in rows]
    (HERE / 'paired_tests_validation.json').write_text(json.dumps(dict(
        protocol='Protocol A: shared 80/20 split, per-seed validation-selected radius for both methods',
        radius_selection_rule=json.loads(SOURCES['selections'].read_text())['rule'],
        difference='NN - SRO-FA (negative = NN cheaper)',
        tests='two-sided paired t-test; exact two-sided Wilcoxon signed-rank; n=10 seeds; '
              'one test per case/N, no multiplicity correction',
        caveats='Validation rows also selected the NN learning rate; test set (seed 9999) '
                'previously inspected: descriptive.',
        sources={k: dict(path=str(p), sha256=sha256(p)) for k, p in SOURCES.items()},
        results=rows), indent=1))
    lines = ['# Paired tests at validation-selected radii (Protocol A): NN vs SRO-FA', '',
             'Both methods: same 40/160/640 training rows, radius chosen per seed on the same 10/40/160',
             'validation rows, same 10,000-row seed-9999 test. Difference = NN - SRO-FA (negative = NN cheaper).',
             'Radius rule (2026-11-18): largest radius within 1e-6 of the minimum validation score.',
             'Shipping: 0.1 per raw q2 > 0 for both. Read-only analysis of saved per-seed test means.', '',
             '| Setting | N | NN | SRO-FA | NN-SRO-FA (SE) | % | NN cheaper | t-test p | Wilcoxon p |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['label']} | {r['Ntotal']} | {r['nn_mean']:.3f} | {r['sro_fa_mean']:.3f} | "
                     f"{r['diff_mean']:+.3f} ({r['diff_se']:.3f}) | {r['diff_pct']:+.1f}% | "
                     f"{r['nn_cheaper']}/10 | {r['t_p']:.2g} | {r['wilcoxon_exact_p']:.2g} |")
    lines += ['', 'With n=10 the smallest exact two-sided Wilcoxon p is 2/1024 = 0.002.',
              'Caveats: validation rows also chose the NN learning rate; the test set was previously',
              'inspected, so results are descriptive. Sources and SHA-256: paired_tests_validation.json.']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
