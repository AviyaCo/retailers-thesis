"""Presentation only; minimal adaptation of comparison_20260924/plot_comparison.

No common/evaluator imports, checkpoint loading, inference, selection or solves.
build_plot_data(manifest, rows) reads only hash-bound frozen baseline JSONs.
render(data, output) writes comparison_shipping.png and comparison_shipping.pdf.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

SIZES = (50, 200, 800)
SEEDS = list(range(10))
GREEN_RADII = [0., .001, .003, .01, .03, .2, .4, .7, 1.5]
TICKS = [0., .001, .003, .01, .03, .1, .2, .4, .7, 1.5]
COLORS = dict(green='#00865a', green_is='#73b998', legacy='#2465aa',
              legacy_is='#80b2d8', sro_fa='#c43d3d', saa_knn='#c99b08')
NEURAL_FEE = dict(amount=.1, comparison='>', threshold=0., quantity='raw q2',
                  projection='common strict Gurobi', native_moreau=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _values(values, shape, name):
    a = np.asarray(values, dtype=float)
    _require(a.shape == shape and np.isfinite(a).all(), f'Invalid/nonfinite {name}')
    return a


def _aggregate(values):
    a = np.asarray(values, dtype=float)
    return dict(per_seed=a.tolist(), mean=a.mean(axis=-1).tolist(),
                p20=np.percentile(a, 20, axis=-1).tolist(),
                p80=np.percentile(a, 80, axis=-1).tolist())


def _grid(values, name, zero=True):
    a = _values(values, (len(values),), name)
    _require(len(a) > 0 and (np.diff(a) > 0).all() and (a >= 0).all()
             and (a <= 1.5).all(), f'Invalid ordered radius grid: {name}')
    _require(not zero or a[0] == 0, f'Missing radius zero: {name}')
    return a.tolist()


def _frozen_json(manifest, source):
    """Never read live originals or substitute a different cache."""
    _require(source in manifest['files'], f'Unbound frozen source: {source}')
    info = manifest['files'][source]
    _require(Path(source).is_absolute() and Path(info['copy']).is_absolute(),
             'Frozen paths must be absolute')
    payload = Path(info['copy']).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == info['sha256'],
             f'Frozen source hash mismatch: {source}')
    return json.loads(payload)


def _key(row):
    _require(row['method'] in ('green', 'legacy') and type(row['Ntotal']) is int
             and row['Ntotal'] in SIZES and type(row['seed']) is int
             and row['seed'] in SEEDS, 'Invalid method/size/seed identity')
    rho = float(row['radius'])
    _require(np.isfinite(rho) and 0 <= rho <= 1.5, 'Invalid radius identity')
    return row['method'], row['Ntotal'], rho, row['seed']


def _index(rows):
    indexed, ids = {}, set()
    for row in rows:
        key = _key(row)
        _require(key not in indexed and row['id'] not in ids, f'Duplicate policy: {key}')
        indexed[key] = row
        ids.add(row['id'])
    return indexed


def _test_protocol(manifest):
    test = manifest['test']
    _require(manifest['n_test'] == test['n_test'] == 10000
             and manifest['test_seed'] == test['seed'] == 9999, 'Test protocol mismatch')
    return test


def _validate_row(row, test):
    key = _key(row)
    _require(row['status'] == 'completed', f'Failed/partial policy: {key}')
    train = 4*row['Ntotal']//5 if row['method'] == 'green' else row['Ntotal']
    _require(row['Ntrain'] == train and row['Nval'] == row['Ntotal']-train,
             f'Training budget mismatch: {key}')
    sizes = row['sizes']
    _require(len(sizes) == train and all(type(s) is int and s > 0 for s in sizes),
             f'Invalid saved pool sizes: {key}')
    count = 10000 + sum(sizes)
    _require(row['solves_attempted'] == row['solves_accepted'] == count,
             f'Incomplete strict solves: {key}')
    names = [name for name in ('status_counts', 'solver_status_counts') if name in row]
    _require(bool(names) and all(row[name] == {'2': count} for name in names),
             f'Invalid strict status counts: {key}')
    if 'test' in row:
        _require(row['test'] == test, f'Worker test identity mismatch: {key}')
    for name, expected in (('n_test', 10000), ('test_seed', 9999)):
        if name in row:
            _require(row[name] == expected, f'Worker test protocol mismatch: {key}')
    if 'fee_rule' in row:
        _require(row['fee_rule'] == NEURAL_FEE, f'Neural fee provenance mismatch: {key}')
    _values([row['mean_oos'], row['finite_pool_mean']], (2,), 'neural costs')


def _neural(index, method, n, radii):
    groups = [[index[method, n, rho, seed] for seed in SEEDS] for rho in radii]
    # Preserve all solver/timing/source fields, without duplicating scenario arrays.
    evidence = [[{k: deepcopy(v) for k, v in r.items() if k not in ('pools', 'centers')}
                 for r in group] for group in groups]
    return dict(method=method, label='Green Moreau' if method == 'green' else 'Historical neural (Clarabel)',
        Ntotal=n, Ntrain=4*n//5 if method == 'green' else n, seeds=SEEDS.copy(), radius=radii,
        oos=_aggregate([[r['mean_oos'] for r in g] for g in groups]),
        finite_pool=_aggregate([[r['finite_pool_mean'] for r in g] for g in groups]),
        provenance=dict(cached=False, evaluation='fresh common strict Gurobi replay',
            training_solver='Moreau' if method == 'green' else 'Clarabel',
            fee_rule=deepcopy(NEURAL_FEE), per_seed=evidence))


def _baselines(manifest, n):
    paths = manifest['baseline_paths'][str(n)]
    cache = _frozen_json(manifest, paths['cache'])
    meta, benchmarks = cache['meta'], cache['benchmarks']
    _require(meta['N_train'] == n and meta['N_test'] == 10000
             and meta['seed_test'] == 9999 and meta['seeds'] == SEEDS,
             'Historical cache protocol/seed order mismatch')
    if n == 800:
        _require(meta.get('experiment') == 'fixed_shipping_N800_corrected_minibatch'
                 and meta.get('sweep_source') == 'N800/rho_sweep_minibatch',
                 'N800 cache must be the corrected minibatch artifact')
    if 'shipping_evaluation' in meta:
        _require(meta['shipping_evaluation'] == 'hard charge 0.1 when Q2 > 1e-6',
                 'Historical fee provenance mismatch')
    knn = benchmarks['saa_knn']
    selected = knn['selected_k_by_seed']
    _require(benchmarks['_knn_k_selection'] == 'shipping_per_seed_loo_grid_v1'
             and knn['selection_metric'] == 'per_seed_loo_validation_cost'
             and len(selected) == 10
             and all(type(k) is int and k in benchmarks['knn_k_grid'] for k in selected),
             'Invalid historical per-seed LOO kNN selection')
    knn_curve = dict(method='saa_knn', label='SAA-kNN', Ntotal=n, Ntrain=n, seeds=SEEDS.copy(),
        oos=_aggregate(_values(knn['oos'], (10,), 'kNN OOS')),
        provenance=dict(source=paths['cache'], cached=True, strict_certification=False,
            current_evaluation_time_s=None, strict_test_status='not_available',
            fee_rule=dict(amount=.1, comparison='>', threshold=1e-6, quantity='q2',
                          basis='Historical shipping evaluator; not common strict replay'),
            metadata=deepcopy(meta), runtime=deepcopy(cache.get('runtime')),
            original=deepcopy(benchmarks),
            interpretation='Saved per-seed training LOO choice; no refitting or new selection.'))

    source = _frozen_json(manifest, paths['sro'])
    by_radius, metadata = {}, {}
    for key, group in source.items():
        # Mixed top-level metadata is not a radius. Unknown keys fail, not vanish.
        if key in ('meta', '_meta', 'metadata', 'runtime'):
            metadata[key] = deepcopy(group)
            continue
        rho = float(key)
        _require(np.isfinite(rho) and rho >= 0 and rho not in by_radius,
                 'Invalid/duplicate SRO-FA radius')
        _require(set(group) == {str(s) for s in SEEDS}, f'Incomplete SRO-FA seeds: {n}, {rho}')
        cohort = [group[str(s)] for s in SEEDS]
        for row in cohort:
            _require(row.get('status', 'completed') == 'completed', 'Failed/partial historical SRO-FA')
            if 'solver_status' in row:
                _require(row['solver_status'] in (2, 9), 'Failed historical SRO-FA solver')
                if row['solver_status'] == 9:
                    _require(row.get('sol_count', 0) > 0, 'SRO-FA time limit without incumbent')
            _values([row['oos']], (1,), 'SRO-FA OOS')
        by_radius[rho] = cohort
    radii = _grid(sorted(r for r in by_radius if r <= 1.5), 'SRO-FA', zero=False)
    groups = [by_radius[r] for r in radii]
    sro = dict(method='sro_fa', label='SRO-FA', Ntotal=n, Ntrain=n, seeds=SEEDS.copy(), radius=radii,
        oos=_aggregate([[r['oos'] for r in g] for g in groups]),
        provenance=dict(source=paths['sro'], cached=True, strict_certification=False,
            current_evaluation_time_s=None, strict_test_status='not_available',
            fee_rule=dict(amount=.1, comparison='>', threshold=1e-10, quantity='q2',
                          basis='Historical shipping SRO-FA evaluator'),
            per_seed=deepcopy(groups), metadata=metadata,
            excluded_radii=sorted(r for r in by_radius if r > 1.5),
            interpretation='Historical descriptive comparator. Missing solver status remains unknown; '
                'saved status 9 incumbents are retained, not certified optimal. '
                'time_lp is the historical field name, including shipping MIP time.'))

    conditional = _frozen_json(manifest, paths['conditional'])
    _require(conditional['N'] * conditional['M'] == conditional['scenarios'] == 1000000
             and conditional['seed'] == 42 and conditional['relaxed'] is False
             and conditional['exact_mip'] is True, 'Conditional reference protocol mismatch')
    _require(conditional['status'] == 'OPTIMAL' and conditional['sol_count'] > 0,
             'Incomplete conditional reference')
    value = _values([conditional['saa_cond_mip_1m_shipping_exact']], (1,), 'conditional reference')[0]
    horizontal = dict(kind='conditional', label='Conditional reference (1M)', value=float(value),
        source=paths['conditional'], cached=True, current_evaluation_time_s=None,
        metadata=deepcopy(conditional), interpretation='Finite 1M-scenario clairvoyant MIP estimate; '
            'not an exact population bound. OPTIMAL is the saved tolerance-based solver status; '
            'the nonzero MIP gap and best bound are preserved. Shared source reuse is not a new solve.')
    return sro, knn_curve, horizontal


def build_plot_data(manifest, rows) -> dict:
    """Validate complete fixed cohorts and return JSON-safe presentation data.

    Requires frozen manifest/test, jobs, files and baseline_paths; legacy_radii
    is optional and otherwise inferred from jobs, never costs or completed rows.
    Rows must copy every job field and include completed status, both costs,
    solve counts and status_counts (or solver_status_counts). All ten seeds
    are mandatory at every radius. Checkpoint/state identities are not inferred.
    """
    _require(manifest['status'] == 'frozen', 'Manifest must be frozen')
    test = _test_protocol(manifest)
    specs, indexed = _index(manifest['jobs']), _index(rows)
    grids = {}
    for n in SIZES:
        inferred = sorted({k[2] for k in specs if k[:2] == ('legacy', n)})
        supplied = manifest.get('legacy_radii', {}).get(str(n), inferred)
        grids[n] = _grid(supplied, f'legacy N={n}')
        _require(grids[n] == inferred, 'Legacy grid differs from manifest jobs')
    expected = {(method, n, rho, seed) for n in SIZES for method in ('green', 'legacy')
                for rho in (GREEN_RADII if method == 'green' else grids[n]) for seed in SEEDS}
    _require(set(specs) == set(indexed) == expected, 'Incomplete/mismatched fixed policy grid')
    for key, row in indexed.items():
        spec = specs[key]
        _require(all(k in row and row[k] == v for k, v in spec.items()),
                 f'Policy identity differs from manifest: {key}')
        _require(spec['checkpoint'] in manifest['files'] and
                 spec['checkpoint_sha256'] == manifest['files'][spec['checkpoint']]['sha256'],
                 f'Checkpoint hash differs from frozen manifest: {key}')
        _validate_row(row, test)
    panels = []
    for n in SIZES:
        sro, knn, reference = _baselines(manifest, n)
        panels.append(dict(Ntotal=n, label=f'N = {n}',
            training_counts=dict(green=4*n//5, legacy=n),
            green=_neural(indexed, 'green', n, GREEN_RADII.copy()),
            legacy=_neural(indexed, 'legacy', n, grids[n]),
            sro_fa=sro, saa_knn=knn, horizontal=[reference]))
    return dict(schema_version=1, status='completed', panels=panels, seeds=SEEDS.copy(),
        legacy_radii={str(n): grids[n] for n in SIZES}, test=deepcopy(test),
        band_definition='Pointwise 20th/80th percentiles across ten training seeds; not confidence intervals.',
        is_definition='Mean of per-center maxima on the returned checkpoint own saved finite MP pool; '
                      'not the full adversarial supremum.',
        caption='Both neural families are fresh common strict Gurobi replays with raw q2 > 0 fees; '
            'not native Moreau behavior. Historical neural training used Clarabel, not SCS. '
            'Green is the approved hybrid of 265 original and 5 advisory-continuation policies. '
            'No black shipping-training comparator exists. Cached kNN uses q2 > 1e-6; '
            'cached SRO-FA uses q2 > 1e-10, retains time-limited incumbents and missing statuses, '
            'and is not freshly evaluated or certified. The conditional 1M MIP estimate is not '
            'an exact population bound. Fixed radii only; no test-based annotation, selection, '
            'interpolated evaluations, refitting or new reference solve.',
        footnote='Neural IS/OOS: common strict Gurobi, raw q2 > 0 (not native Moreau). '
            'Training: green 40/160/640; historical 50/200/800.\n'
            'Bands: seed 20–80%, not CI. Green: 265 original + 5 advisory. '
            'Cached kNN/SRO-FA: q2 > 1e-6 / 1e-10, not certified.\n'
            'Gray: finite 1M MIP estimate, not an exact population bound. '
            'Fixed radii; no test-based selection.',
        provenance=dict(experiment=manifest.get('experiment'), files=deepcopy(manifest['files']),
            workers=manifest.get('workers'), threads=manifest.get('threads'),
            runtime_note='Raw saved phase/solver timings retained where available. Missing timings '
                'are unavailable; rerun required. Current replay time is not training time; '
                'historical loading time is not model runtime. No concurrent times are summed.'))


def _validate_plot_data(data):
    _require(data['status'] == 'completed' and data['seeds'] == SEEDS, 'Incomplete plot/seed order')
    _require(data['test']['n_test'] == 10000 and data['test']['seed'] == 9999, 'Test protocol mismatch')
    _require([p['Ntotal'] for p in data['panels']] == list(SIZES), 'Missing/reordered panels')
    for panel in data['panels']:
        for method in ('green', 'legacy', 'sro_fa', 'saa_knn'):
            curve = panel[method]
            _require(curve['seeds'] == SEEDS, 'Curve seed order mismatch')
            provenance = curve['provenance']
            if method in ('sro_fa', 'saa_knn'):
                fee = provenance['fee_rule']
                _require(fee['amount'] == .1 and fee['comparison'] == '>'
                         and fee['threshold'] == (1e-10 if method == 'sro_fa' else 1e-6)
                         and provenance['cached'] is True
                         and provenance['strict_certification'] is False,
                         'Historical fee/certification provenance mismatch')
            if method == 'saa_knn':
                shape = (10,)
                expected = provenance['original']['saa_knn']['oos']
            else:
                radii = _grid(curve['radius'], method, zero=method != 'sro_fa')
                if method in ('green', 'legacy'):
                    _require(radii == (GREEN_RADII if method == 'green' else
                             data['legacy_radii'][str(panel['Ntotal'])]), 'Neural grid mismatch')
                    _require(provenance['fee_rule'] == NEURAL_FEE, 'Neural fee provenance mismatch')
                shape = (len(radii), 10)
                groups = provenance['per_seed']
                _require(len(groups) == len(radii) and all(len(g) == 10 for g in groups),
                         'Incomplete per-seed provenance')
                if method in ('green', 'legacy'):
                    for rho, group in zip(radii, groups):
                        _require([r['seed'] for r in group] == SEEDS, 'Provenance seed order mismatch')
                        for row in group:
                            _require(_key(row)[:3] == (method, panel['Ntotal'], rho),
                                     'Provenance identity mismatch')
                            _validate_row(row, data['test'])
                expected = [[r['mean_oos' if method in ('green', 'legacy') else 'oos']
                             for r in g] for g in groups]
            for name in ('oos', 'finite_pool') if method in ('green', 'legacy') else ('oos',):
                stats = curve[name]
                if name == 'finite_pool':
                    expected = [[r['finite_pool_mean'] for r in g] for g in provenance['per_seed']]
                actual = _aggregate(_values(stats['per_seed'], shape, f'{method} {name}'))
                _require(stats == actual and stats['per_seed'] == expected,
                         'Aggregates/values differ from per-seed provenance')
        for h in panel['horizontal']:
            _values([h['value']], (1,), 'reference level')
            _require(h['value'] == h['metadata']['saa_cond_mip_1m_shipping_exact'],
                     'Conditional value differs from provenance')


def render(data, output: Path):
    """Validate before writing; output is a directory, not a file prefix."""
    _validate_plot_data(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    output = Path(output)
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'dejavuserif', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.8))
        try:
            handles = {}
            for ax, panel in zip(axes, data['panels']):
                for method in ('green', 'legacy', 'sro_fa'):
                    curve = panel[method]
                    x, stats = curve['radius'], curve['oos']
                    label = curve['label'] + ' OOS'
                    line, = ax.plot(x, stats['mean'], color=COLORS[method], lw=1.9,
                                    marker='o', ms=2.5, label=label, zorder=4)
                    handles.setdefault(label, line)
                    ax.fill_between(x, stats['p20'], stats['p80'], color=COLORS[method], alpha=.12, lw=0)
                    if method != 'sro_fa':
                        label = curve['label'] + ' IS'
                        line, = ax.plot(x, curve['finite_pool']['mean'], '--', color=COLORS[method+'_is'],
                                        lw=1.7, label=label, zorder=3)
                        handles.setdefault(label, line)
                knn = panel['saa_knn']['oos']
                line = ax.axhline(knn['mean'], color=COLORS['saa_knn'], lw=1.8, label='SAA-kNN OOS')
                ax.axhspan(knn['p20'], knn['p80'], color=COLORS['saa_knn'], alpha=.08, lw=0)
                handles.setdefault('SAA-kNN OOS', line)
                for h in panel['horizontal']:
                    line = ax.axhline(h['value'], color='#888888', ls='-.', lw=1.25, label=h['label'])
                    handles.setdefault(h['label'], line)
                ax.set_xscale('symlog', linthresh=.001, linscale=.8)
                ax.set_xlim(0, 1.5)
                ax.set_xticks(TICKS)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(axis='y', color='#dddddd', lw=.6, alpha=.8)
                ax.set_axisbelow(True)
                ax.set_title(panel['label'], fontsize=12, pad=10)
                ax.set_xlabel(r'Radius $\rho$')
            axes[0].set_ylabel('Cost')
            fig.suptitle(r'Fixed-radius shipping policy comparison · $R=3$', fontsize=14, y=.98)
            fig.legend(list(handles.values()), list(handles), loc='lower center',
                       bbox_to_anchor=(.5, .135), ncol=4, frameon=False, fontsize=8.5,
                       handlelength=2.7, columnspacing=1.5)
            fig.text(.5, .025, data['footnote'], ha='center', va='bottom', fontsize=8,
                     linespacing=1.5, color='#555555')
            fig.subplots_adjust(left=.065, right=.988, top=.865, bottom=.36, wspace=.24)
            output.mkdir(parents=True, exist_ok=True)
            fig.savefig(output / 'comparison_shipping.png', dpi=300)
            fig.savefig(output / 'comparison_shipping.pdf')
        finally:
            plt.close(fig)