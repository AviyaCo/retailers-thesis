"""Presentation only: frozen fixed-radius tables in, paper figure out.

No evaluator imports, models, solvers, scenario generation, or test selection.
``output`` in render is a directory. Files are comparison_R3.png (300 dpi)
and comparison_R3.pdf (vector). The parent persists the returned plot data.
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
SCS_COUNTS = {50: 24, 200: 18, 800: 18}
COLORS = dict(green='#00865a', green_is='#73b998', scs='#2465aa',
              scs_is='#80b2d8', sro_fa='#c43d3d', saa_knn='#c99b08')


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


def _grid(values, name, count=None, zero=True):
    a = _values(values, (len(values),), name)
    _require(len(a) > 0 and (np.diff(a) > 0).all() and (a >= 0).all()
             and (a <= 1.5).all(), f'Invalid ordered radius grid: {name}')
    _require(not zero or a[0] == 0, f'Missing radius zero: {name}')
    _require(count is None or len(a) == count, f'Grid count mismatch: {name}')
    return a.tolist()


def _frozen_json(manifest, source):
    """Read only the checksummed copy; never fall back to a live source."""
    _require(str(source) in manifest['files'], f'Unbound frozen source: {source}')
    info = manifest['files'][str(source)]
    payload = Path(info['copy']).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == info['sha256'],
             f'Frozen source hash mismatch: {source}')
    return json.loads(payload)


def _key(row):
    _require(row['method'] in ('green', 'scs') and type(row['Ntotal']) is int
             and row['Ntotal'] in SIZES and type(row['seed']) is int
             and row['seed'] in SEEDS, 'Invalid method/size/seed identity')
    rho = float(row['radius'])
    _require(np.isfinite(rho) and 0 <= rho <= 1.5, 'Invalid radius identity')
    return row['method'], row['Ntotal'], rho, row['seed']


def _index(rows):
    indexed = {}
    for row in rows:
        key = _key(row)
        _require(key not in indexed, f'Duplicate policy: {key}')
        indexed[key] = row
    return indexed


def _array_identity(current, cached):
    for name in ('xi1', 'xi2'):
        a, b = current[name], cached[name]
        _require(a['shape'] == b['shape'] == [10000, 3]
                 and a['dtype'] == b['dtype'] == 'float64'
                 and bool(a['array_sha256']) and a['array_sha256'] == b['array_sha256'],
                 f'Test array identity mismatch: {name}')


def _evidence(row):
    fields = ('id', 'seed', 'radius', 'status', 'mean_oos', 'test_mean', 'finite_pool_mean',
              'checkpoint', 'checkpoint_sha256',
              'state_sha256', 'manifest_sha256', 'selected_candidate_id',
              'stopping_candidate_id', 'source_summary_path', 'wall_time_s',
              'start_utc', 'end_utc', 'phases', 'worker_count', 'solver_threads',
              'iterations', 'environment', 'training_runtime_reference',
              'scs_historical_membership')
    return {k: deepcopy(row[k]) for k in fields if k in row}


def _neural(index, method, n, radii, cached=False):
    cohort = [[index[method, n, rho, seed] for seed in SEEDS] for rho in radii]
    oos = [[r['test_mean' if cached else 'mean_oos'] for r in group] for group in cohort]
    inside = [[r['finite_pool_mean'] for r in group] for group in cohort]
    shape = (len(radii), 10)
    return dict(method=method, label='Green Moreau' if method == 'green' else 'SCS',
                Ntotal=n, Ntrain=4*n//5 if method == 'green' else n, seeds=SEEDS.copy(),
                radius=radii, oos=_aggregate(_values(oos, shape, f'{method} OOS')),
                finite_pool=_aggregate(_values(inside, shape, f'{method} IS')),
                provenance=dict(cached=cached, evaluation='cached strict replay' if cached
                                else 'fresh strict replay',
                                current_evaluation_time_s=None if cached else
                                [[r.get('wall_time_s') for r in group] for group in cohort],
                                per_seed=[[_evidence(r) for r in group] for group in cohort]))


def _references(manifest, baseline, n, old_plot):
    horizontal = []
    cache_source = baseline.get('cache')
    cache = _frozen_json(manifest, cache_source) if cache_source is not None else None
    if cache is not None:
        meta = cache['meta']
        _require(meta['N_train'] == n and meta['N_test'] == 10000
                 and meta['seed_test'] == 9999 and meta['seeds'] == SEEDS,
                 'Historical cache protocol/seed order mismatch')
        actual = sorted(float(r) for r in cache['sweep'] if 0 <= float(r) <= 1.5)
        _require(actual == manifest['scs_radii'][str(n)], 'SCS grid differs from frozen cache')
        _require(cache['benchmarks']['saa_knn']['oos'] == baseline['saa_knn']['oos'],
                 'SAA-kNN differs from frozen cache')
        saa = cache['benchmarks'].get('saa_lb')
        if saa is not None:
            stats = _aggregate(_values(saa['oos'], (10,), 'SAA-LP reference'))
            horizontal.append(dict(kind='saa_lp', label='SAA-LP reference', value=stats['mean'],
                oos=stats, seeds=SEEDS.copy(), source=cache_source,
                source_label=f'SAA-LP (historical {meta["N_test"]:,}-sample test)',
                metadata=deepcopy(meta), interpretation='Historical descriptive oracle reference; '
                'not a newly certified population lower bound.'))
    # The old figure contains one SAA-kNN line and two explicitly identified
    # reference levels. Do not reuse its SAA-kNN or guess at other horizontals.
    for item in old_plot.get('horizontal', []):
        label, value = (item['label'], item['value']) if isinstance(item, dict) else item[:2]
        kind = ('conditional' if label.lower().startswith('conditional lb') else
                'saa_lp' if label.lower().startswith('saa-lp lb') else None)
        if kind is None or any(h['kind'] == kind for h in horizontal):
            continue
        _values([value], (1,), 'historical reference level')
        horizontal.append(dict(kind=kind, label='Conditional reference' if kind == 'conditional'
            else 'SAA-LP reference', value=float(value), source_label=label,
            source=manifest['cached_green_N50']['plot_data'],
            interpretation='Frozen historical shared reference, not an N-specific or matched-test '
            'bound; original sampling qualification retained in source_label.'))
    return horizontal


def build_plot_data(manifest, rows) -> dict:
    """Validate all fixed cohorts, align by seed ID, and return JSON-safe data.

    Asynchronous row arrival order is immaterial: output columns are always
    seeds 0..9. Unlabelled cached vectors require the frozen canonical order.
    Missing/duplicate/failed/nonfinite policies or mismatched grids are fatal.
    Historical SRO-FA has no strict solver-status certificate; absence is
    disclosed, not interpreted as a successful new strict solve.
    """
    _require(manifest['status'] == 'frozen', 'Manifest must be frozen')
    test = manifest['test']
    _require(test['n_test'] == manifest['n_test'] == 10000
             and test['seed'] == manifest['test_seed'] == 9999, 'Test protocol mismatch')
    cached = manifest['cached_green_N50']
    _array_identity(test['arrays'], cached['test_arrays'])
    grids = {n: _grid(manifest['scs_radii'][str(n)], f'SCS N={n}', SCS_COUNTS[n]) for n in SIZES}
    expected = {(method, n, rho, seed) for method in ('green', 'scs') for n in SIZES
                if method == 'scs' or n != 50
                for rho in (GREEN_RADII if method == 'green' else grids[n]) for seed in SEEDS}
    specs, indexed = _index(manifest['jobs']), _index(rows)
    _require(set(specs) == set(indexed) == expected, 'Incomplete/mismatched fixed policy grid')
    for key, row in indexed.items():
        _require(row['status'] == 'completed', f'Failed/partial policy: {key}')
        _require(all(k in row and row[k] == v for k, v in specs[key].items()),
                 f'Policy identity differs from manifest: {key}')
        if 'Ntrain' in row:
            _require(row['Ntrain'] == (4*row['Ntotal']//5 if row['method'] == 'green'
                                      else row['Ntotal']), f'Training budget mismatch: {key}')
        if 'test' in row:
            _require(row['test'] == test, f'Worker test identity mismatch: {key}')
        if 'solves_attempted' in row:
            _require(row['solves_attempted'] == row['solves_accepted'] == 10000 + sum(row['sizes'])
                     and row['status_counts'] == {'2': row['solves_accepted']},
                     f'Incomplete strict solves: {key}')
    old_index = _index([dict(r, Ntotal=50, method='green') for r in cached['jobs']])
    _require(set(old_index) == {('green', 50, rho, s) for rho in GREEN_RADII for s in SEEDS},
             'Incomplete cached green N50 grid')
    _require(all(r['status'] == 'completed' for r in old_index.values()), 'Failed cached green N50')
    old_plot = (_frozen_json(manifest, cached['plot_data']) if 'plot_data' in cached else {})
    if old_plot:
        _array_identity(test['arrays'], old_plot['test_arrays'])
    panels = []
    for n in SIZES:
        baseline = manifest['cached_baselines'][str(n)]
        green = _neural(old_index if n == 50 else indexed, 'green', n, GREEN_RADII.copy(), n == 50)
        scs = _neural(indexed, 'scs', n, grids[n])
        if n == 50 and 'green' in old_plot:
            p = old_plot['green']
            _require(p['radius'] == green['radius'] and p['oos_per_seed'] == green['oos']['per_seed']
                     and p['is_per_seed'] == green['finite_pool']['per_seed'], 'Cached green plot mismatch')
        source = baseline['sro_fa']
        if 'sro_source' in baseline:
            _require(_frozen_json(manifest, baseline['sro_source']) == source,
                     'SRO-FA differs from frozen source')
        by_radius = {}
        for key, group in source.items():
            rho = float(key)
            _require(np.isfinite(rho) and rho >= 0 and rho not in by_radius, 'Invalid SRO-FA radius')
            by_radius[rho] = group
        radii = _grid(sorted(r for r in by_radius if r <= 1.5), 'SRO-FA', zero=False)
        sro_rows = []
        for rho in radii:
            group = by_radius[rho]
            _require(set(group) == {str(s) for s in SEEDS}, f'Incomplete SRO-FA seeds: {n}, {rho}')
            cohort = [group[str(seed)] for seed in SEEDS]
            _require(all(r.get('status', 'not_available') in ('completed', 'not_available') for r in cohort),
                     'Failed/partial historical SRO-FA')
            sro_rows.append(cohort)
        sro = dict(method='sro_fa', label='SRO-FA', Ntotal=n, Ntrain=n, seeds=SEEDS.copy(), radius=radii,
            oos=_aggregate(_values([[r['oos'] for r in group] for group in sro_rows],
                                  (len(radii), 10), 'SRO-FA OOS')),
            provenance=dict(source=baseline.get('sro_source'), cached=True,
                strict_test_status='not_available', current_evaluation_time_s=None,
                interpretation='Historical descriptive test reference, not fresh strict evaluation.',
                per_seed=deepcopy(sro_rows)))
        knn = dict(method='saa_knn', label='SAA-kNN', Ntotal=n, Ntrain=n, seeds=SEEDS.copy(),
            oos=_aggregate(_values(baseline['saa_knn']['oos'], (10,), 'SAA-kNN OOS')),
            provenance=dict(source=baseline.get('cache'), cached=True,
                strict_test_status='not_available', current_evaluation_time_s=None,
                original=deepcopy(baseline['saa_knn']),
                interpretation='Historical descriptive test reference; saved training LOO choice of k.'))
        panels.append(dict(Ntotal=n, label=f'N = {n}', training_counts=dict(green=4*n//5, scs=n),
            green=green, scs=scs, sro_fa=sro, saa_knn=knn,
            horizontal=_references(manifest, baseline, n, old_plot)))
    selected = []
    if 'saved_green_selections' in manifest:
        winners = manifest['saved_green_selections']
        _require(len(winners) == 20 and {(w['Ntotal'], w['seed']) for w in winners}
                 == {(n, s) for n in (200, 800) for s in SEEDS}, 'Incomplete saved validation selections')
        for winner in sorted(winners, key=lambda w: (w['Ntotal'], w['seed'])):
            key = ('green', winner['Ntotal'], float(winner['radius']), winner['seed'])
            _require(key in indexed, 'Saved validation selection outside fixed grid')
            row = indexed[key]
            for field in ('checkpoint_sha256', 'state_sha256'):
                if field in winner:
                    _require(winner[field] == row[field], 'Saved validation checkpoint mismatch')
            selected.append(dict(selection=deepcopy(winner), mean_oos=row['mean_oos'],
                                 finite_pool_mean=row['finite_pool_mean']))
    return dict(schema_version=1, status='completed', panels=panels, seeds=SEEDS.copy(),
        scs_radii={str(n): grids[n] for n in SIZES}, test=deepcopy(test),
        band_definition='Pointwise 20th/80th percentiles across ten training seeds; not confidence intervals.',
        is_definition='Mean of per-center maxima on the returned checkpoint own saved finite MP pool; '
                      'not the full adversarial supremum.',
        caption='Fixed radii only; no test-based radius selection or interpolated evaluations. '
                'Green N50 is cached; green N200/N800 and all SCS evaluations are fresh strict replays. '
                'SCS saved pool coordinates are unchanged: positive box residuals are observationally '
                'reported, not box-certified or repaired. SRO-FA/SAA-kNN and gray reference levels '
                'are historical descriptive comparators; strict historical test status is unavailable.',
        footnote='Training: green 40/160/640; SCS 50/200/800. '
                 'Bands: seed 20–80%, not CI. Historical baselines: descriptive tests.',
        provenance=dict(experiment=manifest.get('experiment'), files=deepcopy(manifest['files']),
            cached_green_summary=cached.get('summary'), workers=manifest.get('workers'),
            threads=manifest.get('threads'),
            runtime_note='Raw saved timings retained where available. Missing timings are unavailable; '
                         'rerun required. Cached loading time is not model runtime.'),
        validation_selected_oos=selected,
        selection_interpretation='Only already-frozen validation selections are reported; no new selection.')


def _validate_plot_data(data):
    _require(data['status'] == 'completed' and data['seeds'] == SEEDS, 'Incomplete plot/seed order')
    _require([p['Ntotal'] for p in data['panels']] == list(SIZES), 'Missing/reordered panels')
    for p in data['panels']:
        n = p['Ntotal']
        for method in ('green', 'scs', 'sro_fa', 'saa_knn'):
            curve = p[method]
            _require(curve['seeds'] == SEEDS, 'Curve seed order mismatch')
            if method == 'saa_knn':
                shape = (10,)
            else:
                radii = _grid(curve['radius'], method, zero=method != 'sro_fa')
                if method == 'green':
                    _require(radii == GREEN_RADII, 'Green grid mismatch')
                if method == 'scs':
                    _require(radii == data['scs_radii'][str(n)] and len(radii) == SCS_COUNTS[n],
                             'SCS grid mismatch')
                shape = (len(radii), 10)
            for name in ('oos', 'finite_pool') if method in ('green', 'scs') else ('oos',):
                stats = curve[name]
                actual = _aggregate(_values(stats['per_seed'], shape, f'{method} {name}'))
                _require(all(stats[k] == actual[k] for k in ('mean', 'p20', 'p80')),
                         'Aggregates differ from per-seed arrays')
                provenance = curve['provenance']
                if method == 'saa_knn':
                    expected = provenance['original']['oos']
                else:
                    groups = provenance['per_seed']
                    _require(len(groups) == len(curve['radius']) and all(len(g) == 10 for g in groups),
                             'Incomplete per-seed provenance')
                    if method in ('green', 'scs'):
                        _require(all([r['seed'] for r in g] == SEEDS for g in groups),
                                 'Provenance seed order mismatch')
                        _require(all(r['radius'] == rho and r['status'] == 'completed'
                                     for rho, g in zip(curve['radius'], groups) for r in g),
                                 'Provenance radius/status mismatch')
                        field = ('finite_pool_mean' if name == 'finite_pool' else
                                 'test_mean' if provenance['cached'] else 'mean_oos')
                    else:
                        field = 'oos'
                    expected = [[r[field] for r in g] for g in groups]
                _require(stats['per_seed'] == expected, 'Per-seed values/order differ from provenance')
        for h in p['horizontal']:
            _values([h['value']], (1,), 'reference level')


def render(data, output: Path):
    """Validate before writing anything; render only the six requested curves."""
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
        fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.4))
        try:
            handles = {}
            for ax, panel in zip(axes, data['panels']):
                for method in ('green', 'scs', 'sro_fa'):
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
                    conditional = h['kind'] == 'conditional'
                    line = ax.axhline(h['value'], color='#888888' if conditional else '#555555',
                                     ls='-.' if conditional else ':', lw=1.25, label=h['label'])
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
            fig.suptitle(r'Fixed-radius policy comparison · $R=3$', fontsize=14, y=.98)
            fig.legend(list(handles.values()), list(handles), loc='lower center',
                       bbox_to_anchor=(.5, .072), ncol=4, frameon=False, fontsize=9,
                       handlelength=2.7, columnspacing=2.)
            fig.text(.5, .022, data['footnote'], ha='center', fontsize=8, color='#555555')
            fig.subplots_adjust(left=.065, right=.988, top=.865, bottom=.31, wspace=.24)
            output.mkdir(parents=True, exist_ok=True)
            fig.savefig(output / 'comparison_R3.png', dpi=300)
            fig.savefig(output / 'comparison_R3.pdf')
        finally:
            plt.close(fig)