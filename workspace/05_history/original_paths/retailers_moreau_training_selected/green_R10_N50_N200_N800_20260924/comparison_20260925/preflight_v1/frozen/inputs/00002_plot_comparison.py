"""Presentation only: frozen R10 fixed-policy replay tables to the R3-style figure.

No evaluator, model, solver, training, scenario generation, or test selection.
The parent persists build_plot_data's JSON-safe result. render writes a 300-dpi
comparison_R10.png and a vector comparison_R10.pdf into its output directory.
Input hashes/identities and per-center maxima are checked here; numerical pool
membership and checkpoint evaluation remain the frozen evaluator's responsibility.
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
SCS_RADII = [0., .001, .005, .01, .02, .05, .1, .15, .2, .3, .4,
             .45, .5, .6, .75, .8, 1., 1.5]
SCS_COUNTS = {50: 18, 200: 18, 800: 18}
TICKS = [0., .001, .003, .01, .03, .1, .2, .4, .7, 1.5]
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


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _frozen_bytes(manifest, source):
    """Read only the checksummed copy; no live-source or missing-data fallback."""
    _require(str(source) in manifest['files'], f'Unbound frozen source: {source}')
    info = manifest['files'][str(source)]
    payload = Path(info['copy']).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == info['sha256'],
             f'Frozen source hash mismatch: {source}')
    return payload


def _frozen_json(manifest, source):
    return json.loads(_frozen_bytes(manifest, source))


def validate_cache_protocol(cache, n, manifest):
    """Pure protocol check shared by freeze/preflight and presentation.

    The sole missing-R schema is the legacy N800 R10 10k cache. Its raw meta
    stays unchanged: dimension evidence certifies the bound replay checkpoint
    and training centers, NOT that adjacent scripts produced the cached costs.
    """
    meta = cache['meta']
    common = {'N_train', 'N_test', 'seed_test', 'seeds', 'projection_method'}
    recorded = 'R' in meta
    _require(type(n) is int and n in SIZES and manifest.get('R') == 10,
             'Historical cache protocol/seed order mismatch: manifest R must be 10')
    _require(set(meta) == (common | {'R'} if recorded else common)
             and meta['N_train'] == n and meta['N_test'] == 10000
             and meta['seed_test'] == 9999 and meta['seeds'] == SEEDS
             and meta['projection_method'] == 'gurobi_qp',
             'Historical cache protocol/seed order mismatch: unsupported meta schema')
    if recorded:
        _require(type(meta['R']) is int and meta['R'] == 10,
                 'Historical cache protocol/seed order mismatch: recorded R must be 10')
    else:
        _require(n == 800, 'Historical cache protocol/seed order mismatch: missing R only allowed for N800')
    baseline = manifest['cached_baselines'][str(n)]
    evidence = baseline.get('dimension_evidence')
    _require(isinstance(evidence, dict), 'Missing bound baseline dimension evidence')
    _require(evidence.get('schema') == 'scs_checkpoint_centers_R10_v1'
             and evidence.get('verified_R') == 10 and evidence.get('Ntotal') == n
             and evidence.get('validator') == 'c.load_model(old)', 'Invalid baseline dimension evidence')
    shapes = {'Q10': [1], 'Q1r': [10], 'fc_layers.0.weight': [16, 21],
              'fc_layers.0.bias': [16], 'fc_layers.2.weight': [16, 16],
              'fc_layers.2.bias': [16], 'fc_layers.4.weight': [10, 16],
              'fc_layers.4.bias': [10], 'proj_layer.q': [12, 12], 'proj_layer.A': [45, 12]}
    _require(evidence.get('state_shapes') == shapes and evidence.get('centers_shape') == [n, 20],
             'Baseline checkpoint/center dimension mismatch')
    jobs = [j for j in manifest['jobs'] if j['id'] == evidence.get('job_id')]
    _require(len(jobs) == 1, 'Unbound baseline dimension job')
    job = jobs[0]
    _require(job['method'] == 'scs' and job['Ntotal'] == job['Ntrain'] == n
             and job['seed'] == 0 and job['radius'] == 0., 'Baseline dimension job identity mismatch')
    _values(job.get('centers'), (n, 20), 'baseline dimension centers')
    for field in ('checkpoint', 'checkpoint_sha256', 'state_sha256'):
        _require(evidence.get(field) == job[field], f'Unbound baseline dimension {field}')
    _require(_digest(evidence['state_sha256']) and _digest(evidence['checkpoint_sha256']),
             'Invalid baseline dimension hashes')
    bindings = evidence.get('source_sha256')
    _require(isinstance(bindings, dict) and bool(baseline.get('source_references'))
             and set(baseline['source_references']).issubset(bindings),
             'Unbound baseline dimension sources')
    required = {baseline['cache'], job['checkpoint']}
    _require(required.issubset(bindings) and bindings[job['checkpoint']] == job['checkpoint_sha256']
             and job['checkpoint'] in job['input_files'], 'Unbound baseline dimension cache/checkpoint')
    _require({'evaluation.py', 'reference.py', 'compare.py'}.issubset({Path(s).name for s in bindings}),
             'Unbound baseline dimension validator sources')
    for source, digest in bindings.items():
        _require(_digest(digest) and source in manifest['files']
                 and manifest['files'][source]['sha256'] == digest, 'Unbound baseline dimension source hash')
    return dict(schema='recorded_R10' if recorded else 'legacy_N800_R10_10k_missing_R',
                recorded_R=meta['R'] if recorded else None, independently_verified_R=10,
                dimension_evidence=deepcopy(evidence))


def _key(row):
    _require(row['method'] in ('green', 'scs') and type(row['Ntotal']) is int
             and row['Ntotal'] in SIZES and type(row['seed']) is int
             and row['seed'] in SEEDS, 'Invalid method/size/seed identity')
    rho = float(row['radius'])
    _require(np.isfinite(rho) and 0 <= rho <= 1.5, 'Invalid radius identity')
    return row['method'], row['Ntotal'], rho, row['seed']


def _index(rows):
    indexed, ids = {}, set()
    for row in rows:
        key = _key(row)
        _require(key not in indexed, f'Duplicate policy: {key}')
        _require(isinstance(row.get('id'), str) and row['id'] and row['id'] not in ids,
                 f'Missing/duplicate policy ID: {key}')
        ids.add(row['id'])
        indexed[key] = row
    return indexed


def _test_identity(test):
    for name in ('xi1', 'xi2'):
        a = test['arrays'][name]
        _require(a['shape'] == [10000, 10] and a['dtype'] == 'float64'
                 and _digest(a['array_sha256']), f'Test array identity mismatch: {name}')


def _evidence(row):
    fields = ('id', 'seed', 'radius', 'status', 'mean_oos', 'finite_pool_mean',
              'per_center_maxima', 'checkpoint', 'checkpoint_sha256', 'state_sha256',
              'pool_sha256', 'pool_path', 'pool_file_sha256', 'sizes', 'input_files',
              'manifest_sha256', 'selected_candidate_id', 'stopping_candidate_id',
              'source_summary_path', 'wall_time_s', 'start_utc', 'end_utc', 'phases',
              'worker_count', 'solver_threads', 'iterations', 'environment',
              'training_runtime_reference', 'scs_historical_membership')
    return {k: deepcopy(row[k]) for k in fields if k in row}


def _neural(index, method, n, radii):
    cohort = [[index[method, n, rho, seed] for seed in SEEDS] for rho in radii]
    shape = (len(radii), 10)
    return dict(method=method, label='Green Moreau' if method == 'green' else 'SCS',
        Ntotal=n, Ntrain=4*n//5 if method == 'green' else n, seeds=SEEDS.copy(), radius=radii,
        oos=_aggregate(_values([[r['mean_oos'] for r in g] for g in cohort], shape, f'{method} OOS')),
        finite_pool=_aggregate(_values([[r['finite_pool_mean'] for r in g] for g in cohort],
                                      shape, f'{method} IS')),
        provenance=dict(cached=False, evaluation='fresh strict replay',
            current_evaluation_time_s=[[r.get('wall_time_s') for r in g] for g in cohort],
            per_seed=[[_evidence(r) for r in g] for g in cohort]))


def _oracle_reference(manifest, oracle):
    """Verify one fresh joint test-fitted LP, not ten independent OOS estimates.

    The parent seals the result artifact/hash separately. Source hashes here
    bind the oracle implementation and every declared LP dependency to frozen
    bytes; test identity includes all recorded array file/hash metadata.
    """
    required = {'method', 'status', 'R', 'n_test', 'test_seed', 'test', 'solver_status',
                'independent_solves', 'solves_attempted', 'solves_accepted',
                'status_counts', 'mean_cost', 'wall_time_s', 'source_sha256'}
    _require(isinstance(oracle, dict) and required.issubset(oracle), 'Missing oracle record fields')
    _require(oracle['method'] == 'empirical_clairvoyant_reference'
             and oracle['status'] == 'completed', 'Oracle method/completion mismatch')
    for field, value in (('R', 10), ('n_test', 10000), ('test_seed', 9999),
                         ('solver_status', 2), ('independent_solves', 1),
                         ('solves_attempted', 1), ('solves_accepted', 1)):
        _require(type(oracle[field]) is int and oracle[field] == value,
                 f'Oracle {field} mismatch')
    _require(oracle['status_counts'] == {'2': 1}
             and type(oracle['status_counts']['2']) is int, 'Oracle OPTIMAL count mismatch')
    _require(oracle['test'] == manifest['test'], 'Oracle test identity mismatch')
    _test_identity(oracle['test'])
    for name in ('xi1', 'xi2'):
        info = oracle['test']['arrays'][name]
        _require(isinstance(info.get('file'), str) and Path(info['file']).is_absolute()
                 and _digest(info.get('file_sha256')), 'Oracle test file identity missing')
    for field in ('mean_cost', 'wall_time_s'):
        _require(type(oracle[field]) in (int, float), f'Invalid oracle {field}')
        _values(oracle[field], (), f'oracle {field}')
    _require(oracle['wall_time_s'] >= 0, 'Negative oracle wall time')
    bindings = oracle['source_sha256']
    _require(isinstance(bindings, dict) and bool(bindings)
             and 'oracle.py' in {Path(s).name for s in bindings}, 'Missing oracle LP source bindings')
    for source, digest in bindings.items():
        _require(_digest(digest) and source in manifest['files']
                 and manifest['files'][source]['sha256'] == digest, 'Oracle source hash mismatch')
        _frozen_bytes(manifest, source)
    return dict(kind='empirical_clairvoyant_reference', label='Empirical clairvoyant reference',
        value=oracle['mean_cost'], independent_solves=1, provenance=deepcopy(oracle),
        interpretation='One fresh joint LP fitted to the common 10,000-sample test; '
                       'not deployable-policy OOS performance or a population lower bound. '
                       'The same scalar is shown in all three panels; no training-seed statistics.')


def build_plot_data(manifest, rows, oracle) -> dict:
    """Require all 270 green + 540 SCS fresh rows and 30 frozen validation choices.

    Async arrival order is immaterial: columns are seeds 0..9. Failures, gaps,
    nonfinite results, changed inputs, and cached neural rows are fatal. Historical
    SCS convergence is NOT a replay acceptance criterion. Cached SRO-FA/kNN are
    descriptive; absent historical strict certificates are never inferred.
    """
    _require(manifest['status'] == 'frozen', 'Manifest must be frozen')
    _require('cached_green_N50' not in manifest, 'No cached green N50 is permitted')
    _require(not manifest.get('new_selection', False), 'New/test selection is not permitted')
    protocols, caches = {}, {}
    for n in SIZES:
        baseline = manifest['cached_baselines'][str(n)]
        _require('cache' in baseline and 'sro_source' in baseline, 'Unbound historical baseline sources')
        caches[n] = _frozen_json(manifest, baseline['cache'])
        protocols[n] = validate_cache_protocol(caches[n], n, manifest)
        for source in protocols[n]['dimension_evidence']['source_sha256']:
            _frozen_bytes(manifest, source)
    test = manifest['test']
    _require(test['n_test'] == manifest['n_test'] == 10000
             and test['seed'] == manifest['test_seed'] == 9999, 'Test protocol mismatch')
    _test_identity(test)
    reference = _oracle_reference(manifest, oracle)
    grids = {n: _grid(manifest['scs_radii'][str(n)], f'SCS N={n}', SCS_COUNTS[n]) for n in SIZES}
    _require(all(g == SCS_RADII for g in grids.values()), 'SCS saved radius grid mismatch')
    expected = {(method, n, rho, seed) for method in ('green', 'scs') for n in SIZES
                for rho in (GREEN_RADII if method == 'green' else grids[n]) for seed in SEEDS}
    specs, indexed = _index(manifest['jobs']), _index(rows)
    _require(set(specs) == set(indexed) == expected and len(indexed) == 810,
             'Incomplete/mismatched fixed policy grid: require 810 fresh policies')
    _require(manifest.get('job_count', 810) == 810, 'Manifest job count mismatch')
    verified, manifest_hashes = set(), set()
    for key, row in indexed.items():
        _require(row['status'] == 'completed', f'Failed/partial policy: {key}')
        _require(row.get('cached') is False, f'Policy must be a fresh replay: {key}')
        _require(all(k in row and row[k] == v for k, v in specs[key].items()),
                 f'Policy identity differs from manifest: {key}')
        ntrain = 4*row['Ntotal']//5 if row['method'] == 'green' else row['Ntotal']
        _require(row['Ntrain'] == ntrain, f'Training budget mismatch: {key}')
        _require(row.get('test') == test, f'Worker test identity mismatch: {key}')
        for field in ('checkpoint_sha256', 'state_sha256', 'pool_sha256', 'manifest_sha256'):
            _require(_digest(row.get(field)), f'Missing/invalid {field}: {key}')
        manifest_hashes.add(row['manifest_sha256'])
        inputs = row['input_files']
        _require(row['checkpoint'] in inputs and bool(inputs), f'Unbound checkpoint inputs: {key}')
        for source in inputs:
            if source not in verified:
                _frozen_bytes(manifest, source)
                verified.add(source)
        _require(manifest['files'][row['checkpoint']]['sha256'] == row['checkpoint_sha256'],
                 f'Checkpoint hash mismatch: {key}')
        sizes = row['sizes']
        _require(len(sizes) == ntrain and all(type(s) is int and s > 0 for s in sizes),
                 f'Invalid returned-checkpoint pool sizes: {key}')
        maxima = _values(row.get('per_center_maxima'), (ntrain,), 'returned-checkpoint pool maxima')
        _require(row['finite_pool_mean'] == float(maxima.mean()), f'Finite-pool maximum mean mismatch: {key}')
        _values([row['mean_oos'], row['finite_pool_mean']], (2,), 'replay costs')
        _require(row.get('solves_attempted') == row.get('solves_accepted') == 10000 + sum(sizes)
                 and row.get('status_counts') == {'2': row['solves_accepted']},
                 f'Incomplete strict solves: {key}')
    _require(len(manifest_hashes) == 1, 'Worker manifest hash mismatch')
    if 'manifest_sha256' in manifest:
        _require(manifest_hashes == {manifest['manifest_sha256']}, 'Manifest hash mismatch')

    panels = []
    for n in SIZES:
        baseline = manifest['cached_baselines'][str(n)]
        cache = caches[n]
        actual = sorted(float(r) for r in cache['sweep'] if 0 <= float(r) <= 1.5)
        _require(actual == grids[n], 'SCS grid differs from frozen cache')
        _require(cache['benchmarks']['saa_knn'] == baseline['saa_knn'], 'SAA-kNN differs from frozen cache')
        source = baseline['sro_fa']
        _require(_frozen_json(manifest, baseline['sro_source']) == source, 'SRO-FA differs from frozen source')
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
            oos=_aggregate(_values([[r['oos'] for r in g] for g in sro_rows], (len(radii), 10), 'SRO-FA OOS')),
            provenance=dict(source=baseline['sro_source'], cached=True, strict_test_status='not_available',
                current_evaluation_time_s=None, per_seed=deepcopy(sro_rows),
                interpretation='Historical descriptive test reference, not fresh strict evaluation.'))
        knn = dict(method='saa_knn', label='SAA-kNN', Ntotal=n, Ntrain=n, seeds=SEEDS.copy(),
            oos=_aggregate(_values(baseline['saa_knn']['oos'], (10,), 'SAA-kNN OOS')),
            provenance=dict(source=baseline['cache'], cached=True, strict_test_status='not_available',
                current_evaluation_time_s=None, original=deepcopy(baseline['saa_knn']),
                interpretation='Historical descriptive test reference; saved validation choice of k.'))
        panels.append(dict(Ntotal=n, label=f'N = {n}', training_counts=dict(green=4*n//5, scs=n),
            cache_protocol=protocols[n],
            green=_neural(indexed, 'green', n, GREEN_RADII.copy()), scs=_neural(indexed, 'scs', n, grids[n]),
            sro_fa=sro, saa_knn=knn, horizontal=[deepcopy(reference)],
            historical_saa_reference=dict(source=baseline['cache'], metadata=deepcopy(cache['meta']),
                original=deepcopy(cache['benchmarks'].get('saa_lb')),
                interpretation='Raw historical cache only; not filled, aggregated, or plotted.')))

    winners = manifest.get('saved_green_selections', [])
    _require(len(winners) == 30 and {(w['Ntotal'], w['seed']) for w in winners}
             == {(n, s) for n in SIZES for s in SEEDS}, 'Incomplete saved validation selections')
    selected = []
    for winner in sorted(winners, key=lambda w: (w['Ntotal'], w['seed'])):
        _require(winner.get('method') == 'green_validation' and winner.get('status') == 'completed',
                 'Only saved completed validation selections are permitted')
        key = ('green', winner['Ntotal'], float(winner['radius']), winner['seed'])
        _require(key in indexed, 'Saved validation selection outside fixed grid')
        row = indexed[key]
        for field, target in (('checkpoint', 'checkpoint'), ('checkpoint_sha256', 'checkpoint_sha256'),
                              ('state_sha256', 'state_sha256'), ('candidate_id', 'selected_candidate_id'),
                              ('stopping_candidate_id', 'stopping_candidate_id')):
            _require(field in winner and winner[field] == row[target], 'Saved validation checkpoint mismatch')
        selected.append(dict(selection=deepcopy(winner), mean_oos=row['mean_oos'],
                             finite_pool_mean=row['finite_pool_mean']))
    return dict(schema_version=1, status='completed', panels=panels, seeds=SEEDS.copy(),
        scs_radii={str(n): grids[n] for n in SIZES}, test=deepcopy(test),
        band_definition='Pointwise 20th/80th percentiles across ten training seeds; not confidence intervals.',
        is_definition='Mean of per-center maxima on the returned checkpoint own saved finite MP pool; '
                      'not the full adversarial supremum.',
        caption='All 810 policies (270 green + 540 SCS) are fresh strict replays, including N50. '
                'Training budget: green 40/160/640 versus SCS 50/200/800. '
                'Fixed saved radius nodes only; no interpolation/resampling or test-based selection; '
                'connecting line segments are visual only. Test seed 9999 is historically known, not a new holdout. '
                'SCS saved pool coordinates are unchanged; historical residuals up to 7.91e-6 are '
                'observational, not box-certified or repaired. Three nonconverged historical N50 '
                'checkpoints are retained. SRO-FA/SAA-kNN are cached historical descriptive comparators; '
                'strict historical test status is unavailable. The gray Empirical clairvoyant reference '
                'is one fresh joint LP fitted to the common 10,000-sample test, repeated across all '
                'three panels, not independent OOS performance or a population lower bound.',
        footnote='Training: green 40/160/640; SCS 50/200/800. Bands: seed 20–80%, not CI. '
                 'Historical test seed: 9999.\n'
                 'Unchanged SCS pools: residuals ≤7.91e−6 observational, not certified; '
                 '3 nonconverged N50 checkpoints retained. Cached baselines: descriptive.\n'
                 'Gray: one common test-fitted clairvoyant LP; not deployable OOS or a population bound.',
        provenance=dict(experiment=manifest.get('experiment'), files=deepcopy(manifest['files']),
            manifest_sha256=next(iter(manifest_hashes)), workers=manifest.get('workers'),
            threads=manifest.get('threads'), runtime_note='Raw saved timings retained where available. '
            'Missing timings are unavailable; rerun required. Cache loading time is not model runtime.'),
        validation_selected_oos=selected,
        selection_interpretation='Only already-frozen validation selections are reported; no new selection.')


def _validate_plot_data(data):
    _require(data['status'] == 'completed' and data['seeds'] == SEEDS, 'Incomplete plot/seed order')
    _require([p['Ntotal'] for p in data['panels']] == list(SIZES), 'Missing/reordered panels')
    _test_identity(data['test'])
    horizontal = data['panels'][0]['horizontal']
    _require(len(horizontal) == 1, 'Require one common oracle reference')
    expected_reference = _oracle_reference(
        dict(test=data['test'], files=data['provenance']['files']), horizontal[0]['provenance'])
    for panel in data['panels']:
        n = panel['Ntotal']
        for method in ('green', 'scs', 'sro_fa', 'saa_knn'):
            curve = panel[method]
            _require(curve['seeds'] == SEEDS, 'Curve seed order mismatch')
            if method == 'saa_knn':
                shape = (10,)
            else:
                radii = _grid(curve['radius'], method, zero=method != 'sro_fa')
                if method == 'green':
                    _require(radii == GREEN_RADII, 'Green grid mismatch')
                if method == 'scs':
                    _require(radii == data['scs_radii'][str(n)] == SCS_RADII
                             and len(radii) == SCS_COUNTS[n], 'SCS grid mismatch')
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
                        _require(provenance['cached'] is False, 'No cached neural results permitted')
                        _require(all([r['seed'] for r in g] == SEEDS for g in groups),
                                 'Provenance seed order mismatch')
                        _require(all(r['radius'] == rho and r['status'] == 'completed'
                                     for rho, g in zip(curve['radius'], groups) for r in g),
                                 'Provenance radius/status mismatch')
                        field = 'finite_pool_mean' if name == 'finite_pool' else 'mean_oos'
                        if name == 'finite_pool':
                            for g in groups:
                                for row in g:
                                    maxima = _values(row['per_center_maxima'], (curve['Ntrain'],), 'pool maxima')
                                    _require(row[field] == float(maxima.mean()), 'Finite-pool maximum mean mismatch')
                    else:
                        field = 'oos'
                    expected = [[r[field] for r in g] for g in groups]
                _require(stats['per_seed'] == expected, 'Per-seed values/order differ from provenance')
        _require(panel['horizontal'] == [expected_reference], 'Common oracle reference mismatch')


def render(data, output: Path):
    """Validate before writing; same three-panel R3 layout and actual saved nodes."""
    _validate_plot_data(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    output = Path(output)
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'dejavuserif', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
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
                    line = ax.axhline(h['value'], color='#555555', ls=':', lw=1.25, label=h['label'])
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
            fig.suptitle(r'Fixed-radius policy comparison · $R=10$', fontsize=14, y=.98)
            fig.legend(list(handles.values()), list(handles), loc='lower center',
                       bbox_to_anchor=(.5, .09), ncol=4, frameon=False, fontsize=9,
                       handlelength=2.7, columnspacing=2.)
            fig.text(.5, .022, data['footnote'], ha='center', fontsize=8, color='#555555')
            fig.subplots_adjust(left=.065, right=.988, top=.865, bottom=.31, wspace=.24)
            output.mkdir(parents=True, exist_ok=True)
            fig.savefig(output / 'comparison_R10.png', dpi=300)
            fig.savefig(output / 'comparison_R10.pdf')
        finally:
            plt.close(fig)