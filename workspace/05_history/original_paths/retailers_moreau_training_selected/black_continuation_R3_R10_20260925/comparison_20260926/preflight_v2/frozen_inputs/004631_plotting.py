"""Display-only black overlays; no evaluation, selection, or final publication.

Parent contract for build_overlay(base_data, rows, R, bindings):
* rows is the complete R-specific cohort (270 or 300 freshly evaluated rows).
* bindings contains input_sha256/source_sha256 (absolute path -> SHA256),
  base_sha256 (str(R) -> canonical JSON SHA256), renderers (str(R) ->
  {path, sha256}), test (str(R) -> common test metadata), jobs (all 570 frozen
  specs, keyed by id), and results (all 570 journal-verified result references).
* Every job contains IDENTITY_FIELDS and SELECTION_FIELDS below. The parent
  establishes earliest-minimum-mean-AP selection on the entire original plus
  continued prefix, including terminal 40, BEFORE any test evaluation.
* Every result reference contains result_sha256, status='completed',
  journal_verified=True, and raw_proofs (absolute path -> SHA256). These are
  parent attestations: this module does not independently verify raw solver
  journals or pretend that a hash alone certifies physical feasibility.
* Rows contain the frozen job fields, result_sha256, mean_oos,
  finite_pool_mean, per_center_maxima, sizes, status='completed', cached=False,
  test (same array identities), solves_attempted/solves_accepted/
  solves_physical_accepted, status_counts={'2': expected}, wall_time_s,
  phases (current invocation only), and source_prefix_timings (separate).
  All extra row evidence, timestamps, concurrency and timings are preserved.

The global journal barrier must attest strict acceptance of ALL 570 results,
not merely job termination. Training nonconvergence is separate from replay
completion. render writes only into a caller-owned empty staging directory;
the parent owns exclusive creation, atomic publication, and base snapshots.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/data/home/aviyac/retailers_moreau_training_selected')
BASE_PATHS = {
    3: ROOT / 'green_R3_N200_N800_20260923/comparison_20260924/results/'
              '2026-09-24T091945.582854+0000/plot_data.json',
    10: ROOT / 'green_R10_N50_N200_N800_20260924/comparison_20260925/'
               'production_v1/published/plot_data.json',
}
RENDERER_PATHS = {
    3: ROOT / 'green_R3_N200_N800_20260923/comparison_20260924/plot_comparison.py',
    10: ROOT / 'green_R10_N50_N200_N800_20260924/comparison_20260925/plot_comparison.py',
}
SIZES = (50, 200, 800)
SEEDS = list(range(10))
RADII = {3: [0., .001, .003, .01, .03, .2, .4, .7, 1.5],
         10: [0., .001, .003, .01, .03, .2, .4, .7, 1.5, 2.]}
IDENTITY_FIELDS = ('id', 'R', 'Ntotal', 'Ntrain', 'seed', 'radius',
                   'checkpoint_sha256', 'state_sha256', 'pool_sha256', 'pool_file_sha256')
SELECTION_FIELDS = ('converged', 'terminal_status', 'selected_candidate_id',
                    'stopping_candidate_id')
SELECTION_RULE = 'earliest_minimum_mean_AP_full_original_plus_continued_prefix_including_terminal'
CAPTION_SUFFIX = (
    ' Black addition: all 570 fixed policies are freshly replayed on the same '
    '10,000 scenarios (seed 9999); no test-based policy/radius selection. '
    'Selection is the earliest minimum mean AP incumbent over the full original '
    'plus continued prefix, including terminal 40 for capped runs. All eleven '
    'nonconverged policies are retained (ten new caps plus cached-source '
    'R3/N50/seed9/radius0.7); cached-source training is not cached evaluation. '
    'Black OOS bands are pointwise seed 20/80 percentiles, not confidence intervals; '
    'dashed light-gray IS is the mean of per-center maxima on each selected '
    'checkpoint\'s own finite pool, not the full adversarial supremum. '
    'Original caption statements and timings describe the original figure, '
    'not this invocation; all original comparators and their qualifications '
    'are unchanged. Black alone extends to radius 2 for R10; no SRO-FA value '
    'at 2 is supplied. Current replay and persisted source-prefix timings '
    'remain separate; missing historical timings are unavailable, requiring rerun.'
)
FOOTNOTE_SUFFIX = (
    '\nBlack: fresh 10k replay; all 11 capped policies retained across R3/R10. '
    'Dashed gray: own-pool IS. Original comparators unchanged.'
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sha256(value):
    """Parent uses this for base snapshots, NOT hashes of raw result bytes."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _hash_map(mapping, name):
    _require(isinstance(mapping, dict) and bool(mapping), f'Missing {name}')
    _require(all(isinstance(p, str) and Path(p).is_absolute() and _digest(h)
                 for p, h in mapping.items()), f'Invalid {name}')


def _finite(value, name):
    _require(type(value) in (int, float) and np.isfinite(value), f'Nonfinite/invalid {name}')


def _key(row):
    for field, allowed in (('R', (3, 10)), ('Ntotal', SIZES), ('seed', SEEDS)):
        _require(type(row[field]) is int and row[field] in allowed, f'Invalid {field}')
    _finite(row['radius'], 'radius')
    _require(row['radius'] in RADII[row['R']], 'Unexpected radius')
    _require(type(row['Ntrain']) is int and row['Ntrain'] == 4*row['Ntotal']//5,
             'Wrong training budget')
    return row['R'], row['Ntotal'], row['radius'], row['seed']


def _expected(R):
    return {(R, n, rho, seed) for n in SIZES for rho in RADII[R] for seed in SEEDS}


def _test_identity(test, R):
    _require(test['n_test'] == 10000 and test['seed'] == 9999, 'Wrong test protocol')
    identity = {}
    for name in ('xi1', 'xi2'):
        a = test['arrays'][name]
        _require(a['shape'] == [10000, R] and a['dtype'] == 'float64'
                 and _digest(a['array_sha256']), 'Wrong test array identity')
        identity[name] = a['array_sha256']
    return identity


def _validate_bindings(bindings):
    _hash_map(bindings['input_sha256'], 'input hashes')
    _hash_map(bindings['source_sha256'], 'source hashes')
    _require(bindings['selection_rule'] == SELECTION_RULE, 'Unapproved selection protocol')
    jobs, results = bindings['jobs'], bindings['results']
    _require(isinstance(jobs, dict) and len(jobs) == 570
             and set(results) == set(jobs), 'Global 570-result barrier incomplete')
    keys, capped = set(), []
    for identifier, job in jobs.items():
        _require(isinstance(identifier, str) and bool(identifier) and job['id'] == identifier,
                 'Job ID mismatch')
        key = _key(job)
        _require(key not in keys, 'Duplicate global job')
        keys.add(key)
        for field in IDENTITY_FIELDS[6:]:
            _require(_digest(job[field]), f'Invalid {field}')
        _require(type(job['converged']) is bool, 'Missing training convergence flag')
        terminal = job['terminal_status']
        _require(terminal in ('iteration_cap', 'converged', 'separation_satisfied')
                 and job['converged'] == (terminal != 'iteration_cap'), 'Inconsistent terminal status')
        selected, stopping = job['selected_candidate_id'], job['stopping_candidate_id']
        _require(type(selected) is int and type(stopping) is int
                 and 0 <= selected <= stopping <= 40, 'Invalid selected/terminal candidate')
        if not job['converged']:
            _require(stopping == 40, 'Capped prefix must include terminal 40')
            capped.append({k: deepcopy(job[k]) for k in IDENTITY_FIELDS + SELECTION_FIELDS})
        ref = results[identifier]
        _require(ref['status'] == 'completed' and ref['journal_verified'] is True
                 and _digest(ref['result_sha256']), 'Global strict-result barrier incomplete')
        _hash_map(ref['raw_proofs'], 'raw strict-solve proof references')
    _require(keys == _expected(3) | _expected(10), 'Global grid mismatch')
    _require(len(capped) == 11, 'Expected all eleven nonconverged policies')
    historical = [r for r in capped if r['R'] == 3 and r['Ntotal'] == 50]
    _require(len(historical) == 1 and historical[0]['seed'] == 9
             and historical[0]['radius'] == .7, 'Historical black cap identity mismatch')
    return sorted(capped, key=_key)


def _load_original(R, bindings, overlay=False):
    """Execute a private hash-bound renderer copy; never patch/import live modules.

    Replacements are exact and presentation-only; original validators and
    method-specific grids stay intact. Hash authority belongs to the parent.
    """
    spec = bindings['renderers'][str(R)]
    path = Path(spec['path'])
    _require(path == RENDERER_PATHS[R] and _digest(spec['sha256'])
             and bindings['source_sha256'].get(str(path)) == spec['sha256'],
             'Unbound original renderer')
    payload = path.read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == spec['sha256'], 'Renderer source changed')
    source = payload.decode('utf-8')
    if overlay:
        replacements = [
            ("                knn = panel['saa_knn']['oos']",
             "                _draw_black(ax, panel['black'], handles)\n"
             "                knn = panel['saa_knn']['oos']"),
            (f"'comparison_R{R}.png'", f"'comparison_R{R}_with_black.png'"),
            (f"'comparison_R{R}.pdf'", f"'comparison_R{R}_with_black.pdf'"),
            # One extra explanatory footnote line; retain the same plot geometry.
            ("fig.text(.5, .022,", "fig.text(.5, .012,"),
        ]
        if R == 10:
            replacements.extend([
                ('ax.set_xlim(0, 1.5)', 'ax.set_xlim(0, 2.)'),
                ('ax.set_xticks(TICKS)', "ax.set_xticks(panel['black']['radius'])"),
            ])
        for old, new in replacements:
            _require(source.count(old) == 1, f'Renderer presentation anchor changed: {old}')
            source = source.replace(old, new, 1)
    namespace = {'__name__': f'_private_black_plot_R{R}', '__file__': str(path),
                 '_draw_black': _draw_black}
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace


def _validate_row(row, R, bindings):
    key = _key(row)
    _require(row['R'] == R and row['id'] in bindings['jobs'], 'Wrong cohort/job')
    job = bindings['jobs'][row['id']]
    _require(all(row[k] == job[k] for k in IDENTITY_FIELDS + SELECTION_FIELDS),
             'Result differs from frozen job selection/input')
    _require(type(row['converged']) is bool
             and type(row['selected_candidate_id']) is int
             and type(row['stopping_candidate_id']) is int, 'Invalid selection field types')
    _require(row['status'] == 'completed' and row['cached'] is False,
             'Require fresh completed replay, including cached training sources')
    _require(_test_identity(row['test'], R) == _test_identity(bindings['test'][str(R)], R),
             'Replay uses different test arrays')
    ref = bindings['results'][row['id']]
    _require(row['result_sha256'] == ref['result_sha256'], 'Unbound raw result hash')
    sizes = row['sizes']
    _require(len(sizes) == row['Ntrain'] and all(type(s) is int and s > 0 for s in sizes),
             'Invalid own-pool sizes')
    maxima = np.asarray(row['per_center_maxima'], dtype=float)
    _require(maxima.shape == (row['Ntrain'],) and np.isfinite(maxima).all(),
             'Invalid own-pool maxima')
    for field in ('mean_oos', 'finite_pool_mean', 'wall_time_s'):
        _finite(row[field], field)
    _require(row['finite_pool_mean'] == float(maxima.mean()), 'Own-pool mean mismatch')
    expected = 10000 + sum(sizes)
    for field in ('solves_attempted', 'solves_accepted', 'solves_physical_accepted'):
        _require(type(row[field]) is int and row[field] == expected,
                 'Incomplete strict/physical QP acceptance')
    _require(row['status_counts'] == {'2': expected}
             and type(row['status_counts']['2']) is int, 'Non-OPTIMAL QP status')
    _require(row['wall_time_s'] >= 0 and isinstance(row['phases'], dict)
             and bool(row['phases']) and isinstance(row['source_prefix_timings'], dict)
             and bool(row['source_prefix_timings']), 'Missing separate timing evidence')
    return key


def build_overlay(base_data, rows, R, bindings) -> dict:
    """Return an independent JSON-safe overlay, preserving every original field.

    Only caption/footnote are appended; panel.black and black_overlay are new.
    Existing provenance, selection arrays, reference scalars, cache flags, method
    grids and training_counts are never rewritten. No training selection occurs.
    """
    _require(type(R) is int and R in RADII, 'Unsupported R')
    capped = _validate_bindings(bindings)
    _require('black_overlay' not in base_data and all('black' not in p for p in base_data['panels']),
             'Input already contains a black overlay')
    _require(canonical_sha256(base_data) == bindings['base_sha256'][str(R)], 'Base snapshot mismatch')
    _require(_test_identity(base_data['test'], R) == _test_identity(bindings['test'][str(R)], R),
             'Base/test binding mismatch')
    _load_original(R, bindings)['_validate_plot_data'](base_data)
    indexed = {}
    for row in rows:
        key = _validate_row(row, R, bindings)
        _require(key not in indexed, 'Duplicate black policy')
        indexed[key] = row
    _require(set(indexed) == _expected(R), 'Incomplete black fixed grid (270/300 required)')
    data = deepcopy(base_data)
    for panel in data['panels']:
        n = panel['Ntotal']
        groups = [[indexed[R, n, rho, seed] for seed in SEEDS] for rho in RADII[R]]
        def aggregate(field):
            a = np.asarray([[r[field] for r in g] for g in groups], dtype=float)
            return dict(per_seed=a.tolist(), mean=a.mean(axis=-1).tolist(),
                        p20=np.percentile(a, 20, axis=-1).tolist(),
                        p80=np.percentile(a, 80, axis=-1).tolist())
        panel['black'] = dict(method='black', label='Black Moreau', Ntotal=n,
            Ntrain=4*n//5, seeds=SEEDS.copy(), radius=RADII[R].copy(),
            oos=aggregate('mean_oos'), finite_pool=aggregate('finite_pool_mean'),
            provenance=dict(cached=False, evaluation='fresh strict replay',
                current_evaluation_time_s=[[r['wall_time_s'] for r in g] for g in groups],
                per_seed=deepcopy(groups)))
    data['caption'] += CAPTION_SUFFIX
    data['footnote'] += FOOTNOTE_SUFFIX
    data['black_overlay'] = dict(R=R, schema_version=1, bindings=deepcopy(bindings),
        black_nonconverged=capped, cohort_nonconverged=[r for r in capped if r['R'] == R],
        global_policy_count=570, cohort_policy_count=len(indexed),
        runtime_note='Rows retain current wall_time_s and phases separately from source_prefix_timings. '
                     'No concurrent sums are reported as sweep elapsed; no cache load is called training.',
        proof_scope='Parent journal attestation plus exact references; no numerical proof replay here.')
    return data


def _draw_black(ax, curve, handles):
    x, stats = curve['radius'], curve['oos']
    line, = ax.plot(x, stats['mean'], color='#000000', lw=1.9, marker='o', ms=2.5,
                    label='Black Moreau OOS', zorder=5)
    handles.setdefault('Black Moreau OOS', line)
    ax.fill_between(x, stats['p20'], stats['p80'], color='#000000', alpha=.12, lw=0)
    line, = ax.plot(x, curve['finite_pool']['mean'], '--', color='#999999', lw=1.7,
                    label='Black Moreau IS', zorder=3)
    handles.setdefault('Black Moreau IS', line)


def render(data, output: Path):
    """Revalidate then write PNG/PDF and JSON into existing parent staging only.

    Never overwrite artifacts. This is not atomic publication: the parent must
    discard staging on any error and publish only after BOTH R values succeed.
    """
    meta = data['black_overlay']
    R, bindings = meta['R'], meta['bindings']
    base = deepcopy(data)
    del base['black_overlay']
    rows = []
    for panel in base['panels']:
        black = panel.pop('black')
        rows.extend(r for group in black['provenance']['per_seed'] for r in group)
    _require(base['caption'].endswith(CAPTION_SUFFIX)
             and base['footnote'].endswith(FOOTNOTE_SUFFIX), 'Overlay explanation changed')
    base['caption'] = base['caption'][:-len(CAPTION_SUFFIX)]
    base['footnote'] = base['footnote'][:-len(FOOTNOTE_SUFFIX)]
    _require(build_overlay(base, rows, R, bindings) == data, 'Overlay data/evidence changed')
    output = Path(output)
    _require(output.is_dir(), 'Parent must create staging directory')
    paths = [output / f'comparison_R{R}_with_black.{ext}' for ext in ('png', 'pdf')]
    paths.append(output / f'plot_data_R{R}.json')
    _require(not any(p.exists() for p in paths), 'Refusing to overwrite plot artifacts')
    _load_original(R, bindings, overlay=True)['render'](data, output)
    with paths[-1].open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
    return paths