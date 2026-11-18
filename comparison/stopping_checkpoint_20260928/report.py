"""STOP-only publication; no selection, training, evaluation, or archive writes.

publish_data(manifest, rows, output: Path) -> list[Path] requires an existing,
empty parent-owned staging directory. The parent verifies selection inputs,
sources and ALL solver journals first, then adds journal_verified=True and
result_path/result_sha256 to each fresh worker row. No manifest self-hash is
required. Raw result bytes and row identities/values are checked here. The
parent owns atomic publication, pipeline elapsed time and the final seal.

JSON artifacts have same-stem .sha256 sidecars; RESULTS.md is deliberately not
sealed here because the parent appends whole-pipeline elapsed time. Runtime
records retain parent-provided concurrency/timestamps, and distinguish persisted
phase sums from measured invocation totals. Missing totals are never inferred.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np


RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
SIZES = (50, 200, 800)
SEEDS = list(range(10))
IDENTITY = ('id', 'method', 'R', 'Ntotal', 'Ntrain', 'seed', 'radius',
            'checkpoint_sha256', 'state_sha256', 'pool_sha256', 'pool_file_sha256',
            'selected_candidate_id', 'stopping_candidate_id', 'converged', 'terminal_status')
VALUES = ('status', 'cached', 'mean_oos', 'finite_pool_mean', 'per_center_maxima',
          'sizes', 'test', 'wall_time_s', 'phases')
CAPTION = (
    'Both GREEN (540) and BLACK (570) use their actual stopping checkpoints, '
    'not minimum-AP checkpoints. All 11 nonconverged BLACK iteration-cap policies '
    'are retained at stop 40; successful replay does not imply training convergence '
    'or globally certified robust optimality. Both are freshly evaluated on the '
    'same 10,000 test scenarios per R (seed 9999). This historically known test '
    'is descriptive, not a new holdout. No new validation, radius selection, '
    'ranking test, or test-based policy selection is performed. Fixed saved nodes '
    'only; connecting lines are visual. Bands are pointwise seed 20/80 percentiles, '
    'not confidence intervals. Dashed neural IS curves are means of per-center '
    'maxima on each checkpoint\'s own finite pool, not a full adversarial supremum. '
    'SCS, SRO-FA, SAA-kNN and all original reference levels and qualifications '
    'are unchanged historical comparators, not fresh results. SCS pools are '
    'unrepaired and not newly box-certified; historical test solver status is '
    'not newly certified. Gray references retain their historical sampling/test-fit '
    'qualifications and are not newly certified population bounds. Only BLACK '
    'extends to radius 2 for R10. Current evaluation and persisted training '
    'timings are separate; unavailable historical totals require a rerun.'
)
FOOTNOTE = (
    'GREEN/BLACK: STOP checkpoints, fresh same-10k replay; all 11 BLACK caps retained. '
    'Bands: seed 20–80%, not CI. Dashed: own-pool IS.\n'
    'Training: neural 40/160/640; SCS 50/200/800. Original comparators/references '
    'unchanged; historical test is descriptive, not a new holdout.'
)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _read_bound(path, digest):
    payload = Path(path).read_bytes()
    _require(_digest(digest) and hashlib.sha256(payload).hexdigest() == digest,
             f'Hash mismatch: {path}')
    return json.loads(payload)


def _finite(value, name, *, duration=False):
    _require(type(value) in (int, float) and np.isfinite(value)
             and (not duration or value >= 0), f'Invalid {name}')
    return value


def _grid(method, r):
    return RADII + ((2.,) if method == 'black' and r == 10 else ())


def _key(job):
    return tuple(job[k] for k in ('method', 'R', 'Ntotal', 'seed', 'radius'))


def _test_identity(test, r):
    _require(test['n_test'] == 10000 and test['seed'] == 9999, 'Wrong test protocol')
    identity = []
    for name in ('xi1', 'xi2'):
        a = test['arrays'][name]
        _require(a['shape'] == [10000, r] and a['dtype'] == 'float64'
                 and _digest(a['array_sha256']) and _digest(a['file_sha256']), 'Wrong test arrays')
        identity.append((a['array_sha256'], a['file_sha256']))
    return identity


def _validated(manifest, rows):
    jobs = manifest['jobs']
    expected = {(m, r, n, s, rho) for m in ('green', 'black') for r in (3, 10)
                for n in SIZES for s in SEEDS for rho in _grid(m, r)}
    _require(len(jobs) == len(rows) == 1110 and len({j['id'] for j in jobs}) == 1110
             and {_key(j) for j in jobs} == expected, 'Incomplete/duplicate manifest cohort')
    _require(manifest.get('return_rule', 'stopping_checkpoint') == 'stopping_checkpoint'
             and manifest.get('selection_rule', 'stopping_checkpoint') == 'stopping_checkpoint',
             'Not a STOP manifest')
    indexed, caps = {}, []
    by_id = {j['id']: j for j in jobs}
    for job in jobs:
        for field in ('R', 'Ntotal', 'Ntrain', 'seed', 'selected_candidate_id', 'stopping_candidate_id'):
            _require(type(job[field]) is int, f'Invalid identity: {field}')
        stop = job['stopping_candidate_id']
        _require(job['Ntrain'] == 4 * job['Ntotal'] // 5 and
                 job['selected_candidate_id'] == stop and 0 <= stop <= 40, 'Changed selected stop')
        _require(type(job['converged']) is bool, 'Invalid convergence flag')
        status = job['terminal_status']
        if job['method'] == 'green':
            _require(job['converged'] and status == 'green_gap_satisfied', 'Invalid GREEN stop')
        else:
            _require((job['converged'] and status in ('converged', 'separation_satisfied')) or
                     (not job['converged'] and status == 'iteration_cap' and stop == 40),
                     'Invalid BLACK convergence/cap')
            if not job['converged']:
                caps.append({k: job[k] for k in IDENTITY})
        for field in IDENTITY:
            if field.endswith('_sha256'):
                _require(_digest(job[field]), f'Invalid {field}')
    _require(len(caps) == 11, 'Must retain all 11 BLACK caps')
    for row in rows:
        _require(row['id'] in by_id and _key(row) not in indexed, 'Unknown/duplicate result')
        job = by_id[row['id']]
        _require(all(row[k] == job[k] and type(row[k]) is type(job[k]) for k in IDENTITY),
                 'Result identity differs from selected stop')
        for field in set(row) & set(job):
            if field.endswith('_sha256') or field in ('checkpoint', 'pool_path', 'mp_count', 'return_rule'):
                _require(row[field] == job[field], f'Result input differs: {field}')
        _require(row['status'] == 'completed' and row['cached'] is False
                 and row['journal_verified'] is True, 'Require fresh completed journal-verified results')
        _require(_test_identity(row['test'], row['R']) ==
                 _test_identity(manifest['tests'][str(row['R'])], row['R']), 'Result test arrays differ')
        _require(row['sizes'] == job['sizes'] and len(row['sizes']) == job['Ntrain']
                 and all(type(s) is int and s > 0 for s in row['sizes']), 'Own-pool sizes differ')
        maxima = np.asarray(row['per_center_maxima'], dtype=float)
        _require(maxima.shape == (job['Ntrain'],) and np.isfinite(maxima).all(), 'Invalid own-pool maxima')
        for field in ('mean_oos', 'finite_pool_mean', 'wall_time_s'):
            _finite(row[field], field, duration=field == 'wall_time_s')
        _require(row['finite_pool_mean'] == float(maxima.mean()), 'Own-pool mean mismatch')
        _require(isinstance(row['phases'], dict) and bool(row['phases']), 'Missing evaluation phases')
        for phase in row['phases'].values():
            _require(phase['status'] == 'completed', 'Incomplete evaluation phase')
            _finite(phase['wall_time_s'], 'phase wall time', duration=True)
        raw = _read_bound(row['result_path'], row['result_sha256'])
        _require(all(raw[k] == row[k] for k in IDENTITY + VALUES), 'Raw result/row mismatch')
        # Preserve any additional raw timing, input and solver evidence supplied by parent.
        _require(all(raw[k] == row[k] for k in set(raw) & set(row)), 'Raw evidence/row mismatch')
        indexed[_key(row)] = row
    _require(set(indexed) == expected, 'Missing seed/radius/method result')
    return indexed, caps


def _stats(values):
    present = [_finite(v, 'runtime', duration=True) for v in values if v is not None]
    return dict(count=len(present), unavailable_count=len(values) - len(present),
                mean=float(np.mean(present)) if present else None,
                median=float(np.median(present)) if present else None,
                min=min(present) if present else None, max=max(present) if present else None)


def _runtime(manifest, indexed):
    records, nominal, grouped = [], {}, defaultdict(list)
    for job in manifest['jobs']:
        row, history = indexed[_key(job)], job['historical_training']
        nominal_key = (job['R'], job['Ntotal'], job['seed'])
        timing = history['nominal']['training_timing']
        _require(timing['status'] == 'completed', 'Incomplete persisted nominal training')
        _finite(timing['wall_time_s'], 'nominal training time', duration=True)
        _require(nominal_key not in nominal or nominal[nominal_key]['training_timing'] == timing,
                 'Nominal selected-arm timing differs across radii/methods')
        nominal[nominal_key] = dict(R=job['R'], Ntotal=job['Ntotal'], Ntrain=job['Ntrain'],
            seed=job['seed'], training_timing=deepcopy(timing),
            scope='persisted selected nominal arm, once per R/N/seed; not selection/CV or fresh training')
        historical = {k: deepcopy(v) for k, v in history.items() if k != 'nominal'}
        record = {k: deepcopy(job[k]) for k in IDENTITY}
        record.update(nominal_key=list(nominal_key), historical_training=historical,
            current_evaluation={k: deepcopy(v) for k, v in row.items()
                                if k not in IDENTITY + ('per_center_maxima', 'sizes', 'test',
                                    'input_sha256', 'historical_training', 'mean_oos', 'finite_pool_mean')})
        records.append(record)
        grouped[job['method'], job['R'], job['Ntotal'], job['radius'],
                'converged' if job['converged'] else 'capped'].append(record)
    aggregates = []
    for (method, r, n, rho, status), group in sorted(grouped.items()):
        histories = [g['historical_training'] for g in group]
        segments = {name: _stats([h[name]['active_phase_time_s'] for h in histories])
                    for name in ('green_through_stop', 'black_after_green_stop')}
        active = [h['green_through_stop']['active_phase_time_s'] +
                  h['black_after_green_stop']['active_phase_time_s'] for h in histories]
        aggregates.append(dict(method=method, R=r, Ntotal=n, radius=rho,
            training_status=status, count=len(group),
            evaluation_wall_time_s=_stats([g['current_evaluation']['wall_time_s'] for g in group]),
            persisted_training_total_wall_time_s=_stats([h['total_wall_time_s'] for h in histories]),
            persisted_training_active_phase_time_s=_stats(active), persisted_segments=segments))
    environments = []
    for row in indexed.values():
        if 'environment' in row and row['environment'] not in environments:
            environments.append(deepcopy(row['environment']))
    return dict(timing_unit='seconds', current_training_performed=False,
        scope='Per-policy wall time, NOT summed concurrent sweep elapsed. Persisted active phase sums '
              'exclude nominal and are NOT end-to-end wall time. Caps remain separate from converged runs.',
        unknown='Unavailable historical totals require rerun; never inferred from phase sums.',
        environment=deepcopy(manifest.get('environment', environments or 'unknown; see parent execution report')),
        whole_pipeline_elapsed_s=None, whole_pipeline_scope='parent appends after publisher returns',
        nominal_selected_arms=list(nominal.values()),
        nominal_aggregates=[dict(R=r, Ntotal=n, training_wall_time_s=_stats([
            record['training_timing']['wall_time_s'] for key, record in nominal.items()
            if key[:2] == (r, n)])) for r in (3, 10) for n in SIZES],
        aggregates=aggregates, records=records,
        sro_fa=dict(total_wall_time_s=None, partition_time_s=None, optimization_time_s=None,
                    evaluation_time_s=None, scope='historical timing unavailable; rerun required'),
        selection_cv=dict(wall_time_s=None, scope='no new validation/CV; historical timing unavailable'),
        nominal_evaluation=dict(wall_time_s=None, scope='not separately measured by this replay'))


def _curve(method, r, n, indexed):
    radii = list(_grid(method, r))
    groups = [[indexed[method, r, n, seed, rho] for seed in SEEDS] for rho in radii]
    def aggregate(field):
        a = np.asarray([[row[field] for row in group] for group in groups])
        return dict(per_seed=a.tolist(), mean=a.mean(axis=-1).tolist(),
                    p20=np.percentile(a, 20, axis=-1).tolist(),
                    p80=np.percentile(a, 80, axis=-1).tolist())
    return dict(method=method, label=f'{method.title()} Moreau STOP', Ntotal=n, Ntrain=4*n//5,
        seeds=SEEDS.copy(), radius=radii, oos=aggregate('mean_oos'), finite_pool=aggregate('finite_pool_mean'),
        provenance=dict(selection_rule='stopping_checkpoint', cached=False,
            proof_scope='Parent verified all solver journals; publisher binds raw results to STOP jobs.',
            historical_training_reference='runtime_report.json: records indexed by id',
            per_seed=[[{k: deepcopy(v) for k, v in row.items()
                        if k not in ('input_sha256', 'historical_training')} for row in group] for group in groups]))


def _plot_data(base, binding, manifest, r, indexed):
    _require(_test_identity(base['test'], r) == _test_identity(manifest['tests'][str(r)], r),
             'Base test arrays differ')
    _require([p['Ntotal'] for p in base['panels']] == list(SIZES), 'Wrong base panels')
    # Old selection/validation claims are historical only, never current provenance.
    retained = ('schema_version', 'seeds', 'scs_radii')
    data = {k: deepcopy(base[k]) for k in retained}
    historical = {k: deepcopy(v) for k, v in base.items()
                  if k not in (*retained, 'panels', 'black_overlay')}
    if 'black_overlay' in base:
        historical['black_overlay'] = dict(base_plot_data=deepcopy(binding), json_pointer='/black_overlay',
            scope='historical min-AP overlay ONLY; full immutable metadata retained in hash-bound base')
    data.update(status='completed', selection_rule='stopping_checkpoint', caption=CAPTION,
        test=deepcopy(manifest['tests'][str(r)]),
        footnote=FOOTNOTE, band_definition='pointwise seed 20/80 percentiles, not confidence intervals',
        is_definition='neural: mean of per-center maxima on the STOP checkpoint own finite pool',
        historical_base_metadata=dict(scope='NOT current results or current selection', **historical),
        provenance=dict(base_plot_data=deepcopy(binding), source_sha256=deepcopy(manifest['source_sha256']),
            tests=deepcopy(manifest['tests'][str(r)]), selection_rule='stopping_checkpoint',
            global_counts=dict(green=540, black=570, black_capped=11),
            global_robust_optimality_certified=False), panels=[])
    for panel in base['panels']:
        updated = {k: deepcopy(v) for k, v in panel.items() if k not in ('green', 'black')}
        for method in ('green', 'black'):
            updated[method] = _curve(method, r, panel['Ntotal'], indexed)
        _require({k: v for k, v in updated.items() if k not in ('green', 'black')} ==
                 {k: v for k, v in panel.items() if k not in ('green', 'black')}, 'Reference changed')
        data['panels'].append(updated)
    return data


def _render(data, r, output):
    """Dimension-generic display adaptation of the archived plot_comparison.render.

    Same colors, actual nodes, symlog, reference styles and three-panel geometry.
    Archived min-AP/validation validators do not describe this STOP publication;
    _validated and _plot_data establish its distinct contract before rendering.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    colors = dict(green='#00865a', green_is='#73b998', scs='#2465aa', scs_is='#80b2d8',
                  sro_fa='#c43d3d', saa_knn='#c99b08', black='#000000', black_is='#999999')
    ticks = [0., .001, .003, .01, .03, .1, .2, .4, .7, 1.5] + ([2.] if r == 10 else [])
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'dejavuserif', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(1, 3, figsize=(14.6, 5.4))
        try:
            handles = {}
            for ax, panel in zip(axes, data['panels']):
                for method in ('green', 'scs', 'sro_fa', 'black'):
                    curve = panel[method]
                    x, stats = curve['radius'], curve['oos']
                    label = curve['label'] + ' OOS'
                    line, = ax.plot(x, stats['mean'], color=colors[method], lw=1.9,
                                    marker='o', ms=2.5, label=label, zorder=5 if method == 'black' else 4)
                    handles.setdefault(label, line)
                    ax.fill_between(x, stats['p20'], stats['p80'], color=colors[method], alpha=.12, lw=0)
                    if method != 'sro_fa':
                        label = curve['label'] + ' IS'
                        line, = ax.plot(x, curve['finite_pool']['mean'], '--', color=colors[method+'_is'],
                                        lw=1.7, label=label, zorder=3)
                        handles.setdefault(label, line)
                knn = panel['saa_knn']['oos']
                line = ax.axhline(knn['mean'], color=colors['saa_knn'], lw=1.8, label='SAA-kNN OOS')
                ax.axhspan(knn['p20'], knn['p80'], color=colors['saa_knn'], alpha=.08, lw=0)
                handles.setdefault('SAA-kNN OOS', line)
                for h in panel['horizontal']:
                    conditional = h['kind'] == 'conditional'
                    line = ax.axhline(h['value'], color='#888888' if conditional else '#555555',
                                     ls='-.' if conditional else ':', lw=1.25, label=h['label'])
                    handles.setdefault(h['label'], line)
                ax.set_xscale('symlog', linthresh=.001, linscale=.8)
                ax.set_xlim(0, 2. if r == 10 else 1.5)
                ax.set_xticks(ticks)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.tick_params(axis='x', rotation=45, labelsize=8)
                ax.grid(axis='y', color='#dddddd', lw=.6, alpha=.8)
                ax.set_axisbelow(True)
                ax.set_title(panel['label'], fontsize=12, pad=10)
                ax.set_xlabel(r'Radius $\rho$')
            axes[0].set_ylabel('Cost')
            fig.suptitle(f'Fixed-radius STOP policy comparison · $R={r}$', fontsize=14, y=.98)
            fig.legend(list(handles.values()), list(handles), loc='lower center',
                       bbox_to_anchor=(.5, .09), ncol=4, frameon=False, fontsize=9,
                       handlelength=2.7, columnspacing=2.)
            fig.text(.5, .012, data['footnote'], ha='center', fontsize=8, color='#555555')
            fig.subplots_adjust(left=.065, right=.988, top=.865, bottom=.31, wspace=.24)
            for ext in ('png', 'pdf'):
                fig.savefig(output / f'comparison_R{r}_stopping.{ext}', dpi=300)
        finally:
            plt.close(fig)


def _fixed_costs(plots):
    rows = []
    for r, data in plots.items():
        for panel in data['panels']:
            green, black = panel['green'], panel['black']
            for i, rho in enumerate(black['radius']):
                g = green['oos']['mean'][i] if rho in green['radius'] else None
                b = black['oos']['mean'][i]
                rows.append(dict(R=r, Ntotal=panel['Ntotal'], radius=rho, black_oos=b, green_oos=g,
                    black_minus_green=b-g if g is not None else None,
                    green_own_pool_is=green['finite_pool']['mean'][i] if g is not None else None,
                    black_own_pool_is=black['finite_pool']['mean'][i],
                    paired_seed_count=10 if g is not None else 0))
    return rows


def _human(value):
    return 'unavailable' if value is None else f'{value:.3f} s ({value/60:.3f} min; {value/3600:.4f} h)'


def _markdown(caps, costs, runtime):
    lines = ['# Stopping-checkpoint comparison', '', CAPTION, '',
        'Counts: GREEN 540 converged; BLACK 559 converged + 11 capped. All 1,110 fresh replays completed.',
        'Cap identities: ' + ', '.join(c['id'] for c in caps) + '.', '',
        '## Fixed-radius matched costs', '',
        '| R | N | radius | GREEN OOS | BLACK OOS | BLACK − GREEN |',
        '|---|---|---|---|---|---|']
    for c in costs:
        fmt = lambda x: 'unavailable (BLACK only)' if x is None else f'{x:.6f}'
        lines.append(f"| {c['R']} | {c['Ntotal']} | {c['radius']:g} | {fmt(c['green_oos'])} | "
                     f"{fmt(c['black_oos'])} | {fmt(c['black_minus_green'])} |")
    lines += ['', '## Runtime', '', runtime['scope'], runtime['unknown'],
        'Each cell is count; mean (s/min/h); median / min / max (seconds). No concurrent times are summed.',
        'Raw phase/iteration timings, timestamps and supplied thread/worker counts are in runtime_report.json.',
        'Hardware/environment: ' + json.dumps(runtime['environment']) + '.', '',
        '| Method | R/N/radius/status | Evaluation | Persisted training total | Persisted active phases |',
        '|---|---|---|---|---|']
    def stats(s):
        values = ' / '.join('unavailable' if s[k] is None else f'{s[k]:.3f}' for k in ('median', 'min', 'max'))
        return f"{s['count']}; {_human(s['mean'])}; {values}; missing={s['unavailable_count']}"
    for a in runtime['nominal_aggregates']:
        lines.append(f"| Nominal selected arm | {a['R']}/{a['Ntotal']}/—/completed | not separately measured | "
                     f"{stats(a['training_wall_time_s'])} | — |")
    for a in runtime['aggregates']:
        label = f"{a['R']}/{a['Ntotal']}/{a['radius']:g}/{a['training_status']}"
        lines.append(f"| {a['method']} | {label} | {stats(a['evaluation_wall_time_s'])} | "
                     f"{stats(a['persisted_training_total_wall_time_s'])} | "
                     f"{stats(a['persisted_training_active_phase_time_s'])} |")
    lines += ['| SRO-FA | historical | unavailable | unavailable; rerun required | unavailable |',
              '| Validation/CV | no new selection | not run | historical unavailable | — |']
    lines += ['', 'No new training was performed. Nominal selected-arm timings are counted once per R/N/seed, '
              'not per radius. Unmeasured nominal evaluation and historical CV are not zero.',
              'Whole-pipeline elapsed time: pending parent completion (not a sum of policy wall times).', '']
    return '\n'.join(lines)


def publish_data(manifest: dict, rows: list[dict], output: Path) -> list[Path]:
    """Validate the full barrier, then write both figures/reports to empty staging.

    Raises on any failed binding, missing job, changed stop/cap or output conflict.
    No partial publication or atomic rename: discard staging on any exception.
    """
    output = Path(output)
    _require(output.is_dir() and not any(output.iterdir()), 'Parent must provide empty staging')
    archive = Path(__file__).resolve().parents[2] / 'archive'
    _require(not output.resolve().is_relative_to(archive), 'Cannot write into archive')
    indexed, caps = _validated(manifest, rows)
    plots = {}
    for r in (3, 10):
        binding = manifest['base_plot_data'][str(r)]
        base = _read_bound(binding['path'], binding['sha256'])
        plots[r] = _plot_data(base, binding, manifest, r, indexed)
    runtime, costs = _runtime(manifest, indexed), _fixed_costs(plots)
    paths = []
    def write_json(name, value):
        path = output / name
        payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
        with path.open('xb') as stream:
            stream.write(payload)
        seal = path.with_suffix('.sha256')
        seal.write_text(hashlib.sha256(payload).hexdigest() + '\n')
        paths.extend([path, seal])
    for r, data in plots.items():
        _render(data, r, output)
        paths.extend(output / f'comparison_R{r}_stopping.{ext}' for ext in ('png', 'pdf'))
        write_json(f'plot_data_R{r}.json', data)
    write_json('runtime_report.json', runtime)
    write_json('fixed_radius_comparison.json', costs)
    write_json('report.json', dict(experiment='stopping_checkpoint_20260928', status='completed',
        selection_rule='stopping_checkpoint', counts=dict(green=540, black=570, black_converged=559, black_capped=11),
        capped=caps, caption=CAPTION, base_plot_data=manifest['base_plot_data'],
        source_sha256=manifest['source_sha256'], tests=manifest['tests'],
        artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Publication only; parent appends pipeline elapsed time and owns final seal.'))
    results = output / 'RESULTS.md'
    with results.open('x') as stream:
        stream.write(_markdown(caps, costs, runtime))
    paths.append(results)
    return paths