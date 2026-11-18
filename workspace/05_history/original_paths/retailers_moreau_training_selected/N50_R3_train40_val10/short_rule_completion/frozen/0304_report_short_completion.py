"""Report completed, isolated mean-rule continuations; never train or evaluate.

Interface: build_report(output, completed_summary) -> reporting timing record.
The caller supplies status='completed', 31 full continuation worker records and
31 strict c.evaluate_job results in the documented continuations/evaluation
sections. It owns final campaign wall time and summary persistence. This module
does not mutate that summary, import the controller, or load model/test arrays.

Freeze before continuation/evaluation: u.OUTPUT/{plot_data.json,summary.json,
frozen/manifest.json}, u.TRAIN/summary.json and its 90 ccg summary_path inputs;
SCS_SELECTED/rho_*/seed_*_summary.json (10), SCS_NOMINAL/seed_*_summary.json
(10), and u.c.SWEEP/rho_*/seed_*_summary.json restricted to the 23 blue plotted
radii (230). Capture this module too. All consumed source paths/hashes and raw
timing evidence are persisted in runtime_comparison.json. No historical zero
total_time_s is treated as a measured runtime.
"""

from copy import deepcopy
from pathlib import Path
import time

import update_comparison as u
import numpy as np


SCS_SELECTED = Path('/data/home/aviyac/retailers_problem/R3/selected_timing_net16_v2_run2/N50')
SCS_NOMINAL = Path('/data/home/aviyac/retailers_problem/R3/N50/nominal_timing_net16_v2')
POSITIVE_RADII = (.001, .003, .01, .03, .2, .4, .7, 1.5)
SEEDS = tuple(range(10))
PHASE_SCOPE = (
    'Original input verification/loading + each candidate evaluation/AP + '
    'incoming-pool evaluation, MP call and returned-pool evaluation before each '
    'next candidate. Nonoverlapping outer phases only; excludes nominal training, '
    'selection/checkpoint publication, uninstrumented controller/persistence '
    'overhead and new resume loading. NOT fresh short-run end-to-end time.'
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seconds(value):
    _require(value is not None and np.isfinite(value) and value >= 0,
             f'Missing/invalid measured duration: {value!r}')
    return float(value)


def _index(rows, expected, description):
    result = {(j['seed'], j['radius']): j for j in rows}
    _require(len(rows) == len(result) and set(result) == expected,
             f'{description}: missing, duplicate or unexpected seed/radius')
    return result


def _phases(record, original=None):
    """Use cached candidate zero only once; never count nested MP/AP timings."""
    initial = record if original is None else original
    result = [dict(phase='input_verification_and_loading', origin='saved',
                   **initial['input_verification_and_loading'])]
    iterations = record['iterations']
    _require([a['candidate_id'] for a in iterations] == list(range(len(iterations))),
             'Unordered/incomplete candidate timing trajectory')
    _require(record['mp_count'] == len(iterations) - 1, 'MP count/trajectory mismatch')
    for i, candidate in enumerate(iterations):
        assessment = candidate
        if original is not None and i == 0:
            assessment = original['iterations'][0]
            for key in ('checkpoint_sha256', 'state_sha256'):
                _require(candidate[key] == assessment[key], 'Cached candidate-zero identity changed')
            for name in ('evaluation', 'ap'):
                _require(candidate[name]['wall_time_s'] == assessment[name]['wall_time_s'],
                         'Cached candidate-zero duration changed')
        for name in ('evaluation', 'ap'):
            result.append(dict(candidate=i, phase=name,
                               origin='current' if original is not None and i > 0 else 'saved',
                               **assessment[name]))
        if i < len(iterations) - 1:
            for name in ('incoming_pool_evaluation', 'mp_call', 'returned_pool_evaluation'):
                result.append(dict(candidate=i, phase=name,
                                   origin='current' if original is not None else 'saved',
                                   **candidate[name]))
    for p in result:
        _require(p['status'] == 'completed', 'Uncompleted phase in completed model')
        _seconds(p['wall_time_s'])
    return result


def _phase_totals(phases):
    return dict(
        phase_sum_s=sum(_seconds(p['wall_time_s']) for p in phases),
        ap_s=sum(p['wall_time_s'] for p in phases if p['phase'] == 'ap'),
        mp_s=sum(p['wall_time_s'] for p in phases if p['phase'] == 'mp_call'),
        saved_phase_s=sum(p['wall_time_s'] for p in phases if p['origin'] == 'saved'),
        current_phase_s=sum(p['wall_time_s'] for p in phases if p['origin'] == 'current'),
        ap_count=sum(p['phase'] == 'ap' for p in phases),
        mp_count=sum(p['phase'] == 'mp_call' for p in phases))


def _check_totals(record, totals):
    for reported, derived in (('measured_prefix_phase_sum_s', 'phase_sum_s'),
                              ('prefix_ap_s', 'ap_s'), ('prefix_mp_s', 'mp_s')):
        _require(np.isclose(_seconds(record[reported]), totals[derived], rtol=1e-12, atol=1e-9),
                 f'Persisted/derived timing mismatch: {reported}')
    _require(record['fresh_short_run_wall_s'] is None, 'Unexpected fresh short-run timing claim')


def _aggregates(rows, metrics, split_origin=False):
    """No failed/capped/cached-origin measurements silently join other statuses."""
    groups = {}
    for row in rows:
        key = (row['method'], row['radius'], row['status'], row['terminal_status'],
               row['origin'] if split_origin else 'all')
        groups.setdefault(key, []).append(row)
    result = []
    for (method, radius, status, terminal, origin), cohort in sorted(groups.items()):
        entry = dict(method=method, radius=radius, status=status,
                     terminal_status=terminal, origin=origin, n=len(cohort),
                     seeds=sorted(j['seed'] for j in cohort), metrics={},
                     origin_counts={o: sum(j['origin'] == o for j in cohort)
                                    for o in sorted({j['origin'] for j in cohort})})
        for metric in metrics:
            values = [j[metric] for j in cohort if j.get(metric) is not None]
            entry['metrics'][metric] = dict(
                unavailable_count=len(cohort) - len(values),
                statistics=u.c.timing_stats([_seconds(v) for v in values]) if values else None)
        result.append(entry)
    return result


def _statistics_cell(entry):
    stats = entry['statistics']
    if stats is None:
        return 'unavailable (rerun required)'
    return ' / '.join(f'{stats[k]:.6f}' for k in ('mean', 'median', 'min', 'max'))


def _runtime_table(lines, title, aggregates, metrics):
    lines += ['', f'## {title}', '',
              '| Method | Radius | Completion / terminal status | Origin | n | Metric | Mean / median / min / max (seconds unless count) | Mean minutes |',
              '|---|---:|---|---|---:|---|---|---:|']
    for a in aggregates:
        for metric in metrics:
            entry = a['metrics'][metric]
            stats = entry['statistics']
            minutes = f'{stats["mean"] / 60:.6f}' if stats and metric.endswith('_s') else '—'
            lines.append(f'| {a["method"]} | {a["radius"]:g} | {a["status"]} / '
                         f'{a["terminal_status"]} | {a["origin"]} | {a["n"]} | '
                         f'{metric} | {_statistics_cell(entry)} | {minutes} |')


def _render(output, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(11, 8))
    try:
        colors = ['#2166ac', '#762a83', '#111111', '#009e73', '#009e73', '#d62728']
        markers = ['o', 'o', 'D', '^', '^', 's']
        for i, curve in enumerate(data['curves']):
            ax.plot(curve['radius'], curve['mean'], marker=markers[i],
                    ls='None' if i == 4 else '--' if i == 5 else '-',
                    color=colors[i], label=curve['label'], ms=5, lw=2)
            ax.fill_between(curve['radius'], curve['p20'], curve['p80'],
                            color=colors[i], alpha=.12)
        for label, value, color, style in data['horizontal']:
            ax.axhline(value, color=color, ls=style, label=label)
        ax.set_xscale('symlog', linthresh=.001, linscale=1)
        ax.set_xlim(-.00015, 1.65)
        ax.set_xticks([0, .001, .003, .01, .03, .1, .2, .4, .7, 1.5])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.set(xlabel=r'Training radius $\rho$ (linear near zero; logarithmic above .001)',
               ylabel='Mean out-of-sample cost',
               title='R=3, total N=50: completed retailer comparison\n'
                     'Same 10,000 scenarios (seed 9999); means and 20/80 seed bands, not CIs')
        ax.grid(axis='y', alpha=.25)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, fontsize=8)
        fig.text(.5, .025,
                 '*31 isolated continuations + 59 recovered/reference results; no test-based choice.\n'
                 'Positive mean-rule radii connect; isolated zero is nominal only. Not a historical-SCS rerun.\n'
                 'Black includes capped seed 9 at .7 (40 MPs), not converged. All other numeric curves unchanged.\n'
                 'Previously inspected evaluation draw: descriptive, not confirmatory. Cached bounds use different samples.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(bottom=.38)
        fig.savefig(output / 'matched_oos_completed.png', dpi=180, bbox_inches='tight')
    finally:
        plt.close(fig)


def build_report(output, completed_summary):
    """Validate completed inputs, write plot/data/runtime/Markdown; return timing.

    Writes only beneath output, which must already exist and must not be a prior
    artifact directory. Re-reporting there is permitted. Does not set parent
    wall_time_s or write summary.json. report_timing.json measures this invocation
    through output hashing, excluding its own final serialization; rendering is
    a separately measured, nested phase, never added again to reporting total.
    Any missing/mismatched input raises; there is no partial-curve fallback.
    """
    start, started = time.perf_counter(), u.c.utc()
    output = Path(output).resolve()
    _require(output.is_dir() and output not in (u.OUTPUT.resolve(), u.TRAIN.resolve(),
                                               u.PREVIOUS.resolve(), (u.PREVIOUS / 'results').resolve()),
             'Reporting requires an existing isolated output directory')
    _require(completed_summary['status'] == 'completed', 'Report requires a completed campaign')
    sources = {}

    def read(path, expected=None):
        path = Path(path).resolve()
        digest = u.c.sha(path)
        if expected is not None:
            _require(digest == expected, f'Input hash mismatch: {path}')
        value = u.c.read_json(path)
        _require(u.c.sha(path) == digest, f'Input changed while reading: {path}')
        sources[str(path)] = dict(sha256=digest)
        return value

    prior_plot = read(u.OUTPUT / 'plot_data.json')
    prior = read(u.OUTPUT / 'summary.json')
    manifest = read(u.OUTPUT / 'frozen/manifest.json', prior['manifest_sha256'])
    train_path = u.TRAIN / 'summary.json'
    train = read(train_path, manifest['files'][str(train_path)]['sha256'])
    _require(prior['status'] == train['status'] == 'completed', 'Incomplete authoritative source')
    u.c.complete_grid(prior, manifest)
    _require(prior_plot['test_arrays'] == prior['test_arrays'] == manifest['test_arrays'],
             'Prior plot/test identities differ')
    grid = {(s, r) for s in SEEDS for r in (0., *POSITIVE_RADII)}
    recoveries = _index(manifest['recovery'], grid, 'Prior recovery inventory')
    missing = {k for k, r in recoveries.items() if not r['recoverable']}
    _require(len(missing) == 31, 'Expected exactly 31 missing prefixes')
    workers = _index(completed_summary['continuations']['jobs'], missing, 'Continuation workers')
    new_evaluation = completed_summary['evaluation']
    fresh = _index(new_evaluation['jobs'], missing, 'Fresh evaluations')
    _require(new_evaluation['test_arrays'] == prior['test_arrays'], 'New evaluation test arrays differ')
    expected_jobs = []
    for key, worker in workers.items():
        _require(worker['status'] == 'completed' and worker['terminal_status'] == 'mean_gap_satisfied',
                 f'Uncompleted mean-rule continuation: {key}')
        j = fresh[key]
        _require(j['comparison_method'] == 'mean_gap' and j['method'] == 'new'
                 and j['id'] == f'mean_gap_r{key[1]:g}_s{key[0]}' and j['cached'] is False,
                 f'Unexpected fresh evaluation identity: {key}')
        for evaluation_key, worker_key in (
                ('selected_iteration', 'selected_candidate_id'),
                ('checkpoint', 'selected_checkpoint'),
                ('checkpoint_sha256', 'selected_checkpoint_sha256'),
                ('state_sha256', 'selected_state_sha256'),
                ('selected_separation_status', 'selected_separation_status')):
            _require(j[evaluation_key] == worker[worker_key], f'Selected policy mismatch: {key}/{evaluation_key}')
        _require(j['n_test'] == 10000 and j['test_seed'] == 9999 and
                 j['n_train'] == 40 and j['n_val'] == 10, 'Unexpected evaluation protocol')
        selected = worker['iterations'][worker['selected_candidate_id']]
        _require(selected['checkpoint_sha256'] == worker['selected_checkpoint_sha256'] and
                 selected['state_sha256'] == worker['selected_state_sha256'],
                 'Worker selection does not match assessed candidate')
        _require(u.c.sha(worker['selected_checkpoint']) == worker['selected_checkpoint_sha256'],
                 'Selected checkpoint changed after evaluation')
        sources[str(Path(worker['selected_checkpoint']).resolve())] = dict(sha256=worker['selected_checkpoint_sha256'])
        expected_jobs.append(dict(id=j['id'], method='new', radius=key[1], seed=key[0],
                                  checkpoint_sha256=worker['selected_checkpoint_sha256'],
                                  state_sha256=worker['selected_state_sha256']))
    u.c.complete_grid(new_evaluation, dict(jobs=expected_jobs))
    recovered_jobs = [j for j in prior['jobs'] if j['comparison_method'] == 'mean_gap']
    _require(len(recovered_jobs) == 59, 'Expected 59 recovered/reference evaluations')
    recovered_index = _index(recovered_jobs, grid - missing, 'Recovered evaluations')
    for key, j in recovered_index.items():
        r = recoveries[key]
        for jk, rk in (('selected_iteration', 'selected_candidate'), ('checkpoint', 'checkpoint'),
                       ('checkpoint_sha256', 'checkpoint_sha256'), ('state_sha256', 'state_sha256'),
                       ('selected_separation_status', 'selected_separation_status')):
            _require(j[jk] == r[rk], f'Recovered policy identity mismatch: {key}')
    combined = recovered_jobs + new_evaluation['jobs']
    _require(len({j['id'] for j in combined}) == 90, 'Duplicate mean-rule evaluation IDs')
    evaluations = _index(combined, grid, 'Completed 90-job mean-rule grid')

    data = deepcopy(prior_plot)
    _require(len(data['curves']) == 6 and data['curves'][4]['radius'] == [0.],
             'Unexpected prior curve layout')
    values = np.asarray([[evaluations[s, r]['mean_oos'] for s in SEEDS] for r in POSITIVE_RADII])
    green = dict(label='Moreau: mean-gap stop / min-AP*', radius=list(POSITIVE_RADII),
                 per_seed=values.tolist(), mean=values.mean(axis=1).tolist(),
                 p20=np.percentile(values, 20, axis=1).tolist(),
                 p80=np.percentile(values, 80, axis=1).tolist())
    for i, r in enumerate(prior_plot['curves'][3]['radius']):
        for field in ('per_seed', 'mean', 'p20', 'p80'):
            _require(green[field][POSITIVE_RADII.index(r)] == prior_plot['curves'][3][field][i],
                     'Previously plotted green value changed')
    _require(data['curves'][4]['per_seed'] == [[evaluations[s, 0.]['mean_oos'] for s in SEEDS]],
             'Nominal reference changed')
    data['curves'][3] = green
    data['curves'][2]['label'] = 'Moreau: every-center stop (cap 40 MPs)'
    for i in (0, 1, 2, 4, 5):
        for field in ('radius', 'per_seed', 'mean', 'p20', 'p80'):
            _require(data['curves'][i][field] == prior_plot['curves'][i][field],
                     'Non-green numeric plot values changed')
    full_jobs = _index(train['ccg']['jobs'], grid, 'Full Moreau grid')
    capped = [k for k, j in full_jobs.items() if j['terminal_status'] == 'iteration_cap']
    _require(capped == [(9, .7)] and full_jobs[9, .7]['mp_count'] == 40,
             'Unexpected black capped configuration')
    models = []
    for key, j in sorted(full_jobs.items()):
        record = read(j['summary_path'], j['summary_sha256'])
        _require((record['seed'], record['radius']) == key and record['status'] == j['status']
                 and record['terminal_status'] == j['terminal_status'], 'Full-model source mismatch')
        phases = _phases(record)
        totals = _phase_totals(phases)
        models.append(dict(method='Moreau every-center', seed=key[0], radius=key[1],
                           status=record['status'], terminal_status=record['terminal_status'],
                           origin='historical_full', n_train=40, n_val=10, workers=16, threads=1,
                           wall_s=_seconds(record['wall_time_s']), current_wall_s=None,
                           source_path=j['summary_path'], source_sha256=j['summary_sha256'],
                           started_at_utc=record['started_at_utc'], ended_at_utc=record['ended_at_utc'],
                           phases=phases, **totals))
        if key in workers:
            short = workers[key]
            phases = _phases(short, record)
            totals = _phase_totals(phases)
            _check_totals(short, totals)
            for name, phase_name in (('current_ap_total_time_s', 'ap'), ('current_mp_total_time_s', 'mp_call')):
                measured = sum(p['wall_time_s'] for p in phases if p['origin'] == 'current' and p['phase'] == phase_name)
                _require(np.isclose(_seconds(short[name]), measured, rtol=1e-12, atol=1e-9),
                         f'Current invocation total mismatch: {name}')
            origin, terminal = 'continued', short['terminal_status']
            raw = dict(worker=short)  # Includes cached_prefix, resume load, selections and all iterations.
            current_wall = _seconds(short['wall_time_s'])
        else:
            short = recoveries[key]
            phases = [dict(p, origin='saved') for p in short['phases']]
            prefix_record = dict(record, iterations=record['iterations'][:short['stop_candidate'] + 1],
                                 mp_count=short['mp_count'])
            _require(phases == _phases(prefix_record), 'Recovered phases differ from original prefix scope')
            totals = _phase_totals(phases)
            _check_totals(short, totals)
            origin = 'nominal_reference' if key[1] == 0 else 'recovered'
            terminal = 'nominal_reference' if key[1] == 0 else 'mean_gap_satisfied'
            raw = dict(recovery=short)
            current_wall = None
        models.append(dict(method='Moreau mean-gap', seed=key[0], radius=key[1], status='completed',
                           terminal_status=terminal, origin=origin, n_train=40, n_val=10,
                           workers=16, threads=1, wall_s=None, fresh_short_run_wall_s=None,
                           current_wall_s=current_wall, source_path=j['summary_path'],
                           source_sha256=j['summary_sha256'], phases=phases, **totals, **raw))

    # Exact plotted fixed-radius SCS evidence: inspect all 230, never average zeros.
    blue_radii = set(data['curves'][0]['radius'])
    directories = {float(p.name[4:].replace('p', '.')): p for p in u.c.SWEEP.glob('rho_*') if p.is_dir()}
    fixed_scs = []
    for radius in sorted(blue_radii):
        for seed in SEEDS:
            path = directories[radius] / f'seed_{seed}_summary.json'
            r = read(path)
            _require(r['seed'] == seed and r['rho'] == radius, 'Fixed-radius SCS source identity mismatch')
            _require(r['total_time_s'] == 0, 'Historical plotted SCS timing assumption changed; review required')
            fixed_scs.append(dict(radius=radius, seed=seed, source_path=str(path),
                                  source_sha256=sources[str(path.resolve())]['sha256'],
                                  reported_total_time_s=r['total_time_s'], wall_s=None,
                                  status='unavailable_zero_placeholder', raw=r))
    _require(len(fixed_scs) == 230, 'Expected 230 plotted fixed-radius SCS timing inputs')

    selected_scs, nominal_scs, nominal_moreau = [], [], []
    for path in sorted(SCS_SELECTED.glob('rho_*/seed_*_summary.json')):
        r = read(path)
        _require(r['n_samples'] == 50 and path.name == f'seed_{r["seed"]}_summary.json'
                 and float(path.parent.name[4:].replace('p', '.')) == r['rho'],
                 'UNMATCHED SCS source configuration mismatch')
        _require(r['status'] == 'completed' and r['completed'] and r['converged'],
                 f'Uncompleted UNMATCHED SCS timing rerun: {path}')
        trajectory = r['ccg_trajectory']
        for a in trajectory:
            _require(a['ap_status'] == 'completed' and
                     a['mp_status'] in ('completed', 'not_run_converged'), 'Unexpected SCS trajectory status')
            _seconds(a['ap_time_s'])
            _seconds(a['mp_time_s'])
            _require(a['mp_status'] != 'not_run_converged' or a['mp_time_s'] == 0,
                     'Unrun SCS MP has nonzero time')
        selected_scs.append(dict(method='SCS UNMATCHED selected-radius rerun', seed=r['seed'],
                                 radius=r['rho'], status='completed', terminal_status='converged',
                                 origin='historical_unmatched_rerun', n_train=50, n_val=0,
                                 workers=30, worker_scope='30 original concurrent jobs across N=50/200/800',
                                 threads=1, wall_s=_seconds(r['total_time_s']),
                                 ap_s=sum(a['ap_time_s'] for a in trajectory if a['ap_status'] == 'completed'),
                                 mp_s=sum(a['mp_time_s'] for a in trajectory if a['mp_status'] == 'completed'),
                                 ap_count=sum(a['ap_status'] == 'completed' for a in trajectory),
                                 mp_count=sum(a['mp_status'] == 'completed' for a in trajectory),
                                 source_path=str(path), source_sha256=sources[str(path.resolve())]['sha256'], raw=r))
    for path in sorted(SCS_NOMINAL.glob('seed_*_summary.json')):
        r = read(path)
        _require(r['n_samples'] == 50 and path.name == f'seed_{r["seed"]}_summary.json',
                 'Nominal SCS source configuration mismatch')
        _require(r['status'] == 'completed' and r['completed'] and r['nominal_training_status'] == 'computed',
                 f'Incomplete nominal SCS timing: {path}')
        nominal_scs.append(dict(method='SCS nominal separate rerun', seed=r['seed'], radius=0.,
                                status='completed', terminal_status='training_completed',
                                origin='historical_nominal_rerun', n_train=50, n_val=0,
                                workers=None, threads=None, wall_s=_seconds(r['nominal_training_time_s']),
                                epochs=r['nominal_epochs_completed'], source_path=str(path),
                                source_sha256=sources[str(path.resolve())]['sha256'], raw=r))
    for j in train['nominal']['jobs']:
        timing = j['winner']['training_timing']
        _require(j['status'] == timing['status'] == 'completed', 'Incomplete selected nominal training')
        nominal_moreau.append(dict(method='Moreau nominal selected winner', seed=j['seed'], radius=0.,
                                   status='completed', terminal_status='training_completed',
                                   origin='historical_selected_winner', n_train=40, n_val=10,
                                   workers=10, threads=1, wall_s=_seconds(timing['wall_time_s']),
                                   epochs=timing['epochs_evaluated'], source_path=str(train_path),
                                   source_sha256=sources[str(train_path.resolve())]['sha256'], raw=j))
    for description, rows in (('Selected SCS', selected_scs), ('Nominal SCS', nominal_scs),
                              ('Nominal Moreau', nominal_moreau)):
        _require(len(rows) == 10 and {r['seed'] for r in rows} == set(SEEDS), f'{description}: incomplete seed cohort')

    metrics = ('wall_s', 'phase_sum_s', 'saved_phase_s', 'current_phase_s', 'current_wall_s',
               'ap_s', 'mp_s', 'ap_count', 'mp_count')
    model_aggregates = _aggregates(models, metrics)
    origin_aggregates = _aggregates(models, metrics, split_origin=True)
    nominal_aggregates = _aggregates(nominal_scs + nominal_moreau, ('wall_s', 'epochs'))
    scs_aggregates = _aggregates(selected_scs, ('wall_s', 'ap_s', 'mp_s', 'ap_count', 'mp_count'))
    scs_overall = {metric: u.c.timing_stats([j[metric] for j in selected_scs])
                   for metric in ('wall_s', 'ap_s', 'mp_s', 'ap_count', 'mp_count')}
    evaluation_rows = []
    for origin, rows in (('prior_evaluation', prior['jobs']), ('current_evaluation', new_evaluation['jobs'])):
        for j in rows:
            evaluation_rows.append(dict(method=f'OOS {j["comparison_method"]}', seed=j['seed'],
                                        radius=j['radius'], status=j['status'], terminal_status='strict_OPTIMAL',
                                        origin=origin, wall_s=_seconds(j['wall_time_s']), raw=j))
    evaluation_aggregates = _aggregates(evaluation_rows, ('wall_s',), split_origin=True)
    cost_rows = []
    blue = dict(zip(data['curves'][0]['radius'], data['curves'][0]['mean']))
    black = dict(zip(data['curves'][2]['radius'], data['curves'][2]['mean']))
    green_means = dict(zip(green['radius'], green['mean']))
    for radius in sorted(blue.keys() & black.keys() & green_means.keys()):
        cost_rows.append(dict(radius=radius, blue_scs=blue[radius], black_full=black[radius],
                              green_short=green_means[radius], black_minus_blue=black[radius] - blue[radius],
                              black_minus_green=black[radius] - green_means[radius]))
    campaign_times = dict(
        prior_nominal_lr_search_s=_seconds(train['nominal']['wall_time_s']),
        prior_full_ccg_sweep_s=_seconds(train['ccg']['wall_time_s']),
        prior_terminal_validation_s=_seconds(train['validation']['wall_time_s']),
        prior_evaluation_invocation_s=_seconds(prior['wall_time_s']),
        new_continuation_sweep_s=_seconds(completed_summary['continuations']['wall_time_s']),
        new_evaluation_sweep_s=_seconds(new_evaluation['wall_time_s']))
    runtime = dict(status='completed', timing_unit='seconds', timing_clock='time.perf_counter',
                   phase_scope=PHASE_SCOPE, sources=sources,
                   campaign_times=campaign_times, current_parent_phases=deepcopy(completed_summary['phases']),
                   parent_final_wall_time_s=None,
                   parent_final_wall_time_note='Caller sets final wall time after reporting returns; not inferred here.',
                   environment=dict(prior_training_host=train['host'], prior_evaluation=prior.get('environment'),
                                    current=completed_summary.get('environment'),
                                    hardware_note='Unrecorded CPU model/memory and historical SCS host are unknown.'),
                   protocol=train['protocol'], models=models, model_aggregates=model_aggregates,
                   origin_aggregates=origin_aggregates, nominal_moreau=nominal_moreau,
                   nominal_scs=nominal_scs, nominal_aggregates=nominal_aggregates,
                   scs_plotted_fixed_radius=dict(status='unavailable', rerun_required=True, jobs=fixed_scs),
                   scs_unmatched_selected=dict(jobs=selected_scs, aggregates=scs_aggregates,
                                              overall_statistics=scs_overall, matched_to_plot=False),
                   evaluation=dict(jobs=evaluation_rows, aggregates=evaluation_aggregates,
                                   manifest_sha256=new_evaluation['manifest_sha256'], test_arrays=new_evaluation['test_arrays']),
                   continuations=deepcopy(completed_summary['continuations']),
                   prior_evaluation_phases=prior['phases'], cost_comparison=cost_rows,
                   sro_fa=dict(status='cached_not_rerun', partition_s=None, solver_s=None, evaluation_s=None, total_s=None),
                   cv=dict(status='not_run', wall_time_s=None),
                   new_hyperparameter_selection=dict(status='not_run', wall_time_s=None))

    lines = ['# Completed mean-rule retailer comparison', '',
             '31 isolated continuation jobs and 31 fresh strict evaluations completed; combined with '
             '59 recovered/reference evaluations, the green grid contains exactly 90 unique seed/radius '
             'results (10 seeds per radius). Eight positive radii connect; zero remains an isolated nominal reference.',
             '', 'Black: **Moreau: every-center stop (cap 40 MPs)**. Green: **Moreau: mean-gap stop / min-AP***.',
             'The green policy is chosen using training/AP information, not test cost. No new test-based '
             'selection, tuning, significance tests, projection solves or retraining occurs in this report. '
             'The same 10,000 scenarios (seed 9999) were previously inspected: results are descriptive, not confirmatory. '
             'Bands are 20th/80th percentiles across seed costs around the mean, **not confidence intervals**.',
             'Only the green positive curve changes numerically. Other curves, isolated nominal and horizontal '
             'references retain their prior numeric values exactly; old outputs are not overwritten.',
             'Black seed 9 at radius .7 exhausted 40 MPs and is **capped, not converged**. Its timing is '
             'aggregated separately. Mean-gap satisfaction does not certify every-center separation or global optimality.',
             '', '## Cost comparison at common radii', '',
             'Signed differences are black minus comparator; negative means lower black cost. '
             'Means below come from the plotted arrays, not a different policy/radius selection.', '',
             '| Radius | Blue SCS | Black full | Green short | Black − blue | Black − green |',
             '|---:|---:|---:|---:|---:|---:|']
    for r in cost_rows:
        lines.append(f'| {r["radius"]:g} | ' + ' | '.join(f'{r[k]:.9f}' for k in
                     ('blue_scs', 'black_full', 'green_short', 'black_minus_blue', 'black_minus_green')) + ' |')
    lines += ['', '## Runtime interpretation', '', PHASE_SCOPE,
              'Full Moreau wall time is the historical per-configuration CCG invocation (nominal training separate). '
              'Short full wall time is unavailable: recovered paths were not fresh runs, and continuation wall time '
              'covers only the current resume invocation. The comparable short phase sum combines persisted '
              'candidate-zero phases with newly measured continuation phases. Saved/current splits and origin '
              'cohort counts are retained below and in JSON. Resume input/loading and selection overhead remain '
              'in current wall time and raw worker records, but not in the matched phase sum.',
              'Model durations are concurrent wall times, not additive sweep elapsed times. Moreau CCG and '
              'continuations use 16 workers, one numerical/solver thread each. Prior strict evaluation used '
              '8 workers; fresh evaluation concurrency is recorded per result. Nominal Moreau used 10 workers. '
              'Unrecorded historical nominal-SCS concurrency/threads are unknown.',
              f'Environment: {runtime["environment"]}.',
              '', '## Campaign and evaluation/selection elapsed times', '',
              '| Phase | Measured seconds / minutes / hours |', '|---|---|']
    for name, seconds in campaign_times.items():
        lines.append(f'| {name} | {u.c.human(seconds)} |')
    for name, p in completed_summary['phases'].items():
        duration = p.get('wall_time_s') if isinstance(p, dict) else p
        shown = u.c.human(_seconds(duration)) if duration is not None else 'unavailable'
        status = p.get('status', 'not recorded') if isinstance(p, dict) else 'duration only'
        lines.append(f'| Current parent phase: {name} ({status}) | {shown} |')
    lines += ['| Parent final elapsed | Pending caller finalization after this function; not inferred |',
              '| SRO-FA partition / solver / evaluation / total | Cached; timing unavailable, not rerun |',
              '| CV / new hyperparameter selection / new benchmark construction | None |',
              '| Exact plotted fixed-radius SCS training/CCG | Unavailable: all 230 total_time_s fields are zero placeholders; rerun required |',
              '| Rendering / reporting | Separate measured report_timing.json; rendering is nested, not additive |']
    _runtime_table(lines, 'Nominal training (separate from CCG)', nominal_aggregates, ('wall_s', 'epochs'))
    _runtime_table(lines, 'Moreau by radius and completion/terminal status', model_aggregates,
                   ('wall_s', 'phase_sum_s', 'ap_s', 'mp_s', 'ap_count', 'mp_count'))
    short_origins = [a for a in origin_aggregates if a['method'] == 'Moreau mean-gap']
    _runtime_table(lines, 'Short-rule saved/current split by origin', short_origins,
                   ('saved_phase_s', 'current_phase_s', 'current_wall_s'))
    lines += ['', '## UNMATCHED selected-radius SCS timing reruns — not the plotted SCS policies', '',
              'These 10 completed/converged timing reruns used validation-selected radii/policies and 50 training '
              'samples versus 40 training / 10 validation for Moreau. The original run launched 30 concurrent '
              'jobs across three N sizes. They are **not timings for the fixed-radius blue curve**. '
              'Do not infer a solver-only speedup: policies, radii, training sizes and algorithms differ. '
              'AP/MP counts use trajectory status fields, not ccg_iters (which can identify an incumbent). '
              'Initialization was cached; nominal SCS training is reported separately. All statistics are '
              'computed from the actual timing files, never substituted with expected values.', '',
              '| UNMATCHED overall metric | n | Mean / median / min / max | Mean minutes |',
              '|---|---:|---|---:|']
    for metric, stats in scs_overall.items():
        minutes = f'{stats["mean"] / 60:.6f}' if metric.endswith('_s') else '—'
        lines.append(f'| {metric} | {stats["count"]} | ' +
                     ' / '.join(f'{stats[k]:.6f}' for k in ('mean', 'median', 'min', 'max')) + f' | {minutes} |')
    _runtime_table(lines, 'UNMATCHED SCS by selected radius/status', scs_aggregates,
                   ('wall_s', 'ap_s', 'mp_s', 'ap_count', 'mp_count'))
    _runtime_table(lines, 'Strict OOS evaluation by radius/status and invocation', evaluation_aggregates, ('wall_s',))
    lines += ['', '## Reproducibility', '',
              'Raw phase durations, AP/MP counts and trajectories, source paths/SHA-256 hashes, nominal '
              'epochs, worker records, timestamps and evaluation identities are in runtime_comparison.json. '
              'Unknown metadata/timings remain null, not zero. Continuation records preserve cached_prefix, '
              'resume input/loading, selected policy evidence and current invocation totals. '
              'No cached timing is labeled as current full-model time. No failed/incomplete or capped '
              'configuration is mixed into a converged aggregate.', '',
              '[Completed plot](matched_oos_completed.png) · [Plot data](plot_data.json) · '
              '[Raw timing and provenance](runtime_comparison.json) · [Reporting time](report_timing.json)']

    # Check all consumed inputs again before publishing report artifacts.
    for path, info in sources.items():
        _require(u.c.sha(path) == info['sha256'], f'Source changed during report: {path}')
    render_start, render_started = time.perf_counter(), u.c.utc()
    _render(output, data)
    render = dict(status='completed', start_utc=render_started, end_utc=u.c.utc(),
                  wall_time_s=time.perf_counter() - render_start)
    runtime['rendering'] = render
    u.c.write_json(output / 'plot_data.json', data)
    u.c.write_json(output / 'runtime_comparison.json', runtime)
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    artifact_hashes = {name: u.c.sha(output / name) for name in
                       ('plot_data.json', 'runtime_comparison.json', 'REPORT.md', 'matched_oos_completed.png')}
    timing = dict(status='completed', start_utc=started, end_utc=u.c.utc(),
                  wall_time_s=time.perf_counter() - start, timing_clock='time.perf_counter',
                  timing_unit='seconds', phases=dict(rendering=render),
                  scope='This reporting invocation through output hashing; excludes final timing-file serialization. '
                        'Rendering is nested. No experiment/evaluation or parent-final-wall measurement.',
                  script_sha256=u.c.sha(Path(__file__)), output=str(output), artifact_sha256=artifact_hashes)
    timing['wall_time_human'] = u.c.human(timing['wall_time_s'])
    u.c.write_json(output / 'report_timing.json', timing)
    return timing