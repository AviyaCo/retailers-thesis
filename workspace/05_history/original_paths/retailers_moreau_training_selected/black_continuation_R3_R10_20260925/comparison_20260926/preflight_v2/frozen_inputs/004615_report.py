"""Reporting only: consume supplied summaries, never load policies or observations.

The caller owns validation-based selection, provenance verification, phase timing,
and final summary.json publication. No training, evaluation, or selection occurs
here; only four presentation artifacts are written beneath output.
"""

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np


RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
COLORS = ('green', 'black')
METRICS = ('reconstructed_phase_s', 'ap_s', 'mp_s', 'mp_count', 'source_ccg_wall_s')
SCOPE = (
    'Saved nonoverlapping phases through STOP, not the returned checkpoint; '
    'excludes nominal training, selection/publication and uninstrumented overhead. '
    'Not fresh end-to-end runtime or parallel sweep elapsed time.'
)
CAVEATS = (
    'Both rules return the earliest minimum mean AP incumbent in their own stopping '
    'prefix. Green is initial-AP eligible; black retains every-center stopping. '
    'Stopping and returned checkpoints are distinct. No new global optimum or '
    'every-center certificate is claimed for the returned minimum-AP policy; '
    'AP incumbent rankings do not resolve differences within AP bounds.',
    'Black seed 9 at radius 0.7 remains capped at 40 MPs even when an earlier '
    'checkpoint is returned. It is included in the complete ten-seed cost curve '
    'but separated from stopping-rule-satisfied runtime cohorts.',
    'All fixed-radius costs use all ten seeds. Supplied radius selections were '
    'frozen using validation only (smaller-radius ties, nominal eligible) before '
    'test lookup. Reporting never selects, tunes, clips costs, or interpolates red.',
    'Cache reuse requires exact seed, checkpoint SHA256 and learned-state SHA256 '
    'under the same evaluator and data contract. Reuse across radii is legitimate '
    'for the same policy because evaluation is independent of training radius. '
    'Repeated cache references are neither new observations nor current runtime.',
    'Red SRO-FA, yellow kNN, blue, purple and horizontal reference numbers are '
    'unchanged. Historical training/selection budgets and reference samples differ; '
    'this is not a matched-protocol or statistical-significance comparison. The '
    'previously inspected seed-9999 test and test-informed historical radius grid '
    'permit descriptive, not confirmatory, conclusions. Wins are not guaranteed.',
    'Bands are empirical 20th/80th percentiles across seed means, not confidence '
    'intervals. Red comparisons exist only at exactly shared radius values.',
    SCOPE,
    'Fresh green and black protocol end-to-end times are unavailable. No new '
    'training, AP search, MP optimization, CV, or benchmark construction was run. '
    'Cached durations must not be counted as current work. Missing historical '
    'timings require a rerun to measure; SRO-FA runtime is unavailable.',
    'Current invocation total and reporting-phase completion are not yet available '
    'here: consult summary.json finalized by the caller. Phase snapshots may show '
    'reporting running. Do not sum nested phases or concurrent model wall times '
    'and present the sum as elapsed campaign runtime.',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seconds(value):
    _require(value is not None and math.isfinite(value) and value >= 0,
             f'Invalid measured duration: {value!r}')
    return float(value)


def _stats(values):
    values = list(values)
    _require(all(math.isfinite(v) for v in values), 'Nonfinite reporting value')
    return dict(n=len(values), mean=mean(values) if values else None,
                median=median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


def _human(seconds):
    return ('unavailable' if seconds is None else
            f'{seconds:.6f} s / {seconds / 60:.6f} min')


def _cohort(terminal):
    if terminal == 'iteration_cap':
        return 'capped'
    if terminal in ('separation_satisfied', 'mean_gap_satisfied'):
        return 'stopping_rule_satisfied'
    return terminal  # Nominal reference and stagnation stay separate.


def _cost_stats(rows):
    values = [r['test_mean'] for r in rows]
    return dict(**_stats(values), p20=float(np.percentile(values, 20)),
                p80=float(np.percentile(values, 80)), seeds=[r['seed'] for r in rows],
                per_seed=values, terminal_status_counts=dict(Counter(
                    r['terminal_status'] for r in rows)),
                capped_seeds=[r['seed'] for r in rows if r['terminal_status'] == 'iteration_cap'])


def _render(path, data):
    # Avoid pyplot/backend/global figure state in the caller.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    fig = Figure(figsize=(11, 8))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    colors = ('#2166ac', '#762a83', '#111111', '#009e73', '#009e73', '#d62728')
    markers = ('o', 'o', 'D', '^', '^', 's')
    for i, curve in enumerate(data['curves']):
        ax.plot(curve['radius'], curve['mean'], marker=markers[i], color=colors[i],
                ls='None' if i == 4 else '--' if i == 5 else '-',
                label=curve['label'], ms=5, lw=2)
        ax.fill_between(curve['radius'], curve['p20'], curve['p80'],
                        color=colors[i], alpha=.12)
    for label, value, color, style in data['horizontal']:
        ax.axhline(value, color=color, ls=style, label=label)
    ax.set_xscale('symlog', linthresh=.001, linscale=1)
    ax.set_xticks([0, .001, .003, .01, .03, .1, .2, .4, .7, 1.5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set(xlabel=r'Training radius $\rho$ (linear near zero; logarithmic above .001)',
           ylabel='Mean out-of-sample cost',
           title='R=3, total N=50: stopping-only / minimum-AP return\n'
                 '10,000 scenarios (seed 9999); 20/80 seed bands, not CIs')
    ax.grid(axis='y', alpha=.25)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=2, fontsize=8)
    fig.text(.5, .025,
             'Green initial AP is eligible; both rules return minimum-AP within their own stop.\n'
             'Black includes capped seed 9 at .7 (40 MPs), not converged. Zero green is nominal.\n'
             'All ten seeds at each radius; reference numbers unchanged; no test-based choice.\n'
             'Previously inspected test: descriptive only; historical protocols differ.',
             ha='center', fontsize=8)
    fig.subplots_adjust(bottom=.38)
    fig.savefig(path, dpi=180, bbox_inches='tight')


def build_report(output: Path, summary: dict, old_plot: dict) -> dict:
    """Write RESULTS.md, runtime_comparison.json, plot_data.json, comparison.png.

    Inputs are not mutated. Requires the complete supplied grid and preselected
    rows; does not open any input files or recompute selections. Runtime JSON
    preserves per-configuration evidence, historical/current origins, and the
    caller's phase snapshot. The caller must finalize its own total afterwards.
    """
    _require(summary['status'] == 'completed', 'Require completed evaluation inputs')
    choices = summary['choices']
    index = {(r['color'], r['seed'], r['radius']): r for r in choices}
    expected = {(c, s, r) for c in COLORS for s in range(10) for r in RADII}
    _require(len(choices) == 180 and set(index) == expected, 'Incomplete/duplicate grid')
    _require(len({r['id'] for r in choices}) == 180, 'Duplicate choice IDs')
    _require(summary['workers'] == summary['threads_per_worker'] == 1,
             'Expected one current worker and thread')
    for row in choices:
        _require(all(math.isfinite(row[k]) for k in ('validation_mean', 'test_mean')),
                 'Nonfinite cost: no partial-curve publication')
        _require(row['fresh_protocol_wall_s'] is None, 'Unexpected fresh training runtime')
        for metric in METRICS:
            _seconds(row[metric])
        for split in ('validation', 'test'):
            _require(row[f'{split}_origin'] in ('cached', 'current'), 'Invalid evaluation origin')
            _seconds(row[f'{split}_wall_s'])
    capped = [r for r in choices if r['terminal_status'] == 'iteration_cap']
    _require([(r['color'], r['seed'], r['radius'], r['mp_count']) for r in capped]
             == [('black', 9, .7, 40)], 'Unexpected capped cohort')

    # Validate supplied identities, not the selection algorithm or its ranking.
    selected_stats = {}
    for color in COLORS:
        selected = sorted(summary['selected_radii'][color], key=lambda r: r['seed'])
        _require([r['seed'] for r in selected] == list(range(10)), 'Incomplete selected cohort')
        for row in selected:
            original = index[color, row['seed'], row['radius']]
            for key in ('id', 'validation_mean', 'test_mean', 'selected_iteration',
                        'stop_candidate', 'terminal_status', 'checkpoint_sha256', 'state_sha256'):
                _require(row[key] == original[key], f'Selected identity mismatch: {key}')
        selected_stats[color] = dict(
            test=_cost_stats(selected), validation=_stats(r['validation_mean'] for r in selected),
            rows=deepcopy(selected), by_terminal_status={
                status: _cost_stats([r for r in selected if r['terminal_status'] == status])
                for status in sorted({r['terminal_status'] for r in selected})})

    fixed = [dict(color=c, radius=r, **_cost_stats([index[c, s, r] for s in range(10)]))
             for c in COLORS for r in RADII]
    fixed_index = {(r['color'], r['radius']): r for r in fixed}
    _require(len(old_plot['curves']) == 6 and old_plot['curves'][4]['radius'] == [0.],
             'Unexpected source curve layout')
    data = deepcopy(old_plot)
    for i, color, radii, label in (
        (2, 'black', RADII, 'Moreau: every-center stop / min-AP (cap 40 MPs)'),
        (3, 'green', RADII[1:], 'Moreau: mean-gap stop / min-AP (initial eligible)'),
        (4, 'green', RADII[:1], 'Moreau: min-AP nominal (green; initial eligible)'),
    ):
        rows = [fixed_index[color, r] for r in radii]
        data['curves'][i] = dict(label=label, radius=list(radii), **{
            key: [r[key] for r in rows] for key in ('per_seed', 'mean', 'p20', 'p80')})
    _require(all(data['curves'][i] == old_plot['curves'][i] for i in (0, 1, 5))
             and data['horizontal'] == old_plot['horizontal'], 'Reference numbers changed')

    yellow = [h for h in old_plot['horizontal'] if h[2] == '#e6a817']
    _require(len(yellow) == 1, 'Expected one yellow kNN reference')
    red = old_plot['curves'][5]
    _require(len(red['radius']) == len(red['mean']) == len(set(red['radius'])),
             'Malformed red reference')
    red_means = dict(zip(red['radius'], red['mean']))
    _require(all(math.isfinite(v) for v in [yellow[0][1], *red_means.values()]),
             'Nonfinite reference')
    comparisons = []
    for row in fixed:
        radius, value = row['radius'], row['mean']
        reference = red_means.get(radius)  # Exact key only; no interpolation.
        comparisons.append(dict(color=row['color'], radius=radius, test_mean=value,
                                yellow_mean=yellow[0][1], test_minus_yellow=value - yellow[0][1],
                                red_mean=reference, test_minus_red=(
                                    None if reference is None else value - reference)))
    comparison = dict(yellow_label=yellow[0][0], red_label=red['label'],
                      interpretation='Negative differences mean lower cost; descriptive only.',
                      rows=comparisons, counts={})
    for color in COLORS:
        comparison['counts'][color] = {}
        for reference in ('yellow', 'red'):
            differences = [r[f'test_minus_{reference}'] for r in comparisons
                           if r['color'] == color and r[f'test_minus_{reference}'] is not None]
            comparison['counts'][color][reference] = dict(
                n=len(differences), lower=sum(v < 0 for v in differences),
                equal=sum(v == 0 for v in differences), higher=sum(v > 0 for v in differences))

    # Preserve saved phase evidence, but never label green's parent black wall
    # time as its own runtime. Historical black wall time is not min-AP rerun time.
    runtime_fields = ('id', 'color', 'seed', 'radius', 'n_train', 'n_val', 'network_seed',
                      'split_seed', 'method', 'training_backend', 'status', 'terminal_status',
                      'source_terminal_status', 'stop_candidate', 'selected_iteration',
                      'checkpoint_sha256', 'state_sha256', 'selected_separation_status',
                      'ap_sweeps', 'pool_evaluation_s', 'source_summary_path', 'source_start_utc',
                      'source_stop_ap_end_utc', 'source_ccg_workers', 'threads_per_worker',
                      'phases', 'timing_scope', 'validation_mean', 'test_mean',
                      'validation_origin', 'test_origin', 'validation_wall_s', 'test_wall_s')
    models = []
    for row in choices:
        item = {k: deepcopy(row[k]) for k in runtime_fields}
        item.update({k: row[k] for k in METRICS})
        item.update(cohort=_cohort(row['terminal_status']), origin='historical_reconstruction',
                    fresh_protocol_wall_s=None)
        if row['color'] != 'black':
            item['source_ccg_wall_s'] = None
        models.append(item)
    groups = []
    for color, radius, status, terminal in sorted({
            (r['color'], r['radius'], r['status'], r['terminal_status']) for r in models}):
        rows = [r for r in models if (r['color'], r['radius'], r['status'], r['terminal_status'])
                == (color, radius, status, terminal)]
        groups.append(dict(color=color, radius=radius, status=status, terminal_status=terminal,
                           cohort=_cohort(terminal), n_train=40, n_val=10, n=len(rows),
                           seeds=sorted(r['seed'] for r in rows), metrics={
                               k: dict(**_stats(r[k] for r in rows if r[k] is not None),
                                       unavailable_count=sum(r[k] is None for r in rows))
                               for k in METRICS}))

    nominal = []
    for seed in range(10):
        timing = index['green', seed, 0.]['cached_nominal']['training_timing']
        _require(timing['status'] == 'completed', 'Incomplete historical nominal training')
        _require(all(r['cached_nominal']['training_timing'] == timing
                     for r in choices if r['seed'] == seed), 'Inconsistent nominal timing reuse')
        nominal.append(dict(seed=seed, radius=0., n_train=40, n_val=10,
                            origin='historical_selected_LR_winner', workers=10, threads_per_worker=1,
                            wall_time_s=_seconds(timing['wall_time_s']),
                            epochs_evaluated=timing['epochs_evaluated'], training_timing=deepcopy(timing),
                            evaluation_by_color={c: {k: index[c, seed, 0.][k] for k in (
                                'id', 'validation_origin', 'test_origin', 'validation_wall_s', 'test_wall_s')}
                                for c in COLORS}))
    evaluation = {}
    evidence_keys = ('id', 'seed', 'radius', 'method', 'status', 'wall_time_s',
                     'start_utc', 'end_utc', 'started_at_utc', 'ended_at_utc', 'phases',
                     'checkpoint_sha256', 'state_sha256', 'sample_count', 'n_test', 'test_seed')
    for split in ('validation', 'test'):
        fresh = summary[f'fresh_{split}']
        _require(len(fresh) == 3, f'Expected three fresh {split} records')
        durations = [_seconds(r['wall_time_s']) for r in fresh]
        evaluation[split] = dict(
            current_records=[{k: deepcopy(r[k]) for k in evidence_keys if k in r} for r in fresh],
            current_wall_s_statistics=_stats(durations), current_record_sum_s=sum(durations),
            choice_origin_counts=dict(Counter(r[f'{split}_origin'] for r in choices)),
            scope='Sum of three actual evaluation calls, not duplicated choice references or full phase elapsed.')

    runtime = dict(
        experiment='stopping_only_min_ap', unit='seconds', scope=SCOPE,
        environment=deepcopy(summary['environment']),
        historical_environment=deepcopy(summary['historical_environment']),
        workers=1, threads_per_worker=1, source_ccg_workers=16, source_nominal_workers=10,
        started_at_utc=summary['started_at_utc'], end_utc=None, current_total_wall_s=None,
        current_total_source='summary.json finalized by caller after reporting',
        current_phases_snapshot=deepcopy(summary['phases']),
        fresh_protocol_wall_s={'green': None, 'black': None},
        models=models, aggregates=groups,
        capped_configurations=[r for r in models if r['cohort'] == 'capped'],
        nominal=dict(per_seed=nominal, training_wall_s_statistics=_stats(r['wall_time_s'] for r in nominal),
                     epochs_statistics=_stats(r['epochs_evaluated'] for r in nominal),
                     evaluation_scope='Per-policy persisted durations shown separately; shared references are not additive.'),
        current_evaluation=evaluation,
        source_campaign={k: _seconds(summary['source_campaign'][k]) for k in (
            'nominal_lr_search_s', 'ccg_sweep_s', 'validation_s', 'overall_s')},
        not_run=['new nominal training', 'new AP search', 'new MP optimization', 'new CV',
                 'new benchmark construction'],
        sro_fa=dict(partition_s=None, solver_s=None, evaluation_s=None, total_s=None,
                    status='unavailable; rerun required to measure'),
        input_sha256=deepcopy(summary['input_sha256']), caveats=list(CAVEATS))

    lines = ['# Stopping-only minimum-AP reporting', '',
             'R=3; N=50 = 40 train + 10 validation; seeds 0–9; network seed 0; '
             'split seed 100000 + seed. Frozen historical test: 10,000 scenarios, seed 9999. '
             'Native validation and strict OPTIMAL-only test evaluation are unchanged.', '',
             '## Interpretation and provenance', '']
    lines += [f'- {text}' for text in CAVEATS]
    lines += ['', '## Fixed-radius means and descriptive references', '',
              'Negative differences indicate lower cost. All rows contain ten seeds, including capped policies. '
              'No red entry means that exact radius is absent, not an interpolated estimate.', '',
              '| Color | Radius | Test mean | 20th / 80th percentile | Minus yellow | Minus red |',
              '|---|---:|---:|---:|---:|---:|']
    for row in comparisons:
        costs = fixed_index[row['color'], row['radius']]
        red_delta = 'not shared' if row['test_minus_red'] is None else f'{row["test_minus_red"]:.6f}'
        lines.append(f'| {row["color"]} | {row["radius"]:g} | {row["test_mean"]:.6f} | '
                     f'{costs["p20"]:.6f} / {costs["p80"]:.6f} | '
                     f'{row["test_minus_yellow"]:.6f} | {red_delta} |')
    for color in COLORS:
        for reference, counts in comparison['counts'][color].items():
            lines.append(f'\n{color} vs {reference}: lower at {counts["lower"]}/{counts["n"]} '
                         f'compared radii, equal at {counts["equal"]}, higher at {counts["higher"]}. '
                         'Descriptive counts only; not significance tests.')
    lines += ['', '## Supplied validation-selected policies', '',
              '| Color | Seed | Radius | Validation mean | Test mean | Returned / stop candidate | Terminal status |',
              '|---|---:|---:|---:|---:|---|---|']
    for color, stats in selected_stats.items():
        for row in stats['rows']:
            lines.append(f'| {color} | {row["seed"]} | {row["radius"]:g} | '
                         f'{row["validation_mean"]:.6f} | {row["test_mean"]:.6f} | '
                         f'{row["selected_iteration"]} / {row["stop_candidate"]} | {row["terminal_status"]} |')
        lines.append(f'\n{color} selected test costs (n=10): mean / median / min / max = '
                     + ' / '.join(f'{stats["test"][k]:.6f}' for k in ('mean', 'median', 'min', 'max')) + '.')
    cap = capped[0]
    lines += ['', '## Capped configuration (not converged)', '',
              f'Black seed 9, radius 0.7: returned candidate {cap["selected_iteration"]}; '
              f'stop candidate {cap["stop_candidate"]}; {cap["mp_count"]} MPs; '
              f'test mean {cap["test_mean"]:.6f}. Status remains iteration_cap independently '
              'of the earlier return or whether this radius was validation-selected.', '',
              '## Runtime overview', '',
              'Durations below are seconds / minutes. Historical source nominal training used 10 workers; '
              'source CCG used 16 workers, one thread each. New evaluations use one worker and one thread. '
              'Saved historical environment metadata are retained; unrecorded historical machine load is unknown.', '',
              'Current environment: ' + json.dumps(summary['environment'], sort_keys=True), '',
              '| Scope | Wall time |', '|---|---|',
              '| Fresh green protocol end-to-end | unavailable (not run) |',
              '| Fresh black protocol end-to-end | unavailable (not run) |',
              '| SRO-FA partition / solver / evaluation / total | unavailable; rerun required |',
              '| New nominal training / AP / MP / CV / benchmark construction | not run |',
              '| Current total, including reporting | pending; caller finalizes summary.json |']
    for key, seconds in runtime['source_campaign'].items():
        lines.append(f'| Historical campaign {key} (separate, not additive) | {_human(seconds)} |')
    for split, record in evaluation.items():
        lines.append(f'| Current {split}: sum of 3 actual calls (not whole phase) | '
                     f'{_human(record["current_record_sum_s"])} |')
    for name, phase in runtime['current_phases_snapshot'].items():
        seconds = phase.get('wall_time_s') if phase['status'] == 'completed' else None
        lines.append(f'| Current phase {name}: {phase["status"]} (snapshot, not additive) | '
                     f'{_human(seconds)} |')
    lines += ['', '## Historical selected nominal training (not full LR search)', '',
              '| Seed | Epochs evaluated | Training seconds / minutes |', '|---:|---:|---|']
    for row in nominal:
        lines.append(f'| {row["seed"]} | {row["epochs_evaluated"]} | {_human(row["wall_time_s"])} |')
    lines += ['', '## Runtime statistics', '',
              'Each row reports n and mean / median / min / max. Seconds are stored in JSON; '
              'counts are dimensionless. Capped and nominal-reference cohorts stay separate. '
              'Black source wall time is historical parent-trajectory time, not a new min-AP total. '
              'Per-candidate AP/MP phases, identities, source timestamps and persisted evaluation durations '
              'are retained in runtime_comparison.json; repeated cached evaluations are not summed.', '',
              '| Cohort | Metric | n | Mean / median / min / max (s unless count) | Mean minutes |',
              '|---|---|---:|---|---:|']

    def statistics_line(label, metric, stats, is_time=True):
        values = ('unavailable' if not stats['n'] else
                  ' / '.join(f'{stats[k]:.6f}' for k in ('mean', 'median', 'min', 'max')))
        minutes = f'{stats["mean"] / 60:.6f}' if stats['n'] and is_time else '—'
        lines.append(f'| {label} | {metric} | {stats["n"]} | {values} | {minutes} |')

    statistics_line('Historical nominal selected winners', 'training_wall_s',
                    runtime['nominal']['training_wall_s_statistics'])
    statistics_line('Historical nominal selected winners', 'epochs_evaluated',
                    runtime['nominal']['epochs_statistics'], False)
    for split, record in evaluation.items():
        statistics_line(f'Current {split} calls', 'wall_time_s', record['current_wall_s_statistics'])
    for group in groups:
        label = f'{group["color"]}, radius {group["radius"]:g}, {group["status"]}/{group["terminal_status"]}'
        for metric, stats in group['metrics'].items():
            statistics_line(label, metric, stats, metric != 'mp_count')
    lines += ['', '## Artifacts', '',
              '- comparison.png: complete fixed-radius curves and unchanged references.',
              '- plot_data.json: all plotted numerical values and original test-array metadata.',
              '- runtime_comparison.json: raw timing evidence, source hashes, status-separated statistics.',
              '- summary.json: caller-owned final invocation timing and provenance.', '']

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = {name: output / name for name in (
        'RESULTS.md', 'runtime_comparison.json', 'plot_data.json', 'comparison.png')}
    _require(all(not path.is_symlink() for path in files.values()),
             'Output artifact must not redirect writes outside output')
    # Serialize with strict JSON before publication; never substitute for NaN.
    runtime_text = json.dumps(runtime, indent=2, allow_nan=False) + '\n'
    plot_text = json.dumps(data, indent=2, allow_nan=False) + '\n'
    _render(files['comparison.png'], data)
    files['runtime_comparison.json'].write_text(runtime_text, encoding='utf-8')
    files['plot_data.json'].write_text(plot_text, encoding='utf-8')
    files['RESULTS.md'].write_text('\n'.join(lines), encoding='utf-8')
    return dict(files_written={k: str(v) for k, v in files.items()}, fixed_radius=fixed,
                selected_cost_statistics=selected_stats,
                descriptive_reference_comparison=comparison,
                capped_configurations=deepcopy(runtime['capped_configurations']),
                runtime_statistics=deepcopy(groups), caveats=list(CAVEATS))