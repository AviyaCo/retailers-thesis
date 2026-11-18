"""Describe persisted shipping runtimes; never train, solve, or alter inputs."""
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
STOP = BASE / 'shipping_stopping_20261005/production_v2'
started = perf_counter()
started_utc = datetime.now(timezone.utc).isoformat()
sources = {}


def read(path, expected=None):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None:
        assert digest == expected, path
    sources[str(path)] = digest
    return json.loads(data)


manifest = read(STOP / 'manifest.json',
                'a0fc73c005be77c512a33ae7b6b842ba8c5ea8745a92dccaab1845754d26c619')
summary = read(STOP / 'summary.json')
plot_path = STOP / 'published/plot_data.json'
plot = read(plot_path, summary['published'][str(plot_path)])


def bound(source):
    binding = manifest['files'][str(Path(source).resolve())]
    return read(binding['copy'], binding['sha256'])


def stats(values):
    assert values and all(math.isfinite(x) and x >= 0 for x in values)
    return dict(count=len(values), mean_s=mean(values), median_s=median(values),
                min_s=min(values), max_s=max(values))


rows, nominal, continuations = [], [], []
groups = defaultdict(list)
evaluation = defaultdict(list)
for panel in plot['panels']:
    n = panel['Ntotal']
    for method in ('legacy', 'green'):
        curve = panel[method]
        for radius, seeds in zip(curve['radius'], curve['provenance']['per_seed']):
            assert sorted(r['seed'] for r in seeds) == list(range(10))
            for row in seeds:
                assert row['status'] == 'completed'
                ref = row['training_runtime_reference']
                item = {k: row[k] for k in ('id', 'Ntotal', 'Ntrain', 'seed', 'radius')}
                item.update(method=method, training_runtime_reference=ref,
                            evaluation={k: row[k] for k in
                                        ('wall_time_s', 'start_utc', 'end_utc', 'phases',
                                         'worker_count', 'solver_threads', 'torch_threads')})
                evaluation[(method, n, radius)].append(item['evaluation'])
                if radius == 0:
                    nominal.append(dict(method=method, Ntotal=n, seed=row['seed'],
                                        timing=ref['nominal']))
                if method == 'green':
                    source = bound(ref['source'])
                    assert source['converged'] and source['complete']
                    wall = ref['current_invocation']['wall_time_s']
                    assert wall == source['wall_time_s']
                    cohort = 'completed_single_invocation'
                    if ref['cached']:
                        cohort = 'completed_continuation_current_only'
                        prior = ref['cached']['cached_prior_job_timing']
                        continuations.append(dict(Ntotal=n, radius=radius, seed=row['seed'],
                            prior_failed_wall_s=prior['wall_time_s'], current_wall_s=wall,
                            summed_active_invocation_wall_s=prior['wall_time_s'] + wall,
                            scope='Sum of two measured invocations, including prior failure; '
                                  'not continuous elapsed time and not a completed-only mean.'))
                    item.update(iteration_count=len(ref['iterations']),
                                mp_count=source['mp_count'], source_status=source['status'])
                elif radius:
                    source = bound(ref['selector_source'])
                    assert source['completed']
                    wall = source['total_time_s']
                    assert wall == ref['ccg']['total_time_s']
                    trajectory = source['ccg_trajectory']
                    # Zero AP timing or zero non-terminal MP timing is not proof of zero work.
                    # Retain this evidence separately rather than repairing historical totals.
                    partial = any(a.get('ap_time_s') == 0 or
                                  (a.get('mp_time_s') == 0 and not a.get('converged')) or
                                  'cached' in str(a.get('ap_status', '')).lower() or
                                  'cached' in str(a.get('mp_status', '')).lower()
                                  for a in trajectory)
                    cohort = 'converged' if source['converged'] else 'not_converged'
                    cohort += '_partial_or_cache_timed' if partial else '_no_cache_flag'
                    if source.get('resumed'):
                        cohort += '_resumed'
                    item.update(iteration_count=len(trajectory),
                                source_status=source.get('status'),
                                converged=source['converged'], completed=source['completed'],
                                resumed=source.get('resumed'),
                                current_invocation_time_s=source.get('current_invocation_time_s'),
                                resume_previous_total_time_s=source.get('resume_previous_total_time_s'),
                                started_at_utc=source.get('started_at_utc'),
                                ended_at_utc=source.get('ended_at_utc'))
                else:
                    rows.append(item)
                    continue
                item.update(cohort=cohort, reported_ccg_wall_s=wall)
                groups[(method, n, radius, cohort)].append(wall)
                rows.append(item)

assert len(rows) == 750 and len(nominal) == 60 and len(continuations) == 5
aggregates = [dict(method=m, Ntotal=n, radius=r, cohort=c, **stats(v))
              for (m, n, r, c), v in sorted(groups.items())]
eval_aggregates = [dict(method=m, Ntotal=n, radius=r,
                       total=stats([v['wall_time_s'] for v in values]),
                       oos=stats([v['phases']['oos']['wall_time_s'] for v in values]),
                       own_pool=stats([v['phases']['is']['wall_time_s'] for v in values]))
                   for (m, n, r), values in sorted(evaluation.items())]
nominal_aggregates = []
for n in (50, 200, 800):
    for method in ('legacy', 'green'):
        selected = [x['timing'] for x in nominal if x['Ntotal'] == n and x['method'] == method]
        values = ([x['training_timing']['wall_time_s'] for x in selected]
                  if method == 'green' else [x['seed_time_s'] for x in selected if 'seed_time_s' in x])
        nominal_aggregates.append(dict(Ntotal=n, method=method,
            scope='Selected fit only' if method == 'green' else 'SAA + CV + final fit + hard evaluation',
            statistics=stats(values) if values else None))

campaigns = []
for directory, version in (
    ('shipping_nominal_R3_N50_N200_N800_20260924', 'production_v1'),
    ('shipping_live_green_20260926', 'production_v1'),
    ('shipping_advisory_continue_20260928', 'continuation_v1'),
    ('shipping_comparison_20260928', 'production_v1'),
    ('shipping_stopping_20261005', 'production_v2'),
):
    record = read(BASE / directory / version / 'summary.json')
    invocation = read(BASE / directory / f'{version}.invocation.json')
    phases = {k: {a: b for a, b in v.items() if not isinstance(b, (dict, list))}
              for k, v in record.get('phases', {}).items()}
    campaigns.append(dict(experiment=directory, invocation=invocation, phases=phases,
                          runner_wall_s=record['wall_time_s'], status=record['status'],
                          workers=record.get('workers', record.get('concurrency'))))

lines = ['# Shipping: blue versus green historical runtimes', '',
    'Read-only analysis of the frozen October 5 STOP comparison. No retraining, solver run, '
    'test-driven selection, or source/result modification. Seeds 0–9. Stored units are seconds.', '',
    '## Scope and limitations', '',
    '- Blue = historical Clarabel neural training; green = Moreau. AP uses Gurobi, not Clarabel.',
    '- Training samples differ: blue 50/200/800, green 40/160/640. Blue N800 uses corrected '
    'minibatch MP (batch 200). Stopping/return rules, optimizers and execution contention differ.',
    '- Green CCG job wall includes loading and native own-pool assessments, but excludes cached '
    'nominal training and held-out OOS replay. Green uses monotonic wall timers.',
    '- Blue totals are reported historical timings, not certified complete end-to-end runtimes. '
    'Old code times cache loading as work and can retain rounded zeros. Absence of a cache flag '
    'does not prove a cache-free history. Historical clock choice is not certified for every run.',
    '- Resume-only times, partial/cache-timed histories and non-converged histories are separate '
    'cohorts below. No missing work is estimated. Reruns would be needed for missing timings.',
    '- Small green radii through 0.03 perform zero MP updates and reuse the nominal policy. '
    'Those times are stopping checks, not fresh neural training.',
    '- Host recorded for green/replay: gold-srv2.iem.technion.ac.il, Linux. Green CCG: '
    '62 single-thread workers; continuation: five. Nominal fits: 52 workers. Replay: 16 workers, '
    'one Torch/Gurobi thread each. Historical blue concurrency is not uniform or fully established; '
    'the N800 launch records include a 24-worker parallel continuation. Exact hardware equivalence '
    'and CPU model are not established. Do not interpret these differences as solver-only speedups.', '',
    '## Main comparison: mean reported CCG minutes (number of runs)', '',
    'Blue is shown only when all ten records converged and have no explicit partial/cache timing '
    'flag or resume flag. This is a conservative display rule, not a timing-completeness certificate. '
    'Green shows uninterrupted completed runs; five continuations are detailed separately. '
    'Dashes are deliberately not replaced by misleading mixed-cohort means.', '',
    '| Radius | N50 blue | N50 green | N200 blue | N200 green | N800 blue | N800 green |',
    '|---:|---:|---:|---:|---:|---:|---:|']
for radius in (0, .001, .003, .01, .03, .2, .4, .7, 1.5):
    cells = []
    for n in (50, 200, 800):
        for method, cohort in [('legacy', 'converged_no_cache_flag'),
                               ('green', 'completed_single_invocation')]:
            values = groups.get((method, n, radius, cohort), [])
            cells.append(f'{mean(values)/60:.2f} ({len(values)})'
                         if values and (method == 'green' or len(values) == 10) else '—')
    lines.append('| ' + ' | '.join([f'{radius:g}'] + cells) + ' |')

lines += ['', 'Radius zero green is the nominal-policy CCG check, not nominal training. '
          'Blue has no CCG job at zero and no matching 0.03 radius.', '',
          '## Nominal timing (different scopes; do not form speedup ratios)', '',
          '| N | Method | Scope | Runs | Mean s (min) | Median s | Min s | Max s |',
          '|---:|---|---|---:|---:|---:|---:|---:|']
for a in nominal_aggregates:
    s = a['statistics']
    cells = ([str(s['count']), f"{s['mean_s']:.3f} ({s['mean_s']/60:.3f})",
              f"{s['median_s']:.3f}", f"{s['min_s']:.3f}", f"{s['max_s']:.3f}"]
             if s else ['0', 'unavailable', '—', '—', '—'])
    lines.append('| ' + ' | '.join([str(a['Ntotal']), a['method'], a['scope']] + cells) + ' |')

lines += ['', '## Five green continuations (not in uninterrupted-run means)', '',
          '| N | Radius | Seed | Prior failed job s | Continuation s | Sum of active invocation times s |',
          '|---:|---:|---:|---:|---:|---:|']
for c in continuations:
    lines.append(f"| {c['Ntotal']} | {c['radius']} | {c['seed']} | {c['prior_failed_wall_s']:.3f} | "
                 f"{c['current_wall_s']:.3f} | {c['summed_active_invocation_wall_s']:.3f} |")

lines += ['', '## Runtime coverage and parallel elapsed time', '',
    '| Component | Coverage |', '|---|---|',
    '| Nominal | Green selected-fit times above; full initialization and 90-fit campaign below. '
    'Blue N50 unavailable; N200/N800 bundled SAA/CV/final-fit time, not isolated fit time. |',
    '| Robust CCG | All saved timing references and AP/MP iteration fields retained in '
    '[comparison.json](comparison.json); cohorts below. |',
    '| SRO-FA partition / optimization / evaluation / total | Not run in this comparison. '
    'No comparable complete phase/total measurements established here; cached reference curve '
    'is not a measured zero-cost run. |',
    '| CV / selection | Blue separate CV duration unavailable; included in N200/N800 seed totals. '
    'Green nominal LR validation stage below, not five-fold CV. No new CV/radius selection. |',
    '| Final OOS / IS | Green Oct5, blue cached Sept28 measurements. Same Gurobi replay, '
    'not native Moreau-vs-Clarabel inference. Separate phase statistics below. |', '',
    'Parallel elapsed times below are observed campaign/phase durations, not sums of worker times. '
    'The original green CCG campaign includes five failed jobs; the continuation is separate. '
    'The September replay includes both colors, so its elapsed time is not a blue-only runtime.', '',
    '| Campaign | Runner seconds (minutes) | Phase elapsed seconds |', '|---|---:|---|']
for c in campaigns:
    phases = '; '.join(f"{k}: {v['wall_time_s']:.3f}" for k, v in c['phases'].items() if 'wall_time_s' in v)
    lines.append(f"| {c['experiment']} | {c['runner_wall_s']:.3f} ({c['runner_wall_s']/60:.3f}) | {phases} |")

lines += ['', '## All CCG cohorts (seconds; failed/partial/resumed kept separate)', '',
          '| Method | N | Radius | Cohort | Runs | Mean | Median | Min | Max |',
          '|---|---:|---:|---|---:|---:|---:|---:|---:|']
for a in aggregates:
    lines.append(f"| {a['method']} | {a['Ntotal']} | {a['radius']:g} | {a['cohort']} | {a['count']} | "
                 f"{a['mean_s']:.3f} | {a['median_s']:.3f} | {a['min_s']:.3f} | {a['max_s']:.3f} |")
lines += ['', '## Replay evaluation by method, sample size and radius (seconds)', '',
          '| Method | N | Radius | Runs | Total mean | Median | Min | Max | OOS mean | Own-pool mean |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for a in eval_aggregates:
    s = a['total']
    lines.append(f"| {a['method']} | {a['Ntotal']} | {a['radius']:g} | {s['count']} | {s['mean_s']:.3f} | "
                 f"{s['median_s']:.3f} | {s['min_s']:.3f} | {s['max_s']:.3f} | "
                 f"{a['oos']['mean_s']:.3f} | {a['own_pool']['mean_s']:.3f} |")
lines += ['', 'Raw per-seed timing references, per-iteration AP/MP fields, available timestamps, '
          'workers, statuses, nominal records, campaign receipts and input SHA-256 bindings: '
          '[comparison.json](comparison.json). Reproduction: run [compare.py](compare.py) with '
          'the existing ap_env Python. No external package or solver is required.', '']

artifact = dict(experiment='shipping_runtime_comparison_20261005', unit='seconds',
    source_hashes=sources, per_policy=rows, nominal=nominal, nominal_aggregates=nominal_aggregates,
    ccg_aggregates=aggregates, evaluation_aggregates=eval_aggregates,
    continuations=continuations, campaigns=campaigns,
    analysis=dict(started_at_utc=started_utc, ended_at_utc=datetime.now(timezone.utc).isoformat(),
                  wall_time_s=perf_counter()-started,
                  scope='Read-only reporting overhead through assembly, excludes output writes; not model runtime'))
(ROOT / 'comparison.json').write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')
(ROOT / 'RESULTS.md').write_text('\n'.join(lines))
print('\n'.join(lines[:55]))
print(f'Validated {len(rows)} policies; {len(sources)} input bindings. Report: {ROOT / "RESULTS.md"}')