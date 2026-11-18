"""Fixed 90-job launcher; no refit, test-set access, hydro, or resume.

Run from this directory with Python run_ccg.py (external tmux/log ownership).
Python run_ccg.py --self-test runs mocked launcher tests, never an experiment.
Raw controller summaries/assessments/checkpoints remain in each job directory;
only compact evidence references and timing records cross process boundaries.
"""
import os

# Set before numerical imports, also when spawn imports this module afresh.
THREAD_ENV = {name: '1' for name in (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
import math
import multiprocessing
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

import ccg
import core
import torch

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'ccg_results'
MAX_WORKERS = 16
TERMINALS = ('separation_satisfied', 'stagnation', 'iteration_cap')
SEPARATIONS = ('separation_satisfied', 'violating')


def job_key(seed, radius):
    return f'seed_{seed}_radius_{radius:g}'


def error_info(error):
    return dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())


def snapshot(source, destination, expected=None):
    """Copy bytes, not links; source path/mode/content are never modified."""
    source, destination = Path(source).resolve(), Path(destination)
    before = core.digest(source)
    if expected is not None and before != expected:
        raise ValueError(f'Frozen input hash mismatch: {source}')
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    if core.digest(destination) != before or core.digest(source) != before:
        raise ValueError(f'Source changed while snapshotting: {source}')
    destination.chmod(0o444)
    return dict(source=str(source), snapshot=str(destination.resolve()), sha256=before)


def verify_provenance(manifest):
    for path, expected in manifest['source_sha256'].items():
        if core.digest(Path(path)) != expected:
            raise ValueError(f'Launch source/dependency changed: {path}')
    for entry in manifest['snapshots']:
        for key in ('source', 'snapshot'):
            if core.digest(Path(entry[key])) != entry['sha256']:
                raise ValueError(f'Launch input changed: {entry[key]}')


def freeze_provenance(output):
    """Extend, never rewrite, the original LR manifest. No model is evaluated."""
    folder = Path(output) / 'provenance'
    folder.mkdir(exist_ok=False)
    entries = []

    def copy(path, expected=None):
        entry = snapshot(path, folder / f'{len(entries):03d}_{Path(path).name}', expected)
        entries.append(entry)
        return Path(entry['snapshot'])

    lr_manifest_path = ccg.LR_SEARCH_ROOT / 'provenance.json'
    lr_manifest = json.loads(copy(lr_manifest_path).read_text())
    lr_hash = entries[-1]['sha256']
    lr_summary = json.loads(copy(ccg.LR_SEARCH_ROOT / 'summary.json').read_text())
    if lr_summary['status'] != 'completed' or lr_summary['provenance_sha256'] != lr_hash:
        raise ValueError('Require completed LR search with unchanged original manifest')
    sources = dict(lr_manifest['source_sha256'])
    for path, expected in sources.items():
        if core.digest(Path(path)) != expected:
            raise ValueError(f'Frozen LR source changed: {path}')
    for path in (Path(core.__file__), Path(ccg.adversary.__file__), Path(ccg.__file__),
                 Path(__file__), ROOT / 'run_lr_search.py', core._BASELINE_PATH, core._loader.SOURCE):
        path = Path(path).resolve()
        copy(path, sources.get(str(path)))
        sources[str(path)] = entries[-1]['sha256']
    rows = lr_summary['seed_results']
    if len(rows) != len(core.SEEDS) or {r['seed'] for r in rows} != set(core.SEEDS):
        raise ValueError('Incomplete/duplicate nominal seed coverage')
    selected = []
    for seed in core.SEEDS:
        nominal = next(r for r in rows if r['seed'] == seed)
        seed_root = ccg.LR_SEARCH_ROOT / f'seed_{seed}'
        stored = json.loads(copy(seed_root / 'summary.json').read_text())
        if stored != nominal or not nominal['complete'] or nominal['status'] != 'completed':
            raise ValueError(f'Incomplete/inconsistent nominal seed {seed}')
        configs = nominal['configurations']
        if (len(configs) != len(core.LR_GRID)
                or {r['lr_net'] for r in configs} != set(core.LR_GRID)
                or any(r['status'] != 'completed' or not finite_number(
                    r['restored_validation_evaluation']['mean']) for r in configs)):
            raise ValueError(f'Incomplete nominal LR grid for seed {seed}')
        best = min(configs, key=lambda r: (r['restored_validation_evaluation']['mean'], r['lr_net']))
        winner = nominal['winner']
        if (winner['lr_net'], winner['validation_mean'], winner['selected_sha256']) != (
                best['lr_net'], best['restored_validation_evaluation']['mean'], best['selected_sha256']):
            raise ValueError(f'Invalid nominal LR winner for seed {seed}')
        path = seed_root / f"lr_{winner['lr_net']:g}" / 'selected.pth'
        if path.resolve() != Path(winner['checkpoint']).resolve():
            raise ValueError(f'Nominal checkpoint path mismatch for seed {seed}')
        copy(path, winner['selected_sha256'])
        copy(seed_root / 'split.json', nominal['split_artifact_sha256'])
        selected.append(dict(seed=seed, **winner))
    versions = {name: version(name) for name in lr_manifest['versions']}
    if versions != lr_manifest['versions']:
        raise ValueError('Dependency versions differ from frozen LR search')
    manifest = dict(source_sha256=sources, snapshots=entries, versions=versions,
        native_dependency=lr_manifest['native_dependency'], original_lr_manifest_sha256=lr_hash,
        selected_lrs=selected, python=sys.version, executable=sys.executable,
        controller_api='ccg.run_configuration(seed, radius, output_root, provenance_hash=None)',
        function_source=str(Path(ccg.__file__).resolve()),
        cached_lr_search=dict(wall_time_s=lr_summary['wall_time_s'],
            seed_workers=lr_summary['seed_workers'], started_at_utc=lr_summary['started_at_utc'],
            ended_at_utc=lr_summary['ended_at_utc'], scope='previous invocation, never current sweep time'),
        protocol=dict(Ntotal=50, Ntrain=40, Nval=10, data_seeds=list(core.SEEDS),
            split_seeds=[100000 + seed for seed in core.SEEDS], network_seed=0,
            radii=list(core.RADII), selection='all nine completed; minimum selected validation mean, smaller radius tie',
            test='not accessed', refit='not performed', hydro='not run'))
    verify_provenance(manifest)
    core.write_json(Path(output) / 'provenance.json', manifest)
    (Path(output) / 'provenance.json').chmod(0o444)
    return manifest


def configure_worker(output, provenance_hash):
    os.environ.update(THREAD_ENV)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = Path(output) / 'provenance.json'
    if core.digest(path) != provenance_hash:
        raise ValueError('Launch manifest hash mismatch')
    verify_provenance(json.loads(path.read_text()))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def completed_trajectory(row):
    return (row.get('status') == 'completed' and row.get('configuration_status') == 'completed'
            and row.get('complete') is True and row.get('terminal_status') in TERMINALS
            and row.get('selected_assessed') is True
            and row.get('selected_separation_status') in SEPARATIONS
            and finite_number(row.get('selected_validation_mean'))
            and all(row.get(key) is not None for key in ('selected_candidate_id',
                'selected_checkpoint', 'selected_state_sha256', 'selected_ap_reference')))


def compact_result(raw):
    """Explicit whitelist: no learned state, AP sample arrays, or epoch histories."""
    keys = ('experiment', 'method', 'seed', 'radius', 'Ntotal', 'Ntrain', 'Nval',
            'split_seed', 'network_seed', 'lr_net', 'max_mp', 'max_epochs', 'patience',
            'lr_inventory', 'status', 'configuration_status', 'complete', 'terminal_status',
            'terminal_reason', 'selected_candidate_id', 'selected_checkpoint',
            'selected_validation_mean', 'selected_state_sha256', 'selected_ap_reference',
            'selected_separation_status', 'provisional_best', 'mp_count',
            'ap_sweeps_attempted', 'ap_sweeps_completed', 'wall_time_s', 'wall_time_human',
            'ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s',
            'input_verification_and_loading', 'selection_and_checkpoint',
            'started_at_utc', 'ended_at_utc', 'summary_path', 'output_directory',
            'provenance_sha256', 'error', 'finalization_error', 'torch_threads',
            'torch_interop_threads', 'gurobi_threads', 'ap_workers_per_job')
    result = {key: raw[key] for key in keys if key in raw}
    candidates = raw.get('iterations', [])
    selected = [r for r in candidates if r['candidate_id'] == raw.get('selected_candidate_id')]
    result['selected_assessed'] = (len(selected) == 1 and selected[0].get('assessed') is True
        and all(selected[0].get(a) == raw.get(b) for a, b in (
            ('validation_mean', 'selected_validation_mean'), ('state_sha256', 'selected_state_sha256'),
            ('ap_reference', 'selected_ap_reference'), ('separation_status', 'selected_separation_status'))))
    # Persist the small per-iteration timing/iteration counts, not the sample payload.
    result['iteration_timings'] = [dict(candidate_id=r['candidate_id'],
        assessed=r.get('assessed', False), separation_status=r.get('separation_status'),
        ap_reference=r.get('ap_reference'), ap_total_time_s=r.get('ap_total_time_s'),
        mp_total_time_s=r.get('mp_total_time_s'), evaluation=r.get('evaluation'),
        ap_replay_time_s=measured_sum([s.get('replay', {}).get('wall_time_s')
                                     for s in r.get('samples', [])]),
        mp={key: r.get('mp', {}).get(key) for key in ('status', 'wall_time_s',
            'started_at_utc', 'ended_at_utc', 'epochs_evaluated', 'optimizer_steps',
            'validation_time_s', 'optimizer_setup_time_s', 'stop_reason')}) for r in candidates]
    if 'cached_nominal' in raw:
        cached = raw['cached_nominal']
        result['cached_nominal'] = {k: v for k, v in cached.items() if k != 'selected_training'}
        result['cached_nominal']['selected_training'] = {k: cached['selected_training'].get(k)
            for k in ('status', 'wall_time_s', 'started_at_utc', 'ended_at_utc',
                      'epochs_evaluated', 'optimizer_steps', 'validation_time_s', 'stop_reason')}
    result['raw_summary_sha256'] = core.digest(Path(raw['summary_path']))
    return core._json_value(result)


def run_job(seed, radius, output, provenance_hash):
    """Worker wrapper preserves controller evidence even on exceptional failure."""
    start = time.perf_counter()
    lifecycle = dict(seed=seed, radius=radius, status='running', pid=os.getpid(),
        started_at_utc=core.utc_now(), started_monotonic_s=start,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        thread_environment={name: os.environ.get(name) for name in THREAD_ENV},
        gurobi_threads=1, ap_workers_per_job=1, job_workers=MAX_WORKERS)
    path = Path(output) / 'jobs' / f'{job_key(seed, radius)}.lifecycle.json'
    core.write_json(path, lifecycle)
    try:
        raw = ccg.run_configuration(seed, radius, output, provenance_hash=provenance_hash)
        result = compact_result(raw)
        if result['seed'] != seed or result['radius'] != radius:
            raise ValueError('Controller returned a mismatched configuration')
        if not completed_trajectory(result):
            result.update(status='failed', complete=False, launcher_error=dict(
                type='IncompleteTrajectory', message='Controller failed or selected policy not fully assessed'))
    except Exception as error:
        result = dict(seed=seed, radius=radius, status='failed', complete=False,
            launcher_error=error_info(error), summary_path=str(
                ccg.configuration_directory(output, seed, radius) / 'summary.json'))
    finally:
        lifecycle.update(ended_monotonic_s=time.perf_counter(), ended_at_utc=core.utc_now())
        lifecycle['wall_time_s'] = lifecycle['ended_monotonic_s'] - start
    result['launcher_execution'] = lifecycle.copy()
    result['launcher_execution']['status'] = result['status']
    # Worker also persists compact evidence before returning it through the future.
    core.write_json(Path(output) / 'jobs' / f'{job_key(seed, radius)}.worker.json', result)
    lifecycle['status'] = result['status']
    core.write_json(path, lifecycle)
    return result


def select_seed(seed, records, *, final_provenance_verified=False):
    rows = [row for row in records if row['seed'] == seed]
    eligible = [row for row in rows if completed_trajectory(row)]
    covered = Counter(row['radius'] for row in rows)
    ready = (covered == Counter(core.RADII) and len(eligible) == len(core.RADII))
    final = ready and final_provenance_verified is True
    best = min(eligible, key=lambda r: (r['selected_validation_mean'], r['radius'])) if eligible else None
    policy = None if best is None else {key: best[key] for key in (
        'seed', 'radius', 'selected_candidate_id', 'selected_validation_mean', 'selected_checkpoint',
        'selected_state_sha256', 'selected_ap_reference', 'selected_separation_status', 'terminal_status')}
    warning = None if final else (
        'PROVISIONAL: incomplete/failed/duplicate radius grid; no final selection' if not ready else
        'PROVISIONAL: end-of-sweep provenance verification pending/failed; no final selection')
    return dict(seed=seed, final_selection=policy if final else None, best_known=policy,
        provisional=not final, eligible_radius_count=len(eligible), warning=warning)


def timing_stats(values, expected):
    known = [value for value in values if finite_number(value) and value >= 0]
    return dict(**core.stats(known), unavailable_count=expected - len(known),
                unavailable_reason='unmeasured/invalid fields require rerun; never substituted' if len(known) < expected else None)


def measured_sum(values):
    """Sequential phase totals only; missing measurements stay unavailable."""
    return sum(values) if all(finite_number(v) and v >= 0 for v in values) else None


def summarize(records):
    groups = []
    for radius in core.RADII:
        rows = [r for r in records if r['radius'] == radius and completed_trajectory(r)]
        phases = {key: [r.get(key) for r in rows] for key in
            ('wall_time_s', 'ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s')}
        phases['input_setup_time_s'] = [r.get('input_verification_and_loading', {}).get('wall_time_s') for r in rows]
        phases['selection_checkpoint_time_s'] = [r.get('selection_and_checkpoint', {}).get('wall_time_s') for r in rows]
        for phase, key in (('mp_validation_time_s', 'validation_time_s'),
                           ('mp_optimizer_setup_time_s', 'optimizer_setup_time_s')):
            phases[phase] = [measured_sum([it['mp'].get(key) for it in r['iteration_timings']
                if it['mp'].get('status') is not None]) if 'iteration_timings' in r else None for r in rows]
        phases['ap_native_replay_time_s'] = [measured_sum([it['ap_replay_time_s']
            for it in r['iteration_timings']]) if 'iteration_timings' in r else None for r in rows]
        groups.append(dict(method='robust_neural_CCG' if radius else 'cached_nominal_current_radius0_assessment',
            Ntotal=50, Ntrain=40, Nval=10, radius=radius, seeds=[r['seed'] for r in rows],
            count=len(rows), terminal_status_counts=dict(Counter(r['terminal_status'] for r in rows)),
            selected_separation_status_counts=dict(Counter(r['selected_separation_status'] for r in rows)),
            joint_status_counts=dict(Counter(f"{r['terminal_status']} / {r['selected_separation_status']}" for r in rows)),
            runtime_s={phase: timing_stats(values, len(rows)) for phase, values in phases.items()}))
    # Historical nominal timings appear once per seed, never once per radius.
    nominal = []
    for seed in core.SEEDS:
        matches = [r for r in records if r['seed'] == seed and completed_trajectory(r) and 'cached_nominal' in r]
        if matches:
            cached = matches[0]['cached_nominal']
            if any(r['cached_nominal'] != cached for r in matches):
                raise ValueError(f'Inconsistent cached nominal timing for seed {seed}')
            nominal.append(dict(seed=seed, **cached))
    cached_phases = {
        'selected_nominal_training': [r['selected_training'].get('wall_time_s') for r in nominal],
        'nominal_train_evaluation': [r['selected_train_evaluation'].get('wall_time_s') for r in nominal],
        'nominal_validation_evaluation': [r['selected_validation_evaluation'].get('wall_time_s') for r in nominal],
        'nominal_training_validation': [r['selected_training'].get('validation_time_s') for r in nominal],
        'saa_setup': [r['saa_initialization'].get('wall_time_s') for r in nominal],
        'seed_lr_search': [r['seed_lr_search_wall_time_s'] for r in nominal],
        'lr_selection': [r['lr_selection'].get('wall_time_s') for r in nominal]}
    return dict(per_radius=groups, cached_nominal=dict(method='nominal_moreau_cached', Ntrain=40,
        radius=0., scope='historical persisted time, excluded from current CCG wall time',
        per_seed=nominal, runtime_s={k: timing_stats(v, len(nominal)) for k, v in cached_phases.items()}),
        failed_or_incomplete=[r for r in records if not completed_trajectory(r)],
        runtime_scope='per-job wall times overlap; overall elapsed is measured separately, not summed',
        phase_scope='AP includes solver/replay and AP persistence; MP includes its validation/setup; phases are not additive',
        not_applicable=dict(SRO_FA='not requested/not run', CV='not run; fixed 40/10 validation, not CV',
                            final_test='not accessed', refit='not performed', hydro='not run'))


def persist_progress(output, sweep, records):
    # Markers are written by executing workers, not Future.running() (which can
    # include queued work). Counts are a point-in-time observation, not telemetry.
    collected = {(r['seed'], r['radius']) for r in records}
    running = awaiting = 0
    for seed, radius in sweep['schedule']:
        if (seed, radius) in collected:
            continue
        path = Path(output) / 'jobs' / f'{job_key(seed, radius)}.lifecycle.json'
        if path.exists():
            state = json.loads(path.read_text())
            running += state['status'] == 'running'
            awaiting += state['status'] != 'running'
    sweep.update(updated_at_utc=core.utc_now(), updated_monotonic_s=time.perf_counter(),
        scheduled_count=len(sweep['schedule']), completed_count=sum(completed_trajectory(r) for r in records),
        failed_count=sum(not completed_trajectory(r) for r in records), collected_count=len(records),
        running_count=running, finished_awaiting_collection_count=awaiting,
        pending_count=len(sweep['schedule']) - len(records) - running - awaiting,
        progress_scope='worker markers sampled on submission/completion; running means started, not yet finished',
        job_results=records, selections=[select_seed(s, records,
            final_provenance_verified=sweep.get('final_provenance_verified', False)) for s in core.SEEDS],
        aggregates=summarize(records))
    core.write_json(Path(output) / 'summary.json', sweep)


def print_report(sweep):
    print('Completed trajectories only | method/radius | phase | n | mean/median/min/max seconds', flush=True)
    aggregates = sweep.get('aggregates', {})
    groups = list(aggregates.get('per_radius', []))
    if 'cached_nominal' in aggregates:
        groups.append(aggregates['cached_nominal'])
    for group in groups:
        for phase, stats in group['runtime_s'].items():
            values = ' / '.join('unavailable' if stats[k] is None else f'{stats[k]:.6f}'
                                for k in ('mean', 'median', 'min', 'max'))
            print(f"{group['method']}/{group['radius']:g} | {phase} | {stats['count']} | {values}", flush=True)
        if 'terminal_status_counts' in group:
            print(f"  terminal={group['terminal_status_counts']}; selected separation={group['selected_separation_status_counts']}", flush=True)
    print('SRO-FA / CV / final test / hydro: not run', flush=True)
    for selection in sweep.get('selections', []):
        print(f"seed={selection['seed']} {selection['warning'] or 'FINAL validation selection'}: "
              f"{selection['best_known']}", flush=True)
    seconds = sweep['wall_time_s']
    print(f"Sweep {sweep['status']}: {seconds:.3f} s / {seconds / 60:.3f} min / {seconds / 3600:.3f} h; "
          'concurrent job walls are NOT summed', flush=True)


def main():
    # Exclusive directory is deliberately outside the failure-writing block.
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / 'jobs').mkdir()
    start = time.perf_counter()
    sweep = dict(experiment='retailer_R3_N50_net16_v2_train40_val10_CCG', status='running',
        started_at_utc=core.utc_now(), started_monotonic_s=start, errors=[], final_provenance_verified=False,
        schedule=[(s, r) for s in core.SEEDS for r in core.RADII], submitted_count=0,
        command_argv=list(getattr(sys, 'orig_argv', [sys.executable, *sys.argv])),
        command=shlex.join(getattr(sys, 'orig_argv', [sys.executable, *sys.argv])),
        cwd=str(Path.cwd()), executable=sys.executable, python=sys.version,
        environment=dict(hostname=platform.node(), platform=platform.platform(),
            processor=platform.processor(), machine=platform.machine(), cpu_count=os.cpu_count(),
            cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_info=Path('/proc/cpuinfo').read_text()),
        concurrency=dict(job_workers=MAX_WORKERS, start_method='spawn', ap_workers_per_job=1,
            torch_threads=1, torch_interop_threads=1, gurobi_threads=1, thread_environment=THREAD_ENV),
        timing_unit='seconds', timing_clock='time.perf_counter', resumed=False, nominal_retrained=False)
    records = []
    try:
        persist_progress(OUTPUT, sweep, records)
        setup = sweep['provenance_setup'] = {}
        with core.timed(setup):
            manifest = freeze_provenance(OUTPUT)
        provenance_hash = core.digest(OUTPUT / 'provenance.json')
        sweep['provenance_sha256'] = provenance_hash
        persist_progress(OUTPUT, sweep, records)
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=multiprocessing.get_context('spawn'),
                initializer=configure_worker, initargs=(OUTPUT, provenance_hash)) as pool:
            futures = {}
            for seed, radius in sweep['schedule']:
                try:
                    futures[pool.submit(run_job, seed, radius, OUTPUT, provenance_hash)] = (seed, radius)
                    sweep['submitted_count'] += 1
                except Exception as error:
                    row = dict(seed=seed, radius=radius, status='not_submitted', complete=False,
                               launcher_error=error_info(error))
                    records.append(row)
                    sweep['errors'].append(dict(seed=seed, radius=radius, phase='submit', **error_info(error)))
                    try:
                        core.write_json(OUTPUT / 'jobs' / f'{job_key(seed, radius)}.json', row)
                    except Exception as write_error:
                        sweep['errors'].append(dict(seed=seed, radius=radius, phase='persist_submission_failure',
                                                   **error_info(write_error)))
            try:
                persist_progress(OUTPUT, sweep, records)
            except Exception as error:
                sweep['errors'].append(dict(phase='persist_submitted', **error_info(error)))
            for future in as_completed(futures):
                seed, radius = futures.pop(future)
                try:
                    row = future.result()
                    if (row['seed'], row['radius']) != (seed, radius):
                        raise ValueError('Future returned wrong configuration')
                except Exception as error:
                    row = dict(seed=seed, radius=radius, status='failed', complete=False,
                        launcher_error=error_info(error), summary_path=str(
                            ccg.configuration_directory(OUTPUT, seed, radius) / 'summary.json'))
                    sweep['errors'].append(dict(seed=seed, radius=radius, phase='worker', **error_info(error)))
                records.append(row)
                # Even if reporting fails, collect the other futures before failing.
                try:
                    core.write_json(OUTPUT / 'jobs' / f'{job_key(seed, radius)}.json', row)
                    persist_progress(OUTPUT, sweep, records)
                except Exception as error:
                    sweep['errors'].append(dict(seed=seed, radius=radius, phase='persist', **error_info(error)))
                print(f"collected {len(records)}/90 seed={seed} radius={radius:g} status={row['status']}", flush=True)
                del future
        verify_provenance(manifest)
        # Only this successful end-of-sweep check permits publication. Unrelated
        # seed failures below do not invalidate another seed's complete grid.
        sweep['final_provenance_verified'] = True
        if sweep['errors'] or len(records) != 90 or any(not completed_trajectory(r) for r in records):
            raise RuntimeError('CCG launch failed/incomplete; all submitted jobs collected, inspect summary and raw job files')
        sweep['status'] = 'completed'
    except BaseException as error:
        sweep.update(status='failed', error=error_info(error))
        raise
    finally:
        records.sort(key=lambda r: (r['seed'], r['radius']))
        end = time.perf_counter()
        sweep.update(ended_at_utc=core.utc_now(), ended_monotonic_s=end, wall_time_s=end - start)
        persist_progress(OUTPUT, sweep, records)
        print_report(sweep)


def self_test():
    """Mocked orchestration/selection tests; temporary outputs, no optimization."""
    import contextlib
    from concurrent.futures import Future
    import io
    import tempfile
    import unittest
    from unittest.mock import patch

    module = sys.modules[__name__]

    def fixture(seed, radius, **changes):
        row = dict(seed=seed, radius=radius, status='completed', configuration_status='completed',
            complete=True, terminal_status='iteration_cap', selected_assessed=True,
            selected_separation_status='violating', selected_candidate_id=0,
            selected_checkpoint='fixture.pth', selected_state_sha256='fixture-sha',
            selected_ap_reference='fixture-assessment', selected_validation_mean=5.,
            wall_time_s=2., ap_total_time_s=1., mp_total_time_s=.5, evaluation_time_s=.1,
            input_verification_and_loading=dict(wall_time_s=.2), selection_and_checkpoint=dict(wall_time_s=.01))
        row.update(changes)
        return row

    class LauncherTests(unittest.TestCase):
        def test_missing_failed_duplicate_unassessed_nonfinite(self):
            rows = [fixture(0, r) for r in core.RADII]
            variants = [rows[:-1], rows + [rows[0]]]
            for change in (dict(status='failed'), dict(complete=False), dict(selected_assessed=False),
                           dict(selected_validation_mean=float('nan')), dict(selected_validation_mean='NaN')):
                variants.append([*rows[:-1], {**rows[-1], **change}])
            for case in variants:
                with self.subTest(case=case[-1]):
                    result = select_seed(0, case, final_provenance_verified=True)
                    self.assertIsNone(result['final_selection'])
                    self.assertTrue(result['provisional'])
                    self.assertIn('PROVISIONAL', result['warning'])

        def test_ties_and_statuses_are_independent(self):
            rows = [fixture(0, r) for r in reversed(core.RADII)]
            selected = select_seed(0, rows, final_provenance_verified=True)['final_selection']
            self.assertEqual(selected['radius'], 0.)
            self.assertEqual(selected['terminal_status'], 'iteration_cap')
            self.assertEqual(selected['selected_separation_status'], 'violating')
            for terminal in TERMINALS:
                rows[0].update(terminal_status=terminal, selected_validation_mean=1.)
                self.assertEqual(select_seed(0, rows, final_provenance_verified=True)['final_selection']['radius'], rows[0]['radius'])
                self.assertEqual(select_seed(0, rows, final_provenance_verified=True)['final_selection']['selected_separation_status'], 'violating')

        def test_only_completed_timings_and_status_counts(self):
            rows = [fixture(0, .2), fixture(1, .2, wall_time_s=4., terminal_status='stagnation'),
                    fixture(2, .2, status='failed', wall_time_s=999.)]
            result = summarize(rows)
            group = next(r for r in result['per_radius'] if r['radius'] == .2)
            self.assertEqual(group['runtime_s']['wall_time_s']['mean'], 3.)
            self.assertEqual(group['count'], 2)
            self.assertEqual(group['selected_separation_status_counts'], {'violating': 2})
            self.assertEqual(len(result['failed_or_incomplete']), 1)
            self.assertEqual(timing_stats([None], 1)['unavailable_count'], 1)
            self.assertIsNone(measured_sum([1., None]))
            self.assertEqual(measured_sum([]), 0)

        def test_progress_counts_actual_start_not_queued_future(self):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                schedule = [(s, r) for s in core.SEEDS for r in core.RADII]
                core.write_json(output / 'jobs' / f'{job_key(0, 0.)}.lifecycle.json', dict(status='running'))
                core.write_json(output / 'jobs' / f'{job_key(0, .001)}.lifecycle.json', dict(status='completed'))
                sweep = dict(schedule=schedule)
                persist_progress(output, sweep, [])
                self.assertEqual((sweep['scheduled_count'], sweep['running_count'],
                    sweep['finished_awaiting_collection_count'], sweep['pending_count']), (90, 1, 1, 88))
                self.assertTrue(all(r['provisional'] for r in sweep['selections']))

        def test_complete_grid_pending_verification_is_provisional(self):
            rows = [fixture(s, r) for s in core.SEEDS for r in core.RADII]
            self.assertIsNone(select_seed(0, rows)['final_selection'])
            self.assertIsNone(select_seed(0, rows, final_provenance_verified=False)['final_selection'])
            with tempfile.TemporaryDirectory() as tmp:
                sweep = dict(schedule=[(r['seed'], r['radius']) for r in rows],
                             status='running', wall_time_s=0.)
                persist_progress(tmp, sweep, rows)
                saved = json.loads((Path(tmp) / 'summary.json').read_text())
                for selection in saved['selections']:
                    self.assertIsNone(selection['final_selection'])
                    self.assertIsNotNone(selection['best_known'])
                    self.assertTrue(selection['provisional'])
                    self.assertIn('PROVISIONAL', selection['warning'])
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    print_report(saved)
                self.assertNotIn('FINAL', stream.getvalue())

        def test_failed_final_verification_revokes_all_final_selections(self):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'ccg_results'

                def freeze(path):
                    source = path / 'fixture-source'
                    source.write_text('intact')
                    manifest = dict(source_sha256={str(source): core.digest(source)}, snapshots=[])
                    core.write_json(path / 'provenance.json', manifest)
                    return manifest

                def submit(function, seed, radius, root, sha):
                    future = Future()
                    future.set_result(fixture(seed, radius))
                    return future

                verify = verify_provenance
                pending = {}

                def final_verify(manifest):
                    pending.update(json.loads((output / 'summary.json').read_text()))
                    self.assertEqual(pending['collected_count'], 90)
                    (output / 'fixture-source').write_text('tampered at final verification')
                    verify(manifest)

                stream = io.StringIO()
                with patch.object(module, 'OUTPUT', output), \
                        patch.object(module, 'freeze_provenance', side_effect=freeze), \
                        patch.object(module, 'verify_provenance', side_effect=final_verify) as checked, \
                        patch.object(module, 'ProcessPoolExecutor') as executor, \
                        contextlib.redirect_stdout(stream):
                    executor.return_value.__enter__.return_value.submit.side_effect = submit
                    with self.assertRaisesRegex(ValueError, 'Launch source/dependency changed'):
                        main()
                checked.assert_called_once()
                saved = json.loads((output / 'summary.json').read_text())
                self.assertEqual(saved['status'], 'failed')
                self.assertEqual(saved['completed_count'], 90)
                self.assertEqual(sum(s['final_selection'] is not None for s in saved['selections']), 0)
                self.assertTrue(all(s['final_selection'] is None for s in pending['selections']))
                self.assertFalse(saved['final_provenance_verified'])
                for selection in saved['selections']:
                    self.assertIsNone(selection['final_selection'])
                    self.assertIsNotNone(selection['best_known'])
                    self.assertTrue(selection['provisional'])
                    self.assertIn('PROVISIONAL', selection['warning'])
                self.assertNotIn('FINAL', stream.getvalue())

        def controller_fixture(self):
            # Reuse existing test fixtures; never read the frozen LR artifacts.
            from test_ccg import ControllerTests
            case = ControllerTests()
            self.addCleanup(case.doCleanups)
            case.setUp()
            case.stack.enter_context(patch.object(torch, 'get_num_threads', return_value=1))
            case.stack.enter_context(patch.object(torch, 'get_num_interop_threads', return_value=1))
            return case

        def test_one_bad_ball_prevents_separation_despite_mean_below_one_percent(self):
            case = self.controller_fixture()
            case.restore = True

            def one_bad_ball(model, center1, center2, radius, row):
                return case.ap(model, center1, center2, radius if row['sample_id'] == 3 else 0., row)

            with patch.object(ccg.adversary, 'solve_ap', side_effect=one_bad_ball):
                result = case.run_case()
            self.assertEqual(result['status'], 'completed', result.get('error'))
            first = result['iterations'][0]
            residuals = [row['residual'] for row in first['samples']]
            self.assertEqual(sum(r > ccg.SEPARATION_TOLERANCE for r in residuals), 1)
            self.assertLess(sum(residuals) / len(residuals), ccg.SEPARATION_TOLERANCE)
            self.assertLess((first['ap_bound_mean'] - first['restricted_mean']) /
                            abs(first['restricted_mean']), .01)
            self.assertEqual(first['separation_status'], 'violating')
            self.assertEqual(first['added_sample_ids'], [3])
            self.assertEqual((case.train_calls, case.ap_calls), (1, 80))
            self.assertEqual(result['terminal_status'], 'stagnation')

        def test_mp_failure_preserves_timing_and_provisional_only(self):
            case = self.controller_fixture()

            def fail_mp(model, loss, val1, val2, lr, folder, record, **kwargs):
                with core.timed(record):
                    record.update(epochs_evaluated=1, optimizer_steps=0)
                    raise RuntimeError('fixture MP failure')

            with patch.object(core, 'train_model', side_effect=fail_mp):
                result = case.run_case()
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['error']['message'], 'fixture MP failure')
            self.assertFalse(result['complete'])
            self.assertIsNone(result['selected_checkpoint'])
            self.assertEqual(result['provisional_best']['candidate_id'], 0)
            self.assertEqual((result['mp_count'], result['ap_sweeps_completed']), (0, 1))
            candidate = result['iterations'][0]
            training = candidate['mp']
            self.assertEqual(training['status'], 'failed')
            self.assertEqual((training['epochs_evaluated'], training['optimizer_steps']), (1, 0))
            self.assertGreater(training['wall_time_s'], 0.)
            self.assertTrue(training['started_at_utc'])
            self.assertTrue(training['ended_at_utc'])
            self.assertGreaterEqual(candidate['mp_total_time_s'], training['wall_time_s'])
            self.assertEqual(result['mp_total_time_s'], candidate['mp_total_time_s'])
            compact = compact_result(result)
            self.assertEqual(compact['iteration_timings'][0]['mp']['wall_time_s'], training['wall_time_s'])
            self.assertEqual(compact['mp_total_time_s'], result['mp_total_time_s'])
            assessed = json.loads(Path(candidate['ap_reference']).read_text())
            self.assertEqual(assessed['mp_total_time_s'], candidate['mp_total_time_s'])

        def test_worker_initializer_thread_settings(self):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'provenance.json'
                core.write_json(path, dict(source_sha256={}, snapshots=[]))
                with patch.object(torch, 'set_num_threads') as intra, \
                        patch.object(torch, 'set_num_interop_threads') as inter:
                    configure_worker(tmp, core.digest(path))
                    intra.assert_called_once_with(1)
                    inter.assert_called_once_with(1)
                    self.assertTrue(all(os.environ[k] == '1' for k in THREAD_ENV))

        def test_snapshot_and_strict_json(self):
            with tempfile.TemporaryDirectory() as tmp:
                src, dst = Path(tmp) / 'source.py', Path(tmp) / 'copy.py'
                src.write_bytes(b'# immutable fixture\n')
                original_mode = src.stat().st_mode
                entry = snapshot(src, dst)
                self.assertEqual(entry['source'], str(src.resolve()))
                self.assertEqual(core.digest(src), entry['sha256'])
                self.assertEqual(src.stat().st_mode, original_mode)
                self.assertNotEqual(src.stat().st_ino, dst.stat().st_ino)
                self.assertEqual(dst.stat().st_mode & 0o222, 0)
                with self.assertRaises(FileExistsError):
                    snapshot(src, dst)
                verify_provenance(dict(source_sha256={}, snapshots=[entry]))
                src.write_bytes(b'# changed fixture\n')
                with self.assertRaises(ValueError):
                    verify_provenance(dict(source_sha256={}, snapshots=[entry]))
                core.write_json(Path(tmp) / 'strict.json', dict(value=float('nan')))
                self.assertEqual(json.loads((Path(tmp) / 'strict.json').read_text()), {'value': 'NaN'})

        def test_worker_exact_api_compact_raw_preserved_and_failure_dict(self):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                raw_path = output / 'raw.json'
                raw = fixture(0, 0.)
                raw.update(summary_path=str(raw_path), iterations=[dict(candidate_id=0, assessed=True,
                    validation_mean=5., state_sha256='fixture-sha', ap_reference='fixture-assessment',
                    separation_status='violating', samples=[{'large_evidence': list(range(1000))}])])
                core.write_json(raw_path, raw)
                before = raw_path.read_bytes()
                with patch.object(ccg, 'run_configuration', return_value=raw) as run:
                    result = run_job(0, 0., output, 'manifest-sha')
                    run.assert_called_once_with(0, 0., output, provenance_hash='manifest-sha')
                self.assertTrue(completed_trajectory(result))
                self.assertNotIn('iterations', result)
                self.assertNotIn('large_evidence', json.dumps(result))
                self.assertEqual(before, raw_path.read_bytes())
                raw.update(status='failed', complete=False)
                core.write_json(raw_path, raw)
                with patch.object(ccg, 'run_configuration', return_value=raw):
                    self.assertEqual(run_job(0, 0., output, 'manifest-sha')['status'], 'failed')

        def test_all90_collected_despite_failure_no_final_incomplete_seed(self):
            submitted = []

            class MockExecutor:
                def __init__(self, **kwargs):
                    self.options = kwargs

                def __enter__(self):
                    self_test_case.assertEqual(self.options['max_workers'], 16)
                    self_test_case.assertEqual(self.options['mp_context'].get_start_method(), 'spawn')
                    self_test_case.assertIs(self.options['initializer'], configure_worker)
                    return self

                def __exit__(self, *args):
                    return False

                def submit(self, function, seed, radius, output, sha):
                    self_test_case.assertIs(function, run_job)
                    submitted.append((seed, radius))
                    future = Future()
                    if (seed, radius) == (0, 0.):
                        future.set_result(fixture(seed, radius, status='failed', complete=False))
                    elif (seed, radius) == (1, 0.):
                        future.set_exception(RuntimeError('mock broken worker'))
                    else:
                        future.set_result(fixture(seed, radius))
                    return future

            self_test_case = self
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'ccg_results'

                def freeze(path):
                    manifest = dict(source_sha256={}, snapshots=[])
                    core.write_json(path / 'provenance.json', manifest)
                    return manifest

                with patch.object(module, 'OUTPUT', output), patch.object(module, 'freeze_provenance', side_effect=freeze), \
                        patch.object(module, 'ProcessPoolExecutor', MockExecutor), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):
                        main()
                    saved = json.loads((output / 'summary.json').read_text())
                    self.assertEqual(len(submitted), 90)
                    self.assertEqual((saved['completed_count'], saved['failed_count'], saved['collected_count']), (88, 2, 90))
                    self.assertEqual(saved['status'], 'failed')
                    self.assertTrue(saved['final_provenance_verified'])
                    self.assertEqual(saved['running_count'], 0)
                    self.assertIsNone(saved['selections'][0]['final_selection'])
                    self.assertIsNotNone(saved['selections'][2]['final_selection'])
                    self.assertEqual(len(list((output / 'jobs').glob('*.json'))), 90)
                    self.assertGreaterEqual(saved['ended_monotonic_s'], saved['started_monotonic_s'])
                    before = (output / 'summary.json').read_bytes()
                    with self.assertRaises(FileExistsError):
                        main()
                    self.assertEqual(before, (output / 'summary.json').read_bytes())

        def test_success_and_submission_failure_collect_remaining(self):
            for submit_failure in (False, True):
                with self.subTest(submit_failure=submit_failure), tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp) / 'ccg_results'
                    attempted = []

                    def submit(function, seed, radius, root, sha):
                        attempted.append((seed, radius))
                        if submit_failure and (seed, radius) == (0, 0.):
                            raise RuntimeError('mock submit error')
                        future = Future()
                        future.set_result(fixture(seed, radius))
                        return future

                    def freeze(path):
                        manifest = dict(source_sha256={}, snapshots=[])
                        core.write_json(path / 'provenance.json', manifest)
                        return manifest

                    with patch.object(module, 'OUTPUT', output), \
                            patch.object(module, 'freeze_provenance', side_effect=freeze), \
                            patch.object(module, 'ProcessPoolExecutor') as executor, \
                            contextlib.redirect_stdout(io.StringIO()):
                        executor.return_value.__enter__.return_value.submit.side_effect = submit
                        if submit_failure:
                            with self.assertRaises(RuntimeError):
                                main()
                        else:
                            main()
                    saved = json.loads((output / 'summary.json').read_text())
                    self.assertEqual(len(attempted), 90)
                    self.assertEqual(saved['collected_count'], 90)
                    self.assertEqual(saved['completed_count'], 89 if submit_failure else 90)
                    self.assertEqual(saved['status'], 'failed' if submit_failure else 'completed')
                    self.assertTrue(saved['final_provenance_verified'])
                    self.assertEqual(sum(s['final_selection'] is not None for s in saved['selections']),
                                     9 if submit_failure else 10)
                    for selection in saved['selections'][1:] if submit_failure else saved['selections']:
                        self.assertFalse(selection['provisional'])
                        self.assertIsNone(selection['warning'])
                    self.assertEqual(len(list((output / 'jobs').glob('*.json'))), 90)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LauncherTests))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        raise SystemExit(self_test())
    if sys.argv[1:]:
        raise SystemExit('Usage: run_ccg.py [--self-test] (fixed full grid; no partial experiment mode)')
    main()