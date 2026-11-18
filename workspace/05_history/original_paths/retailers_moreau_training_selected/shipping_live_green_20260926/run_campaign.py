"""Fresh full shipping campaign: frozen nominal grid -> LR -> 270 CCG -> radius.

Requires --output, --preflight (summary path), and --preflight-sha256.
No test split, refitting, resume, or numerical fallback.
Controller imports are deliberately deferred and use the unambiguous shipping name.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import argparse
import hashlib
import importlib.metadata
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shlex
import sys
import time
import traceback

import shipping_live_common as common
from shipping_live_common import (
    ROOT, PRODUCTION, PROTOCOL, SIZES, SEEDS, LR_GRID, RADII, THREAD_ENV,
    check, core, digest, legacy_ccg, metadata, nominal_common, np, read,
    shipping_model, shipping_resources, torch, verify_sources,
)


def array_hash(*arrays):
    """Exactly the frozen nominal runner's ordered little-endian float64 hash."""
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array, dtype='<f8').tobytes())
    return h.hexdigest()


def select_row(rows, parameter):
    """Minimum native HARD validation mean, then smaller LR/radius on exact ties."""
    check(bool(rows), 'Empty selection grid')
    check(len({r[parameter] for r in rows}) == len(rows), 'Duplicate selection arm')
    for row in rows:
        check(row['status'] == 'completed' and row['complete'], 'Incomplete validation arm')
        check(math.isfinite(row['validation_hard_mean']), 'Nonfinite hard validation cost')
        check(math.isfinite(row[parameter]), 'Nonfinite selection parameter')
    return min(rows, key=lambda row: (row['validation_hard_mean'], row[parameter]))


def require_grid(rows, expected, keys, *, converged=False):
    actual = [tuple(row[key] for key in keys) for row in rows]
    check(len(actual) == len(expected) and set(actual) == set(expected),
          'Incomplete/duplicate grid; no partial selection')
    check(all(row['status'] == 'completed' and row['complete'] for row in rows),
          'Failed/incomplete grid; no next phase')
    if converged:
        check(all(row['converged'] for row in rows), 'Nonconverged CCG grid; no radius validation')


def _bind(mapping, path, expected=None):
    path = str(Path(path).resolve())
    actual = digest(path)
    check(expected is None or actual == expected, f'Changed input: {path}')
    check(path not in mapping or mapping[path] == actual, f'Input changed during barrier: {path}')
    mapping[path] = actual
    return path


def _timing(record):
    timing = {key: record[key] for key in
              ('status', 'wall_time_s', 'epochs_evaluated', 'started_at_utc', 'ended_at_utc')}
    check(timing['status'] == 'completed' and math.isfinite(timing['wall_time_s'])
          and timing['wall_time_s'] >= 0 and 0 < timing['epochs_evaluated'] <= 3000,
          'Missing/invalid persisted nominal training timing; rerun would be required')
    check(timing['started_at_utc'] and timing['ended_at_utc'], 'Missing nominal UTC timing')
    return timing


def load_cached_grid(output, sweep, inputs):
    """Read each large fit once; keep only compact records, never the huge summary."""
    setups, fits = [], []
    for n in SIZES:
        for seed in SEEDS:
            path = PRODUCTION / f'N{n}/initialization/seed{seed}/result.json'
            _bind(inputs, path)
            row = read(path)
            check((row['Ntotal'], row['seed']) == (n, seed), 'Wrong initialization identity')
            check(row['status'] == 'completed' and row['complete'], 'Incomplete initialization')
            for key in ('initial', 'training_data', 'split'):
                _bind(inputs, row[key + '_path'], row[key + '_sha256'])
            _bind(inputs, row['manifest_path'])
            train, split = read(row['training_data_path']), read(row['split_path'])
            x1, x2 = np.asarray(train['x1']), np.asarray(train['x2'])
            check(x1.shape == x2.shape == (4*n//5, 3), 'Wrong training array shape')
            check(array_hash(x1, x2) == train['training_array_sha256']
                  == split['train_array_sha256'], 'Training array hash mismatch')
            check(train['original_sample_ids'] == split['train_ids'], 'Training IDs mismatch')
            check(train['paired_data_sha256'] == split['paired_data_sha256']
                  == row['paired_data_sha256'], 'Original paired hash mismatch')
            for data in (train, split):
                check((data['Ntotal'], data['seed'], data['split_seed']) == (n, seed, 100000+seed),
                      'Wrong saved split identity')
            ids = split['train_ids'] + split['holdout_ids']
            check(len(split['train_ids']) == 4*n//5 and len(split['holdout_ids']) == n//5
                  and len(ids) == n and set(ids) == set(range(n)), 'Invalid/disjoint split IDs')
            compact = {key: row[key] for key in (
                'Ntotal', 'seed', 'status', 'complete', 'initial_path', 'initial_sha256',
                'training_data_path', 'training_data_sha256', 'split_path', 'split_sha256',
                'paired_data_sha256', 'wall_time_s', 'started_at_utc', 'ended_at_utc')}
            compact.update(result_path=str(path), result_sha256=inputs[str(path)],
                           method='cached_nominal_initialization', timing_scope='persisted prior invocation',
                           configured_workers=row['configured_workers'])
            setups.append(compact)
    require_grid(setups, {(n, s) for n in SIZES for s in SEEDS}, ('Ntotal', 'seed'))
    by_id = {(r['Ntotal'], r['seed']): r for r in setups}
    sweep['cached_initialization'] = setups
    sweep['cached_nominal'] = fits
    for n in SIZES:
        for seed in SEEDS:
            setup = by_id[n, seed]
            for lr in LR_GRID:
                path = PRODUCTION / f'N{n}/nominal/seed{seed}/lr{lr:g}/result.json'
                _bind(inputs, path)
                row = read(path)
                check((row['Ntotal'], row['seed'], row['lr_net']) == (n, seed, lr), 'Wrong fit identity')
                check(row['status'] == 'completed' and row['complete']
                      and row['disposable_preflight'] is False, 'Incomplete/nonproduction nominal fit')
                check(row['initial_sha256'] == setup['initial_sha256'], 'Nominal initialization mismatch')
                _bind(inputs, row['checkpoint'], row['checkpoint_sha256'])
                _bind(inputs, row['manifest_path'])
                timing = _timing(row['training'])
                loss = row['training']['selected_training_loss']
                check(math.isfinite(loss), 'Nonfinite saved smooth training loss')
                fits.append(dict(Ntotal=n, seed=seed, lr_net=lr, status='completed', complete=True,
                    method='cached_nominal_training', checkpoint=row['checkpoint'],
                    checkpoint_sha256=row['checkpoint_sha256'], initial_sha256=row['initial_sha256'],
                    result_path=str(path), result_sha256=inputs[str(path)],
                    selected_training_loss=loss, training_timing=timing,
                    wall_time_s=timing['wall_time_s'], epochs_evaluated=timing['epochs_evaluated'],
                    started_at_utc=timing['started_at_utc'], ended_at_utc=timing['ended_at_utc'],
                    nominal_job_wall_time_s=row['wall_time_s'],
                    configured_workers=row['configured_workers'], peak_rss_bytes=row['peak_rss_bytes'],
                    timing_scope='persisted prior training; not current cache-loading time', retrained=False))
                del row
                persist(output, sweep)
    require_grid(fits, {(n, s, lr) for n in SIZES for s in SEEDS for lr in LR_GRID},
                 ('Ntotal', 'seed', 'lr_net'))
    # Recheck every bound byte after all 120 records have been consumed.
    for path, expected in inputs.items():
        check(digest(path) == expected, f'Input changed before complete-grid barrier: {path}')
    sweep['nominal_complete_grid_barrier'] = dict(complete=True, initializations=30, fits=90,
                                                completed_at_utc=core.utc_now())
    return setups, fits


def verified_split(setup):
    """Regenerate exactly the original pairs and train/heldout partition, never test."""
    for key in ('training_data', 'split'):
        check(digest(setup[key + '_path']) == setup[key + '_sha256'], f'Changed {key}')
    train, saved = read(setup['training_data_path']), read(setup['split_path'])
    n, seed = setup['Ntotal'], setup['seed']
    data = nominal_common.split_data(n, seed)
    train_ids, val_ids = data['train_indices'], data['val_indices']
    check(data['xi1'].shape == data['xi2'].shape == (n, 3), 'Regenerated pair dimensions')
    check(np.array_equal(train_ids, train['original_sample_ids'])
          and np.array_equal(train_ids, saved['train_ids'])
          and np.array_equal(val_ids, saved['holdout_ids']), 'Regenerated split IDs differ')
    check(len(train_ids) == 4*n//5 and len(val_ids) == n//5
          and set(np.concatenate((train_ids, val_ids)).tolist()) == set(range(n)), 'Invalid split coverage')
    check(array_hash(data['xi1'], data['xi2']) == setup['paired_data_sha256']
          == train['paired_data_sha256'] == saved['paired_data_sha256'], 'Regenerated paired hash differs')
    a, b = data['xi1'][train_ids], data['xi2'][train_ids]
    check(np.array_equal(a, np.asarray(train['x1'])) and np.array_equal(b, np.asarray(train['x2']))
          and array_hash(a, b) == train['training_array_sha256'] == saved['train_array_sha256'],
          'Regenerated training arrays differ')
    check(array_hash(data['xi1'][val_ids], data['xi2'][val_ids]) == saved['holdout_array_sha256'],
          'Regenerated holdout hash differs')
    for item in (train, saved):
        check((item['Ntotal'], item['seed'], item['split_seed']) == (n, seed, 100000+seed),
              'Changed split identity')
    return data


def native_evaluate(model, x1, x2, record):
    """Same returned q, no clipping; retain observations before physical rejection."""
    frozen = legacy_ccg.state_hash(model)
    with core.timed(record), torch.no_grad():
        model.eval()
        try:
            smooth, hard, q = model.costs_and_shipments(
                torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32))
            check(smooth.shape == hard.shape == (len(x1),) and q.shape == (len(x1), 3),
                  'Wrong native evaluation dimensions')
            check(bool(torch.isfinite(smooth).all() and torch.isfinite(hard).all()
                       and torch.isfinite(q).all()), 'Nonfinite native evaluation')
            q10 = float(torch.nn.functional.softplus(model.Q10).detach())
            check(math.isfinite(q10), 'Nonfinite native capacity')
            q64 = q.detach().to(torch.float64)
            residual = max(0., float(-q64.min()), float((q64.sum(dim=1)-q10).max()))
            record.update(hard_mean=float(hard.mean()), smooth_mean=float(smooth.mean()),
                per_sample_hard_costs=hard.detach().cpu().tolist(),
                per_sample_smooth_costs=smooth.detach().cpu().tolist(),
                shipments=q.detach().cpu().tolist(), sample_count=len(x1),
                native_physical_tolerance=1e-6, native_physical_residual=residual,
                native_physical_guard_passed=residual <= 1e-6,
                hard_fee='0.1 per native q strictly > 0', shipment_clipping=False,
                state_sha256=frozen)
            check(residual <= 1e-6, 'Native validation violates physical constraints')
            check(legacy_ccg.state_hash(model) == frozen, 'Evaluation mutated learned state')
        finally:
            record['projection'] = model.proj_layer.drain_summary()
    return record


def _load_model(path, expected, state_sha256=None):
    check(digest(path) == expected, f'Changed checkpoint: {path}')
    model = shipping_model.model_from_state(torch.load(path, map_location='cpu', weights_only=True))
    if state_sha256 is not None:
        check(legacy_ccg.state_hash(model) == state_sha256, 'Changed checkpoint learned state')
    return model


def select_nominal(params, folder, record):
    setup, fits = params['setup'], params['fits']
    with core.timed(record):
        with core.timed(record.setdefault('split_verification', {})):
            data = verified_split(setup)
        ids = data['val_indices']
        arms = record['arms'] = []
        for fit in fits:
            arm = dict(lr_net=fit['lr_net'], complete=False)
            arms.append(arm)
            try:
                with core.timed(arm):
                    model = _load_model(fit['checkpoint'], fit['checkpoint_sha256'])
                    native_evaluate(model, data['xi1'][ids], data['xi2'][ids],
                                    arm.setdefault('evaluation', {}))
                    arm.update(validation_hard_mean=arm['evaluation']['hard_mean'], complete=True,
                               checkpoint=fit['checkpoint'], checkpoint_sha256=fit['checkpoint_sha256'])
            finally:
                core.write_json(folder / f'lr{fit["lr_net"]:g}.json', arm)
        require_grid(arms, {(lr,) for lr in LR_GRID}, ('lr_net',))
        winner = select_row(arms, 'lr_net')
        fit = next(f for f in fits if f['lr_net'] == winner['lr_net'])
        model = _load_model(fit['checkpoint'], fit['checkpoint_sha256'])
        ids = data['train_indices']
        replay = record['smooth_training_replay'] = {}
        native_evaluate(model, data['xi1'][ids], data['xi2'][ids], replay)
        check(abs(replay['smooth_mean'] - fit['selected_training_loss']) <= 1e-6,
              'Selected nominal smooth training replay mismatch')
        path = folder / 'training_data.json'
        core.write_json(path, dict(centers=np.concatenate((data['xi1'][ids], data['xi2'][ids]), axis=1),
            original_sample_ids=ids, data_sha256=setup['paired_data_sha256']))
        path.chmod(0o444)
        record['nominal_record'] = dict(Ntotal=setup['Ntotal'], seed=setup['seed'], lr_net=fit['lr_net'],
            checkpoint=fit['checkpoint'], checkpoint_sha256=fit['checkpoint_sha256'],
            state_sha256=legacy_ccg.state_hash(model), train_mean=replay['smooth_mean'],
            training_data_path=str(path), training_data_sha256=digest(path),
            data_sha256=setup['paired_data_sha256'], training_timing=fit['training_timing'])
        record.update(selected_lr=fit['lr_net'], validation_hard_mean=winner['validation_hard_mean'], complete=True)


def select_radius(params, folder, record):
    with core.timed(record):
        rows = params['configurations']
        require_grid(rows, {(r,) for r in RADII}, ('radius',), converged=True)
        with core.timed(record.setdefault('split_verification', {})):
            data = verified_split(params['setup'])
        ids = data['val_indices']
        arms = record['arms'] = []
        for result in rows:
            arm = dict(radius=result['radius'], complete=False)
            arms.append(arm)
            try:
                with core.timed(arm):
                    check(digest(result['summary_path']) == result['summary_sha256'], 'Changed CCG summary')
                    model = _load_model(result['selected_checkpoint'], result['selected_checkpoint_sha256'],
                                        result['selected_state_sha256'])
                    native_evaluate(model, data['xi1'][ids], data['xi2'][ids],
                                    arm.setdefault('evaluation', {}))
                    arm.update(validation_hard_mean=arm['evaluation']['hard_mean'], complete=True,
                        checkpoint=result['selected_checkpoint'], checkpoint_sha256=result['selected_checkpoint_sha256'],
                        state_sha256=result['selected_state_sha256'])
            finally:
                core.write_json(folder / f'radius{result["radius"]:g}.json', arm)
        require_grid(arms, {(r,) for r in RADII}, ('radius',))
        winner = select_row(arms, 'radius')
        record.update(selected_radius=winner['radius'], validation_hard_mean=winner['validation_hard_mean'],
                      selected_checkpoint=winner['checkpoint'], selected_checkpoint_sha256=winner['checkpoint_sha256'],
                      selected_state_sha256=winner['state_sha256'], complete=True)


def worker(kind, params, folder, provenance_path):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    record = metadata(params['n_total'], params['seed'], params.get('radius'), kind)
    start = time.perf_counter()
    record.update(status='running', complete=False, pid=os.getpid(), started_at_utc=core.utc_now(),
                  worker_log=str(folder / 'worker.log'), result_path=str(folder / 'result.json'))
    try:
        with (folder / 'worker.log').open('x', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                manifest = read(provenance_path)
                record['configured_workers'] = manifest['workers']
                verify_sources(manifest)
                check(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'Changed worker threads')
                if kind == 'nominal_validation':
                    select_nominal(params, folder, record)
                elif kind == 'radius_validation':
                    select_radius(params, folder, record)
                elif kind == 'ccg':
                    import shipping_ccg
                    check(Path(shipping_ccg.__file__).resolve() == ROOT / 'shipping_ccg.py', 'Wrong CCG module')
                    record.update(shipping_ccg.run_configuration(
                        params['n_total'], params['seed'], params['radius'], params['output_root'],
                        params['nominal_record'], provenance_path))
                else:
                    raise ValueError('Unapproved stage')
                verify_sources(manifest)
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                log.flush()
    except BaseException as error:
        record.update(status='failed', complete=False, error=dict(type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc()))
        if not isinstance(error, Exception):
            raise
    finally:
        record.update(worker_wall_time_s=time.perf_counter()-start, worker_ended_at_utc=core.utc_now(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_rss_scope='process lifetime, including prior jobs in this worker',
            actual_torch_threads=torch.get_num_threads(), actual_interop_threads=torch.get_num_interop_threads())
        if 'wall_time_s' not in record:
            record.update(wall_time_s=record['worker_wall_time_s'], ended_at_utc=record['worker_ended_at_utc'])
        core.write_json(folder / 'result.json', record)
    # Detailed costs/shipments and replay live in the per-worker record only.
    compact = {key: value for key, value in record.items() if key not in ('arms', 'smooth_training_replay')}
    if 'arms' in record:
        compact['arms'] = [{key: value for key, value in arm.items() if key != 'evaluation'}
                           for arm in record['arms']]
        for arm, raw in zip(compact['arms'], record['arms']):
            if 'evaluation' in raw:
                arm['evaluation_timing'] = {key: raw['evaluation'].get(key) for key in
                    ('status', 'wall_time_s', 'started_at_utc', 'ended_at_utc')}
    if 'smooth_training_replay' in record:
        compact['smooth_training_replay_timing'] = {key: record['smooth_training_replay'].get(key)
            for key in ('status', 'wall_time_s', 'started_at_utc', 'ended_at_utc')}
    compact['result_sha256'] = digest(folder / 'result.json')
    return compact


def aggregate(sweep):
    groups, incomplete = [], []
    rows = list(sweep.get('cached_initialization', [])) + list(sweep.get('cached_nominal', []))
    for name, stage in sweep['phases'].items():
        for row in stage.get('jobs', []):
            rows.append({**row, 'method': name, 'timing_scope': 'current invocation'})
            for arm in row.get('arms', []):
                rows.append({**arm, 'Ntotal': row['Ntotal'], 'seed': row['seed'],
                             'method': name + '_arm', 'timing_scope': 'current invocation'})
    grouped = {}
    for row in rows:
        if row.get('status') != 'completed' or not row.get('complete') or (
                row['method'] == 'ccg' and not row.get('converged')):
            incomplete.append({key: row.get(key) for key in
                ('method', 'Ntotal', 'seed', 'radius', 'lr_net', 'status', 'terminal_status',
                 'wall_time_s', 'worker_wall_time_s', 'result_path')})
            continue
        key = (row['method'], row['Ntotal'], row.get('radius'), row.get('lr_net'), row['timing_scope'])
        grouped.setdefault(key, []).append(row)
    for (method, n, radius, lr, scope), members in sorted(grouped.items(), key=lambda item: str(item[0])):
        group = dict(method=method, Ntotal=n, radius=radius, lr_net=lr, timing_scope=scope,
                     seeds=[r['seed'] for r in members], wall_time_s=core.stats([r['wall_time_s'] for r in members]))
        if method == 'ccg':
            for field in ('ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s'):
                measured = [r[field] for r in members if field in r]
                group[field] = core.stats(measured)
                group[field + '_unavailable_count'] = len(members)-len(measured)
        groups.append(group)
    return dict(groups=groups, incomplete=incomplete,
        overall_sweep_wall_time_s=sweep.get('wall_time_s'), workers=sweep.get('workers'),
        environment=sweep.get('environment'),
        timing_scope='Concurrent per-model wall times are NOT summed as sweep elapsed; cached fits are separate',
        not_run=['test evaluation', 'refit', 'SRO-FA', 'CV'],
        runtime_table_coverage=dict(nominal_training='all90 cached fits, persisted prior times',
            robust_neural_CCG='current jobs, failed/nonconverged separate',
            SRO_FA='not part of authorized campaign; unavailable',
            evaluation='nominal and radius validation stages and arms', CV='not run'))


def persist(output, sweep):
    if '_monotonic_start' in sweep:
        sweep['wall_time_s'] = time.perf_counter()-sweep['_monotonic_start']
        seconds = sweep['wall_time_s']
        sweep['wall_time_human'] = f'{seconds:.3f} s / {seconds/60:.3f} min / {seconds/3600:.3f} h'
    core.write_json(output / 'summary.json', {k: v for k, v in sweep.items() if not k.startswith('_')})
    core.write_json(output / 'runtime_aggregates.json', aggregate(sweep))


def run_stage(name, tasks, workers, output, manifest, sweep):
    phase = sweep['phases'][name] = dict(jobs=[], max_workers=workers, planned_count=len(tasks), submitted_count=0)
    stage_start = time.perf_counter()
    futures, observed = {}, set()
    pool = None
    try:
        with core.timed(phase):
            pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                       initializer=core.configure_worker)
            for params, folder in tasks:
                future = pool.submit(worker, name, params, str(folder), str(manifest))
                futures[future] = (params, folder)
                phase['submitted_count'] += 1
                if name == 'ccg':
                    sweep['full_CCG_launched'] = phase['submitted_count'] == 270
                    sweep['protocol']['full_CCG_launched'] = sweep['full_CCG_launched']
                persist(output, sweep)
            for future in as_completed(futures):
                row = future.result()
                observed.add(future)
                phase['jobs'].append(row)
                print(name, row['Ntotal'], row['seed'], row.get('radius'), row['status'], flush=True)
                persist(output, sweep)
            # Ordinary failed/nonconverged jobs never cancel any other configuration.
    except BaseException:
        phase['controller_traceback'] = traceback.format_exc()
        submitted = {str(folder) for _, folder in futures.values()}
        for future, (params, folder) in futures.items():
            if future in observed:
                continue
            cancelled = future.cancel()
            phase['jobs'].append(dict(Ntotal=params['n_total'], seed=params['seed'], radius=params.get('radius'),
                status='cancelled' if cancelled or future.cancelled() else 'unobserved_after_controller_failure',
                complete=False, result_path=str(folder / 'result.json'), wall_time_s=None))
        for params, folder in tasks:
            if str(folder) not in submitted:
                phase['jobs'].append(dict(Ntotal=params['n_total'], seed=params['seed'], radius=params.get('radius'),
                    status='not_dispatched', complete=False, wall_time_s=None))
        raise
    finally:
        try:
            persist(output, sweep)
        finally:
            try:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
            finally:
                # Recover already-running worker results after a controller/pool failure.
                # Never relaunch; interrupted workers without results stay explicit.
                for future, (params, folder) in futures.items():
                    if future in observed or not future.done() or future.cancelled():
                        continue
                    try:
                        recovered = future.result()
                    except BaseException as error:
                        for row in phase['jobs']:
                            if row.get('result_path') == str(folder / 'result.json'):
                                row.update(status='future_failed', future_error=dict(
                                    type=type(error).__name__, message=str(error)))
                                if (folder / 'result.json').is_file():
                                    row['raw_persisted_result_path'] = str(folder / 'result.json')
                                    row['raw_persisted_result_scope'] = (
                                        'Unverified worker artifact; transport/controller failure remains authoritative')
                        continue
                    for index, row in enumerate(phase['jobs']):
                        if row.get('result_path') == str(folder / 'result.json'):
                            phase['jobs'][index] = recovered
                            break
                phase.update(ended_at_utc=core.utc_now(), wall_time_s=time.perf_counter()-stage_start,
                             timing_scope='current full stage including pool shutdown')
                persist(output, sweep)
    return phase['jobs']


def _manifest(path, inputs, workers, environment, source_hashes, *, ccg=False):
    manifest = dict(source_sha256=dict(source_hashes), input_sha256=dict(inputs), workers=workers,
        protocol=dict(PROTOCOL) if ccg else {**PROTOCOL, 'full_CCG_launched': False},
        full_CCG_launched=False, protocol_scope='authorized protocol; dispatch state is in campaign summary',
        environment=environment, thread_environment=THREAD_ENV,
        versions={k: importlib.metadata.version(k) for k in ('torch', 'numpy', 'scipy', 'gurobipy', 'moreau-cpu')})
    verify_sources(manifest)
    core.write_json(path, manifest)
    path.chmod(0o444)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--preflight-sha256', required=True)
    args = parser.parse_args()
    args.preflight = args.preflight.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sweep = dict(protocol={**PROTOCOL, 'full_CCG_launched': False}, phases={},
        full_CCG_launched=False, test_evaluated=False, status='running',
        started_at_utc=core.utc_now(), _monotonic_start=time.perf_counter(),
        command=shlex.join(sys.orig_argv), cwd=str(Path.cwd()),
        timing_clock='time.perf_counter', timing_unit='seconds')
    try:
        inputs = {}
        _bind(inputs, args.preflight, args.preflight_sha256)
        preflight_manifest = args.preflight.parent / 'provenance.json'
        _bind(inputs, preflight_manifest)
        integration = common.validate_integration(args.preflight, args.preflight_sha256)
        source_hashes = read(preflight_manifest)['source_sha256']
        # Bind the checked bytes, not a fresh snapshot of the current sources.
        # These inputs also enter every child manifest and its before/after gate.
        verify_sources(dict(source_sha256=source_hashes, input_sha256=inputs))
        core.configure_worker()
        environment = shipping_resources.resources()
        environment.update(platform=platform.platform(), hostname=platform.node())
        peak = max(713433088, integration['peak_rss_bytes'])
        policy = shipping_resources.choose_workers(environment, peak)
        workers = {**policy, 'ccg': min(270, policy['nominal']),
                   'nominal_validation': min(30, policy['nominal']),
                   'radius_validation': min(30, policy['nominal'])}
        sweep.update(environment=environment, workers=workers,
            resource_peak_evidence=dict(approved_readiness_peak_rss_bytes=713433088,
                integration_peak_rss_bytes=integration['peak_rss_bytes'], peak_rss_bytes=peak,
                preflight_path=str(args.preflight), preflight_sha256=args.preflight_sha256,
                scope='N50 only; not memory certification for larger sample sizes'))
        check(str(ROOT / 'shipping_ccg.py') in source_hashes, 'Missing shipping_ccg controller')
        barrier = sweep['phases']['cached_grid_verification'] = {}
        with core.timed(barrier):
            setups, fits = load_cached_grid(output, sweep, inputs)
        nominal_manifest = _manifest(output / 'nominal_provenance.json', inputs, workers, environment, source_hashes)
        tasks = [(dict(n_total=n, seed=s, setup=next(r for r in setups if (r['Ntotal'], r['seed']) == (n, s)),
                       fits=[r for r in fits if (r['Ntotal'], r['seed']) == (n, s)]),
                  output / f'N{n}/nominal_selection/seed{s}') for n in SIZES for s in SEEDS]
        selected = run_stage('nominal_validation', tasks, workers['nominal_validation'],
                             output, nominal_manifest, sweep)
        require_grid(selected, {(n, s) for n in SIZES for s in SEEDS}, ('Ntotal', 'seed'))
        winners = {(r['Ntotal'], r['seed']): r['nominal_record'] for r in selected}
        for row in selected:
            _bind(inputs, row['result_path'], row['result_sha256'])
            nominal = row['nominal_record']
            _bind(inputs, nominal['training_data_path'], nominal['training_data_sha256'])
        _bind(inputs, nominal_manifest)
        provenance = _manifest(output / 'provenance.json', inputs, workers, environment, source_hashes, ccg=True)
        tasks = [(dict(n_total=n, seed=s, radius=r, output_root=str(output / f'N{n}/ccg'),
                       nominal_record=winners[n, s]), output / f'N{n}/workers/seed{s}/radius{r:g}')
                 for r in RADII for s in SEEDS for n in SIZES]
        configurations = run_stage('ccg', tasks, workers['ccg'], output, provenance, sweep)
        require_grid(configurations, {(n, s, r) for n in SIZES for s in SEEDS for r in RADII},
                     ('Ntotal', 'seed', 'radius'), converged=True)
        sweep['ccg_complete_grid_barrier'] = dict(complete=True, converged_configurations=270,
                                                completed_at_utc=core.utc_now())
        for row in configurations:
            _bind(inputs, row['summary_path'], row['summary_sha256'])
            _bind(inputs, row['selected_checkpoint'], row['selected_checkpoint_sha256'])
        _bind(inputs, provenance)
        radius_manifest = _manifest(output / 'validation_provenance.json', inputs, workers, environment, source_hashes, ccg=True)
        tasks = [(dict(n_total=n, seed=s, setup=next(r for r in setups if (r['Ntotal'], r['seed']) == (n, s)),
                       configurations=[r for r in configurations if (r['Ntotal'], r['seed']) == (n, s)]),
                  output / f'N{n}/radius_selection/seed{s}') for n in SIZES for s in SEEDS]
        radii = run_stage('radius_validation', tasks, workers['radius_validation'], output, radius_manifest, sweep)
        require_grid(radii, {(n, s) for n in SIZES for s in SEEDS}, ('Ntotal', 'seed'))
        verify_sources(read(radius_manifest))
        sweep.update(status='completed', terminal_status='FULL270_COMPLETE_HARD_VALIDATION_SELECTED_TEST_UNTOUCHED')
    except BaseException as error:
        sweep.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
            error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        sweep['ended_at_utc'] = core.utc_now()
        persist(output, sweep)
        print(sweep['status'], sweep['wall_time_human'], flush=True)


if __name__ == '__main__':
    main()