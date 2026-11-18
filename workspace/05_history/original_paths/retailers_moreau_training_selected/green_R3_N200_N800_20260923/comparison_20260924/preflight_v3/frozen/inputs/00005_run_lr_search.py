"""Approved nominal-only 40/10 LR search. Importing this file never launches it.

Launch/log/tmux ownership is external. No test set, refit, AP, robust training,
SCS solve, cache reuse, retries, or initialization repair is performed here.
"""
import os

# Must precede numerical imports, including in spawned seed workers.
THREAD_ENV = {name: '1' for name in (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
import traceback

import core
import moreau_cpu
import numpy as np
import torch

OUTPUT = Path(__file__).resolve().parent / 'lr_search_results_v2'
MAX_WORKERS = 10
CHECK_ATOL = 1e-6  # absolute replay check only; not a solver/training tolerance


def metadata(seed=None, lr=None):
    return dict(experiment='retailer_R3_N50_net16_v2_train40_val10_lr_search',
                method='nominal_moreau', Ntotal=50, Ntrain=40, Nval=10, radius=0.,
                data_seeds=list(core.SEEDS), seed=seed, data_seed=seed,
                split_seed=None if seed is None else 100000 + seed,
                network_seed=0, lr_net=lr, lr_grid=list(core.LR_GRID), lr_inventory=.01,
                max_epochs=3000, patience=300, epochs_evaluated=0,
                optimizer_steps=0, stop_status='not_started',
                seed_workers=MAX_WORKERS, lr_workers_per_seed=1,
                torch_threads=1, torch_interop_threads=1, gurobi_threads=1,
                thread_environment=THREAD_ENV.copy(), device='cpu',
                timing_unit='seconds', timing_clock='time.perf_counter',
                cached=False, resumed=False)


def configure_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def provenance():
    paths = [Path(__file__).resolve(), Path(core.__file__).resolve(),
             core._BASELINE_PATH.resolve(), core._loader.SOURCE.resolve()]
    # Include the isolated native implementation, not merely its distribution name.
    native_root = Path(core.moreau.__file__).resolve().parent
    cpu_root = Path(moreau_cpu.__file__).resolve().parent
    native_paths = sorted(path for root in (native_root, cpu_root) for path in root.rglob('*')
                          if path.is_file() and path.suffix in ('.py', '.so'))
    sources = {str(path): core.digest(path) for path in paths + native_paths}
    return dict(source_sha256=sources,
                versions={name: version(name) for name in
                          ('torch', 'numpy', 'scipy', 'gurobipy', 'cvxpy', 'cvxpylayers', 'moreau-cpu')},
                native_dependency=dict(name='moreau', version=core.moreau.__version__,
                                       path=str(native_root), cpu_path=str(cpu_root)),
                python=sys.version, executable=sys.executable,
                environment=dict(platform=platform.platform(), machine=platform.machine(),
                                 processor=platform.processor(), hostname=platform.node(),
                                 logical_cpu_count=os.cpu_count(),
                                 cpu_affinity=sorted(os.sched_getaffinity(0))),
                model=dict(class_name='canonical Model', hidden_widths=[16, 16],
                           input_width=7, output_width=3, parameter_dtype='float32',
                           projection_dtype='float64', capacity_buffer=0.,
                           native_solver_lifecycle='separate instance per batch size; fresh solves, no warm starts',
                           c=core.baseline.cfg.c, h=core.baseline.cfg.h, b=core.baseline.cfg.b),
                protocol=dict(checkpoint='validation preupdate best, core.train_model',
                              initialization='train40 SAA, network seed0; identical stored learned state for all LRs',
                              selection='per-seed minimum restored validation mean; exact ties smaller LR',
                              replay_check_atol=CHECK_ATOL, replay_check_rtol=0.,
                              test='not used', refit='not performed',
                              SCS='definitions constructed by canonical model; never solved'))


def error_info(error):
    return dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())


def raw_hash(data):
    """Ordered raw float64 xi1 then xi2, C order, before float32 conversion."""
    h = hashlib.sha256()
    for key in ('xi1', 'xi2'):
        h.update(np.asarray(data[key], dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def evaluate(model, x1, x2, record):
    with core.timed(record), torch.no_grad():
        model.eval()
        values = model.per_sample_cost(x1, x2)
        if values.shape != (len(x1),) or not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Restored checkpoint costs must be finite per-sample values')
        mean = float(values.mean())
        if not np.isfinite(mean):
            raise FloatingPointError('Nonfinite restored checkpoint mean')
        record.update(mean=mean, per_sample_costs=values, sample_count=len(x1))
    return mean


def run_lr(seed, lr, data, initial_path, folder, provenance_hash, split_hash):
    record = metadata(seed, lr)
    record.update(provenance_sha256=provenance_hash, raw_data_sha256=split_hash,
                  initial_sha256=core.digest(initial_path))
    training = metadata(seed, lr)
    training.update(provenance_sha256=provenance_hash,
                    initial_sha256=record['initial_sha256'], raw_data_sha256=split_hash)
    record['training'] = training
    try:
        with core.timed(record):
            folder.mkdir(exist_ok=False)
            core.write_json(folder / 'config.json', record)
            record['config_sha256'] = core.digest(folder / 'config.json')
            initialization = record['initial_reconstruction'] = {}
            with core.timed(initialization):
                initial = torch.load(initial_path, map_location='cpu', weights_only=True)
                # This is the fresh seed initial state, never a historical candidate.
                model = core.model_from_state(initial)
                restored = core.weights(model)
                if restored.keys() != initial.keys() or any(
                        not torch.equal(restored[key], value) for key, value in initial.items()):
                    raise AssertionError('Fresh LR reconstruction differs from stored initialization')
            model = core.train_model(
                model, lambda candidate: candidate.per_sample_cost(
                    data['train_x1'], data['train_x2']).mean(),
                data['val_x1'], data['val_x2'], lr, folder, training,
                epochs=3000, patience=300)
            selected_path = folder / 'selected.pth'
            selected = torch.load(selected_path, map_location='cpu', weights_only=True)
            restored = core.weights(model)
            if restored.keys() != selected.keys() or any(
                    not torch.equal(restored[key], value) for key, value in selected.items()):
                raise AssertionError('Returned model is not the saved selected checkpoint')
            record['selected_sha256'] = core.digest(selected_path)
            train_record = record['restored_train_evaluation'] = dict(sample_count=40)
            train_mean = evaluate(model, data['train_x1'], data['train_x2'], train_record)
            val_record = record['restored_validation_evaluation'] = dict(sample_count=10)
            val_mean = evaluate(model, data['val_x1'], data['val_x2'], val_record)
            record['checkpoint_replay_check'] = dict(
                atol=CHECK_ATOL, rtol=0.,
                train_abs_error=abs(train_mean - training['selected_training_loss']),
                validation_abs_error=abs(val_mean - training['best_val']))
            for actual, expected in ((train_mean, training['selected_training_loss']),
                                     (val_mean, training['best_val'])):
                if not np.isfinite(expected) or abs(actual - expected) > CHECK_ATOL:
                    raise AssertionError(f'Selected checkpoint replay mismatch: {actual} vs {expected}; atol=1e-6, rtol=0')
    finally:
        record.update(epochs_evaluated=training['epochs_evaluated'],
                      optimizer_steps=training['optimizer_steps'],
                      stop_status=training.get('stop_reason', record.get('status', 'failed')))
        if record.get('status') == 'failed':
            record['stop_status'] = 'failed'
        training['stop_status'] = training.get('stop_reason', training.get('status', 'not_started'))
        if (folder / 'training.json').exists():
            core.write_json(folder / 'training.json', training)
        core.write_json(folder / 'result.json', record)
    return record


def run_seed(seed, output, provenance_hash):
    folder = Path(output) / f'seed_{seed}'
    record = metadata(seed)
    record.update(provenance_sha256=provenance_hash, configurations=[], winner=None,
                  complete=False)
    errors = []
    try:
        with core.timed(record):
            folder.mkdir(exist_ok=False)
            core.write_json(folder / 'summary.json', record)
            data = core.split_data(seed)
            split_hash = raw_hash(data)
            core.write_json(folder / 'split.json', dict(
                **metadata(seed), arrays=data, raw_data_sha256=split_hash,
                raw_hash_encoding='xi1 then xi2; little-endian float64 C-order bytes'))
            record.update(raw_data_sha256=split_hash,
                          split_artifact_sha256=core.digest(folder / 'split.json'))
            saa = record['saa_initialization'] = metadata(seed)
            try:
                q10, q1r = core.solve_saa(data['xi1'][data['train_indices']],
                                         data['xi2'][data['train_indices']], saa)
            finally:
                core.write_json(folder / 'saa.json', saa)
            initialization = record['initial_model'] = {}
            initial_path = folder / 'initial.pth'
            with core.timed(initialization):
                initial_model = core.make_model(q10, q1r)
                torch.save(core.weights(initial_model), initial_path)
                record.update(initial_sha256=core.digest(initial_path),
                              projection_settings=initial_model.proj_layer.settings)
                del initial_model
            for lr in core.LR_GRID:
                lr_folder = folder / f'lr_{lr:g}'
                try:
                    result = run_lr(seed, lr, data, initial_path, lr_folder,
                                    provenance_hash, split_hash)
                except Exception as error:
                    errors.append(error)
                    result = metadata(seed, lr)
                    result.update(status='failed', stop_status='failed', error=error_info(error),
                                  result_path=str(lr_folder / 'result.json'))
                    # The detailed failed timing/epochs remain in the LR result.
                    if (lr_folder / 'result.json').exists():
                        result = json.loads((lr_folder / 'result.json').read_text())
                    print(f'seed={seed} lr={lr:g} FAILED: {error}', flush=True)
                record['configurations'].append(result)
                core.write_json(folder / 'summary.json', record)
            if errors:
                raise RuntimeError(f'Seed {seed} incomplete: {len(errors)} LR configuration(s) failed') from errors[0]
            selection = record['lr_selection'] = {}
            with core.timed(selection):
                winner = min(record['configurations'], key=lambda row: (
                    row['restored_validation_evaluation']['mean'], row['lr_net']))
                record['winner'] = dict(lr_net=winner['lr_net'],
                    validation_mean=winner['restored_validation_evaluation']['mean'],
                    train_mean=winner['restored_train_evaluation']['mean'],
                    best_epoch=winner['training']['best_epoch'],
                    selected_sha256=winner['selected_sha256'],
                    checkpoint=str(folder / f"lr_{winner['lr_net']:g}" / 'selected.pth'))
            record['complete'] = True
    finally:
        # Unattempted configurations are explicit; never replace them with successes.
        attempted = {row['lr_net'] for row in record['configurations']}
        for lr in core.LR_GRID:
            if lr not in attempted:
                row = metadata(seed, lr)
                row.update(status='not_run', reason='seed initialization or worker interrupted')
                record['configurations'].append(row)
        record['stop_status'] = 'completed' if record['complete'] else 'incomplete'
        record['epochs_evaluated'] = sum(row['epochs_evaluated'] for row in record['configurations'])
        record['optimizer_steps'] = sum(row['optimizer_steps'] for row in record['configurations'])
        core.write_json(folder / 'summary.json', record)
    print(f'seed={seed} completed winner_lr={record["winner"]["lr_net"]:g}', flush=True)
    return record


def summarize(records):
    rows = [row for seed in records for row in seed.get('configurations', [])]
    by_lr = {}
    phases = dict(full_model='wall_time_s', training='training',
                  restored_train_evaluation='restored_train_evaluation',
                  restored_validation_evaluation='restored_validation_evaluation')
    for lr in core.LR_GRID:
        completed = [row for row in rows if row['lr_net'] == lr and row['status'] == 'completed']
        by_lr[str(lr)] = dict(method='nominal_moreau', Ntotal=50, Ntrain=40, Nval=10,
                             radius=0., completed_count=len(completed), runtime_s={})
        for phase, key in phases.items():
            values = [row[key] if key == 'wall_time_s' else row[key]['wall_time_s'] for row in completed]
            by_lr[str(lr)]['runtime_s'][phase] = core.stats(values)
    return dict(per_lr=by_lr,
                completed_seed_wall_time_s=core.stats([
                    seed['wall_time_s'] for seed in records
                    if seed.get('complete') and 'wall_time_s' in seed]),
                winners=[dict(seed=seed['seed'], **seed['winner']) for seed in records
                         if seed.get('complete') and seed.get('winner') is not None],
                incomplete_seeds=[seed['seed'] for seed in records if not seed.get('complete')],
                failed_or_incomplete_configurations=[row for row in rows if row['status'] != 'completed'],
                runtime_scope='current invocation; per-model walls overlap across seeds; never summed as sweep elapsed',
                not_applicable=dict(robust_neural_CCG='not requested/not run', SRO_FA='not requested/not run',
                                    CV='not requested/not run', final_test='not requested/not run'))


def print_report(sweep):
    """Runtime table is descriptive only; no metrics enter model selection."""
    print('Completed configurations only: LR | phase | n | mean / median / min / max (s)', flush=True)
    for lr, row in sweep.get('aggregates', {}).get('per_lr', {}).items():
        for phase, stats in row['runtime_s'].items():
            values = ' / '.join('unavailable' if stats[key] is None else f'{stats[key]:.6f}'
                                for key in ('mean', 'median', 'min', 'max'))
            print(f'{lr} | {phase} | {stats["count"]} | {values}', flush=True)
    print('Robust neural/CCG | SRO-FA | CV | final test: not applicable / not run', flush=True)
    for winner in sweep.get('aggregates', {}).get('winners', []):
        print(f"Winner seed={winner['seed']} lr={winner['lr_net']:g} val={winner['validation_mean']:.9g}", flush=True)
    print(f"Incomplete seeds: {sweep.get('aggregates', {}).get('incomplete_seeds', 'unavailable')}", flush=True)


def main():

    # Refuse to overwrite or resume anything, including a failed prior invocation.
    OUTPUT.mkdir(exist_ok=False)
    sweep = metadata()
    sweep.update(seed_results=[], errors=[])
    try:
        with core.timed(sweep):
            core.write_json(OUTPUT / 'provenance.json', provenance())
            provenance_hash = core.digest(OUTPUT / 'provenance.json')
            sweep['provenance_sha256'] = provenance_hash
            core.write_json(OUTPUT / 'summary.json', sweep)
            # Spawn avoids inherited Torch/native solver thread state.
            failures = []
            with ProcessPoolExecutor(max_workers=MAX_WORKERS,
                    mp_context=multiprocessing.get_context('spawn'), initializer=configure_worker) as pool:
                futures = {}
                for seed in core.SEEDS:
                    try:
                        futures[pool.submit(run_seed, seed, OUTPUT, provenance_hash)] = seed
                    except Exception as error:
                        failures.append(error)
                        sweep['errors'].append(dict(seed=seed, phase='submit', **error_info(error)))
                # Collect every submitted future before rethrowing; no cancellation/retry.
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        record = future.result()
                    except Exception as error:
                        failures.append(error)
                        sweep['errors'].append(dict(seed=seed, phase='worker', **error_info(error)))
                        path = OUTPUT / f'seed_{seed}' / 'summary.json'
                        record = dict(**metadata(seed), configurations=[], winner=None)
                        if path.exists():
                            try:
                                record = json.loads(path.read_text())
                            except Exception as artifact_error:
                                # Explicit reporting failure; still collect other submitted seeds.
                                failures.append(artifact_error)
                                sweep['errors'].append(dict(seed=seed, phase='read_failed_worker_artifact',
                                                            **error_info(artifact_error)))
                        record.update(complete=False, status='failed', stop_status='incomplete')
                    sweep['seed_results'].append(record)
                    core.write_json(OUTPUT / 'summary.json', sweep)
            present = {record['seed'] for record in sweep['seed_results']}
            for seed in core.SEEDS:
                if seed not in present:
                    sweep['seed_results'].append(dict(**metadata(seed), status='not_submitted',
                        complete=False, configurations=[], winner=None))
            sweep['seed_results'].sort(key=lambda record: record['seed'])
            sweep['aggregates'] = summarize(sweep['seed_results'])
            if failures:
                raise RuntimeError(f'LR search failed: {len(failures)} seed submission/worker error(s)') from failures[0]
    finally:
        sweep['stop_status'] = sweep.get('status', 'failed')
        sweep['epochs_evaluated'] = sum(row.get('epochs_evaluated', 0) for row in sweep['seed_results'])
        sweep['optimizer_steps'] = sum(row.get('optimizer_steps', 0) for row in sweep['seed_results'])
        core.write_json(OUTPUT / 'summary.json', sweep)
        print_report(sweep)
        if 'wall_time_s' in sweep:
            seconds = sweep['wall_time_s']
            print(f"Sweep {sweep['stop_status']}: {seconds:.3f} s / {seconds / 60:.3f} min / {seconds / 3600:.3f} h", flush=True)


if __name__ == '__main__':
    main()