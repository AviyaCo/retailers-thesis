"""Five-fold nominal LR CV, fresh full-N refit, existing GREEN CCG and replay.

Orchestration only. Numerical implementations live in the original files bound
by backend.py/reporting.py. One fresh process per job prevents module collisions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
KINDS = ('R3', 'R10', 'shipping_R3')
SIZES = (50, 200, 800)
SEEDS = tuple(range(10))
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
LRS = (.005, .01, .05)
THREADS = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREADS)
os.environ.setdefault('GRB_LICENSE_FILE', '/data/home/aviyac/gurobi.lic')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def timed(record):
    start = time.perf_counter()
    record.update(status='running', started_at_utc=utc(), timing_unit='seconds',
                  timing_clock='time.perf_counter')
    try:
        yield record
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        record.update(ended_at_utc=utc(), wall_time_s=time.perf_counter() - start)


def folds(n, seed):
    from sklearn.model_selection import KFold
    import numpy as np
    result = list(KFold(n_splits=5, shuffle=True, random_state=100000 + seed).split(np.arange(n)))
    if sorted(int(i) for _, held in result for i in held) != list(range(n)):
        raise ValueError('CV held-out folds must partition all rows')
    for train, held in result:
        if set(train) & set(held) or sorted([*train, *held]) != list(range(n)):
            raise ValueError('Invalid train/held-out fold separation')
    return result


def choose_lr(rows):
    import math
    expected = {(f, lr) for f in range(5) for lr in LRS}
    keys = [(r['fold'], r['lr']) for r in rows]
    if (len(keys) != 15 or set(keys) != expected or
            any(r['status'] != 'completed' or not math.isfinite(r['mean']) for r in rows)):
        raise ValueError('Require all 15 completed finite CV arms')
    means = {lr: statistics.mean(r['mean'] for r in rows if r['lr'] == lr) for lr in LRS}
    return min(LRS, key=lambda lr: (means[lr], lr)), means


def backend_for(kind):
    # Identical imports before binding in prepare, nominal and CCG processes.
    import numpy
    import torch
    import sklearn.model_selection
    from backend import load_backend
    b = load_backend(kind)
    b.configure_worker()
    return b


def local_sources():
    return {str(p): sha(p) for p in sorted(ROOT.iterdir()) if p.suffix == '.py' or p.name == 'README.md'}


def verify_local(manifest):
    if manifest['local_sources'] != local_sources():
        raise ValueError('New experiment sources changed after freeze')


def prepare(kind, folder, workers):
    b = backend_for(kind)
    folder.mkdir(parents=True, exist_ok=False)
    sources = {**b.sources(), **local_sources()}
    record = dict(kind=kind, protocol=b.protocol, workers=dict(nominal=workers, ccg=workers),
        source_sha256=sources, local_sources=local_sources(), created_at_utc=utc(),
        folds=dict(implementation='sklearn.model_selection.KFold', n_splits=5,
                   shuffle=True, random_state='100000 + data seed'),
        environment=dict(host=platform.node(), platform=platform.platform(),
            python=platform.python_version(), affinity_cpus=sorted(os.sched_getaffinity(0)),
            thread_environment=THREADS, packages={p: version(p) for p in
            ('numpy', 'scipy', 'torch', 'scikit-learn', 'gurobipy')}))
    b.verify_sources(record)
    write(folder / 'manifest.json', record)
    return dict(status='completed', path=str(folder / 'manifest.json'), sha256=sha(folder / 'manifest.json'))


def timing_ref(record, path):
    keys = ('status', 'started_at_utc', 'ended_at_utc', 'wall_time_s', 'epochs_evaluated',
            'optimizer_steps', 'best_epoch', 'stop_reason', 'solver_status', 'solver_runtime_s', 'iterations')
    return dict({k: record[k] for k in keys if k in record}, path=str(path), sha256=sha(path))


def fit(b, initial, x1, x2, lr, folder, meta):
    model = b.model_from_state(initial)
    if b.state_hash(model) != b.state_hash(b.model_from_state(initial)):
        raise ValueError('Initial state changed')
    record = dict(meta, lr_net=lr)
    model = b.train_model(model, lambda m: m.per_sample_cost(x1, x2).mean(), lr, folder, record)
    replay = dict(meta, method='selected training-loss replay')
    try:
        with b.core.timed(replay), b.torch.no_grad():
            model.eval()
            values = model.per_sample_cost(x1, x2)
            if values.shape != (len(x1),) or not bool(b.torch.isfinite(values).all()):
                raise ValueError('Invalid training replay')
            mean = float(values.mean())
            if abs(mean - record['selected_training_loss']) > 1e-6:
                raise ValueError('Training-best checkpoint replay mismatch')
            saved = b.torch.load(folder / 'selected.pth', map_location='cpu', weights_only=True)
            if b.state_hash(b.model_from_state(saved)) != b.state_hash(model):
                raise ValueError('Saved/returned checkpoint mismatch')
            replay.update(mean=mean, state_sha256=b.state_hash(model), sample_count=len(x1))
    finally:
        b.core.write_json(folder / 'training_replay.json', replay)
    return model, mean, timing_ref(record, folder / 'training.json')


def nominal(kind, n, seed, folder, manifest_path):
    b = backend_for(kind)
    manifest = read(manifest_path)
    verify_local(manifest)
    b.verify_sources(manifest)
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(kind=kind, **b.metadata(n, seed, method='CV and full-N nominal refit'),
                  worker_count=manifest['workers']['nominal'], phases={}, cv_rows=[], test='not accessed')
    np, torch = b.np, b.torch
    try:
        with timed(record):
            data = b.generate(n, seed)
            data_hash = hashlib.sha256(b''.join(np.asarray(data[k], dtype='<f8').tobytes()
                                                for k in ('xi1', 'xi2'))).hexdigest()
            np.savez(folder / 'data.npz', **data)
            record.update(data_sha256=data_hash, raw_data_file_sha256=sha(folder / 'data.npz'))
            splits = folds(n, seed)
            write(folder / 'folds.json', [dict(fold=f, train_ids=t.tolist(), heldout_ids=v.tolist())
                                         for f, (t, v) in enumerate(splits)])
            record['folds_sha256'] = sha(folder / 'folds.json')
            cv = record['phases']['cross_validation'] = dict(folds=[])
            with timed(cv):
                for f, (train_ids, held_ids) in enumerate(splits):
                    fold_dir = folder / f'fold_{f}'
                    fold_dir.mkdir()
                    meta = dict(b.metadata(n, seed, method='nominal CV'), Ntrain=len(train_ids),
                                Nval=len(held_ids), fold=f, worker_count=record['worker_count'])
                    entry = dict(meta, initialization={}, arms=[])
                    cv['folds'].append(entry)
                    try:
                        with timed(entry):
                            # No held-out array is passed into SAA, model initialization or loss.
                            init = entry['initialization']
                            init.update(meta)
                            try:
                                q10, q1 = b.solve_saa(data['xi1'][train_ids], data['xi2'][train_ids], init)
                            finally:
                                b.core.write_json(fold_dir / 'saa.json', init)
                            initial_model = b.make_model(q10, q1)
                            initial = b.weights(initial_model)
                            initial_hash = b.state_hash(initial_model)
                            torch.save(initial, fold_dir / 'initial.pth')
                            x1, x2 = (torch.tensor(data[k][train_ids], dtype=torch.float32) for k in ('xi1', 'xi2'))
                            arms = []
                            for lr in LRS:
                                arm_dir = fold_dir / f'lr_{lr:g}'
                                model, mean, timing = fit(b, initial, x1, x2, lr, arm_dir, meta)
                                arm = dict(lr=lr, training=timing, train_mean=mean,
                                    initial_state_sha256=initial_hash, state_sha256=b.state_hash(model),
                                    checkpoint=str(arm_dir / 'selected.pth'),
                                    checkpoint_sha256=sha(arm_dir / 'selected.pth'))
                                arms.append(arm)
                                entry['arms'].append(arm)
                            # All three arms frozen before held-out scoring; no epoch tuning on held-out rows.
                            write(fold_dir / 'frozen_arms.json', arms)
                            for arm in arms:
                                if sha(arm['checkpoint']) != arm['checkpoint_sha256']:
                                    raise ValueError('CV checkpoint changed')
                                model = b.model_from_state(torch.load(arm['checkpoint'], map_location='cpu', weights_only=True))
                                score = dict(meta, lr=arm['lr'], method='native hard held-out CV score')
                                try:
                                    v1, v2 = (torch.tensor(data[k][held_ids], dtype=torch.float32) for k in ('xi1', 'xi2'))
                                    b.evaluate(model, v1, v2, score)
                                finally:
                                    b.core.write_json(Path(arm['checkpoint']).parent / 'cv_score.json', score)
                                arm['validation'] = timing_ref(score, Path(arm['checkpoint']).parent / 'cv_score.json')
                                record['cv_rows'].append(dict(fold=f, lr=arm['lr'], status=score['status'], mean=score['mean']))
                    finally:
                        b.core.write_json(fold_dir / 'summary.json', entry)
            selection = record['phases']['lr_selection'] = {}
            with timed(selection):
                chosen, means = choose_lr(record['cv_rows'])
                selection.update(lr=chosen, mean_hard_cost_by_lr=means, exact_tie='smaller LR')
            write(folder / 'lr_selection.json', selection)
            # Fresh initialization from ALL rows, never a fold checkpoint.
            meta = dict(b.metadata(n, seed, method='full-N nominal'), worker_count=record['worker_count'])
            init = record['phases']['fullN_initialization'] = dict(meta)
            try:
                q10, q1 = b.solve_saa(data['xi1'], data['xi2'], init)
            finally:
                b.core.write_json(folder / 'fullN_saa.json', init)
            initial = b.weights(b.make_model(q10, q1))
            torch.save(initial, folder / 'fullN_initial.pth')
            x1, x2 = (torch.tensor(data[k], dtype=torch.float32) for k in ('xi1', 'xi2'))
            model, mean, training = fit(b, initial, x1, x2, chosen, folder / 'fullN', meta)
            record['phases']['fullN_training'] = training
            training_data = folder / 'training_data.json'
            write(training_data, dict(centers=np.concatenate([data['xi1'], data['xi2']], axis=1).tolist(),
                original_sample_ids=list(range(n)), data_sha256=data_hash))
            cp = folder / 'fullN/selected.pth'
            result = dict(Ntotal=n, seed=seed, lr_net=chosen, checkpoint=str(cp), checkpoint_sha256=sha(cp),
                state_sha256=b.state_hash(model), train_mean=mean, training_timing=training,
                training_data_path=str(training_data), training_data_sha256=sha(training_data), data_sha256=data_hash)
            write(folder / 'nominal.json', result)
            record.update(selected_lr=chosen, nominal=result)
            verify_local(manifest)
            b.verify_sources(manifest)
    finally:
        b.core.write_json(folder / 'summary.json', record)
    return dict(status='completed', kind=kind, Ntotal=n, seed=seed,
                summary_path=str(folder / 'summary.json'), summary_sha256=sha(folder / 'summary.json'),
                wall_time_s=record['wall_time_s'], nominal=result)


def worker(spec):
    action = spec['action']
    folder = Path(spec['folder'])
    if action == 'prepare':
        return prepare(spec['kind'], folder, spec['workers'])
    if action == 'nominal':
        return nominal(spec['kind'], spec['n'], spec['seed'], folder, spec['manifest'])
    if action == 'ccg':
        b = backend_for(spec['kind'])
        verify_local(read(spec['manifest']))
        result = b.run_configuration(spec['n'], spec['seed'], spec['radius'], folder,
                                     spec['nominal'], spec['manifest'])
        result['kind'] = spec['kind']
        return result
    if action == 'evaluate':
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        from reporting import evaluate_configuration
        return evaluate_configuration(spec['kind'], spec['compact'], folder)
    raise ValueError('Unrecognized worker action')


def invoke(spec, output):
    name = spec['id']
    jobs = output / 'jobs'
    path = jobs / f'{name}.json'
    write(path, spec)
    result_path = jobs / f'{name}.result.json'
    receipt = dict(id=name, action=spec['action'], command=[sys.executable, '-B', '-u',
        str(ROOT / 'runner.py'), '--worker', str(path)], cwd=str(ROOT), log=str(jobs / f'{name}.log'))
    try:
        with timed(receipt), Path(receipt['log']).open('x') as log:
            process = subprocess.run(receipt['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            receipt['returncode'] = process.returncode
            if process.returncode != 0:
                raise RuntimeError(f"Worker {name} exited {process.returncode}; see {receipt['log']}")
            result = read(result_path)
            if result['status'] != 'completed':
                raise RuntimeError(f"Worker {name} {result['status']}; no fallback")
    except Exception:
        # Preserve nonconvergence/failed solver timings, not merely a nonzero exit.
        result = read(result_path) if result_path.is_file() else dict(status='failed')
        if result.get('status') == 'completed':
            result['status'] = 'failed'
        result.update(id=name, result_path=str(result_path), invocation_error=receipt.get('error'))
        for key in ('kind', 'seed', 'radius'):
            if key in spec:
                result.setdefault(key, spec[key])
        if 'n' in spec:
            result.setdefault('Ntotal', spec['n'])
        raw_summary = Path(spec['folder']) / 'summary.json'
        if raw_summary.is_file() and 'summary_path' not in result:
            raw = read(raw_summary)
            result.update(summary_path=str(raw_summary), summary_sha256=sha(raw_summary),
                          measured_model_wall_time_s=raw.get('wall_time_s'))
    finally:
        write(jobs / f'{name}.invocation.json', receipt)
    if result['status'] != 'completed':
        result.update(invocation_path=str(jobs / f'{name}.invocation.json'),
                      invocation_wall_time_s=receipt['wall_time_s'])
    print(utc(), name, result['status'], flush=True)
    return result


def stage(name, specs, output, summary, workers):
    record = summary['phases'][name] = dict(planned=len(specs), workers=workers, results=[])
    try:
        with timed(record), ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(invoke, spec, output) for spec in specs]
            for future in as_completed(futures):
                record['results'].append(future.result())
                write(output / 'summary.json', summary)
            if len(record['results']) != len(specs) or any(r['status'] != 'completed' for r in record['results']):
                raise RuntimeError(f'{name}: incomplete/failed grid; next stage not permitted')
    finally:
        write(output / 'summary.json', summary)
    return record['results']


def aggregate(values):
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), unit='seconds')


def runtime_report(output, summary):
    rows = []
    for stage_name in ('nominal', 'ccg', 'evaluation'):
        results = summary.get('phases', {}).get(stage_name, {}).get('results', [])
        for kind in KINDS:
            for n in SIZES:
                groups = RADII if stage_name in ('ccg', 'evaluation') else (None,)
                for rho in groups:
                    matches = [r for r in results if r.get('status') == 'completed'
                        and r.get('kind') == kind and r.get('Ntotal') == n
                        and (rho is None or r.get('radius') == rho)]
                    if not matches:
                        continue
                    phases = {'total': [r['wall_time_s'] for r in matches]}
                    if stage_name == 'nominal':
                        saved = [read(r['summary_path']) for r in matches]
                        for key in ('cross_validation', 'lr_selection', 'fullN_initialization', 'fullN_training'):
                            phases[key] = [r['phases'][key]['wall_time_s'] for r in saved]
                        for label, key in (('CV_SAA', 'initialization'),):
                            phases[label] = [sum(f[key]['wall_time_s'] for f in r['phases']['cross_validation']['folds']) for r in saved]
                        phases['CV_training'] = [sum(a['training']['wall_time_s'] for f in r['phases']['cross_validation']['folds'] for a in f['arms']) for r in saved]
                        phases['CV_evaluation'] = [sum(a['validation']['wall_time_s'] for f in r['phases']['cross_validation']['folds'] for a in f['arms']) for r in saved]
                    if stage_name == 'ccg':
                        phases.update(AP=[r['ap_total_time_s'] for r in matches], MP=[r['mp_total_time_s'] for r in matches])
                    for phase, values in phases.items():
                        rows.append(dict(stage=stage_name, phase=phase, kind=kind, Ntotal=n, radius=rho, **aggregate(values)))
    failures = [dict(stage=k, result=r) for k, v in summary.get('phases', {}).items()
                for r in v.get('results', []) if r.get('status') != 'completed']
    report = dict(status=summary['status'], runtime_rows=rows, failed_or_incomplete=failures,
        overall_wall_time_s=summary.get('wall_time_s'), workers=summary['workers'],
        environment=summary['environment'], SRO_FA='not run', blue='not run',
        warning='CCG excludes shared nominal/CV initialization; concurrent walls are NOT sweep elapsed.')
    write(output / 'runtime_report.json', report)
    lines = ['# Full-N green CV experiment', '', 'Status: ' + summary['status'], '',
        'Full-N refit after five-fold nominal-LR CV; original numerical code reused. Previous validation runs unchanged.',
        'Test-curve minima are descriptive hindsight on the fixed grid, not deployable tuning or fresh confirmatory results.',
        '', '## Wall-clock runtimes', '',
        '| Kind | N | Radius | Stage / phase | Runs | Mean s (min) | Median s | Min s | Max s |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['kind']} | {r['Ntotal']} | {r['radius']} | {r['stage']} / {r['phase']} | {r['count']} | {r['mean']:.3f} ({r['mean']/60:.3f}) | {r['median']:.3f} | {r['minimum']:.3f} | {r['maximum']:.3f} |")
    lines += ['', 'SRO-FA and blue: not run. Failed/incomplete jobs are excluded from completed aggregates and preserved in runtime_report.json.',
              f"Overall measured sweep seconds: {summary.get('wall_time_s', 'not yet available')}; workers: {summary['workers']}.",
              'CCG times exclude reused nominal/CV time; no sum of concurrent walls is labelled sweep elapsed.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    return report


def run(output, workers):
    if not 1 <= workers <= min(16, len(os.sched_getaffinity(0))):
        raise ValueError('Workers must be 1..16 within process CPU affinity')
    if output.parent != ROOT / 'runs' or output.is_symlink():
        raise ValueError('Output must be a fresh direct child of this experiment/runs')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'jobs').mkdir()
    summary = dict(experiment=ROOT.name, status='running', workers=workers, phases={},
        environment=dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                         affinity_cpus=sorted(os.sched_getaffinity(0)), threads=THREADS),
        local_sources=local_sources(), planned=dict(nominal_CV_fits=1350, fullN_refits=90, CCG=810),
        test_accessed=False, previous_results_modified=False)
    try:
        with timed(summary):
            prepared = stage('prepare', [dict(id='prepare_' + k, action='prepare', kind=k,
                folder=str(output / k), workers=workers) for k in KINDS], output, summary, 3)
            manifests = {k: str(output / k / 'manifest.json') for k in KINDS}
            nominals = stage('nominal', [dict(id=f'nominal_{k}_N{n}_s{s}', action='nominal', kind=k,
                n=n, seed=s, folder=str(output / k / f'N{n}/nominal/seed_{s}'), manifest=manifests[k])
                for k in KINDS for n in SIZES for s in SEEDS], output, summary, workers)
            lookup = {(r['kind'], r['Ntotal'], r['seed']): r['nominal'] for r in nominals}
            trained = stage('ccg', [dict(id=f'ccg_{k}_N{n}_s{s}_r{rho:g}', action='ccg', kind=k,
                n=n, seed=s, radius=rho, nominal=lookup[k, n, s], manifest=manifests[k],
                folder=str(output / k / f'N{n}/ccg')) for k in KINDS for n in SIZES
                for s in SEEDS for rho in RADII], output, summary, workers)
            from reporting import _grid, source_bindings, publish
            _grid(trained)
            # Every checkpoint and own pool is fixed before the first test generation.
            for r in trained:
                if sha(r['summary_path']) != r['summary_sha256'] or sha(r['selected_checkpoint']) != r['selected_checkpoint_sha256']:
                    raise ValueError('Training seal checkpoint/summary changed')
                saved = read(r['summary_path'])
                if saved['selected_candidate_id'] != saved['stopping_candidate_id']:
                    raise ValueError('Not a STOP selection')
            gate_path = output / 'training_seal.json'
            write(gate_path, dict(status='sealed', sealed_at_utc=utc(), allow_test_regeneration=True,
                source_bindings=source_bindings(), evaluation_workers=workers, configurations=trained))
            gate = dict(path=str(gate_path), sha256=sha(gate_path))
            summary.update(evaluation_gate=gate, test_accessed=True)
            write(output / 'summary.json', summary)
            evaluations = stage('evaluation', [dict(id=f"eval_{r['kind']}_N{r['Ntotal']}_s{r['seed']}_r{r['radius']:g}",
                action='evaluate', kind=r['kind'], compact=dict(r, evaluation_gate=gate),
                folder=str(output / r['kind'] / f"N{r['Ntotal']}/evaluation/seed_{r['seed']}/radius_{r['radius']:g}"))
                for r in trained], output, summary, workers)
            publication = summary['phases']['publication'] = {}
            with timed(publication):
                result = publish(output / 'published', evaluations, dict(evaluation_gate=gate,
                    final_runtime_report_path=str(output / 'runtime_report.json'),
                    runtime_note='Final elapsed time is written after publication; see the final runtime report.',
                    workers=workers, environment=summary['environment']))
                publication['artifacts'] = result['artifacts']
    finally:
        summary['runtime_report'] = runtime_report(output, summary)
        write(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if args.worker:
        record = {}
        result = None
        try:
            with timed(record):
                result = worker(read(args.worker))
            if result['status'] != 'completed':
                return 1
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            write(args.worker.with_suffix('.process.json'), record)
            if result is not None:
                write(args.worker.with_suffix('.result.json'), result)
        return 0
    if args.output is None:
        parser.error('--output required')
    run(args.output.resolve(), args.workers)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())