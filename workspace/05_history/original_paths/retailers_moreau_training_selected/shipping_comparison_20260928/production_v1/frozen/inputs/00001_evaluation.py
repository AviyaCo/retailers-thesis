"""Fresh common Gurobi replay; fixed policies only, no training or selection."""
import json
import os
from pathlib import Path
import resource
import time
import traceback
import common as c
from common import np, check
from plot_comparison import NEURAL_FEE

_MANIFEST = _JOBS = _ARRAYS = _MANIFEST_PATH = _MANIFEST_SHA = None


def initialize(manifest_path):
    global _MANIFEST, _JOBS, _ARRAYS, _MANIFEST_PATH, _MANIFEST_SHA
    c.initialize_worker()
    _MANIFEST_PATH = Path(manifest_path)
    _MANIFEST_SHA = c.sha(_MANIFEST_PATH)
    check(_MANIFEST_SHA == _MANIFEST_PATH.with_suffix('.sha256').read_text().strip(), 'Changed manifest seal')
    _MANIFEST = c.read(manifest_path)
    check(_MANIFEST['status'] == 'frozen' and _MANIFEST['versions'] == c.versions(), 'Invalid manifest environment')
    check(_MANIFEST['threads'] == 1 and _MANIFEST['workers'] == 16, 'Changed concurrency')
    for path in _MANIFEST['files']:
        if Path(path).suffix == '.py' or Path(path).name == 'README.md':
            c.bound_file(_MANIFEST, path)
    _JOBS = {j['id']: j for j in _MANIFEST['jobs']}
    check(len(_JOBS) == 750, 'Incomplete manifest jobs')
    _ARRAYS = []
    for info in _MANIFEST['test']['arrays'].values():
        check(c.sha(info['file']) == info['file_sha256'], 'Changed test file')
    for name in ('xi1', 'xi2'):
        info = _MANIFEST['test']['arrays'][name]
        a = np.load(info['file'], mmap_mode='r', allow_pickle=False)
        check(a.shape == (10000, 3) and a.dtype == np.float64 and np.isfinite(a).all() and (a >= 0).all(),
              'Invalid test array')
        import hashlib
        check(hashlib.sha256(a.tobytes()).hexdigest() == info['array_sha256'], 'Changed test array bytes')
        _ARRAYS.append(a)


def evaluate(model, x1, x2, name, record, folder, sizes=None):
    """One all-rows float32 target call, then unchanged fresh QP per row."""
    with c.phase(record['phases'], name) as phase:
        phase.update(sample_count=len(x1), solver_threads=1, iterations=0., barrier_iterations=0,
                     solves_attempted=0, solves_accepted=0)
        with c.phase(phase, 'targets'):
            z, q10, q1r = model.targets(x1)
        q2 = np.empty((len(x1), 3), dtype=np.float64)
        with c.phase(phase, 'projection'), (folder / f'{name}_solves.jsonl').open('x') as journal, c.c.gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            with c.c.gp.Model(env=env) as settings:
                phase['solver_settings'] = {k: settings.getParamInfo(k)[2] for k in
                    ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'Method', 'Threads')}
            for i in range(len(x1)):
                diag = dict(row=i, start_utc=c.utc(), accepted=False)
                record['solves_attempted'] += 1
                phase['solves_attempted'] += 1
                try:
                    q2[i] = c.strict_project(z[i], q10, env, diag)
                    check(diag['status'] == 2 and diag['accepted'], 'Unaccepted projection')
                    phase['solves_accepted'] += 1
                    record['solves_accepted'] += 1
                    # Preserve the raw sign/support that determines hard fees.
                    diag.update(q2=q2[i].tolist(), positive_support=(q2[i] > 0).tolist())
                finally:
                    diag['end_utc'] = c.utc()
                    status = str(diag.get('status'))
                    record['status_counts'][status] = record['status_counts'].get(status, 0) + 1
                    phase['iterations'] += diag.get('iterations') or 0
                    phase['barrier_iterations'] += diag.get('barrier_iterations') or 0
                    journal.write(json.dumps(diag, allow_nan=False) + '\n')
                    if not diag['accepted']:
                        record['failed_projection'] = diag
        with c.phase(phase, 'cost'):
            costs = c.scenario_costs(q10, q1r, q2, x1, x2)
            check(costs.shape == (len(x1),) and costs.dtype == np.float64 and np.isfinite(costs).all(), 'Invalid costs')
            np.savez(folder / f'{name}_raw.npz', x1=x1, x2=x2, targets=z, q10=np.array(q10), q1r=q1r,
                     shipments=q2, costs=costs)
            phase['raw_path'] = str(folder / f'{name}_raw.npz')
            phase['raw_sha256'] = c.sha(phase['raw_path'])
            phase['fee_rule'] = NEURAL_FEE
            if sizes is None:
                record['mean_oos'] = float(costs.mean())
            else:
                record['finite_pool_mean'], record['per_center_maxima'] = c.finite_pool_cost(costs, sizes)
                saved = record['saved_native_finite_pool_mean']
                record['difference_from_saved_native'] = None if saved is None else record['finite_pool_mean'] - saved


def worker(job_id, output):
    spec = _JOBS[job_id]
    folder = Path(output) / job_id
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    # The manifest owns full scenario arrays; records reference its immutable seal.
    record = {k: v for k, v in spec.items() if k not in ('pools', 'centers')}
    record.update(status='running', start_utc=c.utc(), phases={}, status_counts={},
        solves_attempted=0, solves_accepted=0, mean_oos=None, finite_pool_mean=None,
        manifest_sha256=_MANIFEST_SHA, test=_MANIFEST['test'], fee_rule=NEURAL_FEE,
        worker_count=16, solver_threads=1, torch_threads=c.torch.get_num_threads(),
        torch_interop_threads=c.torch.get_num_interop_threads(), pid=os.getpid())
    try:
        with c.phase(record['phases'], 'input_loading'):
            check(record['torch_threads'] == record['torch_interop_threads'] == 1, 'Wrong worker threads')
            for source in spec['input_files']:
                c.bound_file(_MANIFEST, source)
            model = c.load_model(c.bound_file(_MANIFEST, spec['checkpoint']), spec['method'])
            check(c.c.state_hash(model) == spec['state_sha256'], 'Learned state mismatch')
            pools = [np.asarray(p, dtype=np.float64) for p in spec['pools']]
            check(c.pool_digest(pools) == spec['pool_sha256'] and [len(p) for p in pools] == spec['sizes'],
                  'Pool mismatch')
        evaluate(model, *_ARRAYS, 'oos', record, folder)
        points = np.concatenate(pools)
        evaluate(model, points[:, :3], points[:, 3:], 'is', record, folder, spec['sizes'])
        check(c.c.state_hash(model) == spec['state_sha256'], 'Replay changed learned state')
        check(c.sha(_MANIFEST_PATH) == _MANIFEST_SHA, 'Manifest changed')
        expected = 10000 + sum(spec['sizes'])
        check(record['solves_attempted'] == record['solves_accepted'] == expected
              and record['status_counts'] == {'2': expected}, 'Incomplete strict replay')
        record['status'] = 'completed'
    except BaseException:
        record.update(status='failed', error=traceback.format_exc(), mean_oos=None, finite_pool_mean=None)
    finally:
        record.update(end_utc=c.utc(), wall_time_s=time.perf_counter()-start,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        c.write(folder / 'result.json', record)
    return dict(id=job_id, status=record['status'], wall_time_s=record['wall_time_s'],
                result_path=str(folder / 'result.json'), result_sha256=c.sha(folder / 'result.json'))


def verify_result(spec, compact, manifest):
    """Authenticate raw inputs, physical residuals, journals and cost aggregates."""
    check(compact['id'] == spec['id'] and compact['status'] == 'completed'
          and c.sha(compact['result_path']) == compact['result_sha256'], 'Failed or changed worker result')
    r = c.read(compact['result_path'])
    check(r['status'] == 'completed' and r['manifest_sha256'] == manifest['seal'], 'Unbound result')
    total = 10000 + sum(spec['sizes'])
    check(r['solves_attempted'] == r['solves_accepted'] == total and r['status_counts'] == {'2': total},
          'Wrong global solver counts')
    for key, value in spec.items():
        if key not in ('pools', 'centers'):
            check(r[key] == value, f'Worker identity mismatch: {key}')
    test = []
    for name in ('xi1', 'xi2'):
        info = manifest['test']['arrays'][name]
        check(c.sha(info['file']) == info['file_sha256'], 'Changed authoritative test file')
        a = np.load(info['file'], mmap_mode='r', allow_pickle=False)
        check(a.shape == (10000, 3) and a.dtype == np.float64 and np.isfinite(a).all(), 'Invalid authoritative test')
        test.append(a)
    pools = [np.asarray(p, dtype=np.float64) for p in spec['pools']]
    check(len(pools) == len(spec['sizes']) and [len(p) for p in pools] == spec['sizes'], 'Invalid authoritative pools')
    points = np.concatenate(pools)
    check(points.shape == (sum(spec['sizes']), 6) and c.pool_digest(pools) == spec['pool_sha256'], 'Changed authoritative pools')
    folder = Path(compact['result_path']).parent
    for name, count in (('oos', 10000), ('is', sum(spec['sizes']))):
        phase = r['phases'][name]
        check(phase['status'] == 'completed' and phase['solves_accepted'] == phase['solves_attempted'] == count,
              'Incomplete phase')
        check(c.sha(phase['raw_path']) == phase['raw_sha256'], 'Changed raw arrays')
        with np.load(phase['raw_path'], allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        check(all(np.isfinite(a).all() for a in raw.values()), 'Nonfinite raw evidence')
        check(raw['shipments'].shape == (count, 3) and raw['shipments'].dtype == np.float64, 'Wrong raw projection count')
        check(raw['targets'].shape == raw['x1'].shape == raw['x2'].shape == (count, 3), 'Wrong raw input count')
        check(raw['q10'].shape == () and raw['q10'] >= 0 and raw['q1r'].shape == (3,)
              and (raw['q1r'] >= 0).all(), 'Invalid raw inventory')
        expected_inputs = test if name == 'oos' else (points[:, :3], points[:, 3:])
        for key, expected in zip(('x1', 'x2'), expected_inputs):
            check(raw[key].dtype == np.float64 and np.array_equal(raw[key], expected), 'Raw scenarios differ from sealed inputs')
        residuals = np.maximum(np.maximum(0., -raw['shipments'].min(axis=1)),
                               raw['shipments'].sum(axis=1) - raw['q10'].item())
        check(np.isfinite(residuals).all() and (residuals <= c.PHYSICAL_TOL).all(), 'Raw physical violation')
        costs = c.scenario_costs(raw['q10'].item(), raw['q1r'], raw['shipments'], raw['x1'], raw['x2'])
        check(np.array_equal(costs, raw['costs']) and np.isfinite(costs).all(), 'Raw shipping arithmetic mismatch')
        with (folder / f'{name}_solves.jsonl').open() as stream:
            rows = 0
            for i, line in enumerate(stream):
                d = json.loads(line)
                check(i < count and d['row'] == i and d['accepted'] and d['status'] == 2, 'Unaccepted/out-of-order journal')
                check(d['physical_residual'] == residuals[i], 'Journal/raw physical residual mismatch')
                check(np.array_equal(d['q2'], raw['shipments'][i])
                      and d['positive_support'] == (raw['shipments'][i] > 0).tolist(), 'Journal/raw mismatch')
                rows += 1
        check(rows == count, 'Incomplete journal')
        expected = float(costs.mean()) if name == 'oos' else c.finite_pool_cost(costs, spec['sizes'])[0]
        check(expected == r['mean_oos' if name == 'oos' else 'finite_pool_mean'], 'Wrong aggregate')
    return {**spec, **r}