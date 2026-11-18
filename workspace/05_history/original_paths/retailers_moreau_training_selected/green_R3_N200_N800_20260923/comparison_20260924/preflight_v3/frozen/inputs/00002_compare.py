"""Frozen, descriptive R3/N50 comparison. No training, selection, or fallback.

Commands: freeze (no scenarios), run (full held-out evaluation), plot (completed
outputs only). Tests/preflight are separate and never call the test generator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

# Set before importing numerical libraries, including in spawned workers.
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('GRB_LICENSE_FILE', '/data/home/aviyac/gurobi.lic')

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy
from scipy.stats import truncnorm
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
OLD = Path('/data/home/aviyac/retailers_problem')
SWEEP = OLD / 'R3/N50/rho_sweep_net16_v2'
NOMINAL = OLD / 'R3/N50/saainit_net16_v2'
OLD_EVAL = SWEEP / 'eval_sweep_net16_v2_10k.py'
CACHE = SWEEP / 'sweep_eval_net16_v2_10k.json'
SRO = OLD / 'R3/N50/sro_fa/sro_fa_results_N50_R3.json'
COND = OLD / 'R3/saa_cond_lb_large.json'
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
SEEDS = tuple(range(10))
FINAL_ITER_040 = {0: 3, 1: 5, 2: 4, 3: 5, 4: 7, 5: 4, 6: 7, 7: 5, 8: 5, 9: 5}
N_TEST, TEST_SEED, WORKERS, PHYSICAL_TOL = 10000, 9999, 8, 1e-4
FROZEN = HERE / 'frozen'
RESULTS = HERE / 'results'


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


@contextmanager
def phase(record, name):
    t = time.perf_counter()
    entry = record[name] = {'start_utc': utc(), 'status': 'running'}
    try:
        yield entry
    except BaseException:
        entry['status'] = 'failed'
        raise
    else:
        entry['status'] = 'completed'
    finally:
        entry.update(end_utc=utc(), wall_time_s=time.perf_counter() - t)


def initialize_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


class Model(nn.Module):
    """Only the eight learned tensors; no differentiable projection backend."""
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(nn.Linear(7, 16), nn.ReLU(),
                                       nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 3))
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(3))

    @torch.no_grad()
    def targets(self, xi1):
        # Exact legacy precision/order: q1r.sum() in cost is still float32.
        x = torch.tensor(xi1, dtype=torch.float32)
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        features = torch.cat([x, torch.tensor(q1r).unsqueeze(0).expand(len(x), -1),
                              torch.full((len(x), 1), q10)], dim=1)
        z = self.fc_layers(features).numpy()
        if not np.isfinite(z).all() or not np.isfinite([q10, *q1r]).all():
            raise FloatingPointError('Nonfinite target or inventory')
        return z, q10, q1r


def load_model(path, method):
    # Local, trusted historical checkpoints contain sparse projection buffers.
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state = payload['model_state_dict'] if method == 'old' else payload
    model = Model()
    expected = model.state_dict()
    extra = set(state) - set(expected)
    # These two known CvxpyLayer buffers are not learned parameters. No wildcard.
    if method == 'old':
        if extra != {'proj_layer.q', 'proj_layer.A'}:
            raise ValueError(f'Unexpected legacy buffer schema: {extra}')
        if tuple(state['proj_layer.q'].shape) != (5, 5) or tuple(state['proj_layer.A'].shape) != (17, 5):
            raise ValueError('Unexpected legacy projection buffer dimensions')
    elif method != 'new' or extra:
        raise ValueError(f'Unexpected learned schema: {extra}')
    if set(expected) - set(state):
        raise ValueError(f'Missing learned tensors: {set(expected) - set(state)}')
    learned = {k: state[k] for k in expected}
    for k, v in learned.items():
        if v.dtype != torch.float32 or v.shape != expected[k].shape or not torch.isfinite(v).all():
            raise ValueError(f'Invalid learned tensor: {k}')
    model.load_state_dict(learned, strict=True)
    return model.eval()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gen(n, seed):
    """Legacy global NumPy RNG and SciPy truncnorm, not default_rng."""
    np.random.seed(seed)
    xi1 = truncnorm.rvs((0.0 - 6.0) / 2.5, np.inf, loc=6.0, scale=2.5, size=(n, 3))
    mu2 = 2 * (6.0 - xi1) ** 2
    xi2 = truncnorm.rvs((0.0 - mu2) / 2.5, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2


def scenario_costs(q10, q1r, q2, xi1, xi2):
    s1 = .25 * (q10 + q1r.sum()) + .05 * q10
    v = np.maximum(.5 * (xi1 + xi2 - q1r - q2) - .05 * q2,
                   np.maximum(.05 * (q1r - xi1 - xi2),
                              .5 * (xi1 - q1r) - .05 * xi2))
    return s1 + v.sum(axis=1)


def old_choice(rho, seed, directories, nominal=NOMINAL):
    """Exact legacy precedence: real rho directory wins over special .4 source."""
    if rho == 0:
        return nominal / f'seed_{seed}_nominal.pth', 0, None
    if rho in directories:
        folder = directories[rho]
        summary = folder / f'seed_{seed}_summary.json'
        fmp = read_json(summary).get('final_mp_iter', 0)
        if not isinstance(fmp, int) or fmp < 0:
            raise ValueError('Invalid final_mp_iter')
        path = (nominal / f'seed_{seed}_nominal.pth' if fmp == 0 else
                folder / f'seed_{seed}_mp_iter{fmp}.pth')
        return path, fmp, summary
    if rho == .4:
        i = FINAL_ITER_040[seed]
        return nominal / f'seed_{seed}_mp_iter{i}.pth', i, None
    raise FileNotFoundError(f'No original source for rho={rho}')


def plotted_radii(cache):
    # Same domain and nonempty-array rule as extract_curve; values never rank choices.
    radii = []
    for key in sorted(cache['sweep'], key=float):
        entry = cache['sweep'][key]
        if 0 < float(key) <= 1.5 and len(entry['rob_oos']):
            if entry['n_seeds'] != 10 or len(entry['rob_oos']) != 10:
                raise ValueError(f'Original chart lacks all ten seeds at {key}')
            radii.append(float(key))
    return radii


def freeze():
    """Freeze before ANY test generation. Existing freeze is never overwritten."""
    start = time.perf_counter()
    FROZEN.mkdir(exist_ok=False)
    (FROZEN / 'inputs').mkdir()
    manifest = dict(experiment='matched_R3_N50_OOS', start_utc=utc(), status='freezing',
                    n_test=N_TEST, test_seed=TEST_SEED, seeds=list(SEEDS), workers=WORKERS,
                    threads=1, physical_tolerance=PHYSICAL_TOL, jobs=[], files={},
                    versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                  torch=torch.__version__, gurobi=list(gp.gurobi.version())))

    def capture(path):
        path = Path(path).resolve()
        key = str(path)
        if key not in manifest['files']:
            digest = sha(path)
            dest = FROZEN / 'inputs' / f'{len(manifest["files"]):04d}_{path.name}'
            shutil.copyfile(path, dest)
            if sha(dest) != digest or sha(path) != digest:
                raise RuntimeError(f'Input changed while freezing: {path}')
            dest.chmod(0o444)
            manifest['files'][key] = dict(sha256=digest, copy=str(dest.relative_to(FROZEN)),
                                         mtime_ns=path.stat().st_mtime_ns)
        return manifest['files'][key]

    try:
        for path in (HERE / 'compare.py', HERE / 'test_compare.py', HERE / 'preflight.py', HERE / 'README.md',
                     OLD_EVAL, OLD / 'plot_combined_N50_R3.py',
                     NOMINAL / 'run_all_net16_v2.py', SWEEP / 'run_rho_seed_net16_v2.py',
                     NEW / 'core.py', NEW / 'ccg.py', NEW / 'run_ccg.py',
                     NEW / 'ccg_results/summary.json', NEW / 'ccg_results/provenance.json',
                     CACHE, SRO, COND):
            capture(path)
        cache = read_json(CACHE)
        if cache['meta'] != {**cache['meta'], 'N_train': 50, 'N_test': 10000,
                             'seed_test': 9999, 'seeds': list(SEEDS)}:
            raise ValueError('Historical cache protocol mismatch')
        old_radii = plotted_radii(cache)
        runtime = cache['runtime']
        if runtime['status'] != 'completed' or runtime['force_all'] is not True:
            raise ValueError('Expected completed full historical re-evaluation')
        if not set(old_radii).issubset(runtime['recomputed_rhos']):
            raise ValueError('Full cache omitted plotted radii')
        cache_end = datetime.fromisoformat(runtime['end_utc']).timestamp()
        directories = {}
        for p in sorted(SWEEP.iterdir()):
            if p.is_dir() and p.name.startswith('rho_'):
                r = float(p.name[4:].replace('p', '.'))
                if r in directories:
                    raise ValueError(f'Ambiguous radius directory: {r}')
                directories[r] = p
        manifest['old_radii'] = old_radii
        manifest['new_radii'] = list(RADII)
        manifest['historical_mapping_evidence'] = dict(
            evaluator_version='gurobi_qp_final_mp_is_v2', full_cache_runtime=runtime,
            rule='Exact legacy source discovery and final_mp_iter; zero means nominal',
            limitation='Historical cache has no checkpoint hashes. Rule, seed order, full rerun '
                       'metadata and pre-rerun mtimes support mapping, not cryptographic historical identity.')
        for rho in old_radii:
            e = cache['sweep'][str(rho)]
            if e['evaluator_version'] != 'gurobi_qp_final_mp_is_v2':
                raise ValueError('Unexpected historical evaluator')
            for seed in SEEDS:
                ckpt, iteration, summary = old_choice(rho, seed, directories)
                for p in [ckpt] + ([summary] if summary else []):
                    if p.stat().st_mtime > cache_end:
                        raise ValueError(f'Historical identity unresolved: modified after cache: {p}')
                    capture(p)
                model = load_model(ckpt, 'old')
                info = capture(ckpt)
                manifest['jobs'].append(dict(id=f'old_r{rho:g}_s{seed}', method='old', radius=rho,
                    seed=seed, n_train=50, n_val=0, training_backend='SCS', selected_iteration=iteration,
                    checkpoint=str(ckpt), checkpoint_copy=info['copy'], checkpoint_sha256=info['sha256'],
                    state_sha256=state_hash(model), selector=str(summary) if summary else 'FINAL_ITER_040'))
        sweep = read_json(NEW / 'ccg_results/summary.json')
        if sweep['status'] != 'completed' or sweep['final_provenance_verified'] is not True:
            raise ValueError('New sweep not completed and verified')
        manifest['saved_new_selections'] = sweep['selections']  # No re-selection.
        for rho in RADII:
            for seed in SEEDS:
                folder = NEW / f'ccg_results/seed_{seed}/radius_{rho:g}'
                summary = folder / 'summary.json'
                capture(summary)
                d = read_json(summary)
                if not d['complete'] or d['configuration_status'] != 'completed':
                    raise ValueError(f'Incomplete new job: {summary}')
                ckpt = folder / 'selected.pth'
                if Path(d['selected_checkpoint']) != ckpt or d['seed'] != seed or d['radius'] != rho:
                    raise ValueError('New selected identity mismatch')
                model = load_model(ckpt, 'new')
                digest = state_hash(model)
                if digest != d['selected_state_sha256']:
                    raise ValueError('Selected state hash mismatch')
                capture(Path(d['selected_ap_reference']))
                info = capture(ckpt)
                manifest['jobs'].append(dict(id=f'new_r{rho:g}_s{seed}', method='new', radius=rho,
                    seed=seed, n_train=40, n_val=10, training_backend='Moreau',
                    selected_iteration=d['selected_candidate_id'], checkpoint=str(ckpt),
                    checkpoint_copy=info['copy'], checkpoint_sha256=info['sha256'], state_sha256=digest,
                    selector=str(summary), selected_separation_status=d['selected_separation_status'],
                    terminal_status=d['terminal_status']))
        for selection in sweep['selections']:
            s = selection['final_selection']
            job = next(j for j in manifest['jobs'] if j['method'] == 'new' and
                       j['seed'] == s['seed'] and j['radius'] == s['radius'])
            if selection['provisional'] or job['state_sha256'] != s['selected_state_sha256']:
                raise ValueError('Saved final selection inconsistent with radius checkpoint')
            for key in ('selected_separation_status', 'terminal_status'):
                if job[key] != s[key]:
                    raise ValueError('Saved final selection status mismatch')
            if job['selected_iteration'] != s['selected_candidate_id']:
                raise ValueError('Saved final candidate mismatch')
        manifest.update(status='frozen', end_utc=utc(), wall_time_s=time.perf_counter() - start,
                        job_count=len(manifest['jobs']), expected_projection_count=len(manifest['jobs']) * N_TEST)
        write_json(FROZEN / 'manifest.json', manifest)
        (FROZEN / 'manifest.json').chmod(0o444)
        with (FROZEN / 'manifest.sha256').open('x') as f:
            f.write(sha(FROZEN / 'manifest.json') + '\n')
        (FROZEN / 'manifest.sha256').chmod(0o444)
        verify_manifest()
        print(json.dumps({k: manifest[k] for k in ('status', 'job_count', 'expected_projection_count', 'wall_time_s')}))
    except BaseException:
        write_json(FROZEN / 'freeze_failure.json', dict(status='failed', error=traceback.format_exc(),
                   wall_time_s=time.perf_counter() - start, end_utc=utc()))
        raise


def verify_manifest():
    path = FROZEN / 'manifest.json'
    if sha(path) != (FROZEN / 'manifest.sha256').read_text().strip():
        raise ValueError('Manifest hash changed')
    manifest = read_json(path)
    if manifest['status'] != 'frozen':
        raise ValueError('Manifest is not frozen')
    for original, info in manifest['files'].items():
        if sha(original) != info['sha256'] or sha(FROZEN / info['copy']) != info['sha256']:
            raise ValueError(f'Frozen provenance mismatch: {original}')
    versions = dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                    torch=torch.__version__, gurobi=list(gp.gurobi.version()))
    if manifest['versions'] != versions:
        raise ValueError('Runtime versions changed since freeze')
    return manifest


def strict_project(z, q10, env, diagnostic):
    """Fresh exact legacy QP for EVERY row; no solution reuse, buffer or clipping."""
    start = time.perf_counter()
    diagnostic.update(status=None, iterations=None, barrier_iterations=None)
    try:
        if np.shape(z) != (3,) or not np.isfinite(z).all() or not np.isfinite(q10) or q10 < 0:
            raise ValueError('Invalid projection input')
        with gp.Model(env=env) as m:
            m.Params.OutputFlag = 0
            m.Params.Threads = 1
            q2 = m.addVars(3, lb=0., name='q2')
            m.addConstr(gp.quicksum(q2[r] for r in range(3)) <= q10, name='budget')
            m.setObjective(gp.quicksum((q2[r] - float(z[r])) * (q2[r] - float(z[r]))
                                       for r in range(3)), GRB.MINIMIZE)
            try:
                m.optimize()
            finally:
                diagnostic.update(status=int(m.Status), iterations=float(m.IterCount),
                                  barrier_iterations=int(m.BarIterCount))
            if m.Status != GRB.OPTIMAL:
                raise RuntimeError(f'Projection requires OPTIMAL, got {m.Status}')
            result = np.array([q2[r].X for r in range(3)], dtype=np.float64)
            if not np.isfinite(result).all() or not np.isfinite(m.ObjVal):
                raise FloatingPointError('Nonfinite projection solution/objective')
            residual = float(max(0., -result.min(), result.sum() - q10))
            diagnostic['physical_residual'] = residual
            if residual > PHYSICAL_TOL:
                raise RuntimeError(f'Physical residual {residual} exceeds {PHYSICAL_TOL}')
            diagnostic['accepted'] = True
            return result
    except BaseException as exc:
        diagnostic.update(accepted=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        diagnostic['wall_time_s'] = time.perf_counter() - start


def evaluate_job(job, manifest_sha, test_info, output=RESULTS):
    start = time.perf_counter()
    folder = Path(output) / job['id']
    folder.mkdir(exist_ok=False)
    record = dict(**job, experiment='matched_R3_N50_OOS', n_test=N_TEST, test_seed=TEST_SEED,
                  manifest_sha256=manifest_sha, test_arrays=test_info, status='running', start_utc=utc(),
                  worker_count=WORKERS, solver_threads=1, torch_threads=torch.get_num_threads(),
                  pid=os.getpid(), phases={}, status_counts={}, solves_attempted=0, solves_accepted=0,
                  iterations=0., barrier_iterations=0, mean_oos=None, cached=False)
    write_json(folder / 'result.json', record)
    counts = Counter()
    try:
        with phase(record['phases'], 'input_loading'):
            ckpt = FROZEN / job['checkpoint_copy']
            if sha(ckpt) != job['checkpoint_sha256']:
                raise ValueError('Checkpoint copy changed')
            model = load_model(ckpt, job['method'])
            if state_hash(model) != job['state_sha256']:
                raise ValueError('Learned state changed')
            arrays = []
            for key in ('xi1', 'xi2'):
                info = test_info[key]
                path = Path(output) / info['file']
                if sha(path) != info['file_sha256']:
                    raise ValueError('Test array file changed')
                arrays.append(np.load(path, mmap_mode='r', allow_pickle=False))
            xi1, xi2 = arrays
        with phase(record['phases'], 'target_evaluation'):
            z, q10, q1r = model.targets(xi1)
        with phase(record['phases'], 'projection_qp'):
            q2 = np.empty((N_TEST, 3), dtype=np.float64)
            with gp.Env(empty=True) as env, (folder / 'solves.jsonl').open('x', buffering=1) as journal:
                env.setParam('OutputFlag', 0)
                env.setParam('Threads', 1)
                env.start()
                # Numerical defaults unchanged from legacy; store actual values.
                with gp.Model(env=env) as settings_model:
                    record['solver_settings'] = {k: settings_model.getParamInfo(k)[2] for k in
                        ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'Method', 'Threads')}
                for row in range(N_TEST):
                    diagnostic = dict(row=row)
                    record['solves_attempted'] += 1
                    try:
                        q2[row] = strict_project(z[row], q10, env, diagnostic)
                        record['solves_accepted'] += 1
                    finally:
                        counts[str(diagnostic['status'])] += 1
                        record['iterations'] += diagnostic['iterations'] or 0
                        record['barrier_iterations'] += diagnostic['barrier_iterations'] or 0
                        journal.write(json.dumps(diagnostic, allow_nan=False) + '\n')
                        if not diagnostic.get('accepted'):
                            record['failed_projection'] = diagnostic
        with phase(record['phases'], 'cost_evaluation'):
            costs = scenario_costs(q10, q1r, q2, xi1, xi2)
            if costs.dtype != np.float64 or not np.isfinite(costs).all():
                raise FloatingPointError('Invalid float64 test costs')
            record['mean_oos'] = float(costs.mean())
        if state_hash(model) != job['state_sha256']:
            raise ValueError('Evaluation modified learned state')
        record['status'] = 'completed'
    except BaseException:
        record.update(status='failed', error=traceback.format_exc(), mean_oos=None)
    finally:
        record.update(end_utc=utc(), wall_time_s=time.perf_counter() - start, status_counts=dict(counts))
        record['wall_time_human'] = human(record['wall_time_s'])
        write_json(folder / 'result.json', record)
    return record


def human(seconds):
    return f'{seconds:.6f} s / {seconds / 60:.6f} min / {seconds / 3600:.6f} h'


def timing_stats(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()))


def run():
    start = time.perf_counter()
    RESULTS.mkdir(exist_ok=False)  # No resume, cache substitution or overwrite.
    summary = dict(start_utc=utc(), status='running', phases={}, jobs=[],
                   environment=dict(host=platform.node(), platform=platform.platform(),
                                    processor=platform.processor(), cpu_count=os.cpu_count()),
                   workers=WORKERS, threads_per_worker=1, timing_clock='time.perf_counter',
                   timing_unit='seconds', command=sys.argv, cwd=str(Path.cwd()))
    try:
        with phase(summary['phases'], 'verify_frozen_inputs'):
            manifest = verify_manifest()
            manifest_sha = sha(FROZEN / 'manifest.json')
            summary['manifest_sha256'] = manifest_sha
        with phase(summary['phases'], 'test_generation'):
            xi1, xi2 = gen(N_TEST, TEST_SEED)  # FIRST test read; freeze mandatory.
            test_info = {}
            for key, array in [('xi1', xi1), ('xi2', xi2)]:
                path = RESULTS / f'{key}.npy'
                np.save(path, array, allow_pickle=False)
                path.chmod(0o444)
                test_info[key] = dict(file=path.name, file_sha256=sha(path),
                    array_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest(),
                    dtype=str(array.dtype), shape=list(array.shape))
            summary['test_arrays'] = test_info
        write_json(RESULTS / 'summary.json', summary)
        with phase(summary['phases'], 'parallel_evaluation'):
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize_worker) as pool:
                futures = {pool.submit(evaluate_job, job, manifest_sha, test_info): job
                           for job in manifest['jobs']}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        record = future.result()
                    except BaseException:
                        # Worker crashes are explicit, not retried or given invented timings.
                        record = dict(**job, status='worker_failed', wall_time_s=None,
                                      mean_oos=None, error=traceback.format_exc())
                    summary['jobs'].append(record)
                    write_json(RESULTS / 'summary.json', summary)
                    print(f'{len(summary["jobs"])}/{len(manifest["jobs"])} {job["id"]}: {record["status"]}', flush=True)
        verify_manifest()
        if any(j['status'] != 'completed' for j in summary['jobs']):
            raise RuntimeError('Failed jobs retained. Full-grid plot forbidden; no retry.')
        summary['status'] = 'completed'
        write_json(RESULTS / 'summary.json', summary)
        with phase(summary['phases'], 'plot'):
            plot(summary, manifest)
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        summary.update(end_utc=utc(), wall_time_s=time.perf_counter() - start)
        summary['wall_time_human'] = human(summary['wall_time_s'])
        write_json(RESULTS / 'summary.json', summary)
        runtime_report(summary)


def complete_grid(summary, manifest):
    expected = {j['id']: j for j in manifest['jobs']}
    actual = summary['jobs']
    if len(actual) != len(expected) or {j['id'] for j in actual} != set(expected):
        raise ValueError('Missing or duplicated seed/radius outputs')
    for j in actual:
        if (j['status'] != 'completed' or j['solves_accepted'] != N_TEST or
            j['solves_attempted'] != N_TEST or j['status_counts'] != {str(GRB.OPTIMAL): N_TEST} or
            not np.isfinite(j['mean_oos'])):
            raise ValueError(f'Incomplete strict evaluation: {j["id"]}')
        for k in ('method', 'radius', 'seed', 'checkpoint_sha256', 'state_sha256'):
            if j[k] != expected[j['id']][k]:
                raise ValueError(f'Job provenance mismatch: {j["id"]}')
        if j['manifest_sha256'] != summary['manifest_sha256'] or j['test_arrays'] != summary['test_arrays']:
            raise ValueError('Outputs do not share frozen inputs and the same test arrays')


def plot(summary, manifest):
    complete_grid(summary, manifest)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    def frozen_json(path):
        return read_json(FROZEN / manifest['files'][str(path)]['copy'])

    fig, ax = plt.subplots(figsize=(10, 7))
    for method, color, label in [('old', '#2166ac', 'Old neural: 50 train (SCS training)'),
                                  ('new', '#762a83', 'New neural: 40 train / 10 val (Moreau training)')]:
        radii = manifest[f'{method}_radii']
        values = np.array([[next(j['mean_oos'] for j in summary['jobs'] if
                                j['method'] == method and j['radius'] == r and j['seed'] == s)
                            for s in SEEDS] for r in radii])
        ax.plot(radii, values.mean(axis=1), 'o-', color=color, label=label, ms=4)
        ax.fill_between(radii, np.percentile(values, 20, axis=1),
                        np.percentile(values, 80, axis=1), color=color, alpha=.16)
    sro = frozen_json(SRO)
    radii = sorted(float(k) for k in sro if 0 < float(k) <= 1.5)
    values = []
    for r in radii:
        entries = sro[str(r)]
        if set(entries) != {str(s) for s in SEEDS}:
            raise ValueError('Cached SRO-FA lacks common ten seeds')
        values.append([entries[str(s)]['oos'] for s in SEEDS])
    values = np.asarray(values)
    ax.plot(radii, values.mean(axis=1), 's--', color='#d62728', label='SRO-FA (historical cached reference)', ms=3)
    ax.fill_between(radii, np.percentile(values, 20, axis=1), np.percentile(values, 80, axis=1),
                    color='#d62728', alpha=.12)
    benchmarks = frozen_json(CACHE)['benchmarks']
    ax.axhline(np.mean(benchmarks['saa_knn']['oos']), color='#e6a817', ls='-.',
               label='SAA-kNN (historical 10k; seed-wise LOO k)')
    ax.axhline(np.mean(benchmarks['saa_lb']['oos']), color='#1a9641', ls='--',
               label='SAA-LP clairvoyant LB (cached 2k, not matched 10k)')
    ax.axhline(frozen_json(COND)['saa_cond_lb_large'], color='#1b7837', ls='-.',
               label='Conditional LB (separate 1M; seed 42)')
    ax.set_xscale('symlog', linthresh=.001, linscale=1)
    ax.set_xticks([0, .001, .003, .01, .03, .1, .2, .4, .7, 1.5])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.set(xlabel=r'Training radius $\rho$ (linear near zero, logarithmic above .001)',
           ylabel='Mean out-of-sample cost', title='R=3, total N=50: frozen neural policies, matched strict Gurobi evaluation\n'
           'Same 10,000 scenarios, seed 9999; means and 20/80 seed percentiles (not CIs)')
    ax.grid(axis='y', alpha=.3)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, fontsize=8)
    fig.text(.5, .02, 'Descriptive curves only; no test-selected minima or winners. Training/selection protocols differ.\n'
             'Saved new final selections for seeds 2, 6, 9 are labeled violating: selection is not a separation guarantee.\n'
             'Cached references were not re-solved or status-audited; lower bounds use different samples.',
             ha='center', fontsize=8)
    fig.subplots_adjust(bottom=.36)
    fig.savefig(RESULTS / 'matched_oos.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def runtime_report(summary):
    lines = ['# Matched OOS runtime report', '',
             'Per-configuration current evaluation wall times, seconds. Concurrent times are not summed.', '',
             '| Method | N train | Radius | Completed n | Mean s | Median s | Min s | Max s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    groups = {}
    for j in summary['jobs']:
        if j['status'] == 'completed':
            groups.setdefault((j['method'], j['n_train'], j['radius']), []).append(j['wall_time_s'])
    summary['completed_runtime_aggregates'] = []
    for (method, n, radius), values in sorted(groups.items()):
        s = timing_stats(values)
        summary['completed_runtime_aggregates'].append(dict(method=method, n_train=n, radius=radius, **s))
        lines.append(f'| {method} | {n} | {radius:g} | {s["count"]} | {s["mean"]:.6f} | '
                     f'{s["median"]:.6f} | {s["min"]:.6f} | {s["max"]:.6f} |')
    lines += ['', f'Overall invocation: {human(summary["wall_time_s"])}; status: {summary["status"]}.',
              f'Failed/incomplete: {sum(j["status"] != "completed" for j in summary["jobs"])}; excluded above.',
              'Cached-only evaluations: zero (identical states are intentionally re-evaluated).', '',
              '| Phase/category | Current wall time |', '|---|---|']
    for k, v in summary['phases'].items():
        lines.append(f'| {k} ({v["status"]}) | {human(v["wall_time_s"])} |')
    lines += ['| Nominal neural training | Not run; historical per-model timing not imported; rerun needed if unavailable |',
              '| Robust neural/CCG AP, MP, total | Not run; historical training artifacts remain separate |',
              '| SRO-FA partition, solver, total | Not run; reference cached, not a current timing |',
              '| Hyperparameter selection / CV | Not run; choices frozen |',
              '| Neural final test evaluation | Per-model table and raw phase timings above |', '',
              f'Environment: {json.dumps(summary["environment"])}; {WORKERS} workers, one solver/Torch/BLAS thread each.',
              'Unknown historical timings are not inferred from cache loading. Shared-host contention is uncontrolled.']
    (RESULTS / 'runtime_report.md').write_text('\n'.join(lines) + '\n')
    write_json(RESULTS / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'plot'))
    command = parser.parse_args().command
    initialize_worker()
    if command == 'freeze':
        freeze()
    elif command == 'run':
        run()
    else:
        manifest = verify_manifest()
        summary = read_json(RESULTS / 'summary.json')
        if summary['status'] != 'completed':
            raise ValueError('Only completed campaigns can be plotted')
        with phase(summary['phases'], 'plot_regeneration'):
            plot(summary, manifest)
        write_json(RESULTS / 'summary.json', summary)


if __name__ == '__main__':
    main()