"""Bounded train-leaf feasibility study; no training, selection, or test evaluation."""
import os

THREAD_ENV = {name: '1' for name in (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'RAYON_NUM_THREADS')}
os.environ.update(THREAD_ENV)

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
from numbers import Number, Real
from pathlib import Path
import platform
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
import core  # Deliberately loads the isolated native stack and definitions-only baseline.
import diffcp
import diffcp.cone_program
import diffcp.utils
import gurobipy as gp
import moreau_cpu
import numpy as np
from scipy import sparse
import scs
import torch

CONFIGS = [dict(name='moreau', method='Moreau exact', eps=None, tau=None, rho=None)] + [
    dict(name=f'scs_eps{eps:g}_tau{tau:g}', method='SCS directQP LPGD',
         eps=eps, tau=tau, rho=0.)
    for eps in (1e-4, 1e-8) for tau in (1e-2, 1e-3, 1e-4)]
CERTIFICATES = {'res_infeas', 'res_unbdd_a', 'res_unbdd_p'}
REPEATS = 5
FD_ROWS = (0, 19, 39)
FD_STEPS = (1e-2, 1e-3, 1e-4)
THRESHOLDS = dict(forward_maxabs=1e-6, physical=1e-6,
                  gradient_floor_scaled=1e-4, active_set=1e-7)


def finite(value, label):
    if sparse.issparse(value):
        value = value.data
    if not np.isfinite(np.asarray(value)).all():
        raise FloatingPointError(f'Nonfinite {label}')


def error_info(error):
    return dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())


@contextmanager
def phase(record):
    start = time.perf_counter()
    record.update(started_at_utc=core.utc_now(), started_monotonic_s=start, status='running')
    try:
        yield
    except BaseException as error:
        record.update(status='failed', error=error_info(error))
        raise
    else:
        record['status'] = 'completed'
    finally:
        end = time.perf_counter()
        record.update(ended_at_utc=core.utc_now(), ended_monotonic_s=end, wall_time_s=end-start)


def active_set(x, capacity):
    x = np.asarray(x)
    slack = float(capacity - x.sum())
    return dict(zero_coordinates=(x <= THRESHOLDS['active_set']).tolist(),
                capacity_active=abs(slack) <= THRESHOLDS['active_set'], capacity_slack=slack)


def branch(signature):
    return signature['zero_coordinates'], signature['capacity_active']


class SCSObserver:
    """Serial, transparent scs.solve interception; retain failed results before raising."""
    def __init__(self):
        self.records = []
        self.row = None
        self.stage = None

    def __enter__(self):
        self.original = scs.solve

        def checked(data, cone, **settings):
            record = dict(row=self.row, phase=self.stage, settings=dict(settings),
                          warm_start_supplied=all(k in data for k in ('x', 'y', 's')),
                          b=data['b'].copy(), c=data['c'].copy())
            self.records.append(record)
            with phase(record):
                # Guard configuration, without editing the passthrough arguments.
                if settings.get('acceleration_lookback') != 10:
                    raise ValueError('Explicit lookback=10 required: no diffcp retry')
                result = self.original(data, cone, **settings)
                record['info'] = dict(result['info'])
                record['solution'] = {k: np.asarray(result[k]).copy() for k in ('x', 'y', 's')}
                record['active_set'] = active_set(result['x'], data['b'][-1])
                if result['info']['status_val'] != 1:
                    raise RuntimeError(f'SCS strict status failure: {result["info"]}')
                for key in ('x', 'y', 's'):
                    finite(result[key], f'SCS {key}')
                for key in ('res_pri', 'res_dual', 'gap', 'pobj', 'dobj'):
                    value = result['info'][key]
                    if not isinstance(value, Real) or isinstance(value, bool):
                        raise TypeError(f'SCS info.{key} must be a real number')
                    finite(value, f'SCS info.{key}')
                for key, value in result['info'].items():
                    if key in CERTIFICATES:
                        # SCS initializes these to NaN when certificate normalization
                        # is inapplicable, including valid status-1 solutions.
                        if not isinstance(value, Real) or isinstance(value, bool):
                            raise TypeError(f'SCS info.{key} must be a real number or NaN')
                        if np.isinf(value):
                            raise FloatingPointError(f'Infinite SCS info.{key}')
                    elif isinstance(value, Number):
                        finite(value, f'SCS info.{key}')
                return result

        scs.solve = checked
        return self

    def __exit__(self, *exc):
        scs.solve = self.original


class DirectQP:
    def __init__(self, config):
        self.config = config
        self.P = sparse.eye(3, format='csc')
        self.A = sparse.csc_matrix(np.vstack([-np.eye(3), np.ones((1, 3))]))

    def forward(self, z, capacity, observer):
        xs, adjoints = [], []
        for i in range(len(z)):
            observer.row, observer.stage = i, 'forward'
            x, y, s, _, adjoint = diffcp.solve_and_derivative(
                self.A, np.r_[np.zeros(3), capacity[i, 0]], -z[i], {'l': 4},
                P=self.P, solve_method='SCS', mode='lpgd',
                derivative_kwargs={'tau': self.config['tau'], 'rho': 0},
                warm_start=None, eps_abs=self.config['eps'], eps_rel=self.config['eps'],
                max_iters=200000, acceleration_lookback=10)
            for key, value in (('x', x), ('y', y), ('s', s)):
                finite(value, key)
            xs.append(x)
            adjoints.append(adjoint)
        return np.stack(xs), adjoints

    def backward(self, adjoints, upstream, observer, raw):
        gz, gc = [], []
        for i, (adjoint, g) in enumerate(zip(adjoints, upstream)):
            observer.row, observer.stage = i, 'backward_plus_then_minus'
            dA, db, dc = adjoint(g, np.zeros(4), np.zeros(4))
            raw.append(dict(dA_data=dA.data.copy(), db=db.copy(), dc=dc.copy()))
            for key, value in (('dA', dA), ('db', db), ('dc', dc)):
                finite(value, key)
            gz.append(-dc)  # c=-z: use the supported adjoint, not a custom Jacobian.
            gc.append(db[-1])
        return np.stack(gz), np.asarray(gc)[:, None]


def run_pass(arm, fixture, record):
    """Identical double leaves; setup/forward/differentiation measured separately."""
    observer = SCSObserver()
    try:
        with phase(record), observer:
            with phase(record.setdefault('setup', {})):
                z = np.asarray(fixture['z'], dtype=np.float64).copy()
                capacity = np.asarray(fixture['capacity'], dtype=np.float64).copy()
                g = np.asarray(fixture['upstream'], dtype=np.float64).copy()
                if isinstance(arm, core.NativeProjection):
                    zt = torch.tensor(z, requires_grad=True)
                    ct = torch.tensor(capacity, requires_grad=True)
                    gt = torch.tensor(g)
            with phase(record.setdefault('forward', {})):
                if isinstance(arm, core.NativeProjection):
                    xt, = arm(zt, ct, core._SOURCE_SOLVER_ARGS)
                else:
                    x, adjoints = arm.forward(z, capacity, observer)
            record['x'] = xt.detach().numpy().copy() if isinstance(arm, core.NativeProjection) else x.copy()
            with phase(record.setdefault('backward', {})):
                if isinstance(arm, core.NativeProjection):
                    gz_t, gc_t = torch.autograd.grad(xt, (zt, ct), grad_outputs=gt)
                    x, gz, gc = [v.detach().numpy().copy() for v in (xt, gz_t, gc_t)]
                else:
                    record['raw_adjoints'] = []
                    gz, gc = arm.backward(adjoints, g, observer, record['raw_adjoints'])
            record.update(x=x, grad_z=gz, grad_capacity=gc)
            for key in ('x', 'grad_z', 'grad_capacity'):
                finite(record[key], key)
    finally:
        record['scs_solves'] = list(observer.records)
        observer.records.clear()
        if isinstance(arm, core.NativeProjection):
            record['native_solves'] = list(arm.calls)
            record['native_summary'] = arm.drain_summary()
        if record['scs_solves']:
            record['scs_internal_setup_s'] = sum(
                r.get('info', {}).get('setup_time', 0.) / 1000 for r in record['scs_solves'])
            record['scs_internal_solve_s'] = sum(
                r.get('info', {}).get('solve_time', 0.) / 1000 for r in record['scs_solves'])
            record['iterations'] = [r.get('info', {}).get('iter') for r in record['scs_solves']]


def reference_qp(z, capacity, records, label):
    record = dict(label=label, z=np.asarray(z).copy(), capacity=float(capacity), threads=1)
    records.append(record)
    with phase(record):
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            with gp.Model(env=env) as model:
                model.Params.Threads = 1
                model.Params.FeasibilityTol = 1e-9
                model.Params.OptimalityTol = 1e-9
                model.Params.BarConvTol = 1e-9
                x = model.addMVar(3, lb=0.)
                model.addConstr(x.sum() <= float(capacity))
                model.setObjective(.5 * (x @ x) - np.asarray(z) @ x)
                try:
                    model.optimize()
                finally:
                    record.update(solver_status=int(model.Status), iterations=float(model.IterCount),
                                  barrier_iterations=int(model.BarIterCount), solver_time_s=float(model.Runtime))
                    if model.SolCount:
                        record.update(x=x.X.copy(), objective=float(model.ObjVal))
                if model.Status != gp.GRB.OPTIMAL:
                    raise RuntimeError(f'Gurobi nonoptimal: {model.Status}')
                finite(record['x'], 'Gurobi x')
                finite(record['objective'], 'Gurobi objective')
                physical = max(0., float(-record['x'].min()), float(record['x'].sum()-capacity))
                record.update(physical_residual=physical, active_set=active_set(record['x'], capacity))
                if physical > THRESHOLDS['physical']:
                    raise RuntimeError(f'Gurobi physical residual: {physical}')
                return record['x']


def references(fixture, record):
    record.update(solves=[], fd=[])
    with phase(record):
        z, capacity, g, directions = [np.asarray(fixture[k]) for k in
                                      ('z', 'capacity', 'upstream', 'target_direction')]
        with phase(record.setdefault('forward_reference', {})):
            x = np.stack([reference_qp(zi, ci[0], record['solves'], f'base/{i}')
                          for i, (zi, ci) in enumerate(zip(z, capacity))])
            record['x'] = x
        with phase(record.setdefault('finite_difference', {})):
            for i in FD_ROWS:
                for target in ('z', 'capacity'):
                    for h in FD_STEPS:
                        row = dict(row=i, original_train_index=fixture['train_indices'][i],
                                   target=target, step=h, status='running')
                        record['fd'].append(row)
                        if target == 'capacity' and capacity[i, 0]-h < 0:
                            row.update(status='invalid_capacity', reason='central minus capacity < 0; no one-sided substitution')
                            continue
                        values, signatures = [], [active_set(x[i], capacity[i, 0])]
                        for sign in (1, -1):
                            zp = z[i] + sign*h*directions[i] if target == 'z' else z[i]
                            cp = capacity[i, 0] + sign*h if target == 'capacity' else capacity[i, 0]
                            xp = reference_qp(zp, cp, record['solves'], f'fd/{i}/{target}/{h}/{sign}')
                            values.append(float(g[i] @ xp))
                            signatures.append(active_set(xp, cp))
                        row.update(status='completed', derivative=(values[0]-values[1])/(2*h),
                                   scalar_values=values, active_sets=signatures,
                                   branch_changed=any(branch(s) != branch(signatures[0]) for s in signatures[1:]))


def metrics(result, native, reference, fixture):
    x, gz, gc = [np.asarray(result[k]) for k in ('x', 'grad_z', 'grad_capacity')]
    refx = np.asarray(reference['x'])
    c = np.asarray(fixture['capacity'])[:, 0]
    physical = np.maximum.reduce([np.zeros(len(x)), -x.min(axis=1), x.sum(axis=1)-c])
    forward_error = np.max(np.abs(x-refx), axis=1)
    report = dict(forward_maxabs_per_row=forward_error, physical_per_row=physical,
                  comparable_forward_accuracy=bool((forward_error <= 1e-6).all() and (physical <= 1e-6).all()),
                  active_sets=[active_set(xi, ci) for xi, ci in zip(x, c)],
                  reference_active_sets=[active_set(xi, ci) for xi, ci in zip(refx, c)], fd=[])
    if native is not None:
        ng, nc = np.asarray(native['grad_z']), np.asarray(native['grad_capacity'])
        n, a = np.linalg.norm(ng, axis=1), np.linalg.norm(gz, axis=1)
        diff = gz-ng
        cosine = [float(gi @ ni / (ai*nn)) if ai > 0 and nn > 0 else None
                  for gi, ni, ai, nn in zip(gz, ng, a, n)]
        rel = np.linalg.norm(diff, axis=1)/np.maximum(1., n)
        ce = np.abs(gc-nc)[:, 0]
        report.update(target_gradient_normrel=rel, target_gradient_cosine=cosine,
                      target_gradient_maxabs=np.abs(diff).max(axis=1), capacity_abs_error=ce,
                      capacity_floor_scaled_error=ce/np.maximum(1., np.abs(nc[:, 0])),
                      gradient_diagnostic_pass=bool((rel <= 1e-4).all() and
                          (ce/np.maximum(1., np.abs(nc[:, 0])) <= 1e-4).all()))
        report['moreau_active_set_differs'] = [
            branch(active_set(xi, ci)) != branch(active_set(ni, ci))
            for xi, ni, ci in zip(x, np.asarray(native['x']), c)
        ]
    else:
        report['gradient_comparison_unavailable'] = 'Moreau failed; no substitute derivative'
    for fd in reference.get('fd', []):
        row = {k: fd[k] for k in ('row', 'target', 'step', 'status')}
        if fd['status'] == 'completed':
            i = fd['row']
            predicted = float(gz[i] @ np.asarray(fixture['target_direction'][i])) if fd['target'] == 'z' else float(gc[i, 0])
            err = abs(predicted-fd['derivative'])
            row.update(predicted=predicted, reference=fd['derivative'], abs_error=err,
                       floor_scaled_error=err/max(1., abs(fd['derivative'])),
                       diagnostic_pass=err/max(1., abs(fd['derivative'])) <= 1e-4,
                       branch_changed=fd['branch_changed'])
        report['fd'].append(row)
    # LPGD perturbs the cost, not the finite-difference leaf direction.
    report['lpgd_branch_changes'] = []
    solves = result.get('scs_solves', [])
    for i in range(len(x)):
        rows = [s for s in solves if s['row'] == i and 'active_set' in s]
        if rows:
            report['lpgd_branch_changes'].append(dict(row=i, active_sets=[r['active_set'] for r in rows],
                branch_changed=any(branch(r['active_set']) != branch(rows[0]['active_set']) for r in rows[1:])))
    return report


def array_hash(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.asarray(array, dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def bind_fixture(seed, root):
    """Read saved selection, not losses for reselection; held-out arrays are hash-only."""
    folder = root / f'seed_{seed}'
    summary_path, split_path = folder/'summary.json', folder/'split.json'
    summary = json.loads(summary_path.read_text())
    if not summary['complete']:
        raise ValueError(f'Incomplete LR selection for seed {seed}')
    winner = summary['winner']
    checkpoint = Path(winner['checkpoint']).resolve()
    if not checkpoint.is_relative_to(folder.resolve()):
        raise ValueError('Checkpoint outside selected seed folder')
    if core.digest(checkpoint) != winner['selected_sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    if core.digest(split_path) != summary['split_artifact_sha256']:
        raise ValueError('Saved split hash mismatch')
    split = json.loads(split_path.read_text())
    arrays = split['arrays']
    raw1, raw2 = np.asarray(arrays['xi1']), np.asarray(arrays['xi2'])
    if raw1.shape != (50, 3) or raw2.shape != (50, 3):
        raise ValueError('Expected original 50 paired observations')
    # Whole-data fingerprint authenticates the historical split, but no held-out
    # values enter features, normalization, costs, references or diagnostics.
    raw_hash = array_hash(raw1, raw2)
    if raw_hash != summary['raw_data_sha256'] or raw_hash != split['raw_data_sha256']:
        raise ValueError('Raw paired-data hash mismatch')
    indices = np.asarray(arrays['train_indices'], dtype=np.int64)
    expected = np.random.default_rng(100000+seed).permutation(50)[:40]
    if not np.array_equal(indices, expected):
        raise ValueError('Original train40 indices mismatch')
    train1, train2 = raw1[indices], raw2[indices]
    for saved, actual in ((arrays['train_x1'], train1), (arrays['train_x2'], train2)):
        if not np.array_equal(np.asarray(saved, dtype=np.float32), actual.astype(np.float32)):
            raise ValueError('Stored training tensor/raw pair mismatch')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = core.model_from_state(state).eval()
    with torch.no_grad():
        xi1 = torch.tensor(train1, dtype=torch.float32)
        q10 = torch.nn.functional.softplus(model.Q10)
        q1r = torch.nn.functional.softplus(model.Q1r)
        features = torch.cat([xi1, q1r.unsqueeze(0).expand(40, -1), q10.expand(40, 1)], dim=1)
        z = model.fc_layers(features).double().numpy().copy()
        capacity = q10.expand(40, 1).double().numpy().copy()
    rng = np.random.default_rng(20260909+seed)
    upstream = rng.normal(size=(40, 3))
    upstream /= np.linalg.norm(upstream, axis=1, keepdims=True)
    # A separate deterministic stream makes leaf directions independent of g.
    directions = np.random.default_rng(20260909+100000+seed).normal(size=(40, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    finite(z, 'frozen z')
    finite(capacity, 'frozen capacity')
    return dict(seed=seed, data_seed=seed, split_seed=100000+seed, upstream_seed=20260909+seed,
                direction_seed=20260909+100000+seed, lr_net=winner['lr_net'],
                source_bindings={str(p): core.digest(p) for p in (summary_path, split_path, checkpoint)},
                checkpoint=str(checkpoint), checkpoint_sha256=winner['selected_sha256'],
                raw_data_sha256=raw_hash, train_pair_sha256=array_hash(train1, train2),
                train_indices=indices.tolist(), train_x1=train1, train_x2=train2,
                features_float32=features.numpy(), z=z, capacity=capacity, upstream=upstream,
                target_direction=directions, leaf_sha256=array_hash(z, capacity, upstream, directions),
                normalization='none, matching frozen baseline; upstream/directions row L2 only')


def provenance():
    paths = {Path(__file__), HERE/'test_study.py', Path(core.__file__),
             core._BASELINE_PATH, core._loader.SOURCE}
    # Freeze actual installed code/binaries as well as distribution versions.
    for module in (core.moreau, moreau_cpu, diffcp, scs):
        paths.update(p for p in Path(module.__file__).resolve().parent.rglob('*')
                     if p.is_file() and p.suffix in ('.py', '.so'))
    return dict(source_sha256={str(p.resolve()): core.digest(p) for p in sorted(paths)},
                versions={name: version(name) for name in
                          ('torch', 'numpy', 'scipy', 'gurobipy', 'diffcp', 'scs', 'moreau-cpu', 'cvxpy', 'cvxpylayers')},
                moreau_version=core.moreau.__version__, python=sys.version, executable=sys.executable,
                environment=dict(platform=platform.platform(), hostname=platform.node(),
                                 processor=platform.processor(), logical_cpus=os.cpu_count(),
                                 affinity=sorted(os.sched_getaffinity(0))),
                threads=THREAD_ENV, workers=1, n_jobs=1, torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads())


def verify_bindings(manifest):
    bindings = dict(manifest['provenance']['source_sha256'])
    for fixture in manifest['fixtures']:
        bindings.update(fixture['source_bindings'])
        if array_hash(*(fixture[k] for k in ('z', 'capacity', 'upstream', 'target_direction'))) != fixture['leaf_sha256']:
            raise ValueError('Frozen leaf hash mismatch')
        if array_hash(fixture['train_x1'], fixture['train_x2']) != fixture['train_pair_sha256']:
            raise ValueError('Frozen train pair hash mismatch')
    for path, expected in bindings.items():
        if core.digest(Path(path)) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def repeat_stats(passes):
    # First pass NEVER contributes to repeated-call statistics.
    rows = [p for p in passes if p['kind'] == 'repeat' and p['status'] == 'completed']
    return {key: core.stats([p['wall_time_s'] if key == 'total' else p[key]['wall_time_s'] for p in rows])
            for key in ('setup', 'forward', 'backward', 'total')}


def run_case(config, fixture, native, reference, path, manifest_hash):
    record = dict(config=config, seed=fixture['seed'], method=config['method'], sample_size=40,
                  radius=0., lr_net=fixture['lr_net'], workers=1, threads=1, n_jobs=1,
                  manifest_sha256=manifest_hash, passes=[], cached=False, resumed=False,
                  iterations_semantics='solver iterations, no optimizer steps', optimizer_steps=0)
    try:
        with phase(record):
            with phase(record.setdefault('construction', {})):
                arm = core.NativeProjection() if config['name'] == 'moreau' else DirectQP(config)
                record['native_settings'] = arm.settings if isinstance(arm, core.NativeProjection) else None
            for repeat in range(REPEATS+1):
                one = dict(kind='first' if repeat == 0 else 'repeat', repeat=repeat)
                record['passes'].append(one)
                try:
                    run_pass(arm, fixture, one)
                    with phase(one.setdefault('diagnostic_time', {})):
                        if reference.get('status') == 'completed':
                            one['metrics'] = metrics(one, one if config['name'] == 'moreau' else native,
                                                     reference, fixture)
                        else:
                            one['metrics_unavailable'] = 'Gurobi reference/FD failed; no substitute'
                finally:
                    core.write_json(path, record)
    except Exception as error:
        record['error'] = error_info(error)
    finally:
        record['repeat_statistics_s'] = repeat_stats(record['passes'])
        record['unattempted_passes'] = REPEATS+1-len(record['passes'])
        core.write_json(path, record)
    return record


def report_text(summary):
    lines = ['# Direct-QP gradient and speed feasibility', '',
             'Training/test/optimizer steps: none. No automatic selection.', '',
             '## Runtime table (seconds; completed configurations only)',
             '| Method/configuration | n | mean | median | min | max |',
             '|---|---:|---:|---:|---:|---:|']
    for config in CONFIGS:
        rows = [r for r in summary['cases'] if r['name'] == config['name'] and r['status'] == 'completed']
        stats = core.stats([r['wall_time_s'] for r in rows])
        lines.append('| '+config['name']+' | '+str(stats['count'])+' | '+
                     ' | '.join('unavailable' if stats[k] is None else f'{stats[k]:.6f}'
                                for k in ('mean', 'median', 'min', 'max'))+' |')
    refs = [r['wall_time_s'] for r in summary['references'] if r['status'] == 'completed']
    for label, values in [('Gurobi reference + FD', refs), ('freeze inputs', [summary['freeze']['wall_time_s']])]:
        stats = core.stats(values)
        lines.append('| '+label+' | '+str(stats['count'])+' | '+' | '.join(
            'unavailable' if stats[k] is None else f'{stats[k]:.6f}' for k in ('mean', 'median', 'min', 'max'))+' |')
    for label in ('Nominal neural training', 'Robust neural/CCG total', 'SRO-FA total', 'CV / held-out evaluation'):
        lines.append(f'| {label} | 0 | not run | not applicable | — | — |')
    elapsed = summary.get('wall_time_s')
    if elapsed is not None:
        lines.append(f'\nSerial campaign elapsed: {elapsed:.6f} s ({elapsed/60:.4f} min; {elapsed/3600:.6f} h).')
    lines.extend(['', 'First-call and five-repeat setup/forward/backward distributions are separate in each case JSON.',
                  'Case totals include diagnostics/persistence; solver-stage timings exclude JSON writes.',
                  'Failed/interrupted cases are excluded from completed aggregates and retained in summary JSON.',
                  'LPGD is approximate; perturbation warm starts and serial SCS versus batched native integration differ.',
                  'No speed claim is valid without checking raw forward/physical and gradient diagnostics.',
                  'Historical training times are not remeasured or inferred from checkpoint loading.'])
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-name', required=True, help='New folder name inside this isolated directory')
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('.', '..'):
        parser.error('output-name must be one new directory basename')
    output = HERE/args.output_name
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summary = dict(experiment='train40_directQP_SCS_gradient_speed_feasibility', cases=[], references=[],
                   seeds=list(range(10)), workers=1, threads=1, radius=0., sample_size=40,
                   timing_clock='time.perf_counter', timing_unit='seconds', status='not_started')
    try:
        with phase(summary):
            with phase(summary.setdefault('freeze', {})):
                if diffcp.__version__ != '1.1.8':
                    raise RuntimeError('Reviewed supported LPGD API requires diffcp 1.1.8')
                manifest = dict(provenance=provenance(), configs=CONFIGS, thresholds=THRESHOLDS,
                    fd_rows=FD_ROWS, fd_steps=FD_STEPS, repeats=REPEATS,
                    finite_info_exception=sorted(CERTIFICATES),
                    certificate_diagnostics_policy='After status 1 only: finite or NaN; infinity rejected. '
                        'SCS initializes certificates to NaN when normalization is inapplicable; raw info preserved.',
                    protocol='train-only frozen leaves; no validation/test evaluation, normalization fit or training',
                    fixtures=[bind_fixture(seed, PARENT/'lr_search_results_v2') for seed in range(10)])
                core.write_json(output/'manifest.json', manifest)
                manifest_hash = core.digest(output/'manifest.json')
                verify_bindings(manifest)
                summary.update(manifest_sha256=manifest_hash, environment=manifest['provenance'])
                # Reread the frozen bytes, rather than benchmarking an unpersisted object.
                manifest = json.loads((output/'manifest.json').read_text())
                verify_bindings(manifest)
            core.write_json(output/'summary.json', summary)
            for fixture in manifest['fixtures']:
                seed = fixture['seed']
                folder = output/f'seed_{seed}'
                folder.mkdir()
                reference = dict(seed=seed, method='Gurobi strict P1 + central FD', sample_size=40, threads=1)
                try:
                    references(fixture, reference)
                except Exception as error:
                    reference['error'] = error_info(error)
                finally:
                    core.write_json(folder/'reference.json', reference)
                    summary['references'].append({k: reference[k] for k in ('seed', 'status', 'wall_time_s')})
                native = None
                for config in CONFIGS:
                    if core.digest(output/'manifest.json') != manifest_hash:
                        raise ValueError('Frozen manifest changed')
                    verify_bindings(manifest)
                    path = folder/(config['name']+'.json')
                    case = run_case(config, fixture, native, reference, path, manifest_hash)
                    if config['name'] == 'moreau' and case['passes'] and case['passes'][0]['status'] == 'completed':
                        native = case['passes'][0]
                    summary['cases'].append(dict(seed=seed, name=config['name'], status=case['status'],
                        wall_time_s=case['wall_time_s'], path=str(path),
                        repeat_statistics_s=case['repeat_statistics_s']))
                    core.write_json(output/'summary.json', summary)
                    print(f'seed={seed} config={config["name"]} status={case["status"]} seconds={case["wall_time_s"]:.6f}', flush=True)
            verify_bindings(manifest)
    finally:
        summary['complete'] = (len(summary['cases']) == 70 and
            all(r['status'] == 'completed' for r in summary['cases']+summary['references']) and
            summary['status'] == 'completed')
        summary['unattempted_configurations'] = [dict(seed=s, name=c['name']) for s in range(10) for c in CONFIGS
            if not any(r['seed'] == s and r['name'] == c['name'] for r in summary['cases'])]
        core.write_json(output/'summary.json', summary)
        if 'wall_time_s' in summary.get('freeze', {}):
            (output/'REPORT.md').write_text(report_text(summary))
    if not summary['complete']:
        raise SystemExit('Feasibility campaign incomplete: inspect raw failure artifacts; no retries performed')


if __name__ == '__main__':
    main()