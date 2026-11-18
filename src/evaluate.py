"""Strict standalone fixed-policy replay; no campaign imports or test generator.

Arithmetic is transcribed from the original R3 matched_oos_comparison/compare.py
Model.targets, scenario_costs and strict_project. R10 changes dimension only,
as in comparison_20260925/reference.py. Learned targets/inventories stay float32;
scenarios, QP shipments and costs are float64, with the original operation order.
The caller seals OOS inputs before passing arrays here. `sizes` means contiguous
groups from the returned checkpoint's OWN pool, never a later stopping pool.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn

from .artifacts import _json_value, _phase, digest, timed, write_json

PHYSICAL_TOL = 1e-4


class Model(nn.Module):
    """Learned-only evaluation model; no native/differentiable solver dependency."""
    def __init__(self, R):
        super().__init__()
        if type(R) is not int or R not in (3, 10):
            raise ValueError('R must be 3 or 10')
        self.R = R
        self.fc_layers = nn.Sequential(nn.Linear(2 * R + 1, 16), nn.ReLU(),
                                       nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, R))
        self.Q10 = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.Q1r = nn.Parameter(torch.zeros(R, dtype=torch.float32))

    def targets(self, xi1):
        return targets(self, xi1)


def state_hash(model):
    hasher = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith('proj_layer.'):
            hasher.update(name.encode())
            hasher.update(str((value.dtype, tuple(value.shape))).encode())
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def load_model(path, R):
    """New learned-only checkpoints, not ambiguous historical payload adapters."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = Model(R)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('Require a learned-only state dictionary')
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or value.shape != expected[name].shape or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid learned tensor: {name}')
    model.load_state_dict(state, strict=True)
    return model.eval()


@torch.no_grad()
def targets(model, xi1):
    for name, value in model.named_parameters():
        if (value.device.type != 'cpu' or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Require finite CPU float32 learned parameters: {name}')
    x = torch.tensor(xi1, dtype=torch.float32)
    q10 = torch.nn.functional.softplus(model.Q10).item()
    q1r = torch.nn.functional.softplus(model.Q1r).detach().numpy()
    features = torch.cat([x, torch.tensor(q1r).unsqueeze(0).expand(len(x), -1),
                          torch.full((len(x), 1), q10, dtype=torch.float32)], dim=1)
    z = model.fc_layers(features).detach().numpy()
    if not np.isfinite(z).all() or not np.isfinite([q10, *q1r]).all():
        raise FloatingPointError('Nonfinite target or inventory')
    return z, q10, q1r


def scenario_costs(q10, q1r, q2, xi1, xi2):
    # Do not promote q1r before sum, reassociate additions, or reorder maxima.
    s1 = .25 * (q10 + q1r.sum()) + .05 * q10
    v = np.maximum(.5 * (xi1 + xi2 - q1r - q2) - .05 * q2,
                   np.maximum(.05 * (q1r - xi1 - xi2),
                              .5 * (xi1 - q1r) - .05 * xi2))
    return s1 + v.sum(axis=1)


def finite_pool_cost(costs, sizes):
    a = np.asarray(costs, dtype=np.float64)
    if not len(sizes) or any(type(n) is not int or n <= 0 for n in sizes):
        raise ValueError('Require nonempty positive integer pool sizes')
    if a.ndim != 1 or len(a) != sum(sizes) or not np.isfinite(a).all():
        raise ValueError('Invalid own-pool costs/size coverage')
    maxima = np.maximum.reduceat(a, np.r_[0, np.cumsum(sizes)[:-1]])
    return float(maxima.mean()), maxima.tolist()


def strict_project(z, q10, env, diagnostic):
    """One fresh original Gurobi QP per row, OPTIMAL only, no clipping."""
    import gurobipy as gp
    from gurobipy import GRB

    start = time.perf_counter()
    diagnostic.update(status=None, iterations=None, barrier_iterations=None, accepted=False)
    try:
        R = len(z)
        if (R not in (3, 10) or np.shape(z) != (R,) or not np.isfinite(z).all()
                or not np.isfinite(q10) or q10 < 0):
            raise ValueError('Invalid projection input')
        with gp.Model(env=env) as m:
            m.Params.OutputFlag = 0
            m.Params.Threads = 1
            q2 = m.addVars(R, lb=0., name='q2')
            m.addConstr(gp.quicksum(q2[r] for r in range(R)) <= q10, name='budget')
            m.setObjective(gp.quicksum((q2[r] - float(z[r])) * (q2[r] - float(z[r]))
                                       for r in range(R)), GRB.MINIMIZE)
            try:
                m.optimize()
            finally:
                diagnostic.update(status=int(m.Status), iterations=float(m.IterCount),
                                  barrier_iterations=int(m.BarIterCount),
                                  solver_runtime_s=float(m.Runtime))
            if m.Status != GRB.OPTIMAL:
                raise RuntimeError(f'Projection requires OPTIMAL, got {m.Status}')
            result = np.array([q2[r].X for r in range(R)], dtype=np.float64)
            diagnostic.update(shipments=result.tolist(), objective=float(m.ObjVal))
            if not np.isfinite(result).all() or not np.isfinite(m.ObjVal):
                raise FloatingPointError('Nonfinite projection solution/objective')
            residual = float(max(0., -result.min(), result.sum() - q10))
            diagnostic['physical_residual'] = residual
            if residual > PHYSICAL_TOL:
                raise RuntimeError(f'Physical residual {residual} exceeds {PHYSICAL_TOL}')
            diagnostic['accepted'] = True
            return result
    except BaseException as error:
        diagnostic['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        diagnostic['wall_time_s'] = time.perf_counter() - start


def evaluate_policy(model, xi1, xi2, output, record, sizes=None):
    """Persist all targets, shipments, costs and per-QP diagnostics; return record.

    OOS callers must supply record['role']='oos' and a prewritten input_seal
    (path + sha256) binding checkpoint and test files. Validation/own-pool callers
    need no test seal. This routine never generates, discovers, or selects data.
    Failures are persisted and re-raised; no partially successful mean is returned.
    """
    import gurobipy as gp

    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    names = ('result.json', 'solves.jsonl', 'targets.npy', 'shipments.npy', 'costs.npy')
    if any((folder / name).exists() for name in names):
        raise FileExistsError('Evaluation output already contains replay artifacts')
    counts = Counter()
    record.update(phases={}, status_counts={}, solves_attempted=0, solves_accepted=0,
                  solves_physical_accepted=0, iterations=0., barrier_iterations=0,
                  mean=None, cached=False, solver_threads=1, torch_threads=torch.get_num_threads(),
                  physical_tolerance=PHYSICAL_TOL, worker_count=record.get('worker_count', 1),
                  environment=dict(platform=platform.platform(), hostname=platform.node()))
    before = state_hash(model)
    was_training = model.training
    try:
        with timed(record):
            with _phase(record['phases'], {}, 'input_validation'):
                if record.get('role') == 'oos':
                    seal = record.get('input_seal', {})
                    if not seal.get('path') or digest(seal['path']) != seal.get('sha256'):
                        raise ValueError('OOS requires an intact pre-evaluation input seal')
                    sealed = json.loads(Path(seal['path']).read_text())
                    if sealed['state_sha256'] != before:
                        raise ValueError('OOS policy differs from input seal')
                    if type(sealed['test']['seed']) is not int:
                        raise ValueError('OOS seal requires an explicit test seed')
                    if digest(sealed['checkpoint']) != sealed['checkpoint_sha256']:
                        raise ValueError('Sealed checkpoint file changed')
                x1, x2 = np.asarray(xi1), np.asarray(xi2)
                R = model.Q1r.numel()
                if (x1.ndim != 2 or x1.shape != x2.shape or not len(x1)
                        or x1.shape[1] != R or R not in (3, 10)
                        or x1.dtype != np.float64 or x2.dtype != np.float64
                        or not np.isfinite(x1).all() or not np.isfinite(x2).all()):
                    raise ValueError('Require finite paired nonempty float64 arrays (n,R)')
                if sizes is not None:
                    finite_pool_cost(np.zeros(len(x1)), sizes)
                if record.get('role') == 'oos':
                    for key, array in (('xi1', x1), ('xi2', x2)):
                        info = sealed['test']['arrays'][key]
                        actual = hashlib.sha256(array.tobytes(order='C')).hexdigest()
                        if (actual != info['array_sha256'] or list(array.shape) != info['shape']
                            or str(array.dtype) != info['dtype']
                            or len(array) != sealed['test']['n_test']):
                            raise ValueError(f'OOS array differs from seal: {key}')
                record.update(sample_count=len(x1), R=R, state_sha256=before)
            model.eval()
            with _phase(record['phases'], {}, 'target_evaluation'):
                z, q10, q1r = targets(model, x1)
                np.save(folder / 'targets.npy', z, allow_pickle=False)
                record.update(q10=q10, q1r=q1r.tolist(), inventory_dtype=str(q1r.dtype))
            with _phase(record['phases'], {}, 'projection_qp'):
                q2 = np.empty((len(x1), R), dtype=np.float64)
                with gp.Env(empty=True) as env, (folder / 'solves.jsonl').open('x', buffering=1) as journal:
                    env.setParam('OutputFlag', 0)
                    env.setParam('Threads', 1)
                    env.start()
                    with gp.Model(env=env) as settings_model:
                        record['solver_settings'] = {k: settings_model.getParamInfo(k)[2] for k in
                            ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'Method', 'Threads')}
                    for row in range(len(x1)):
                        diagnostic = dict(row=row, status=None, iterations=None, barrier_iterations=None)
                        record['solves_attempted'] += 1
                        try:
                            q2[row] = strict_project(z[row], q10, env, diagnostic)
                            record['solves_accepted'] += 1
                            record['solves_physical_accepted'] += 1
                        finally:
                            counts[str(diagnostic['status'])] += 1
                            record['iterations'] += diagnostic['iterations'] or 0
                            record['barrier_iterations'] += diagnostic['barrier_iterations'] or 0
                            journal.write(json.dumps(_json_value(diagnostic), allow_nan=False) + '\n')
                            if not diagnostic.get('accepted'):
                                record['failed_projection'] = diagnostic
                np.save(folder / 'shipments.npy', q2, allow_pickle=False)
            with _phase(record['phases'], {}, 'cost_evaluation'):
                costs = scenario_costs(q10, q1r, q2, x1, x2)
                if costs.dtype != np.float64 or not np.isfinite(costs).all():
                    raise FloatingPointError('Invalid float64 evaluation costs')
                np.save(folder / 'costs.npy', costs, allow_pickle=False)
                if sizes is None:
                    record['mean'] = float(costs.mean())
                    if record.get('role') == 'oos':
                        record['mean_oos'] = record['mean']
                else:
                    mean, maxima = finite_pool_cost(costs, sizes)
                    record.update(mean=mean, finite_pool_mean=mean,
                                  per_center_maxima=maxima, sizes=list(sizes))
            if state_hash(model) != before:
                raise ValueError('Evaluation changed learned state')
    except BaseException:
        record['mean'] = None
        record.pop('mean_oos', None)
        record.pop('finite_pool_mean', None)
        raise
    finally:
        model.train(was_training)
        record['status_counts'] = dict(counts)
        if 'wall_time_s' in record:
            record['wall_time_human'] = f"{record['wall_time_s']:.3f} s / {record['wall_time_s'] / 60:.3f} min"
        write_json(folder / 'result.json', record)
    return record