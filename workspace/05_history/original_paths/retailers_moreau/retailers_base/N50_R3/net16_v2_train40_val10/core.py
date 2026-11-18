"""Fresh 40/10 retailer training primitives; campaign/AP are deliberately external.

Epochs are evaluated before updating. Validation selects checkpoints and drives
the scheduler, never the gradient objective supplied by the caller. No capacity
buffer, warm starts, solver fallback, or historical checkpoint loading is used.
"""
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).parent
_DEPS = Path('/data/home/aviyac/hydro_rcr_rescaled/.moreau_diagnostic_deps')
sys.path.insert(0, str(_DEPS))

import gurobipy as gp
from gurobipy import GRB
import moreau
from moreau.torch import Solver
import numpy as np
from scipy import sparse
import torch

if moreau.__version__ != '0.4.0' or not Path(moreau.__file__).resolve().is_relative_to(_DEPS):
    raise ImportError('This experiment requires isolated Moreau 0.4.0')

_BASELINE_PATH = Path('/data/home/aviyac/retailers_moreau_N50_R3_net16_v2/baseline.py')
_spec = importlib.util.spec_from_file_location('_retailer_definition_loader', _BASELINE_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f'Cannot load {_BASELINE_PATH}')
_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loader)
load_baseline = _loader.load_baseline
baseline = load_baseline()
baseline.cfg.N_SAMPLES = 50
baseline.cfg.R = 3
baseline.cfg.HIDDEN_WIDTH = 16
baseline.cfg.NET_SEED = 0
baseline.cfg.LR_Q = .01

LR_GRID = (.005, .01, .05)
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
SEEDS = tuple(range(10))
_SOURCE_SOLVER_ARGS = {'solve_method': 'SCS', 'eps': 1e-4, 'max_iters': 200000}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def weights(model) -> dict:
    """Detached independent learned state, including non-projection buffers."""
    return {key: value.detach().clone() for key, value in model.state_dict().items()
            if not key.startswith('proj_layer.')}


def _json_value(value):
    """Encode nonfinite diagnostics explicitly; this never repairs computation."""
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_json(path, obj):
    """Strict JSON, atomically replaced in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(_json_value(obj), stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def timed(record):
    start = time.perf_counter()
    record.update(started_at_utc=utc_now(), status='running')
    try:
        yield record
    except BaseException as error:
        record.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    else:
        record['status'] = 'completed'
    finally:
        record.update(ended_at_utc=utc_now(), wall_time_s=time.perf_counter() - start)


def stats(values):
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise FloatingPointError('Nonfinite input to timing statistics')
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


def split_data(seed):
    """Generate 50 original paired observations, then independently permute 40/10."""
    baseline.cfg.N_SAMPLES = 50
    # The original generator seeds NumPy's global RNG. Do not let it alter callers.
    rng = np.random.get_state()
    try:
        xi1, xi2 = baseline.generate_data(seed)
    finally:
        np.random.set_state(rng)
    if xi1.shape != (50, 3) or xi2.shape != (50, 3):
        raise ValueError('Expected exactly 50 paired R3 samples')
    indices = np.random.default_rng(100000 + seed).permutation(50)
    train_indices, val_indices = indices[:40], indices[40:]
    return dict(xi1=xi1, xi2=xi2, train_indices=train_indices, val_indices=val_indices,
                train_x1=torch.tensor(xi1[train_indices], dtype=torch.float32),
                train_x2=torch.tensor(xi2[train_indices], dtype=torch.float32),
                val_x1=torch.tensor(xi1[val_indices], dtype=torch.float32),
                val_x2=torch.tensor(xi2[val_indices], dtype=torch.float32))


def solve_saa(train1, train2, record):
    """Canonical LP on train40 ONLY. The supplied record retains failure details."""
    record.update(method='SAA initialization', n_train=40, threads=1,
                  solver_status=None, solver_runtime_s=None, iterations=None,
                  q10=None, q1r=None, objective=None)
    with timed(record):
        x1, x2 = np.asarray(train1, dtype=np.float64), np.asarray(train2, dtype=np.float64)
        if x1.shape != (40, 3) or x2.shape != (40, 3):
            raise ValueError('SAA initialization requires train40 paired R3 arrays')
        if not np.isfinite(x1).all() or not np.isfinite(x2).all():
            raise FloatingPointError('Nonfinite SAA input')
        cfg = baseline.cfg
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            with gp.Model(env=env) as model:
                model.Params.Threads = 1
                q10 = model.addVar(lb=0., name='Q10')
                q1r = model.addVars(3, lb=0., name='Q1r')
                q2 = model.addVars(40, 3, lb=0., name='Q2')
                v = model.addVars(40, 3, lb=-GRB.INFINITY, name='v')
                for i in range(40):
                    model.addConstr(gp.quicksum(q2[i, r] for r in range(3)) <= q10)
                    for r in range(3):
                        model.addConstr(v[i, r] >= cfg.b * (x1[i, r] + x2[i, r]
                            - q2[i, r] - q1r[r]) - cfg.h * q2[i, r])
                        model.addConstr(v[i, r] >= cfg.h * (q1r[r] - x1[i, r] - x2[i, r]))
                        model.addConstr(v[i, r] >= cfg.b * (x1[i, r] - q1r[r]) - cfg.h * x2[i, r])
                objective = (cfg.c * (q10 + gp.quicksum(q1r[r] for r in range(3)))
                             + cfg.h * q10
                             + gp.quicksum(v[i, r] for i in range(40) for r in range(3)) / 40)
                model.setObjective(objective, GRB.MINIMIZE)
                try:
                    model.optimize()
                finally:
                    record.update(solver_status=int(model.Status),
                                  solver_runtime_s=float(model.Runtime),
                                  iterations=float(model.IterCount))
                    if model.SolCount:
                        record.update(q10=float(q10.X), q1r=[float(q1r[r].X) for r in range(3)],
                                      objective=float(model.ObjVal))
                if model.Status != GRB.OPTIMAL:
                    raise RuntimeError(f'SAA did not reach GRB.OPTIMAL: {model.Status}')
                if not np.isfinite([record['q10'], *record['q1r'], record['objective']]).all():
                    raise FloatingPointError('Nonfinite SAA solution')
    return record['q10'], np.asarray(record['q1r'], dtype=np.float64)


class NativeProjection(torch.nn.Module):
    """P=I, q=-z, x>=0, sum(x)<=capacity; native exact differentiation."""
    def __init__(self):
        super().__init__()
        p = sparse.eye(3, format='csr')
        a = sparse.csr_matrix(np.vstack([-np.eye(3), np.ones((1, 3))]))
        self.p = torch.tensor(p.data, dtype=torch.float64)
        self.a = torch.tensor(a.data, dtype=torch.float64)
        settings = moreau.Settings(solver='active_set', device='cpu', enable_grad=True,
            auto_tune=False, active_set_settings=moreau.ActiveSetSettings(diff_method='exact'))
        # Convert infinity only in serialized metadata, not the settings passed to Moreau.
        self.settings = _json_value(settings.model_dump(mode='python'))
        self._solver_kwargs = dict(n=3, m=4,
            P_row_offsets=torch.tensor(p.indptr, dtype=torch.int64),
            P_col_indices=torch.tensor(p.indices, dtype=torch.int64),
            A_row_offsets=torch.tensor(a.indptr, dtype=torch.int64),
            A_col_indices=torch.tensor(a.indices, dtype=torch.int64),
            cones=moreau.Cones(num_nonneg_cones=4), settings=settings)
        # Moreau 0.4.0 backward can recreate a differently sized native solver
        # without invalidating its shared-matrix setup cache. Keep each batch
        # size's lifecycle separate; only structure/matrices, not solves, are reused.
        self._solvers = {}
        self.calls = deque(maxlen=256)
        self._status_counts = Counter()
        self._iteration_counts = Counter()
        self._call_count = 0

    def _new_solver(self):
        solver = Solver(**{**self._solver_kwargs,
                           'settings': self._solver_kwargs['settings'].model_copy(deep=True)})
        original = solver._impl.solve

        def checked(*args, **kwargs):
            result = original(*args, **kwargs)
            status = result['status'].detach().reshape(-1).tolist()
            iterations = result['iterations'].detach().reshape(-1).tolist()
            self.calls.append(dict(status_codes=status, iterations=iterations))
            self._call_count += 1
            self._status_counts.update(status)
            self._iteration_counts.update(iterations)
            if not all(code == 1 for code in status):
                raise RuntimeError(f'Moreau non-solved status: {status}')
            for key in ('x', 'z', 's', 'obj_val'):
                if not bool(torch.isfinite(result[key]).all()):
                    raise FloatingPointError(f'Nonfinite Moreau {key}')
            return result

        solver._impl.solve = checked
        return solver

    def drain_summary(self):
        """Summarize ALL solves since drain, even when the recent-call ring wraps."""
        count = sum(self._iteration_counts.values())
        total = sum(value * n for value, n in self._iteration_counts.items())
        summary = dict(count=self._call_count, sample_count=sum(self._status_counts.values()),
                       statuses=dict(self._status_counts),
                       iterations=dict(count=count, total=total,
                           mean=total / count if count else None,
                           min=min(self._iteration_counts) if count else None,
                           max=max(self._iteration_counts) if count else None))
        self.calls.clear()
        self._call_count = 0
        self._status_counts.clear()
        self._iteration_counts.clear()
        return summary

    def forward(self, z, capacity, solver_args):
        if solver_args != _SOURCE_SOLVER_ARGS:
            raise ValueError(f'Unexpected baseline solver_args: {solver_args}')
        if z.device.type != 'cpu' or capacity.device.type != 'cpu':
            raise ValueError('This projection is CPU-only')
        if z.ndim != 2 or z.shape[1] != 3 or capacity.shape != (z.shape[0], 1):
            raise ValueError('Expected z=(batch,3) and capacity=(batch,1)')
        z, capacity = z.to(torch.float64), capacity.to(torch.float64)
        if not bool(torch.isfinite(z).all() and torch.isfinite(capacity).all()):
            raise FloatingPointError('Nonfinite projection input')
        if bool((capacity < 0).any()):
            raise ValueError('Negative physical capacity')
        b = torch.cat([torch.zeros_like(z), capacity], dim=1)
        batch_size = z.shape[0]
        if batch_size not in self._solvers:
            self._solvers[batch_size] = self._new_solver()
        solution = self._solvers[batch_size].solve(self.p, self.a, -z, b, warm_start=None)
        x = solution.x
        if not bool(torch.isfinite(x).all()):
            raise FloatingPointError('Nonfinite projection output')
        residual = torch.maximum((-x).clamp_min(0).amax(),
                                 (x.sum(dim=1, keepdim=True) - capacity).clamp_min(0).amax())
        if float(residual.detach()) > 1e-6:
            raise RuntimeError(f'Physical projection residual exceeds 1e-6: {float(residual.detach())}')
        return (x,)


def _check_weights(model):
    for key, value in model.state_dict().items():
        if not key.startswith('proj_layer.') and not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f'Nonfinite learned state: {key}')


def _load_learned(model, state):
    if any(key.startswith('proj_layer.') for key in state):
        raise AssertionError('Expected learned-only state, not a saved projection')
    result = model.load_state_dict(state, strict=False)
    assert all(key.startswith('proj_layer.') for key in result.missing_keys), result.missing_keys
    assert all(key.startswith('proj_layer.') for key in result.unexpected_keys), result.unexpected_keys
    _check_weights(model)


def make_model(q10, q1r):
    model = baseline.make_model_saa_init(q10, q1r, net_seed=0)
    model.proj_layer = NativeProjection()
    _check_weights(model)
    return model


def model_from_state(state):
    """Rebuild an explicit candidate; do not use this to initialize LR comparisons."""
    torch.manual_seed(0)
    model = baseline.Model()
    model.proj_layer = NativeProjection()
    _load_learned(model, state)
    return model


@contextmanager
def _phase(record, epoch_record, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        epoch_record[name] = epoch_record.get(name, 0.) + elapsed
        record[name] = record.get(name, 0.) + elapsed


def train_model(model, loss_fn, val_x1, val_x2, lr, folder, record, epochs=3000, patience=300):
    """Train from the given fresh/candidate state, restore validation PREUPDATE best.

    Caller supplies data seed/radius/method and the train40 objective. The selected
    artifact is learned-only. The final artifact preserves the *postupdate* state
    and optimizer/scheduler/RNG, plus the independent candidate reference snapshot.
    No post-restoration objective evaluation is performed here.
    """
    folder = Path(folder)
    optimizer = scheduler = reference = None
    best_state = None
    step_times = []
    projection_statuses = Counter()
    record.update(n_train=40, n_val=10, network_seed=0, lr_net=lr, lr_inventory=.01,
                  max_epochs=epochs, patience=patience, improvement_threshold=1e-4,
                  scheduler_factor=.7, scheduler_patience=150, scheduler_threshold=1e-3,
                  epoch_semantics='preupdate; update final evaluated epoch before restoration',
                  torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                  device='cpu', epochs_evaluated=0, optimizer_steps=0, best_val=None,
                  best_epoch=None, selected_training_loss=None, last_train=None,
                  validation_time_s=0., forward_time_s=0., backward_time_s=0., update_time_s=0.,
                  optimizer_setup_time_s=0., checkpoint_time_s=0.,
                  projection_call_count=0, projection_sample_count=0, projection_iterations=0)
    projection = getattr(model, 'proj_layer', None)

    def drain():
        if not isinstance(projection, NativeProjection):
            return None
        summary = projection.drain_summary()
        projection_statuses.update(summary['statuses'])
        record['projection_statuses'] = dict(projection_statuses)
        record['projection_call_count'] += summary['count']
        record['projection_sample_count'] += summary['sample_count']
        record['projection_iterations'] += summary['iterations']['total']
        return summary

    def snapshot():
        return dict(model=weights(model), optimizer=optimizer.state_dict() if optimizer else None,
                    scheduler=scheduler.state_dict() if scheduler else None,
                    rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                             python=random.getstate()),
                    candidate_reference_model_state=reference,
                    selected_model_state=best_state,
                    epochs_evaluated=record['epochs_evaluated'], optimizer_steps=record['optimizer_steps'],
                    best_epoch=record['best_epoch'], best_val=record['best_val'],
                    selected_training_loss=record['selected_training_loss'])

    try:
        with timed(record):
            training_start = time.perf_counter()
            folder.mkdir(parents=True, exist_ok=True)
            try:
                if not 1 <= epochs <= 3000 or patience < 1:
                    raise ValueError('Require 1..3000 evaluated epochs and positive patience')
                if val_x1.shape != (10, 3) or val_x2.shape != (10, 3):
                    raise ValueError('Validation must contain exactly ten paired R3 samples')
                if not bool(torch.isfinite(val_x1).all() and torch.isfinite(val_x2).all()):
                    raise FloatingPointError('Nonfinite validation data')
                _check_weights(model)
                reference = weights(model)
                if isinstance(projection, NativeProjection):
                    record['projection_settings'] = projection.settings
                    record['preexisting_projection_calls'] = projection.drain_summary()
                with _phase(record, {}, 'optimizer_setup_time_s'):
                    optimizer = torch.optim.Adam([
                        {'params': model.fc_layers.parameters(), 'lr': lr},
                        {'params': [model.Q10, model.Q1r], 'lr': .01}])
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', factor=.7, patience=150, threshold=1e-3)
                no_improvement = 0
                with (folder / 'epochs.jsonl').open('x') as stream:
                    for epoch in range(1, epochs + 1):
                        entry = dict(epoch=epoch, status='running', started_at_utc=utc_now())
                        epoch_start = time.perf_counter()
                        try:
                            model.train()
                            with _phase(record, entry, 'forward_time_s'):
                                optimizer.zero_grad(set_to_none=True)
                                loss = loss_fn(model)
                                entry['train_loss'] = loss.detach().cpu().tolist()
                                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                                    raise FloatingPointError('Training objective must be a finite scalar')
                                train_value = float(loss.detach())
                                entry['train_loss'] = train_value
                                record['last_train'] = train_value
                            model.eval()
                            with _phase(record, entry, 'validation_time_s'), torch.no_grad():
                                costs = model.per_sample_cost(val_x1, val_x2)
                                if costs.shape != (10,) or not bool(torch.isfinite(costs).all()):
                                    entry['invalid_validation_costs'] = costs.detach().cpu().tolist()
                                    raise FloatingPointError('Validation costs must be ten finite values')
                                val = costs.mean()
                                if not bool(torch.isfinite(val)):
                                    raise FloatingPointError('Nonfinite validation mean')
                                val_value = float(val)
                                entry['validation_mean'] = val_value
                            record['epochs_evaluated'] += 1
                            # Clone the exact state that produced BOTH recorded objectives.
                            if record['best_val'] is None or val_value < record['best_val'] - 1e-4:
                                with _phase(record, entry, 'checkpoint_time_s'):
                                    best_state = weights(model)
                                record.update(best_val=val_value, best_epoch=epoch,
                                              selected_training_loss=train_value)
                                no_improvement = 0
                            else:
                                no_improvement += 1
                            model.train()
                            with _phase(record, entry, 'backward_time_s'):
                                loss.backward()
                            with _phase(record, entry, 'update_time_s'):
                                for name, parameter in model.named_parameters():
                                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                                        raise FloatingPointError(f'Nonfinite gradient: {name}')
                                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                                entry['gradient_norm_preclip'] = float(norm)
                                optimizer.step()
                                record['optimizer_steps'] += 1
                                _check_weights(model)
                                scheduler.step(val.detach())
                            entry.update(status='completed', optimizer_steps=record['optimizer_steps'],
                                         no_improvement=no_improvement,
                                         learning_rates=[group['lr'] for group in optimizer.param_groups])
                        except BaseException as error:
                            entry.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
                            raise
                        finally:
                            entry['projection'] = drain()
                            entry['optimizer_steps'] = record['optimizer_steps']
                            entry['train_step_time_s'] = sum(entry.get(key, 0.) for key in
                                ('forward_time_s', 'backward_time_s', 'update_time_s'))
                            if entry['status'] == 'completed':
                                step_times.append(entry['train_step_time_s'])
                            entry.update(ended_at_utc=utc_now(), wall_time_s=time.perf_counter() - epoch_start)
                            stream.write(json.dumps(_json_value(entry), allow_nan=False) + '\n')
                            stream.flush()
                        if epoch % 100 == 0:
                            record['train_step_times'] = stats(step_times)
                            record['current_invocation_elapsed_time_s'] = time.perf_counter() - training_start
                            write_json(folder / 'training.json', record)
                        if no_improvement >= patience:
                            record['stop_reason'] = 'validation_patience'
                            break
                    else:
                        record['stop_reason'] = 'max_evaluated_epochs'
                # Preserve every update, including the last, before best restoration.
                torch.save(snapshot(), folder / 'final_state.pth')
                if best_state is None:
                    raise RuntimeError('No finite evaluated checkpoint')
                torch.save(best_state, folder / 'selected.pth')
                _load_learned(model, best_state)
                model.eval()
            except BaseException:
                record['failure_projection'] = drain()
                torch.save(snapshot(), folder / 'failure_state.pth')
                raise
    finally:
        record['train_step_times'] = stats(step_times)
        record['train_compute_time_s'] = sum(record[key] for key in
            ('forward_time_s', 'backward_time_s', 'update_time_s'))
        write_json(folder / 'training.json', record)
    return model