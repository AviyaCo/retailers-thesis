"""Fixed-policy AP: canonical ReLU indicators, projection KKT, exact max3.

Source: retailers_problem/R3/N50/rho_sweep_net16_v2/run_rho_seed_net16_v2.py,
load_weights/solve_ap_one. Inventory softplus is evaluated by Torch in float32;
subsequent coefficient folding is float64. There is no capacity buffer, policy
optimization, time cap, fallback, or access to any dataset outside the centers.
The caller owns persistence of the supplied (strict-JSON-safe) timing record.
"""
import time

import core
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch


PARAMETERS = dict(Threads=1, MIPGap=1e-4, FeasibilityTol=1e-9,
                  OptimalityTol=1e-9, IntFeasTol=1e-9, NumericFocus=3,
                  OutputFlag=0)
REPLAY_TOLERANCE = 1e-4


def _load_weights(model):
    """Fold fixed inventories into the first-layer bias, never xi2."""
    learned = [model.Q10, model.Q1r, *model.fc_layers.parameters()]
    for value in learned:
        if value.dtype != torch.float32 or value.device.type != 'cpu':
            raise ValueError('AP requires the native CPU float32 policy')
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite AP policy parameter')
    with torch.no_grad():
        q10 = float(torch.nn.functional.softplus(model.Q10).item())
        q1r = torch.nn.functional.softplus(model.Q1r).numpy().astype(np.float64)
    layers = [model.fc_layers[index] for index in (0, 2, 4)]
    matrices = [layer.weight.detach().numpy().astype(np.float64) for layer in layers]
    biases = [layer.bias.detach().numpy().astype(np.float64) for layer in layers]
    first = matrices[0]
    folded = biases[0] + first[:, 3:6] @ q1r + first[:, 6:] @ np.array([q10])
    values = dict(W1=first[:, :3], b1=folded, W2=matrices[1], b2=biases[1],
                  W3=matrices[2], b3=biases[2], Q10=q10, Q1r=q1r)
    if not all(np.isfinite(value).all() for value in values.values()):
        raise FloatingPointError('Nonfinite folded AP coefficients')
    return values


def _solver_snapshot(solver, variables, record):
    """Retain diagnostics even on failure; unavailable attributes are explicit."""
    unavailable = {}
    for key, attribute in [('solver_status', 'Status'), ('bound', 'ObjBound'),
                           ('solcount', 'SolCount'), ('solver_seconds', 'Runtime'),
                           ('nodecount', 'NodeCount'), ('iterations', 'IterCount'),
                           ('barrier_iterations', 'BarIterCount'), ('mip_gap', 'MIPGap')]:
        try:
            record[key] = core._json_value(solver.getAttr(attribute))
        except (gp.GurobiError, AttributeError) as error:
            record[key] = None
            unavailable[key] = str(error)
    record['unavailable_solver_attributes'] = unavailable
    if record['solcount'] is not None and record['solcount'] > 0:
        record['incumbent'] = core._json_value(dict(
            cost=float(solver.ObjVal),
            **{name: [float(values[r].X) for r in range(3)]
               for name, values in variables.items()}))


def solve_ap(model, center1: np.ndarray, center2: np.ndarray, radius: float,
             record: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Maximize fixed policy cost on two nonnegative radius boxes.

    Only GRB.OPTIMAL plus a successful native replay returns a result. Replay
    discrepancy is abs(AP-native)/max(1, abs(AP), abs(native)), at most 1e-4.
    All failures propagate, with diagnostics and aggregate wall time in record.
    """
    with core.timed(record):
        record.update(method='fixed-policy canonical AP', n_samples=1, threads=1,
                      radius=core._json_value(radius), parameters=dict(PARAMETERS),
                      time_limit='Infinity', capacity_buffer=0.,
                      replay_tolerance=REPLAY_TOLERANCE,
                      solver_status=None, bound=None, solcount=None,
                      solver_seconds=None, solver_wall_time_s=None,
                      nodecount=None, iterations=None, incumbent=None)
        x1h = np.asarray(center1, dtype=np.float64)
        x2h = np.asarray(center2, dtype=np.float64)
        radius = float(radius)
        record.update(center1=core._json_value(x1h), center2=core._json_value(x2h))
        if x1h.shape != (3,) or x2h.shape != (3,):
            raise ValueError('AP centers must each have shape (3,)')
        if not np.isfinite(x1h).all() or not np.isfinite(x2h).all() or not np.isfinite(radius):
            raise FloatingPointError('Nonfinite AP input')
        if radius < 0 or (x1h < 0).any() or (x2h < 0).any():
            raise ValueError('AP radius and demand centers must be nonnegative')
        lower1, upper1 = np.maximum(0., x1h - radius), x1h + radius
        lower2, upper2 = np.maximum(0., x2h - radius), x2h + radius
        if not np.isfinite([lower1, upper1, lower2, upper2]).all():
            raise FloatingPointError('Nonfinite AP uncertainty bounds')
        w = _load_weights(model)
        q10, q1r = w['Q10'], w['Q1r']
        record.update(q10=q10, q1r=q1r.tolist(),
                      uncertainty_bounds=dict(lower1=lower1.tolist(), upper1=upper1.tolist(),
                                              lower2=lower2.tolist(), upper2=upper2.tolist()))
        cfg = core.baseline.cfg
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            with gp.Model('fixed_policy_AP', env=env) as solver:
                for name, value in PARAMETERS.items():
                    solver.setParam(name, value)
                xi1 = solver.addVars(3, lb=lower1, ub=upper1, name='xi1')
                xi2 = solver.addVars(3, lb=lower2, ub=upper2, name='xi2')

                def relu_layer(inputs, matrix, bias, tag):
                    n = matrix.shape[0]
                    positive = solver.addVars(n, lb=0., name=f'xp{tag}')
                    negative = solver.addVars(n, lb=0., name=f'xm{tag}')
                    inactive = solver.addVars(n, vtype=GRB.BINARY, name=f'z{tag}')
                    for j in range(n):
                        affine = gp.quicksum(matrix[j, k] * inputs[k]
                                             for k in range(matrix.shape[1])) + bias[j]
                        solver.addConstr(affine == positive[j] - negative[j])
                        solver.addGenConstrIndicator(inactive[j], 1, positive[j] == 0)
                        solver.addGenConstrIndicator(inactive[j], 0, negative[j] == 0)
                    return positive

                layer1 = relu_layer(xi1, w['W1'], w['b1'], '1')
                layer2 = relu_layer(layer1, w['W2'], w['b2'], '2')
                target = solver.addVars(3, lb=-GRB.INFINITY, name='target')
                for r in range(3):
                    solver.addConstr(target[r] == gp.quicksum(
                        w['W3'][r, k] * layer2[k] for k in range(w['W3'].shape[1])) + w['b3'][r])
                q2 = solver.addVars(3, lb=0., name='q2')
                lam = solver.addVar(lb=0., name='lambda')
                mu = solver.addVars(3, lb=0., name='mu')
                for r in range(3):
                    solver.addConstr(2 * q2[r] - 2 * target[r] + lam - mu[r] == 0)
                total_q2 = gp.quicksum(q2[r] for r in range(3))
                solver.addConstr(total_q2 <= q10, name='exact_capacity')
                ul = solver.addVar(vtype=GRB.BINARY, name='ul')
                solver.addGenConstrIndicator(ul, 1, lam == 0)
                solver.addGenConstrIndicator(ul, 0, q10 - total_q2 == 0)
                um = solver.addVars(3, vtype=GRB.BINARY, name='um')
                for r in range(3):
                    solver.addGenConstrIndicator(um[r], 1, mu[r] == 0)
                    solver.addGenConstrIndicator(um[r], 0, q2[r] == 0)
                v = solver.addVars(3, lb=-GRB.INFINITY, name='max3')
                for r in range(3):
                    g1, g2, g3 = (solver.addVar(lb=-GRB.INFINITY) for _ in range(3))
                    solver.addConstr(g1 == cfg.b * (xi1[r] + xi2[r] - q1r[r] - q2[r]) - cfg.h * q2[r])
                    solver.addConstr(g2 == cfg.h * (q1r[r] - xi1[r] - xi2[r]))
                    solver.addConstr(g3 == cfg.b * (xi1[r] - q1r[r]) - cfg.h * xi2[r])
                    solver.addGenConstrMax(v[r], [g1, g2, g3])
                stage1 = cfg.c * (q10 + q1r.sum()) + cfg.h * q10
                solver.setObjective(stage1 + gp.quicksum(v[r] for r in range(3)), GRB.MAXIMIZE)
                start = time.perf_counter()
                record['solver_started_at_utc'] = core.utc_now()
                try:
                    solver.optimize()
                finally:
                    record.update(solver_wall_time_s=time.perf_counter() - start,
                                  solver_ended_at_utc=core.utc_now())
                    _solver_snapshot(solver, dict(worst1=xi1, worst2=xi2, q2=q2,
                                                  target=target), record)
                if solver.Status != GRB.OPTIMAL:
                    raise RuntimeError(f'AP did not reach GRB.OPTIMAL: {solver.Status}')
                worst1 = np.array([xi1[r].X for r in range(3)], dtype=np.float64)
                worst2 = np.array([xi2[r].X for r in range(3)], dtype=np.float64)
                allocation = np.array([q2[r].X for r in range(3)], dtype=np.float64)
                targets = np.array([target[r].X for r in range(3)], dtype=np.float64)
                cost = float(solver.ObjVal)
                if not np.isfinite(cost) or not np.isfinite([worst1, worst2, allocation, targets]).all():
                    raise FloatingPointError('Nonfinite AP incumbent')
                residual = max(0., float((lower1 - worst1).max()), float((worst1 - upper1).max()),
                               float((lower2 - worst2).max()), float((worst2 - upper2).max()),
                               float((-allocation).max()), float(allocation.sum() - q10))
                record['output_bound_violation'] = residual
                if residual > PARAMETERS['FeasibilityTol']:
                    raise RuntimeError(f'AP output bound violation: {residual}')

        replay = record['replay'] = {}
        with core.timed(replay):
            replay['preexisting_projection_calls'] = model.proj_layer.drain_summary()
            try:
                with torch.no_grad():
                    replay1 = torch.tensor(worst1[None, :], dtype=torch.float32)
                    replay2 = torch.tensor(worst2[None, :], dtype=torch.float32)
                    if not bool(torch.isfinite(replay1).all() and torch.isfinite(replay2).all()):
                        raise FloatingPointError('Nonfinite float32 AP replay input')
                    native = model.per_sample_cost(replay1, replay2)
                    replay['costs'] = core._json_value(native)
                    if native.shape != (1,) or not bool(torch.isfinite(native).all()):
                        raise FloatingPointError('AP native replay must return one finite cost')
                    native_cost = float(native.item())
            finally:
                replay['projection'] = model.proj_layer.drain_summary()
            discrepancy = abs(cost - native_cost) / max(1., abs(cost), abs(native_cost))
            replay.update(cost=native_cost, ap_cost=cost, floor_scaled_discrepancy=discrepancy)
            if discrepancy > REPLAY_TOLERANCE:
                raise RuntimeError(f'AP/native replay discrepancy {discrepancy} exceeds '
                                   f'{REPLAY_TOLERANCE}: AP={cost}, native={native_cost}')
        return worst1, worst2, cost