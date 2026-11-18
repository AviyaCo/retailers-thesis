"""One test-fitted empirical clairvoyant LP, NOT a learned/OOS policy.

Parent contract: bind SOURCE_FILES in manifest['files'] and ['runner_sources'],
seal policies_manifest.json before drawing the shared test, then seal manifest.json.
run(manifest, output) verifies those seals and writes output/oracle/result.json.
No selection, test generation, solver creation, or historical script import occurs
on import. Existing oracle output is never overwritten or treated as a cached run.

Training-only preflight: solve(xi1_train[:2], xi2_train[:2], record, path_to_npz).
The caller supplies genuine training rows and their seed/sample IDs in record;
this low-level function neither reads nor generates test data. It mutates and
returns record, including failed-run timings, and propagates every failure.
Only run enforces the full sealed 10,000-row test protocol.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import gurobipy as gp
from gurobipy import GRB
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = Path('/data/home/aviyac/retailers_problem/R10/N800/rho_sweep_net16_v2/eval_sweep_net16_v2_N800_R10_10k.py')
SOURCE_SHA256 = 'e829a78b50248d9c1d6386b2dde987dc89cb9ea4e8bd18c2c1ebb01edb077446'
SOURCE_FILES = (SOURCE, Path(__file__).resolve(), HERE / 'test_oracle.py')
PHYSICAL_TOL = 1e-4
OBJECTIVE_ABS_TOL = 1e-4  # Acceptance only; no solver tolerance change or repair.
R = 10
c, h, b = .25, .05, .5


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _phase(record, name):
    row = dict(start_utc=_utc(), status='running')
    record.setdefault('phases', {})[name] = row
    start = time.perf_counter()
    try:
        yield
        row['status'] = 'completed'
    except BaseException:
        row['status'] = 'failed'
        raise
    finally:
        row.update(end_utc=_utc(), wall_time_s=time.perf_counter() - start)


def _definitions():
    """Reuse original algebra AST unchanged; replace only lifecycle/extraction.

    The original Env/Model creation and optimize/solution reads are handled by
    solve so resources close and OPTIMAL is checked before any solution access.
    No original top-level statements (including gen/argparse) are executed.
    """
    data = SOURCE.read_bytes()
    _check(hashlib.sha256(data).hexdigest() == SOURCE_SHA256, 'Historical oracle source changed')
    tree = ast.parse(data, filename=str(SOURCE))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    original = functions['saa_lp_full']
    body = original.body
    cut = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call)
               and ast.unparse(n.value.func) == 'm.optimize')
    # Semicolon-separated Env setup is four AST statements after N assignment.
    _check([ast.unparse(n) for n in body[1:6]] == [
        'env = gp.Env(empty=True)', "env.setParam('OutputFlag', 0)",
        'env.start()', 'm = gp.Model(env=env)',
        "Q10v = m.addVar(lb=0.0, name='Q10')"], 'Historical lifecycle layout changed')
    builder = ast.parse('def build(m, xi1, xi2):\n    pass\n').body[0]
    builder.body = [deepcopy(body[0])] + deepcopy(body[5:cut]) + [
        ast.parse('return Q10v, Q1rv, Q2v').body[0]]
    module = ast.fix_missing_locations(ast.Module(
        body=[builder, deepcopy(functions['scenario_costs'])], type_ignores=[]))
    namespace = dict(np=np, gp=gp, GRB=GRB, R=R, c=c, h=h, b=b)
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    return namespace['build'], namespace['scenario_costs']


def _audit(q10, q1, q2, xi1, xi2, objective, costs):
    _check(np.isfinite(q10) and np.isfinite(q1).all() and np.isfinite(q2).all()
           and np.isfinite(objective), 'Nonfinite oracle solution/objective')
    values = costs(q10, q1, q2, xi1, xi2)
    _check(np.isfinite(values).all(), 'Nonfinite replay cost')
    residual = float(max(0., -q10, -q1.min(), -q2.min(), (q2.sum(axis=1) - q10).max()))
    mean = float(values.mean())
    return dict(physical_residual=residual, mean_cost=mean,
                objective_abs_error=abs(mean - objective),
                physical_tolerance=PHYSICAL_TOL, objective_abs_tolerance=OBJECTIVE_ABS_TOL,
                objective_relative_tolerance=0., repair_applied=False)


def solve(xi1, xi2, record, solution_path):
    """Solve supplied R10 rows once; preflight callers must supply training only.

    Records total and phase wall times even on failure. Never changes Gurobi
    defaults except Env OutputFlag=0 and Threads=1. Uses default Method/Seed,
    tolerances, bounds, and limits; no clipping or degraded retry is permitted.
    """
    start = time.perf_counter()
    record.update(start_utc=_utc(), status='running', accepted=False, phases={},
                  method='empirical_clairvoyant_reference', radius=None,
                  radius_applicability='not_applicable', learned_policy=False,
                  worker_count=1, solver_threads=1, timing_clock='time.perf_counter',
                  timing_unit='seconds', solver_status=None, iterations=None,
                  barrier_iterations=None, source=str(SOURCE), source_sha256=SOURCE_SHA256,
                  environment=dict(hostname=platform.node(), platform=platform.platform(),
                      python=platform.python_version(), numpy=np.__version__,
                      gurobi=list(gp.gurobi.version()), pid=os.getpid(),
                      cpu_affinity=sorted(os.sched_getaffinity(0))),
                  solver_parameters=dict(Threads=1, OutputFlag=0),
                  other_solver_parameters='Gurobi defaults; Method and Seed unchanged')
    try:
        with ExitStack() as resources:
            with _phase(record, 'construction'):
                xi1, xi2 = np.asarray(xi1), np.asarray(xi2)
                _check(xi1.ndim == 2 and xi1.shape[1] == R and len(xi1) > 0
                       and xi2.shape == xi1.shape, 'Expected matching nonempty (N,10) rows')
                _check(xi1.dtype == xi2.dtype == np.float64 and np.isfinite(xi1).all()
                       and np.isfinite(xi2).all() and (xi1 >= 0).all() and (xi2 >= 0).all(),
                       'Rows must be finite nonnegative float64')
                n = len(xi1)
                record.update(N=n, retailers=R, dimensions=dict(variables=1+R+2*n*R,
                                                              constraints=n*(1+3*R)))
                build, costs = _definitions()
                env = resources.enter_context(gp.Env(empty=True))
                env.setParam('OutputFlag', 0)
                env.setParam('Threads', 1)
                env.start()
                model = resources.enter_context(gp.Model(env=env))
                q10v, q1v, q2v = build(model, xi1, xi2)
                model.update()
                _check(model.NumVars == record['dimensions']['variables']
                       and model.NumConstrs == record['dimensions']['constraints'],
                       'Unexpected LP dimensions')
            with _phase(record, 'optimization'):
                model.optimize()
                record.update(solver_status=int(model.Status), iterations=float(model.IterCount),
                              barrier_iterations=int(model.BarIterCount))
                _check(model.Status == GRB.OPTIMAL, f'Oracle requires OPTIMAL, got {model.Status}')
            with _phase(record, 'cost_replay_and_solution_serialization'):
                q10 = float(q10v.X)
                q1 = np.array([q1v[r].X for r in range(R)])
                q2 = np.array([[q2v[i, r].X for r in range(R)] for i in range(n)])
                objective = float(model.ObjVal)
                path = Path(solution_path).resolve()
                with path.open('xb') as stream:
                    np.savez(stream, Q10=np.asarray(q10), Q1r=q1, Q2=q2)
                    stream.flush()
                    os.fsync(stream.fileno())
                record['solution'] = dict(file=str(path), sha256=_sha(path),
                                          arrays=['Q10', 'Q1r', 'Q2'], raw=True)
                record.update(_audit(q10, q1, q2, xi1, xi2, objective, costs))
                record['objective'] = objective
                _check(record['physical_residual'] <= PHYSICAL_TOL, 'Oracle physical residual rejected')
                _check(record['objective_abs_error'] <= OBJECTIVE_ABS_TOL,
                       'Oracle objective/replay disagreement')
        record.update(status='completed', accepted=True)
        return record
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.update(end_utc=_utc(), wall_time_s=time.perf_counter() - start)


def _sealed(path):
    path = Path(path)
    digest = _sha(path)
    _check(digest == path.with_suffix('.sha256').read_text().strip(), 'Sealed manifest digest changed')
    return json.loads(path.read_text()), digest


def _inputs(manifest, output):
    path = Path(output) / 'frozen/manifest.json'
    sealed, digest = _sealed(path)
    _check(sealed == manifest and manifest['status'] == 'frozen', 'Manifest differs from sealed test manifest')
    policies, _ = _sealed(path.parent / 'policies_manifest.json')
    _check(policies == {k: v for k, v in manifest.items() if k != 'test'}
           and 'test' not in policies, 'Policy-only seal mismatch')
    for source, info in manifest['files'].items():
        _check(_sha(source) == _sha(info['copy']) == info['sha256'], f'Frozen source/input changed: {source}')
    for source, digest_bound in manifest['runner_sources'].items():
        _check(_sha(source) == manifest['files'][source]['sha256'] == digest_bound,
               f'Runner source changed: {source}')
    for source in SOURCE_FILES:
        _check(str(source) in manifest['runner_sources'], f'Oracle source not runner-bound: {source}')
    _check(manifest['files'][str(SOURCE)]['sha256'] == SOURCE_SHA256, 'Historical source lock mismatch')
    test = manifest['test']
    _check(test['n_test'] == 10000 and test['seed'] == 9999
           and set(test['arrays']) == {'xi1', 'xi2'}, 'Wrong shared test identity')
    arrays = []
    for key in ('xi1', 'xi2'):
        info = test['arrays'][key]
        _check(_sha(info['file']) == info['file_sha256'], 'Frozen test file changed')
        array = np.load(info['file'], allow_pickle=False)
        _check(array.shape == (10000, 10) and info['shape'] == [10000, 10]
               and array.dtype == np.float64 and info['dtype'] == 'float64'
               and hashlib.sha256(array.tobytes(order='C')).hexdigest() == info['array_sha256'],
               'Frozen test array identity changed')
        arrays.append(array)
    return arrays, digest


def run(manifest, output):
    """After the parent seals its common test, solve ONCE and persist evidence.

    manifest is the same dictionary passed to every policy. Exceptions propagate
    after result.json records failure; a pre-existing oracle directory is refused.
    The parent owns scheduling (one sequential process) and experiment reporting.
    """
    start, started = time.perf_counter(), _utc()
    folder = Path(output).resolve() / 'oracle'
    folder.mkdir(exist_ok=False)
    record = dict(experiment='R10_fixed_policy_comparison_20260925',
                  method='empirical_clairvoyant_reference', status='running',
                  N=10000, seed=9999, test_seed=9999, training_seed=None, radius=None,
                  learned_policy=False, independent_replicates=1,
                  interpretation='One common test-fitted clairvoyant LP; not independent OOS policy performance.',
                  test=deepcopy(manifest.get('test')), invocation='fresh', phases={})
    try:
        arrays, digest = _inputs(manifest, output)
        record['manifest_sha256'] = digest
        solve(*arrays, record, folder / 'solution.npz')
        record['solve_wall_time_s'] = record['wall_time_s']
        _, final_digest = _inputs(manifest, output)
        _check(final_digest == digest, 'Manifest changed during oracle solve')
        _check(_sha(record['solution']['file']) == record['solution']['sha256'], 'Solution artifact changed')
        return record
    except BaseException as error:
        record.update(status='failed', accepted=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record.update(start_utc=started, end_utc=_utc(), wall_time_s=time.perf_counter() - start,
                      timing_clock='time.perf_counter', timing_unit='seconds', worker_count=1,
                      solver_threads=1)
        with (folder / 'result.json').open('x') as stream:
            json.dump(record, stream, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())