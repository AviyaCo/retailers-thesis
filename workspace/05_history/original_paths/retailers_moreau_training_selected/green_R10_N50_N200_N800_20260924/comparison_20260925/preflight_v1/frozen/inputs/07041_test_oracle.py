"""Solver-free contracts only. All artificial rows/solutions are test fixtures.

Real Env/Model constructors are forbidden throughout; no scenario generator or
production LP is invoked. Parent numerical preflight supplies real training rows.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import oracle as o


class Linear:
    """Tiny symbolic recorder, not an optimizer or production algebra backend."""
    def __init__(self, coefficients=None, constant=0.):
        self.coefficients = coefficients or {}
        self.constant = float(constant)

    def __add__(self, other):
        if not isinstance(other, Linear):
            other = Linear(constant=other)
        keys = self.coefficients.keys() | other.coefficients.keys()
        return Linear({k: self.coefficients.get(k, 0.) + other.coefficients.get(k, 0.)
                       for k in keys}, self.constant + other.constant)

    __radd__ = __add__

    def __mul__(self, value):
        return Linear({k: v*value for k, v in self.coefficients.items()}, self.constant*value)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __le__(self, other):
        return '<=', self - other

    def __ge__(self, other):
        return '>=', self - other


class Variable(Linear):
    def __init__(self, name, model, value):
        super().__init__({name: 1.})
        self.model, self.value = model, value

    @property
    def X(self):
        if not self.model.optimized or self.model.Status != o.GRB.OPTIMAL:
            raise AssertionError('Solution attribute read before strict OPTIMAL')
        self.model.solution_reads += 1
        return self.value


class FakeModel:
    def __init__(self, status=2, q10=0., q2=0., objective=5.):
        self.Status, self.IterCount, self.BarIterCount = status, 7., 2
        self.q10, self.q2, self.objective_value = q10, q2, objective
        self.NumVars = self.NumConstrs = self.solution_reads = 0
        self.variables, self.constraints = {}, []
        self.closed = self.optimized = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def addVar(self, lb, name):
        value = self.q10 if name == 'Q10' else self.q2 if name.startswith('Q2[') else 2.
        variable = Variable(name, self, value)
        self.variables[name] = (lb, variable)
        self.NumVars += 1
        return variable

    def addVars(self, *shape, lb, name):
        result = {}
        for index in itertools.product(*(range(n) for n in shape)):
            key = index[0] if len(index) == 1 else index
            result[key] = self.addVar(lb, f'{name}[{",".join(map(str, index))}]')
        return result

    def addConstr(self, constraint):
        self.constraints.append(constraint)
        self.NumConstrs += 1

    def setObjective(self, expression, sense):
        self.expression, self.sense = expression, sense

    def update(self):
        pass

    def optimize(self):
        self.optimized = True

    @property
    def ObjVal(self):
        if not self.optimized or self.Status != o.GRB.OPTIMAL:
            raise AssertionError('Objective read before strict OPTIMAL')
        self.solution_reads += 1
        return self.objective_value


class FakeEnv:
    def __init__(self):
        self.parameters, self.closed = {}, False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def setParam(self, name, value):
        self.parameters[name] = value

    def start(self):
        pass


def sealed(path, value):
    path.write_text(json.dumps(value))
    path.with_suffix('.sha256').write_text(o._sha(path))


def manifest_fixture(root):
    folder = root / 'frozen'
    folder.mkdir()
    manifest = dict(status='frozen', files={}, runner_sources={}, threads=1)
    for i, source in enumerate(o.SOURCE_FILES):
        copy = folder / f'source_{i}'
        copy.write_bytes(source.read_bytes())
        digest = o._sha(source)
        manifest['files'][str(source)] = dict(copy=str(copy), sha256=digest)
        manifest['runner_sources'][str(source)] = digest
    sealed(folder / 'policies_manifest.json', manifest)
    manifest['test'] = dict(n_test=10000, seed=9999, arrays={})
    for name in ('xi1', 'xi2'):
        array = np.ones((10000, 10), dtype=np.float64)  # Test fixture, never production.
        path = folder / f'{name}.npy'
        np.save(path, array, allow_pickle=False)
        manifest['test']['arrays'][name] = dict(file=str(path), file_sha256=o._sha(path),
            array_sha256=hashlib.sha256(array.tobytes()).hexdigest(), shape=[10000, 10], dtype='float64')
    sealed(folder / 'manifest.json', manifest)
    return manifest


class OracleTests(unittest.TestCase):
    def setUp(self):
        for name in ('Env', 'Model'):
            guard = patch.object(o.gp, name, side_effect=AssertionError('Real solver forbidden'))
            guard.start()
            self.addCleanup(guard.stop)

    def fake_backend(self, model, env):
        return patch.object(o, 'gp', SimpleNamespace(Env=lambda **kw: env, Model=lambda **kw: model,
                            quicksum=sum, gurobi=SimpleNamespace(version=lambda: (0, 0, 0))))

    def test_source_lock_ast_and_public_interfaces(self):
        self.assertEqual(o._sha(o.SOURCE), o.SOURCE_SHA256)
        self.assertEqual(list(inspect.signature(o.run).parameters), ['manifest', 'output'])
        self.assertEqual(list(inspect.signature(o.solve).parameters), ['xi1', 'xi2', 'record', 'solution_path'])
        # Exact ordered original algebra statements, with only lifecycle replaced.
        original = next(n for n in ast.parse(o.SOURCE.read_text()).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'saa_lp_full')
        captured = []
        real_compile = compile
        def capture(tree, *args, **kwargs):
            if isinstance(tree, ast.Module):
                captured.append(tree)
            return real_compile(tree, *args, **kwargs)
        with patch('builtins.compile', side_effect=capture):
            o._definitions()
        builder = captured[-1].body[0]
        stop = next(i for i, n in enumerate(original.body) if ast.unparse(n) == 'm.optimize()')
        self.assertEqual([ast.dump(n) for n in builder.body[1:-1]],
                         [ast.dump(n) for n in original.body[5:stop]])
        with patch.object(o, 'SOURCE_SHA256', '0'*64), self.assertRaisesRegex(ValueError, 'source changed'):
            o._definitions()
        self.assertEqual(1+10+2*10000*10, 200011)
        self.assertEqual(10000*(1+3*10), 310000)

    def test_symbolic_formulation_all_branches_bounds_and_objective(self):
        model, env = FakeModel(), FakeEnv()
        x1, x2 = np.ones((2, 10)), np.full((2, 10), 2.)
        with self.fake_backend(model, env):
            build, _ = o._definitions()
            build(model, x1, x2)
        self.assertFalse(model.optimized)
        self.assertEqual((model.NumVars, model.NumConstrs), (51, 62))
        for name, (lb, _) in model.variables.items():
            self.assertEqual(lb, -o.GRB.INFINITY if name.startswith('v[') else 0.)
        sense, cap = model.constraints[0]
        # Python prefers the reflected comparison on the Variable subclass.
        if sense == '>=':
            cap = -cap
        self.assertEqual(cap.coefficients, {'Q10': -1., **{f'Q2[0,{r}]': 1. for r in range(10)}})
        expected = [({'v[0,0]': 1., 'Q1r[0]': .5, 'Q2[0,0]': .55}, -1.5),
                    ({'v[0,0]': 1., 'Q1r[0]': -.05}, .15),
                    ({'v[0,0]': 1., 'Q1r[0]': .5}, -.4)]
        for (sense, expr), (coefficients, constant) in zip(model.constraints[1:4], expected):
            self.assertEqual(sense, '>=')
            self.assertEqual(expr.coefficients, coefficients)
            self.assertAlmostEqual(expr.constant, constant)
        self.assertEqual(model.sense, o.GRB.MINIMIZE)
        self.assertEqual(model.expression.coefficients, {'Q10': .3,
            **{f'Q1r[{r}]': .25 for r in range(10)},
            **{f'v[{i},{r}]': .5 for i in range(2) for r in range(10)}})

    def test_optimal_raw_serialization_replay_and_disjoint_timing(self):
        # Tiny negative numerical residual is accepted but never clipped.
        q10 = -1e-6
        model, env = FakeModel(q10=q10, objective=5.+.3*q10), FakeEnv()
        with tempfile.TemporaryDirectory() as tmp, self.fake_backend(model, env):
            record = dict(seed=0, original_sample_ids=[1, 3], scope='training_only')
            result = o.solve(np.ones((2, 10)), np.ones((2, 10)), record, Path(tmp)/'solution.npz')
            self.assertIs(result, record)
            self.assertEqual(record['status'], 'completed')
            self.assertAlmostEqual(record['physical_residual'], 1e-6)
            self.assertEqual(record['solution']['sha256'], o._sha(record['solution']['file']))
            with np.load(record['solution']['file']) as data:
                self.assertEqual(set(data.files), {'Q10', 'Q1r', 'Q2'})
                self.assertEqual(data['Q10'].item(), q10)
                self.assertEqual(data['Q2'].shape, (2, 10))
            phases = record['phases']
            self.assertEqual(list(phases), ['construction', 'optimization', 'cost_replay_and_solution_serialization'])
            self.assertLessEqual(sum(p['wall_time_s'] for p in phases.values()), record['wall_time_s'])
            previous = record['start_utc']
            for phase in phases.values():
                self.assertEqual(phase['status'], 'completed')
                self.assertGreaterEqual(phase['wall_time_s'], 0.)
                self.assertLessEqual(previous, phase['start_utc'])
                previous = phase['end_utc']
            json.dumps(record, allow_nan=False)
        self.assertTrue(model.closed and env.closed)
        self.assertEqual(env.parameters, {'Threads': 1, 'OutputFlag': 0})

    def test_nonoptimal_never_reads_solution_and_closes_resources(self):
        for status in (3, 9, 12, 13):
            model, env, record = FakeModel(status=status), FakeEnv(), {}
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp, self.fake_backend(model, env):
                path = Path(tmp)/'solution.npz'
                with self.assertRaisesRegex(ValueError, 'requires OPTIMAL'):
                    o.solve(np.ones((2, 10)), np.ones((2, 10)), record, path)
                self.assertEqual(model.solution_reads, 0)
                self.assertFalse(path.exists())
                self.assertEqual(record['solver_status'], status)
                self.assertEqual(record['status'], 'failed')
                self.assertEqual(record['phases']['optimization']['status'], 'failed')
                self.assertGreaterEqual(record['wall_time_s'], 0.)
            self.assertTrue(model.closed and env.closed)

    def test_residual_objective_nonfinite_rejected_without_repair(self):
        cases = [dict(q10=-.01), dict(q2=1.), dict(objective=7.), dict(objective=float('nan'))]
        for case in cases:
            model, env, record = FakeModel(**case), FakeEnv(), {}
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp, self.fake_backend(model, env):
                with self.assertRaises(ValueError):
                    o.solve(np.ones((2, 10)), np.ones((2, 10)), record, Path(tmp)/'solution.npz')
                self.assertEqual(record['status'], 'failed')
                self.assertFalse(record['accepted'])
                json.dumps(record, allow_nan=False)
            self.assertTrue(model.closed and env.closed)

    def test_invalid_training_rows_fail_before_environment(self):
        for array in (np.ones((2, 3)), np.ones((2, 10), dtype=np.float32),
                      np.full((2, 10), np.nan), np.full((2, 10), -1.)):
            record = {}
            with self.assertRaises(ValueError):
                o.solve(array, array, record, Path('/unused'))
            self.assertEqual(record['status'], 'failed')
        o.gp.Env.assert_not_called()

    def test_sealed_test_and_source_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = manifest_fixture(root)
            arrays, digest = o._inputs(manifest, root)
            self.assertEqual(arrays[0].shape, (10000, 10))
            self.assertEqual(digest, o._sha(root/'frozen/manifest.json'))
            for change in (lambda m: m['test'].update(seed=1),
                           lambda m: m['test']['arrays']['xi1'].update(array_sha256='wrong'),
                           lambda m: m['test']['arrays']['xi1'].update(file_sha256='wrong'),
                           lambda m: m['test']['arrays']['xi1'].update(shape=[10000, 3])):
                bad = deepcopy(manifest)
                change(bad)
                sealed(root/'frozen/manifest.json', bad)
                with self.assertRaises(ValueError):
                    o._inputs(bad, root)
            sealed(root/'frozen/manifest.json', manifest)
            source = manifest['files'][str(o.SOURCE)]['copy']
            Path(source).write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'Frozen source'):
                o._inputs(manifest, root)

    def test_run_once_common_identity_persistence_and_no_replicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = manifest_fixture(root)
            def fake_solve(x1, x2, record, path):
                self.assertEqual(x1.shape, (10000, 10))
                self.assertEqual(x2.shape, x1.shape)
                path.write_bytes(b'FAKE SOLUTION FOR TEST ONLY')
                record.update(status='completed', accepted=True, wall_time_s=.1,
                              solution=dict(file=str(path), sha256=o._sha(path)))
                return record
            with patch.object(o, 'solve', side_effect=fake_solve) as solve:
                record = o.run(manifest, root)
                solve.assert_called_once()
                self.assertEqual(record['test'], manifest['test'])
                self.assertEqual(record['independent_replicates'], 1)
                self.assertIsNone(record['training_seed'])
                self.assertFalse(record['learned_policy'])
                self.assertEqual(json.loads((root/'oracle/result.json').read_text()), record)
                with self.assertRaises(FileExistsError):
                    o.run(manifest, root)
                solve.assert_called_once()

    def test_run_rejects_unsealed_and_persists_solver_failure(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(o, 'solve') as solve:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                o.run({}, root)
            solve.assert_not_called()
            self.assertEqual(json.loads((root/'oracle/result.json').read_text())['status'], 'failed')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = manifest_fixture(root)
            with patch.object(o, 'solve', side_effect=RuntimeError('fake solver failure')):
                with self.assertRaisesRegex(RuntimeError, 'fake solver failure'):
                    o.run(manifest, root)
            record = json.loads((root/'oracle/result.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertGreaterEqual(record['wall_time_s'], 0.)


if __name__ == '__main__':
    unittest.main()