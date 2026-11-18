"""Analytic replay contracts only: no real QP, training, freeze, or test data.

Solver doubles return prescribed raw values, not simulated optimizers. The
10,000-row verifier fixture is a repeated analytic row, never generated demand.
Rejection tests intentionally expose implementation gaps rather than xfail them.
"""
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
from unittest.mock import MagicMock, Mock, patch

import common as c
import evaluation as e

np, torch = c.np, c.torch


class ReturnedVariable(float):
    """Numeric expression stand-in whose X is the prescribed solver output."""
    @property
    def X(self):
        return float(self)


def solver(solution, status=2, objective=1.):
    model = MagicMock()
    model.__enter__.return_value = model
    model.Params = SimpleNamespace()
    model.Status, model.IterCount, model.BarIterCount = status, 7., 2
    model.ObjVal = objective
    model.addVars.return_value = {i: ReturnedVariable(v) for i, v in enumerate(solution)}
    settings = dict(FeasibilityTol=1e-6, OptimalityTol=1e-6,
                    BarConvTol=1e-8, Method=-1, Threads=1)
    model.getParamInfo.side_effect = lambda key: (key, float, settings[key])
    return model


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Guards cover both evaluation's module and the extracted function globals.
        self.environment = MagicMock()
        self.environment.__enter__.return_value = self.environment
        self.env_factory = self.enterContext(patch.object(c.c.gp, 'Env', return_value=self.environment))
        self.model_factory = self.enterContext(patch.object(
            c.c.gp, 'Model', side_effect=AssertionError('Real QP forbidden in unit tests')))
        self.enterContext(patch.object(c.c.gp, 'quicksum', side_effect=sum))
        self.enterContext(patch.object(c.c, 'gen', side_effect=AssertionError('Real data forbidden')))

    def prescribe(self, *models):
        self.model_factory.side_effect = list(models)

    def inputs(self, count=3):
        return (np.tile([4., 1., 0.], (count, 1)),
                np.tile([2., 1., 0.], (count, 1)))

    def policy(self, count=3):
        return SimpleNamespace(targets=Mock(return_value=(
            np.zeros((count, 3), dtype=np.float32), 2., np.array([1., 2., 3.], dtype=np.float32))))

    def record(self):
        return dict(phases={}, solves_attempted=0, solves_accepted=0, status_counts={},
                    mean_oos=None, finite_pool_mean=None, saved_native_finite_pool_mean=None)

    def assert_timing(self, phase, status):
        self.assertEqual(phase['status'], status)
        self.assertGreater(phase['wall_time_s'], 0)
        self.assertIn('+00:00', phase['start_utc'])
        self.assertIn('+00:00', phase['end_utc'])

    def test_hard_fee_raw_tiny_positive_zero_negative_not_cutoff_or_clipped(self):
        tiny = np.nextafter(0., 1.)
        q2 = np.array([[tiny, 0., -tiny], [5e-7, -5e-7, 0.],
                       [1e-6, 0., -1e-6], [2e-6, 0., 0.], [0., -0., -5e-7]])
        before = q2.copy()
        x1, x2 = self.inputs(len(q2))
        q1r = np.array([1., 2., 3.], dtype=np.float32)
        base = c.c.scenario_costs(2., q1r, q2, x1, x2)
        actual = c.scenario_costs(2., q1r, q2, x1, x2)
        np.testing.assert_allclose(actual - base, [.1, .1, .1, .1, 0.], rtol=0, atol=1e-15)
        self.assertFalse(np.array_equal(actual, base + .1 * (q2 > 1e-6).sum(axis=1)))
        np.testing.assert_array_equal(q2, before)
        self.assertEqual(q2.tobytes(), before.tobytes())
        # This negative coordinate changes the variable cost; clipping is not benign.
        self.assertNotEqual(c.scenario_costs(2., q1r, np.array([[-5e-7, 0., 0.]]),
                                            x1[:1], x2[:1])[0],
                            c.scenario_costs(2., q1r, np.zeros((1, 3)), x1[:1], x2[:1])[0])

    def test_true_cost_three_branches_and_float32_inventory_sum(self):
        # The three coordinates select, respectively, shortage, surplus, and
        # first-stage shortage: 5.45, .15, 1.25. Initial inventory costs 3.6.
        q1r = np.array([2., 4., 0.], dtype=np.float64)
        actual = c.scenario_costs(7., q1r, np.array([[1., 0., 6.]]),
                                  np.array([[4., 0., 3.]]), np.array([[10., 1., 5.]]))
        np.testing.assert_allclose(actual, [10.65], rtol=0, atol=2e-15)
        self.assertEqual(actual.dtype, np.float64)
        q1r = np.array([2**24, 1., 1.], dtype=np.float32)
        x = np.asarray(q1r, dtype=np.float64)[None, :]
        expected = .25 * (2. + q1r.sum()) + .1
        actual = c.scenario_costs(2., q1r, np.zeros((1, 3)), x, np.zeros((1, 3)))
        self.assertEqual(actual[0], expected)
        self.assertNotEqual(actual[0], .25 * (2. + q1r.sum(dtype=np.float64)) + .1)

    def test_finite_pool_equal_center_weight_not_row_mean(self):
        mean, maxima = c.finite_pool_cost([1., 9., -3., 2., 4.], [2, 3])
        self.assertEqual(maxima, [9., 4.])
        self.assertEqual(mean, 6.5)
        self.assertNotEqual(mean, np.mean([1., 9., -3., 2., 4.]))
        self.assertEqual(c.finite_pool_cost([-5., -2., -7.], [2, 1]), (-4.5, [-2., -7.]))

    def test_finite_pool_rejects_invalid_shapes_sizes_and_nonfinite(self):
        for costs, sizes in (([], []), ([1.], [0]), ([1.], [-1]), ([1.], [True]),
                             ([1.], [1.]), ([1.], [np.int64(1)]), ([1.], [2]),
                             ([[1.]], [1]), (np.ones((1, 1, 1)), [1]),
                             ([float('nan')], [1]), ([float('inf')], [1])):
            with self.subTest(costs=costs, sizes=sizes), self.assertRaises(ValueError):
                c.finite_pool_cost(costs, sizes)

    def seeded_state(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1729)
            return c.c.Model().state_dict()

    def test_load_green_and_legacy_exact_learned_state_and_targets_without_qp(self):
        state = self.seeded_state()
        x1, _ = self.inputs()
        outputs = []
        for method, buffers in (('green', {}), ('legacy', {}), ('legacy', {
                'proj_layer.q': torch.zeros(5, 5), 'proj_layer.A': torch.zeros(17, 5)})):
            with self.subTest(method=method, buffers=list(buffers)):
                payload = dict(state, **buffers)
                if method == 'legacy':
                    payload = {'model_state_dict': payload, 'seed': 1729}
                path = self.root / 'checkpoint.pth'
                torch.save(payload, path)
                model = c.load_model(path, method)
                self.assertFalse(model.training)
                self.assertEqual(set(model.state_dict()), set(state))
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(value, state[key]), key)
                outputs.append(model.targets(x1))
        for z, q10, q1r in outputs:
            self.assertEqual(z.shape, (3, 3))
            self.assertEqual(z.dtype, np.float32)
            self.assertEqual(q1r.dtype, np.float32)
            self.assertTrue(np.isfinite(z).all())
            self.assertGreater(q10, 0)
            for actual, expected in zip((z, q10, q1r), outputs[0]):
                np.testing.assert_array_equal(actual, expected)
        self.model_factory.assert_not_called()
        self.env_factory.assert_not_called()

    def test_load_rejects_missing_bad_learned_tensors_and_unknown_buffers(self):
        state = self.seeded_state()
        for method in ('green', 'legacy'):
            for mutation in ('missing', 'shape', 'dtype', 'nan', 'inf', 'unknown', 'wildcard'):
                bad = deepcopy(state)
                if mutation == 'missing':
                    del bad['Q10']
                elif mutation == 'shape':
                    bad['Q10'] = torch.zeros(2)
                elif mutation == 'dtype':
                    bad['Q10'] = bad['Q10'].double()
                elif mutation in ('nan', 'inf'):
                    bad['Q10'][0] = float(mutation)
                else:
                    bad['proj_layer.unrecognized' if mutation == 'wildcard' else 'unknown'] = torch.zeros(1)
                path = self.root / 'bad.pth'
                torch.save(bad if method == 'green' else {'model_state_dict': bad}, path)
                with self.subTest(method=method, mutation=mutation), self.assertRaises(ValueError):
                    c.load_model(path, method)

    def test_load_rejects_wrong_wrapper_method_and_green_projection_buffers(self):
        state = self.seeded_state()
        for method, payload in (('green', {'model_state_dict': state}), ('legacy', state),
                                ('unknown', state), ('green', dict(state, **{'proj_layer.q': torch.zeros(1)}))):
            path = self.root / 'schema.pth'
            torch.save(payload, path)
            with self.subTest(method=method, keys=list(payload)), self.assertRaises((ValueError, KeyError)):
                c.load_model(path, method)

    def test_projection_unchanged_reference_body_and_local_physical_tolerance(self):
        self.assertEqual(c.PHYSICAL_TOL, 1e-6)
        self.assertEqual(c.strict_project.__globals__['PHYSICAL_TOL'], 1e-6)
        self.assertEqual(c.c.PHYSICAL_TOL, 1e-4)
        self.assertEqual(c.c.strict_project.__globals__['PHYSICAL_TOL'], 1e-4)
        self.assertEqual(Path(c.strict_project.__code__.co_filename), c.REFERENCE)
        self.assertEqual(inspect.getsource(c.strict_project), inspect.getsource(c.c.strict_project))
        # Compare with a fresh extraction, not import-time module bytecode, which
        # need not be identical to separately compiled function bytecode.
        nodes = [node for node in ast.parse(c.REFERENCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name == 'strict_project']
        self.assertEqual(len(nodes), 1)
        namespace = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(c.REFERENCE), 'exec'), namespace)
        self.assertEqual(c.strict_project.__code__, namespace['strict_project'].__code__)

    def test_source_binding_checks_live_copy_and_expected_hash(self):
        source, copy = self.root / 'reference.py', self.root / 'frozen_reference.py'
        for mutation in ('none', 'source', 'copy', 'hash'):
            source.write_text('unchanged reference\n')
            copy.write_bytes(source.read_bytes())
            manifest = {'files': {str(source.resolve()): {'copy': str(copy), 'sha256': c.sha(source)}}}
            if mutation in ('source', 'copy'):
                (source if mutation == 'source' else copy).write_text('changed\n')
            elif mutation == 'hash':
                manifest['files'][str(source)]['sha256'] = 'wrong'
            with self.subTest(mutation=mutation):
                if mutation == 'none':
                    self.assertEqual(c.bound_file(manifest, source), copy)
                else:
                    with self.assertRaisesRegex(ValueError, 'Changed frozen file'):
                        c.bound_file(manifest, source)

    def test_projection_preserves_raw_signs_at_physical_boundary_and_fresh_models(self):
        raw = np.array([np.nextafter(0., 1.), 0., -1e-6])
        models = [solver(raw), solver([1e-6, 0., 0.])]
        self.prescribe(*models)
        for expected, budget, model in zip((raw, np.array([1e-6, 0., 0.])), (1., 0.), models):
            diag = {}
            result = c.strict_project(np.zeros(3), budget, self.environment, diag)
            self.assertEqual(result.tobytes(), expected.tobytes())
            self.assertEqual(diag['physical_residual'], 1e-6)
            self.assertTrue(diag['accepted'])
            self.assertEqual((diag['status'], diag['iterations'], diag['barrier_iterations']), (2, 7., 2))
            self.assertGreater(diag['wall_time_s'], 0)
            self.assertEqual(vars(model.Params), {'OutputFlag': 0, 'Threads': 1})
            model.addVars.assert_called_once_with(3, lb=0., name='q2')
            model.optimize.assert_called_once_with()
        self.assertEqual(self.model_factory.call_count, 2)

    def test_projection_rejects_excess_residual_nonoptimal_and_nonfinite(self):
        cases = [(solver([-np.nextafter(1e-6, np.inf), 0., 0.]), 1., RuntimeError),
                 (solver([np.nextafter(1e-6, np.inf), 0., 0.]), 0., RuntimeError),
                 (solver([0., 0., 0.], status=9), 1., RuntimeError),
                 (solver([float('nan'), 0., 0.]), 1., FloatingPointError),
                 (solver([float('inf'), 0., 0.]), 1., FloatingPointError),
                 (solver([0., 0., 0.], objective=float('nan')), 1., FloatingPointError)]
        for model, budget, exception in cases:
            self.prescribe(model)
            diag = {}
            with self.subTest(budget=budget, status=model.Status), self.assertRaises(exception):
                c.strict_project(np.zeros(3), budget, self.environment, diag)
            self.assertFalse(diag['accepted'])
            self.assertIn('error', diag)
            self.assertGreater(diag['wall_time_s'], 0)

    def test_projection_invalid_inputs_fail_before_model_construction(self):
        for z, budget in ((np.zeros(2), 1.), (np.zeros((1, 3)), 1.),
                          ([float('nan'), 0., 0.], 1.), (np.zeros(3), -1.),
                          (np.zeros(3), float('inf'))):
            with self.subTest(z=z, budget=budget), self.assertRaises(ValueError):
                c.strict_project(z, budget, self.environment, {})
        self.model_factory.assert_not_called()

    def test_evaluate_raw_journal_hard_fees_and_finite_pool_aggregation(self):
        raw = np.array([[5e-7, 0., -5e-7], [0., 0., 0.], [1., 0., 0.]])
        x1, x2 = self.inputs()
        for name, sizes, saved in (('oos', None, None), ('is', [2, 1], 4.), ('unsaved', [2, 1], None)):
            self.prescribe(solver([0., 0., 0.]), *(solver(row) for row in raw))
            model, record = self.policy(), self.record()
            record['saved_native_finite_pool_mean'] = saved
            with self.subTest(name=name):
                e.evaluate(model, x1, x2, name, record, self.root, sizes)
                model.targets.assert_called_once()
                self.assertIs(model.targets.call_args.args[0], x1)
                p = record['phases'][name]
                self.assert_timing(p, 'completed')
                for part in ('targets', 'projection', 'cost'):
                    self.assert_timing(p[part], 'completed')
                self.assertEqual((p['iterations'], p['barrier_iterations']), (21., 6))
                self.assertEqual(p['sample_count'], 3)
                self.assertEqual(record['status_counts'], {'2': 3})
                self.assertEqual((record['solves_attempted'], record['solves_accepted']), (3, 3))
                self.assertEqual(p['solver_settings']['FeasibilityTol'], 1e-6)
                self.assertEqual(p['fee_rule'], e.NEURAL_FEE)
                self.assertEqual(c.sha(p['raw_path']), p['raw_sha256'])
                with np.load(p['raw_path'], allow_pickle=False) as data:
                    np.testing.assert_array_equal(data['shipments'], raw)
                    np.testing.assert_array_equal(data['x1'], x1)
                    np.testing.assert_array_equal(data['x2'], x2)
                    costs = data['costs'].copy()
                # Analytic variable costs, with the mandated float32 inventory
                # scalar arithmetic retained (2.1 is not exactly float32).
                initial = float(.25 * (2. + np.array([1., 2., 3.], dtype=np.float32).sum()) + .1)
                expected = initial + np.array([2.75 - .55 * 5e-7, 2.65, 2.2])
                np.testing.assert_allclose(costs, expected, rtol=0, atol=2e-15)
                journal = [json.loads(line) for line in (self.root / f'{name}_solves.jsonl').read_text().splitlines()]
                self.assertEqual([d['row'] for d in journal], [0, 1, 2])
                for d, row in zip(journal, raw):
                    self.assertEqual(d['q2'], row.tolist())
                    self.assertEqual(d['positive_support'], (row > 0).tolist())
                    self.assertTrue(d['accepted'])
                if sizes is None:
                    self.assertEqual(record['mean_oos'], float(costs.mean()))
                else:
                    self.assertEqual(record['per_center_maxima'], [max(costs[:2]), costs[2]])
                    self.assertEqual(record['finite_pool_mean'], (max(costs[:2]) + costs[2]) / 2)
                    self.assertEqual(record['difference_from_saved_native'], None if saved is None
                                     else record['finite_pool_mean'] - saved)

    def test_evaluate_failure_journals_attempt_and_preserves_failed_phase_time(self):
        self.prescribe(solver([0., 0., 0.]), solver([5e-7, 0., -5e-7]),
                       solver([0., 0., 0.], status=9))
        record = self.record()
        # Deterministic elapsed-time instrumentation test, not a measured runtime.
        with patch.object(c.c.time, 'perf_counter', side_effect=itertools.count(1., .125)), \
             self.assertRaisesRegex(RuntimeError, 'OPTIMAL'):
            e.evaluate(self.policy(), *self.inputs(), 'failed', record, self.root)
        p = record['phases']['failed']
        self.assert_timing(p, 'failed')
        self.assert_timing(p['projection'], 'failed')
        self.assert_timing(p['targets'], 'completed')
        self.assertNotIn('cost', p)
        self.assertEqual((record['solves_attempted'], record['solves_accepted']), (2, 1))
        self.assertEqual((p['solves_attempted'], p['solves_accepted']), (2, 1))
        self.assertEqual(record['status_counts'], {'2': 1, '9': 1})
        self.assertEqual((p['iterations'], p['barrier_iterations']), (14., 4))
        journal = [json.loads(line) for line in (self.root / 'failed_solves.jsonl').read_text().splitlines()]
        self.assertEqual(len(journal), 2)
        self.assertEqual(journal[-1], record['failed_projection'])
        self.assertFalse(journal[-1]['accepted'])
        self.assertEqual(journal[-1]['status'], 9)
        self.assertGreater(journal[-1]['wall_time_s'], 0)
        self.assertNotIn('q2', journal[-1])  # No zero-shipment replacement.
        self.assertIsNone(record['mean_oos'])
        self.assertIsNone(record['finite_pool_mean'])
        self.assertFalse((self.root / 'failed_raw.npz').exists())
        self.assertEqual(self.model_factory.call_count, 3)  # Settings + two attempts, no retry.

    def test_evaluate_target_failure_times_phase_without_solves_or_fallback(self):
        policy, record = self.policy(), self.record()
        policy.targets.side_effect = FloatingPointError('nonfinite target')
        with self.assertRaisesRegex(FloatingPointError, 'nonfinite target'):
            e.evaluate(policy, *self.inputs(), 'targets_failed', record, self.root)
        p = record['phases']['targets_failed']
        self.assert_timing(p, 'failed')
        self.assert_timing(p['targets'], 'failed')
        self.assertEqual(record['solves_attempted'], 0)
        self.assertNotIn('projection', p)
        self.assertIsNone(record['mean_oos'])
        self.assertFalse(list(self.root.glob('targets_failed_*')))
        self.model_factory.assert_not_called()

    def result_fixture(self):
        """Write self-consistent analytic artifacts without invoking evaluate."""
        if hasattr(self, '_baseline'):
            for name, data in self._baseline_files.items():
                (self.root / name).write_bytes(data)
            return deepcopy(self._baseline)
        manifest = {'seal': 'synthetic-seal', 'test': {'arrays': {}}}
        oos_inputs = self.inputs(10000)
        for name, array in zip(('xi1', 'xi2'), oos_inputs):
            path = self.root / f'{name}.npy'
            np.save(path, array, allow_pickle=False)
            manifest['test']['arrays'][name] = dict(file=str(path), file_sha256=c.sha(path),
                array_sha256=hashlib.sha256(array.tobytes()).hexdigest())
        # Distinct rows across centers, but identical true costs: total first-
        # coordinate demand stays six and its same shortage branch stays active.
        is_x1, is_x2 = self.inputs(3)
        is_x1[:, 0] = [4., 4.5, 5.]
        is_x2[:, 0] = [2., 1.5, 1.]
        points = np.concatenate((is_x1, is_x2), axis=1)
        pools = [points[:2], points[2:]]
        spec = dict(id='analytic_green', method='green', seed=1729, radius=.2,
                    Ntotal=50, state_sha256='synthetic-state', sizes=[2, 1],
                    pools=[pool.tolist() for pool in pools], pool_sha256=c.pool_digest(pools), centers=[])
        record = {k: v for k, v in spec.items() if k not in ('pools', 'centers')}
        record.update(status='completed', manifest_sha256='synthetic-seal', phases={},
                      solves_attempted=10003, solves_accepted=10003, status_counts={'2': 10003})
        for name, count in (('oos', 10000), ('is', 3)):
            x1, x2 = oos_inputs if name == 'oos' else (is_x1, is_x2)
            q2 = np.tile([5e-7, 0., -5e-7], (count, 1))
            q1r = np.array([1., 2., 3.], dtype=np.float32)
            costs = c.scenario_costs(2., q1r, q2, x1, x2)
            path = self.root / f'{name}_raw.npz'
            np.savez(path, x1=x1, x2=x2, targets=np.zeros((count, 3), dtype=np.float32),
                     q10=np.array(2.), q1r=q1r, shipments=q2, costs=costs)
            record['phases'][name] = dict(status='completed', solves_attempted=count,
                solves_accepted=count, raw_path=str(path), raw_sha256=c.sha(path))
            with (self.root / f'{name}_solves.jsonl').open('w') as journal:
                for i in range(count):
                    journal.write(json.dumps(dict(row=i, accepted=True, status=2, physical_residual=5e-7,
                                                  q2=q2[i].tolist(), positive_support=[True, False, False])) + '\n')
            record['mean_oos' if name == 'oos' else 'finite_pool_mean'] = (
                float(costs.mean()) if name == 'oos' else c.finite_pool_cost(costs, spec['sizes'])[0])
        compact = dict(id=spec['id'], status='completed', result_path=str(self.root / 'result.json'))
        self.seal_result(record, compact)
        # Restore bytes between mutations instead of rebuilding 10,000 identical
        # analytic rows for each subtest. No production cache/freeze is involved.
        self._baseline = spec, compact, manifest, record
        self._baseline_files = {p.name: p.read_bytes() for p in self.root.iterdir()}
        return deepcopy(self._baseline)

    def seal_result(self, record, compact):
        # allow_nan permits deliberately malformed JSON in rejection cases.
        Path(compact['result_path']).write_text(json.dumps(record))
        compact['result_sha256'] = c.sha(compact['result_path'])

    def change_raw(self, record, field, transform):
        p = record['phases']['oos']
        with np.load(p['raw_path'], allow_pickle=False) as raw:
            arrays = {key: raw[key].copy() for key in raw.files}
        arrays[field] = transform(arrays[field])
        np.savez(p['raw_path'], **arrays)
        p['raw_sha256'] = c.sha(p['raw_path'])

    def test_verify_accepts_complete_analytic_artifacts(self):
        spec, compact, manifest, record = self.result_fixture()
        verified = e.verify_result(spec, compact, manifest)
        self.assertEqual(verified['mean_oos'], record['mean_oos'])
        self.assertEqual(verified['sizes'], [2, 1])
        self.model_factory.assert_not_called()
        self.env_factory.assert_not_called()

    def test_verify_rejects_wrong_oos_inputs_with_consistent_costs(self):
        for field in ('x1', 'x2'):
            spec, compact, manifest, record = self.result_fixture()
            phase = record['phases']['oos']
            journal = (self.root / 'oos_solves.jsonl').read_bytes()
            with np.load(phase['raw_path'], allow_pickle=False) as archive:
                raw = {key: archive[key] for key in archive.files}
            previous_costs = raw['costs'].copy()
            raw[field][:, 0] += 1.
            raw['costs'] = c.scenario_costs(raw['q10'].item(), raw['q1r'], raw['shipments'],
                                          raw['x1'], raw['x2'])
            self.assertFalse(np.array_equal(raw['costs'], previous_costs))
            record['mean_oos'] = float(raw['costs'].mean())
            np.savez(phase['raw_path'], **raw)
            phase['raw_sha256'] = c.sha(phase['raw_path'])
            self.seal_result(record, compact)
            # Neither the authoritative arrays nor journal identities changed.
            for info in manifest['test']['arrays'].values():
                self.assertEqual(c.sha(info['file']), info['file_sha256'])
            self.assertEqual((self.root / 'oos_solves.jsonl').read_bytes(), journal)
            with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, 'Raw scenarios differ from sealed inputs'):
                e.verify_result(spec, compact, manifest)

    def test_verify_rejects_cross_center_input_permutation_with_unchanged_costs(self):
        spec, compact, manifest, record = self.result_fixture()
        phase = record['phases']['is']
        journal = (self.root / 'is_solves.jsonl').read_bytes()
        with np.load(phase['raw_path'], allow_pickle=False) as archive:
            raw = {key: archive[key] for key in archive.files}
        # Exchange rows between the two centers, retaining row numbers, raw
        # shipments, targets, costs, aggregates and all journal identities.
        for field in ('x1', 'x2'):
            reordered = raw[field][[2, 1, 0]]
            self.assertFalse(np.array_equal(reordered, raw[field]))
            raw[field] = reordered
        np.testing.assert_array_equal(raw['costs'], c.scenario_costs(
            raw['q10'].item(), raw['q1r'], raw['shipments'], raw['x1'], raw['x2']))
        self.assertEqual(c.finite_pool_cost(raw['costs'], spec['sizes'])[0], record['finite_pool_mean'])
        np.savez(phase['raw_path'], **raw)
        phase['raw_sha256'] = c.sha(phase['raw_path'])
        self.seal_result(record, compact)
        self.assertEqual((self.root / 'is_solves.jsonl').read_bytes(), journal)
        self.assertEqual(c.pool_digest(spec['pools']), spec['pool_sha256'])
        with self.assertRaisesRegex(ValueError, 'Raw scenarios differ from sealed inputs'):
            e.verify_result(spec, compact, manifest)

    def test_verify_rejects_raw_physical_violation_hidden_by_journal_residual(self):
        for mutation, shipment in (('negative', [-2e-6, 0., 0.]),
                                   ('capacity', [2. + 2e-6, 0., 0.])):
            spec, compact, manifest, record = self.result_fixture()
            phase = record['phases']['is']
            with np.load(phase['raw_path'], allow_pickle=False) as archive:
                raw = {key: archive[key] for key in archive.files}
            raw['shipments'][0] = shipment
            raw['costs'] = c.scenario_costs(raw['q10'].item(), raw['q1r'], raw['shipments'],
                                          raw['x1'], raw['x2'])
            record['finite_pool_mean'] = c.finite_pool_cost(raw['costs'], spec['sizes'])[0]
            np.savez(phase['raw_path'], **raw)
            phase['raw_sha256'] = c.sha(phase['raw_path'])
            path = self.root / 'is_solves.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['q2'] = raw['shipments'][0].tolist()
            rows[0]['positive_support'] = (raw['shipments'][0] > 0).tolist()
            # Keep the falsely acceptable residual; all other raw/journal and
            # cost evidence is consistent, so the raw physical guard must fire.
            self.assertLessEqual(rows[0]['physical_residual'], c.PHYSICAL_TOL)
            self.assertGreater(max(0., -min(shipment), sum(shipment) - raw['q10'].item()),
                               c.PHYSICAL_TOL)
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.seal_result(record, compact)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'Raw physical violation'):
                e.verify_result(spec, compact, manifest)

    def test_verify_rejects_wrong_raw_costs_shipments_count_and_aggregate(self):
        for mutation in ('cost', 'shipment', 'count', 'oos_mean', 'pool_mean'):
            spec, compact, manifest, record = self.result_fixture()
            if mutation in ('cost', 'shipment', 'count'):
                field = 'costs' if mutation == 'cost' else 'shipments'
                self.change_raw(record, field, (lambda a: a[:-1]) if mutation == 'count' else (lambda a: a + 1.))
            else:
                record['mean_oos' if mutation == 'oos_mean' else 'finite_pool_mean'] += 1.
            self.seal_result(record, compact)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                e.verify_result(spec, compact, manifest)

    def test_verify_rejects_wrong_journal_fields_and_missing_or_extra_rows(self):
        for mutation in ('row', 'accepted', 'status', 'residual', 'negative_residual',
                         'nan_residual', 'q2', 'support', 'missing', 'extra'):
            spec, compact, manifest, _ = self.result_fixture()
            path = self.root / 'is_solves.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            changes = dict(row=('row', 1), accepted=('accepted', False), status=('status', 9),
                           residual=('physical_residual', np.nextafter(1e-6, np.inf)),
                           negative_residual=('physical_residual', -1.),
                           nan_residual=('physical_residual', float('nan')),
                           q2=('q2', [0., 0., 0.]), support=('positive_support', [False]*3))
            if mutation == 'missing':
                rows.pop()
            elif mutation == 'extra':
                rows.append(dict(rows[-1], row=3))
            else:
                key, value = changes[mutation]
                rows[0][key] = value
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, IndexError)):
                e.verify_result(spec, compact, manifest)

    def test_verify_rejects_phase_counts_hashes_status_and_record_identity(self):
        for mutation in ('attempted', 'accepted', 'phase_status', 'raw_hash', 'result_hash',
                         'manifest', 'compact_status', 'status', 'id', 'method', 'seed', 'state_sha256'):
            spec, compact, manifest, record = self.result_fixture()
            if mutation in ('attempted', 'accepted'):
                record['phases']['oos']['solves_' + mutation] -= 1
            elif mutation == 'phase_status':
                record['phases']['oos']['status'] = 'failed'
            elif mutation == 'raw_hash':
                record['phases']['oos']['raw_sha256'] = 'wrong'
            elif mutation == 'manifest':
                manifest['seal'] = 'wrong'
            elif mutation == 'compact_status':
                compact['status'] = 'failed'
            elif mutation != 'result_hash':
                record[mutation] = 'wrong'
            self.seal_result(record, compact)
            if mutation == 'result_hash':
                Path(compact['result_path']).write_text(Path(compact['result_path']).read_text() + '\n')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                e.verify_result(spec, compact, manifest)

    def test_verify_rejects_nonfinite_cost_inputs_and_aggregates(self):
        for field in ('shipments', 'costs', 'q10', 'q1r', 'x1', 'x2', 'mean_oos', 'finite_pool_mean'):
            for value in (float('nan'), float('inf')):
                spec, compact, manifest, record = self.result_fixture()
                if field in ('mean_oos', 'finite_pool_mean'):
                    record[field] = value
                else:
                    self.change_raw(record, field, lambda a: np.full(a.shape, value, dtype=a.dtype))
                self.seal_result(record, compact)
                with self.subTest(field=field, value=value), np.errstate(invalid='ignore'), self.assertRaises(ValueError):
                    e.verify_result(spec, compact, manifest)

    def test_verify_rejects_wrong_global_solve_counts(self):
        for key, value in (('solves_attempted', 10004), ('solves_accepted', 10002),
                           ('status_counts', {'2': 10002, '9': 1})):
            spec, compact, manifest, record = self.result_fixture()
            record[key] = value
            self.seal_result(record, compact)
            with self.subTest(key=key), self.assertRaises(ValueError):
                e.verify_result(spec, compact, manifest)

    def test_verify_rejects_wrong_compact_identity(self):
        spec, compact, manifest, _ = self.result_fixture()
        compact['id'] = 'another_job'
        with self.assertRaises(ValueError):
            e.verify_result(spec, compact, manifest)

    def test_verify_rejects_nonfinite_saved_targets(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            spec, compact, manifest, record = self.result_fixture()
            self.change_raw(record, 'targets', lambda a: np.full_like(a, value))
            self.seal_result(record, compact)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Nonfinite raw evidence'):
                e.verify_result(spec, compact, manifest)


if __name__ == '__main__':
    unittest.main()