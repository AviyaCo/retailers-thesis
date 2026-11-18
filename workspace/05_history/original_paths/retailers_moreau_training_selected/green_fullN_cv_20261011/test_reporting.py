"""Synthetic/analytic adapter tests only; never generate/load historical test data."""
import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('_reporting_under_test', Path(__file__).with_name('reporting.py'))
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def synthetic_rows():
    """TEST ONLY: invented values, not trained policies or research results."""
    return [dict(kind=k, Ntotal=n, seed=s, radius=rho, status='completed',
                 converged=True, complete=True, mean_oos=(rho - .03)**2 + s,
                 finite_pool_mean=rho + s, test=dict(n_test=10000, seed=9999,
                 already_seen=True, arrays={name: dict(shape=[10000, 10 if k == 'R10' else 3],
                 array_sha256='synthetic-test-only') for name in ('xi1', 'xi2')}))
            for k in r.KINDS for n in r.SIZES for s in r.SEEDS for rho in r.RADII]


class SourceAndBarrierTests(unittest.TestCase):
    def test_import_performs_no_file_or_data_reads(self):
        source = Path(r.__file__).read_text()
        with patch('builtins.open', side_effect=AssertionError('Unexpected import IO')), \
                patch.object(Path, 'read_text', side_effect=AssertionError('Unexpected import IO')), \
                patch.object(Path, 'open', side_effect=AssertionError('Unexpected import IO')):
            exec(compile(source, r.__file__, 'exec'), {'__file__': r.__file__})

    def test_source_bindings_only_python_and_reject_mutation(self):
        bindings = r.source_bindings()
        self.assertTrue(all(Path(p).suffix == '.py' and len(sha) == 64 for p, sha in bindings.items()))
        with patch.object(r, '_sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Changed original source'):
                r.source_bindings()

    def test_grid_cannot_drop_duplicate_or_fail(self):
        rows = synthetic_rows()
        r._grid(rows)
        for bad in (rows[:-1], rows[:-1] + [rows[0]], [dict(x, converged=False) for x in rows]):
            with self.assertRaises(ValueError):
                r._grid(bad)

    def test_no_gate_no_test_access(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, '_runtime', side_effect=AssertionError('Test access')):
            output = Path(tmp) / 'out'
            with self.assertRaises(KeyError):
                r.evaluate_configuration('R3', {}, output)
            self.assertFalse(output.exists())

    def test_explicit_authorization_and_complete_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'seal.json'
            gate = dict(status='sealed', allow_test_regeneration=True,
                        configurations=synthetic_rows(), source_bindings=r.source_bindings())
            r._write(path, gate)
            info = dict(path=str(path), sha256=r._sha(path))
            self.assertEqual(len(r._gate(info)['configurations']), 810)
            for mutation in (dict(allow_test_regeneration=False), dict(configurations=synthetic_rows()[:-1]),
                             dict(source_bindings={})):
                changed = dict(gate, **mutation)
                with patch.object(r, '_read', return_value=changed):
                    with self.assertRaises(ValueError):
                        r._gate(info)
            with self.assertRaisesRegex(ValueError, 'Changed training barrier'):
                r._gate(dict(info, sha256='not-the-hash'))


class NumericalReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    def test_no_common_collision_and_no_old_initializer(self):
        r._runtime('shipping_R3')  # Load extensions before patch.dict restores sys.modules.
        sentinel = object()
        with patch.dict(sys.modules, common=sentinel):
            shared, ns, load, evaluate = r._runtime('shipping_R3')
            self.assertIs(sys.modules['common'], sentinel)
            self.assertEqual(Path(evaluate.__code__.co_filename), r.SHIPPING / 'evaluation.py')
            self.assertEqual(Path(load.__code__.co_filename), r.SHIPPING / 'common.py')
            self.assertEqual(Path(evaluate.__globals__['c'].strict_project.__code__.co_filename), r.COMPARE)
            self.assertEqual(shared.PHYSICAL_TOL, 1e-6)
            self.assertNotIn('initialize', evaluate.__globals__)

    def test_original_float32_targets_and_cost_order(self):
        import numpy as np
        import torch
        shared, ns, _, evaluate = r._runtime('shipping_R3')
        common = evaluate.__globals__['c']
        torch.manual_seed(17)
        model = common.c.Model()
        # Synthetic small batch only. No scenario generator or solver invocation.
        x1 = np.arange(12, dtype=np.float64).reshape(4, 3) / 10
        x2 = x1 + .01
        original = model.targets(x1)
        adapted = shared.targets(model, x1)
        for first, second in zip(original, adapted):
            np.testing.assert_array_equal(first, second)
        z, q10, q1r = original
        self.assertEqual(z.dtype, np.float32)
        self.assertEqual(q1r.dtype, np.float32)
        q2 = np.array([[0, 1e-14, -.00000001]] * 4, dtype=np.float64)
        expected = common.c.scenario_costs(q10, q1r, q2, x1, x2)
        np.testing.assert_array_equal(shared.scenario_costs(q10, q1r, q2, x1, x2), expected)
        np.testing.assert_array_equal(common.scenario_costs(q10, q1r, q2, x1, x2), expected + .1)
        self.assertEqual(common.strict_project.__globals__['PHYSICAL_TOL'], 1e-6)

    def test_r10_generator_dimension_with_mocked_draws_only(self):
        import numpy as np
        _, ns, _, evaluator = r._runtime('R10')
        self.assertIsNone(evaluator)
        first, second = np.ones((2, 10)), np.full((2, 10), 2.)
        with patch.object(ns['np'].random, 'seed') as seed, \
                patch.object(ns['truncnorm'], 'rvs', side_effect=[first, second]) as draws:
            result = ns['gen'](2, 123)  # Not 10000/9999; draws are synthetic mocks.
        seed.assert_called_once_with(123)
        self.assertEqual(draws.call_args_list[0].kwargs['size'], (2, 10))
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)

    def pool_fixture(self, folder, ns, R=3):
        import numpy as np
        centers = np.zeros((50, 2 * R), dtype=np.float64)
        pools = [x[None, :].copy() for x in centers]
        pools[0][0, 0] = -5e-10  # Approved native membership residual; must remain raw.
        digest = ns['pool_digest'](pools)
        r._write(folder / 'pool.json', dict(pools=[p.tolist() for p in pools],
            pool_sha256=digest, original_sample_ids=list(range(50))))
        (folder / 'inputs').mkdir()
        data_hash = ns['raw_data_hash'](dict(xi1=centers[:, :R], xi2=centers[:, R:]))
        r._write(folder / 'inputs/training_data.json', dict(centers=centers.tolist(),
            original_sample_ids=list(range(50)), data_sha256=data_hash))
        chosen = dict(candidate_id=1, assessed=True, state_sha256='state', checkpoint_sha256='checkpoint',
            ap_reference='ap', checkpoint=str(folder / 'candidate.pth'),
            pool_sha256=digest, pool_file_sha256=r._sha(folder / 'pool.json'),
            ap_incumbent_mean=1., restricted_mean=1.)
        first = dict(candidate_id=0, assessed=True, ap_incumbent_mean=0., restricted_mean=1.)
        for row in (first, chosen):
            row.update(ns['terminal_decision'](row))
        return dict(Ntotal=50, selected_candidate_id=1, stopping_candidate_id=1,
            selection_rule='stopping_checkpoint', radius=0., summary_path=str(folder / 'summary.json'),
            training_data_sha256=r._sha(folder / 'inputs/training_data.json'), data_sha256=data_hash,
            selected_state_sha256='state', selected_checkpoint_sha256='checkpoint',
            selected_ap_reference='ap', iterations=[first, chosen])

    def test_pool_rejects_prestop_selection_and_malformed_prefix(self):
        _, ns, _, _ = r._runtime('R3')
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.pool_fixture(Path(tmp), ns)
            for mutation in (dict(selected_candidate_id=0), dict(selected_candidate_id=True),
                             dict(stopping_candidate_id=True), dict(selection_rule='minAP')):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    r._own_pool(dict(summary, **mutation), ns, 3)
            for index, mutation in ((0, dict(assessed=False)), (0, dict(candidate_id=9)),
                    (0, dict(ap_incumbent_mean=float('nan'))), (0, dict(green_pass=True)),
                    (0, dict(ap_incumbent_mean=1., green_gap=0., green_pass=True)),
                    (1, dict(green_pass=False)), (1, dict(green_gap=.1))):
                bad = deepcopy(summary)
                bad['iterations'][index].update(mutation)
                with self.subTest(index=index, mutation=mutation), self.assertRaises(ValueError):
                    r._own_pool(bad, ns, 3)
            summary['iterations'].append({})  # Never inspect unassessed future candidates.
            self.assertEqual(len(r._own_pool(summary, ns, 3)[0]), 50)

    def test_stop_strict_boundary_and_floor(self):
        import numpy as np
        _, ns, _, _ = r._runtime('R3')
        with tempfile.TemporaryDirectory() as tmp:
            summary = self.pool_fixture(Path(tmp), ns)
            row = summary['iterations'][1]
            for pool in (0., 100.):
                # Exactly representable quotient at .01, including the 1e-9 floor.
                ap = .01 * 1e-9 if pool == 0 else 99.
                row.update(restricted_mean=pool, ap_incumbent_mean=ap,
                           green_gap=.01, green_pass=True)
                self.assertEqual(ns['terminal_decision'](row)['green_gap'], .01)
                with self.assertRaises(ValueError):
                    r._own_pool(summary, ns, 3)
                row['ap_incumbent_mean'] = np.nextafter(ap, pool).item()
                row.update(ns['terminal_decision'](row))
                self.assertTrue(row['green_pass'])
                self.assertEqual(len(r._own_pool(summary, ns, 3)[0]), 50)

    def test_raw_membership_training_authentication_and_r10(self):
        import numpy as np
        for kind, R in (('R3', 3), ('R10', 10), ('shipping_R3', 3)):
            _, ns, _, _ = r._runtime(kind)
            self.assertEqual(Path(ns['check_membership'].__code__.co_filename), r.CCG)
            self.assertEqual(ns['MEMBERSHIP_TOLERANCE'], 1e-9)
            with tempfile.TemporaryDirectory() as tmp:
                summary = self.pool_fixture(Path(tmp), ns, R)
                pools, binding = r._own_pool(summary, ns, R)
                self.assertEqual(pools[0][0, 0], -5e-10)
                self.assertEqual(binding['sha256'], summary['iterations'][1]['pool_file_sha256'])
                for coordinate in (-1e-9, 1e-9):
                    point = np.zeros((1, 2 * R), dtype=np.float64)
                    point[0, 0] = coordinate
                    self.assertEqual(ns['check_membership'](point, np.zeros(2 * R), 0.), 1e-9)
                    point[0, 0] = np.nextafter(coordinate, np.sign(coordinate) * np.inf)
                    with self.assertRaisesRegex(ValueError, 'Box membership'):
                        ns['check_membership'](point, np.zeros(2 * R), 0.)
                with self.assertRaisesRegex(ValueError, 'Changed training centers'):
                    r._own_pool(dict(summary, training_data_sha256='changed'), ns, R)
                with self.assertRaisesRegex(ValueError, 'raw training data hash'):
                    r._own_pool(dict(summary, data_sha256='changed'), ns, R)
                training_path = Path(binding['training_data_path'])
                training = r._read(training_path)
                training['original_sample_ids'][0:2] = [1, 0]
                with patch.object(r, '_read', side_effect=lambda p: training if Path(p) == training_path
                                  else json.loads(Path(p).read_text())):
                    with self.assertRaisesRegex(ValueError, 'misaligned'):
                        r._own_pool(summary, ns, R)
            mean, maxima = ns['finite_pool_cost'](np.array([1., 3., 2.]), [2, 1])
            self.assertEqual((mean, maxima), (2.5, [3., 2.]))

    def test_successful_mocked_replay_both_branches_and_final_binding_check(self):
        import numpy as np
        for kind in r.KINDS:
            for mutate in (False, True):
                with self.subTest(kind=kind, mutate=mutate), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    R = 10 if kind == 'R10' else 3
                    _, ns, _, _ = r._runtime(kind)  # Definitions only; all solver/generator calls mocked below.
                    pools = [np.zeros((1, 2 * R), dtype=np.float64)]
                    r._write(root / 'pool.json', {'synthetic': True})
                    r._write(root / 'train.json', {'synthetic': True})
                    binding = dict(path=str(root / 'pool.json'), sha256=r._sha(root / 'pool.json'),
                        training_data_path=str(root / 'train.json'), training_data_sha256=r._sha(root / 'train.json'))
                    compact = {k: 0 for k in r.IDENTITY}
                    compact.update(selected_state_sha256='synthetic-state', evaluation_gate={'synthetic': True})
                    calls = []

                    def replay(model, a, b, output, raw, sizes=None):
                        calls.append(raw['role'])
                        raw.update(status='completed', solves_attempted=len(a), solves_accepted=len(a),
                                   status_counts={'2': len(a)}, mean=2. if sizes is None else 3.)
                        if mutate and sizes is not None:
                            (root / 'pool.json').write_text('synthetic mutation')

                    def shipping(model, a, b, role, raw, folder, sizes=None):
                        raw['role'] = role
                        replay(model, a, b, folder, raw, sizes)
                        raw['mean_oos' if sizes is None else 'finite_pool_mean'] = raw['mean']

                    shared = SimpleNamespace(state_hash=lambda _: 'synthetic-state', evaluate_policy=replay)
                    with patch.dict(ns, gen=lambda n, seed: (np.zeros((n, R)), np.ones((n, R)))), \
                            patch.object(r, '_runtime', return_value=(shared, ns, lambda *_: object(),
                                shipping if kind == 'shipping_R3' else None)), \
                            patch.object(r, '_gate', return_value={'source_bindings': {}, 'evaluation_workers': 1}), \
                            patch.object(r, '_identity', return_value={'selected_restricted_mean': 3.}), \
                            patch.object(r, '_own_pool', return_value=(pools, binding)):
                        folder = root / 'output'
                        if mutate:
                            with self.assertRaisesRegex(ValueError, 'Pool changed during replay'):
                                r.evaluate_configuration(kind, compact, folder)
                        else:
                            result = r.evaluate_configuration(kind, compact, folder)
                            self.assertEqual(result['status'], 'completed')
                            self.assertEqual((result['mean_oos'], result['finite_pool_mean']), (2., 3.))
                            self.assertEqual(result['own_pool_binding'], binding)
                            self.assertEqual(result['own_pool']['solves_accepted'], 1)
                            self.assertEqual(r._sha(result['result_path']), result['result_sha256'])
                        self.assertEqual(calls, ['oos', 'own_pool'])
                        self.assertEqual(r._read(folder / 'result.json')['status'], 'failed' if mutate else 'completed')

    def test_runtime_failure_persisted_and_propagated(self):
        compact = {k: 0 for k in r.IDENTITY}
        compact['evaluation_gate'] = {'path': 'synthetic-only', 'sha256': 'fixture'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, '_gate', return_value={'source_bindings': {}}), \
                patch.object(r, '_identity', return_value={}), \
                patch.object(r, '_runtime', side_effect=RuntimeError('synthetic failure')):
            folder = Path(tmp) / 'failed'
            with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                r.evaluate_configuration('R3', compact, folder)
            record = json.loads((folder / 'result.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertIsNone(record['mean_oos'])
            self.assertGreaterEqual(record['wall_time_s'], 0)


class ReportingTests(unittest.TestCase):
    def test_three_panels_all_radii_and_mean_only_hindsight(self):
        renderer = r._shared('plot')
        for kind in r.KINDS:
            data, minima, R = r._plot_data(kind, synthetic_rows())
            renderer.validate_plot_data(data, R)
            self.assertEqual([p['Ntotal'] for p in data['panels']], list(r.SIZES))
            self.assertEqual([m['radii'] for m in minima], [[.03]] * 3)
            self.assertTrue(all(set(p) == {'Ntotal', 'label', 'green'} for p in data['panels']))
            self.assertIn('already seen', data['footnote'])
            self.assertIn('descriptive hindsight', data['footnote'])

    def test_unpaired_or_nonfinite_results_fail(self):
        rows = synthetic_rows()
        rows[0]['mean_oos'] = float('nan')
        with self.assertRaises(ValueError):
            r._plot_data('R3', rows)
        rows = synthetic_rows()
        rows[0]['test']['arrays']['xi1']['array_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            r._plot_data('R3', rows)

    def test_publish_rejects_incomplete_grid_before_creating_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(r, '_gate', return_value={}):
            folder = Path(tmp) / 'plots'
            with self.assertRaises(ValueError):
                r.publish(folder, synthetic_rows()[:-1], {'evaluation_gate': {}})
            self.assertFalse(folder.exists())

    def test_original_renderer_synthetic_smoke(self):
        # Temporary test artifacts only, removed on exit; not published model results.
        data, _, R = r._plot_data('shipping_R3', synthetic_rows())
        with tempfile.TemporaryDirectory() as tmp:
            paths = r._shared('plot').render(data, Path(tmp) / 'synthetic-only', R)
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(p.is_file() for p in paths))


def train_only_preflight(kind, output, record):
    """Real native strict QPs on two saved TRAIN rows; no generator or gate bypass API."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summary_path = r.ROOT / 'backend_checks_v3' / kind / 'full50/ccg/seed_0/radius_0/summary.json'
    summary_hash = r._sha(summary_path)
    summary = r._read(summary_path)
    r._check(summary['status'] == 'completed' and summary['converged'] is True
             and summary['complete'] is True and summary['Ntotal'] == summary['Ntrain'] == 50
             and summary['Nval'] == 0 and summary['seed'] == 0 and summary['radius'] == 0.,
             'Require completed train-only full50 scratch probe')
    shared, ns, load, shipping = r._runtime(kind)
    with patch.dict(ns, gen=lambda *_: (_ for _ in ()).throw(AssertionError('No generator in preflight'))), \
            patch.object(r, 'evaluate_configuration', side_effect=AssertionError('No post-barrier evaluation')):
        R = 10 if kind == 'R10' else 3
        _, binding = r._own_pool(summary, ns, R)
        training = r._read(binding['training_data_path'])
        ids = training['original_sample_ids']
        centers = np.asarray(training['centers'], dtype=np.float64)
        rows = centers[[ids.index(0), ids.index(1)]]
        checkpoint = summary['selected_checkpoint']
        r._check(r._sha(checkpoint) == summary['selected_checkpoint_sha256'], 'Changed scratch checkpoint')
        model = load(checkpoint, 'green')
        r._check(shared.state_hash(model) == summary['selected_state_sha256'], 'Changed scratch state')
        record.update(method='strict_train_only_evaluator_probe', kind=kind, Ntotal=50, sample_count=2,
            original_sample_ids=[0, 1], seed=0, radius=0., selected_candidate_id=summary['selected_candidate_id'],
            summary_path=str(summary_path), summary_sha256=summary_hash, own_pool_binding=binding,
            selected_checkpoint=checkpoint, selected_checkpoint_sha256=summary['selected_checkpoint_sha256'],
            source_bindings=r.source_bindings(), current_invocation_only=True,
            cached_training_time='excluded; not measured by this invocation', torch_threads=1,
            solver_threads=1, worker_count=1, physical_tolerance=1e-6,
            input_sha256={name: r.hashlib.sha256(a.tobytes()).hexdigest()
                          for name, a in (('xi1', rows[:, :R]), ('xi2', rows[:, R:]))})
        raw = dict(role='train_preflight', kind=kind, Ntotal=50, seed=0, radius=0., worker_count=1)
        record['evaluation'] = raw
        with ns['phase'](record.setdefault('phases', {}), 'strict_train_evaluation'):
            if shipping is None:
                shared.evaluate_policy(model, rows[:, :R], rows[:, R:], output / 'train', raw)
                r._check(raw['status'] == 'completed', 'Incomplete strict train evaluation')
                record['iterations'] = raw['iterations']
            else:
                raw.update(phases={}, status_counts={}, solves_attempted=0, solves_accepted=0)
                shipping(model, rows[:, :R], rows[:, R:], 'train_preflight', raw, output)
                record['iterations'] = raw['phases']['train_preflight']['iterations']
            r._check(raw['solves_attempted'] == raw['solves_accepted'] == 2
                     and raw['status_counts'] == {'2': 2}, 'Unaccepted train-only QPs')
        r._check(shared.state_hash(model) == summary['selected_state_sha256'], 'Changed state in preflight')
        r._check(r._sha(summary_path) == summary_hash, 'Changed scratch summary')
        r._check(r._sha(checkpoint) == summary['selected_checkpoint_sha256'], 'Changed scratch checkpoint')
        r._own_pool(summary, ns, R)
        r._check(record['source_bindings'] == r.source_bindings(), 'Changed preflight sources')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, help='Existing empty directory for measured test/probe artifacts')
    parser.add_argument('--preflight-kind', choices=r.KINDS, help='Real two-TRAIN-row strict QP probe only')
    args = parser.parse_args()
    if args.preflight_kind and args.scratch is None:
        parser.error('--preflight-kind requires --scratch')
    if args.scratch is not None and (not args.scratch.is_dir() or any(args.scratch.iterdir())):
        parser.error('--scratch must be an existing empty directory')
    start = time.perf_counter()
    record = dict(started_at_utc=r.datetime.now(r.timezone.utc).isoformat(), status='running',
        scope='real train-only strict evaluator preflight' if args.preflight_kind else 'synthetic unit tests; not research results',
        timing_clock='time.perf_counter', timing_unit='seconds', command=sys.argv,
        environment=dict(platform=r.platform.platform(), hostname=r.platform.node()), workers=1)
    try:
        if args.preflight_kind:
            train_only_preflight(args.preflight_kind, args.scratch, record)
            success = True
        else:
            result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
            record.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors))
            success = result.wasSuccessful()
        record['status'] = 'completed' if success else 'failed'
    except BaseException as error:
        record.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        record.update(ended_at_utc=r.datetime.now(r.timezone.utc).isoformat(), wall_time_s=time.perf_counter()-start)
        record['wall_time_human'] = f"{record['wall_time_s']:.3f} s / {record['wall_time_s']/60:.3f} min"
        if args.scratch is not None:
            r._write(args.scratch / 'test_summary.json', record)
    raise SystemExit(0 if success else 1)