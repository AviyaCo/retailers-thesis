"""Small solver fixtures only; no trained checkpoint or dataset benchmark."""
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import study as s

np, torch = s.np, s.torch
ARTIFACTS = []


def fixture():
    g = np.array([[.4, -.3, .5], [-.3, .7, .1], [.2, -.1, .8]])
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return dict(z=np.array([[1., 2., 3.], [3., 2., -1.], [-2., 1., 4.]]),
                capacity=np.array([[10.], [2.], [2.]]), upstream=g)


class Preflight(unittest.TestCase):
    def test_actual_supported_backward_sign_capacity_and_reference_fd(self):
        f = fixture()
        record = dict(test=self.id(), moreau={}, scs={}, gurobi=[])
        ARTIFACTS.append(record)
        s.run_pass(s.core.NativeProjection(), f, record['moreau'])
        s.run_pass(s.DirectQP(dict(eps=1e-8, tau=1e-3)), f, record['scs'])
        native, actual = record['moreau'], record['scs']
        for key in ('x', 'grad_z', 'grad_capacity'):
            np.testing.assert_allclose(actual[key], native[key], atol=1e-5, rtol=0)
        # Interior projection is identity: this catches the c=-z chain sign.
        np.testing.assert_allclose(actual['grad_z'][0], f['upstream'][0], atol=1e-5, rtol=0)
        self.assertAlmostEqual(actual['grad_capacity'][0, 0], 0., places=6)
        self.assertAlmostEqual(actual['grad_capacity'][1, 0], f['upstream'][1, :2].mean(), places=5)
        calls = actual['scs_solves']
        self.assertEqual(len(calls), 9)
        self.assertTrue(all(r['info']['status_val'] == 1 for r in calls))
        self.assertTrue(all(not r['warm_start_supplied'] for r in calls[:3]))
        self.assertTrue(all(r['warm_start_supplied'] for r in calls[3:]))
        self.assertTrue(all(r['settings']['acceleration_lookback'] == 10 for r in calls))
        # Independent central derivatives, separately perturbing z and capacity.
        direction = np.array([1., -2., .5]); direction /= np.linalg.norm(direction)
        for i, (z, c) in enumerate(zip(f['z'], f['capacity'][:, 0])):
            x = s.reference_qp(z, c, record['gurobi'], f'base/{i}')
            np.testing.assert_allclose(x, native['x'][i], atol=1e-6, rtol=0)
            for target in ('z', 'capacity'):
                h = 1e-3
                scalars = []
                for sign in (1, -1):
                    zp = z+sign*h*direction if target == 'z' else z
                    cp = c+sign*h if target == 'capacity' else c
                    xp = s.reference_qp(zp, cp, record['gurobi'], f'{i}/{target}/{sign}')
                    scalars.append(f['upstream'][i] @ xp)
                fd = (scalars[0]-scalars[1])/(2*h)
                predicted = actual['grad_z'][i] @ direction if target == 'z' else actual['grad_capacity'][i, 0]
                self.assertLess(abs(fd-predicted), 1e-4)

    def test_all_six_settings_small_fixture_record_raw_diagnostics(self):
        # A feasibility diagnostic is not a claim that loose settings are accurate.
        f = {k: v[:1] for k, v in fixture().items()}
        for config in s.CONFIGS[1:]:
            with self.subTest(config=config['name']):
                record = dict(test=self.id(), config=config)
                ARTIFACTS.append(record)
                s.run_pass(s.DirectQP(config), f, record)
                self.assertEqual(len(record['scs_solves']), 3)
                self.assertEqual(len(record['raw_adjoints']), 1)

    def test_inaccurate_forward_rejected_without_retry(self):
        fake = dict(x=np.ones(3), y=np.zeros(4), s=np.zeros(4),
                    info=dict(status_val=2, status='solved (inaccurate - reached max_iters)', iter=200000))
        observer = s.SCSObserver()
        with patch.object(s.scs, 'solve', return_value=fake) as solve:
            with observer, self.assertRaisesRegex(RuntimeError, 'strict status'):
                f = fixture()
                s.DirectQP(dict(eps=1e-4, tau=1e-3)).forward(f['z'][:1], f['capacity'][:1], observer)
            self.assertEqual(solve.call_count, 1)
        self.assertEqual(observer.records[0]['info']['status_val'], 2)
        self.assertEqual(observer.records[0]['status'], 'failed')
        ARTIFACTS.append(dict(test=self.id(), observed=observer.records))

    def test_inaccurate_backward_rejected_logged_and_no_retry(self):
        original = s.scs.solve
        for fail_call, stage in enumerate(('base', 'plus', 'minus'), start=1):
            for field, value, error, message in (
                ('status_val', 2, RuntimeError, 'strict status'),
                ('res_dual', np.nan, FloatingPointError, 'info.res_dual'),
                *((key, value, FloatingPointError, f'info.{key}')
                  for key in sorted(s.CERTIFICATES) for value in (np.inf, -np.inf)),
            ):
                with self.subTest(stage=stage, field=field, value=value):
                    count = 0

                    def fail_solve(*args, **kwargs):
                        nonlocal count
                        count += 1
                        result = original(*args, **kwargs)
                        if count == fail_call:
                            result['info'][field] = value
                        return result

                    record = dict(test=self.id(), injected_stage=stage,
                                  injected_field=field, injected_value=value)
                    ARTIFACTS.append(record)
                    with patch.object(s.scs, 'solve', side_effect=fail_solve):
                        with self.assertRaisesRegex(error, message):
                            s.run_pass(s.DirectQP(dict(eps=1e-8, tau=1e-3)),
                                       {k: v[:1] for k, v in fixture().items()}, record)
                    self.assertEqual(count, fail_call)
                    self.assertEqual(record['status'], 'failed')
                    self.assertEqual(len(record['scs_solves']), fail_call)
                    self.assertEqual(record['scs_solves'][-1]['warm_start_supplied'], fail_call > 1)

    def test_nonfinite_info_and_gradient_rejected_preserved(self):
        base = dict(x=np.ones(3), y=np.zeros(4), s=np.zeros(4),
                    info=dict(status_val=1, status='solved', res_pri=float('nan'),
                              res_dual=0., gap=0., pobj=0., dobj=0.))
        observer = s.SCSObserver()
        with patch.object(s.scs, 'solve', return_value=base):
            with observer, self.assertRaisesRegex(FloatingPointError, 'info.res_pri'):
                f = fixture()
                s.DirectQP(dict(eps=1e-8, tau=1e-3)).forward(f['z'][:1], f['capacity'][:1], observer)
        raw = []
        adjoint = lambda *args: (s.sparse.eye(3, format='csc'), np.zeros(4), np.full(3, np.nan))
        with self.assertRaisesRegex(FloatingPointError, 'dc'):
            s.DirectQP(dict(eps=1e-8, tau=1e-3)).backward([adjoint], np.ones((1, 3)), observer, raw)
        self.assertTrue(np.isnan(raw[0]['dc']).all())
        encoded = s.core._json_value(dict(solves=observer.records, adjoints=raw))
        self.assertEqual(encoded['solves'][0]['info']['res_pri'], 'NaN')
        self.assertEqual(encoded['adjoints'][0]['dc'][0], 'NaN')
        ARTIFACTS.append(dict(test=self.id(), **encoded))

    def test_certificate_nan_semantics_are_narrow_and_raw_preserved(self):
        for key in sorted(s.CERTIFICATES):
            for value in (np.nan, 0., 1., np.inf, -np.inf, None, 'NaN', True, 1j):
                with self.subTest(field=key, value=value):
                    base = dict(x=np.ones(3), y=np.zeros(4), s=np.zeros(4),
                                info=dict(status_val=1, status='solved', res_pri=0.,
                                          res_dual=0., gap=0., pobj=0., dobj=0.,
                                          **{k: np.nan for k in s.CERTIFICATES}))
                    base['info'][key] = value
                    observer = s.SCSObserver()
                    with patch.object(s.scs, 'solve', return_value=base) as solve, observer:
                        data = dict(b=np.array([0., 0., 0., 10.]), c=-np.ones(3))
                        if value is None or isinstance(value, (str, bool, complex)):
                            with self.assertRaisesRegex(TypeError, f'info.{key}'):
                                s.scs.solve(data, {'l': 4}, acceleration_lookback=10)
                        elif np.isinf(value):
                            with self.assertRaisesRegex(FloatingPointError, f'info.{key}'):
                                s.scs.solve(data, {'l': 4}, acceleration_lookback=10)
                        else:
                            actual = s.scs.solve(data, {'l': 4}, acceleration_lookback=10)
                            self.assertIs(actual, base)
                            encoded = s.core._json_value(observer.records[0]['info'])
                            for certificate in s.CERTIFICATES:
                                expected = value if certificate == key and not np.isnan(value) else 'NaN'
                                self.assertEqual(encoded[certificate], expected)
                        self.assertEqual(solve.call_count, 1)
                    self.assertIs(observer.records[0]['info'][key], value)

    def test_actual_diagnostics_required_numeric_and_finite(self):
        for key in ('res_pri', 'res_dual', 'gap', 'pobj', 'dobj'):
            for value in ('missing', None, '0', True, 1j, np.nan, np.inf, -np.inf):
                with self.subTest(field=key, value=value):
                    info = dict(status_val=1, status='solved', res_pri=0., res_dual=0.,
                                gap=0., pobj=0., dobj=0.)
                    if value == 'missing':
                        del info[key]
                        error = KeyError
                    else:
                        info[key] = value
                        error = TypeError if value is None or isinstance(value, (str, bool, complex)) else FloatingPointError
                    base = dict(x=np.ones(3), y=np.zeros(4), s=np.zeros(4), info=info)
                    observer = s.SCSObserver()
                    with patch.object(s.scs, 'solve', return_value=base) as solve, observer:
                        with self.assertRaisesRegex(error, key):
                            s.scs.solve(dict(b=np.array([0., 0., 0., 10.]), c=-np.ones(3)),
                                        {'l': 4}, acceleration_lookback=10)
                        self.assertEqual(solve.call_count, 1)
                    self.assertEqual(observer.records[0]['status'], 'failed')
                    self.assertEqual(set(observer.records[0]['info']), set(info))

    def test_preflight_and_study_provenance_include_test_source_hash(self):
        path = Path(__file__).resolve()
        self.assertEqual(s.provenance()['source_sha256'][str(path)], s.core.digest(path))

    def test_first_repeat_separation_and_native_record_drain(self):
        f = fixture()
        arm = s.core.NativeProjection()
        records = []
        for i in range(2):
            r = dict(kind='first' if i == 0 else 'repeat')
            s.run_pass(arm, f, r)
            records.append(r)
            self.assertEqual(len(arm.calls), 0)
            self.assertEqual(arm.drain_summary()['count'], 0)
        stats = s.repeat_stats(records)
        self.assertEqual(stats['total']['count'], 1)
        self.assertEqual(stats['total']['mean'], records[1]['wall_time_s'])
        for record in records:
            self.assertGreaterEqual(record['wall_time_s'], sum(record[k]['wall_time_s'] for k in ('setup', 'forward', 'backward')))
        ARTIFACTS.append(dict(test=self.id(), passes=records, repeat_stats=stats))

    def test_checkpoint_train_and_leaf_hash_binding(self):
        # Fabricated file-format fixture ONLY, never used as experimental data.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed_folder = root/'seed_0'; seed_folder.mkdir()
            checkpoint = seed_folder/'selected.pth'
            model = s.core.model_from_state(s.core.weights(s.core.baseline.Model()))
            torch.save(s.core.weights(model), checkpoint)
            indices = np.random.default_rng(100000).permutation(50)[:40]
            x1 = np.arange(150, dtype=np.float64).reshape(50, 3)/100
            x2 = x1+1
            raw_hash = s.array_hash(x1, x2)
            split = dict(raw_data_sha256=raw_hash, arrays=dict(xi1=x1, xi2=x2,
                train_indices=indices, train_x1=x1[indices].astype(np.float32),
                train_x2=x2[indices].astype(np.float32)))
            split_path = seed_folder/'split.json'
            s.core.write_json(split_path, split)
            summary = dict(complete=True, raw_data_sha256=raw_hash,
                split_artifact_sha256=s.core.digest(split_path), winner=dict(checkpoint=str(checkpoint),
                selected_sha256=s.core.digest(checkpoint), lr_net=.01))
            summary_path = seed_folder/'summary.json'
            s.core.write_json(summary_path, summary)
            frozen = s.bind_fixture(0, root)
            np.testing.assert_array_equal(frozen['train_indices'], indices)
            self.assertEqual(frozen['train_pair_sha256'], s.array_hash(x1[indices], x2[indices]))
            manifest = dict(provenance=dict(source_sha256={}), fixtures=[frozen])
            s.verify_bindings(manifest)
            frozen['z'][0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'leaf hash'):
                s.verify_bindings(manifest)
            frozen['z'][0, 0] -= 1
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
                s.bind_fixture(0, root)
            # Restore checkpoint, then modify raw train data without updating its digest.
            torch.save(s.core.weights(model), checkpoint)
            summary['winner']['selected_sha256'] = s.core.digest(checkpoint)
            split['arrays']['xi1'][indices[0], 0] += 1
            s.core.write_json(split_path, split)
            summary['split_artifact_sha256'] = s.core.digest(split_path)
            s.core.write_json(summary_path, summary)
            with self.assertRaisesRegex(ValueError, 'Raw paired-data hash'):
                s.bind_fixture(0, root)

    def test_fd_invalid_capacity_no_one_sided_and_branch_flags(self):
        f = dict(z=np.ones((40, 3)), capacity=np.full((40, 1), 1e-5),
                 upstream=np.ones((40, 3))/np.sqrt(3), target_direction=np.ones((40, 3))/np.sqrt(3),
                 train_indices=list(range(40)))
        record = {}
        # Solver-free control-flow fixture; not an accuracy measurement.
        with patch.object(s, 'reference_qp', return_value=np.zeros(3)) as qp:
            s.references(f, record)
        invalid = [r for r in record['fd'] if r['status'] == 'invalid_capacity']
        self.assertEqual(len(invalid), 9)
        self.assertEqual(qp.call_count, 40+18)
        self.assertEqual(len([r for r in record['fd'] if r['status'] == 'completed']), 9)
        self.assertNotEqual(s.branch(s.active_set(np.zeros(3), 0.)),
                            s.branch(s.active_set(np.zeros(3), 1.)))

    def test_zero_gradient_metrics_no_denominator_substitution(self):
        result = dict(x=np.zeros((1, 3)), grad_z=np.zeros((1, 3)), grad_capacity=np.zeros((1, 1)))
        report = s.metrics(result, result, dict(x=np.zeros((1, 3)), fd=[]),
                           dict(capacity=np.ones((1, 1))))
        np.testing.assert_array_equal(report['target_gradient_normrel'], [0.])
        self.assertEqual(report['target_gradient_cosine'], [None])
        self.assertTrue(report['gradient_diagnostic_pass'])

    def test_failure_continues_independent_config_and_serializes(self):
        f = fixture(); f.update(seed=0, lr_net=.01)
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/'bad.json', Path(tmp)/'good.json'
            with patch.object(s.DirectQP, 'forward', side_effect=RuntimeError('fixture failure')):
                bad = s.run_case(s.CONFIGS[1], f, None, {}, first, 'fixture')
            self.assertEqual(bad['status'], 'failed')
            self.assertEqual(bad['unattempted_passes'], 5)
            self.assertEqual(bad['repeat_statistics_s']['total']['count'], 0)
            good = s.run_case(s.CONFIGS[0], f, None, {}, second, 'fixture')
            self.assertEqual(good['status'], 'completed')
            self.assertEqual(good['repeat_statistics_s']['total']['count'], 5)
            ARTIFACTS.append(dict(test=self.id(), failed=bad, independent=good))


class TimedResult(unittest.TextTestResult):
    def startTest(self, test):
        super().startTest(test)
        self.record = dict(test=test.id(), started_at_utc=s.core.utc_now(),
                           started_monotonic_s=s.time.perf_counter())

    def stopTest(self, test):
        end = s.time.perf_counter()
        self.record.update(ended_at_utc=s.core.utc_now(), ended_monotonic_s=end,
                           wall_time_s=end-self.record['started_monotonic_s'],
                           status='failed' if any(t is test or getattr(t, 'test_case', None) is test
                               for t, _ in self.failures+self.errors) else 'completed')
        ARTIFACTS.append(self.record)
        super().stopTest(test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('.', '..'):
        parser.error('output-name must be a new basename')
    output = s.HERE/args.output_name
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report = dict(scope='small fixtures only; no trained-checkpoint/data benchmark', artifacts=ARTIFACTS,
                  environment=s.provenance(), workers=1, threads=1)
    try:
        with s.phase(report):
            result = unittest.TextTestRunner(verbosity=2, resultclass=TimedResult).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(Preflight))
            report.update(tests_run=result.testsRun, failures=[(str(t), e) for t, e in result.failures],
                          errors=[(str(t), e) for t, e in result.errors], success=result.wasSuccessful())
            if not result.wasSuccessful():
                raise RuntimeError('Preflight failures; see preserved test artifacts')
    finally:
        s.core.write_json(output/'preflight.json', report)


if __name__ == '__main__':
    main()