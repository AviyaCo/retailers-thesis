"""Solver-free fixtures: never call the scenario generator or a real solver."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import evaluation as e


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e.initialize_worker()

    def test_unbalanced_mean_of_maxima(self):
        mean, maxima = e.finite_pool_cost([1, 9, 4, 2, 3, 6], [2, 1, 3])
        self.assertEqual(maxima, [9, 4, 6])
        self.assertEqual(mean, 19 / 3)
        self.assertNotEqual(mean, 25 / 6)
        self.assertEqual(e.finite_pool_cost([1, 3], [1, 1]), (2., [1., 3.]))

    def test_invalid_pool_costs(self):
        for costs, sizes in [([], []), ([1], [0]), ([1], [2]), ([float('nan')], [1]), ([1], [True])]:
            with self.assertRaises(ValueError):
                e.finite_pool_cost(costs, sizes)

    def test_pool_grouping_is_hashed(self):
        a = [[1.] * 6, [2.] * 6]
        self.assertNotEqual(e.pool_digest([a]), e.pool_digest([[a[0]], [a[1]]]))
        self.assertNotEqual(e.pool_digest([a]), e.pool_digest([a[::-1]]))

    def test_membership_and_center_count(self):
        center = [[1.] * 6]
        for pools in ([], [[[2.] * 6]], [[[1.] * 6, [1.21] * 6]], [[[1.] * 6, [float('nan')] * 6]]):
            with self.assertRaises(ValueError):
                e.validate_pools(pools, center, .2)
        self.assertEqual(len(e.validate_pools([[[1.] * 6, [1.1] * 6]], center, .2)[0]), 2)

    def test_selected_pool_not_stopping_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            candidates = []
            ids = [7]
            for i in range(3):
                sub = folder / str(i)
                sub.mkdir()
                pools = [[[1.] * 6] + [[1.01 * k] * 6 for k in range(1, i + 1)]]
                digest = e.pool_digest(pools)
                e.write_json(sub / 'pool.json', dict(pools=pools, original_sample_ids=ids, pool_sha256=digest))
                candidates.append(dict(candidate_id=i, assessed=True, failed_sample_ids=[],
                    samples=[dict(sample_id=0, original_sample_id=7, controller_status='completed',
                                  candidate_sha256=f'state{i}', pool_sha256=digest)],
                    checkpoint=str(sub / 'candidate.pth'), checkpoint_sha256=f'file{i}',
                    state_sha256=f'state{i}', pool_file_sha256=e.c.sha(sub / 'pool.json'), pool_sha256=digest))
            for selected, stop in [(1, 2), (0, 1)]:
                record = dict(selected_candidate_id=selected, stopping_candidate_id=stop,
                    iterations=candidates[:stop + 1], Ntrain=1, radius=.2,
                    selected_checkpoint_sha256=f'file{selected}', selected_state_sha256=f'state{selected}')
                _, path, pools = e.selected_green_pool(record, dict(original_sample_ids=ids, centers=[[1.] * 6]))
                self.assertEqual(path.parent.name, str(selected))
                self.assertEqual(len(pools[0]), selected + 1)
                record['iterations'][0]['failed_sample_ids'] = [0]
                with self.assertRaisesRegex(ValueError, 'Incomplete green'):
                    e.selected_green_pool(record, dict(original_sample_ids=ids, centers=[[1.] * 6]))
                record['iterations'][0]['failed_sample_ids'] = []

    def test_scs_real_directory_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real, nominal = root / 'rho_0p400', root / 'nominal'
            real.mkdir()
            path = real / 'seed_0_summary.json'
            e.write_json(path, dict(final_mp_iter=2, seed=0, rho=.4, completed=True))
            chosen, iteration, selector = e.scs_choice(50, .4, 0, {.4: real}, nominal)
            self.assertEqual((chosen, iteration, selector), (real / 'seed_0_mp_iter2.pth', 2, path))
            special, iteration, selector = e.scs_choice(50, .4, 0, {}, nominal)
            self.assertEqual((special, iteration, selector), (nominal / 'seed_0_mp_iter3.pth', 3, None))
            e.write_json(path, dict(final_mp_iter=0, seed=0, rho=.4, completed=True))
            self.assertEqual(e.scs_choice(50, .4, 0, {.4: real}, nominal)[0], nominal / 'seed_0_nominal.pth')
            e.write_json(path, dict(seed=0, rho=.4, completed=True))
            with self.assertRaisesRegex(ValueError, 'final_mp_iter'):
                e.scs_choice(50, .4, 0, {.4: real}, nominal)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                e.scs_choice(50, .4, 0, {.4: real}, nominal)
            with self.assertRaises(ValueError):
                e.scs_choice(200, .4, 0, {}, nominal)

    def test_derived_assessment_timing_only(self):
        candidate = dict(state_sha256='state', mp_call={'wall_time_s': 2.}, mp_total_time_s=2.)
        assessment = dict(candidate, mp_total_time_s=0.)
        e.validate_assessment(assessment, candidate)
        with self.assertRaisesRegex(ValueError, 'state_sha256'):
            e.validate_assessment(dict(assessment, state_sha256='different'), candidate)
        with self.assertRaisesRegex(ValueError, 'Unexplained'):
            e.validate_assessment(dict(assessment, mp_total_time_s=1.), candidate)

    def test_scs_prefix_and_full_coverage(self):
        centers = [[1.] * 6, [2.] * 6]
        rows = [dict(ball_id=i, xi1_hat=v[:3], xi2_hat=v[3:], worst_xi1=v[:3],
                     worst_xi2=v[3:], max_cost=1.) for i, v in enumerate(centers)]
        reader = MagicMock(return_value=rows)
        pools, membership = e.scs_pools(centers, .2, ['AP1', 'AP2'], reader)
        self.assertEqual([len(p) for p in pools], [3, 3])
        self.assertEqual(membership['saved_points_checked'], 4)
        self.assertEqual(membership['positive_residual_count'], 0)
        self.assertEqual(membership['max_positive_residual'], 0.)
        self.assertEqual([call.args[0] for call in reader.call_args_list], ['AP1', 'AP2'])
        for invalid in (rows[:1], [rows[0], rows[0]]):
            with self.assertRaisesRegex(ValueError, 'coverage failure'):
                e.scs_pools(centers, .2, ['bad'], lambda _: invalid)
        for field, value in [('worst_xi1', [float('nan')] * 3),
                             ('xi1_hat', [9.] * 3), ('max_cost', float('inf'))]:
            invalid = deepcopy(rows)
            invalid[0][field] = value
            with self.assertRaises(ValueError):
                e.scs_pools(centers, .2, ['bad'], lambda _: invalid)
        with self.assertRaises(FileNotFoundError):
            e.scs_pools(centers, .2, ['/definitely/missing/AP.json'])

    def test_historical_scs_retains_and_discloses_strict_green_offenders(self):
        centers = [[1.] * 6, [0.] * 6, [2.] * 6]
        points = [[1.2 + 7.15e-6] + [1.] * 5, [-5e-10] + [0.] * 5,
                  [200.] + [2.] * 5]  # No hidden numerical acceptance ceiling.
        rows = [dict(ball_id=i, xi1_hat=v[:3], xi2_hat=v[3:], worst_xi1=p[:3],
                     worst_xi2=p[3:], max_cost=1.) for i, (v, p) in enumerate(zip(centers, points))]
        original = deepcopy(rows)
        pools, report = e.scs_pools(centers, .2, ['saved/AP1.json'], lambda _: rows)
        self.assertEqual(rows, original)
        self.assertEqual([p.tolist() for p in pools], [[v, p] for v, p in zip(centers, points)])
        with self.assertRaisesRegex(ValueError, 'Pool membership failure center=0'):
            e.validate_pools(pools, centers, .2)
        self.assertEqual(report['policy'], e.SCS_MEMBERSHIP_POLICY)
        self.assertFalse(report['box_certified'])
        self.assertTrue(report['coordinates_unchanged'])
        self.assertEqual(report['reporting_threshold'], 1e-9)
        self.assertEqual(report['saved_points_checked'], 3)
        self.assertEqual(report['positive_residual_count'], 3)
        self.assertEqual(report['count_above_reporting_threshold'], 2)
        self.assertEqual(report['max_positive_residual'], 197.8)
        self.assertAlmostEqual(report['offenders'][0]['positive_residual'], 7.15e-6)
        self.assertEqual(report['offenders'][1]['positive_residual'], 5e-10)
        self.assertEqual([(o['path'], o['ball_id'], o['pool_row']) for o in report['offenders']],
                         [('saved/AP1.json', i, 1) for i in range(3)])
        with self.assertRaisesRegex(ValueError, 'Unknown membership policy'):
            e.validate_pools(pools, centers, .2, membership_policy='relaxed')

    def test_unknown_buffers_not_stripped(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'policy.pth'
            state = e.c.Model().state_dict()
            e.c.torch.save(state, file)
            e.c.load_model(file, 'new')
            state['unexpected'] = e.c.torch.zeros(1)
            e.c.torch.save(state, file)
            with self.assertRaisesRegex(ValueError, 'Unexpected learned schema'):
                e.c.load_model(file, 'new')
            e.c.torch.save({'model_state_dict': state}, file)
            with self.assertRaisesRegex(ValueError, 'Unexpected legacy buffer schema'):
                e.c.load_model(file, 'old')

    def test_snapshot_wrong_hash_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / 'policy.pth'
            e.c.torch.save(e.c.Model().state_dict(), file)
            snap = e._Snapshot(root / 'frozen', dict(files={}))
            with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
                snap.capture(file, 'bad')
            with self.assertRaisesRegex(ValueError, 'Learned state mismatch'):
                snap.model(file, 'green', state='bad')
            with self.assertRaises(FileNotFoundError):
                snap.capture(root / 'missing')

    def test_identity_and_manifest_tampering_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'manifest.json'
            spec = dict(id='one', method='green')
            manifest = dict(status='frozen', versions=e.versions(), membership_tolerance=e.MEMBERSHIP_TOL,
                            threads=1, workers=1, jobs=[spec])
            e.write_json(file, manifest)
            file.with_suffix('.sha256').write_text(e.c.sha(file))
            with self.assertRaisesRegex(ValueError, 'Per-policy identity'):
                e._load_inputs(dict(spec, method='scs'), str(file))
            file.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest digest'):
                e._load_inputs(spec, str(file))

    def test_worker_missing_input_persists_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = e.worker(dict(id='missing'), str(Path(tmp) / 'missing.json'), tmp)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['solves_attempted'], 0)
            self.assertEqual(result['phases']['input_loading']['status'], 'failed')
            self.assertGreaterEqual(result['wall_time_s'], 0)
            self.assertEqual(e.read_json(Path(tmp) / 'missing/result.json'), result)

    def test_test_array_identity_verified_without_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = e.c.Model()
            checkpoint = root / 'model.pth'
            e.c.torch.save(model.state_dict(), checkpoint)
            pools = [[[1.] * 6]]
            pool_file = root / 'pool.json'
            e.write_json(pool_file, dict(pools=pools, original_sample_ids=[0]))
            spec = dict(id='fixture', method='green', checkpoint=str(checkpoint),
                checkpoint_sha256=e.c.sha(checkpoint), state_sha256=e.c.state_hash(model),
                input_files=[str(checkpoint), str(pool_file)], pool_path=str(pool_file),
                pool_file_sha256=e.c.sha(pool_file), pool_sha256=e.pool_digest(pools),
                original_sample_ids=[0], centers=[[1.] * 6], radius=0., Ntrain=1, sizes=[1])
            manifest = dict(status='frozen', versions=e.versions(), membership_tolerance=e.MEMBERSHIP_TOL,
                workers=1, threads=1, jobs=[spec], files={str(p): dict(sha256=e.c.sha(p), copy=str(p))
                    for p in (checkpoint, pool_file)}, n_test=10000, test_seed=9999,
                test=dict(n_test=10000, seed=9999, arrays={}), cached_green_N50=dict(test_arrays={}))
            for key in ('xi1', 'xi2'):
                array = e.np.ones((10000, 3), dtype=e.np.float64)  # Unit fixture, not experimental data.
                file = root / f'{key}.npy'
                e.np.save(file, array)
                info = dict(file=str(file), file_sha256=e.c.sha(file),
                    array_sha256=e.hashlib.sha256(array.tobytes()).hexdigest(), dtype='float64', shape=[10000, 3])
                manifest['test']['arrays'][key] = info
                manifest['cached_green_N50']['test_arrays'][key] = dict(info)
            path = root / 'manifest.json'
            def seal():
                e.write_json(path, manifest)
                path.with_suffix('.sha256').write_text(e.c.sha(path))
            seal()
            with patch.object(e.c, 'gen', side_effect=AssertionError('generator forbidden')):
                self.assertEqual(e._load_inputs(spec, str(path))[3][0].shape, (10000, 3))
                manifest['test']['arrays']['xi1']['array_sha256'] = 'wrong'
                seal()
                with self.assertRaisesRegex(ValueError, 'Unmatched historical'):
                    e._load_inputs(spec, str(path))
                manifest['test']['arrays']['xi1']['file_sha256'] = 'wrong'
                seal()
                with self.assertRaisesRegex(ValueError, 'Test file changed'):
                    e._load_inputs(spec, str(path))

    def test_freeze_failure_is_persisted_and_never_generates_test(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(e, '_freeze_green', side_effect=ValueError('bad artifact')), \
                patch.object(e.c, 'gen', side_effect=AssertionError('generator forbidden')) as generator:
            with self.assertRaisesRegex(ValueError, 'bad artifact'):
                e.freeze(Path(tmp))
            generator.assert_not_called()
            failure = e.read_json(Path(tmp) / 'frozen/freeze_failure.json')
            self.assertEqual(failure['status'], 'failed')
            self.assertFalse(failure['test_generated'])
            self.assertGreaterEqual(failure['wall_time_s'], 0)

    def test_strict_failure_journalled_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = MagicMock()
            model.targets.return_value = (e.np.zeros((2, 3)), 1., e.np.zeros(3))
            record = dict(phases={}, status_counts={}, solves_attempted=0, solves_accepted=0,
                          iterations=0., barrier_iterations=0)
            def fail(z, q, env, diagnostic):
                diagnostic.update(status=9, iterations=4., barrier_iterations=1)
                raise RuntimeError('Projection requires OPTIMAL, got 9')
            with patch.object(e.c.gp, 'Env'), patch.object(e.c.gp, 'Model') as setting, \
                    patch.object(e.c, 'strict_project', side_effect=fail) as solve:
                setting.return_value.__enter__.return_value.getParamInfo.return_value = (None, None, 1)
                with self.assertRaisesRegex(RuntimeError, 'OPTIMAL'):
                    e._evaluate(model, e.np.zeros((2, 3)), e.np.zeros((2, 3)), 'is', record, Path(tmp))
            solve.assert_called_once()
            self.assertEqual(record['solves_attempted'], 1)
            self.assertEqual(record['solves_accepted'], 0)
            self.assertEqual(record['status_counts'], {'9': 1})
            self.assertEqual(record['iterations'], 4.)
            self.assertEqual(record['phases']['is_projection']['status'], 'failed')
            rows = [json.loads(line) for line in (Path(tmp) / 'is_solves.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]['accepted'])
            self.assertIn('end_utc', rows[0])


if __name__ == '__main__':
    unittest.main()