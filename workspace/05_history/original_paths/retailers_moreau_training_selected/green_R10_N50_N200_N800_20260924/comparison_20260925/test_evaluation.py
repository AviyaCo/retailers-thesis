"""Solver-free unit fixtures only: no experiment, preflight or test generation."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import evaluation as e


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e.initialize_worker()

    def setUp(self):
        self.generator_guard = patch.object(e.c, 'gen', side_effect=AssertionError('generator forbidden'))
        self.generator_guard.start()
        self.addCleanup(self.generator_guard.stop)
        self.env_guard = patch.object(e.c.gp, 'Env', side_effect=AssertionError('real solver forbidden'))
        self.env_guard.start()
        self.addCleanup(self.env_guard.stop)
        self.model_guard = patch.object(e.c.gp, 'Model', side_effect=AssertionError('real solver forbidden'))
        self.model_guard.start()
        self.addCleanup(self.model_guard.stop)

    def test_dimension_ten_targets_and_first_stage_isolation(self):
        e.c.torch.manual_seed(3)
        model = e.c.Model().eval()
        x1 = e.np.arange(30, dtype=float).reshape(3, 10)
        before = e.c.state_hash(model)
        z, q10, q1r = model.targets(x1)
        self.assertEqual(z.shape, (3, 10))
        self.assertEqual(q1r.shape, (10,))
        self.assertEqual(z.dtype, e.np.float32)
        self.assertEqual(q1r.dtype, e.np.float32)
        self.assertEqual(list(inspect.signature(model.targets).parameters), ['xi1'])
        # Later demand can change cost, but is never an input to policy targets.
        q2 = e.np.zeros((3, 10))
        a = e.c.scenario_costs(q10, q1r, q2, x1, e.np.zeros_like(x1))
        b = e.c.scenario_costs(q10, q1r, q2, x1, e.np.full_like(x1, 100))
        self.assertFalse(e.np.array_equal(a, b))
        self.assertTrue(e.np.array_equal(z, model.targets(x1)[0]))
        self.assertEqual(e.c.state_hash(model), before)
        # Exact legacy float32 softplus/features and cost order, not float64 inference.
        with e.c.torch.no_grad():
            features = e.c.torch.cat([e.c.torch.tensor(x1, dtype=e.c.torch.float32),
                e.c.torch.tensor(q1r).expand(3, -1), e.c.torch.full((3, 1), q10)], dim=1)
            self.assertTrue(e.np.array_equal(z, model.fc_layers(features).numpy()))

    def test_generator_dimension_contract_without_generating(self):
        # Inspect the executed generator's constants: never call gen(10000,9999).
        e.c.gen.assert_not_called()
        namespace = dict(e.c.__dict__)
        e.c.load_definitions(namespace, e.c.COMPARE, ('gen',),
                             (('size=(n, 3)', 'size=(n, 10)', 1),))
        first = e.np.ones((2, 10))
        second = e.np.full((2, 10), 2.)
        with patch.object(e.c.np.random, 'seed') as seed, \
                patch.object(e.c.truncnorm, 'rvs', side_effect=[first, second]) as draw:
            got = namespace['gen'](2, 3)  # Mocked distribution only; not a scenario draw.
        self.assertIs(got[0], first)
        self.assertIs(got[1], second)
        seed.assert_called_once_with(3)
        self.assertEqual(draw.call_args_list[0].kwargs['size'], (2, 10))
        self.assertTrue(e.np.array_equal(draw.call_args_list[1].kwargs['loc'], 2 * (6. - first)**2))

    def test_learned_schema_dtype_shape_finiteness_and_legacy_buffers(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'model.pth'
            state = e.c.Model().state_dict()
            e.c.torch.save(state, file)
            model = e.c.load_model(file, 'new')
            legacy = dict(state, **{'proj_layer.q': e.c.torch.zeros(12, 12).to_sparse(),
                                   'proj_layer.A': e.c.torch.zeros(45, 12).to_sparse()})
            e.c.torch.save({'model_state_dict': legacy}, file)
            self.assertEqual(e.c.state_hash(e.c.load_model(file, 'old')), e.c.state_hash(model))
            for key, value in [('Q10', e.c.torch.zeros(1, dtype=e.c.torch.float64)),
                               ('Q1r', e.c.torch.zeros(3)),
                               ('Q10', e.c.torch.full((1,), float('nan')))]:
                bad = dict(state, **{key: value})
                e.c.torch.save(bad, file)
                with self.assertRaisesRegex(ValueError, 'Invalid learned tensor'):
                    e.c.load_model(file, 'new')
            for bad in (dict(state, unexpected=e.c.torch.zeros(1)),
                        {k: v for k, v in state.items() if k != 'Q10'}):
                e.c.torch.save(bad, file)
                with self.assertRaises(ValueError):
                    e.c.load_model(file, 'new')
            legacy['proj_layer.A'] = e.c.torch.zeros(17, 5)
            e.c.torch.save({'model_state_dict': legacy}, file)
            with self.assertRaisesRegex(ValueError, 'projection buffer dimensions'):
                e.c.load_model(file, 'old')
            legacy['extra_buffer'] = e.c.torch.zeros(1)
            e.c.torch.save({'model_state_dict': legacy}, file)
            with self.assertRaisesRegex(ValueError, 'legacy buffer schema'):
                e.c.load_model(file, 'old')

    def test_finite_max_and_group_order(self):
        self.assertEqual(e.finite_pool_cost([1, 9, 4, 2, 3, 6], [2, 1, 3]), (19/3, [9., 4., 6.]))
        a, b = [1.] * 20, [2.] * 20
        self.assertNotEqual(e.pool_digest([[a, b]]), e.pool_digest([[a], [b]]))
        self.assertNotEqual(e.pool_digest([[a, b]]), e.pool_digest([[b, a]]))
        for costs, sizes in [([], []), ([1], [0]), ([1], [2]), ([float('nan')], [1]), ([1], [True])]:
            with self.assertRaises(ValueError):
                e.finite_pool_cost(costs, sizes)

    def test_pool_dimension_membership_and_centers(self):
        center = [[1.] * 20]
        for pools in ([], [[[2.] * 20]], [[[1.] * 20, [1.21] * 20]],
                      [[[1.] * 20, [float('nan')] * 20]], [[[1.] * 6]]):
            with self.assertRaises(ValueError):
                e.validate_pools(pools, center, .2)
        self.assertEqual(len(e.validate_pools([[[1.] * 20, [1.1] * 20]], center, .2)[0]), 2)

    def green_fixture(self, root, selected=1):
        identity = dict(Ntotal=50, Ntrain=1, seed=0, radius=.2)
        training = dict(centers=[[1.] * 20], original_sample_ids=[7])
        candidates = []
        for i in range(3):
            folder = root / str(i)
            folder.mkdir()
            pools = [[[1.] * 20] + [[1. + .01 * k] * 20 for k in range(1, i+1)]]
            digest = e.pool_digest(pools)
            e.write_json(folder / 'pool.json', dict(pools=pools, original_sample_ids=[7], pool_sha256=digest))
            a = dict(identity, candidate_id=i, assessed=True, failed_sample_ids=[],
                checkpoint=str(folder / 'candidate.pth'), checkpoint_sha256=f'file{i}', state_sha256=f'state{i}',
                pool_sha256=digest, pool_file_sha256=e.c.sha(folder / 'pool.json'),
                ap_reference=str(folder / 'assessment.json'), ap_incumbent_mean=[3., 1., 2.][i],
                restricted_mean=1., green_gap=[.1, .02, .005][i],
                samples=[dict(identity, sample_id=0, original_sample_id=7, controller_status='completed',
                    candidate_id=i, candidate_sha256=f'state{i}', checkpoint_sha256=f'file{i}',
                    pool_sha256=digest, center1=[1.] * 10, center2=[1.] * 10)])
            e.write_json(folder / 'assessment.json', a)
            candidates.append(a)
        record = dict(identity, selected_candidate_id=selected, stopping_candidate_id=2, iterations=candidates,
            selected_checkpoint_sha256=f'file{selected}', selected_state_sha256=f'state{selected}',
            selected_ap_reference=candidates[selected]['ap_reference'],
            selected_green_gap=candidates[selected]['green_gap'], stopping_green_gap=.005,
            selection_rule='earliest minimum mean AP incumbent over complete executed prefix including stop')
        return record, training

    def test_own_selected_pool_retains_nonpassing_selected_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, training = self.green_fixture(Path(tmp))
            a, path, pools = e.selected_green_pool(record, training)
            self.assertEqual(a['candidate_id'], 1)
            self.assertEqual(path.parent.name, '1')
            self.assertEqual(len(pools[0]), 2)  # Stopping pool has three rows.
            self.assertGreaterEqual(a['green_gap'], .01)

    def test_entire_prefix_and_ap_references_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, training = self.green_fixture(Path(tmp))
            for mutate in (lambda r: r['iterations'][0]['failed_sample_ids'].append(0),
                           lambda r: r['iterations'].pop(),
                           lambda r: r['iterations'][2]['samples'][0].update(candidate_sha256='bad'),
                           lambda r: r.update(selected_ap_reference='wrong')):
                bad = deepcopy(record)
                mutate(bad)
                with self.assertRaises(ValueError):
                    e.selected_green_pool(bad, training)
            # A later stopping pool or an earlier AP reference must also be bound.
            path = Path(record['iterations'][2]['checkpoint']).parent / 'pool.json'
            original = path.read_text()
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Prefix pool file changed'):
                e.selected_green_pool(record, training)
            path.write_text(original)
            Path(record['iterations'][0]['ap_reference']).unlink()
            with self.assertRaises(FileNotFoundError):
                e.selected_green_pool(record, training)

    def test_no_reselection_and_earliest_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            record, training = self.green_fixture(Path(tmp), selected=2)
            with self.assertRaisesRegex(ValueError, 'earliest minimum AP'):
                e.selected_green_pool(record, training)
            record['iterations'][2]['ap_incumbent_mean'] = 1.
            e.write_json(record['iterations'][2]['ap_reference'], record['iterations'][2])
            with self.assertRaisesRegex(ValueError, 'earliest minimum AP'):
                e.selected_green_pool(record, training)

    def test_derived_assessment_timing_only(self):
        candidate = dict(state_sha256='state', mp_call={'wall_time_s': 2.}, mp_total_time_s=2.)
        assessment = dict(candidate, mp_total_time_s=0.)
        e.validate_assessment(assessment, candidate)
        with self.assertRaises(ValueError):
            e.validate_assessment(dict(assessment, state_sha256='different'), candidate)
        with self.assertRaises(ValueError):
            e.validate_assessment(dict(assessment, mp_total_time_s=1.), candidate)

    def test_final_mp_selector_not_ccg_iters_nonconverged_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real, nominal = root / 'rho_0p400', root / 'nominal'
            real.mkdir()
            path = real / 'seed_0_summary.json'
            summary = dict(final_mp_iter=2, ccg_iters=7, seed=0, rho=.4, completed=True, converged=False)
            e.write_json(path, summary)
            self.assertEqual(e.scs_choice(50, .4, 0, {.4: real}, nominal),
                             (real / 'seed_0_mp_iter2.pth', 2, path))
            e.write_json(path, dict(summary, final_mp_iter=0))
            self.assertEqual(e.scs_choice(50, .4, 0, {.4: real}, nominal)[0], nominal / 'seed_0_nominal.pth')
            with self.assertRaises(ValueError):
                e.scs_choice(50, .4, 0, {}, nominal)  # No R3 special-.4 fallback.
            for field, value in [('final_mp_iter', True), ('final_mp_iter', -1), ('seed', 3), ('completed', False)]:
                e.write_json(path, dict(summary, **{field: value}))
                with self.assertRaises(ValueError):
                    e.scs_choice(50, .4, 0, {.4: real}, nominal)
            e.write_json(path, {k: v for k, v in summary.items() if k != 'final_mp_iter'})
            with self.assertRaises(KeyError):
                e.scs_choice(50, .4, 0, {.4: real}, nominal)

    def test_scs_exact_coordinates_prefix_and_coverage(self):
        centers = [[1.] * 20, [0.] * 20, [2.] * 20]
        points = [[1.2 + 7.15e-6] + [1.] * 19, [-5e-10] + [0.] * 19, [200.] + [2.] * 19]
        rows = [dict(ball_id=i, xi1_hat=v[:10], xi2_hat=v[10:], worst_xi1=p[:10],
                     worst_xi2=p[10:], max_cost=1.) for i, (v, p) in enumerate(zip(centers, points))]
        original = deepcopy(rows)
        reader = MagicMock(return_value=rows)
        pools, report = e.scs_pools(centers, .2, ['AP1', 'AP2'], reader)
        self.assertEqual([call.args[0] for call in reader.call_args_list], ['AP1', 'AP2'])
        self.assertEqual(rows, original)
        self.assertEqual([p.tolist() for p in pools], [[v, p, p] for v, p in zip(centers, points)])
        self.assertFalse(report['box_certified'])
        self.assertTrue(report['coordinates_unchanged'])
        self.assertEqual(report['positive_residual_count'], 6)
        self.assertEqual(report['count_above_reporting_threshold'], 4)
        self.assertEqual(report['max_positive_residual'], 197.8)
        with self.assertRaises(ValueError):
            e.validate_pools(pools, centers, .2)
        for invalid in (rows[:1], [rows[0]] * 3):
            with self.assertRaisesRegex(ValueError, 'coverage failure'):
                e.scs_pools(centers, .2, ['bad'], lambda _: invalid)
        bad = deepcopy(rows)
        bad[0]['worst_xi1'][0] = float('nan')
        with self.assertRaises(ValueError):
            e.scs_pools(centers, .2, ['bad'], lambda _: bad)
        with self.assertRaises(FileNotFoundError):
            e.scs_pools(centers, .2, ['/definitely/missing/AP.json'])

    def test_knn_uses_loo_not_test_and_checks_cached_selection(self):
        a = {'is': [4.] * 10, 'val': [1.] * 10, 'oos': [99.] * 10}
        b = {'is': [3.] * 10, 'val': [1.] * 10, 'oos': [0.] * 10}
        benchmark = dict(_knn_k_selection='per_seed_loo_grid_v3', knn_k_grid=[5, 3],
            saa_knn_by_k={'3': a, '5': b}, saa_knn=dict(deepcopy(a), selected_k_by_seed=[3] * 10))
        self.assertEqual(e.validate_knn(benchmark)['selected_k_by_seed'], [3] * 10)
        bad = deepcopy(benchmark)
        bad['saa_knn']['selected_k_by_seed'][0] = 5
        with self.assertRaisesRegex(ValueError, 'LOO selection mismatch'):
            e.validate_knn(bad)
        bad = deepcopy(benchmark)
        bad['saa_knn']['oos'][0] = 0
        with self.assertRaisesRegex(ValueError, 'cache values mismatch'):
            e.validate_knn(bad)

    def test_snapshot_hash_and_state_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / 'policy.pth'
            e.c.torch.save(e.c.Model().state_dict(), file)
            snap = e._Snapshot(root / 'frozen', dict(files={}))
            with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
                snap.capture(file, 'bad')
            with self.assertRaisesRegex(ValueError, 'Learned state mismatch'):
                snap.model(file, 'green', state='bad')

    def nominal_fixture(self, root, n=800, seed=2):
        state = dict(e.c.Model().state_dict(), **{
            'proj_layer.q': e.c.torch.zeros(12, 12).to_sparse(),
            'proj_layer.A': e.c.torch.zeros(45, 12).to_sparse()})
        payload = dict(model_state_dict=state, seed=seed, n_samples=n,
                       saa_obj=60., saa_Q10=90., saa_Q1r=[8.] * 10,
                       irrelevant_tensor=e.c.torch.ones(1))
        path = root / f'seed_{seed}_nominal.pth'
        e.c.torch.save(payload, path)
        return path, payload

    def test_predetermined_nominal_metadata_mapping(self):
        self.assertEqual(set(e.NOMINAL_METADATA_KINDS), {50, 200, 800})
        actual = {(n, seed) for n, kinds in e.NOMINAL_METADATA_KINDS.items()
                  for seed, kind in enumerate(kinds) if kind == 'checkpoint_metadata'}
        self.assertEqual(actual, {(800, seed) for seed in range(2, 10)})
        for n in (50, 200, 800):
            self.assertEqual(len(e.NOMINAL_METADATA_KINDS[n]), 10)
            for seed in range(10):
                self.assertEqual(e.NOMINAL_METADATA_KINDS[n][seed],
                                 'checkpoint_metadata' if (n, seed) in actual else 'json')

    def test_all_eight_original_checkpoint_metadata_sources_without_generation(self):
        # Read the eight confirmed legacy artifacts; no training, scenarios or solvers.
        nominal = e.c.OLD / 'R10/N800/saainit_net16_v2'
        with tempfile.TemporaryDirectory() as tmp:
            snap = e._Snapshot(Path(tmp) / 'frozen', dict(files={}))
            for seed in range(2, 10):
                with self.subTest(seed=seed):
                    info = e.nominal_metadata(snap, 800, seed, nominal)
                    path = nominal / f'seed_{seed}_nominal.pth'
                    original = e.c.torch.load(path, map_location='cpu', weights_only=False)
                    self.assertEqual(info['source_kind'], 'checkpoint_metadata')
                    self.assertEqual(info['source'], str(path))
                    self.assertEqual(info['source_sha256'], e.c.sha(path))
                    self.assertEqual(info['checkpoint_sha256'], e.c.sha(path))
                    self.assertEqual(info['metadata'], {k: original[k] for k in
                        ('seed', 'n_samples', 'saa_obj', 'saa_Q10', 'saa_Q1r')})
                    self.assertEqual(info['summary_status'], 'missing')
                    self.assertIsNone(info['full_training_time_s'])
                    self.assertIn('unavailable', info['full_training_time_status'])
                    self.assertEqual(info['state_shapes']['fc_layers.0.weight'], [16, 21])
                    self.assertEqual(info['state_shapes']['fc_layers.4.weight'], [10, 16])
                    self.assertEqual(info['verified_n_samples'], 800)
                    self.assertEqual(info['verified_R'], 10)
                    json.dumps(info, allow_nan=False)
            self.assertEqual(len(snap.manifest['files']), 8)
        e.c.gen.assert_not_called()

    def test_json_mapping_never_falls_back_to_existing_checkpoint(self):
        for n, seed in ((50, 2), (200, 2), (800, 0), (800, 1)):
            with self.subTest(n=n, seed=seed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path, payload = self.nominal_fixture(root, n, seed)
                snap = e._Snapshot(root / 'frozen', dict(files={}))
                with self.assertRaises(FileNotFoundError):
                    e.nominal_metadata(snap, n, seed, root)
                summary = root / f'seed_{seed}_summary.json'
                record = {k: payload[k] for k in ('seed', 'saa_obj', 'saa_Q10', 'saa_Q1r')}
                record['total_time_s'] = 123.  # Historical combined time is NOT nominal training time.
                e.write_json(summary, record)
                info = e.nominal_metadata(snap, n, seed, root)
                self.assertEqual(info['source_kind'], 'json')
                self.assertEqual(info['source'], str(summary))
                self.assertEqual(info['metadata'], record)
                self.assertEqual(info['summary_status'], 'present')
                self.assertIsNone(info['full_training_time_s'])
                self.assertEqual(set(snap.manifest['files']), {str(path), str(summary)})

    def test_checkpoint_mapping_missing_expected_or_new_json_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snap = e._Snapshot(root / 'frozen', dict(files={}))
            with self.assertRaises(FileNotFoundError):
                e.nominal_metadata(snap, 800, 2, root)
            self.nominal_fixture(root)
            e.write_json(root / 'seed_2_summary.json', {})
            with self.assertRaisesRegex(ValueError, 'Unexpected nominal JSON'):
                e.nominal_metadata(snap, 800, 2, root)

    def test_nominal_metadata_identity_shapes_primitives_and_binding_fail_closed(self):
        mutations = [lambda p: p.update(seed=3), lambda p: p.update(seed=True),
                     lambda p: p.update(n_samples=200), lambda p: p.pop('saa_obj'),
                     lambda p: p.update(saa_obj=float('nan')),
                     lambda p: p.update(saa_Q1r=[8.] * 3),
                     lambda p: p.update(saa_Q10=e.c.torch.tensor(90.)),
                     lambda p: p['model_state_dict'].update(Q1r=e.c.torch.zeros(3)),
                     lambda p: p['model_state_dict'].update(**{'fc_layers.0.weight': e.c.torch.zeros(16, 7)})]
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path, payload = self.nominal_fixture(root)
                mutate(payload)
                e.c.torch.save(payload, path)
                snap = e._Snapshot(root / 'frozen', dict(files={}))
                with self.assertRaises((ValueError, KeyError)):
                    e.nominal_metadata(snap, 800, 2, root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, payload = self.nominal_fixture(root)
            snap = e._Snapshot(root / 'frozen', dict(files={}))
            info = e.nominal_metadata(snap, 800, 2, root)
            self.assertNotIn('irrelevant_tensor', info['metadata'])
            json.dumps(info, allow_nan=False)
            payload['saa_obj'] += 1
            e.c.torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'Source changed during freeze'):
                e.nominal_metadata(snap, 800, 2, root)

    def test_nominal_json_must_match_bound_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, payload = self.nominal_fixture(root, 50, 0)
            record = {k: payload[k] for k in ('seed', 'saa_obj', 'saa_Q10', 'saa_Q1r')}
            record['saa_obj'] += 1
            e.write_json(root / 'seed_0_summary.json', record)
            snap = e._Snapshot(root / 'frozen', dict(files={}))
            with self.assertRaisesRegex(ValueError, 'JSON/checkpoint metadata mismatch'):
                e.nominal_metadata(snap, 50, 0, root)

    def test_cache_validator_sources_captured_before_freeze_and_no_clipping_adaptations(self):
        required = {e.HERE / 'plot_comparison.py', e.REFERENCE, e.COMPARE, Path(e.__file__).resolve()}
        self.assertTrue(required <= set(e.SOURCE_FILES))
        def check_captured(snap, manifest):
            self.assertTrue({str(p) for p in required} <= set(manifest['files']))
            for path in required:
                self.assertEqual(e.c.sha(path), manifest['files'][str(path)]['sha256'])
                e._bound_file(manifest, path)
            raise ValueError('source-order checked')
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, '_freeze_green', side_effect=check_captured):
            with self.assertRaisesRegex(ValueError, 'source-order checked'):
                e.freeze(Path(tmp))
        for change in e.c.ADAPTATIONS:
            self.assertNotIn('clip', change['after'].lower())
            self.assertNotIn('np.maximum', change['after'])

    def test_frozen_test_arrays_without_n50_cache_or_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = e.c.Model()
            checkpoint, pool_file = root / 'model.pth', root / 'pool.json'
            e.c.torch.save(model.state_dict(), checkpoint)
            pools = [[[1.] * 20]]
            e.write_json(pool_file, dict(pools=pools, original_sample_ids=[0]))
            spec = dict(id='fixture', method='green', checkpoint=str(checkpoint),
                checkpoint_sha256=e.c.sha(checkpoint), state_sha256=e.c.state_hash(model),
                input_files=[str(checkpoint), str(pool_file)], pool_path=str(pool_file),
                pool_file_sha256=e.c.sha(pool_file), pool_sha256=e.pool_digest(pools),
                original_sample_ids=[0], centers=[[1.] * 20], radius=0., Ntrain=1, sizes=[1])
            manifest = dict(status='frozen', versions=e.versions(), membership_tolerance=e.MEMBERSHIP_TOL,
                workers=1, threads=1, jobs=[spec], files={str(p): dict(sha256=e.c.sha(p), copy=str(p))
                    for p in (checkpoint, pool_file)}, n_test=10000, test_seed=9999,
                test=dict(n_test=10000, seed=9999, arrays={}))
            for key in ('xi1', 'xi2'):
                array = e.np.ones((10000, 10), dtype=e.np.float64)  # Unit fixture only.
                file = root / f'{key}.npy'
                e.np.save(file, array)
                manifest['test']['arrays'][key] = dict(file=str(file), file_sha256=e.c.sha(file),
                    array_sha256=e.hashlib.sha256(array.tobytes()).hexdigest(), dtype='float64', shape=[10000, 10])
            path = root / 'manifest.json'
            def seal():
                e.write_json(path, manifest)
                path.with_suffix('.sha256').write_text(e.c.sha(path))
            seal()
            self.assertEqual(e._load_inputs(spec, str(path))[3][0].shape, (10000, 10))
            with self.assertRaisesRegex(ValueError, 'Per-policy identity'):
                e._load_inputs(dict(spec, method='scs'), str(path))
            manifest['test']['arrays']['xi1']['array_sha256'] = 'bad'
            seal()
            with self.assertRaisesRegex(ValueError, 'Unmatched frozen'):
                e._load_inputs(spec, str(path))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest digest'):
                e._load_inputs(spec, str(path))

    def test_worker_failure_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = e.worker(dict(id='missing'), str(Path(tmp) / 'missing.json'), tmp)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['solves_attempted'], 0)
            self.assertIsNone(result['mean_oos'])
            self.assertEqual(result['phases']['input_loading']['status'], 'failed')
            self.assertGreaterEqual(result['wall_time_s'], 0)
            self.assertEqual(e.read_json(Path(tmp) / 'missing/result.json'), result)

    def test_freeze_failure_never_generates_test(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(e, '_freeze_green', side_effect=ValueError('bad artifact')):
            with self.assertRaisesRegex(ValueError, 'bad artifact'):
                e.freeze(Path(tmp))  # Deliberate unit failure before any real input freeze.
            failure = e.read_json(Path(tmp) / 'frozen/freeze_failure.json')
            self.assertEqual(failure['status'], 'failed')
            self.assertFalse(failure['test_generated'])
            self.assertGreaterEqual(failure['wall_time_s'], 0)

    def test_each_optimal_row_journalled_and_only_xi1_passed_to_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = MagicMock()
            model.targets.return_value = (e.np.zeros((2, 10)), 1., e.np.zeros(10))
            record = dict(phases={}, status_counts={}, solves_attempted=0, solves_accepted=0,
                iterations=0., barrier_iterations=0, original_sample_ids=[7], saved_native_finite_pool_mean=0.)
            def solve(z, q10, env, diag):
                diag.update(status=2, accepted=True, iterations=1., barrier_iterations=0)
                return e.np.zeros(10)
            x1, x2 = e.np.zeros((2, 10)), e.np.full((2, 10), 5.)
            with patch.object(e.c.gp, 'Env'), patch.object(e.c.gp, 'Model') as settings, \
                    patch.object(e.c, 'strict_project', side_effect=solve) as project:
                settings.return_value.__enter__.return_value.getParamInfo.return_value = (None, None, 1)
                e._evaluate(model, x1, x2, 'is', record, root, [2])
            self.assertIs(model.targets.call_args.args[0], x1)
            self.assertEqual(len(model.targets.call_args.args), 1)
            self.assertEqual(project.call_count, 2)
            self.assertEqual(record['status_counts'], {'2': 2})
            self.assertEqual(record['solves_accepted'], 2)
            rows = [json.loads(s) for s in (root / 'is_solves.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r['accepted'] and r['original_sample_id'] == 7 for r in rows))

    def test_projection_failure_journalled_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = MagicMock()
            model.targets.return_value = (e.np.zeros((2, 10)), 1., e.np.zeros(10))
            record = dict(phases={}, status_counts={}, solves_attempted=0, solves_accepted=0,
                          iterations=0., barrier_iterations=0)
            def fail(z, q, env, diag):
                diag.update(status=9, iterations=4., barrier_iterations=1)
                raise RuntimeError('Projection requires OPTIMAL, got 9')
            with patch.object(e.c.gp, 'Env'), patch.object(e.c.gp, 'Model') as settings, \
                    patch.object(e.c, 'strict_project', side_effect=fail) as solve:
                settings.return_value.__enter__.return_value.getParamInfo.return_value = (None, None, 1)
                with self.assertRaisesRegex(RuntimeError, 'OPTIMAL'):
                    e._evaluate(model, e.np.zeros((2, 10)), e.np.zeros((2, 10)), 'is', record, Path(tmp))
            solve.assert_called_once()
            self.assertEqual(record['solves_attempted'], 1)
            self.assertEqual(record['solves_accepted'], 0)
            self.assertEqual(record['status_counts'], {'9': 1})
            self.assertEqual(record['phases']['is_projection']['status'], 'failed')
            row = json.loads((Path(tmp) / 'is_solves.jsonl').read_text())
            self.assertFalse(row['accepted'])
            self.assertIn('controller_wall_time_s', row)

    def test_strict_project_never_clips_and_rejects_nonoptimal(self):
        # Fake Gurobi arithmetic only; exercise the real strict acceptance logic.
        result = [-1e-8] + [.1] * 9
        variables = {i: MagicMock(X=v) for i, v in enumerate(result)}
        model = MagicMock(Status=2, IterCount=0., BarIterCount=0, ObjVal=1.)
        model.addVars.return_value = variables
        expression = MagicMock()
        expression.__le__.return_value = True
        with patch.object(e.c.gp, 'Model') as factory, patch.object(e.c.gp, 'quicksum', return_value=expression):
            factory.return_value.__enter__.return_value = model
            diag = {}
            actual = e.c.strict_project(e.np.zeros(10), 1., None, diag)
            self.assertTrue(e.np.array_equal(actual, result))
            self.assertLess(actual[0], 0)  # Tolerance does not authorize clipping.
            self.assertTrue(diag['accepted'])
            model.Status = 9
            with self.assertRaisesRegex(RuntimeError, 'OPTIMAL'):
                e.c.strict_project(e.np.zeros(10), 1., None, {})
            model.Status = 2
            variables[0].X = -.1
            with self.assertRaisesRegex(RuntimeError, 'Physical residual'):
                e.c.strict_project(e.np.zeros(10), 1., None, {})


if __name__ == '__main__':
    unittest.main()