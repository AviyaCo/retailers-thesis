"""Bounded synthetic/mocked tests only; never collect, launch, or solve real QPs."""
from copy import deepcopy
from concurrent.futures import Future
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run = load('_stop_run_tests', 'run.py')
launch = load('_stop_launch_tests', 'launch.py')


class SealsAndGates(unittest.TestCase):
    def test_seal_is_exclusive_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'value.json'
            digest = run.seal(p, {'v': 3})
            self.assertEqual(run.sealed(p, digest), {'v': 3})
            with self.assertRaises(FileExistsError):
                run.seal(p, {'v': 3})
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                run.sealed(p, '0'*64)
            p.chmod(0o644)
            p.write_text('{"v":4}')
            with self.assertRaisesRegex(ValueError, 'Seal'):
                run.sealed(p)

    def test_failed_phase_has_monotonic_timing(self):
        rows = {}
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with run.phase(rows, 'failed'):
                raise RuntimeError('deliberate')
        self.assertEqual(rows['failed']['status'], 'failed')
        run.timing(rows['failed'])

    def test_nonfinite_and_negative_durations_rejected(self):
        for value in (float('nan'), float('inf'), -1, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run.seconds(value)

    def test_resource_exact_reserves_and_no_degradation(self):
        snapshot = dict(affinity=list(range(24)), cpu_quota_cores=None,
                        load_average=[0., 0., 0.], memory_available_bytes=114*run.GIB)
        gate = run.resource_gate(snapshot, 4*run.GIB)
        self.assertEqual(gate['workers'], 16)
        self.assertEqual(gate['reserved_cpu_slots'], 8)
        self.assertEqual(gate['reserved_memory_bytes'], 50*run.GIB)
        for field, value in (('cpu_quota_cores', 23.9), ('load_average', [.1, 0., 0.]),
                             ('memory_available_bytes', 114*run.GIB-1), ('affinity', list(range(23)))):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'no automatic'):
                run.resource_gate(dict(snapshot, **{field: value}))
        with self.assertRaises(ValueError):
            run.resource_gate(snapshot, 4*run.GIB+1)

    def test_all_local_files_and_shared_runtime_bound(self):
        sources = run.local_sources()
        for filename in ('run.py', 'test_run.py', 'launch.py', 'README.md',
                         'selection.py', 'report.py', 'test_selection.py', 'test_report.py'):
            self.assertIn(str(run.HERE / filename), sources)
        for filename in ('src/evaluate.py', 'src/artifacts.py', 'src/__init__.py', 'archive_paths.py'):
            self.assertIn(str(run.ROOT / filename), sources)
        run.verify_sources(sources)
        sources[str(run.HERE / 'run.py')] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source changed'):
            run.verify_sources(sources)

    def test_added_local_source_is_not_ignored(self):
        expected = {str(run.HERE / 'run.py'): 'a'*64}
        with patch.object(run, 'local_sources', return_value=dict(expected, **{str(run.HERE / 'extra.py'): 'b'*64})):
            with self.assertRaisesRegex(ValueError, 'source set'):
                run.verify_sources(expected)

    def test_conflicting_source_binding_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            run.merge_sources({'p': 'a'}, {'p': 'b'})

    def test_selection_receipt_required_even_when_manifest_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'manifest.json'
            digest = run.seal(p, {})
            with self.assertRaises(FileNotFoundError):
                run.selection_bundle(p, digest, Path(tmp) / 'absent.json', '0'*64)

    def test_failed_selection_receipt_blocks_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, receipt = Path(tmp) / 'manifest.json', Path(tmp) / 'receipt.json'
            m = {'verification_receipt': str(receipt), 'input_sha256': {}}
            digest = run.seal(p, m)
            rd = run.seal(receipt, dict(selection_manifest_sha256=digest, status='failed'))
            with patch.object(run, 'cohort'), self.assertRaisesRegex(ValueError, 'did not complete'):
                run.selection_bundle(p, digest, receipt, rd)

    def test_missing_probes_block(self):
        self.assertEqual(len(run.PROBES), 12)
        self.assertEqual({(p[1], p[2]) for p in run.PROBES},
                         {(r, n) for r in (3, 10) for n in (50, 200, 800)})
        self.assertEqual({p[0] for p in run.PROBES}, {'green', 'black'})
        with self.assertRaisesRegex(ValueError, 'Missing parity'):
            run.validate_probes([], [])

    def test_fresh_directory_never_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                run.fresh_directory(tmp)
        with self.assertRaisesRegex(ValueError, 'archive'):
            run.fresh_directory(run.ARCHIVE / 'forbidden-test-output')

    def test_cli_requires_production_binding(self):
        with patch.object(run, 'production') as production:
            with self.assertRaisesRegex(ValueError, 'requires'):
                run.main(['production', '/unused'])
            production.assert_not_called()

    def test_unit_command_uses_current_interpreter_and_all_local_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            def fake_call(command, **kwargs):
                self.assertEqual(command[:3], [sys.executable, '-B', '-u'])
                self.assertEqual(command[4], '_unit')
                self.assertEqual(kwargs['cwd'], run.HERE)
                run.seal(output / 'unit_evidence.json', {'status': 'completed'})
                return SimpleNamespace(returncode=0)
            with patch.object(run.subprocess, 'run', side_effect=fake_call):
                evidence = run.run_units(output)
            self.assertEqual(evidence['returncode'], 0)
            self.assertEqual(evidence['sha256'], run.sha(output / 'unit_evidence.json'))


class NumericalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = run.numerical()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.e.torch.manual_seed(20260928)
        self.model = self.e.Model(3).eval()
        self.x1 = self.e.np.array([[1., 2., 3.], [4., 5., 6.]], dtype=self.e.np.float64)
        self.x2 = self.x1 + 1.

    def fixture(self, folder, model=None, x1=None, x2=None, sizes=None, record=None):
        """Synthetic journal: zero shipments, NOT a solved policy or experiment."""
        model = self.model if model is None else model
        x1 = self.x1 if x1 is None else x1
        x2 = self.x2 if x2 is None else x2
        folder.mkdir(parents=True, exist_ok=True)
        e = self.e
        z, q10, q1r = e.targets(model, x1)
        shipments = e.np.zeros(x1.shape, dtype=e.np.float64)
        costs = e.scenario_costs(q10, q1r, shipments, x1, x2)
        for name, a in (('targets', z), ('shipments', shipments), ('costs', costs)):
            e.np.save(folder / (name + '.npy'), a, allow_pickle=False)
        n, r = x1.shape
        raw = dict(record or {}, status='completed', cached=False, sample_count=n, R=r,
            state_sha256=e.state_hash(model), solver_threads=1, torch_threads=1, worker_count=16,
            physical_tolerance=1e-4, solver_settings={'Threads': 1}, solves_attempted=n,
            solves_accepted=n, solves_physical_accepted=n, status_counts={'2': n},
            timing_clock='time.perf_counter', timing_unit='seconds', wall_time_s=1.,
            started_at_utc=run.utc(), ended_at_utc=run.utc(),
            phases={k: .1 for k in ('input_validation', 'target_evaluation', 'projection_qp', 'cost_evaluation')},
            q10=q10, q1r=q1r.tolist(), inventory_dtype='float32', iterations=float(n), barrier_iterations=n,
            role='oos' if sizes is None else 'own_pool')
        if sizes is None:
            raw.update(mean=float(costs.mean()), mean_oos=float(costs.mean()))
        else:
            mean, maxima = e.finite_pool_cost(costs, sizes)
            raw.update(mean=mean, finite_pool_mean=mean, per_center_maxima=maxima, sizes=sizes)
        (folder / 'result.json').write_text(json.dumps(raw))
        diagnostics = [dict(row=i, status=2, accepted=True, physical_residual=0., wall_time_s=.01,
            iterations=1., barrier_iterations=1, solver_runtime_s=.005,
            objective=float(sum(float(t)**2 for t in z[i])), shipments=shipments[i].tolist()) for i in range(n)]
        (folder / 'solves.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in diagnostics))
        return raw

    def mutate_journal(self, folder, mutation):
        path = folder / 'solves.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutation(rows)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_verified_synthetic_replay_counts_all_physical_acceptances(self):
        folder = self.path / 'oos'
        self.fixture(folder)
        raw, proofs = run.verify_replay(folder, self.model, self.x1, self.x2)
        self.assertEqual(raw['solves_physical_accepted'], 2)
        self.assertEqual(set(proofs), {'result.json', 'targets.npy', 'shipments.npy', 'costs.npy', 'solves.jsonl'})

    def test_every_raw_status_residual_ordinal_timing_checked(self):
        mutations = [('status', 9), ('accepted', False), ('physical_residual', 1.001e-4),
                     ('physical_residual', float('nan')), ('row', 1), ('iterations', -1),
                     ('wall_time_s', float('inf')), ('solver_runtime_s', -1), ('objective', float('nan'))]
        for i, (key, value) in enumerate(mutations):
            folder = self.path / str(i)
            self.fixture(folder)
            self.mutate_journal(folder, lambda rows: rows[0].update({key: value}))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                run.verify_replay(folder, self.model, self.x1, self.x2)

    def test_missing_and_extra_journal_rows_rejected(self):
        for i, operation in enumerate((lambda rows: rows.pop(), lambda rows: rows.append(dict(rows[-1], row=2)))):
            folder = self.path / str(i)
            self.fixture(folder)
            self.mutate_journal(folder, operation)
            with self.subTest(i=i), self.assertRaises(ValueError):
                run.verify_replay(folder, self.model, self.x1, self.x2)

    def test_arrays_targets_shipments_and_costs_verified(self):
        for name in ('targets', 'shipments', 'costs'):
            folder = self.path / name
            self.fixture(folder)
            a = self.e.np.load(folder / (name+'.npy'))
            a.flat[0] += 1.
            self.e.np.save(folder / (name+'.npy'), a)
            with self.subTest(name=name), self.assertRaises(ValueError):
                run.verify_replay(folder, self.model, self.x1, self.x2)

    def test_raw_counts_inventory_and_mean_verified(self):
        for i, (key, value) in enumerate((('solves_physical_accepted', 1), ('q10', 123.),
                ('worker_count', 8), ('mean_oos', -999.), ('iterations', 9.))):
            folder = self.path / str(i)
            raw = self.fixture(folder)
            raw[key] = value
            (folder / 'result.json').write_text(json.dumps(raw))
            with self.subTest(key=key), self.assertRaises(ValueError):
                run.verify_replay(folder, self.model, self.x1, self.x2)

    def test_own_pool_maxima_recomputed(self):
        folder = self.path / 'pool'
        raw = self.fixture(folder, sizes=[2])
        run.verify_replay(folder, self.model, self.x1, self.x2, [2])
        raw['per_center_maxima'][0] += 1.
        (folder / 'result.json').write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'Own-pool'):
            run.verify_replay(folder, self.model, self.x1, self.x2, [2])

    def test_ast_reference_r3_r10_targets_exact_on_synthetic_full_batches(self):
        # No solver calls, archived policy reads, or test data. Only definitions and synthetic tensors.
        for r in (3, 10):
            ref, changes = run.reference(r)
            self.assertEqual({k for k in ref if k in ('Model', 'load_model', 'state_hash', 'scenario_costs', 'strict_project')},
                             {'Model', 'load_model', 'state_hash', 'scenario_costs', 'strict_project'})
            self.assertNotIn('gen', ref)
            self.assertEqual(len(changes), 0 if r == 3 else 8)
            m = self.e.Model(r).eval()
            old = ref['Model']().eval()
            old.load_state_dict(m.state_dict())
            x = self.e.np.arange(40*r, dtype=self.e.np.float64).reshape(40, r)/100
            z, q10, q1r = self.e.targets(m, x)
            oz, oq10, oq1r = old.targets(x)
            self.assertTrue(self.e.np.array_equal(z, oz))
            self.assertEqual(q10, oq10)
            self.assertTrue(self.e.np.array_equal(q1r, oq1r))
            self.assertEqual(self.e.state_hash(m), ref['state_hash'](old))

    def test_ast_reference_lock_failure_stops(self):
        with patch.object(run, 'COMPARE_SHA', '0'*64), self.assertRaisesRegex(ValueError, 'lock'):
            run.reference(3)

    def test_failed_real_probe_retained_without_solver_or_test_access(self):
        job = dict(id='synthetic_probe', method='black', R=3, Ntotal=50, Ntrain=40,
            radius=.4, seed=0, checkpoint_sha256='a'*64, state_sha256='b'*64, selected_candidate_id=4)
        with patch.object(run, 'own_inputs', side_effect=RuntimeError('synthetic failure')):
            row = run.probe(job, None, self.path)
        self.assertEqual(row['status'], 'failed')
        self.assertFalse(row['test_evaluated'])
        self.assertEqual(row['phases']['input_loading']['status'], 'failed')
        run.timing(row)
        self.assertEqual(run.sealed(row['path'], row['sha256'])['status'], 'failed')

    def test_successful_r10_probe_survives_json_roundtrip(self):
        e = self.e
        model = e.Model(10).eval()
        checkpoint = self.path / 'synthetic.pth'
        e.torch.save(model.state_dict(), checkpoint)
        job = dict(id='green_synthetic_r10', method='green', R=10, Ntotal=50, Ntrain=40,
            radius=.4, seed=0, checkpoint=str(checkpoint), checkpoint_sha256=run.sha(checkpoint),
            state_sha256=e.state_hash(model), selected_candidate_id=4)
        ref, changes = run.reference(10)
        def fake_projection(z, q10, env, diagnostic):
            diagnostic.update(status=2, accepted=True, physical_residual=0., wall_time_s=.01,
                              iterations=1., barrier_iterations=1)
            return e.np.zeros(len(z), dtype=e.np.float64)
        ref['strict_project'] = fake_projection
        centers = e.np.ones((40, 20), dtype=e.np.float64)
        import gurobipy as gp
        with patch.object(run, 'own_inputs', return_value=(model, centers, [])), \
                patch.object(run, 'reference', return_value=(ref, changes)), \
                patch.object(e, 'strict_project', side_effect=fake_projection), \
                patch.object(gp, 'Env'), patch.object(run, 'PROBES', (('green', 10, 50, 0, .4),)):
            row = run.probe(job, None, self.path)
            self.assertEqual(row['status'], 'completed')
            run.validate_probes([row], [job])
        self.assertEqual(len(row['adaptations']), 8)
        self.assertEqual(row['full_batch_size'], 40)

    def test_historical_version_difference_not_ignored(self):
        m = {'source_old_manifest': {'path': '/unused', 'sha256': 'a'*64}}
        old = {'base_comparisons': {str(r): {'versions': {}} for r in (3, 10)}}
        with patch.object(run, 'sealed', return_value=old), self.assertRaisesRegex(ValueError, 'differs'):
            run.runtime_environment(m)

    def test_test_arrays_reject_wrong_protocol_before_loading(self):
        # Small fixture deliberately fails the fixed 10k protocol before any file access.
        m = {'tests': {'3': {'seed': 1, 'n_test': 2}}}
        with patch.object(self.e.np, 'load') as loader:
            with self.assertRaisesRegex(ValueError, 'protocol'):
                run.test_arrays(m, 3)
            loader.assert_not_called()

    def test_synthetic_test_arrays_bind_file_and_content_hashes(self):
        arrays = {}
        for name in ('xi1', 'xi2'):
            p = self.path / (name+'.npy')
            a = self.e.np.zeros((10000, 3), dtype=self.e.np.float64)
            self.e.np.save(p, a, allow_pickle=False)
            arrays[name] = dict(file=str(p), file_sha256=run.sha(p), shape=[10000, 3], dtype='float64',
                array_sha256=run.hashlib.sha256(a.tobytes()).hexdigest())
        m = {'tests': {'3': dict(seed=9999, n_test=10000, arrays=arrays)}}
        a, b = run.test_arrays(m, 3)
        self.assertFalse(a.flags.writeable)
        self.assertFalse(b.flags.writeable)
        for field in ('file_sha256', 'array_sha256'):
            changed = deepcopy(m)
            changed['tests']['3']['arrays']['xi1'][field] = '0'*64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'changed'):
                run.test_arrays(changed, 3)

    def test_failed_preflight_blocks_production_and_publication(self):
        output = self.path / 'production'
        with patch.object(run, 'validate_preflight', side_effect=ValueError('failed preflight')), \
            patch.object(run, 'test_arrays') as tests, \
                patch.object(run, 'ProcessPoolExecutor') as executor:
            code = run.production(output, self.path / 'absent.json', 'a'*64)
        self.assertEqual(code, 1)
        tests.assert_not_called()
        executor.assert_not_called()
        self.assertFalse((output / 'published').exists())
        summary = run.sealed(output / 'pipeline_summary.json')
        self.assertEqual(summary['status'], 'failed')
        self.assertEqual(summary['phases']['preflight_validation']['status'], 'failed')

    def test_incomplete_sweep_cannot_reach_publisher(self):
        output = self.path / 'incomplete'
        pre = {k: '/synthetic' for k in ('selection_path', 'selection_sha256',
               'selection_verification_path', 'selection_verification_sha256')}
        pre.update(source_sha256={}, environment={}, probes=[{'peak_rss_bytes': 1}])
        manifest = dict(jobs=[{'id': 'incomplete_job'}], tests={})
        future = Future()
        future.set_exception(RuntimeError('synthetic worker crash'))
        with patch.object(run, 'validate_preflight', return_value=(pre, manifest)), \
                patch.object(run, 'verify_inputs'), patch.object(run, 'resource_snapshot', return_value={}), \
                patch.object(run, 'resource_gate', return_value={}), patch.object(run, 'test_arrays'), \
                patch.object(run, 'ProcessPoolExecutor') as executor, patch.object(run, 'verify_job') as verify:
            executor.return_value.__enter__.return_value.submit.return_value = future
            code = run.production(output, self.path / 'preflight.json', 'a'*64)
        self.assertEqual(code, 1)
        verify.assert_not_called()
        self.assertFalse((output / '.staging').exists())
        self.assertFalse((output / 'published').exists())
        failed = run.sealed(output / 'jobs/incomplete_job/worker_failure.json')
        self.assertIsNone(failed['wall_time_s'])
        self.assertNotIn('mean_oos', failed)
        summary = run.sealed(output / 'pipeline_summary.json')
        self.assertEqual(summary['completed_jobs'], 0)
        self.assertEqual(summary['failed_jobs'], ['incomplete_job'])

    def test_full_worker_synthetic_roundtrip_parent_adds_verification_only(self):
        e = self.e
        checkpoint, training, poolpath = (self.path / n for n in ('checkpoint.pth', 'training.json', 'pool.json'))
        e.torch.save(self.model.state_dict(), checkpoint)
        training.write_text('{}')
        poolpath.write_text('{}')
        x1 = e.np.tile(self.x1, (5000, 1))
        x2 = e.np.tile(self.x2, (5000, 1))
        centers = e.np.concatenate((self.x1, self.x2), axis=1)
        pools = [centers[:1], centers[1:]]
        job = dict(id='black_synthetic', method='black', R=3, Ntotal=50, Ntrain=2, seed=0, radius=.4,
            checkpoint=str(checkpoint), checkpoint_sha256=run.sha(checkpoint), state_sha256=e.state_hash(self.model),
            pool_path=str(poolpath), pool_file_sha256=run.sha(poolpath), pool_sha256='a'*64,
            training_data_path=str(training), training_data_sha256=run.sha(training), training_centers_sha256='b'*64,
            selected_candidate_id=4, stopping_candidate_id=4, converged=True, terminal_status='converged',
            mp_count=4, sizes=[1, 1])
        # This is only a temporary synthetic worker protocol fixture, never a publishable cohort.
        test = {'n_test': 10000, 'seed': 9999, 'arrays': {}}
        manifest = dict(jobs=[job], tests={'3': test})
        execution = dict(output=str(self.path), selection_sha256='c'*64, selection_path='/synthetic/selection.json')
        (self.path / 'jobs').mkdir()
        def replay(model, a, b, output, record, sizes=None):
            return self.fixture(Path(output), model, a, b, sizes, record)
        own = (self.model, centers, pools)
        with patch.object(run, '_CONTEXT', (execution, manifest, e, None, {3: (x1, x2)})), \
                patch.object(run, 'own_inputs', return_value=own), \
                patch.object(e, 'evaluate_policy', side_effect=replay):
            envelope = run.worker(0)
            self.assertEqual(envelope['status'], 'completed')
            row = run.verify_job(job, envelope, execution, manifest, None, {3: (x1, x2)})
        raw = run.sealed(envelope['result_path'], envelope['result_sha256'])
        self.assertNotIn('journal_verified', raw)
        self.assertTrue(row['journal_verified'])
        self.assertEqual(row['solves_physical_accepted'], 10002)
        self.assertNotIn('input_sha256', raw)
        self.assertNotIn('historical_training', raw)
        self.assertTrue(all(raw[k] == row[k] for k in raw))
        failed_job = dict(job, id='black_synthetic_failure')
        with patch.object(run, '_CONTEXT', (execution, dict(manifest, jobs=[failed_job]), e, None, {3: (x1, x2)})), \
                patch.object(run, 'own_inputs', return_value=own), \
                patch.object(e, 'evaluate_policy', side_effect=RuntimeError('synthetic solve failure')):
            failed = run.worker(0)
        failure = run.sealed(failed['result_path'], failed['result_sha256'])
        self.assertEqual(failure['status'], 'failed')
        run.timing(failure)
        self.assertFalse(failure['cached'])
        self.assertNotIn('mean_oos', failure)
        self.assertNotIn('finite_pool_mean', failure)


class LaunchContracts(unittest.TestCase):
    def test_command_uses_same_interpreter_and_explicit_binding(self):
        command = launch.command_for('production', '/new', '/preflight.json', 'a'*64)
        self.assertEqual(command[:3], [sys.executable, '-B', '-u'])
        self.assertEqual(command[-4:], ['--preflight', '/preflight.json', '--sha', 'a'*64])
        self.assertNotIn('--preflight', launch.command_for('preflight', '/new'))

    def test_command_rejects_missing_hash_wrong_mode_or_implicit_production(self):
        for args in (('production', '/new'), ('production', '/new', '/pre', 'bad'),
                     ('preflight', '/new', '/pre', 'a'*64), ('other', '/new')):
            with self.subTest(args=args), self.assertRaises(ValueError):
                launch.command_for(*args)

    def test_tmux_unavailable_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launch.shutil, 'which', return_value=None), \
                patch.object(launch.subprocess, 'run') as child:
            with self.assertRaisesRegex(RuntimeError, 'tmux unavailable'):
                launch.main(['preflight', str(Path(tmp) / 'new'), '--session', 'test_only'])
            child.assert_not_called()

    def test_child_records_failure_duration_and_signals_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            spec = dict(mode='preflight', output=str(folder / 'not_created'), command=['synthetic-command'],
                        tmux_session='test_only', completion_channel='test_only_done')
            launch.write(folder / 'invocation.json', spec)
            with patch.object(launch.subprocess, 'call', return_value=7), \
                    patch.object(launch.subprocess, 'run') as signal:
                code = launch.child(folder)
            self.assertEqual(code, 7)
            row = run.sealed(folder / 'child_runtime.json')
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(row['exit_code'], 7)
            run.timing(row)
            signal.assert_called_once_with(['tmux', 'wait-for', '-S', 'test_only_done'], check=True)


if __name__ == '__main__':
    unittest.main()