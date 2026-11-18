"""Solver-free artificial contracts; one real spawn exercises worker failure I/O."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_comparison as r
import preflight as p
import launch
import test_oracle as ot

e = r.e


def grid_fixture():
    jobs = [dict(id=f'{method}_{n}_{rho}_{seed}', method=method, Ntotal=n,
                 Ntrain=4*n//5 if method == 'green' else n, radius=rho, seed=seed,
                 centers=[[1.]*20 for _ in range(4*n//5 if method == 'green' else n)],
                 original_sample_ids=list(range(4*n//5 if method == 'green' else n)))
            for method in ('green', 'scs') for n in (50, 200, 800)
            for rho in (e.RADII if method == 'green' else e.SCS_RADII) for seed in range(10)]
    return dict(jobs=jobs, job_count=810, status='frozen', files={},
                scs_radii={str(n): list(e.SCS_RADII) for n in (50, 200, 800)})


def timed(**kwargs):
    return dict(start_utc='2026-09-25T00:00:00+00:00', end_utc='2026-09-25T00:00:01+00:00',
                wall_time_s=.1, **kwargs)


def evidence_fixture(manifest, root):
    phases = {name: timed(status='completed') for name in ('syntax', 'unit_tests', 'resources',
              'full_freeze', 'numerical_training_only', 'oracle_training_only', 'final_verification')}
    phases['syntax'].update(bash_returncode=0,
                           python_files=[str(path) for path in r.local_sources() if path.suffix == '.py'])
    phases['unit_tests'].update(returncode=0, command=[sys.executable, '-B', '-m', 'unittest', '-v',
                                                      'test_evaluation', 'test_plot', 'test_runner', 'test_oracle'])
    rows = []
    for spec in manifest['jobs']:
        if spec['seed'] != 0 or spec['radius'] not in (0., .2):
            continue
        row = {key: spec[key] for key in ('id', 'method', 'Ntotal', 'Ntrain', 'radius', 'seed')}
        row.update(status='completed', worker_count=1, solver_threads=1, costs=[1., 3.],
                   training_subset_mean=2., original_sample_ids=spec['original_sample_ids'][:2],
                   phases={'training_only_replay': timed(status='completed')},
                   checks=[timed(status=2, accepted=True, iterations=1., barrier_iterations=0,
                                 physical_residual=0.) for _ in range(2)])
        rows.append(row)
    result = timed(status='completed', test_generated=False, job_count=810, versions=e.versions(),
                 timing_clock='time.perf_counter', timing_unit='seconds', phases=phases, checks=rows,
                 solves_attempted=24, solves_accepted=24, status_counts={'2': 24}, peak_rss_bytes=12345,
                 solve_counts_scope='strict_training_QP_only; oracle LP recorded separately',
                 output=str(root), manifest_sha256='a'*64)
    with ot.OracleTests().fake_backend(ot.FakeModel(), ot.FakeEnv()):
        p.oracle_training_check(manifest, result, root)
    return result


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch.object(e.c, 'gen', side_effect=AssertionError('Scenario generation forbidden'))
        self.guard.start()
        self.addCleanup(self.guard.stop)
        for name in ('Env', 'Model'):
            guard = patch.object(e.c.gp, name, side_effect=AssertionError('Real solver forbidden'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_exact_locked_definitions_and_apis(self):
        for path, digest in r.LOCKS.items():
            self.assertEqual(e.c.sha(path), digest)
        self.assertEqual(list(inspect.signature(r.run_worker).parameters), ['spec', 'manifest_path', 'output'])
        self.assertEqual(list(inspect.signature(p.training_checks).parameters), ['manifest', 'report'])
        self.assertEqual(list(inspect.signature(e.c.load_model).parameters), ['path', 'method'])
        self.assertEqual(r.run_worker.__module__, 'run_comparison')
        self.assertEqual(e.c.Model().fc_layers[-1].out_features, 10)
        self.assertEqual(len(r.EXPECTED_LOCAL), 13)
        self.assertTrue({'oracle.py', 'test_oracle.py'}.issubset(r.EXPECTED_LOCAL))
        self.assertTrue(set(r.oracle.SOURCE_FILES).issubset(r.local_sources()))
        self.assertEqual(list(inspect.signature(r.plot.build_plot_data).parameters), ['manifest', 'rows', 'oracle'])

    def test_full_fresh_grid_rejects_n50_cache_missing_and_duplicates(self):
        manifest = grid_fixture()
        r.require_grid(manifest)
        for change in (lambda m: m['jobs'].pop(), lambda m: m['jobs'].append(m['jobs'][0]),
                       lambda m: m.update(cached_green_N50={}),
                       lambda m: m['scs_radii']['50'].__setitem__(1, .003),
                       lambda m: m.update(job_count=780)):
            bad = deepcopy(manifest)
            change(bad)
            with self.assertRaises(ValueError):
                r.require_grid(bad)

    def test_nested_preflight_rejects_forged_success(self):
        manifest = grid_fixture()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        pref = evidence_fixture(manifest, Path(tmp.name))
        r.validate_preflight_evidence(pref, manifest)
        changes = [lambda d: d['phases']['unit_tests'].update(returncode=1),
            lambda d: d['phases']['unit_tests']['command'].pop(),
            lambda d: d['phases']['syntax'].update(bash_returncode=1),
            lambda d: d['phases']['syntax']['python_files'].pop(),
            lambda d: d['checks'].pop(), lambda d: d['checks'].__setitem__(1, deepcopy(d['checks'][0])),
            lambda d: d['checks'][0].update(seed=1), lambda d: d['checks'][0].update(status='running'),
            lambda d: d['checks'][0]['checks'].pop(),
            lambda d: d['checks'][0]['checks'][0].update(accepted=False),
            lambda d: d['checks'][0]['checks'][0].update(status=9),
            lambda d: d['checks'][0]['checks'][0].update(physical_residual=.01),
            lambda d: d['checks'][0]['checks'][0].update(iterations=None),
            lambda d: d['checks'][0]['checks'][0].update(wall_time_s=float('nan')),
            lambda d: d['checks'][0]['checks'][0].update(end_utc='2026-09-24T00:00:00+00:00'),
            lambda d: d['phases']['full_freeze'].update(wall_time_s=-1),
            lambda d: d['checks'][0]['phases']['training_only_replay'].update(status='failed'),
            lambda d: d['checks'][0].update(costs=[float('nan'), 3.]),
            lambda d: d.update(solves_accepted=23), lambda d: d.update(test_generated=True),
            lambda d: d['phases']['oracle_training_only'].update(status='failed'),
            lambda d: d['oracle_training_only'].update(status='failed'),
            lambda d: d['oracle_training_only'].update(accepted=False),
            lambda d: d['oracle_training_only'].update(solver_status=9),
            lambda d: d['oracle_training_only'].update(Ntotal=200),
            lambda d: d['oracle_training_only'].update(seed=9999),
            lambda d: d['oracle_training_only'].update(original_sample_ids=[5, 6]),
            lambda d: d['oracle_training_only'].update(training_points_sha256='b'*64),
            lambda d: d['oracle_training_only'].update(manifest_sha256='b'*64),
            lambda d: d['oracle_training_only'].update(mean_cost=float('nan')),
            lambda d: d['oracle_training_only'].update(iterations=None),
            lambda d: d['oracle_training_only'].update(wall_time_s=-1),
            lambda d: d['oracle_training_only'].update(physical_residual=.1),
            lambda d: d['oracle_training_only'].update(objective_abs_error=.1),
            lambda d: d['oracle_training_only']['phases'].pop('optimization'),
            lambda d: d['oracle_training_only']['phases']['optimization'].update(status='failed'),
            lambda d: d['oracle_training_only']['solution'].update(sha256='c'*64)]
        for i, change in enumerate(changes):
            bad = deepcopy(pref)
            change(bad)
            with self.subTest(case=i), self.assertRaises((ValueError, KeyError)):
                r.validate_preflight_evidence(bad, manifest)
        Path(pref['oracle_training_only']['solution']['file']).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'solution hash'):
            r.validate_preflight_evidence(pref, manifest)

    def test_oracle_training_one_mock_lp_saved_first_two_only_and_failure(self):
        manifest = grid_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = dict(output=str(root), manifest_sha256='a'*64)
            with ot.OracleTests().fake_backend(ot.FakeModel(), ot.FakeEnv()), \
                    patch.object(r.oracle, 'solve', wraps=r.oracle.solve) as solve:
                p.oracle_training_check(manifest, report, root)
                solve.assert_called_once()
                self.assertEqual(solve.call_args.args[0].shape, (2, 10))
                self.assertEqual(solve.call_args.args[1].shape, (2, 10))
            r.require_training_oracle(report, manifest)
            e.c.gen.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            report = dict(output=tmp, manifest_sha256='a'*64)
            with ot.OracleTests().fake_backend(ot.FakeModel(status=9), ot.FakeEnv()), \
                    self.assertRaisesRegex(ValueError, 'requires OPTIMAL'):
                p.oracle_training_check(manifest, report, Path(tmp))
            self.assertEqual(report['oracle_training_only']['status'], 'failed')
            self.assertEqual(report['oracle_training_only']['solver_status'], 9)

    def test_oracle_plot_adapter_counts_hashes_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = ot.manifest_fixture(root)
            # The fake symbolic backend keeps even this 10k identity test tiny:
            # fabricate ONLY the artifact fixture, never run a real solver.
            with ot.OracleTests().fake_backend(ot.FakeModel(), ot.FakeEnv()):
                small = r.oracle.solve(e.np.ones((2, 10)), e.np.ones((2, 10)), {}, root/'small.npz')
            folder = root/'oracle'
            folder.mkdir()
            with (folder/'solution.npz').open('xb') as stream:
                e.np.savez(stream, Q10=e.np.asarray(0.), Q1r=e.np.full(10, 2.), Q2=e.np.zeros((10000, 10)))
            row = dict(small, N=10000, seed=9999, test_seed=9999, training_seed=None,
                invocation='fresh', independent_replicates=1, test=manifest['test'],
                manifest_sha256=e.c.sha(root/'frozen/manifest.json'),
                dimensions=dict(variables=200011, constraints=310000), solve_wall_time_s=small['wall_time_s'])
            row['solution'] = dict(small['solution'], file=str(folder/'solution.npz'),
                                   sha256=e.c.sha(folder/'solution.npz'))
            path = folder/'result.json'
            e.write_json(path, row)
            binding = dict(file=str(path), sha256=e.c.sha(path), solution=row['solution'])
            adapted = r.oracle_plot_record(manifest, root, binding)
            self.assertEqual(adapted['solves_attempted'], adapted['solves_accepted'])
            self.assertEqual(adapted['independent_solves'], 1)
            self.assertEqual(adapted['status_counts'], {'2': 1})
            self.assertEqual(set(adapted['source_sha256']), {str(s) for s in r.oracle.SOURCE_FILES})
            self.assertIsInstance(row['source_sha256'], str)  # Raw schema untouched.
            for field, value in [('solver_status', 9), ('accepted', False), ('test_seed', 0),
                                 ('independent_replicates', 8), ('mean_cost', 7.)]:
                bad = dict(row, **{field: value})
                e.write_json(path, bad)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    r.oracle_plot_record(manifest, root, dict(binding, sha256=e.c.sha(path)))
            e.write_json(path, row)
            changed = deepcopy(manifest)
            changed['runner_sources'].pop(str(r.oracle.HERE/'test_oracle.py'))
            with self.assertRaises(KeyError):
                r.oracle_plot_record(changed, root, binding)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'result hash'):
                r.oracle_plot_record(manifest, root, binding)
            e.write_json(path, row)
            (folder/'solution.npz').write_bytes(b'corruption')
            with self.assertRaisesRegex(ValueError, 'solution hash'):
                r.oracle_plot_record(manifest, root, binding)

    def test_policy_result_hash_recheck_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'result.json'
            path.write_text('{}')
            bindings = {str(path): e.c.sha(path)}
            r.verify_result_hashes(bindings)
            path.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'Policy result hash'):
                r.verify_result_hashes(bindings)

    def test_seal_original_copy_and_expected_digest_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'inputs').mkdir()
            source = root / 'input.txt'
            source.write_text('original')
            manifest = dict(files={})
            r.capture_sources(manifest, root, [source])
            r.verify_files(manifest)
            path = root / 'manifest.json'
            digest = r.seal(path, manifest)
            self.assertEqual(r.read_sealed(path), manifest)
            with self.assertRaises(ValueError):
                r.seal(path, manifest)
            with self.assertRaisesRegex(ValueError, 'Exact preflight'):
                r.validate_preflight(path, '0'*64)
            source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
                r.verify_files(manifest)
            self.assertEqual(e.c.sha(path), digest)
            path.chmod(0o644)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'digest changed'):
                r.read_sealed(path)

    def test_test_arrays_require_presealed_grid_and_are_r10_float64(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = grid_fixture()
            with self.assertRaises(FileNotFoundError):
                r.freeze_test(manifest, root)
            e.c.gen.assert_not_called()
            r.seal(root / 'policies_manifest.json', manifest)
            arrays = (e.np.ones((10000, 10)), e.np.full((10000, 10), 2.))
            with patch.object(e.c, 'gen', return_value=arrays) as draw:
                r.freeze_test(manifest, root)
                draw.assert_called_once_with(10000, 9999)  # Artificial return only; no RNG draw.
            self.assertNotIn('cached_green_N50', manifest)
            for name, array in zip(('xi1', 'xi2'), arrays):
                info = manifest['test']['arrays'][name]
                self.assertEqual(info['shape'], [10000, 10])
                self.assertEqual(info['dtype'], 'float64')
                self.assertEqual(info['array_sha256'], hashlib.sha256(array.tobytes()).hexdigest())
                self.assertEqual(Path(info['file']).stat().st_mode & 0o222, 0)
            with self.assertRaises(ValueError):
                r.freeze_test(manifest, root)

    def test_resource_gate_limits_and_no_fallback(self):
        resources = dict(affinity=list(range(128)), cpu_quota_cores=64.,
                         load_average=[1., 2., 3.], memory_available_bytes=256*r.GIB)
        self.assertEqual(r.choose_workers(resources, r.GIB)['workers'], 32)
        self.assertEqual(r.choose_workers(dict(resources, cpu_quota_cores=12.5), r.GIB)['workers'], 5)
        for bad in (dict(resources, memory_available_bytes=16*r.GIB),
                    dict(resources, cpu_quota_cores=4.), dict(resources, load_average=[200.]*3)):
            with self.assertRaisesRegex(ValueError, 'headroom'):
                r.choose_workers(bad, r.GIB)

    def test_runtime_status_separation_not_sum_of_concurrent_times(self):
        rows = [dict(method='green', Ntotal=50, radius=.2, status=status, wall_time_s=value)
                for status, value in [('completed', 2.), ('completed', 4.), ('failed', 10.), ('process_failed', None)]]
        totals = [g for g in r.runtime_aggregates(rows)['groups'] if g['scope'] == 'total']
        completed = next(g for g in totals if g['status'] == 'completed')
        self.assertEqual(completed['wall_time_s'], dict(n=2, count=2, mean=3., median=3., min=2., max=4.))
        self.assertEqual(next(g for g in totals if g['status'] == 'process_failed')['unavailable'], 1)

    def test_training_smoke_adapter_24_mocked_qps_and_r10_inputs(self):
        from unittest.mock import MagicMock
        manifest = grid_fixture()
        for spec in manifest['jobs']:
            spec.update(checkpoint='synthetic', state_sha256='state', saved_native_finite_pool_mean=1.,
                        centers=[[1.]*20, [2.]*20])
        model = MagicMock()
        model.targets.return_value = e.np.zeros((2, 10)), 1., e.np.zeros(10)

        def solve(target, q10, env, diagnostic):
            diagnostic.update(status=2, accepted=True, iterations=1., barrier_iterations=0,
                              physical_residual=0., wall_time_s=.01)
            return e.np.zeros(10)

        report = {}
        with patch.object(e, '_bound_file', return_value=Path('/synthetic')), \
                patch.object(e.c, 'load_model', return_value=model), \
                patch.object(e.c, 'state_hash', return_value='state'), \
                patch.object(e.c.gp, 'Env'), patch.object(e.c, 'strict_project', side_effect=solve) as qp:
            p.training_checks(manifest, report)
        self.assertEqual(qp.call_count, 24)
        self.assertEqual(len(report['checks']), 12)
        self.assertTrue(all(row['status'] == 'completed' and len(row['checks']) == 2 for row in report['checks']))
        self.assertTrue(all(call.args[0].shape == (2, 10) for call in model.targets.call_args_list))
        e.c.gen.assert_not_called()

    def test_historical_runtime_report_includes_n50_and_distinct_sro_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = dict(phases={name: dict(jobs=[]) for name in
                ('nominal', 'ccg', 'initialization', 'lr_validation', 'radius_validation')})
            for n in (50, 200, 800):
                for lr in (.005, .01, .05):
                    training['phases']['nominal']['jobs'].append(dict(Ntotal=n, lr_net=lr, status='completed',
                        training=dict(wall_time_s=4.), train_evaluation=dict(wall_time_s=1.)))
                for rho in e.RADII:
                    training['phases']['ccg']['jobs'].append(dict(Ntotal=n, radius=rho, status='completed', wall_time_s=8.))
                for name in ('initialization', 'lr_validation', 'radius_validation'):
                    training['phases'][name]['jobs'].append(dict(Ntotal=n, status='completed', wall_time_s=2.))
            path = root / 'training.json'
            e.write_json(path, training)
            manifest = dict(green_training_reference=dict(summary=str(path)), files={str(path): dict(copy=str(path))},
                cached_baselines={str(n): dict(sro_fa={'0.2': {'0': dict(time_partition=1., time_lp=2.,
                    time_eval=3., time_total=7.)}}) for n in (50, 200, 800)},
                historically_nonconverged_scs=[], jobs=[])
            summary = dict(status='completed', resources=dict(host='synthetic', affinity=[0]), workers=1,
                           preflight='synthetic', preflight_sha256='a'*64, wall_time_s=1., figures=[], plot_data=None)
            aggregates = r.runtime_aggregates([])
            r.write_report(root, manifest, summary, aggregates)
            historical = aggregates['historical']
            self.assertEqual(len([g for g in historical if g['Ntotal'] == 50 and g['method'] == 'Green nominal training']), 3)
            self.assertEqual(len([g for g in historical if g['Ntotal'] == 50 and g['method'].startswith('Green CCG')]), 9)
            for label, value in [('partition', 1.), ('solver', 2.), ('OOS', 3.), ('total', 7.)]:
                group = next(g for g in historical if g['Ntotal'] == 50 and g['method'] == 'SRO-FA historical '+label)
                self.assertEqual(group['wall_time_s']['mean'], value)
            self.assertIn('not byte-certified', (root / 'RESULTS.md').read_text())
            self.assertIn('ONE fresh joint LP', (root / 'RESULTS.md').read_text())
            self.assertIn('replay separately (unmeasured)', (root / 'RESULTS.md').read_text())
            self.assertIsNone(aggregates['common_oracle']['result'])
            self.assertIsNone(aggregates['common_oracle']['separate_serialization_time_s'])
            (root/'oracle').mkdir()
            raw = timed(status='failed', solver_status=9, phases={'optimization': timed(status='failed')})
            e.write_json(root/'oracle/result.json', raw)
            r.write_report(root, manifest, summary, aggregates)
            self.assertEqual(aggregates['common_oracle']['result'], raw)
            self.assertEqual(aggregates['common_oracle']['independent_solves'], 1)
            self.assertIn('| optimization | failed | 0.100000 |', (root/'RESULTS.md').read_text())

    def test_real_spawn_wrapper_failure_no_solver_no_generator(self):
        # The child uses a deliberately absent manifest and fails in input loading,
        # before any model, generator or solver can be reached. Real spawn + logs.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'logs').mkdir()
            (root / 'policies').mkdir()
            spec = dict(id='synthetic_missing_manifest', method='green', Ntotal=50, radius=0., seed=0)
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=e.initialize_worker) as pool:
                row = pool.submit(r.run_worker, spec, str(root / 'absent.json'), str(root)).result(timeout=90)
            self.assertEqual(row['status'], 'failed')
            self.assertEqual(row['solves_attempted'], 0)
            self.assertGreaterEqual(row['wall_time_s'], 0)
            self.assertEqual(row['phases']['input_loading']['status'], 'failed')
            self.assertTrue((root / 'logs/synthetic_missing_manifest.log').is_file())
            self.assertEqual(e.read_json(row['result_path'])['status'], 'failed')

    def test_journal_counts_reject_missing_unaccepted_and_bad_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'policies/fixture'
            folder.mkdir(parents=True)
            d = timed(row=0, phase='oos', status=2, accepted=True, physical_residual=0., controller_wall_time_s=.2)
            with (folder / 'oos_solves.jsonl').open('x') as file:
                for i in range(10000):
                    file.write(json.dumps(dict(d, row=i))+'\n')
            path = folder / 'is_solves.jsonl'
            path.write_text(json.dumps(dict(d, phase='is'))+'\n')
            manifest = dict(jobs=[dict(id='fixture', sizes=[1])])
            self.assertEqual(r.verify_journals(manifest, root)['solves_accepted'], 10001)
            for bad in ('', json.dumps(dict(d, phase='is', accepted=False))+'\n',
                        json.dumps(dict(d, phase='is', wall_time_s=-1))+'\n'):
                path.write_text(bad)
                with self.assertRaises(ValueError):
                    r.verify_journals(manifest, root)

    def test_launch_command_only_no_launcher_execution(self):
        args = argparse.Namespace(mode='preflight', output=Path('/tmp/not_launched'), preflight=None,
                                  preflight_sha256=None)
        self.assertIn(str(r.ROOT / 'preflight.py'), launch.command_for(args))
        args.mode = 'production'
        with self.assertRaises(ValueError):
            launch.command_for(args)
        args.preflight, args.preflight_sha256 = Path('/tmp/not_read'), 'a'*64
        self.assertEqual(launch.command_for(args)[-2:], ['--preflight-sha256', 'a'*64])


    def test_monitor_completion_signal_after_persisted_exit_and_runtime(self):
        from types import SimpleNamespace
        for returncode in (0, 1):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                data = dict(mode='preflight', session='fixture', completion_channel='fixture_done',
                    command=['not-executed'], exact_command='not-executed', cwd=str(root),
                    output=str(root/'absent'), log=str(root/'child.log'), runtime=str(root/'runtime.json'),
                    exit=str(root/'child.exit'), source_sha256={})
                invocation = root/'invocation.json'
                launch.write_once(invocation, data)
                def subprocess(command, **kwargs):
                    if command == ['tmux', 'display-message', '-p', '#S']:
                        return SimpleNamespace(stdout='fixture\n')
                    if command == ['not-executed']:
                        return SimpleNamespace(returncode=returncode)
                    self.assertEqual(command, ['tmux', 'wait-for', '-S', 'fixture_done'])
                    self.assertEqual((root/'child.exit').read_text(), f'{returncode}\n')
                    record = e.read_json(root/'runtime.json')
                    self.assertEqual(record['returncode'], returncode)
                    self.assertEqual(record['completion_channel'], 'fixture_done')
                    self.assertGreaterEqual(record['wall_time_s'], 0.)
                with patch.dict(os.environ, TMUX='artificial'), \
                        patch.object(launch.subprocess, 'run', side_effect=subprocess) as calls:
                    self.assertEqual(launch.monitor(invocation, launch.sha(invocation)), returncode)
                    self.assertEqual(calls.call_count, 3)

    def test_main_preflight_or_oracle_failure_never_dispatches_or_plots(self):
        from contextlib import ExitStack
        for failure in ('preflight', 'oracle'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                output = Path(tmp)/'output'
                previous = dict(files={}, jobs=[])
                pref = dict(peak_rss_bytes=1)
                validation = stack.enter_context(patch.object(r, 'validate_preflight', return_value=(pref, previous)))
                if failure == 'preflight':
                    validation.side_effect = ValueError('fixture preflight failure')
                def fake_freeze(folder, previous):
                    (folder/'frozen').mkdir()
                    return dict(jobs=[], files={})
                stack.enter_context(patch.object(r, 'freeze_all', side_effect=fake_freeze))
                for name, value in [('resources', {}), ('choose_workers', {'workers': 1}),
                                    ('disk_gate', {}),
                                    ('freeze_test', None), ('membership_report', {}), ('write_report', None)]:
                    stack.enter_context(patch.object(r, name, return_value=value))
                lp = stack.enter_context(patch.object(r.oracle, 'run', side_effect=ValueError('fixture oracle failure')))
                execute = stack.enter_context(patch.object(r, 'execute'))
                plot = stack.enter_context(patch.object(r.plot, 'build_plot_data'))
                with self.assertRaisesRegex(ValueError, f'fixture {failure} failure'):
                    r.main(['--output', str(output), '--preflight', str(Path(tmp)/'unused'),
                            '--preflight-sha256', 'a'*64])
                self.assertEqual(lp.call_count, int(failure == 'oracle'))
                execute.assert_not_called()
                plot.assert_not_called()
                self.assertFalse((output/'published').exists())
                self.assertEqual(e.read_json(output/'summary.json')['status'], 'failed')


if __name__ == '__main__':
    unittest.main()