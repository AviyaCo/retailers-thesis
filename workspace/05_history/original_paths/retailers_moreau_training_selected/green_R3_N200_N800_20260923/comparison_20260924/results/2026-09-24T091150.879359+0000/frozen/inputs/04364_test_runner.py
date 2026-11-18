"""Solver-free synthetic contracts. Never launch the campaign or real preflight."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import run_comparison as r
import preflight as p
from test_plot import fixture

e = r.e


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def grid(self):
        manifest, rows = fixture(self.root)
        for spec, row in zip(manifest['jobs'], rows):
            spec.update(sizes=[1])
            row.update(sizes=[1], manifest_sha256='fixture-digest', solves_attempted=10001,
                solves_accepted=10001, status_counts={'2':10001}, wall_time_s=3.)
        return manifest, rows

    def test_worker_budget_affinity_load_memory_and_cap(self):
        s = dict(affinity=list(range(72)), cpu_quota_cores=None, load_average=[1.,2.,3.],
                 memory_available_bytes=256*r.GIB)
        self.assertEqual(r.choose_workers(s, 100)['workers'], 32)
        s['affinity'] = list(range(12))
        self.assertEqual(r.choose_workers(s, 100)['workers'], 5)
        s['cpu_quota_cores'] = 10.5
        self.assertEqual(r.choose_workers(s, 100)['workers'], 3)
        s['memory_available_bytes'] = 20*r.GIB
        self.assertEqual(r.choose_workers(s, 100)['workers'], 2)
        self.assertEqual(r.choose_workers(s, 2*r.GIB)['workers'], 1)
        s['memory_available_bytes'] = 17*r.GIB
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            r.choose_workers(s, 100)
        s['memory_available_bytes'] = 256*r.GIB
        s['load_average'] = [100.,1.,1.]
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            r.choose_workers(s, 100)

    def test_exact_all_cohorts(self):
        m, _ = self.grid()
        r.require_grid(m)
        self.assertEqual(len(m['jobs']), 780)
        for n, count in ((50,240),(200,180),(800,180)):
            self.assertEqual(sum(j['method']=='scs' and j['Ntotal']==n for j in m['jobs']), count)
        for mutate in (lambda d:d['jobs'].pop(), lambda d:d['jobs'].append(d['jobs'][0]),
                       lambda d:d['jobs'][0].update(seed=11),
                       lambda d:d['jobs'][0].update(id=d['jobs'][1]['id']),
                       lambda d:d['scs_radii']['800'].pop()):
            bad = deepcopy(m)
            mutate(bad)
            with self.assertRaises(ValueError):
                r.require_grid(bad)

    def test_completed_requires_identity_all_solves_finite_and_hash(self):
        m, rows = self.grid()
        r.require_completed(m, rows, 'fixture-digest')
        for change in (lambda rows:rows.pop(), lambda rows:rows.append(rows[0]),
                       lambda rows:rows[0].update(status='failed'),
                       lambda rows:rows[0].update(solves_accepted=0),
                       lambda rows:rows[0].update(status_counts={'9':10001}),
                       lambda rows:rows[0].update(manifest_sha256='changed'),
                       lambda rows:rows[0].update(radius=2.),
                       lambda rows:rows[0].update(mean_oos=float('nan')),
                       lambda rows:rows[0].update(wall_time_s=-1.)):
            bad = deepcopy(rows)
            change(bad)
            with self.assertRaises(ValueError):
                r.require_completed(m, bad, 'fixture-digest')

    def test_seal_is_exclusive_and_rejects_tampering(self):
        path = self.root/'manifest.json'
        digest = r.seal(path, dict(status='frozen'))
        self.assertEqual(e.c.sha(path), digest)
        self.assertEqual(r.read_sealed(path), dict(status='frozen'))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            r.seal(path, {})
        path.chmod(0o644)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'digest changed'):
            r.read_sealed(path)

    def test_extend_existing_snapshot_and_detect_source_or_copy_change(self):
        source = self.root/'runner.py'
        source.write_text('value = 1\n')
        folder = self.root/'frozen'
        (folder/'inputs').mkdir(parents=True)
        m = dict(files={})
        r.capture_sources(m, folder, [source])
        r.capture_sources(m, folder, [source])
        self.assertEqual(len(m['files']),1)
        r.verify_files(m)
        source.write_text('value = 2\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            r.capture_sources(m, folder, [source])
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            r.verify_files(m)
        source.write_text('value = 1\n')
        copied = Path(m['files'][str(source)]['copy'])
        copied.chmod(0o644)
        copied.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Frozen file changed'):
            r.verify_files(m)

    def test_full_freeze_hashset_mismatch_before_test_generation(self):
        m, _ = self.grid()
        m['files'] = {}
        with patch.object(r, 'local_sources', return_value=[]), \
                patch.object(e, 'freeze', return_value=m), \
                patch.object(e.c, 'gen', side_effect=AssertionError('test forbidden')) as gen:
            r.freeze_all(self.root, dict(files={}, runner_sources={}))
            with self.assertRaisesRegex(ValueError, 'hashset differs'):
                r.freeze_all(self.root, dict(files={'new':dict(sha256='changed')},runner_sources={}))
            gen.assert_not_called()

    def preflight_fixture(self):
        m, _ = self.grid()
        m.pop('test')
        m.update(files={}, runner_sources={}, versions=e.versions())
        manifest_path = self.root/'pref_manifest.json'
        digest = r.seal(manifest_path, m)
        pref = dict(status='completed', test_generated=False, job_count=780,
            phases={name:dict(status='completed') for name in ('syntax','unit_tests','resources',
                'full_freeze','numerical_training_only','final_verification')},
            manifest=str(manifest_path), manifest_sha256=digest, source_sha256={})
        return m, pref

    def test_preflight_complete_and_exact_binding(self):
        _, pref = self.preflight_fixture()
        path = self.root/'preflight.json'
        r.seal(path,pref)
        with patch.object(r,'local_sources',return_value=[]):
            actual, _ = r.validate_preflight(path)
            self.assertEqual(actual,pref)
        source = self.root/'later_doc.md'
        source.write_text('New, not preflighted')
        with patch.object(r,'local_sources',return_value=[source]):
            with self.assertRaisesRegex(ValueError,'exact source/doc set'):
                r.validate_preflight(path)

    def test_failed_or_incomplete_preflight_cannot_generate_test(self):
        _, pref = self.preflight_fixture()
        for i, change in enumerate((lambda p:p.update(status='failed'),
                lambda p:p.update(test_generated=True), lambda p:p.update(job_count=779),
                lambda p:p['phases']['unit_tests'].update(status='failed'),
                lambda p:p.update(source_sha256={'unexpected':'hash'}),
                lambda p:p.update(manifest_sha256='wrong'))):
            bad = deepcopy(pref)
            change(bad)
            path = self.root/f'bad{i}.json'
            r.seal(path,bad)
            with patch.object(r,'local_sources',return_value=[]), \
                    patch.object(e.c,'gen',side_effect=AssertionError('forbidden')) as gen:
                with self.assertRaises(ValueError):
                    r.main(['--preflight',str(path),'--output',str(self.root/f'run{i}')])
                gen.assert_not_called()
                self.assertFalse((self.root/f'run{i}').exists())

    def test_test_generation_only_after_grid_and_matches_cached_float64(self):
        m, _ = self.grid()
        arrays = [e.np.ones((10000,3),dtype=e.np.float64) for _ in range(2)]
        for name,a in zip(('xi1','xi2'),arrays):
            m['cached_green_N50']['test_arrays'][name]['array_sha256'] = r.hashlib.sha256(a.tobytes()).hexdigest()
        with patch.object(e.c,'gen',return_value=arrays) as gen:
            r.freeze_test(m,self.root)
            gen.assert_called_once_with(10000,9999)
        self.assertEqual(m['test']['arrays']['xi1']['dtype'],'float64')
        m['jobs'].pop()
        with patch.object(e.c,'gen') as gen:
            with self.assertRaises(ValueError):
                r.freeze_test(m,self.root)
            gen.assert_not_called()
        m, _ = self.grid()
        with patch.object(e.c,'gen',return_value=arrays):
            with self.assertRaisesRegex(ValueError,'differs from cached'):
                r.freeze_test(m,self.root)

    def test_compact_progress_excludes_large_policy_fields(self):
        row = dict(id='x', status='completed', pools=[1], centers=[1], training_runtime_reference=[1],
                   per_center_maxima=[1], scs_historical_membership=dict(offenders=[1]), phases={})
        self.assertEqual(r.compact(row,'x.json'),dict(id='x',status='completed',phases={},result_path='x.json'))

    def test_process_failure_does_not_invent_worker_walltime(self):
        (self.root/'process_failures').mkdir()
        row = r.process_failure(dict(id='x',method='green',Ntotal=200,radius=.2,seed=0),self.root,'crash',90.)
        self.assertIsNone(row['wall_time_s'])
        self.assertEqual(row['parent_queue_elapsed_s'],90.)
        stats = r.runtime_aggregates([row])['groups']
        self.assertTrue(all(s['wall_time_s'] is None for s in stats))

    def test_runtime_status_separation_and_phase_totals(self):
        row = dict(method='green',Ntotal=200,radius=.2,status='completed',wall_time_s=10.,
                   phases={name:dict(status='completed',wall_time_s=1.) for name in
                           ('oos_targets','oos_projection','oos_cost','is_targets','is_projection','is_cost')})
        rows = [row,dict(row,wall_time_s=20.),dict(row,status='failed',wall_time_s=2.,phases={})]
        groups = r.runtime_aggregates(rows)['groups']
        total = next(g for g in groups if g['status']=='completed' and g['scope']=='total')
        self.assertEqual(total['wall_time_s'],dict(n=2,count=2,mean=15.,median=15.,min=10.,max=20.))
        oos = next(g for g in groups if g['status']=='completed' and g['scope']=='oos')
        self.assertEqual(oos['wall_time_s']['mean'],3.)
        failed = next(g for g in groups if g['status']=='failed' and g['scope']=='oos')
        self.assertEqual(failed['unavailable'],1)

    def test_disk_gate_no_cap_or_override(self):
        with patch.object(r.shutil,'disk_usage',return_value=MagicMock(free=100)):
            self.assertEqual(r.disk_gate(self.root,100,'fixture')['required_bytes'],100)
            with self.assertRaisesRegex(ValueError,'Insufficient disk'):
                r.disk_gate(self.root,101,'fixture')

    def test_training_smoke_success_and_rejection_without_test_or_solver(self):
        specs = [dict(id=f'{method}{n}{rho}',method=method,Ntotal=n,Ntrain=n,seed=0,radius=rho,
            checkpoint='fixture',state_sha256='state',centers=[[1.]*6,[2.]*6],original_sample_ids=[3,4],
            saved_native_finite_pool_mean=123.) for method,ns in [('green',(200,800)),('scs',(50,200,800))]
            for n in ns for rho in (0.,.2)]
        model = MagicMock()
        model.targets.return_value=(e.np.ones((2,3)),3.,e.np.ones(3))

        def project(target, inventory, env, diagnostic):
            diagnostic.update(status=2,accepted=True,wall_time_s=.1)
            return e.np.ones(3)

        with patch.object(e,'_bound_file',return_value='fixture'), \
                patch.object(e.c,'load_model',return_value=model), \
                patch.object(e.c,'state_hash',return_value='state'), \
                patch.object(e.c.gp,'Env'), patch.object(e.c,'strict_project',side_effect=project) as solve, \
                patch.object(e.c,'gen',side_effect=AssertionError('test forbidden')) as gen:
            report = {}
            p.training_checks(dict(jobs=specs),report)
            self.assertEqual(len(report['checks']),10)
            self.assertEqual(solve.call_count,20)
            gen.assert_not_called()
            self.assertTrue(all(row['status']=='completed' for row in report['checks']))
            def reject(target, inventory, env, diagnostic):
                diagnostic.update(status=9,accepted=False)
                return e.np.ones(3)
            solve.side_effect=reject
            with self.assertRaisesRegex(ValueError,'acceptance failed'):
                p.training_checks(dict(jobs=specs),{})

    def test_orchestration_freeze_before_test_and_no_failed_publication(self):
        # All I/O here is synthetic test data under a temporary directory.
        m, rows = self.grid()
        m['cached_green_N50']['test_arrays'] = {}
        m['runner_sources'] = {}
        for j in m['jobs']:
            if j['method'] == 'scs':
                j['scs_historical_membership'] = dict(saved_points_checked=0,
                    positive_residual_count=0,count_above_reporting_threshold=0,max_positive_residual=0.)
        for spec,row in zip(m['jobs'],rows):
            row.update(spec)
        pref_path = self.root/'preflight.json'
        pref_path.write_text('{}')
        snapshot = dict(host='test',affinity=list(range(72)),cpu_quota_cores=None,
                        load_average=[0.,0.,0.],memory_available_bytes=256*r.GIB)
        for failure in (None,'policy','render'):
            events = []
            output = self.root/str(failure)

            def freeze(*args):
                events.append('freeze')
                (output/'frozen').mkdir()
                return deepcopy(m)

            def test_arrays(manifest,folder):
                self.assertEqual(events,['freeze'])
                events.append('test')

            def execute(manifest,path,out,summary):
                for row in deepcopy(rows):
                    row['manifest_sha256'] = e.c.sha(path)
                    if failure == 'policy' and row['id'] == rows[0]['id']:
                        row['status'] = 'failed'
                    folder = out/'policies'/row['id']
                    folder.mkdir()
                    e.write_json(folder/'result.json',row)
                    summary['jobs'].append(r.compact(row,folder/'result.json'))

            def render(data,stage):
                (stage/'comparison_R3.png').write_text('synthetic renderer fixture')
                if failure == 'render':
                    raise RuntimeError('mock render failure')
                (stage/'comparison_R3.pdf').write_text('synthetic renderer fixture')

            with patch.object(r,'ROOT',self.root), patch.object(r,'resources',return_value=snapshot), \
                    patch.object(r,'validate_preflight',return_value=(dict(peak_rss_bytes=100),m)), \
                    patch.object(r,'disk_gate',return_value={}), patch.object(r,'freeze_all',side_effect=freeze), \
                    patch.object(r,'freeze_test',side_effect=test_arrays), patch.object(r,'execute',side_effect=execute), \
                    patch.object(r,'final_verify'), patch.object(r,'write_report'), \
                    patch.object(r.plot,'build_plot_data',return_value={}), patch.object(r.plot,'render',side_effect=render), \
                    patch.object(e.c,'gen',side_effect=AssertionError('real generator forbidden')):
                args = ['--preflight',str(pref_path),'--output',str(output)]
                if failure is None:
                    self.assertEqual(r.main(args),output)
                else:
                    with self.assertRaises(ValueError):
                        r.main(args)
            self.assertEqual(events,['freeze','test'])
            summary = e.read_json(output/'summary.json')
            self.assertEqual(summary['status'],'failed' if failure else 'completed')
            self.assertEqual((output/'comparison_R3.png').exists(),failure is None)
            self.assertEqual((output/'comparison_R3.pdf').exists(),failure is None)
            self.assertEqual((output/'plot_data.json').exists(),failure is None)
            if failure is None:
                self.assertEqual(summary['completed_jobs'],780)
                self.assertEqual(summary['missing_job_ids'],[])


if __name__ == '__main__':
    unittest.main()