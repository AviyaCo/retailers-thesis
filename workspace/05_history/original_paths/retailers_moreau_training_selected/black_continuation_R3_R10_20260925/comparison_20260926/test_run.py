"""Solver-free orchestration tests; synthetic fixtures are not experiment results.

No historical held-out arrays are opened and no subprocess is actually launched.
The separately owned test_plotting suite checks the actual large base schemas.
"""
from copy import deepcopy
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch
import run as r


def timed(**fields):
    return dict(start_utc='2026-09-26T00:00:00+00:00', end_utc='2026-09-26T00:00:01+00:00',
                wall_time_s=1., **fields)


def jobs():
    result = []
    # Fixture caps deliberately include the three required capped smoke identities.
    caps = {(3, 50, 9, .7), (10, 50, 8, .2), (10, 200, 6, .7)}
    caps |= {(10, 800, seed, 1.5) for seed in range(8)}
    for R in (3, 10):
        for n in (50, 200, 800):
            for seed in range(10):
                for rho in r.backend().RADII + ((2.,) if R == 10 else ()):
                    cap = (R, n, seed, rho) in caps
                    stop = 40 if cap else 1
                    identity = dict(checkpoint_sha256='1'*64, state_sha256='2'*64,
                                    pool_sha256='3'*64, pool_file_sha256='4'*64)
                    result.append(dict(id=f'R{R}_N{n}_s{seed}_r{rho:g}', R=R, Ntotal=n, seed=seed,
                        radius=rho, Ntrain=4*n//5, capped_nonconverged=cap, converged=not cap,
                        comparison_derived_cap_selection=cap and R == 10, retained=True,
                        terminal_status='iteration_cap' if cap else 'converged', stopping_candidate_id=stop,
                        selected_candidate_id=0, assessed_candidate_count=stop+1,
                        selection_rule=r.backend().SELECTION, **identity,
                        selection_evidence=[dict(candidate_id=i, ap_incumbent_mean=1., **identity)
                                            for i in range(stop+1)]))
    return result


class CohortTests(unittest.TestCase):
    def test_all570_and_all11_caps(self):
        data = jobs()
        r.cohort(data)
        self.assertEqual(len(data), 570)
        self.assertEqual(sum(j['capped_nonconverged'] for j in data), 11)
        selected = r.smoke_jobs(data)
        self.assertEqual([(j['R'], j['Ntotal'], j['seed'], j['radius']) for j in selected], list(r.SMOKES))
        self.assertEqual({(j['R'], j['Ntotal']) for j in selected},
                         {(R, n) for R in (3, 10) for n in (50, 200, 800)})

    def test_duplicate_missing_and_filtered_caps_rejected(self):
        data = jobs()
        for invalid in (data[:-1], data[:-1]+data[:1], [j for j in data if not j['capped_nonconverged']]):
            with self.assertRaises(ValueError):
                r.cohort(invalid)

    def test_terminal40_minimum_and_exact_earliest_tie(self):
        data = jobs()
        cap = next(j for j in data if j['capped_nonconverged'])
        cap['selection_evidence'][-1]['ap_incumbent_mean'] = .5
        with self.assertRaisesRegex(ValueError, 'earliest'):
            r.cohort(data)
        cap['selected_candidate_id'] = 40
        r.cohort(data)
        cap['selection_evidence'][2]['ap_incumbent_mean'] = .5
        with self.assertRaisesRegex(ValueError, 'earliest'):
            r.cohort(data)
        cap['selected_candidate_id'] = 2
        r.cohort(data)

    def test_fullprefix_and_ownpool_are_bound(self):
        data = jobs()
        data[0]['selection_evidence'].pop()
        with self.assertRaisesRegex(ValueError, 'prefix'):
            r.cohort(data)
        data = jobs()
        data[0]['pool_sha256'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'own-policy'):
            r.cohort(data)


class IntegrityTests(unittest.TestCase):
    def test_streamed_seal_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            digest = r.seal(path, {'large': list(range(1000))})
            self.assertEqual(r.sealed(path, digest)['large'][-1], 999)
            with self.assertRaises(FileExistsError):
                r.seal(path, {})
            path.chmod(0o644)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                r.sealed(path, digest)

    def test_original_and_private_copy_must_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, copy = root / 'input', root / 'copy'
            source.write_text('training evidence')
            copy.write_bytes(source.read_bytes())
            copy.chmod(0o444)
            digest = r.sha(source)
            manifest = dict(input_sha256={str(source): digest}, source_sha256={},
                            files={str(source): dict(copy=str(copy), sha256=digest)})
            with patch.object(r, 'local_sources', return_value=[]):
                r.verify_files(manifest)
                source.write_text('changed origin')
                with self.assertRaisesRegex(ValueError, 'Original/copy'):
                    r.verify_files(manifest)
                source.write_bytes(copy.read_bytes())
                copy.chmod(0o644)
                with self.assertRaisesRegex(ValueError, 'writable'):
                    r.verify_files(manifest)
                copy.write_text('changed copy')
                copy.chmod(0o444)
                with self.assertRaisesRegex(ValueError, 'Original/copy'):
                    r.verify_files(manifest)

    def test_compact_base_keeps_verbatim_manifest_reference(self):
        collection = dict(base_comparisons={'3': dict(manifest_path='/original/base.json',
            manifest_sha256='a'*64, test={'seed': 9999}, versions={'python': 'same'},
            files={'/large/original': {'copy': '/large/frozen'}})})
        r.compact_bases(collection)
        base = collection['base_comparisons']['3']
        self.assertNotIn('files', base)
        self.assertEqual(base['files_metadata_reference'], '/original/base.json')
        self.assertEqual(base['test'], {'seed': 9999})

    def test_unsealed_policy_cannot_touch_test_or_plot_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'policy.json'
            path.write_text('{}')
            with patch.object(r, 'plotting', side_effect=AssertionError('test/plot access forbidden')):
                with self.assertRaises(ValueError):
                    r.bind_after_seal({}, path, '0'*64, root)

    def test_production_reuses_preflight_without_collection(self):
        tree = ast.parse(Path(r.__file__).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'production')
        calls = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertNotIn('collect_policies', calls)
        self.assertNotIn('materialize', calls)
        self.assertNotIn('gen', calls)


class GateTests(unittest.TestCase):
    def test_preflight_nested_phases_counts_and_six_exact_probes(self):
        data = jobs()
        manifest = dict(status='frozen', jobs=data, workers=16, threads=1,
            policies_frozen_before_test=True, physical_tolerance=1e-4, membership_tolerance=1e-9,
            test_bytes_accessed=False, test_evaluated=False, source_sha256={}, files={},
            base_comparisons={str(R): {'test': {'arrays': {}}} for R in (3, 10)})
        phases = {name: timed(status='completed') for name in r.PREFLIGHT_PHASES}
        phases['syntax']['files'] = []
        pref = timed(status='completed', mode='preflight', test_bytes_accessed=False, job_count=570,
            workers=16, phases=phases, source_sha256={}, solves_attempted=12, solves_accepted=12,
            status_counts={'2': 12}, unit_tests={},
            processes=[dict(id=job['id']) for job in r.smoke_jobs(data)])
        with patch.object(r, 'validate_units'), patch.object(r, 'local_sources', return_value=[]):
            r.validate_preflight_evidence(pref, manifest)
            for name in r.PREFLIGHT_PHASES:
                broken = deepcopy(pref)
                broken['phases'][name]['status'] = 'failed'
                with self.subTest(phase=name), self.assertRaises(ValueError):
                    r.validate_preflight_evidence(broken, manifest)
            for field, value in [('solves_accepted', 10), ('job_count', 569),
                                 ('processes', pref['processes'][:-1]), ('test_bytes_accessed', True)]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    r.validate_preflight_evidence({**pref, field: value}, manifest)

    def test_exact16_with_reserves_no_silent_reduction(self):
        snapshot = dict(affinity=list(range(128)), cpu_quota_cores=None,
                        load_average=[1., 1., 1.], memory_available_bytes=500*r.GIB)
        self.assertEqual(r.resource_gate(snapshot, [2*r.GIB])['workers'], 16)
        for change in ({'cpu_quota_cores': 16}, {'memory_available_bytes': 20*r.GIB},
                       {'load_average': [120., 120., 120.]}):
            with self.assertRaisesRegex(ValueError, '16 workers'):
                r.resource_gate({**snapshot, **change}, [2*r.GIB])

    def test_unit_counts_and_nested_failure_not_just_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log, path = root / 'unit.log', root / 'unit.json'
            log.write_text('ARTIFICIAL unit evidence, not an executed run')
            raw = timed(status='completed', modules=list(r.UNIT_MODULES), tests_run=3, tests_discovered=3,
                test_ids=[m+'.Fixture.test_example' for m in r.UNIT_MODULES], skipped=0, errors=0,
                failures=0, expected_failures=0, unexpected_successes=0)
            r.write(path, raw)
            evidence = dict(raw, path=str(path), sha256=r.sha(path), log=str(log), log_sha256=r.sha(log), returncode=0)
            evidence['command'] = r.unit_command() + [str(path)]
            r.validate_units(evidence)
            for field, value in [('tests_run', 2), ('skipped', 1), ('errors', 1), ('returncode', 1),
                                 ('test_ids', ['one']*3)]:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    r.validate_units({**evidence, field: value})

    def test_duration_rejects_nan_negative_boolean(self):
        for value in (float('nan'), float('inf'), -1., True, None):
            with self.assertRaises(ValueError):
                r.seconds(value)

    def test_runtime_groups_do_not_sum_concurrent_elapsed(self):
        rows = [dict(status='completed', cached=False, R=3, Ntotal=50, radius=.7, wall_time_s=t)
                for t in (2., 6.)]
        rows += [dict(rows[0], status='failed', wall_time_s=999.), dict(rows[0], cached=True)]
        group = r.runtime_groups(rows)[0]
        self.assertEqual((group['count'], group['mean'], group['median'], group['min'], group['max']),
                         (2, 4., 4., 2., 6.))
        self.assertNotIn('elapsed', group)


class JournalTests(unittest.TestCase):
    def fixture(self, root, smoke=True):
        job = dict(id='ARTIFICIAL', R=3, Ntotal=50, Ntrain=2, seed=9, radius=.7,
                   sizes=[1, 2], original_sample_ids=[8, 4], training_runtime_reference={'status': 'persisted_not_current'})
        folder = root / 'policies' / job['id']
        folder.mkdir(parents=True)
        names = [('training_probe', 2)] if smoke else [('oos', 10000), ('is', 3)]
        record = dict(job, journals={}, phases={}, iterations=0., barrier_iterations=0,
                      solves_attempted=sum(n for _, n in names), solves_accepted=sum(n for _, n in names))
        record['status_counts'] = {'2': record['solves_accepted']}
        for name, count in names:
            path = folder / f'{name}_solves.jsonl'
            with path.open('x') as stream:
                for i in range(count):
                    d = timed(row=i, phase=name, status=2, accepted=True, physical_residual=1e-5,
                              iterations=1., barrier_iterations=0, controller_wall_time_s=1.)
                    if name == 'is':
                        center = 0 if i == 0 else 1
                        d.update(center_id=center, original_sample_id=job['original_sample_ids'][center])
                    stream.write(json.dumps(d)+'\n')
            record['journals'][name+'_solves'] = dict(path=str(path), sha256=r.sha(path))
            record['phases'][name+'_projection'] = timed(status='completed', solves_attempted=count,
                solves_accepted=count, sample_count=count, status_counts={'2': count}, iterations=float(count),
                barrier_iterations=0, solver_threads=1, solver_settings={'Threads': 1})
            record['iterations'] += count
        return job, record

    def change_first(self, record, name, field, value):
        info = record['journals'][name+'_solves']
        path = Path(info['path'])
        lines = path.read_text().splitlines()
        first = json.loads(lines[0])
        first[field] = value
        lines[0] = json.dumps(first)
        path.write_text('\n'.join(lines)+'\n')
        # Even recomputing the hash cannot make wrong numerical evidence acceptable.
        info['sha256'] = r.sha(path)

    def test_actual_rows_not_attestation_counts(self):
        for field, value in [('row', 1), ('phase', 'oos'), ('status', 9), ('accepted', False),
                             ('physical_residual', 1.01e-4), ('iterations', float('nan')),
                             ('controller_wall_time_s', -1.), ('barrier_iterations', True)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                job, row = self.fixture(root)
                self.assertEqual(r.journal_verify(row, job, root, True)[0], 2)
                self.change_first(row, 'training_probe', field, value)
                with self.assertRaises(ValueError):
                    r.journal_verify(row, job, root, True)

    def test_ownpool_original_ids_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, row = self.fixture(root, False)
            self.assertEqual(r.journal_verify(row, job, root)[0], 10003)
            self.change_first(row, 'is', 'original_sample_id', 4)
            with self.assertRaisesRegex(ValueError, 'center mapping'):
                r.journal_verify(row, job, root)

    def test_missing_row_even_with_updated_digest_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, row = self.fixture(root)
            info = row['journals']['training_probe_solves']
            path = Path(info['path'])
            path.write_text(path.read_text().splitlines()[0]+'\n')
            info['sha256'] = r.sha(path)
            with self.assertRaisesRegex(ValueError, 'journal count'):
                r.journal_verify(row, job, root, True)

    def test_normalization_does_not_rewrite_hashed_backend_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job, row = self.fixture(root)
            manifest_path = root / 'manifest.json'
            manifest_path.write_text('{}')
            row.update(status='completed', inputs_unchanged=True, cached=False,
                manifest_path=str(manifest_path), manifest_sha256=r.sha(manifest_path),
                preflight_only=True, test_evaluated=False, solver_threads=1, torch_threads=1,
                torch_interop_threads=1, worker_count=16, timing_clock='time.perf_counter',
                timing_unit='seconds', training_probe_original_sample_ids=[8, 4],
                mean_oos=None, finite_pool_mean=None, training_probe_mean=5., **timed())
            path = root / 'policies/ARTIFICIAL/result.json'
            r.write(path, row)
            original = path.read_bytes()
            envelope = timed(id=job['id'], status='completed', peak_rss_bytes=1024,
                result_path=str(path), result_sha256=r.sha(path), model_wall_time_s=1.)
            ep = root / 'envelope.json'
            r.write(ep, envelope)
            process = timed(id=job['id'], status='completed', returncode=0, envelope_path=str(ep),
                            envelope_sha256=r.sha(ep), envelope=envelope)
            normalized, ref = r.normalize_result(job, process, manifest_path, root, True)
            self.assertEqual(path.read_bytes(), original)
            self.assertNotIn('result_sha256', r.read(path))
            self.assertEqual(normalized['solves_physical_accepted'], 2)
            self.assertEqual(normalized['source_prefix_timings'], job['training_runtime_reference'])
            self.assertTrue(ref['journal_verified'])
            self.assertEqual(ref['result_sha256'], r.sha(path))

    def test_global_missing_duplicate_worker_barrier(self):
        data = jobs()
        for processes in ([], [dict(id=data[0]['id'])]*570):
            with self.assertRaisesRegex(ValueError, 'Missing/duplicate'):
                r.verify_all_results(data, processes, Path('/unused'), Path('/unused'))


class LauncherTests(unittest.TestCase):
    def test_only_local_run_not_parent_campaign(self):
        command = launch.command_for('preflight', '/tmp/black_pref')
        self.assertEqual(Path(command[3]), r.ROOT / 'run.py')
        self.assertEqual(command[4], 'preflight')
        self.assertNotIn('campaign.py', ' '.join(command))
        command = launch.command_for('production', '/tmp/black_prod', '/tmp/pref/summary.json', 'a'*64)
        self.assertEqual(command[-4:], ['--preflight', '/tmp/pref/summary.json', '--sha', 'a'*64])

    def test_production_binding_and_preflight_no_auto_production(self):
        for path, digest in [(None, None), ('/tmp/pref', None), ('/tmp/pref', 'short')]:
            with self.assertRaises(ValueError):
                launch.command_for('production', '/tmp/out', path, digest)
        with self.assertRaises(ValueError):
            launch.command_for('preflight', '/tmp/out', '/tmp/pref', 'a'*64)

    def test_launcher_syntax_and_no_implicit_launch_on_import(self):
        for path in (r.ROOT / 'launch.py', r.ROOT / 'run.py'):
            tree = ast.parse(path.read_text())
            compile(tree, str(path), 'exec')
            calls = [node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
            self.assertEqual(calls, [])

    def test_missing_tmux_stops_without_subprocess(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(launch.shutil, 'which', return_value=None), \
                patch.object(launch.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'tmux unavailable'):
                launch.main(['preflight', str(Path(directory) / 'new'), '--session', 'black_pref'])
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()