"""Bounded, training-only tests. Never calls collect or numerical evaluation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('stopping_selection', Path(__file__).with_name('selection.py'))
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def candidate(i, ap, pool, closed=False):
    return dict(candidate_id=i, assessed=True, ap_incumbent_mean=ap, restricted_mean=pool,
                separation_status='separation_satisfied' if closed else 'violating')


class PrefixSelectionTests(unittest.TestCase):
    def test_green_first_pass_includes_zero(self):
        rows = [candidate(0, 100., 100.), candidate(1, 50., 50.)]
        self.assertIs(selection._stopping(rows, 'green'), rows[0])

    def test_green_not_minimum_ap(self):
        rows = [candidate(0, 1., 2.), candidate(1, 4., 4.)]
        self.assertIs(selection._stopping(rows, 'green', stored_stop=1,
                                         terminal_status='green_gap_satisfied'), rows[1])

    def test_green_strict_threshold_absolute_gap_and_denominator(self):
        rows = [candidate(0, 99., 100.), candidate(1, 0., 2e-11), candidate(2, 0., 0.)]
        self.assertEqual(selection._stopping(rows, 'green')['candidate_id'], 2)

    def test_saved_green_cannot_skip_first_pass(self):
        rows = [candidate(0, 2., 2.), candidate(1, 3., 3.)]
        with self.assertRaisesRegex(ValueError, 'first pass'):
            selection._stopping(rows, 'green', stored_stop=1, terminal_status='green_gap_satisfied')

    def test_black_uses_terminal_not_minimum(self):
        rows = [candidate(0, 1., 2.), candidate(1, 4., 4., True)]
        self.assertIs(selection._stopping(rows, 'black', stored_stop=1,
                                         terminal_status='converged'), rows[-1])

    def test_black_checks_from_retained_green_stop_not_restart(self):
        rows = [candidate(0, 1., 1., True), candidate(1, 3., 1.), candidate(2, 4., 4., True)]
        self.assertIs(selection._stopping(rows, 'black', boundary=1, stored_stop=2,
                                         terminal_status='converged'), rows[-1])
        with self.assertRaisesRegex(ValueError, 'continued after'):
            selection._stopping(rows, 'black', boundary=0, stored_stop=2, terminal_status='converged')

    def test_cap40_is_kept_even_with_lower_earlier_ap(self):
        rows = [candidate(i, i + 1., i + 2.) for i in range(41)]
        self.assertIs(selection._stopping(rows, 'black', stored_stop=40,
                                         terminal_status='iteration_cap'), rows[-1])
        with self.assertRaises(ValueError):
            selection._stopping(rows[:-1], 'black', stored_stop=39, terminal_status='iteration_cap')
        with self.assertRaises(ValueError):
            selection._stopping(rows, 'black', stored_stop=40, terminal_status='converged')

    def test_all_earlier_candidates_are_finite_contiguous_assessed(self):
        for field, value in [('ap_incumbent_mean', float('nan')), ('candidate_id', 3), ('assessed', False)]:
            rows = [candidate(0, 1., 2.), candidate(1, 3., 3., True)]
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                selection._stopping(rows, 'black', stored_stop=1, terminal_status='converged')

    def test_green_timing_excludes_outgoing_mp_without_editing_evidence(self):
        phase = dict(status='completed', wall_time_s=2.)
        row = dict(candidate_id=0, origin='original', ap=phase, pool_evaluation=phase,
                   mp=phase, mp_origin='original', epochs=10,
                   incoming_pool_evaluation=phase, returned_pool_evaluation=phase)
        old = dict(R=3, Ntotal=50, assessed_candidate_count=1, training_runtime_reference=dict(
            iterations=[row], nominal={'training_timing': phase}, original_prefix={'wall_time_s': 100.},
            continuation=None))
        before = deepcopy(old)
        green = selection._historical(old, 0, 0, 'green', {}, None)
        self.assertNotIn('mp', green['iterations'][0])
        self.assertEqual(green['green_through_stop']['active_phase_time_s'], 4.)
        self.assertEqual(green['black_after_green_stop']['active_phase_time_s'], 0.)
        self.assertIsNone(green['total_wall_time_s'])
        self.assertEqual(old, before)

    def test_timing_black_boundary_mp_not_double_counted(self):
        phase = dict(status='completed', wall_time_s=2.)
        rows = [dict(candidate_id=0, ap=phase, pool_evaluation=phase, mp=phase, epochs=1,
                     incoming_pool_evaluation=phase, returned_pool_evaluation=phase),
                dict(candidate_id=1, ap=phase, pool_evaluation=phase)]
        old = dict(R=10, Ntotal=50, assessed_candidate_count=2, training_runtime_reference=dict(
            iterations=rows, nominal={'training_timing': phase}, original_prefix={}, continuation={}))
        result = selection._historical(old, 1, 0, 'black', {}, {'wall_time_s': 20.})
        self.assertEqual(result['green_through_stop']['active_phase_time_s'], 4.)
        self.assertEqual(result['black_after_green_stop']['active_phase_time_s'], 10.)
        self.assertIsNone(result['total_wall_time_s'])

    def test_fresh_black_preserves_measured_invocation_total(self):
        phase = dict(status='completed', wall_time_s=2.)
        old = dict(R=10, Ntotal=50, assessed_candidate_count=1, training_runtime_reference=dict(
            iterations=[dict(candidate_id=0, ap=phase, pool_evaluation=phase)],
            nominal={'training_timing': phase}, original_prefix=None, continuation={}))
        result = selection._historical(old, 0, None, 'black', {'wall_time_s': 6.}, None)
        self.assertEqual(result['total_wall_time_s'], 6.)
        self.assertEqual(result['black_after_green_stop']['active_phase_time_s'], 4.)

    def test_full_cohort_keeps_all_seeds_radii_and_caps(self):
        jobs = []
        for method in ('green', 'black'):
            for r in (3, 10):
                for n in (50, 200, 800):
                    for seed in range(10):
                        for rho in selection.RADII + ((2.,) if method == 'black' and r == 10 else ()):
                            jobs.append(dict(id=f'{method}_{r}_{n}_{seed}_{rho}', method=method,
                                R=r, Ntotal=n, seed=seed, radius=rho, converged=True,
                                terminal_status='green_gap_satisfied' if method == 'green' else 'converged',
                                mp_count=0, selected_candidate_id=0, stopping_candidate_id=0))
        for job in [j for j in jobs if j['method'] == 'black'][:11]:
            job.update(converged=False, terminal_status='iteration_cap', mp_count=40,
                       selected_candidate_id=40, stopping_candidate_id=40)
        selection._cohort(jobs, selected=True)
        self.assertEqual(len(jobs), 1110)
        with self.assertRaises(ValueError):
            selection._cohort(jobs[:-1], selected=True)
        with self.assertRaises(ValueError):
            selection._cohort(jobs + [jobs[0]], selected=True)
        jobs[-1]['selected_candidate_id'] = 1
        with self.assertRaisesRegex(ValueError, 'stop/pass'):
            selection._cohort(jobs, selected=True)


class BindingTests(unittest.TestCase):
    def test_local_resolution_hash_conflicts_and_test_byte_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'archive').mkdir()
            (root / 'archive_paths.json').write_text(json.dumps(dict(source_home='/old', archive_root='archive')))
            local = root / 'archive/evidence.json'
            local.write_text('{"training": true}')
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            cap = selection._Bindings(root)
            self.assertEqual(cap.path('/old/evidence.json'), local)
            cap.bind('/old/evidence.json', digest)
            cap.forbidden.add(str(local))
            with self.assertRaisesRegex(ValueError, 'Test bytes forbidden'):
                cap.read('/old/evidence.json')
            cap.forbidden.clear()
            self.assertEqual(cap.read('/old/evidence.json'), {'training': True})
            self.assertEqual(len(cap.verified), 1)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                cap.bind('/old/evidence.json', '0' * 64)
            with self.assertRaises(ValueError):
                cap.path('/external/evidence.json')

    def test_bad_hash_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'archive_paths.json').write_text(json.dumps(dict(source_home='/old', archive_root='archive')))
            (root / 'evidence.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                selection._Bindings(root).add(root / 'evidence.json', '0' * 64)

    def test_seal_is_exclusive_and_binds_json_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'selection_manifest.json'
            digest = selection._seal(path, {'return_rule': 'stopping_checkpoint'})
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(path.with_suffix('.sha256').read_text().strip(), digest)
            self.assertEqual(path.stat().st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                selection._seal(path, {})


class BoundedArchiveTest(unittest.TestCase):
    def test_one_actual_job_without_full_collection_or_test_bytes(self):
        cap = selection._Bindings()
        manifest = cap.sealed(selection.OLD_MANIFEST, selection.OLD_MANIFEST_SHA256)
        old = next(j for j in manifest['jobs'] if j['id'] == 'R3_N200_s0_rho0.2')
        for test in manifest['tests'].values():
            for array in test['arrays'].values():
                cap.forbidden.add(str(cap.path(array['file'])))
        # This bounded test verifies this job only, not the complete 73GB audit.
        for path, digest in old['input_sha256'].items():
            cap.add(path, digest)
        backend_path = cap.path('comparison/black_green_evaluation/backend.py')
        expected = next(d for p, d in manifest['source_sha256'].items() if cap.path(p) == backend_path)
        cap.add(backend_path, expected)
        backend = selection._load(backend_path, '_bounded_stopping_backend')
        record, original, prefix, green, black = selection._prefixes(old, cap, backend)
        self.assertEqual((green['candidate_id'], black['candidate_id']), (2, 5))
        self.assertLess(old['selected_candidate_id'], black['candidate_id'])
        for method, chosen in [('green', green), ('black', black)]:
            job = selection._terminal_job(old, chosen, method, record, original, green, cap, backend)
            self.assertEqual(job['selected_candidate_id'], job['stopping_candidate_id'])
            self.assertEqual(job['pool_sha256'], prefix[job['stopping_candidate_id']]['pool_sha256'])
            self.assertTrue(Path(job['checkpoint']).is_relative_to(selection.ROOT / 'archive'))
            self.assertTrue(job['converged'])
        self.assertFalse(set(cap.verified) & cap.forbidden)


if __name__ == '__main__':
    unittest.main()