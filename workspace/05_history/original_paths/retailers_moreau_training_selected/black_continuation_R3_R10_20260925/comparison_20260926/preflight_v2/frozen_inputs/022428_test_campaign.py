"""Pure orchestration tests. No solver, experiment, or historical writes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import campaign as c


def inventory():
    jobs = []
    for r, n in c.GROUPS:
        radii = c.RADII + ((2.,) if r == 10 else ())
        for index, (seed, radius) in enumerate((s, rho) for s in range(10) for rho in c.RADII):
            jobs.append(dict(id=c.job_id(r, n, seed, radius), R=r, Ntotal=n, seed=seed,
                radius=radius, source_summary_path='/original/summary.json',
                expected={'stop_every_center_pass': index < c.EXPECTED_PASS[r, n]}))
        if 2. in radii:
            jobs.extend(dict(id=c.job_id(r, n, s, 2.), R=r, Ntotal=n, seed=s, radius=2.,
                source_summary_path=None, expected={'stop_every_center_pass': False}) for s in range(10))
    cached = [dict(id=c.job_id(3, 50, seed, radius), R=3, Ntotal=50, seed=seed,
        radius=radius, terminal_status='iteration_cap' if (seed, radius) == (9, .7)
        else 'separation_satisfied') for seed in range(10) for radius in c.RADII]
    return dict(jobs=jobs, cached_black=cached)


def candidate():
    return dict(candidate_id=0, assessed=True, state_sha256='state', checkpoint_sha256='checkpoint',
        pool_sha256='pool', pool_costs=[100.], ap_incumbent_mean=100.,
        separation_status='separation_satisfied', samples=[dict(sample_id=0, candidate_id=0,
            candidate_sha256='state', checkpoint_sha256='checkpoint', pool_sha256='pool',
            controller_status='completed', status='completed', bound=101., pool_cost=100.,
            residual=1./101., incumbent={'cost': 100.})])


class InventoryTests(unittest.TestCase):
    def test_exact_570_450_30_90(self):
        manifest = inventory()
        c.check_inventory(manifest)
        self.assertEqual(len(manifest['jobs']), 480)
        self.assertEqual(sum(j['source_summary_path'] is not None for j in manifest['jobs']), 450)
        self.assertEqual(len(manifest['cached_black']), 90)

    def test_duplicate_missing_extra(self):
        for mutation in ('duplicate', 'missing', 'extra'):
            with self.subTest(mutation=mutation):
                manifest = inventory()
                if mutation == 'duplicate':
                    manifest['jobs'][0] = copy.deepcopy(manifest['jobs'][1])
                elif mutation == 'missing':
                    manifest['jobs'].pop()
                else:
                    manifest['jobs'].append(copy.deepcopy(manifest['jobs'][0]))
                with self.assertRaises(ValueError):
                    c.check_inventory(manifest)

    def test_no_nominal_restart_of_existing_job(self):
        manifest = inventory()
        manifest['jobs'][0]['source_summary_path'] = None
        with self.assertRaises(ValueError):
            c.check_inventory(manifest)

    def test_known_green_stop_inventory(self):
        manifest = inventory()
        manifest['jobs'][0]['expected']['stop_every_center_pass'] = False
        with self.assertRaises(ValueError):
            c.check_inventory(manifest)

    def test_preserve_cached_black_cap(self):
        manifest = inventory()
        next(j for j in manifest['cached_black'] if j['terminal_status'] == 'iteration_cap')['terminal_status'] = 'separation_satisfied'
        with self.assertRaises(ValueError):
            c.check_inventory(manifest)

    def test_total_budget_includes_old_mps(self):
        self.assertEqual(c.remaining_mp(38), 2)
        self.assertEqual(c.remaining_mp(40), 0)
        self.assertEqual(c.remaining_mp(0), 40)
        for invalid in (-1, 41, True, 2.5):
            with self.assertRaises(ValueError):
                c.remaining_mp(invalid)

    def test_interleaved_groups_and_seed_round_robin(self):
        jobs = c.schedule(inventory()['jobs'])
        self.assertEqual([(j['R'], j['Ntotal']) for j in jobs[:5]], list(c.GROUPS))
        self.assertTrue(all(j['seed'] == 0 for j in jobs[:5]))
        self.assertTrue(all(j['seed'] == 1 for j in jobs[5:10]))


class BindingTests(unittest.TestCase):
    def test_capture_does_not_write_original_and_detects_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.json'
            path.write_text('{"original": true}')
            before = path.stat()
            capture = c.Capture()
            sha = capture.add(path)
            self.assertEqual(capture.read(path, sha), {'original': True})
            self.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)
            c.verify_hashes(capture.hashes)
            path.write_text('{"original": false}')
            with self.assertRaises(ValueError):
                c.verify_hashes(capture.hashes)

    def test_original_hash_never_recaptured_as_new_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.py'
            path.write_text('changed source')
            with self.assertRaises(ValueError):
                c.Capture().add(path, '0' * 64)
        with self.assertRaises(ValueError):
            c.merge_hashes({'/a.py': '0' * 64}, {'/a.py': '1' * 64})

    def test_all_centers_required_and_bound_formula_recomputed(self):
        good = candidate()
        self.assertTrue(c.candidate_binding(good, 1))
        for field, value in (('residual', 0.), ('candidate_sha256', 'other'),
                             ('sample_id', 1), ('controller_status', 'failed')):
            bad = copy.deepcopy(good)
            bad['samples'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.candidate_binding(bad, 1)
        with self.assertRaises(ValueError):
            c.candidate_binding(good, 2)

    def test_candidate_cost_nonfinite_rejected(self):
        bad = candidate()
        bad['samples'][0]['bound'] = float('nan')
        with self.assertRaises(ValueError):
            c.candidate_binding(bad, 1)

    def test_source_capture_includes_parent_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('campaign.py', 'test_campaign.py', 'continuation.py', 'README.md', 'launcher.py'):
                (root / name).write_text('source')
            with patch.object(c, 'ROOT', root):
                self.assertEqual({p.name for p in c.local_paths()},
                    {'campaign.py', 'test_campaign.py', 'continuation.py', 'README.md', 'launcher.py'})


class ExecutionTests(unittest.TestCase):
    def test_failure_does_not_cancel_other_jobs_or_retry(self):
        jobs = inventory()['jobs'][:12]
        calls, published = [], []

        def launch(job):
            calls.append(job['id'])
            if job['id'] == jobs[0]['id']:
                raise RuntimeError('intentional unit-test failure')
            return dict(job, status='completed', current_invocation_wall_time_s=1.)

        rows = c.run_all(jobs, 2, launch, lambda rows: published.append(len(rows)))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len(set(calls)), 12)
        self.assertEqual(len(calls), 12)
        self.assertEqual(sum(r['status'] == 'failed' for r in rows), 1)
        self.assertEqual(published[-1], 12)

    def test_honest_adapter_statuses(self):
        self.assertEqual(c.classify(dict(status='nonconverged', complete=False,
                                        terminal_status='iteration_cap')), 'nonconverged')
        good = dict(status='completed', complete=True, converged=True, terminal_status='converged')
        self.assertEqual(c.classify(good), 'completed')
        self.assertEqual(c.classify(dict(good, cached_only=True)), 'cached_only')
        self.assertEqual(c.classify(dict(status='failed')), 'failed')
        with self.assertRaises(ValueError):
            c.classify(dict(status='completed', complete=True, terminal_status='unknown'))

    def test_nested_failure_is_not_hidden_by_top_level_success(self):
        for bad in (dict(status='completed', checks={'ap': {'status': 'failed'}}),
                    dict(status='completed', nested=[{'returncode': 1}])):
            with self.assertRaises(ValueError):
                c.nested_success(bad)

    def test_resource_policy_one_joint_cap_reserves_and_rss(self):
        snapshot = dict(affinity=list(range(72)), cpu_quota_cores=60,
                        load_average=[2., 3., 1.], memory_available_bytes=200 * 1024**3)
        policy = c.choose_workers(snapshot, [2 * 1024**3, 4 * 1024**3])
        self.assertEqual(policy['ccg'], 16)
        self.assertEqual(policy['reserve_cpu_slots'], 6)
        self.assertEqual(policy['per_worker_memory_bytes'], 6 * 1024**3)
        self.assertGreaterEqual(policy['memory_reserve_bytes'], 20 * 1024**3)
        with self.assertRaises(ValueError):
            c.choose_workers(dict(snapshot, cpu_quota_cores=4), [1024])
        with self.assertRaises(ValueError):
            c.choose_workers(snapshot, [])

    def test_runtime_groups_never_mix_cached_failed_nonconverged(self):
        rows = [dict(R=3, Ntotal=200, radius=.7, status=status, wall=duration)
                for status, duration in [('completed', 1.), ('completed', 3.),
                    ('failed', 8.), ('nonconverged', 12.), ('cached_only', None)]]
        result = {row['status']: row for row in c.aggregates(rows, 'wall')}
        self.assertEqual(result['completed']['mean_s'], 2.)
        self.assertEqual(result['completed']['median_s'], 2.)
        self.assertEqual(result['completed']['n'], 2)
        self.assertEqual(result['cached_only']['unavailable_n'], 1)
        self.assertIsNone(result['cached_only']['mean_s'])
        self.assertEqual(result['failed']['mean_s'], 8.)

    def test_manifest_seal_detects_mutation_before_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps({'changed': True}))
            with self.assertRaisesRegex(ValueError, 'seal mismatch'):
                c.verified_manifest(path, '0' * 64)


if __name__ == '__main__':
    unittest.main()