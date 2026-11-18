"""Synthetic freeze-contract tests only: no AP, QP, production freeze or test set."""
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import freeze as f


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return path


def cohorts():
    jobs = [dict(Ntotal=n, seed=s, radius=r,
                 status='failed' if (n, s, r) in f.REPLACEMENTS else 'completed')
            for n in f.c.SIZES for s in f.c.SEEDS for r in f.c.RADII]
    replacements = [dict(Ntotal=n, seed=s, radius=r, status='completed', continued=True)
                    for n, s, r in sorted(f.REPLACEMENTS)]
    return (dict(test_evaluated=False, phases={'ccg': {'jobs': jobs}}),
            dict(test_evaluated=False, status='completed', all_five_converged=True,
                 results=replacements))


def green_fixture(root):
    centers = [[2.]*6, [3.]*6]
    training = dict(centers=centers, original_sample_ids=[7, 2])
    record = dict(Ntotal=50, Ntrain=2, seed=0, radius=.5, selected_candidate_id=1,
                  stopping_candidate_id=2, iterations=[])
    for i, (score, restricted) in enumerate(((10., 5.), (8., 5.), (8., 8.))):
        folder = root / f'candidate_{i}'
        pools = [[center] + [[v + .1*j for v in center] for j in range(1, i+1)]
                 for center in centers]
        pool_hash = f.c.pool_digest(pools)
        pool_path = put(folder / 'pool.json', dict(pools=pools, original_sample_ids=[7, 2],
                                                  pool_sha256=pool_hash))
        a = dict(candidate_id=i, Ntotal=50, Ntrain=2, seed=0, radius=.5, assessed=True,
            failed_sample_ids=[], samples=[], state_sha256=f'state{i}', checkpoint_sha256=f'bytes{i}',
            checkpoint=str(folder / 'candidate.pth'), ap_reference=str(folder / 'assessment.json'),
            pool_sha256=pool_hash, pool_file_sha256=f.c.sha(pool_path),
            ap_incumbent_mean=score, restricted_mean=restricted,
            green_gap=abs(score-restricted)/abs(restricted), green_pass=i == 2)
        for k, center in enumerate(centers):
            a['samples'].append(dict(sample_id=k, original_sample_id=training['original_sample_ids'][k],
                candidate_id=i, controller_status='completed', candidate_sha256=a['state_sha256'],
                checkpoint_sha256=a['checkpoint_sha256'], pool_sha256=pool_hash,
                center1=center[:3], center2=center[3:],
                incumbent=dict(cost=score, worst1=center[:3], worst2=center[3:])))
        put(Path(a['ap_reference']), a)
        record['iterations'].append(a)
    selected = record['iterations'][1]
    record.update(selected_checkpoint_sha256=selected['checkpoint_sha256'],
                  selected_state_sha256=selected['state_sha256'],
                  selected_ap_incumbent_mean=8., selected_restricted_mean=5.,
                  selected_ap_reference=selected['ap_reference'])
    return record, training


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Any accidental inference/solver/test-generation call fails the suite.
        for target in ('strict_project', 'scenario_costs'):
            p = patch.object(f.c, target, side_effect=AssertionError('No evaluation in freeze tests'))
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(f.c.c, 'gen', side_effect=AssertionError('No data generation in synthetic tests'))
        p.start()
        self.addCleanup(p.stop)

    def snapshot(self):
        folder = self.root / 'snapshot'
        folder.mkdir()
        return f.Snapshot(folder, dict(files={}, historical_bindings={}))

    def test_exact_hybrid_270(self):
        live, continued = cohorts()
        rows = f.green_cohort(live, continued)
        self.assertEqual(len(rows), 270)
        self.assertEqual(sum(r.get('continued', False) for r in rows), 5)
        for mutation in ('missing', 'wrong', 'failed', 'extra', 'test_seen'):
            a, b = deepcopy(live), deepcopy(continued)
            if mutation == 'missing':
                b['results'].pop()
            elif mutation == 'wrong':
                b['results'][0]['seed'] = 3
            elif mutation == 'failed':
                b['results'][0]['status'] = 'failed'
            elif mutation == 'extra':
                a['phases']['ccg']['jobs'].append(a['phases']['ccg']['jobs'][0])
            else:
                a['test_evaluated'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                f.green_cohort(a, b)

    def test_selected_own_pool_not_terminal_and_earliest_tie(self):
        r, training = green_fixture(self.root)
        f.validate_prefix(r, training)
        a, path, pools = f.selected_green_pool(r, training)
        self.assertEqual(a['candidate_id'], 1)
        self.assertEqual(path.parent.name, 'candidate_1')
        self.assertEqual([len(p) for p in pools], [2, 2])
        r['selected_candidate_id'] = 2
        with self.assertRaisesRegex(ValueError, 'earliest'):
            f.validate_prefix(r, training)

    def test_green_prefix_rejects_unassessed_missing_future_and_membership(self):
        record, training = green_fixture(self.root)
        for mutation in ('unassessed', 'missing', 'future', 'wrongstate', 'outside', 'early_stop'):
            r = deepcopy(record)
            if mutation == 'unassessed':
                r['iterations'][0]['assessed'] = False
            elif mutation == 'missing':
                r['iterations'][0]['samples'].pop()
            elif mutation == 'future':
                r['iterations'].append(deepcopy(r['iterations'][-1]))
            elif mutation == 'wrongstate':
                r['iterations'][0]['samples'][0]['candidate_sha256'] = 'wrong'
            elif mutation == 'outside':
                r['iterations'][0]['samples'][0]['incumbent']['worst1'][0] = 2.500000002
            else:
                r['iterations'][0]['green_pass'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                f.validate_prefix(r, training)

    def test_selected_pool_numeric_hash_and_file_hash(self):
        r, training = green_fixture(self.root)
        a = r['iterations'][1]
        path = Path(a['checkpoint']).parent / 'pool.json'
        obj = f.c.read(path)
        obj['pools'][0][1][0] += .01
        put(path, obj)
        with self.assertRaisesRegex(ValueError, 'file changed'):
            f.selected_green_pool(r, training)
        a['pool_file_sha256'] = f.c.sha(path)
        with self.assertRaisesRegex(ValueError, 'digest changed'):
            f.selected_green_pool(r, training)

    def test_assessment_only_known_flush_timing_difference(self):
        a = dict(candidate_id=0, mp_total_time_s=3., mp_call={'wall_time_s': 3.})
        saved = dict(a, mp_total_time_s=0.)
        f.validate_assessment(saved, a)
        saved['candidate_id'] = 1
        with self.assertRaises(ValueError):
            f.validate_assessment(saved, a)

    def test_legacy_all_positive_residuals_preserved(self):
        center = [1.]*6
        rows = []
        for value in (1.5 + 5e-10, 1.5 + 6.898892e-6):
            point = [value] + [1.]*5
            rows.append([dict(ball_id=0, xi1_hat=center[:3], xi2_hat=center[3:],
                              worst_xi1=point[:3], worst_xi2=point[3:], max_cost=3.)])
        paths = [put(self.root / f'ap{i}.json', row) for i, row in enumerate(rows)]
        before = [p.read_bytes() for p in paths]
        pools, disclosure = f.legacy_pools([center], .5, paths)
        self.assertEqual(pools[0][1, 0], rows[0][0]['worst_xi1'][0])
        self.assertEqual(pools[0][2, 0], rows[1][0]['worst_xi1'][0])
        self.assertEqual(disclosure['positive_residual_count'], 2)
        self.assertEqual(disclosure['count_above_reporting_threshold'], 1)
        self.assertEqual(disclosure['saved_points_checked'], 2)
        self.assertEqual(len(disclosure['offenders']), 2)
        self.assertFalse(disclosure['box_certified'])
        self.assertEqual(before, [p.read_bytes() for p in paths])
        with self.assertRaisesRegex(ValueError, 'membership'):
            f.validate_pools(pools, [center], .5)

    def test_legacy_full_coverage_center_nonfinite_and_hole(self):
        center = [1.]*6
        row = dict(ball_id=0, xi1_hat=center[:3], xi2_hat=center[3:],
                   worst_xi1=center[:3], worst_xi2=center[3:], max_cost=1.)
        for rows in ([], [row, row], [dict(row, ball_id=1)],
                     [dict(row, xi1_hat=[2.]*3)], [dict(row, max_cost=float('nan'))]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                f.legacy_pools([center], .5, ['synthetic'], read=lambda _: rows)
        with self.assertRaises(FileNotFoundError):
            f.legacy_pools([center], .5, [self.root / 'missing.json'])

    def test_selector_mandatory_no_test_cost_selection(self):
        snap = self.snapshot()
        directory = self.root / 'rho_0p2'
        p = put(directory / 'seed_0_summary.json', dict(seed=0, rho=.2, completed=True, final_mp_iter=2))
        cp, i, source, _ = f.legacy_choice(snap, 50, 0, .2, self.root, {.2: directory})
        self.assertEqual((cp.name, i, source), ('seed_0_mp_iter2.pth', 2, p))
        # Separate snapshots are not needed to test schema errors with a read-only reader.
        class Reader:
            def read(self, _):
                return self.row
        reader = Reader()
        for row in (dict(seed=0, rho=.2, completed=True),
                    dict(seed=0, rho=.2, completed=False, final_mp_iter=0),
                    dict(seed=0, rho=.2, completed=True, final_mp_iter=True)):
            reader.row = row
            with self.assertRaises((ValueError, KeyError)):
                f.legacy_choice(reader, 50, 0, .2, self.root, {.2: directory})

    def test_grid_metadata_only_and_exact_seed_coverage(self):
        grid = [.001, .003, .005, .01, .02, .05, .1, .15, .2, .3, .4, .5, .7, 1., 1.5]
        cache = {'sweep': {str(r): {'n_seeds': 10, 'rob_oos': 'must never inspect costs'} for r in grid}}
        for r in grid:
            for s in f.c.SEEDS:
                put(self.root / f'rho_{r:g}' / f'seed_{s}_summary.json', {})
        radii, directories = f.legacy_grid(self.root, cache)
        self.assertEqual(radii, [0.] + grid)
        (directories[.001] / 'seed_9_summary.json').unlink()
        with self.assertRaisesRegex(ValueError, 'seed coverage'):
            f.legacy_grid(self.root, cache)

    def test_snapshot_readonly_no_overwrite_and_mutation_detection(self):
        snap = self.snapshot()
        original = put(self.root / 'input.json', {'value': 1})
        key = snap.capture(original)
        copy = Path(snap.manifest['files'][key]['copy'])
        self.assertTrue(copy.is_absolute())
        self.assertEqual(copy.stat().st_mode & 0o222, 0)
        self.assertEqual(snap.read(original), {'value': 1})
        with self.assertRaises(FileExistsError):
            f.Snapshot(snap.folder, {'files': {}})
        put(original, {'value': 2})
        with self.assertRaises(ValueError):
            snap.capture(original)

    def test_learned_hash_wrapped_legacy_and_known_buffers_only(self):
        snap = self.snapshot()
        state = f.c.c.Model().state_dict()
        raw, legacy = self.root / 'green.pth', self.root / 'legacy.pth'
        f.c.torch.save(state, raw)
        f.c.torch.save({'model_state_dict': dict(state, **{
            'proj_layer.q': f.c.torch.zeros(1), 'proj_layer.A': f.c.torch.zeros(1)})}, legacy)
        a, b = snap.model(raw, 'green'), snap.model(legacy, 'legacy')
        self.assertEqual(a['state_sha256'], b['state_sha256'])
        with self.assertRaisesRegex(ValueError, 'Learned state'):
            snap.model(raw, 'green', state='incorrect')
        bad = self.root / 'bad.pth'
        f.c.torch.save({'model_state_dict': dict(state, unknown=f.c.torch.zeros(1))}, bad)
        with self.assertRaisesRegex(ValueError, 'Unknown nonlearned'):
            snap.model(bad, 'legacy')

    def test_cache_provenance_not_invented(self):
        cp = self.root / 'checkpoint.pth'
        self.assertFalse(f._cache_checkpoint({}, .2, 0, cp, 1, {}, 'hash')['checkpoint_hash_persisted'])
        rows = {str(s): dict(checkpoint=str(cp), final_mp_iter=1, best_ap_iter=2) for s in f.c.SEEDS}
        cache = dict(checkpoint_provenance={'0.2': rows})
        actual = f._cache_checkpoint(cache, .2, 0, cp, 1, {'best_ap_iter': 2}, 'hash')
        self.assertEqual(actual['status'], 'selector_metadata_verified')
        rows['0']['checkpoint_sha256'] = 'bad'
        with self.assertRaisesRegex(ValueError, 'bytes'):
            f._cache_checkpoint(cache, .2, 0, cp, 1, {'best_ap_iter': 2}, 'hash')

    def test_hash_only_dependency_verification(self):
        source = put(self.root / 'dependency.json', {})
        manifest = dict(files={}, versions=f.c.versions(), historical_bindings={str(source): f.c.sha(source)})
        f.verify(manifest)
        put(source, {'changed': True})
        with self.assertRaisesRegex(ValueError, 'historical binding'):
            f.verify(manifest)

    def test_compact_timing_no_epoch_histories_or_invented_fields(self):
        row = dict(wall_time_s=3., epochs_evaluated=12, epoch_history=[{'loss': 4}], train_step_times=[1])
        self.assertEqual(f.compact_timing(row), dict(wall_time_s=3., epochs_evaluated=12))
        self.assertEqual(f.compact_timing({}), {})

    def test_historical_manifests_verify_but_do_not_copy_unconsumed_sources(self):
        snap = self.snapshot()
        dependency = put(self.root / 'research.json', {'untouched': True})
        digest = f.c.sha(dependency)
        live, cont = self.root / 'live', self.root / 'continued'
        for folder in (live, cont):
            put(folder / 'provenance.json', dict(source_sha256={str(dependency): digest},
                                               input_sha256={str(dependency): digest}))
        with patch.multiple(f, LIVE=live, CONTINUED=cont):
            f._historical_bindings(snap, {}, {})
        self.assertEqual(snap.manifest['historical_bindings'], {str(dependency): digest})
        self.assertNotIn(str(dependency), snap.manifest['files'])
        self.assertEqual(len(snap.manifest['files']), 2)
        put(dependency, {'changed': True})
        snap.manifest['historical_bindings'].clear()
        with patch.multiple(f, LIVE=live, CONTINUED=cont), self.assertRaisesRegex(ValueError, 'binding'):
            f._historical_bindings(snap, {}, {})

    def test_legacy_collector_raw_prefix_native_mean_and_compact_runtime(self):
        snap = self.snapshot()
        snap.manifest.update(jobs=[], baseline_paths={}, legacy_radii={})
        base = self.root / 'N50'
        sweep = base / 'rho_sweep'
        directory = sweep / 'rho_0p2'
        cp = f.c.c.Model().state_dict()
        put(base / 'seed_0_summary.json', dict(seed_time_s=7.))
        f.c.torch.save(dict(model_state_dict=cp, seed=0, n_samples=50,
                            hard_loss=4., best_loss=3.), base / 'seed_0_nominal.pth')
        put(directory / 'seed_0_summary.json', dict(seed=0, rho=.2, completed=True,
            final_mp_iter=1, total_time_s=9., ccg_trajectory=[dict(iter=1, ap_time_s=2., mp_time_s=7.)]))
        f.c.torch.save(dict(model_state_dict=cp, seed=0, rho=.2, ccg_iter=1, best_loss=3.),
                       directory / 'seed_0_mp_iter1.pth')
        rows = [dict(ball_id=i, xi1_hat=[1.]*3, xi2_hat=[1.]*3,
                     worst_xi1=[1.200001, 1., 1.], worst_xi2=[1.]*3, max_cost=5.) for i in range(50)]
        put(directory / 'seed_0_ap_iter1.json', rows)
        # A post-selected-MP file must not be read, nor added to the returned pool.
        later = put(directory / 'seed_0_ap_iter2.json', {'invalid_post_selection_file': True})
        put(sweep / 'sweep_eval_shipping_10k.json', dict(
            meta=dict(N_train=50, N_test=10000, seed_test=9999, seeds=[0]), sweep={}))
        put(base / 'sro_fa/sro_fa_shipping_results_N50_R3.json', {})
        put(base / 'saa_cond_mip_1m_shipping_exact.json', {})
        with patch.object(f, 'LEGACY', self.root), patch.multiple(f.c, SIZES=(50,), SEEDS=[0]), \
             patch.object(f, 'legacy_grid', return_value=([0., .2], {.2: directory})), \
             patch.object(f, '_training_arrays', return_value=(f.np.ones((50, 3)), f.np.ones((50, 3)))):
            f._freeze_legacy(snap)
        nominal, robust = snap.manifest['jobs']
        self.assertEqual(nominal['saved_native_finite_pool_mean'], 4.)
        self.assertIsNone(robust['saved_native_finite_pool_mean'])
        self.assertEqual(robust['sizes'], [2]*50)
        self.assertEqual(robust['legacy_membership']['positive_residual_count'], 50)
        self.assertNotIn(str(later), snap.manifest['files'])
        self.assertEqual(robust['pool_sha256'], f.c.pool_digest(robust['pools']))
        self.assertEqual(robust['training_runtime_reference']['iterations'],
                         [dict(iter=1, ap_time_s=2., mp_time_s=7.)])
        self.assertTrue(all(Path(p).is_absolute() and p in snap.manifest['files']
                            for row in (nominal, robust) for p in row['input_files']))

    def test_failed_freeze_persists_utc_monotonic_and_never_overwrites(self):
        # Missing README is an intentional synthetic failure before any sources/data.
        put(self.root / 'local.py', {})
        output = self.root / 'out'
        with patch.object(f, 'ROOT', self.root), self.assertRaises(FileNotFoundError):
            f.freeze(output)
        failure = f.c.read(output / 'frozen/freeze_failure.json')
        self.assertEqual(failure['status'], 'failed')
        self.assertFalse(failure['test_generated'])
        self.assertGreaterEqual(failure['timing']['wall_time_s'], 0)
        self.assertIn('+00:00', failure['timing']['start_utc'])
        before = (output / 'frozen/freeze_failure.json').read_bytes()
        with self.assertRaises(FileExistsError):
            f.freeze(output)
        self.assertEqual(before, (output / 'frozen/freeze_failure.json').read_bytes())

    def test_successful_orchestration_750_and_local_capture_before_collectors(self):
        local = self.root / 'local'
        put(local / 'freeze.py', {})
        put(local / 'runner.py', {})
        put(local / 'README.md', {})
        ref, poolref = put(self.root / 'ref.py', {}), put(self.root / 'poolref.py', {})
        live, continued = self.root / 'live', self.root / 'continued'
        put(live / 'summary.json', {})
        put(continued / 'summary.json', {})
        def green(snap, *_):
            self.assertTrue(all(str(p) in snap.manifest['files'] for p in
                                [local/'freeze.py', local/'runner.py', local/'README.md', ref, poolref]))
            for n in f.c.SIZES:
                for s in f.c.SEEDS:
                    for r in f.c.RADII:
                        snap.manifest['jobs'].append(dict(id=f._id('green', n, r, s), method='green',
                                                         Ntotal=n, seed=s, radius=r))
        def legacy(snap):
            for n in f.c.SIZES:
                radii = [i / 10 for i in range(16)]
                snap.manifest['legacy_radii'][str(n)] = radii
                for s in f.c.SEEDS:
                    for r in radii:
                        snap.manifest['jobs'].append(dict(id=f._id('legacy', n, r, s), method='legacy',
                                                         Ntotal=n, seed=s, radius=r))
        with patch.multiple(f, ROOT=local, LIVE=live, CONTINUED=continued), \
             patch.multiple(f.c, REFERENCE=ref, POOL_REFERENCE=poolref), \
             patch.object(f, '_historical_bindings'), patch.object(f, '_freeze_green', side_effect=green), \
             patch.object(f, '_freeze_legacy', side_effect=legacy):
            manifest = f.freeze(self.root / 'success')
        self.assertEqual(manifest['job_count'], 750)
        self.assertEqual(manifest['status'], 'frozen')
        self.assertIsNone(manifest['workers'])
        self.assertTrue(all(re.fullmatch(r'[A-Za-z0-9_]+', j['id']) for j in manifest['jobs']))
        self.assertEqual(f.c.read(self.root / 'success/frozen/freeze_manifest.json'), manifest)


if __name__ == '__main__':
    unittest.main()