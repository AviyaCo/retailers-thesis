"""Synthetic unit-only contracts; no production freeze, QPs, models or test data.

Reuse the frozen parent's analytic fixtures and actual validators. Checkpoint
hash strings below are synthetic identities, not claims about learned models.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# selection appends PARENT: keep the local runner ahead of the frozen modules.
import selection as s
import run_comparison as runner
from test_freeze import green_fixture, put
from test_plot import fixture as plot_fixture

c, f, p = s.c, s.f, runner.p
LEGACY_RADII = [0., .001, .003, .005, .01, .02, .05, .1,
                .15, .2, .3, .4, .5, .7, 1., 1.5]


def stopping_fixture(root):
    record, training = green_fixture(root)
    record.update(status='completed', complete=True, converged=True,
                  terminal_status='green_gap_satisfied')
    return record, training


def grid_fixture():
    """Only grid metadata; no policy execution or observed costs."""
    jobs = []
    for method, radii in (('green', c.RADII), ('legacy', LEGACY_RADII)):
        for n in c.SIZES:
            for seed in c.SEEDS:
                for radius in radii:
                    job = dict(id=f._id(method, n, radius, seed), method=method,
                               Ntotal=n, seed=seed, radius=radius)
                    if method == 'green':
                        job.update(selected_candidate_id=2, stopping_candidate_id=2,
                                   return_rule=s.RETURN_RULE, green_pass=True)
                    jobs.append(job)
    return jobs


class SyntheticOnly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Fail loudly if a test accidentally leaves analytic fixture code.
        for module, names in ((c, ('strict_project', 'scenario_costs', 'load_model')),
                              (c.c, ('gen',)),
                              (runner.e, ('worker',))):
            for name in names:
                guard = patch.object(module, name, side_effect=AssertionError(
                    'No production evaluation, model loading or generation in unit tests'))
                guard.start()
                self.addCleanup(guard.stop)


class StoppingPoolTests(SyntheticOnly):
    def assert_rejected(self, record, training, message=None, error=ValueError):
        before = deepcopy((record, training))
        context = (self.assertRaisesRegex(error, message) if message
                   else self.assertRaises(error))
        with context:
            s.stopping_pool(record, training)
        # JSON also makes comparisons involving deliberate NaNs deterministic.
        self.assertEqual(json.dumps((record, training), sort_keys=True),
                         json.dumps(before, sort_keys=True))

    def test_imports_bind_local_runner_and_frozen_shipping_dependencies(self):
        self.assertEqual(Path(runner.__file__).resolve().parent, s.ROOT)
        for module in (c, f, runner.e, p):
            self.assertEqual(Path(module.__file__).resolve().parent, s.PARENT)
        self.assertIs(runner.s, s)

    def test_first_pass_not_earliest_min_ap_and_no_mutation(self):
        record, training = stopping_fixture(self.root)
        before = deepcopy((record, training))
        files = {path: path.read_bytes() for path in self.root.rglob('*.json')}
        reads = []

        def read(path):
            reads.append(Path(path))
            return c.read(path)

        candidate, path, pools = s.stopping_pool(record, training, read)
        self.assertEqual(record['selected_candidate_id'], 1)  # Historical AP tie winner.
        self.assertEqual(candidate['candidate_id'], 2)  # First passing assessment.
        self.assertIs(candidate, record['iterations'][2])
        self.assertTrue(candidate['green_pass'])
        self.assertFalse(record['iterations'][1]['green_pass'])
        self.assertEqual(path, self.root / 'candidate_2/pool.json')
        self.assertEqual([len(pool) for pool in pools], [3, 3])
        self.assertEqual([pool.tolist() for pool in pools], c.read(path)['pools'])
        self.assertEqual(reads, [Path(candidate['ap_reference']), path])
        self.assertEqual((record, training), before)
        self.assertEqual({path: path.read_bytes() for path in files}, files)

    def test_stopping_ap_can_be_worse_than_historical_minimum(self):
        record, training = stopping_fixture(self.root)
        stop = record['iterations'][2]
        stop.update(ap_incumbent_mean=9., restricted_mean=9., green_gap=0.)
        for sample in stop['samples']:
            sample['incumbent']['cost'] = 9.
        put(Path(stop['ap_reference']), stop)
        candidate, _, _ = s.stopping_pool(record, training)
        self.assertEqual(candidate['candidate_id'], 2)
        self.assertGreater(candidate['ap_incumbent_mean'], record['selected_ap_incumbent_mean'])
        self.assertEqual(record['selected_candidate_id'], 1)

    def test_candidate_zero_truncated_prefix_with_consistent_historical_metadata(self):
        record, training = stopping_fixture(self.root)
        candidate = record['iterations'][0]
        candidate.update(restricted_mean=10., green_gap=0., green_pass=True)
        record['iterations'] = [candidate]
        record.update(selected_candidate_id=0, stopping_candidate_id=0,
            selected_checkpoint_sha256=candidate['checkpoint_sha256'],
            selected_state_sha256=candidate['state_sha256'],
            selected_ap_incumbent_mean=candidate['ap_incumbent_mean'],
            selected_restricted_mean=candidate['restricted_mean'],
            selected_ap_reference=candidate['ap_reference'])
        put(Path(candidate['ap_reference']), candidate)
        before = deepcopy(record)
        returned, path, pools = s.stopping_pool(record, training)
        self.assertEqual(returned['candidate_id'], 0)
        self.assertEqual(path, self.root / 'candidate_0/pool.json')
        self.assertEqual([pool.tolist() for pool in pools], [[x] for x in training['centers']])
        self.assertEqual(record, before)

    def test_identical_state_and_checkpoint_bytes_do_not_merge_assessments_or_pools(self):
        record, training = stopping_fixture(self.root)
        earlier, stop = record['iterations'][1:]
        # Equal synthetic hashes represent identical weights/bytes, while the
        # assessment ID, restricted pool and checkpoint location remain distinct.
        for key in ('state_sha256', 'checkpoint_sha256'):
            stop[key] = earlier[key]
        for sample in stop['samples']:
            sample.update(candidate_sha256=stop['state_sha256'],
                          checkpoint_sha256=stop['checkpoint_sha256'])
        put(Path(stop['ap_reference']), stop)
        before = deepcopy(record)
        candidate, path, pools = s.stopping_pool(record, training)
        self.assertEqual(candidate['candidate_id'], 2)
        self.assertEqual(candidate['state_sha256'], earlier['state_sha256'])
        self.assertNotEqual(candidate['pool_sha256'], earlier['pool_sha256'])
        self.assertNotEqual(candidate['checkpoint'], earlier['checkpoint'])
        self.assertEqual(path.parent.name, 'candidate_2')
        self.assertEqual([len(pool) for pool in pools], [3, 3])
        self.assertEqual(record, before)

    def test_failed_incomplete_nonconverged_and_wrong_terminal_status(self):
        original, training = stopping_fixture(self.root)
        for key, value in (('status', 'failed'), ('status', 'running'),
                           ('complete', False), ('converged', False),
                           ('terminal_status', 'iteration_limit')):
            with self.subTest(key=key, value=value):
                record = deepcopy(original)
                record[key] = value
                self.assert_rejected(record, training, 'Incomplete green trajectory')

    def test_no_terminal_fallback_when_no_assessment_passes(self):
        record, training = stopping_fixture(self.root)
        stop = record['iterations'][2]
        stop.update(restricted_mean=5., green_gap=.6, green_pass=False)
        put(Path(stop['ap_reference']), stop)
        # Complete files and an otherwise valid minimum-AP policy cannot save it.
        self.assert_rejected(record, training, 'first strict green stop')

    def test_full_prefix_rejects_unassessed_failed_and_earlier_passing_candidates(self):
        original, training = stopping_fixture(self.root)
        changes = {
            'unassessed': lambda a: a.update(assessed=False),
            'failed_ball': lambda a: a.update(failed_sample_ids=[7]),
            'failed_controller': lambda a: a['samples'][0].update(controller_status='failed'),
            'earlier_pass': lambda a: a.update(restricted_mean=10., green_gap=0., green_pass=True),
        }
        for name, change in changes.items():
            with self.subTest(mutation=name):
                record = deepcopy(original)
                change(record['iterations'][0])
                self.assert_rejected(record, training)
        for i in (1, 2):
            with self.subTest(unassessed_candidate=i):
                record = deepcopy(original)
                record['iterations'][i]['assessed'] = False
                self.assert_rejected(record, training, 'Unassessed')

    def test_prefix_ids_coverage_binding_configuration_and_historical_selector(self):
        original, training = stopping_fixture(self.root)
        changes = {
            'missing_prefix': lambda r: r['iterations'].pop(0),
            'future_candidate': lambda r: r['iterations'].append(deepcopy(r['iterations'][-1])),
            'negative_stop': lambda r: r.update(stopping_candidate_id=-1),
            'bool_stop': lambda r: r.update(stopping_candidate_id=True),
            'float_stop': lambda r: r.update(stopping_candidate_id=2.),
            'nonminimum_selected': lambda r: r.update(selected_candidate_id=2),
            'bool_selected': lambda r: r.update(selected_candidate_id=True),
            'candidate_id': lambda r: r['iterations'][2].update(candidate_id=1),
            'bool_candidate': lambda r: r['iterations'][0].update(candidate_id=False),
            'configuration': lambda r: r['iterations'][0].update(seed=1),
            'missing_sample': lambda r: r['iterations'][0]['samples'].pop(),
            'duplicate_sample': lambda r: r['iterations'][0]['samples'][1].update(sample_id=0),
            'original_id': lambda r: r['iterations'][0]['samples'][0].update(original_sample_id=2),
            'ap_candidate': lambda r: r['iterations'][0]['samples'][0].update(candidate_id=1),
            'state': lambda r: r['iterations'][2]['samples'][0].update(candidate_sha256='tampered'),
            'checkpoint': lambda r: r['iterations'][2]['samples'][0].update(checkpoint_sha256='tampered'),
            'pool_binding': lambda r: r['iterations'][2]['samples'][0].update(pool_sha256='tampered'),
            'center': lambda r: r['iterations'][0]['samples'][0].update(center1=[1.]*3),
            'mean': lambda r: r['iterations'][2].update(ap_incumbent_mean=9.),
            'gap': lambda r: r['iterations'][2].update(green_gap=.5),
            'historical_summary': lambda r: r.update(selected_ap_incumbent_mean=10.),
            'historical_pool_mean': lambda r: r.update(selected_restricted_mean=8.),
            'historical_reference': lambda r: r.update(selected_ap_reference='wrong'),
        }
        for name, change in changes.items():
            with self.subTest(mutation=name):
                record = deepcopy(original)
                change(record)
                self.assert_rejected(record, training)

    def test_nonfinite_ap_values_and_out_of_box_incumbents(self):
        original, training = stopping_fixture(self.root)
        for value in (float('nan'), float('inf'), -float('inf')):
            for field in ('ap_incumbent_mean', 'restricted_mean', 'green_gap', 'cost', 'point'):
                with self.subTest(value=value, field=field):
                    record = deepcopy(original)
                    candidate = record['iterations'][2]
                    if field == 'cost':
                        candidate['samples'][0]['incumbent']['cost'] = value
                    elif field == 'point':
                        candidate['samples'][0]['incumbent']['worst1'][0] = value
                    else:
                        candidate[field] = value
                    self.assert_rejected(record, training)
        record = deepcopy(original)
        record['iterations'][0]['samples'][0]['incumbent']['worst1'][0] = 2.500000002
        self.assert_rejected(record, training, 'membership')

    def test_saved_assessment_tampering_and_schema_changes(self):
        for field, value in (('candidate_id', 1), ('state_sha256', 'wrong'),
                             ('checkpoint_sha256', 'wrong'), ('pool_sha256', 'wrong'),
                             ('ap_incumbent_mean', 99.), ('green_pass', False),
                             ('extra_field', 'unexpected')):
            with self.subTest(field=field):
                record, training = stopping_fixture(self.root)
                stop = record['iterations'][2]
                saved = c.read(stop['ap_reference'])
                saved[field] = value
                put(Path(stop['ap_reference']), saved)
                self.assert_rejected(record, training, 'assessment')

    def test_checkpoint_and_assessment_must_share_pool_directory(self):
        record, training = stopping_fixture(self.root)
        stop = record['iterations'][2]
        stop['checkpoint'] = record['iterations'][1]['checkpoint']
        put(Path(stop['ap_reference']), stop)
        self.assert_rejected(record, training, 'Checkpoint/pool assessment mismatch')

    def test_missing_stop_files_never_fall_back_to_existing_min_ap_files(self):
        for name in ('assessment.json', 'pool.json'):
            with self.subTest(missing=name):
                record, training = stopping_fixture(self.root)
                (self.root / 'candidate_2' / name).unlink()
                self.assertTrue((self.root / 'candidate_1' / name).is_file())
                self.assert_rejected(record, training, error=FileNotFoundError)

    def test_pool_file_hash_then_numeric_digest_are_independent_gates(self):
        record, training = stopping_fixture(self.root)
        stop = record['iterations'][2]
        path = Path(stop['checkpoint']).parent / 'pool.json'
        saved = c.read(path)
        saved['pools'][0][1][0] += .01
        put(path, saved)
        self.assert_rejected(record, training, 'pool file changed')
        # Reseal only this temporary fixture to reach the numeric digest gate.
        stop['pool_file_sha256'] = c.sha(path)
        put(Path(stop['ap_reference']), stop)
        self.assert_rejected(record, training, 'pool digest changed')

    def test_resealed_pool_rejects_ids_centers_membership_shape_and_digest(self):
        changes = {
            'ids_reordered': lambda obj: obj['original_sample_ids'].reverse(),
            'ids_duplicate': lambda obj: obj.update(original_sample_ids=[7, 7]),
            'missing_pool': lambda obj: obj['pools'].pop(),
            'empty_pool': lambda obj: obj['pools'].__setitem__(0, []),
            'shape': lambda obj: obj['pools'][0][1].pop(),
            'center': lambda obj: obj['pools'][0][0].__setitem__(0, 2.1),
            'outside': lambda obj: obj['pools'][0][1].__setitem__(0, 2.500000002),
            'digest': lambda obj: obj.update(pool_sha256='tampered'),
        }
        for name, change in changes.items():
            with self.subTest(mutation=name):
                record, training = stopping_fixture(self.root)
                stop = record['iterations'][2]
                path = Path(stop['checkpoint']).parent / 'pool.json'
                saved = c.read(path)
                change(saved)
                put(path, saved)
                stop['pool_file_sha256'] = c.sha(path)
                put(Path(stop['ap_reference']), stop)
                self.assert_rejected(record, training)

    def test_nonfinite_pool_coordinates_rejected_with_injected_fixture_reader(self):
        record, training = stopping_fixture(self.root)
        stop = record['iterations'][2]
        path = Path(stop['checkpoint']).parent / 'pool.json'
        for value in (float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value):
                saved = c.read(path)
                saved['pools'][0][1][0] = value

                def read(source):
                    return saved if Path(source) == path else c.read(source)

                # Parent put disallows NaN JSON; inject only the decoded pool,
                # keeping real validators and the temporary byte hash intact.
                with self.assertRaisesRegex(ValueError, 'Invalid pool'):
                    s.stopping_pool(record, training, read)


class GridTests(unittest.TestCase):
    def test_exact_grid_and_order_independence_without_mutation(self):
        jobs = grid_fixture()
        before = deepcopy(jobs)
        green = s.validate_grid(jobs)
        self.assertEqual(len(jobs), 750)
        self.assertEqual(len(green), 270)
        self.assertEqual(green, [j for j in jobs if j['method'] == 'green'])
        self.assertEqual(s.validate_grid(list(reversed(jobs))), list(reversed(green)))
        self.assertEqual(jobs, before)

    def test_incomplete_duplicate_and_nonstopping_grid_rejected(self):
        changes = {
            'missing_green': lambda jobs: jobs.pop(0),
            'missing_legacy': lambda jobs: jobs.pop(),
            'extra_job': lambda jobs: jobs.append(deepcopy(jobs[0])),
            'duplicate_id': lambda jobs: jobs[-1].update(id=jobs[0]['id']),
            'duplicate_green_key': lambda jobs: jobs[1].update(
                Ntotal=jobs[0]['Ntotal'], seed=jobs[0]['seed'], radius=jobs[0]['radius']),
            'wrong_seed': lambda jobs: jobs[0].update(seed=10),
            'wrong_radius': lambda jobs: jobs[0].update(radius=.123),
            'wrong_method': lambda jobs: jobs[0].update(method='legacy'),
            'minimum_ap_return': lambda jobs: jobs[0].update(selected_candidate_id=1),
            'wrong_rule': lambda jobs: jobs[0].update(return_rule='minimum_ap'),
            'failed_stop': lambda jobs: jobs[0].update(green_pass=False),
            'truthy_not_true': lambda jobs: jobs[0].update(green_pass=1),
        }
        for name, change in changes.items():
            with self.subTest(mutation=name):
                jobs = grid_fixture()
                change(jobs)
                before = deepcopy(jobs)
                with self.assertRaises(ValueError):
                    s.validate_grid(jobs)
                self.assertEqual(jobs, before)


class UpdatePlotTests(SyntheticOnly):
    def setUp(self):
        super().setUp()
        manifest, rows = plot_fixture(self.root)
        # Expand the parent's four-radius analytic legacy fixture to 16 radii
        # so the new fixed 270+480 grid gate is exercised, never mocked out.
        existing = {row['radius'] for row in rows if row['method'] == 'legacy'}
        templates = [row for row in rows if row['method'] == 'legacy' and row['radius'] == 0.]
        for template in templates:
            spec = next(job for job in manifest['jobs'] if job['id'] == template['id'])
            for radius in LEGACY_RADII:
                if radius in existing:
                    continue
                identity = f._id('legacy', spec['Ntotal'], radius, spec['seed'])
                checkpoint = '/nonexistent/checkpoints/' + identity
                digest = hashlib.sha256(identity.encode()).hexdigest()
                job = deepcopy(spec)
                job.update(id=identity, radius=radius, checkpoint=checkpoint,
                           checkpoint_sha256=digest, state_sha256=digest)
                manifest['files'][checkpoint] = dict(copy=str(self.root / identity), sha256=digest)
                manifest['jobs'].append(job)
                row = deepcopy(template)
                row.update(job)
                row.update(mean_oos=template['mean_oos'] + radius,
                           finite_pool_mean=template['finite_pool_mean'] + radius)
                rows.append(row)
        self.old = p.build_plot_data(manifest, rows)
        self.manifest = deepcopy(manifest)
        self.manifest.update(seal='unit-only-selection-seal', selection_changes=[dict(
            id=rows[0]['id'], previous_candidate_id=1, stopping_candidate_id=2,
            candidate_changed=True, state_changed=True, pool_changed=True)])
        for job in self.manifest['jobs']:
            if job['method'] == 'green':
                job.update(selected_candidate_id=2, stopping_candidate_id=2,
                           return_rule=s.RETURN_RULE, green_pass=True)
        self.rows = [deepcopy(row) for row in rows if row['method'] == 'green']
        for row in self.rows:
            row.update(selected_candidate_id=2, stopping_candidate_id=2,
                       return_rule=s.RETURN_RULE, green_pass=True)
            # Deliberately distinct analytic observations, not measured results.
            row['mean_oos'] += 100.
            row['finite_pool_mean'] += 200.

    def test_only_green_changes_comparators_exact_and_inputs_unmodified(self):
        before = deepcopy((self.old, self.manifest, self.rows))
        data = runner.update_plot(self.old, self.manifest, self.rows)
        self.assertEqual((self.old, self.manifest, self.rows), before)
        self.assertIsNot(data, self.old)
        self.assertEqual(data, runner.update_plot(self.old, self.manifest, list(reversed(self.rows))))
        for old_panel, panel in zip(self.old['panels'], data['panels']):
            self.assertEqual({k: v for k, v in panel.items() if k != 'green'},
                             {k: v for k, v in old_panel.items() if k != 'green'})
            self.assertNotEqual(panel['green'], old_panel['green'])
            self.assertIsNot(panel['legacy'], old_panel['legacy'])
            green = panel['green']
            self.assertEqual(green['radius'], c.RADII)
            self.assertEqual(green['seeds'], c.SEEDS)
            expected = c.np.asarray([[panel['Ntotal']/100 + seed + radius + 100.
                                     for seed in c.SEEDS] for radius in c.RADII])
            for name, values in (('oos', expected), ('finite_pool', expected + 102.)):
                c.np.testing.assert_allclose(green[name]['per_seed'], values, rtol=0, atol=1e-12)
                c.np.testing.assert_allclose(green[name]['mean'], values.mean(axis=1), rtol=0, atol=1e-12)
                for percentile in (20, 80):
                    c.np.testing.assert_allclose(green[name][f'p{percentile}'],
                        c.np.percentile(values, percentile, axis=1), rtol=0, atol=1e-12)
        changed = {'panels', 'return_rule', 'caption', 'footnote', 'current_revision'}
        self.assertEqual({k: v for k, v in data.items() if k not in changed},
                         {k: v for k, v in self.old.items() if k not in changed})
        p._validate_plot_data(data)
        json.dumps(data, allow_nan=False)

    def test_labels_scope_and_original_provenance_are_explicit(self):
        data = runner.update_plot(self.old, self.manifest, self.rows)
        self.assertEqual(data['return_rule'], s.RETURN_RULE)
        self.assertEqual(data['current_revision'], dict(
            parent_manifest_sha256=s.MANIFEST_SHA, parent_summary_sha256=s.SUMMARY_SHA,
            manifest_sha256=self.manifest['seal'], fresh_green_policies=270,
            cached_legacy_policies=480, manifest_relative_path='../manifest.json',
            comparator_metadata_scope='unchanged September28 parent invocation; cached in this revision',
            selection_changes=self.manifest['selection_changes']))
        self.assertTrue(data['caption'].endswith(self.old['caption']))
        self.assertIn('first passing stopping assessment and its own pool', data['caption'])
        self.assertIn('Only green is freshly replayed in this revision', data['caption'])
        self.assertIn('all other curves are unchanged parent results', data['caption'])
        for text in ('Green: stopping checkpoint', 'blue: unchanged Sept 28 replay',
                     'Raw q2 > 0', 'green 40/160/640', 'historical 50/200/800',
                     'not CI', '265 original + 5 advisory', 'No test-based selection'):
            self.assertIn(text, data['footnote'])
        for panel, old_panel in zip(data['panels'], self.old['panels']):
            self.assertEqual(panel['green']['label'], 'Green Moreau (stopping)')
            provenance = panel['green']['provenance']
            self.assertEqual(provenance['return_rule'], s.RETURN_RULE)
            self.assertIs(provenance['current_invocation'], True)
            self.assertIs(provenance['cached'], False)
            self.assertEqual(provenance['fee_rule'], p.NEURAL_FEE)
            for name in ('legacy', 'saa_knn', 'sro_fa', 'horizontal'):
                self.assertEqual(panel[name], old_panel[name])
            self.assertIn('Clarabel', panel['legacy']['label'])
            self.assertNotIn('current_invocation', panel['legacy']['provenance'])

    def test_missing_extra_duplicate_failed_nonfinite_and_wrong_protocol_rows(self):
        changes = {
            'missing': lambda rows: rows.pop(),
            'extra': lambda rows: rows.append(deepcopy(rows[0])),
            'duplicate_id': lambda rows: rows[1].update(id=rows[0]['id']),
            'duplicate_key': lambda rows: rows[1].update(seed=rows[0]['seed']),
            'failed': lambda rows: rows[0].update(status='failed'),
            'incomplete': lambda rows: rows[0].update(status='running'),
            'nan_oos': lambda rows: rows[0].update(mean_oos=float('nan')),
            'infinite_is': lambda rows: rows[0].update(finite_pool_mean=float('inf')),
            'test_identity': lambda rows: rows[0]['test'].update(seed=1),
            'training_budget': lambda rows: rows[0].update(Ntrain=50),
            'missing_solve': lambda rows: rows[0].update(solves_accepted=10000),
            'nonoptimal_solve': lambda rows: rows[0].update(status_counts={'9': 10040}),
            'wrong_fee': lambda rows: rows[0].update(fee_rule=dict(p.NEURAL_FEE, threshold=1e-6)),
        }
        for name, change in changes.items():
            with self.subTest(mutation=name):
                rows = deepcopy(self.rows)
                change(rows)
                before = json.dumps((self.old, self.manifest, rows), sort_keys=True)
                with self.assertRaises(ValueError):
                    runner.update_plot(self.old, self.manifest, rows)
                self.assertEqual(json.dumps((self.old, self.manifest, rows), sort_keys=True), before)

    def test_nonstopping_manifest_rejected_before_green_plot_replacement(self):
        self.manifest['jobs'][0]['selected_candidate_id'] = 1
        with self.assertRaisesRegex(ValueError, 'not its passing stopping assessment'):
            runner.update_plot(self.old, self.manifest, self.rows)


if __name__ == '__main__':
    unittest.main()