"""Analytic resume fixtures only: no native projection, AP solver or training."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import continuation as ccg

core, np, torch = ccg.core, ccg.np, ccg.torch


class ResumeTests(unittest.TestCase):
    def setUp(self):
        # Reuse the parent's policy, checkpoint loader, AP/MP fixtures and
        # temporary artifact setup, but build its controller without imports.
        self.frozen = ModuleType('frozen_shipping_fixture')
        self.frozen.__dict__.update(vars(ccg))
        self.frozen.PROTOCOL = ccg.parent.PROTOCOL
        ccg._load_shipping_definitions(self.frozen.__dict__, {})
        fixture_namespace = dict(globals(), ccg=self.frozen, common=ccg.parent)
        ccg._definitions(fixture_namespace, ccg.PARENT / 'test_live.py',
                         ('Policy', 'ControllerTests'))
        self.fixture = fixture_namespace['ControllerTests']('runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.Policy = fixture_namespace['Policy']
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(ccg, 'verify_sources'))
        stack.enter_context(patch.object(ccg, 'validate_ap', side_effect=self.fixture.validate))
        stack.enter_context(patch.object(ccg, 'train_model', side_effect=self.train))
        stack.enter_context(patch.object(ccg, 'EXPECTED_BOUNDARIES', {(50, 0, .4): 1}))
        ccg.adversary.solve_ap.side_effect = self.ap
        core.write_json(self.fixture.manifest, dict(protocol=ccg.parent.PROTOCOL,
            workers={'ccg': 1}, input_sha256={
                self.fixture.nominal[key]: self.fixture.nominal[hash_key]
                for key, hash_key in (('checkpoint', 'checkpoint_sha256'),
                                      ('training_data_path', 'training_data_sha256'))}))

    def ap(self, model, x1, x2, radius, row):
        result = self.fixture.ap(model, x1, x2, radius, row)
        row['incumbent'].update(worst1=result[0].tolist(), worst2=result[1].tolist())
        return result

    def train(self, model, loss_fn, lr, folder, record, epochs, patience):
        self.fixture.mp_calls += 1
        self.assertEqual((epochs, patience), (3000, 300))
        with core.timed(record), torch.no_grad():
            first = float(loss_fn(model))
            model.bias.sub_(.01)
            best = float(loss_fn(model))
            record.update(first_train=first, best_train=best, selected_training_loss=best,
                          epochs_evaluated=1, optimizer_steps=1)
            torch.save(core.weights(model), folder / 'selected.pth')
        return model

    def parent_failure(self, repeated=False):
        def failed_ap(*args):
            if self.fixture.ap_calls == 40:
                self.fixture.ap_calls += 1
                raise RuntimeError('historical AP failure')
            return self.ap(*args)

        def unchanged(model, loss_fn, lr, folder, record, epochs, patience):
            self.fixture.mp_calls += 1
            with core.timed(record), torch.no_grad():
                loss = float(loss_fn(model))
                record.update(first_train=loss, best_train=loss, selected_training_loss=loss,
                              epochs_evaluated=1, optimizer_steps=1)
                torch.save(core.weights(model), folder / 'selected.pth')
            return model

        with patch.object(ccg.adversary, 'solve_ap', side_effect=failed_ap), \
             patch.object(self.frozen, 'train_model', side_effect=unchanged if repeated else self.train):
            result = self.fixture.run_case(.4)
        self.assertEqual(result['status'], 'failed', result['error'])
        self.assertEqual((self.fixture.ap_calls, self.fixture.mp_calls), (80, 1))
        self.prior_path = Path(result['summary_path'])
        self.prior = ccg.read(self.prior_path)
        self.assertEqual(self.prior['iterations'][1]['repeated_state'], repeated)
        self.nominal = copy.deepcopy(self.fixture.nominal)
        self.nominal.update(resume_summary_path=str(self.prior_path),
                            resume_summary_sha256=core.digest(self.prior_path))
        self.manifest = self.root / 'resume_manifest.json'
        self.bind_inputs()
        self.fixture.ap_calls = self.fixture.mp_calls = 0

    def bind_inputs(self):
        paths = [p for p in self.prior_path.parent.rglob('*') if p.is_file()]
        paths += [Path(self.nominal['checkpoint']), Path(self.nominal['training_data_path'])]
        self.nominal['resume_summary_sha256'] = core.digest(self.prior_path)
        core.write_json(self.manifest, dict(protocol=ccg.PROTOCOL, workers={'ccg': 1},
                        input_sha256={str(p.resolve()): core.digest(p) for p in paths}))

    def rewrite_candidate(self, index):
        candidate = self.prior['iterations'][index]
        core.write_json(candidate['ap_reference'], candidate)
        core.write_json(self.prior_path, self.prior)
        self.bind_inputs()

    def resume(self):
        result = ccg.run_configuration(50, 0, .4, self.root / 'continued',
                                       self.nominal, self.manifest)
        return result, ccg.read(result['summary_path'])

    def assert_rejected(self, text):
        result, record = self.resume()
        self.assertEqual(result['status'], 'failed')
        self.assertIn(text, result['error']['message'])
        self.assertEqual((self.fixture.ap_calls, self.fixture.mp_calls), (0, 0))
        self.assertIsNone(result['selected_checkpoint'])
        self.assertEqual(record['iterations'], [])

    def test_boundary_replay_keeps_prefix_and_smooth_loss(self):
        self.parent_failure()
        hashes = ccg.read(self.manifest)['input_sha256'].copy()
        result, record = self.resume()
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual((self.fixture.ap_calls, self.fixture.mp_calls), (40, 0))
        self.assertEqual((result['cached_mp_count'], result['new_mp_count'], result['mp_count']), (1, 0, 1))
        self.assertEqual(record['max_mp'], 40)
        self.assertEqual(record['iterations'][0], self.prior['iterations'][0])
        resumed = record['iterations'][1]
        self.assertEqual(resumed['candidate_id'], 1)
        self.assertAlmostEqual(resumed['smooth_training_replay']['mean'],
                               self.prior['iterations'][0]['mp']['selected_training_loss'])
        self.assertNotEqual(resumed['restricted_mean'], resumed['smooth_training_replay']['mean'])
        self.assertNotEqual(resumed['ap_reference'], self.prior['iterations'][1]['ap_reference'])
        self.assertNotIn('controller_error', resumed)
        self.assertEqual(record['historical_failed_assessment']['path'],
                         self.prior['iterations'][1]['ap_reference'])
        self.assertTrue(all(core.digest(p) == h for p, h in hashes.items()))
        self.assertEqual(result['summary_sha256'], core.digest(result['summary_path']))

    def test_repeated_state_green_pass_precedes_stagnation_and_earliest_tie(self):
        self.parent_failure(repeated=True)
        result, record = self.resume()
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(result['terminal_status'], 'green_gap_satisfied')
        self.assertTrue(record['iterations'][1]['repeated_state'])
        self.assertTrue(record['iterations'][1]['unchanged_from_previous'])
        self.assertEqual(result['selected_candidate_id'], 0)
        self.assertEqual(core.digest(result['selected_checkpoint']),
                         self.prior['iterations'][0]['checkpoint_sha256'])
        self.assertNotEqual(result['selected_checkpoint'], self.prior['iterations'][0]['checkpoint'])
        self.assertEqual(result['selected_ap_reference'], self.prior['iterations'][0]['ap_reference'])
        self.assertEqual(self.fixture.mp_calls, 0)

    def elevated_ap(self, *args):
        a, b, cost = self.ap(*args)
        row = args[-1]
        row['incumbent']['cost'] = cost + 1.
        row['bound'] = cost + 1.
        return a, b, cost + 1.

    def test_seen_is_restored_not_reset(self):
        self.parent_failure(repeated=True)
        with patch.object(ccg.adversary, 'solve_ap', side_effect=self.elevated_ap):
            result, record = self.resume()
        self.assertEqual(result['status'], 'nonconverged', result['error'])
        self.assertEqual(record['terminal_reason'], 'repeated_assessed_state_without_green_pass')
        self.assertEqual(self.fixture.mp_calls, 0)
        self.assertIsNone(result['selected_checkpoint'])

    def test_cap_is_total_not_a_fresh_budget(self):
        self.parent_failure()
        with patch.object(ccg, 'MAX_MP', 1), \
             patch.object(ccg.adversary, 'solve_ap', side_effect=self.elevated_ap):
            result, record = self.resume()
        self.assertEqual(result['terminal_status'], 'iteration_cap', result['error'])
        self.assertEqual(result['mp_count'], 1)
        self.assertEqual(result['new_mp_count'], 0)
        self.assertEqual(len(record['iterations']), 2)
        self.assertIsNone(result['selected_checkpoint'])

    def test_new_mp_is_added_to_retained_count(self):
        self.parent_failure()

        def next_point(*args):
            if self.fixture.ap_calls < 40:
                a, b, cost = self.elevated_ap(*args)
                a[1] += args[3]
                args[-1]['incumbent']['worst1'] = a.tolist()
                return a, b, cost
            return self.ap(*args)

        with patch.object(ccg.adversary, 'solve_ap', side_effect=next_point):
            result, record = self.resume()
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual((result['mp_count'], result['cached_mp_count'], result['new_mp_count']), (2, 1, 1))
        self.assertEqual((self.fixture.ap_calls, self.fixture.mp_calls), (80, 1))
        self.assertEqual(record['iterations'][1]['mp']['mp_iteration'], 2)
        self.assertEqual(record['iterations'][2]['candidate_id'], 2)

    def test_cached_and_current_timing_are_disjoint(self):
        self.parent_failure()
        result, record = self.resume()
        current = ccg._phase_totals(record['iterations'][1:])
        cached = ccg._phase_totals(self.prior['iterations'][:1])
        for key in ('ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s'):
            self.assertEqual(result[key], current[key])
            self.assertEqual(result[f'cached_prefix_{key}'], cached[key])
            self.assertEqual(result['logical_prefix_totals'][key], cached[key] + current[key])
        self.assertEqual(result['ap_sweeps_attempted'], 1)
        self.assertEqual(result['logical_prefix_totals']['ap_sweeps_attempted'], 2)
        self.assertEqual(result['cached_prior_job_timing']['wall_time_s'], self.prior['wall_time_s'])
        self.assertEqual(result['cached_failed_assessment_timing']['ap_total_time_s'],
                         self.prior['iterations'][1]['ap']['wall_time_s'])
        self.assertEqual(result['wall_time_s'], result['current_invocation_elapsed_time_s'])
        self.assertGreaterEqual(result['wall_time_s'], result['ap_total_time_s'])
        self.assertEqual(result['cached_nominal']['training_timing']['wall_time_s'], 10.)

    def test_redo_failure_keeps_full_new_sweep_and_no_selection(self):
        self.parent_failure()

        def fail(*args):
            if self.fixture.ap_calls == 3:
                self.fixture.ap_calls += 1
                raise RuntimeError('new AP failure')
            return self.ap(*args)

        with patch.object(ccg.adversary, 'solve_ap', side_effect=fail):
            result, record = self.resume()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.fixture.ap_calls, 40)
        self.assertFalse(record['iterations'][1]['assessed'])
        self.assertEqual(result['ap_sweeps_completed'], 0)
        self.assertEqual(record['iterations'][0], self.prior['iterations'][0])
        self.assertGreater(result['ap_total_time_s'], 0)
        self.assertEqual(result['mp_count'], 1)
        self.assertIsNone(result['selected_checkpoint'])

    def test_summary_digest_is_required(self):
        self.parent_failure()
        self.nominal['resume_summary_sha256'] = 'wrong'
        self.assert_rejected('Resume input hash mismatch')

    def test_every_sample_must_be_hash_bound(self):
        self.parent_failure()
        manifest = ccg.read(self.manifest)
        del manifest['input_sha256'][str(self.prior_path.parent / 'candidate_00/sample_00.json')]
        core.write_json(self.manifest, manifest)
        self.assert_rejected('Unbound resume input')

    def test_assessment_mirror_is_required(self):
        self.parent_failure()
        path = self.prior['iterations'][0]['ap_reference']
        mirror = ccg.read(path)
        mirror['restricted_mean'] += 1
        core.write_json(path, mirror)
        self.bind_inputs()
        self.assert_rejected('Assessment mirror differs')

    def test_sample_mirror_is_required(self):
        self.parent_failure()
        path = self.prior_path.parent / 'candidate_00/sample_00.json'
        sample = ccg.read(path)
        sample['incumbent']['cost'] += 1
        core.write_json(path, sample)
        self.bind_inputs()
        self.assert_rejected('Sample mirror differs')

    def test_learned_checkpoint_hash_is_recomputed(self):
        self.parent_failure()
        self.prior['iterations'][0]['state_sha256'] = 'wrong learned state'
        self.rewrite_candidate(0)
        self.assert_rejected('Checkpoint state mismatch')

    def test_numerical_pool_hash_is_recomputed(self):
        self.parent_failure()
        path = self.prior_path.parent / 'candidate_00/pool.json'
        pool = ccg.read(path)
        pool['pools'][0][0][0] += .01
        path.chmod(0o644)
        core.write_json(path, pool)
        self.prior['iterations'][0]['pool_file_sha256'] = core.digest(path)
        self.rewrite_candidate(0)
        self.assert_rejected('Numerical pool hash mismatch')

    def test_selected_mp_chain_is_required(self):
        self.parent_failure()
        self.prior['iterations'][1]['pool_sha256'] = 'wrong next pool'
        self.rewrite_candidate(1)
        self.assert_rejected('Selected MP to next candidate chain mismatch')

    def test_failed_boundary_cannot_have_outgoing_mp(self):
        self.parent_failure()
        self.prior['iterations'][1]['mp'] = {'status': 'running'}
        self.rewrite_candidate(1)
        self.assert_rejected('Failed boundary has outgoing MP activity')

    def test_parent_mp_count_must_match_boundary(self):
        self.parent_failure()
        self.prior['mp_count'] = 0
        core.write_json(self.prior_path, self.prior)
        self.bind_inputs()
        self.assert_rejected('Parent boundary/MP count mismatch')

    def test_cached_smooth_best_guard_is_preserved(self):
        self.parent_failure()
        training = self.prior['iterations'][0]['mp']
        training['selected_training_loss'] += .5
        path = self.prior_path.parent / 'candidate_00/mp/controller_training.json'
        core.write_json(path, training)
        self.rewrite_candidate(0)
        self.assert_rejected('Cached MP smooth best/unchanged guard differs')


if __name__ == '__main__':
    unittest.main()