"""Solver-free controller fixtures: real Torch costs and frozen numerical gates."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
import ccg_training as ccg

np, torch, core = common.np, common.torch, common.core


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(20.))

    def per_sample_cost(self, x1, x2):
        if len(x1) == 10:
            raise AssertionError('Held-out-sized call forbidden in fixture')
        return x1[:, 0].double() + self.bias.double()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.centers = np.ones((40, 6), dtype=np.float64)
        self.ids = list(range(40))
        self.ap_calls = self.train_calls = 0
        self.restore = False
        self.ap_fault = self.train_fault = None
        model = Policy()
        checkpoint = self.root / 'nominal.pth'
        torch.save(core.weights(model), checkpoint)
        data_path = self.root / 'train.json'
        core.write_json(data_path, dict(centers=self.centers,
            original_sample_ids=self.ids, data_sha256='parent-original-data-hash'))
        self.provenance = self.root / 'provenance.json'
        core.write_json(self.provenance, dict(source_sha256={}))
        self.nominal = dict(seed=0, lr_net=.005, checkpoint=str(checkpoint),
            checkpoint_sha256=core.digest(checkpoint), state_sha256=ccg.state_hash(model),
            train_mean=21., data_sha256='parent-original-data-hash',
            training_data_path=str(data_path), training_data_sha256=core.digest(data_path),
            training_timing=dict(status='completed', wall_time_s=1000., epochs_evaluated=3))
        self.nominal_bytes = checkpoint.read_bytes()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(torch, 'get_num_threads', return_value=1))
        stack.enter_context(patch.object(torch, 'get_num_interop_threads', return_value=1))
        stack.enter_context(patch.object(core, 'model_from_state', side_effect=self.from_state))
        stack.enter_context(patch.object(ccg, 'train_model', side_effect=self.train))
        stack.enter_context(patch.object(ccg.adversary, 'solve_ap', side_effect=self.ap))
        for target, name in ((core, 'split_data'), (core, 'train_model'),
                             (common, 'evaluate'), (ccg.legacy_ccg, 'assess_candidate'),
                             (ccg.legacy_ccg, 'run_configuration'), (ccg.legacy_ccg, 'load_nominal')):
            stack.enter_context(patch.object(target, name, side_effect=AssertionError(f'Forbidden {name}')))

    @staticmethod
    def from_state(state):
        model = Policy()
        model.load_state_dict(state)
        return model

    def train(self, model, loss_fn, lr, folder, record, epochs=3000, patience=300):
        self.assertEqual((lr, epochs, patience), (.005, 3000, 300))
        self.train_calls += 1
        with core.timed(record), torch.no_grad():
            pools = json.loads((folder / 'pool.json').read_text())['pools']
            expected = torch.stack([model.per_sample_cost(
                torch.tensor(p, dtype=torch.float32)[:, :3],
                torch.tensor(p, dtype=torch.float32)[:, 3:]).max() for p in pools]).mean()
            torch.testing.assert_close(loss_fn(model), expected, atol=0., rtol=0.)
            first = float(loss_fn(model))
            if not self.restore:
                model.bias.sub_(.01)
            best = float(loss_fn(model))
            record.update(first_train=first, best_train=best, selected_training_loss=best,
                          epochs_evaluated=1, optimizer_steps=1)
            torch.save(core.weights(model), folder / 'selected.pth')
            if self.train_fault is not None:
                self.train_fault(model, record, folder)
        return model

    def ap(self, model, center1, center2, radius, row):
        sweep, sample_id = divmod(self.ap_calls, 40)
        self.ap_calls += 1
        self.assertEqual(row['sample_id'], sample_id)
        self.assertEqual(center1.dtype, np.float64)
        np.testing.assert_array_equal(center1, self.centers[sample_id, :3])
        np.testing.assert_array_equal(center2, self.centers[sample_id, 3:])
        x1, x2 = center1.copy(), center2.copy()
        x1[0] += min(radius, .5 * (sweep + 1))
        with core.timed(row), torch.no_grad():
            cost = float(model.per_sample_cost(torch.tensor(x1[None], dtype=torch.float32),
                                               torch.tensor(x2[None], dtype=torch.float32))[0])
            row.update(solver_status=2, solcount=1, bound=cost, iterations=1,
                incumbent=dict(cost=cost, worst1=x1.tolist(), worst2=x2.tolist()),
                replay=dict(status='completed', cost=cost, floor_scaled_discrepancy=0.))
        result = (x1, x2, cost)
        return self.ap_fault(result, row, sweep, sample_id) if self.ap_fault else result

    def run_case(self, radius=1.5):
        result = ccg.run_configuration(0, radius, self.root / 'out', self.nominal, self.provenance)
        self.summary = json.loads(Path(result['summary_path']).read_text())
        self.assertEqual(result['summary_sha256'], core.digest(result['summary_path']))
        self.assertNotIn('iterations', result)
        self.assertEqual(result['status'], self.summary['status'])
        json.dumps(result, allow_nan=False)
        self.assertGreaterEqual(result['wall_time_s'], 0.)
        return result

    def test_zero_radius_train_only_nominal_frozen_and_exclusive(self):
        result = self.run_case(0.)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual((self.ap_calls, self.train_calls), (40, 0))
        self.assertEqual(result['selected_candidate_id'], 0)
        self.assertEqual(Path(result['selected_checkpoint']).read_bytes(), self.nominal_bytes)
        self.assertEqual(Path(self.nominal['checkpoint']).read_bytes(), self.nominal_bytes)
        self.assertEqual(self.summary['cached_nominal']['training_timing']['wall_time_s'], 1000.)
        frozen = Path(result['output_directory']) / 'inputs' / 'nominal.pth'
        self.assertEqual(frozen.stat().st_mode & 0o222, 0)
        before = Path(result['summary_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            ccg.run_configuration(0, 0., self.root / 'out', self.nominal, self.provenance)
        self.assertEqual(before, Path(result['summary_path']).read_bytes())

    def test_terminal_selection_cap_and_final_full_ap(self):
        # An irrelevant held-out preference must neither be read nor select c0.
        class Poison:
            def __float__(self):
                raise AssertionError('Held-out preference accessed')
        self.nominal['validation_mean'] = Poison()
        with patch.object(ccg, 'MAX_MP', 2):
            result = self.run_case()
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(result['terminal_status'], 'iteration_cap')
        self.assertFalse(result['converged'])
        self.assertEqual((self.train_calls, self.ap_calls), (2, 120))
        self.assertEqual(result['selected_candidate_id'], 2)
        final = self.summary['iterations'][-1]
        self.assertEqual(Path(result['selected_checkpoint']).read_bytes(), Path(final['checkpoint']).read_bytes())
        self.assertEqual(final['state_sha256'], result['selected_state_sha256'])
        self.assertEqual(Path(self.nominal['checkpoint']).read_bytes(), self.nominal_bytes)
        for candidate in self.summary['iterations']:
            self.assertTrue(candidate['assessed'])
            self.assertEqual(len(candidate['samples']), 40)
            self.assertGreaterEqual(candidate['ap_total_time_s'], 0.)
            self.assertGreaterEqual(candidate['mp_total_time_s'], 0.)
            self.assertNotIn('validation_mean', candidate)
        for candidate in self.summary['iterations'][:-1]:
            training = candidate['mp']
            self.assertLessEqual(training['best_train'], training['first_train'])
            self.assertAlmostEqual(training['selected_training_loss'], training['returned_pool_mean'])

    def test_unchanged_weights_closure_recorded_not_fake_mp_improvement(self):
        self.restore = True
        result = self.run_case(.4)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(result['selected_candidate_id'], 1)
        self.assertEqual(result['terminal_status'], 'separation_satisfied')
        self.assertTrue(result['closure_with_unchangedweights'])
        terminal = self.summary['iterations'][-1]
        self.assertTrue(terminal['repeated_state'])
        self.assertTrue(terminal['unchanged_from_previous'])
        self.assertEqual(self.summary['iterations'][0]['mp']['same_pool_change'], 0.)

    def test_repetition_without_separation_is_stagnation(self):
        self.restore = True
        result = self.run_case()
        self.assertEqual(result['terminal_status'], 'stagnation', result['error'])
        self.assertEqual(result['selected_candidate_id'], 1)
        self.assertFalse(result['converged'])
        self.assertFalse(result['closure_with_unchangedweights'])
        self.assertEqual((self.train_calls, self.ap_calls), (1, 80))

    def test_failed_ap_retains_all40_and_stops_no_selection(self):
        def fault(result, row, sweep, sample_id):
            if sweep == 1 and sample_id == 3:
                row['solver_status'] = 9
                raise RuntimeError('fixture AP failure')
            return result
        self.ap_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual((self.train_calls, self.ap_calls), (1, 80))
        self.assertIsNone(result['selected_checkpoint'])
        self.assertFalse((Path(result['output_directory']) / 'selected.pth').exists())
        candidate = self.summary['iterations'][-1]
        self.assertEqual(candidate['failed_sample_ids'], [3])
        self.assertFalse(candidate['assessed'])
        self.assertEqual(len(candidate['samples']), 40)
        failed = candidate['samples'][3]
        self.assertEqual(failed['solver_status'], 9)
        self.assertIn('fixture AP failure', failed['controller_error']['traceback'])
        for row in candidate['samples']:
            self.assertGreaterEqual(row['controller_wall_time_s'], 0.)
            path = Path(candidate['ap_reference']).parent / f"sample_{row['sample_id']:02d}.json"
            self.assertTrue(path.exists())
        self.assertGreaterEqual(result['mp_total_time_s'], 0.)

    def test_bad_nominal_hash_or_state_stops_before_ap(self):
        self.nominal['state_sha256'] = 'wrong'
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.ap_calls, 0)
        self.assertIn('state mismatch', result['error']['message'])

    def test_training_data_byte_tamper_stops_before_ap(self):
        path = Path(self.nominal['training_data_path'])
        path.write_text(path.read_text() + ' ')
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.ap_calls, 0)
        self.assertIn('hash mismatch', result['error']['message'])

    def test_mp_returned_state_must_match_saved_checkpoint(self):
        def fault(model, record, folder):
            other = Policy()
            torch.save(core.weights(other), folder / 'selected.pth')
        self.train_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.ap_calls, 40)
        self.assertIsNone(result['selected_checkpoint'])
        self.assertIn('state mismatch', result['error']['message'])

    def test_failed_training_preserves_partial_timing(self):
        def fault(model, record, folder):
            raise RuntimeError('fixture training failure')
        self.train_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.ap_calls, 40)
        training = self.summary['iterations'][0]['mp']
        self.assertEqual(training['status'], 'failed')
        self.assertGreaterEqual(training['wall_time_s'], 0.)
        self.assertIn('fixture training failure', training['controller_error']['traceback'])

    def test_bad_first_train_rejected(self):
        def fault(model, record, folder):
            record['first_train'] += 1.
        self.train_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('incoming/first', result['error']['message'])

    def test_duplicate_ids_retain_full_coverage_attempt_and_fail(self):
        def fault(result, row, sweep, sample_id):
            row['sample_id'] = 0
            return result
        self.ap_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual((self.ap_calls, self.train_calls), (40, 0))
        self.assertEqual(result['ap_sweeps_completed'], 0)
        self.assertEqual(len(self.summary['iterations'][0]['samples']), 40)

    def test_nonfinite_ap_bound_retained_not_repaired(self):
        def fault(result, row, sweep, sample_id):
            if sample_id == 5:
                row['bound'] = float('nan')
            return result
        self.ap_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual((self.ap_calls, self.train_calls), (40, 0))
        self.assertEqual(self.summary['iterations'][0]['samples'][5]['bound'], 'NaN')

    def test_final_source_verification_failure_does_not_publish_selection(self):
        with patch.object(ccg, 'verify_sources', side_effect=[None, ValueError('source changed')]):
            result = self.run_case(0.)
        self.assertEqual(result['status'], 'failed')
        self.assertIsNone(result['selected_checkpoint'])
        self.assertFalse((Path(result['output_directory']) / 'selected.pth').exists())
        self.assertIn('source changed', result['error']['traceback'])

    def test_worsened_selected_training_loss_is_rejected(self):
        def fault(model, record, folder):
            record['best_train'] = record['selected_training_loss'] = record['first_train'] + 1.
        self.train_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.ap_calls, 40)
        self.assertIn('worsened incoming', result['error']['message'])

    def test_numerically_violating_near_nonexact_point_is_inserted(self):
        original = ccg.restricted_objective
        def objective(pools):
            evaluate = original(pools)
            singleton = len(pools[0]) == 1
            def shifted(model):
                values = evaluate(model)
                if singleton:
                    values = values.clone()
                    values[0] -= .00009 * values[0]
                return values
            return shifted
        def fault(result, row, sweep, sample_id):
            if sample_id:
                return result
            point = self.centers[0].copy()
            point[0] += 5e-10
            native = result[2] - .5 * (sweep + 1)
            incumbent = native + .00009 * native
            row.update(bound=incumbent, incumbent=dict(cost=incumbent,
                worst1=point[:3].tolist(), worst2=point[3:].tolist()))
            row['replay'].update(cost=native,
                floor_scaled_discrepancy=abs(incumbent-native)/max(1., abs(incumbent), abs(native)))
            return point[:3], point[3:], incumbent
        # Fixture-only allowed signed numerical discrepancies on either side of
        # the pool/incumbent comparison, not an altered production tolerance.
        self.nominal['train_mean'] -= .00009 * 21. / 40
        self.ap_fault = fault
        with patch.object(ccg, 'restricted_objective', side_effect=objective), patch.object(ccg, 'MAX_MP', 1):
            result = self.run_case()
        self.assertEqual(result['status'], 'completed', result['error'])
        first = self.summary['iterations'][0]
        self.assertIn(0, first['added_sample_ids'])
        self.assertTrue(first['samples'][0]['membership']['near_duplicate'])
        self.assertTrue(first['samples'][0]['membership']['float32_collision'])
        pool = json.loads((Path(first['ap_reference']).parent / 'mp' / 'pool.json').read_text())
        self.assertEqual(len(pool['pools'][0]), 2)
        self.assertNotEqual(pool['pools'][0][0], pool['pools'][0][1])


class FrozenGateTests(unittest.TestCase):
    def test_exact_and_near_identity_and_membership(self):
        center = np.ones(6, dtype=np.float64)
        pool = center[None].copy()
        self.assertTrue(ccg.point_membership(center, pool)['exact_duplicate'])
        near = center.copy()
        near[0] += 5e-10
        membership = ccg.point_membership(near, pool)
        self.assertFalse(membership['exact_duplicate'])
        self.assertTrue(membership['near_duplicate'])
        self.assertTrue(membership['float32_collision'])
        self.assertNotEqual(ccg.pool_hash([pool]), ccg.pool_hash([np.vstack([pool, near])]))
        with self.assertRaisesRegex(ValueError, 'Box membership'):
            ccg.check_membership((center + 2.)[None], center, 1.)

    def test_duplicate_residual_and_negative_bound_are_integrity_failures(self):
        center = np.ones(6, dtype=np.float64)
        pool = center[None].copy()
        def row(bound):
            return dict(sample_id=0, status='completed', solver_status=2, solcount=1,
                bound=bound, incumbent=dict(cost=20., worst1=center[:3].tolist(), worst2=center[3:].tolist()),
                replay=dict(status='completed', cost=20., floor_scaled_discrepancy=0.))
        with self.assertRaisesRegex(ValueError, 'Duplicate/colliding'):
            ccg.validate_ap((center[:3], center[3:], 20.), row(21.), 0, center, pool, 1., 20.)
        with self.assertRaisesRegex(ValueError, 'negative bound'):
            ccg.validate_ap((center[:3], center[3:], 20.), row(19.), 0, center, pool, 1., 20.)


if __name__ == '__main__':
    unittest.main()