"""Controller fixtures only: mocked AP/training, real split/artifact verification."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import ccg
import core


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(20.))

    def per_sample_cost(self, x1, x2):
        return x1[:, 0].double() + self.bias.double()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lr_root = self.root / 'lr'
        self.seed_folder = self.lr_root / 'seed_0'
        self.seed_folder.mkdir(parents=True)
        self.data = core.split_data(0)
        self.ap_calls = 0
        self.train_calls = 0
        self.restore = False
        self.ap_fault = None
        self.fixture_nominal()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(ccg, 'LR_SEARCH_ROOT', self.lr_root))
        self.stack.enter_context(patch.object(core, 'model_from_state', side_effect=self.from_state))
        self.stack.enter_context(patch.object(core, 'train_model', side_effect=self.train))
        self.stack.enter_context(patch.object(ccg.adversary, 'solve_ap', side_effect=self.ap))

    def from_state(self, state):
        model = Policy()
        model.load_state_dict(state)
        return model

    def fixture_nominal(self):
        model = Policy()
        val = float(model.per_sample_cost(self.data['val_x1'], self.data['val_x2']).mean().detach())
        train = float(model.per_sample_cost(self.data['train_x1'], self.data['train_x2']).mean().detach())
        core.write_json(self.lr_root / 'provenance.json', dict(source_sha256={}))
        provenance = core.digest(self.lr_root / 'provenance.json')
        core.write_json(self.seed_folder / 'split.json', dict(
            arrays=self.data, raw_data_sha256=ccg.raw_data_hash(self.data)))
        rows = []
        for lr in core.LR_GRID:
            folder = self.seed_folder / f'lr_{lr:g}'
            folder.mkdir()
            torch.save(core.weights(model), folder / 'selected.pth')
            rows.append(dict(status='completed', lr_net=lr,
                             selected_sha256=core.digest(folder / 'selected.pth'),
                             training=dict(status='completed', wall_time_s=1., epochs_evaluated=1),
                             wall_time_s=2., restored_train_evaluation=dict(mean=train, wall_time_s=.1),
                             restored_validation_evaluation=dict(mean=val, wall_time_s=.1)))
        selected = self.seed_folder / 'lr_0.005' / 'selected.pth'
        winner = dict(lr_net=.005, validation_mean=val, train_mean=train,
                      selected_sha256=core.digest(selected), checkpoint=str(selected))
        self.nominal = dict(seed=0, split_seed=100000, status='completed', complete=True,
            configurations=rows, winner=winner, provenance_sha256=provenance,
            raw_data_sha256=ccg.raw_data_hash(self.data), wall_time_s=6.,
            split_artifact_sha256=core.digest(self.seed_folder / 'split.json'),
            saa_initialization=dict(wall_time_s=.5), lr_selection=dict(wall_time_s=.001))
        self.save_nominal()

    def save_nominal(self):
        core.write_json(self.seed_folder / 'summary.json', self.nominal)
        core.write_json(self.lr_root / 'summary.json', dict(status='completed',
            provenance_sha256=self.nominal['provenance_sha256'], seed_results=[self.nominal],
            wall_time_s=8., seed_workers=10))

    def train(self, model, loss, val1, val2, lr, folder, record, epochs, patience):
        self.assertEqual((epochs, patience, lr), (3000, 300, .005))
        torch.testing.assert_close(val1, self.data['val_x1'], atol=0., rtol=0.)
        snapshot = json.loads((folder / 'pool.json').read_text())
        pools = [np.asarray(pool) for pool in snapshot['pools']]
        expected = torch.stack([model.per_sample_cost(torch.tensor(p[:, :3], dtype=torch.float32),
            torch.tensor(p[:, 3:], dtype=torch.float32)).max() for p in pools]).mean()
        torch.testing.assert_close(loss(model), expected, atol=0., rtol=0.)
        self.train_calls += 1
        with core.timed(record), torch.no_grad():
            if not self.restore:
                model.bias.sub_(.01)
            record.update(epochs_evaluated=1, optimizer_steps=1)
            torch.save(core.weights(model), folder / 'selected.pth')
        return model

    def ap(self, model, center1, center2, radius, row):
        self.assertEqual(center1.dtype, np.float64)
        self.assertEqual(center2.dtype, np.float64)
        sample_id = self.ap_calls % 40
        sweep = self.ap_calls // 40
        self.ap_calls += 1
        self.assertEqual(row['sample_id'], sample_id)
        np.testing.assert_array_equal(center1, self.data['xi1'][self.data['train_indices'][sample_id]])
        x1, x2 = center1.copy(), center2.copy()
        x1[0] += min(radius, sweep + 1.)
        with core.timed(row), torch.no_grad():
            a = float(model.per_sample_cost(torch.tensor(x1[None], dtype=torch.float32),
                                            torch.tensor(x2[None], dtype=torch.float32))[0])
            row.update(solver_status=2, solcount=1, bound=a, iterations=1, threads=1,
                       incumbent=dict(cost=a, worst1=x1.tolist(), worst2=x2.tolist()),
                       replay=dict(status='completed', cost=a, floor_scaled_discrepancy=0.))
        result = (x1, x2, a)
        if self.ap_fault is not None:
            return self.ap_fault(result, row, center1, center2, sweep, sample_id)
        return result

    def run_case(self, radius=100.):
        result = ccg.run_configuration(0, radius, self.root / 'out', 'fixture-provenance')
        saved = json.loads(Path(result['summary_path']).read_text())
        self.assertEqual(result, saved)
        json.dumps(result, allow_nan=False)
        self.assertGreaterEqual(result['wall_time_s'], 0.)
        return result

    def test_zero_radius_all40_cached_nominal_and_exclusive_output(self):
        result = self.run_case(0.)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['terminal_status'], 'separation_satisfied')
        self.assertEqual((self.ap_calls, self.train_calls), (40, 0))
        self.assertEqual(result['selected_candidate_id'], 0)
        self.assertEqual(result['selected_separation_status'], 'separation_satisfied')
        self.assertEqual(result['cached_nominal']['whole_lr_search_wall_time_s'], 8.)
        self.assertTrue(result['cached_nominal']['reused'])
        folder = Path(result['output_directory'])
        nominal = folder / 'inputs' / 'nominal.pth'
        self.assertEqual(core.digest(nominal), self.nominal['winner']['selected_sha256'])
        self.assertEqual(nominal.stat().st_mode & 0o222, 0)
        pool = json.loads((folder / 'candidate_00' / 'pool.json').read_text())
        centers = np.concatenate([self.data['xi1'][self.data['train_indices']],
                                  self.data['xi2'][self.data['train_indices']]], axis=1)
        np.testing.assert_array_equal(np.asarray(pool['pools'])[:, 0, :], centers)
        before = Path(result['summary_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            ccg.run_configuration(0, 0., self.root / 'out')
        self.assertEqual(Path(result['summary_path']).read_bytes(), before)

    def test_cap40_has_final41st_ap_and_selected_checkpoint_alignment(self):
        result = self.run_case()
        self.assertEqual(result['terminal_status'], 'iteration_cap', result.get('error'))
        self.assertEqual((self.train_calls, self.ap_calls), (40, 1640))
        self.assertEqual(result['selected_candidate_id'], 40)
        self.assertEqual(result['selected_separation_status'], 'violating')
        self.assertEqual(result['ap_sweeps_completed'], 41)
        state = torch.load(result['selected_checkpoint'], weights_only=True)
        self.assertEqual(ccg.state_hash(self.from_state(state)), result['selected_state_sha256'])
        assessed = json.loads(Path(result['selected_ap_reference']).read_text())
        self.assertEqual(assessed['candidate_id'], 40)
        self.assertEqual(assessed['state_sha256'], result['selected_state_sha256'])
        self.assertEqual(assessed['validation_mean'], result['selected_validation_mean'])
        for candidate in result['iterations']:
            self.assertTrue(candidate['assessed'])
            self.assertEqual(len(candidate['samples']), 40)
            self.assertGreaterEqual(candidate['ap_total_time_s'], 0.)
            self.assertGreaterEqual(candidate['mp_total_time_s'], 0.)
            for row in candidate['samples']:
                self.assertEqual(row['candidate_sha256'], candidate['state_sha256'])
                self.assertEqual(row['pool_sha256'], candidate['pool_sha256'])
        self.assertNotEqual(result['iterations'][0]['pool_sha256'], result['iterations'][1]['pool_sha256'])

    def test_validation_restoration_assessed_then_stagnates_tie_earlier(self):
        self.restore = True
        result = self.run_case()
        self.assertEqual(result['terminal_status'], 'stagnation')
        self.assertEqual((self.train_calls, self.ap_calls), (1, 80))
        self.assertEqual(result['selected_candidate_id'], 0)
        self.assertEqual(result['selected_separation_status'], 'violating')
        self.assertEqual(result['iterations'][0]['state_sha256'], result['iterations'][1]['state_sha256'])
        self.assertNotEqual(result['iterations'][0]['pool_sha256'], result['iterations'][1]['pool_sha256'])

    def test_terminal_separation_not_borrowed_by_earlier_selected_candidate(self):
        self.restore = True
        result = self.run_case(1.)
        self.assertEqual(result['terminal_status'], 'separation_satisfied')
        self.assertEqual(result['selected_candidate_id'], 0)
        self.assertEqual(result['selected_separation_status'], 'violating')
        self.assertEqual(result['iterations'][1]['separation_status'], 'separation_satisfied')

    def test_missing_ap_persists_all40_and_failure_provisional_only(self):
        def fault(result, row, c1, c2, sweep, sample_id):
            return None if sweep == 1 and sample_id == 3 else result
        self.ap_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['complete'])
        self.assertEqual(self.ap_calls, 80)
        self.assertIsNone(result['selected_checkpoint'])
        self.assertEqual(result['provisional_best']['candidate_id'], 0)
        failed = result['iterations'][1]['samples'][3]
        self.assertIn('Missing/malformed AP', failed['controller_error']['message'])
        self.assertGreaterEqual(failed['controller_wall_time_s'], 0.)
        self.assertFalse((Path(result['output_directory']) / 'selected.pth').exists())

    def test_duplicate_id_not_silently_averaged(self):
        def fault(result, row, *args):
            row['sample_id'] = 0
            return result
        self.ap_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.ap_calls, 40)
        self.assertEqual(self.train_calls, 0)
        self.assertEqual(result['ap_sweeps_completed'], 0)

    def test_near_duplicate_retained_in_mp_raw_pool_collision_logged(self):
        def fault(result, row, c1, c2, sweep, sample_id):
            if sample_id != 0:
                return result
            point = np.concatenate([c1, c2])
            point[0] += 5e-10
            with torch.no_grad():
                native = float(Policy().per_sample_cost(torch.tensor(point[None, :3], dtype=torch.float32),
                                                       torch.tensor(point[None, 3:], dtype=torch.float32))[0])
            # Two independent numerical discrepancies, each <= replay tolerance.
            # Their difference exceeds the incumbent/pool ordering threshold.
            a = native + .00009 * native
            row.update(bound=a, incumbent=dict(cost=a, worst1=point[:3].tolist(), worst2=point[3:].tolist()))
            row['replay'].update(cost=native, floor_scaled_discrepancy=abs(a-native)/max(1., abs(a), abs(native)))
            return point[:3], point[3:], a
        # One assessed iteration suffices to inspect real insertion. Lower the
        # fixture's first-pool cost by another allowed numerical discrepancy.
        original = ccg.restricted_objective
        def objective(pools):
            evaluate = original(pools)
            def shifted(model):
                values = evaluate(model)
                if len(pools[0]) == 1:
                    values = values.clone()
                    values[0] -= .00009 * values[0]
                return values
            return shifted
        self.ap_fault = fault
        self.nominal['winner']['train_mean'] -= .00009 * float(
            Policy().per_sample_cost(self.data['train_x1'], self.data['train_x2'])[0].detach()) / 40
        self.save_nominal()
        with patch.object(ccg, 'restricted_objective', side_effect=objective), patch.object(ccg, 'MAX_MP', 1):
            result = self.run_case()
        self.assertEqual(self.train_calls, 1, result.get('error'))
        row = result['iterations'][0]['samples'][0]
        self.assertTrue(row['membership']['near_duplicate'])
        self.assertTrue(row['membership']['float32_collision'])
        pool = json.loads((Path(result['output_directory']) / 'candidate_00' / 'mp' / 'pool.json').read_text())
        self.assertEqual(len(pool['pools'][0]), 2)
        self.assertNotEqual(pool['pools'][0][0], pool['pools'][0][1])

    def test_final_copy_failure_is_persisted_not_completed(self):
        original = ccg._copy
        def fail_selected(source, destination, expected=None):
            if Path(destination).name == 'selected.pth':
                raise OSError('fixture selected copy failed')
            return original(source, destination, expected)
        with patch.object(ccg, '_copy', side_effect=fail_selected):
            result = self.run_case(0.)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['complete'])
        self.assertIsNone(result['selected_checkpoint'])
        self.assertIn('copy failed', result['finalization_error']['message'])

    def test_negative_bound_pool_gap_preserves_raw_sign(self):
        def fault(result, row, *args):
            row['bound'] = -100.
            return result
        self.ap_fault = fault
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        row = result['iterations'][0]['samples'][0]
        self.assertLess(row['raw_bound_minus_pool'], 0.)
        self.assertLess(row['signed_floor_scaled_gap'], 0.)
        self.assertEqual(row['residual'], 0.)
        self.assertEqual(row['controller_status'], 'failed')

    def test_split_hash_tampering_fails_before_any_ap(self):
        split = self.seed_folder / 'split.json'
        split.write_text(split.read_text() + ' ')
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('hash mismatch', result['error']['message'])
        self.assertEqual(self.ap_calls, 0)

    def test_regenerated_indices_verified_even_with_updated_artifact_hash(self):
        path = self.seed_folder / 'split.json'
        split = json.loads(path.read_text())
        split['arrays']['train_indices'].reverse()
        core.write_json(path, split)
        self.nominal['split_artifact_sha256'] = core.digest(path)
        self.save_nominal()
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Regenerated split differs', result['error']['message'])

    def test_nominal_checkpoint_hash_tampering(self):
        with Path(self.nominal['winner']['checkpoint']).open('ab') as stream:
            stream.write(b'changed')
        result = self.run_case()
        self.assertEqual(result['status'], 'failed')
        self.assertIn('hash mismatch', result['error']['message'])


class PoolAndAPContractTests(unittest.TestCase):
    def setUp(self):
        self.center = np.ones(6, dtype=np.float64)
        self.pool = self.center[None, :].copy()

    def row(self, point, a=20., b=20.):
        return dict(sample_id=0, status='completed', solver_status=2, solcount=1, bound=b,
            incumbent=dict(cost=a, worst1=point[:3].tolist(), worst2=point[3:].tolist()),
            replay=dict(status='completed', cost=a, floor_scaled_discrepancy=0.))

    def validate(self, point, a=20., b=20., m=20.):
        row = self.row(point, a, b)
        result = (point[:3], point[3:], a)
        ccg.validate_ap(result, row, 0, self.center, self.pool, 1., m)
        return row

    def test_exact_duplicate_vs_near_float64_and_float32_collision(self):
        exact = self.validate(self.center)
        self.assertTrue(exact['membership']['exact_duplicate'])
        self.assertFalse(exact['membership']['near_duplicate'])
        point = self.center.copy()
        point[0] += 5e-10
        near = self.validate(point, a=20.01, b=20.01)
        self.assertFalse(near['membership']['exact_duplicate'])
        self.assertTrue(near['membership']['near_duplicate'])
        self.assertTrue(near['membership']['float32_collision'])
        self.assertTrue(near['incumbent_violating'])
        # Diagnostic near/collision is not a compression rule: raw points survive.
        extended = np.vstack([self.pool, point])
        self.assertEqual(len(extended), 2)
        self.assertNotEqual(ccg.pool_hash([self.pool]), ccg.pool_hash([extended]))
        self.assertLess(near['membership']['nearest_linf_distance'], 1e-9)
        point[0] = 1. + 1e-8
        collision = self.validate(point)
        self.assertFalse(collision['membership']['near_duplicate'])
        self.assertTrue(collision['membership']['float32_collision'])

    def test_duplicate_and_collision_material_residual_integrity_failures(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate/colliding'):
            self.validate(self.center, a=21., b=21.)
        point = self.center.copy()
        point[0] += 1e-8
        with self.assertRaisesRegex(ValueError, 'Duplicate/colliding'):
            self.validate(point, a=21., b=21.)

    def test_unresolved_bound_without_violating_incumbent(self):
        point = self.center + .1
        with self.assertRaisesRegex(ValueError, 'Unresolved AP bound'):
            self.validate(point, a=20., b=21.)

    def test_numerical_negative_gap_not_material_preserves_raw(self):
        row = self.validate(self.center, a=19.999, b=19.999)
        self.assertLess(row['raw_bound_minus_pool'], 0.)
        self.assertEqual(row['residual'], 0.)

    def test_box_membership_not_clipped_and_nonfinite_bound_rejected(self):
        point = self.center.copy()
        point[0] = 2. + 2e-9
        with self.assertRaisesRegex(ValueError, 'Box membership'):
            self.validate(point)
        self.assertEqual(point[0], 2. + 2e-9)
        with self.assertRaisesRegex(ValueError, 'AP bound'):
            self.validate(self.center, b=float('nan'))

    def test_ragged_group_max_gradients_not_flat_mean_and_frozen_flatten(self):
        pools = [np.array([[1.] * 6, [4.] * 6]), np.array([[8.] * 6])]
        objective = ccg.restricted_objective(pools)
        model = Policy()
        pools[0][1, 0] = 999.  # Closure tensors were flattened once, not rebuilt.
        per_ball = objective(model)
        torch.testing.assert_close(per_ball, torch.tensor([24., 28.], dtype=torch.float64))
        per_ball.mean().backward()
        self.assertEqual(model.bias.grad.item(), 1.)
        self.assertEqual(per_ball.mean().item(), 26.)


if __name__ == '__main__':
    unittest.main()