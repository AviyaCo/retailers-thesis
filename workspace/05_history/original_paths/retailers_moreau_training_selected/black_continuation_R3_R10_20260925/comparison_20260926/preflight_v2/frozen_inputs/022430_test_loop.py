"""Production-loop integration with a deterministic NUMERICAL UNIT FIXTURE.

No campaign imports, solvers, training, experiment data, or research results.
The one-center metadata deliberately bypasses load_job's production size gate.
Only the numerical/source boundary and source re-verification are substituted;
_continue, _candidate, _points, _insert, _train and their gates run unchanged.
All fixture artifacts (including JSON disguised as .pth) live in TemporaryDirectory.
Fake phase times/statuses are fixture values, NOT measured solver evidence.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import continuation as controller


class NumericalFailure(RuntimeError):
    """Sentinel for an explicitly injected native numerical failure."""


class NumericalFixture:
    """Small NumPy backend; no replacement of any continuation orchestration."""

    def __init__(self, root, R, stop, fail=None):
        self.root = root.resolve()
        self.R, self.stop, self.fail = R, stop, fail
        self.centers = np.ones((1, 2 * R), dtype=np.float64)
        self.ids = [7]
        self.training_calls = []
        self.assessment_calls = []
        self.validation_calls = []
        self.verified = []
        self.writes = []
        self.building_prefix = True

    def path(self, path):
        path = Path(path).resolve()
        assert path.is_relative_to(self.root), 'Unit fixture write escaped temporary root'
        return path

    def write_json(self, path, value):
        path = self.path(path)
        path.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
        self.writes.append(path)

    def state_hash(self, model):
        return hashlib.sha256(f'UNIT-ONLY-model-{model.index}'.encode()).hexdigest()

    def checkpoint(self, path, model):
        self.write_json(path, dict(unit_fixture_only=True, index=model.index))
        return controller._digest(path)

    def _copy(self, source, destination, expected):
        assert controller._digest(source) == expected
        destination = self.path(destination)
        with destination.open('xb') as stream:
            stream.write(Path(source).read_bytes())
        assert controller._digest(destination) == expected
        self.writes.append(destination)

    def _verify_checkpoint(self, path, byte_hash, learned_hash):
        assert controller._digest(path) == byte_hash
        saved = controller._read(path)
        assert saved['unit_fixture_only'] is True
        model = SimpleNamespace(index=saved['index'])
        assert self.state_hash(model) == learned_hash
        self.verified.append((str(path), model.index))
        return model

    def pool_hash(self, pools):
        digest = hashlib.sha256()
        for pool in pools:
            assert pool.dtype == np.float64
            digest.update(str(pool.shape).encode())
            digest.update(pool.tobytes())
        return digest.hexdigest()

    def _save_pool(self, path, pools, ids):
        self.write_json(path, dict(unit_fixture_only=True,
            pools=[pool.tolist() for pool in pools], original_sample_ids=ids,
            pool_sha256=self.pool_hash(pools)))
        return controller._digest(path)

    def point(self, index):
        point = self.centers[0].copy()
        point[0] += (index + 1) * .0200000001
        return point

    def expected_pool(self, index):
        # Candidate k has the center plus points from candidates 0,...,k-1.
        return np.vstack([self.centers[0]] + [self.point(i) for i in range(index)])

    def check_membership(self, points, center, radius):
        assert points.dtype == np.float64 and points.shape[1] == 2 * self.R
        assert np.isfinite(points).all()
        violation = max(0., float((np.maximum(0., center - radius) - points).max()),
                        float((points - center - radius).max()))
        assert violation <= 1e-9
        return violation

    def point_membership(self, point, pool):
        distances = np.max(np.abs(pool - point), axis=1)
        exact = np.all(pool == point, axis=1)
        collisions = np.all(pool.astype(np.float32) == point.astype(np.float32), axis=1)
        return dict(exact_duplicate=bool(exact.any()),
            nearest_linf_distance=float(distances.min()),
            near_duplicate=bool(((distances <= 1e-9) & ~exact).any()),
            float32_collision=bool((collisions & ~exact).any()),
            float32_matches=np.flatnonzero(collisions).tolist())

    def costs(self, model, points):
        return 100. - model.index + 1000. * (points[:, 0] - 1.)

    def restricted_objective(self, pools):
        assert all(p.dtype == np.float64 for p in pools)
        return lambda model: np.asarray([self.costs(model, p).max() for p in pools],
                                        dtype=np.float64)

    def _pool_mean(self, model, pools, record):
        values = self.restricted_objective(pools)(model)
        record.update(unit_fixture_only=True, status='completed', wall_time_s=0.,
            mean=float(values.mean()), per_ball_costs=values.tolist(),
            sample_count=len(pools), state_sha256=self.state_hash(model))
        return values

    def metadata(self, n_total, seed, radius, method):
        return dict(unit_fixture_only=True, R=self.R, Ntotal=n_total, Ntrain=1,
            seed=seed, radius=radius, method=method, torch_threads=1, gurobi_threads=1)

    def validate_ap(self, result, row, sample_id, center, pool, radius, pool_cost):
        # Recompute the complete saved-gate schema, rather than echoing saved fields.
        assert row['sample_id'] == sample_id
        assert row['status'] == 'completed' and row['solver_status'] == 2
        assert row['solcount'] >= 1 and row['replay']['status'] == 'completed'
        x1, x2, cost = result
        assert x1.shape == x2.shape == (self.R,)
        point = np.concatenate((x1, x2))
        a, b = row['incumbent']['cost'], row['bound']
        assert cost == a == row['replay']['cost']
        assert row['replay']['floor_scaled_discrepancy'] == 0.
        np.testing.assert_array_equal(x1, row['incumbent']['worst1'])
        np.testing.assert_array_equal(x2, row['incumbent']['worst2'])
        gap, scale = b - pool_cost, max(1., abs(b), abs(pool_cost))
        row.update(pool_cost=pool_cost,
            membership_violation=self.check_membership(point[None, :], center, radius),
            membership=self.point_membership(point, pool), raw_bound_minus_pool=gap,
            signed_floor_scaled_gap=gap / scale, raw_bound_minus_incumbent=b - a,
            raw_incumbent_minus_pool=a - pool_cost, residual=max(0., gap) / scale,
            incumbent_violating=a - pool_cost > 1e-4 * max(1., abs(a), abs(pool_cost)))
        assert b >= a and b >= pool_cost
        if row['residual'] > .01:
            assert row['incumbent_violating']
            assert not row['membership']['exact_duplicate']
            assert not row['membership']['float32_collision']
        if not self.building_prefix:
            self.validation_calls.append(row['candidate_id'])
        return point

    def assess_candidate(self, model, pools, centers, ids, radius, candidate, folder, publish):
        index = candidate['candidate_id']
        assert model.index == index
        assert candidate['Ntrain'] == len(ids) == len(centers) == 1
        assert candidate['state_sha256'] == self.state_hash(model)
        assert candidate['pool_sha256'] == self.pool_hash(pools)
        assert candidate['pool_file_sha256'] == controller._digest(folder / 'pool.json')
        self._verify_checkpoint(candidate['checkpoint'], candidate['checkpoint_sha256'],
                                candidate['state_sha256'])
        np.testing.assert_array_equal(pools[0], self.expected_pool(index))
        if not self.building_prefix:
            self.assessment_calls.append(index)
        candidate.update(assessed=False, samples=[], failed_sample_ids=[],
                         separation_status='unassessed')
        evaluation = candidate['evaluation'] = {}
        values = self._pool_mean(model, pools, evaluation)
        candidate.update(pool_costs=values.tolist(), restricted_mean=evaluation['mean'])
        candidate['ap'] = dict(unit_fixture_only=True, status='running', wall_time_s=0.)
        if not self.building_prefix and self.fail == 'AP':
            candidate['ap']['status'] = 'failed'
            candidate['failed_sample_ids'] = [0]
            self.write_json(folder / 'assessment.json', candidate)
            publish()
            raise NumericalFailure('UNIT native AP failure')
        point = pools[0][-1].copy() if index == self.stop else self.point(index)
        cost = float(self.costs(model, point[None, :])[0])
        row = dict(Ntotal=candidate['Ntotal'], Ntrain=1, seed=candidate['seed'],
            radius=radius, sample_id=0, original_sample_id=ids[0], candidate_id=index,
            candidate_sha256=candidate['state_sha256'],
            checkpoint_sha256=candidate['checkpoint_sha256'], pool_sha256=candidate['pool_sha256'],
            center1=centers[0, :self.R].tolist(), center2=centers[0, self.R:].tolist(),
            status='completed', controller_status='completed', solver_status=2, solcount=1,
            bound=cost, incumbent=dict(cost=cost, worst1=point[:self.R].tolist(),
                                      worst2=point[self.R:].tolist()),
            replay=dict(status='completed', cost=cost, floor_scaled_discrepancy=0.))
        self.validate_ap((point[:self.R], point[self.R:], cost), row, 0, centers[0],
                         pools[0], radius, float(values[0]))
        candidate.update(samples=[row], assessed=True, ap_incumbent_mean=cost,
            ap_bound_mean=cost, ap_total_time_s=0.,
            separation_status='separation_satisfied' if row['residual'] <= .01 else 'violating')
        candidate['ap']['status'] = 'completed'
        self.write_json(folder / 'sample_00.json', row)
        self.write_json(folder / 'assessment.json', candidate)
        publish()

    def train_model(self, model, loss_fn, lr, folder, training, epochs, patience):
        assert epochs == 3000 and patience == 300 and lr == .005
        assert training['preflight_only'] is False
        assert training['incoming_state_sha256'] == self.state_hash(model)
        assert training['mp_iteration'] == training['output_candidate_id'] == model.index + 1
        saved = controller._read(folder / 'pool.json')
        pools = [np.asarray(p, dtype=np.float64) for p in saved['pools']]
        np.testing.assert_array_equal(pools[0], self.expected_pool(model.index + 1))
        assert saved['original_sample_ids'] == self.ids
        assert training['pool_sha256'] == saved['pool_sha256'] == self.pool_hash(pools)
        first = float(loss_fn(model))  # Exercise the actual _train objective closure.
        assert first == float(self.restricted_objective(pools)(model).mean())
        assert first == training['incoming_pool_mean']
        self.training_calls.append(dict(index=model.index, pools=pools, first=first))
        if self.fail == 'MP':
            training.update(status='failed', unit_fixture_only=True)
            self.write_json(folder / 'training.json', training)
            raise NumericalFailure('UNIT native MP failure')
        learned = SimpleNamespace(index=model.index + 1)
        best = float(loss_fn(learned))
        assert best < first - 1e-4
        training.update(unit_fixture_only=True, status='completed', wall_time_s=0.,
            max_epochs=epochs, patience=patience, lr_net=lr, first_train=first,
            best_train=best, selected_training_loss=best, epochs_evaluated=2, best_epoch=2)
        self.checkpoint(folder / 'selected.pth', learned)
        self.write_json(folder / 'training.json', training)
        return learned


class ProductionLoop(unittest.TestCase):
    def fixture(self, root, *, k=1, stop=3, R=3, fail=None):
        backend = NumericalFixture(root, R, stop, fail)
        original = root / 'original_UNIT_ONLY'
        original.mkdir()
        output = root / 'continuation_UNIT_ONLY'
        output.mkdir()
        model = SimpleNamespace(index=0)
        nominal_path = original / 'nominal.pth'
        nominal_hash = backend.checkpoint(nominal_path, model)
        ctx = SimpleNamespace(R=R, np=np, source=backend,
            common=SimpleNamespace(metadata=backend.metadata),
            core=SimpleNamespace(write_json=backend.write_json))
        nominal = dict(checkpoint=str(nominal_path), checkpoint_sha256=nominal_hash,
            state_sha256=backend.state_hash(model), train_mean=100., lr_net=.005)
        state = SimpleNamespace(ctx=ctx, model=model, nominal=nominal,
            job=dict(R=R, Ntotal=200 if R == 3 else 50, seed=0, radius=1.5 if R == 3 else 2.),
            workers=1, centers=backend.centers.copy(), ids=backend.ids.copy(),
            pools=[backend.expected_pool(0)], prefix=[])
        # Construct a complete contiguous saved prefix without loading numerical campaigns.
        # No historical solver claims: these are explicitly marked unit records.
        for index in range(k + 1):
            state.model = SimpleNamespace(index=index)
            state.pools = [backend.expected_pool(index)]
            checkpoint = original / f'state_{index:02d}.pth'
            byte_hash = backend.checkpoint(checkpoint, state.model)
            folder = original / f'candidate_{index:02d}'
            candidate = controller._candidate(state, folder, index, checkpoint, byte_hash, state.pools)
            backend.assess_candidate(state.model, state.pools, state.centers, state.ids,
                state.job['radius'], candidate, folder, lambda: None)
            state.prefix.append(candidate)
        for previous, following in zip(state.prefix, state.prefix[1:]):
            previous['mp'] = dict(unit_fixture_only=True, status='completed',
                mp_iteration=following['candidate_id'], output_candidate_id=following['candidate_id'],
                incoming_state_sha256=previous['state_sha256'],
                returned_state_sha256=following['state_sha256'],
                selected_checkpoint_sha256=following['checkpoint_sha256'],
                selected_training_loss=following['restricted_mean'], pool_sha256=following['pool_sha256'])
            previous['mp_call'] = dict(unit_fixture_only=True, status='completed', wall_time_s=0.)
            backend.write_json(previous['ap_reference'], previous)
        state.source_summary = dict(unit_fixture_only=True, iterations=deepcopy(state.prefix),
            stopping_candidate_id=k if k >= 0 else None, selected_candidate_id=0,
            selected_checkpoint=state.prefix[0]['checkpoint'] if state.prefix else str(nominal_path),
            mp_count=max(0, k))
        backend.write_json(original / 'summary.json', state.source_summary)
        frozen = {p: p.read_bytes() for p in original.rglob('*') if p.is_file()}
        original_prefix = deepcopy(state.prefix)
        record = dict(unit_fixture_only=True, new_iterations=[], new_mp_count=0,
                      mp_count=max(0, k), converged=False)

        def save():
            backend.write_json(output / 'loop_record.json', record)

        backend.building_prefix = False
        backend.writes.clear()
        return SimpleNamespace(backend=backend, state=state, output=output,
            record=record, save=save, frozen=frozen, original_prefix=original_prefix)

    def assert_immutable(self, fixture):
        self.assertEqual(fixture.state.prefix, fixture.original_prefix)
        for path, contents in fixture.frozen.items():
            self.assertEqual(path.read_bytes(), contents, str(path))
        self.assertTrue(all(p.is_relative_to(fixture.output) for p in fixture.backend.writes))

    def run_loop(self, fixture):
        # Source/manifest validation alone is out of scope, not any loop helper.
        with patch.object(controller, '_reverify') as reverify:
            controller._continue(fixture.state, fixture.output, fixture.record, fixture.save)
        self.assertEqual(reverify.call_count, int(fixture.record['converged']))
        if fixture.record['converged']:
            reverify.assert_called_once_with(fixture.state)
        self.assert_immutable(fixture)

    def assert_chain(self, fixture, outgoing_ids):
        backend, record = fixture.backend, fixture.record
        self.assertEqual([c['index'] for c in backend.training_calls], outgoing_ids)
        self.assertEqual(record['new_mp_count'], len(outgoing_ids))
        self.assertEqual(record['mp_count'], max(0, len(fixture.state.prefix) - 1) + len(outgoing_ids))
        candidates = ({record['extension_boundary']['candidate_id']: record['extension_boundary']}
                      if 'extension_boundary' in record else {})
        candidates.update({c['candidate_id']: c for c in record['new_iterations']})
        for index in outgoing_ids:
            before, after = candidates[index], candidates[index + 1]
            mp = before['mp']
            self.assertEqual(before['added_sample_ids'], [0])
            self.assertEqual(mp['incoming_state_sha256'], before['state_sha256'])
            self.assertEqual(mp['returned_state_sha256'], after['state_sha256'])
            self.assertNotEqual(before['state_sha256'], after['state_sha256'])
            self.assertEqual(mp['selected_checkpoint_sha256'], after['checkpoint_sha256'])
            self.assertEqual(mp['pool_sha256'], after['pool_sha256'])
            self.assertEqual(mp['selected_training_loss'], after['restricted_mean'])
            self.assertEqual(mp['first_train'], before['incoming_pool_evaluation']['mean'])
            self.assertEqual(mp['best_train'], before['returned_pool_evaluation']['mean'])
            self.assertEqual(mp['controller_status'], 'completed')
            self.assertEqual(before['mp_call']['status'], 'completed')
            self.assertGreaterEqual(before['mp_call']['wall_time_s'], 0.)
            for phase in ('incoming_pool_evaluation', 'returned_pool_evaluation'):
                self.assertIsInstance(before[phase]['per_ball_costs'], list)
                self.assertEqual(before[phase]['status'], 'completed')
                self.assertIn('wall_time_s', before[phase])
            folder = fixture.output / 'extension_boundary' / f'candidate_{index:02d}' / 'mp'
            self.assertEqual(controller._read(folder / 'controller_training.json'), mp)
            self.assertEqual((folder / 'selected.pth').read_bytes(), Path(after['checkpoint']).read_bytes())
            self.assertEqual(controller._read(after['ap_reference']), after)
            row = after['samples'][0]
            for key in ('Ntotal', 'Ntrain', 'seed', 'radius', 'candidate_id',
                        'checkpoint_sha256', 'pool_sha256'):
                self.assertEqual(row[key], after[key])
            self.assertEqual(row['candidate_sha256'], after['state_sha256'])
            self.assertEqual(row['original_sample_id'], fixture.state.ids[0])
            np.testing.assert_array_equal(row['center1'] + row['center2'], fixture.state.centers[0])
        final_index = outgoing_ids[-1] + 1
        np.testing.assert_array_equal(fixture.state.pools[0], backend.expected_pool(final_index))
        self.assertEqual(fixture.state.pools[0].dtype, np.float64)
        self.assertFalse(np.array_equal(fixture.state.pools[0],
                         fixture.state.pools[0].astype(np.float32).astype(np.float64)))

    def test_resume_last_stop_two_mps_selects_earlier_full_prefix(self):
        with tempfile.TemporaryDirectory(prefix='UNIT_loop_') as tmp:
            f = self.fixture(Path(tmp))
            self.assertEqual(f.state.source_summary['selected_candidate_id'], 0)
            self.assertEqual(f.state.model.index, 1)
            self.run_loop(f)
            self.assert_chain(f, [1, 2])
            self.assertEqual(f.backend.assessment_calls, [2, 3])
            # Candidate 2 is validated both during fake AP and by the real next loop.
            self.assertEqual(f.backend.validation_calls, [1, 2, 2, 3])
            self.assertEqual(f.record['stopping_candidate_id'], 3)
            self.assertEqual(f.record['terminal_status'], 'converged')
            self.assertEqual(f.record['selected_candidate_id'], 0)
            self.assertEqual(f.record['selected_checkpoint'], f.state.prefix[0]['checkpoint'])
            self.assertEqual(f.record['selected_state_sha256'], f.state.prefix[0]['state_sha256'])
            self.assertEqual(f.record['selected_separation_status'], 'violating')
            self.assertEqual(f.record['selected_ap_incumbent_mean'],
                min(c['ap_incumbent_mean'] for c in f.state.prefix + f.record['new_iterations']))
            self.assertEqual(f.state.model.index, 3)
            self.assertNotIn('mp', f.state.prefix[-1])

    def test_total_budget_39_to_40_without_separation_never_41(self):
        self.check_budget(stop=None, expected='iteration_cap')

    def test_total_budget_39_to_40_with_separation_selects_without_41(self):
        self.check_budget(stop=40, expected='converged')

    def check_budget(self, stop, expected):
        with tempfile.TemporaryDirectory(prefix='UNIT_loop_') as tmp:
            f = self.fixture(Path(tmp), k=39, stop=stop)
            self.run_loop(f)
            self.assert_chain(f, [39])
            self.assertEqual(f.record['mp_count'], 40)
            self.assertEqual(f.record['stopping_candidate_id'], 40)
            self.assertEqual(f.record['terminal_status'], expected)
            self.assertEqual(f.backend.assessment_calls, [40])
            self.assertEqual([c['candidate_id'] for c in f.record['new_iterations']], [40])
            self.assertFalse((f.output / 'candidate_41').exists())
            self.assertFalse((f.output / 'extension_boundary' / 'candidate_40').exists())
            self.assertEqual(f.record['converged'], stop == 40)
            if stop == 40:
                self.assertEqual(f.record['selected_candidate_id'], 0)
            else:
                self.assertFalse(any(key.startswith('selected_') for key in f.record))

    def test_fresh_r10_radius2_assesses_zero_then_two_mps(self):
        with tempfile.TemporaryDirectory(prefix='UNIT_loop_') as tmp:
            f = self.fixture(Path(tmp), k=-1, stop=2, R=10)
            self.run_loop(f)
            self.assert_chain(f, [0, 1])
            self.assertEqual(f.state.job['radius'], 2.)
            self.assertEqual(f.backend.assessment_calls, [0, 1, 2])
            self.assertEqual(f.backend.validation_calls, [0, 0, 1, 1, 2])
            self.assertEqual([c['candidate_id'] for c in f.record['new_iterations']], [0, 1, 2])
            self.assertNotIn('extension_boundary', f.record)
            zero = f.record['new_iterations'][0]
            self.assertEqual(zero['checkpoint_sha256'], f.state.nominal['checkpoint_sha256'])
            self.assertEqual(zero['restricted_mean'], f.state.nominal['train_mean'])
            self.assertEqual(f.record['stopping_candidate_id'], 2)
            self.assertEqual(f.record['selected_candidate_id'], 0)
            self.assertEqual(f.state.model.index, 2)

    def test_native_mp_failure_propagates_no_fallback(self):
        self.check_failure('MP')

    def test_native_ap_failure_after_mp_propagates_no_fallback(self):
        self.check_failure('AP')

    def check_failure(self, phase):
        with tempfile.TemporaryDirectory(prefix='UNIT_loop_') as tmp:
            f = self.fixture(Path(tmp), fail=phase)
            with patch.object(controller, '_reverify') as reverify:
                with self.assertRaisesRegex(NumericalFailure, f'UNIT native {phase} failure'):
                    controller._continue(f.state, f.output, f.record, f.save)
            # Direct _continue intentionally raises; _execute status wrapping is not mocked/tested here.
            reverify.assert_not_called()
            self.assert_immutable(f)
            self.assertEqual([c['index'] for c in f.backend.training_calls], [1])
            self.assertFalse(f.record['converged'])
            self.assertFalse(any(key.startswith('selected_') for key in f.record))
            self.assertFalse((f.output / 'candidate_03').exists())
            boundary = f.record['extension_boundary']
            if phase == 'MP':
                self.assertEqual(f.record['new_mp_count'], 0)
                self.assertEqual(f.record['mp_count'], 1)
                self.assertEqual(f.backend.assessment_calls, [])
                self.assertEqual(boundary['mp']['controller_status'], 'failed')
                self.assertEqual(boundary['mp']['controller_error']['type'], 'NumericalFailure')
                self.assertEqual(boundary['mp_call']['status'], 'failed')
                self.assertEqual(f.state.model.index, 1)
                self.assertFalse((f.output / 'candidate_02').exists())
            else:
                self.assertEqual(f.record['new_mp_count'], 1)
                self.assertEqual(f.record['mp_count'], 2)
                self.assertEqual(f.backend.assessment_calls, [2])
                self.assertEqual(boundary['mp']['controller_status'], 'completed')
                failed = f.record['new_iterations'][0]
                self.assertFalse(failed['assessed'])
                self.assertEqual(failed['ap']['status'], 'failed')
                self.assertEqual(failed['failed_sample_ids'], [0])
                self.assertEqual(f.state.model.index, 2)


if __name__ == '__main__':
    unittest.main()