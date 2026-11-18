"""Pure/controller tests; never import campaigns or access experiment state.

Numerical test arrays are unit fixtures only. Do not run these in a real worker.
"""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


PATH = Path(__file__).with_name('continuation.py')
SPEC = importlib.util.spec_from_file_location('_continuation_under_test', PATH)
assert SPEC is not None and SPEC.loader is not None
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def candidate(index=0, bound=100., pool=100., state=None, mean=100.):
    row = dict(sample_id=0, status='completed', controller_status='completed',
               solver_status=2, solcount=1, bound=bound, pool_cost=pool,
               residual=max(0., bound - pool) / max(1., abs(bound), abs(pool)))
    return dict(candidate_id=index, Ntrain=1, assessed=True, samples=[row],
                pool_costs=[pool], state_sha256=str(index) if state is None else state,
                ap_incumbent_mean=mean)


def membership(point, pool):
    """Fixture identity only; production uses the original numerical helper."""
    return dict(exact_duplicate=bool(np.all(pool == point, axis=1).any()))


def context():
    def check(points, center, radius):
        if points.dtype != np.float64 or not np.isfinite(points).all():
            raise ValueError('raw dtype')
        if np.any(points < np.maximum(0., center - radius)) or np.any(points > center + radius):
            raise ValueError('membership')
    return SimpleNamespace(np=np, source=SimpleNamespace(
        point_membership=membership, check_membership=check))


class Decisions(unittest.TestCase):
    def test_inclusive_one_percent_not_mean(self):
        self.assertTrue(controller._decision(candidate(bound=100., pool=99.)))
        self.assertFalse(controller._decision(candidate(bound=100., pool=98.999999, mean=98.999999)))

    def test_all_centers_required(self):
        c = candidate()
        c['Ntrain'] = 2
        with self.assertRaisesRegex(ValueError, 'coverage'):
            controller._decision(c)
        second = candidate(bound=100., pool=90.)['samples'][0]
        second['sample_id'] = 1
        c['samples'].append(second)
        c['pool_costs'].append(90.)
        self.assertFalse(controller._decision(c))

    def test_bad_residual_and_solver_fail(self):
        c = candidate()
        c['samples'][0]['residual'] = .1
        with self.assertRaisesRegex(ValueError, 'formula'):
            controller._decision(c)
        c = candidate()
        c['samples'][0]['solver_status'] = 9
        with self.assertRaisesRegex(ValueError, 'solver'):
            controller._decision(c)

    def test_total_budget_not_extra_budget(self):
        self.assertEqual(controller._next_action(candidate(39, 100., 90.), set()), 'mp')
        self.assertEqual(controller._next_action(candidate(40, 100., 90.), set()), 'iteration_cap')
        self.assertEqual(controller._next_action(candidate(40), {'40'}), 'converged')

    def test_stop_before_repetition(self):
        self.assertEqual(controller._next_action(candidate(state='same'), {'same'}), 'converged')
        self.assertEqual(controller._next_action(candidate(bound=100., pool=90., state='same'),
                                                 {'same'}), 'stagnation')

    def test_seen_excludes_last_only(self):
        prefix = [candidate(0, state='a'), candidate(1, 100., 90., state='b')]
        seen = {c['state_sha256'] for c in prefix[:-1]}
        self.assertEqual(controller._next_action(prefix[-1], seen), 'mp')
        prefix[-1]['state_sha256'] = 'a'
        self.assertEqual(controller._next_action(prefix[-1], seen), 'stagnation')

    def test_full_prefix_earliest_min_including_stop(self):
        prefix = [candidate(0, 100., 90., mean=3.), candidate(1, mean=3.),
                  candidate(2, mean=5.), candidate(3, mean=1.)]
        selected = controller._choose(prefix, 2)
        self.assertIs(selected, prefix[0])
        self.assertFalse(controller._decision(selected))  # selected need not pass
        self.assertIs(controller._choose(prefix, 3), prefix[3])
        prefix[1]['assessed'] = False
        with self.assertRaises(ValueError):
            controller._choose(prefix, 3)


class Insertion(unittest.TestCase):
    def test_raw_float64_near_point_not_merged(self):
        pools = [np.array([[1., 1.]], dtype=np.float64)]
        point = np.array([1. + 1e-10, 1.], dtype=np.float64)
        self.assertTrue(np.array_equal(point.astype(np.float32), pools[0][0].astype(np.float32)))
        row = dict(incumbent={'cost': 2.}, pool_cost=1., incumbent_violating=True,
                   membership=membership(point, pools[0]))
        added = controller._insert(context(), pools, [point], [row], np.array([[1., 1.]]), 1.)
        self.assertEqual(added, [0])
        self.assertEqual(pools[0].dtype, np.float64)
        np.testing.assert_array_equal(pools[0][1], point)

    def test_duplicate_and_nonviolating_are_not_inserted(self):
        pools = [np.array([[1., 1.]])]
        point = pools[0][0].copy()
        row = dict(incumbent={'cost': 2.}, pool_cost=1., incumbent_violating=True,
                   membership=membership(point, pools[0]))
        self.assertEqual(controller._insert(context(), pools, [point], [row],
                                            np.array([[1., 1.]]), 1.), [])
        point = np.array([1.2, 1.])
        row.update(incumbent={'cost': 1.00001}, incumbent_violating=False,
                   membership=membership(point, pools[0]))
        self.assertEqual(controller._insert(context(), pools, [point], [row],
                                            np.array([[1., 1.]]), 1.), [])
        with self.assertRaisesRegex(ValueError, 'float64'):
            controller._insert(context(), pools, [point.astype(np.float32)], [row],
                               np.array([[1., 1.]]), 1.)


class Integrity(unittest.TestCase):
    def test_cached_last_stop_does_not_reassess_or_train(self):
        prefix = [candidate(0, 100., 90., mean=1.), candidate(1, mean=2.)]
        for c in prefix:
            c.update(checkpoint=f'private_fixture_{c["candidate_id"]}',
                     checkpoint_sha256='bytes', ap_reference='fixture_assessment')
        original = deepcopy(prefix)
        src = SimpleNamespace(_verify_checkpoint=Mock(), assess_candidate=Mock(), train_model=Mock())
        state = SimpleNamespace(ctx=SimpleNamespace(source=src), prefix=prefix)
        record = dict(new_iterations=[], converged=False, mp_count=1, new_mp_count=0)
        with patch.object(controller, '_reverify'):
            controller._continue(state, Path('/unused_unit_fixture'), record, lambda: None)
        self.assertTrue(record['converged'])
        self.assertEqual(record['stopping_candidate_id'], 1)
        self.assertEqual(record['selected_candidate_id'], 0)
        self.assertEqual(record['selected_separation_status'], 'violating')
        self.assertEqual(prefix, original)
        src.assess_candidate.assert_not_called()
        src.train_model.assert_not_called()

    def test_capped_continuation_never_selects_or_trains(self):
        prefix = [candidate(i, 100., 90.) for i in range(41)]
        src = SimpleNamespace(assess_candidate=Mock(), train_model=Mock())
        state = SimpleNamespace(ctx=SimpleNamespace(source=src), prefix=prefix)
        record = dict(new_iterations=[], converged=False, mp_count=40, new_mp_count=0)
        controller._continue(state, Path('/unused_unit_fixture'), record, lambda: None)
        self.assertEqual(record['terminal_status'], 'iteration_cap')
        self.assertFalse(record['converged'])
        self.assertNotIn('selected_checkpoint', record)
        src.assess_candidate.assert_not_called()
        src.train_model.assert_not_called()

    def test_only_documented_assessment_timing_difference(self):
        c = candidate()
        c.update(mp_call={'wall_time_s': 12.}, mp_total_time_s=12.)
        disk = deepcopy(c)
        disk['mp_total_time_s'] = 0.
        controller._same_assessment(c, disk)
        self.assertEqual(disk['mp_total_time_s'], 0.)
        disk['ap_incumbent_mean'] += 1.
        with self.assertRaisesRegex(ValueError, 'differs'):
            controller._same_assessment(c, disk)

    def test_native_mp_consistency(self):
        row = dict(status='completed', first_train=10., best_train=9., selected_training_loss=9.)
        controller._mp_checks(row, {'mean': 10.}, {'mean': 9.})
        with self.assertRaises(ValueError):
            controller._mp_checks(row, {'mean': 10.}, {'mean': 8.})

    def test_private_protocol_and_no_historical_flush(self):
        # Inspect only the new controller, not any live experiment/source tree.
        tree = ast.parse(PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, '_flush')
                self.assertNotEqual(node.func.attr, 'run_configuration')
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    text = ast.unparse(target)
                    self.assertNotIn('.PROTOCOL', text)
        before = deepcopy(controller.PROTOCOL)
        c = candidate(40)
        controller._next_action(c, set())
        controller._choose([candidate()], 0)
        self.assertEqual(controller.PROTOCOL, before)
        self.assertIsNone(controller._CONTEXT)

    def test_only_converged_branch_selects(self):
        tree = ast.parse(PATH.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_continue')
        selection_blocks = [n for n in function.body if isinstance(n, ast.If)
                            and ast.unparse(n.test) == "record['converged']"]
        self.assertEqual(len(selection_blocks), 1)
        choices = [n for n in ast.walk(selection_blocks[0]) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == '_choose']
        self.assertEqual(len(choices), 1)


if __name__ == '__main__':
    unittest.main()