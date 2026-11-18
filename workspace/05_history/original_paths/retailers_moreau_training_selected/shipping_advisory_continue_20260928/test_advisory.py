"""Advisory/retained-guard regressions; fixtures only, no AP solve or training."""
import copy
from fractions import Fraction as F
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import advisory as advisory
from continue_common import ROOT, definitions, np, read, torch


# Extract only hash-bound definitions, never import old test modules or expose
# their TestCase classes to discovery (several assert now-obsolete rejections).
_fixtures = dict(np=np, torch=torch, copy=copy, F=F, unittest=unittest,
                 SimpleNamespace=SimpleNamespace, read=read,
                 REPLAY=advisory.prior.REPLAY,
                 classify_support=advisory.classify_support,
                 finish_replay=advisory.finish_replay)
definitions(_fixtures, ROOT.parent / 'shipping_epsilon_green_R3_20260925/test_epsilon.py',
            ('weights', 'evidence'))
definitions(_fixtures, advisory.prior.REPLAY / 'test_epsilon.py', ('NativeFixture',))
definitions(_fixtures, advisory.OBSERVED / 'test_observed.py', ('ObservedTests',))
weights, evidence = _fixtures['weights'], _fixtures['evidence']
NativeFixture = _fixtures['NativeFixture']


class AdvisoryTests(unittest.TestCase):
    def activation(self, target, ap_q, flags, native_q, *, coords=None):
        """Allow paired coordinates for controller tests without a real model."""
        point = np.zeros(6, dtype=np.float64) if coords is None else np.asarray(coords, dtype=np.float64)
        row = evidence(target, ap_q, flags)
        cost = .1 * sum(flags)
        row['incumbent'].update(cost=cost, worst1=point[:3].tolist(),
                                worst2=point[3:].tolist())
        before = copy.deepcopy(row['incumbent'])
        model = NativeFixture(native_q)
        advisory.validate_activation(model, weights(target), point[:3], point[3:], row)
        advisory.finish_replay(row, cost)
        self.assertEqual(model.calls, 1)
        self.assertEqual(row['incumbent'], before)
        self.assertEqual(row['activation_replay']['native_q'], native_q)
        self.assertEqual(row['activation_replay']['status'], 'completed')
        self.assertFalse(row['activation_replay']['original_box_certificate'])
        self.assertEqual(row['replay_consistency'], 'recorded_only')
        self.assertFalse(row['replay_rejection_enabled'])
        self.assertTrue(row['replay']['raw_hard_cost_unchanged'])
        return row

    def controller_case(self, *, point=None):
        center = np.ones(6, dtype=np.float64)
        point = center + .1 if point is None else point
        row = self.activation([.5, -1., -1.], [.5, 0., 0.], [1., 0., 0.],
                              [.5, 0., 0.], coords=point)
        row.update(sample_id=0, status='completed', solver_status=2,
                   solcount=1, bound=row['incumbent']['cost'])
        result = (np.asarray(row['incumbent']['worst1']),
                  np.asarray(row['incumbent']['worst2']), row['incumbent']['cost'])
        return row, center, result

    def validate(self, row, center, result, *, pool=None, pool_cost=0.):
        return advisory.validate_ap(result, row, 0, center,
                                    center[None, :] if pool is None else pool, .2, pool_cost)

    def test_activation_lower_and_upper_links_are_advisory(self):
        for q, flag in ((advisory.EPSILON / 2, 1.), (.5, 0.)):
            with self.subTest(q=q, flag=flag):
                row = self.activation([q, -1., -1.], [q, 0., 0.],
                                      [flag, 0., 0.], [q, 0., 0.])
                a = row['activation_replay']
                self.assertGreater(a['link_residual'], advisory.PARAMETERS['FeasibilityTol'])
                self.assertFalse(a['activation_link_guard_passed'])
                self.assertFalse(a['activation_link_rejection_enabled'])

    def test_exact_binary_support_mismatch_both_directions_advisory(self):
        for target, q, flag in ((0., advisory.EPSILON, 1.), (.5, 0., 0.)):
            with self.subTest(target=target):
                a = self.activation([target, -1., -1.], [q, 0., 0.],
                                    [flag, 0., 0.], [q, 0., 0.])['activation_replay']
                self.assertFalse(a['exact_support_matches'])
                self.assertFalse(a['exact_support_rejection_enabled'])
                self.assertNotEqual(a['binary_support'], a['exact_support'])

    def test_exact_positive_support_is_not_rounded_to_float_zero(self):
        a = {}
        advisory.classify_support(np.zeros(3, dtype=bool), np.zeros(3),
                                  (F(1, 10**400), F(0), F(0)), np.zeros(3), a)
        self.assertEqual(a['exact_support'], [True, False, False])
        self.assertFalse(a['exact_support_matches'])
        self.assertFalse(a['exact_support_rejection_enabled'])

    def test_ap_positive_epsilon_and_paid_zero_are_advisory(self):
        for target, q in ((-1., 0.), (.5, 0.), (0., advisory.EPSILON),
                          (advisory.EPSILON / 2, advisory.EPSILON / 2)):
            with self.subTest(target=target, q=q):
                row = self.activation([target, -1., -1.], [q, 0., 0.],
                                      [1., 0., 0.], [q, 0., 0.])
                a = row['activation_replay']
                self.assertFalse(a['ap_positive_epsilon_matches'])
                self.assertFalse(a['ap_positive_epsilon_rejection_enabled'])
                self.assertEqual(row['replay']['ap_cost'], .1)
                self.assertEqual(row['replay']['cost'], .1 if q > 0 else 0.)

    def test_valid_epsilon_equality_records_matching_diagnostics(self):
        q = advisory.EPSILON
        a = self.activation([q, -1., -1.], [q, 0., 0.], [1., 0., 0.],
                            [q, 0., 0.])['activation_replay']
        for key in ('activation_link_guard_passed', 'exact_support_matches',
                    'ap_positive_epsilon_matches', 'native_physical_guard_passed'):
            self.assertTrue(a[key], key)
        self.assertEqual(a['native_hard_cost'], .1)

    def test_native_support_and_epsilon_disagreement_remain_advisory(self):
        for target, q, flag, native in ((-1., 0., 0., 1e-15),
                                       (.5, .5, 1., 0.),
                                       (.5, .5, 1., advisory.EPSILON / 2)):
            with self.subTest(native=native):
                row = self.activation([target, -1., -1.], [q, 0., 0.],
                                      [flag, 0., 0.], [native, 0., 0.])
                self.assertEqual(row['activation_replay']['support_check_status'], 'recorded_only')
                self.assertEqual(row['replay']['cost'], .1 if native > 0 else 0.)

    def test_flag_integrality_and_range_remain_hard(self):
        for flag in (.9, -1., 2., -5e-10, 1. + 5e-10):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'Nonbinary'):
                self.activation([.5, -1., -1.], [.5, 0., 0.],
                                [flag, 0., 0.], [.5, 0., 0.])

    def test_nonfinite_ap_support_remains_hard(self):
        for key in ('zship', 'q2', 'mu', 'target', 'lambda'):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(key=key, value=value):
                    row = evidence([.5, -1., -1.], [.5, 0., 0.], [1., 0., 0.])
                    row['incumbent'][key][0] = value
                    model = NativeFixture([.5, 0., 0.])
                    with self.assertRaisesRegex(ValueError, 'Nonfinite AP support'):
                        advisory.validate_activation(model, weights([.5, -1., -1.]),
                                                     np.zeros(3), np.zeros(3), row)
                    self.assertEqual(model.calls, 0)
                    self.assertEqual(row['activation_replay']['status'], 'failed')

    def test_nonfinite_native_shipments_and_costs_remain_hard(self):
        for output in range(3):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(output=output, value=value):
                    row = evidence([.5, -1., -1.], [.5, 0., 0.], [1., 0., 0.])
                    model = NativeFixture([.5, 0., 0.])
                    values = [torch.zeros(1), torch.tensor([.1]), torch.tensor([[.5, 0., 0.]])]
                    values[output].reshape(-1)[0] = value
                    with patch.object(model, 'costs_and_shipments', return_value=tuple(values)), \
                         self.assertRaisesRegex(ValueError, 'Nonfinite native replay'):
                        advisory.validate_activation(model, weights([.5, -1., -1.]),
                                                     np.zeros(3), np.zeros(3), row)
                    self.assertEqual(row['activation_replay']['status'], 'failed')
                    self.assertIn('projection', row['activation_replay'])

    def test_nonfinite_replay_cost_remains_hard(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            for ap, native in ((value, 1.), (1., value)):
                with self.subTest(ap=ap, native=native), self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    advisory.check_replay_cost(ap, dict(native_hard_cost=native), {})

    def test_native_physical_boundary_accepted_without_clipping(self):
        tol = advisory.PHYSICAL_TOLERANCE
        self.assertEqual(tol, 1e-6)
        for q in (-6e-7, -tol, 1. + tol):
            with self.subTest(q=q):
                a = self.activation([.5, -1., -1.], [.5, 0., 0.], [1., 0., 0.],
                                    [q, 0., 0.])['activation_replay']
                self.assertTrue(a['native_physical_guard_passed'])
                self.assertEqual(a['native_physical_tolerance'], tol)
                self.assertEqual(a['native_physical_residual'], max(0., -q, q - 1.))

    def test_native_physical_nextafter_boundary_rejected_even_with_advisory_failures(self):
        tol = advisory.PHYSICAL_TOLERANCE
        for q in (np.nextafter(-tol, -np.inf), np.nextafter(1. + tol, np.inf)):
            with self.subTest(q=q):
                row = evidence([-1.] * 3, [0.] * 3, [1., 0., 0.])
                row['incumbent']['cost'] = .1
                with self.assertRaisesRegex(ValueError, 'physical constraints'):
                    advisory.validate_activation(NativeFixture([q, 0., 0.]), weights([-1.] * 3),
                                                 np.zeros(3), np.zeros(3), row)
                a = row['activation_replay']
                self.assertFalse(a['activation_link_guard_passed'])
                self.assertFalse(a['exact_support_matches'])
                self.assertFalse(a['ap_positive_epsilon_matches'])
                self.assertFalse(a['native_physical_guard_passed'])
                self.assertEqual(a['status'], 'failed')
                self.assertEqual(a['native_q'], [q, 0., 0.])
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    advisory.finish_replay(row, .1)

    def test_solver_tolerances_unchanged(self):
        self.assertEqual(advisory.PARAMETERS, advisory.prior.PARAMETERS)
        self.assertEqual(advisory.PARAMETERS['FeasibilityTol'], 1e-9)
        self.assertEqual(advisory.PARAMETERS['IntFeasTol'], 1e-9)
        self.assertEqual(advisory.EPSILON, 1e-6)

    def test_controller_accepts_advisory_activation_evidence(self):
        row, center, _ = self.controller_case()
        point = center + .1
        bad = self.activation([-1.] * 3, [0.] * 3, [1., 0., 0.], [0.] * 3, coords=point)
        row.update(bad)
        result = (point[:3], point[3:], .1)
        np.testing.assert_array_equal(self.validate(row, center, result), point)
        self.assertEqual(row['replay']['cost'], 0.)
        self.assertEqual(row['incumbent']['cost'], .1)

    def test_controller_nonoptimal_or_incomplete_solver_rejected(self):
        for field, value in (('solver_status', 9), ('solver_status', 13),
                             ('status', 'failed'), ('solcount', 0)):
            with self.subTest(field=field, value=value):
                row, center, result = self.controller_case()
                row[field] = value
                with self.assertRaisesRegex(ValueError, 'Incomplete AP solver'):
                    self.validate(row, center, result)

    def test_controller_nonfinite_values_rejected(self):
        for field in ('incumbent', 'bound', 'replay', 'discrepancy', 'returned'):
            for value in (float('nan'), float('inf'), -float('inf')):
                with self.subTest(field=field, value=value):
                    row, center, result = self.controller_case()
                    if field in ('incumbent', 'replay'):
                        row[field]['cost'] = value
                    elif field == 'discrepancy':
                        row['replay']['floor_scaled_discrepancy'] = value
                    elif field == 'returned':
                        result = (*result[:2], value)
                    else:
                        row['bound'] = value
                    with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                        self.validate(row, center, result)

    def test_controller_membership_preserved(self):
        for value, message in ((1.20001, 'Box membership'),
                               (float('nan'), 'Nonfinite pool/AP'),
                               (float('inf'), 'Nonfinite pool/AP')):
            with self.subTest(value=value):
                row, center, result = self.controller_case()
                result[0][0] = value
                row['incumbent']['worst1'] = result[0].tolist()
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(row, center, result)

    def test_controller_requires_raw_float64_points(self):
        row, center, result = self.controller_case()
        result = (result[0].astype(np.float32), result[1].astype(np.float32), result[2])
        with self.assertRaisesRegex(ValueError, 'raw float64'):
            self.validate(row, center, result)

    def test_controller_incumbent_bound_order_preserved(self):
        row, center, result = self.controller_case()
        row['bound'] = result[2] - .01
        with self.assertRaisesRegex(ValueError, 'negative bound/incumbent'):
            self.validate(row, center, result)

    def test_controller_unresolved_bound_preserved_including_exact_duplicate(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                row, center, result = self.controller_case()
                row['bound'] = .5
                pool = np.concatenate(result[:2])[None, :] if duplicate else center[None, :]
                with self.assertRaisesRegex(ValueError, 'Unresolved AP bound'):
                    self.validate(row, center, result, pool=pool, pool_cost=result[2])
                self.assertEqual(row['membership']['exact_duplicate'], duplicate)

    def test_exact_duplicate_material_residual_is_advisory(self):
        row, center, result = self.controller_case()
        point = np.concatenate(result[:2])
        pool = point[None, :].copy()
        before = pool.copy()
        np.testing.assert_array_equal(self.validate(row, center, result, pool=pool), point)
        self.assertTrue(row['membership']['exact_duplicate'])
        self.assertFalse(row['membership']['float32_collision'])
        self.assertGreater(row['residual'], advisory.SEPARATION_TOLERANCE)
        self.assertTrue(row['exact_duplicate_material_separation'])
        self.assertFalse(row['exact_duplicate_separation_rejection_enabled'])
        np.testing.assert_array_equal(pool, before)
        # Reinsertion/deduplication belongs to the separate controller tests.

    def test_float32_collision_still_rejected_even_alongside_exact_duplicate(self):
        for include_exact in (False, True):
            with self.subTest(include_exact=include_exact):
                row, center, result = self.controller_case()
                point = np.concatenate(result[:2])
                collision = point.copy()
                collision[0] = np.nextafter(collision[0], np.inf)
                self.assertFalse(np.array_equal(collision, point))
                np.testing.assert_array_equal(collision.astype(np.float32), point.astype(np.float32))
                pool = np.vstack((point, collision)) if include_exact else collision[None, :]
                with self.assertRaisesRegex(ValueError, 'Float32-colliding point'):
                    self.validate(row, center, result, pool=pool)
                self.assertEqual(row['membership']['exact_duplicate'], include_exact)
                self.assertTrue(row['membership']['float32_collision'])

    def test_collision_without_material_residual_is_not_newly_rejected(self):
        row, center, result = self.controller_case()
        collision = np.concatenate(result[:2])
        collision[0] = np.nextafter(collision[0], np.inf)
        self.validate(row, center, result, pool=collision[None, :], pool_cost=result[2])
        self.assertTrue(row['membership']['float32_collision'])
        self.assertEqual(row['residual'], 0.)

    def test_saved_native_disagreement_and_negative_pool_gap_recorded_raw(self):
        row, center, result = _fixtures['ObservedTests']().saved()
        raw = row['replay']['cost']
        self.validate(row, center, result, pool_cost=raw)
        self.assertGreater(row['replay']['floor_scaled_discrepancy'], advisory.REPLAY_TOLERANCE)
        self.assertTrue(row['material_negative_bound_pool_gap'])
        self.assertFalse(row['negative_bound_pool_rejection_enabled'])
        self.assertEqual(row['bound_pool_consistency_status'], 'recorded_only')
        self.assertLess(row['raw_bound_minus_pool'], 0.)
        self.assertEqual(row['pool_cost'], raw)
        self.assertEqual(row['replay']['cost'], raw)

    def test_controller_replay_arithmetic_and_evidence_integrity_preserved(self):
        for mutation, message in (
                ('discrepancy', 'discrepancy'), ('cost', 'Changed replay cost evidence'),
                ('support', 'Changed activation evidence'),
                ('binding', 'not bound to AP incumbent'), ('status', 'Incomplete activation')):
            with self.subTest(mutation=mutation):
                row, center, result = _fixtures['ObservedTests']().saved()
                if mutation == 'discrepancy':
                    row['replay']['floor_scaled_discrepancy'] = 0.
                elif mutation == 'cost':
                    row['replay'].update(cost=result[2], floor_scaled_discrepancy=0.)
                elif mutation == 'support':
                    row['activation_replay']['exact_support_matches'] = False
                elif mutation == 'binding':
                    row['activation_replay']['solver_q'] = [0.] * 3
                else:
                    row['activation_replay']['status'] = 'failed'
                with self.assertRaisesRegex(ValueError, message):
                    self.validate(row, center, result, pool_cost=result[2] - .05)


if __name__ == '__main__':
    unittest.main()