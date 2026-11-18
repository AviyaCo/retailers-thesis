"""Analytic/selection regression tests; no test observations or solver calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_validation as v


class ValidationTests(unittest.TestCase):
    def arms(self):
        return [dict(radius=r, mean_validation=cost, status='completed')
                for r, cost in [(0., 20.), (.2, 19.), (.4, 19.)]]

    def test_validation_minimum_and_exact_tie(self):
        # Rule since 2026-11-18: the largest radius within 1e-6 of the minimum validation mean.
        self.assertEqual(v.choose(self.arms(), [0., .2, .4])['radius'], .4)

    def test_tie_tolerance(self):
        rows = self.arms()
        rows[1]['mean_validation'] -= 5e-7
        self.assertEqual(v.choose(rows, [0., .2, .4])['radius'], .4)
        rows[1]['mean_validation'] -= 2e-6
        self.assertEqual(v.choose(rows, [0., .2, .4])['radius'], .2)

    def test_order_independent(self):
        self.assertEqual(v.choose(self.arms()[::-1], [0., .2, .4])['radius'], .4)

    def test_missing_arm_rejected(self):
        with self.assertRaises(ValueError):
            v.choose(self.arms()[:-1], [0., .2, .4])

    def test_duplicate_arm_rejected(self):
        rows = self.arms(); rows[2]['radius'] = .2
        with self.assertRaises(ValueError):
            v.choose(rows, [0., .2, .4])

    def test_failure_not_dropped(self):
        rows = self.arms(); rows[2]['status'] = 'failed'
        with self.assertRaises(ValueError):
            v.choose(rows, [0., .2, .4])

    def test_nonfinite_rejected(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            rows = self.arms(); rows[2]['mean_validation'] = value
            with self.assertRaises(ValueError):
                v.choose(rows, [0., .2, .4])

    def test_blue_does_not_add_zero_radius(self):
        with self.assertRaises(ValueError):
            v.choose(self.arms(), [.2, .4])

    def test_test_cost_is_ignored_in_selection(self):
        rows = self.arms()
        for i, row in enumerate(rows):
            row['test_mean'] = -1e9 if i == 0 else 1e9
        self.assertEqual(v.choose(rows, [0., .2, .4])['radius'], .4)  # .2/.4 tie -> larger (rule of 2026-11-18)

    def test_strict_positive_shipping(self):
        np = v.c.np
        q = np.array([[0., 1e-12, 1e-7], [-1e-12, 0., 1.]])
        x = np.zeros((2, 3)); q1 = np.zeros(3)
        difference = v.c.scenario_costs(2., q1, q, x, x)-v.c.c.scenario_costs(2., q1, q, x, x)
        np.testing.assert_allclose(difference, [.2, .1], rtol=0, atol=1e-15)

    def test_no_test_lookup_without_seal(self):
        with tempfile.TemporaryDirectory() as path, patch.object(v, 'checked_read') as read:
            with self.assertRaises(FileNotFoundError):
                v.attach_tests(Path(path), {})
            read.assert_not_called()

    def test_blue_generator_matches_original_definition(self):
        source = Path('/data/home/aviyac/retailers_with_shipping/select_rho_by_validation_net16_v2.py')
        gen = v.c.definition(source, 'generate_data', dict(np=v.c.np, truncnorm=v.c.c.truncnorm, R=3))
        for a, b in zip(gen(3, 123), v.c.c.gen(3, 123)):
            v.c.np.testing.assert_array_equal(a, b)

    def test_hash_detects_row_order(self):
        a = v.c.np.arange(12.).reshape(4, 3)
        self.assertNotEqual(v.array_hash(a), v.array_hash(a[::-1]))

    def test_input_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'input.json'; p.write_text('{}')
            before = v.c.sha(p); p.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                v.checked_read(p, before)


if __name__ == '__main__':
    unittest.main()