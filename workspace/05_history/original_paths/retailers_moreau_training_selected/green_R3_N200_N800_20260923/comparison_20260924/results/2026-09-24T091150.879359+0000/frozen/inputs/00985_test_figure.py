"""Aggregation tests use synthetic costs, never synthetic experiment results."""
import unittest
import numpy as np
from make_figure import finite_pool_cost, pool_digest


class FigureTests(unittest.TestCase):
    def test_mean_of_maxima_not_mean_of_all_points(self):
        costs = np.r_[1., 9., np.full(39, 2.)]
        value, maxima = finite_pool_cost(costs, [2] + [1]*39)
        self.assertEqual(value, (9+39*2)/40)
        self.assertEqual(maxima, [9.] + [2.]*39)

    def test_singletons_equal_empirical_mean(self):
        value, _ = finite_pool_cost(np.arange(40), [1]*40)
        self.assertEqual(value, 19.5)

    def test_nonfinite_or_empty_pool_rejected(self):
        for costs, sizes in (([float('nan')]*40, [1]*40), (np.arange(39), [0]+[1]*39),
                             (np.arange(41), [1]*40)):
            with self.assertRaises(ValueError):
                finite_pool_cost(costs, sizes)

    def test_pool_grouping_and_point_order_in_hash(self):
        a=np.arange(12,dtype=np.float64).reshape(2,6)
        self.assertNotEqual(pool_digest([a]),pool_digest([a[:1],a[1:]]))
        self.assertNotEqual(pool_digest([a]),pool_digest([a[::-1]]))


if __name__ == '__main__':
    unittest.main()