"""Small synthetic UNIT fixtures; no historical/test data or real solver runs."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src import evaluate as e


class FakeModel:
    """Solver-status fixture, not a numerical projection implementation."""
    def __init__(self, status=2, q=None):
        self.Status = status
        self.q = np.array([.1, .2, .3] if q is None else q, dtype=float)
        self.IterCount = 4.
        self.BarIterCount = 2
        self.Runtime = .001
        self.ObjVal = 1.
        self.Params = SimpleNamespace()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def addVars(self, R, **kwargs):
        class Variable(float):
            @property
            def X(self):
                return float(self)
        return [Variable(x) for x in self.q]

    def addConstr(self, *args, **kwargs):
        pass

    def setObjective(self, *args):
        pass

    def optimize(self):
        pass

    def setParam(self, *args):
        pass

    def start(self):
        pass

    def getParamInfo(self, key):
        return (key, float, 1.)


def fake_gp(model):
    return SimpleNamespace(Model=lambda **kw: model, Env=lambda **kw: FakeModel(),
                           quicksum=sum, GRB=SimpleNamespace(OPTIMAL=2, MINIMIZE=1))


class EvaluationTests(unittest.TestCase):
    def test_original_targets_and_cost_order_both_dimensions(self):
        for R in (3, 10):
            torch.manual_seed(712)
            model = e.Model(R)
            with torch.no_grad():
                model.Q1r.copy_(torch.linspace(-2., 4., R))
            x1 = np.arange(4 * R, dtype=np.float64).reshape(4, R) / 7
            x2 = x1[:, ::-1].copy() * 2.3
            z, q10, q1r = e.targets(model, x1)
            with torch.no_grad():
                expected_q10 = torch.nn.functional.softplus(model.Q10).item()
                expected_q1r = torch.nn.functional.softplus(model.Q1r).numpy()
                features = torch.cat([torch.tensor(x1, dtype=torch.float32),
                    torch.tensor(expected_q1r).unsqueeze(0).expand(4, -1),
                    torch.full((4, 1), expected_q10)], dim=1)
                expected_z = model.fc_layers(features).numpy()
            np.testing.assert_array_equal(z, expected_z)
            self.assertEqual(z.dtype, np.float32)
            self.assertEqual(q1r.dtype, np.float32)
            q2 = np.full((4, R), .01, dtype=np.float64)
            stage1 = .25 * (q10 + q1r.sum()) + .05 * q10
            expected = stage1 + np.maximum(.5 * (x1 + x2 - q1r - q2) - .05 * q2,
                np.maximum(.05 * (q1r - x1 - x2), .5 * (x1 - q1r) - .05 * x2)).sum(axis=1)
            actual = e.scenario_costs(q10, q1r, q2, x1, x2)
            self.assertEqual(actual.dtype, np.float64)
            np.testing.assert_array_equal(actual, expected)

    def test_own_pool_unequal_sizes_not_flat_mean(self):
        mean, maxima = e.finite_pool_cost([1., 9., 2., 3., 4.], [2, 3])
        self.assertEqual(maxima, [9., 4.])
        self.assertEqual(mean, 6.5)
        for sizes in ([], [0, 5], [2, 2], [True, 4], [2., 3]):
            with self.assertRaises(ValueError):
                e.finite_pool_cost([1., 9., 2., 3., 4.], sizes)

    def test_optimal_only_physical_guard_no_clipping(self):
        for status, q, error in ((9, [.1, .2, .3], RuntimeError),
                                  (2, [-.0002, .2, .3], RuntimeError),
                                  (2, [float('nan'), .2, .3], FloatingPointError)):
            diagnostic = {}
            with patch.dict(sys.modules, gurobipy=fake_gp(FakeModel(status, q))):
                with self.assertRaises(error):
                    e.strict_project(np.zeros(3), 1., None, diagnostic)
            self.assertFalse(diagnostic['accepted'])
            self.assertEqual(diagnostic['status'], status)
            self.assertGreaterEqual(diagnostic['wall_time_s'], 0)
        q = [-.00001, .2, .3]  # accepted residual is retained, NOT clipped
        diagnostic = {}
        with patch.dict(sys.modules, gurobipy=fake_gp(FakeModel(q=q))):
            result = e.strict_project(np.zeros(3), 1., None, diagnostic)
        np.testing.assert_array_equal(result, q)
        self.assertTrue(diagnostic['accepted'])

    def test_replay_journals_every_row_and_failure(self):
        for status in (2, 9):
            with tempfile.TemporaryDirectory() as tmp:
                model = e.Model(3)
                before = e.state_hash(model)
                record = dict(role='unit_fixture', seed=0, radius=0., method='green')
                with patch.dict(sys.modules, gurobipy=fake_gp(FakeModel(status))):
                    if status == 2:
                        result = e.evaluate_policy(model, np.ones((3, 3)), np.ones((3, 3)),
                                                   tmp, record, sizes=[1, 2])
                        self.assertEqual(result['status'], 'completed')
                        self.assertEqual(result['solves_accepted'], 3)
                        self.assertEqual(result['iterations'], 12.)
                    else:
                        with self.assertRaises(RuntimeError):
                            e.evaluate_policy(model, np.ones((3, 3)), np.ones((3, 3)), tmp, record)
                        self.assertEqual(record['status'], 'failed')
                        self.assertIsNone(record['mean'])
                rows = [json.loads(line) for line in (Path(tmp) / 'solves.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows), 3 if status == 2 else 1)
                self.assertEqual(record['status_counts'], {str(status): len(rows)})
                self.assertEqual(before, e.state_hash(model))
                self.assertTrue(model.training)
                saved = json.loads((Path(tmp) / 'result.json').read_text())
                self.assertGreaterEqual(saved['wall_time_s'], 0)

    def test_oos_requires_seal_before_any_projection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, gurobipy=fake_gp(FakeModel())):
            with patch.object(e, 'strict_project') as projection, self.assertRaises(ValueError):
                e.evaluate_policy(e.Model(3), np.ones((1, 3)), np.ones((1, 3)), tmp, {'role': 'oos'})
            projection.assert_not_called()


if __name__ == '__main__':
    unittest.main()