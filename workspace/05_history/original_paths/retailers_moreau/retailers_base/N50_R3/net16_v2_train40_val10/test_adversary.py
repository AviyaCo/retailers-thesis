"""One deterministic synthetic AP fixture, not a scientific experiment.

Requires the real Gurobi and native Moreau backends; failures are not skipped,
retuned, or replaced by an alternative solver. The parent runs this test.
"""
import json
import unittest

import core
import numpy as np
import torch

from adversary import PARAMETERS, REPLAY_TOLERANCE, solve_ap


class AdversaryTests(unittest.TestCase):
    def test_positive_radius_native_replay(self):
        model = core.make_model(5., np.array([3., 4., 5.]))
        with torch.no_grad():
            model.fc_layers[4].bias.copy_(torch.tensor([8., 7., 6.]))
        before = core.weights(model)
        center1, center2 = np.array([2., 4., 6.]), np.array([7., 3., 1.])
        record = dict(seed=0, fixture=True, sample_id=0)
        radius = .2
        worst1, worst2, cost = solve_ap(model, center1, center2, radius, record)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['solver_status'], 2)
        for point, center in ((worst1, center1), (worst2, center2)):
            self.assertEqual(point.dtype, np.float64)
            self.assertTrue((point >= np.maximum(0., center - radius) - 1e-9).all())
            self.assertTrue((point <= center + radius + 1e-9).all())
        self.assertTrue(np.isfinite(record['bound']))
        self.assertGreaterEqual(record['bound'] - cost, -1e-4 * max(1., abs(cost)))
        self.assertLessEqual(record['replay']['floor_scaled_discrepancy'], REPLAY_TOLERANCE)
        with torch.no_grad():
            nominal = float(model.per_sample_cost(torch.tensor(center1[None], dtype=torch.float32),
                                                   torch.tensor(center2[None], dtype=torch.float32))[0])
        self.assertGreaterEqual(record['bound'] - nominal, -1e-4 * max(1., abs(nominal)))
        self.assertGreaterEqual(record['wall_time_s'], record['solver_wall_time_s'])
        for name, value in before.items():
            torch.testing.assert_close(core.weights(model)[name], value, atol=0., rtol=0.)
        json.dumps(record, allow_nan=False)

    def test_radius_zero_native_replay_capacity_and_information_boundary(self):
        model = core.make_model(5., np.array([3., 4., 5.]))
        # A prescribed active-capacity fixture, not fitted to observed solver results.
        with torch.no_grad():
            model.fc_layers[4].bias.copy_(torch.tensor([8., 7., 6.]))
        before = core.weights(model)
        center1 = np.array([2., 4., 6.])
        center2 = np.array([7., 3., 1.])
        record = dict(seed=0, fixture=True)
        worst1, worst2, cost = solve_ap(model, center1, center2, 0., record)
        self.assertEqual(record['solver_status'], 2)
        self.assertEqual(record['status'], 'completed')
        self.assertGreater(record['solcount'], 0)
        self.assertTrue(np.isfinite(cost))
        self.assertTrue(np.isfinite(worst1).all() and np.isfinite(worst2).all())
        np.testing.assert_allclose(worst1, center1, atol=1e-9, rtol=0.)
        np.testing.assert_allclose(worst2, center2, atol=1e-9, rtol=0.)
        self.assertEqual(record['parameters'], PARAMETERS)
        self.assertEqual(record['time_limit'], 'Infinity')
        self.assertEqual(record['capacity_buffer'], 0.)
        q10 = float(torch.nn.functional.softplus(model.Q10).item())
        self.assertEqual(record['q10'], q10)
        allocation = np.asarray(record['incumbent']['q2'])
        self.assertTrue(np.isfinite(allocation).all())
        self.assertTrue(np.isfinite(record['incumbent']['target']).all())
        self.assertGreaterEqual(allocation.min(), -1e-9)
        self.assertLessEqual(abs(allocation.sum() - q10), 1e-9)
        self.assertGreater(abs(allocation.sum() - (q10 + 1e-6)), 9e-7)
        self.assertLessEqual(record['output_bound_violation'], 1e-9)
        self.assertLessEqual(record['replay']['floor_scaled_discrepancy'], REPLAY_TOLERANCE)
        self.assertEqual(record['replay']['projection']['statuses'], {1: 1})
        self.assertEqual(record['replay']['projection']['sample_count'], 1)

        network_inputs, allocations = [], []
        input_hook = model.fc_layers.register_forward_pre_hook(
            lambda module, args: network_inputs.append(args[0].detach().clone()))
        output_hook = model.proj_layer.register_forward_hook(
            lambda module, args, output: allocations.append(output[0].detach().clone()))
        try:
            with torch.no_grad():
                x1 = torch.tensor(center1[None, :], dtype=torch.float32)
                x2 = torch.tensor(center2[None, :], dtype=torch.float32)
                nominal = float(model.per_sample_cost(x1, x2).item())
                future_changed = model.per_sample_cost(x1, x2 + 10.)
        finally:
            input_hook.remove()
            output_hook.remove()
        self.assertTrue(bool(torch.isfinite(future_changed).all()))
        torch.testing.assert_close(network_inputs[0], network_inputs[1], atol=0., rtol=0.)
        torch.testing.assert_close(allocations[0], allocations[1], atol=0., rtol=0.)
        np.testing.assert_allclose(allocations[0].numpy()[0], allocation, atol=1e-5, rtol=0.)
        self.assertLessEqual(abs(cost - nominal) / max(1., abs(cost), abs(nominal)), REPLAY_TOLERANCE)
        self.assertEqual(model.proj_layer.drain_summary()['statuses'], {1: 2})
        for name, value in before.items():
            torch.testing.assert_close(core.weights(model)[name], value, atol=0., rtol=0.)
        for key in ('solver_seconds', 'solver_wall_time_s', 'nodecount', 'iterations'):
            self.assertTrue(np.isfinite(record[key]))
            self.assertGreaterEqual(record[key], 0.)
        self.assertGreaterEqual(record['wall_time_s'], record['solver_wall_time_s'])
        self.assertGreaterEqual(record['wall_time_s'], record['replay']['wall_time_s'])
        self.assertIn('started_at_utc', record)
        self.assertIn('ended_at_utc', record)
        json.dumps(record, allow_nan=False)


if __name__ == '__main__':
    unittest.main()