"""Implementation fixtures, not retailer experiment observations."""
import unittest
import json

import torch
from torch import nn
from baseline import load_baseline


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Linear(1, 1, bias=False)
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(3))
        with torch.no_grad():
            self.fc_layers.weight.fill_(1.)


class CheckpointTests(unittest.TestCase):
    def test_native_settings_are_strict_json(self):
        from native_projection import NativeProjection
        layer = NativeProjection()
        json.dumps(layer.settings, allow_nan=False)
        self.assertEqual(layer.settings["time_limit"], "Infinity (unlimited)")

    def test_native_target_and_capacity_derivatives(self):
        from native_projection import derivative_fixture
        report = derivative_fixture()
        self.assertLess(report["z_gradient_max_error"], 1e-6)
        self.assertLess(report["capacity_gradient_max_error"], 1e-6)

    def test_one_update_keeps_evaluated_state(self):
        baseline = load_baseline()
        model = TinyModel()
        def loss(m):
            return m.fc_layers.weight.square().sum() + m.Q10.square().sum() + m.Q1r.square().sum()
        best, state, epochs, seconds = baseline.train(model, loss, "fixture", epochs=1, log=False)
        self.assertEqual(epochs, 1)
        self.assertGreater(seconds, 0)
        self.assertEqual(best, 1.)
        self.assertEqual(float(loss(model).detach()), best)
        self.assertEqual(float(state["fc_layers.weight"].item()), 1.)


if __name__ == "__main__":
    unittest.main()