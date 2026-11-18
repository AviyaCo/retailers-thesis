"""Smooth training and native strict-positive fee on the SAME returned q2."""
import shipping_core as core
from shipping_core import np, torch, check


def charges(q2):
    return (.1 * (1 - torch.exp(-5 * q2)).sum(dim=1),
            .1 * (q2 > 0).to(q2.dtype).sum(dim=1))


class Model(core.Model):
    def costs_and_shipments(self, x1, x2):
        check(x1.ndim == 2 and x1.shape == x2.shape and x1.shape[1] == 3,
              'Expected paired (batch,3) inputs')
        q10 = torch.nn.functional.softplus(self.Q10)
        q1r = torch.nn.functional.softplus(self.Q1r)
        n = len(x1)
        z = self.fc_layers(torch.cat([x1, q1r.expand(n, -1), q10.expand(n, 1)], dim=1))
        q2, = self.proj_layer(z, q10.expand(n, 1), core._SOURCE_SOLVER_ARGS)
        stage1 = .25 * (q10 + q1r.sum()) + .05 * q10
        t1 = .5 * (x2 + x1 - q2 - q1r) - .05 * q2
        t2 = .05 * (q1r - x1 - x2)
        t3 = .5 * (x1 - q1r) - .05 * x2
        base = stage1 + torch.maximum(torch.maximum(t1, t2), t3).sum(dim=1)
        smooth, hard = charges(q2)
        return base + smooth, base + hard, q2

    def per_sample_cost(self, x1, x2):
        return self.costs_and_shipments(x1, x2)[0]

    @torch.no_grad()
    def hard_shipping_cost(self, x1, x2):
        """Native floating-program diagnostic, NOT a real-arithmetic certificate."""
        return self.costs_and_shipments(x1, x2)[1]


def make_model(q10, q1r):
    torch.manual_seed(0)
    model = Model()
    with torch.no_grad():
        model.Q10.copy_(torch.tensor(core.softplus_inverse(q10), dtype=torch.float32))
        model.Q1r.copy_(torch.tensor(core.softplus_inverse(q1r), dtype=torch.float32))
    # In particular, log(expm1(0))=-inf is a FAILURE, never floored/replaced.
    core._check_weights(model)
    return model


def model_from_state(state):
    torch.manual_seed(0)
    model = Model()
    core._load_learned(model, state)
    return model