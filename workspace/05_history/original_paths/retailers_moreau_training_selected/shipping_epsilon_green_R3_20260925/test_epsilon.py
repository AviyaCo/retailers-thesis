"""Acceptance-rule fixtures; no trained policy or heldout data is accessed."""
import unittest
from epsilon_common import np, torch
from epsilon_adversary import (EPSILON, ActivationMismatch, exact_folded_projection,
                                validate_activation)


def weights(target):
    return dict(W1=np.eye(3),b1=np.zeros(3),W2=np.eye(3),b2=np.zeros(3),
                W3=np.zeros((3,3)),b3=np.asarray(target),Q10=1.)


class NativeFixture:
    def __init__(self,q):
        self.q=torch.tensor([q],dtype=torch.float64)

    def costs_and_shipments(self,x1,x2):
        return (torch.tensor([0.],dtype=self.q.dtype),
            .1*(self.q>0).to(self.q.dtype).sum(dim=1),self.q)


def evidence(target,q,z):
    return dict(incumbent=dict(q2=q,zship=z,target=target,
        mu=[2*(a-b) for a,b in zip(q,target)],**{'lambda':[0.,0.,0.]}))


class ActivationTests(unittest.TestCase):
    def test_zero_no_fee(self):
        target=[-1.,-1.,-1.]
        row=evidence(target,[0.,0.,0.],[0.,0.,0.])
        validate_activation(NativeFixture([0.,0.,0.]),weights(target),np.zeros(3),np.zeros(3),row)
        self.assertTrue(row['activation_replay']['support_matches'])

    def test_equality_pays(self):
        target=[EPSILON,-1.,-1.]
        row=evidence(target,[EPSILON,0.,0.],[1.,0.,0.])
        validate_activation(NativeFixture([EPSILON,0.,0.]),weights(target),np.zeros(3),np.zeros(3),row)
        self.assertEqual(row['activation_replay']['native_hard_cost'],.1)

    def test_paid_zero_rejected(self):
        target=[-1.,-1.,-1.]
        with self.assertRaises(ValueError):
            validate_activation(NativeFixture([0.,0.,0.]),weights(target),np.zeros(3),np.zeros(3),
                                evidence(target,[0.,0.,0.],[1.,0.,0.]))

    def test_tolerance_does_not_allow_false_support(self):
        target=[0.,-1.,-1.]
        # Both modeled and native allocations are positive near epsilon, but
        # exact point replay has zero: aggregate cost is not a support proof.
        with self.assertRaises(ActivationMismatch):
            validate_activation(NativeFixture([EPSILON,0.,0.]),weights(target),np.zeros(3),np.zeros(3),
                                evidence(target,[EPSILON,0.,0.],[1.,0.,0.]))

    def test_native_tiny_positive_not_clipped(self):
        target=[-1.,-1.,-1.]
        row=evidence(target,[0.,0.,0.],[0.,0.,0.])
        with self.assertRaises(ActivationMismatch):
            validate_activation(NativeFixture([2.**-50,0.,0.]),weights(target),np.zeros(3),np.zeros(3),row)
        self.assertEqual(row['activation_replay']['native_q'][0],2.**-50)

    def test_positive_below_epsilon_rejected(self):
        target=[EPSILON/2,-1.,-1.]
        with self.assertRaises(ValueError):
            validate_activation(NativeFixture([EPSILON/2,0.,0.]),weights(target),np.zeros(3),np.zeros(3),
                                evidence(target,[EPSILON/2,0.,0.],[1.,0.,0.]))

    def test_exact_replay_simplex(self):
        _,q=exact_folded_projection(weights([2.,1.,-1.]),np.zeros(3))
        self.assertEqual(tuple(map(float,q)),(1.,0.,0.))


if __name__=='__main__':
    unittest.main()