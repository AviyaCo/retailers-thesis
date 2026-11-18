"""Regression: permit only explained zero residues; retain unrelated failures."""
import copy
from fractions import Fraction as F
from types import SimpleNamespace
import unittest
from epsilon_common import np,torch,read,SAVED_FAILURE,PREVIOUS,definitions
from epsilon_adversary import (EPSILON,REPLAY_ZERO_TOLERANCE,ActivationMismatch,
    classify_support,check_replay_cost,validate_activation,finish_replay)

definitions(globals(),PREVIOUS/'test_epsilon.py',('weights','evidence'))


class NativeFixture:
    def __init__(self,q):
        self.q=torch.tensor([q],dtype=torch.float64)
        self.calls=0
        self.proj_layer=SimpleNamespace(drain_summary=lambda:dict(test_fixture=True,calls=self.calls))

    def costs_and_shipments(self,x1,x2):
        self.calls+=1
        return (torch.zeros(1,dtype=self.q.dtype),
                .1*(self.q>0).to(self.q.dtype).sum(dim=1),self.q)


def classification(native=(1.,0.,1e-15),solver=(1.,0.,0.),exact=(F(1),F(0),F(0)),
                   active=(True,False,False)):
    replay={}
    classify_support(np.array(active),np.array(solver),exact,np.array(native),replay)
    return replay


class ReplayTests(unittest.TestCase):
    def test_zero_no_exception(self):
        r=classification(native=(1.,0.,0.))
        self.assertTrue(r['support_matches'])
        self.assertEqual(r['support_check_status'],'passed')

    def test_small_residue_logged_not_consistent(self):
        r=classification()
        self.assertFalse(r['support_matches'])
        self.assertEqual(r['discrepancy_indices_zero_based'],[2])
        self.assertEqual(r['expected_fee_delta'],.1)

    def test_tolerance_equality_accepted(self):
        r=classification(native=(1.,0.,REPLAY_ZERO_TOLERANCE))
        self.assertEqual(r['support_check_status'],'exception_pending_cost_check')

    def test_just_above_tolerance_rejected(self):
        with self.assertRaises(ActivationMismatch):
            classification(native=(1.,0.,np.nextafter(REPLAY_ZERO_TOLERANCE,np.inf)))

    def test_epsilon_sized_residue_not_eligible(self):
        with self.assertRaises(ActivationMismatch):
            classification(native=(1.,0.,EPSILON))

    def test_reverse_mismatch_rejected(self):
        with self.assertRaises(ActivationMismatch):
            classification(native=(0.,0.,0.))

    def test_swapped_support_rejected(self):
        with self.assertRaises(ActivationMismatch):
            classification(native=(0.,1e-15,0.))

    def test_true_exact_positive_not_rounded(self):
        with self.assertRaises(ActivationMismatch):
            classification(exact=(F(1),F(0),F(1,10**400)))

    def test_nonzero_AP_residue_rejected(self):
        with self.assertRaises(ActivationMismatch):
            classification(solver=(1.,0.,1e-16))

    def test_nonfinite_native_rejected(self):
        for value in (float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ActivationMismatch):
                classification(native=(1.,0.,value))

    def test_multiple_residues_counted(self):
        a=classification(native=(1.,1e-15,1e-15))
        a['native_hard_cost']=10.2
        r={}
        check_replay_cost(10.,a,r)
        self.assertEqual(r['expected_fee_delta'],.2)
        self.assertEqual(r['consistency_status'],'accepted_with_replay_tolerance')

    def test_cost_difference_explained_raw_unchanged(self):
        a=classification()
        a['native_hard_cost']=10.100001
        before=copy.deepcopy(a)
        r={}
        check_replay_cost(10.,a,r)
        self.assertEqual(a,before)
        self.assertEqual(r['cost'],10.100001)
        self.assertEqual(r['ap_cost'],10.)
        self.assertEqual(r['consistency_status'],'accepted_with_replay_tolerance')
        self.assertGreater(r['floor_scaled_discrepancy'],1e-4)

    def test_unexplained_cost_difference_rejected(self):
        a=classification()
        a['native_hard_cost']=10.2
        r={}
        with self.assertRaises(ValueError):
            check_replay_cost(10.,a,r)
        self.assertEqual(r['consistency_status'],'failed')

    def test_matching_support_cost_gate_unchanged(self):
        a=classification(native=(1.,0.,0.))
        a['native_hard_cost']=10.1
        with self.assertRaises(ValueError):
            check_replay_cost(10.,a,{})
        a['native_hard_cost']=10.000001
        r={}
        check_replay_cost(10.,a,r)
        self.assertEqual(r['consistency_status'],'passed')

    def test_nonfinite_cost_rejected(self):
        for native,cost in ((float('nan'),10.),(10.1,float('inf'))):
            a=classification();a['native_hard_cost']=native
            with self.assertRaises(ValueError):
                check_replay_cost(cost,a,{})

    def test_saved_failed_case_regression(self):
        record=read(SAVED_FAILURE)
        a=record['activation_replay']
        r=classification(native=a['native_q'],solver=a['solver_q'],
            exact=tuple(F(q) for q in a['exact_q']),active=a['binary_support'])
        r['native_hard_cost']=a['native_hard_cost']
        costcheck={}
        check_replay_cost(record['incumbent']['cost'],r,costcheck)
        self.assertEqual(costcheck['consistency_status'],'accepted_with_replay_tolerance')
        self.assertFalse(r['support_matches'])
        self.assertAlmostEqual(costcheck['unexplained_cost_residual'],-5.024392017161095e-7)

    def test_native_invocation_once_and_raw_fee_retained(self):
        q=[1e-15,0.,0.]; model=NativeFixture(q)
        record=evidence([-1.,-1.,-1.],[0.,0.,0.],[0.,0.,0.])
        record['incumbent']['cost']=0.
        validate_activation(model,weights([-1.,-1.,-1.]),np.zeros(3),np.zeros(3),record)
        finish_replay(record,0.)
        self.assertEqual(model.calls,1)
        self.assertEqual(record['activation_replay']['native_q'],q)
        self.assertEqual(record['replay']['cost'],.1)
        self.assertTrue(record['consistency_exception'])

    def test_physical_failure_after_eligible_classification(self):
        target=[.5,-1.,-1.]
        record=evidence(target,[.5,0.,0.],[1.,0.,0.])
        with self.assertRaises(ValueError):
            validate_activation(NativeFixture([2.,0.,1e-15]),weights(target),np.zeros(3),np.zeros(3),record)
        self.assertEqual(record['activation_replay']['status'],'failed')
        self.assertIn('projection',record['activation_replay'])
        with self.assertRaises(ValueError):
            finish_replay(record,0.)

    def test_active_epsilon_failure_not_bypassed(self):
        target=[.5,-1.,-1.]
        record=evidence(target,[.5,0.,0.],[1.,0.,0.])
        with self.assertRaises(ActivationMismatch):
            validate_activation(NativeFixture([EPSILON/2,0.,1e-15]),weights(target),np.zeros(3),np.zeros(3),record)

    def test_epsilon_equality_still_pays(self):
        target=[EPSILON,-1.,-1.]
        record=evidence(target,[EPSILON,0.,0.],[1.,0.,0.])
        validate_activation(NativeFixture([EPSILON,0.,0.]),weights(target),np.zeros(3),np.zeros(3),record)
        self.assertEqual(record['activation_replay']['native_hard_cost'],.1)

    def test_ghost_fee_still_rejected(self):
        target=[-1.,-1.,-1.]
        record=evidence(target,[0.,0.,0.],[1.,0.,0.])
        with self.assertRaises(ValueError):
            validate_activation(NativeFixture([1e-15,0.,0.]),weights(target),np.zeros(3),np.zeros(3),record)


if __name__=='__main__':
    unittest.main()