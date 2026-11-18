"""Native disagreement is advisory; solver, physical and signed gates stay hard."""
import copy
from fractions import Fraction as F
import unittest
from observed_common import np,torch,read,REPLAY,previous,definitions
from observed_adversary import (EPSILON,PARAMETERS,classify_support,
    validate_activation,finish_replay,check_replay_cost)
from observed_pool import validate_ap,hard_objective,restricted_objective

definitions(globals(),previous.previous.PREVIOUS/'test_epsilon.py',('weights','evidence'))
from types import SimpleNamespace
definitions(globals(),REPLAY/'test_epsilon.py',('NativeFixture',))


class ObservedTests(unittest.TestCase):
    def activation(self,target,ap_q,flags,native_q):
        record=evidence(target,ap_q,flags)
        model=NativeFixture(native_q)
        validate_activation(model,weights(target),np.zeros(3),np.zeros(3),record)
        finish_replay(record,.1*sum(flags))
        self.assertEqual(model.calls,1)
        self.assertEqual(record['activation_replay']['native_q'],native_q)
        self.assertEqual(record['replay_consistency'],'recorded_only')
        return record

    def saved(self):
        row=copy.deepcopy(read(REPLAY/'ap_gate_v1/N800_radius0.2.json'))
        a=row['activation_replay'];inc=row['incumbent']
        classify_support(np.asarray(inc['zship'])>.5,np.asarray(inc['q2']),
            tuple(F(q) for q in a['exact_q']),np.asarray(a['native_q']),a)
        finish_replay(row,inc['cost'])
        center=np.concatenate((row['center1'],row['center2']))
        result=(np.asarray(inc['worst1']),np.asarray(inc['worst2']),inc['cost'])
        return row,center,result

    def test_native_positive_at_ap_zero_advisory(self):
        for q in (1e-15,1e-6,.1):
            with self.subTest(q=q):
                r=self.activation([-1.]*3,[0.]*3,[0.]*3,[q,0.,0.])
                self.assertFalse(r['activation_replay']['support_matches'])
                self.assertEqual(r['replay']['cost'],.1)

    def test_native_zero_at_active_ap_advisory(self):
        r=self.activation([.5,-1.,-1.],[.5,0.,0.],[1.,0.,0.],[0.,0.,0.])
        self.assertFalse(r['activation_replay']['support_matches'])

    def test_native_epsilon_failure_advisory(self):
        r=self.activation([EPSILON,-1.,-1.],[EPSILON,0.,0.],[1.,0.,0.],[EPSILON/2,0.,0.])
        self.assertEqual(r['activation_replay']['native_epsilon_violations_zero_based'],[0])

    def test_native_negative_beyond_physical_tolerance_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical constraints'):
            self.activation([.5,-1.,-1.],[.5,0.,0.],[1.,0.,0.],[-6e-7,0.,0.])

    def test_native_capacity_violation_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical constraints'):
            self.activation([.5,-1.,-1.],[.5,0.,0.],[1.,0.,0.],[2.,0.,0.])

    def test_native_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite native'):
            self.activation([.5,-1.,-1.],[.5,0.,0.],[1.,0.,0.],[float('nan'),0.,0.])

    def test_ap_epsilon_link_still_required(self):
        with self.assertRaisesRegex(ValueError,'link violation'):
            self.activation([EPSILON/2,-1.,-1.],[EPSILON/2,0.,0.],[1.,0.,0.],[EPSILON/2,0.,0.])

    def test_ghost_ap_fee_still_rejected(self):
        with self.assertRaises(ValueError):
            self.activation([-1.]*3,[0.]*3,[1.,0.,0.],[0.]*3)

    def test_ap_exact_support_still_required(self):
        with self.assertRaisesRegex(ValueError,'exact folded-policy support'):
            self.activation([0.,-1.,-1.],[EPSILON,0.,0.],[1.,0.,0.],[EPSILON,0.,0.])

    def test_ap_integrality_still_required(self):
        with self.assertRaisesRegex(ValueError,'Nonbinary'):
            self.activation([.5,-1.,-1.],[.5,0.,0.],[.9,0.,0.],[.5,0.,0.])

    def test_arbitrary_finite_cost_difference_recorded(self):
        a=dict(native_hard_cost=100.)
        before=copy.deepcopy(a);r={}
        check_replay_cost(1.,a,r)
        self.assertEqual(a,before)
        self.assertEqual(r['cost'],100.)
        self.assertFalse(r['cost_matches_previous_tolerance'])

    def test_nonfinite_cost_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            check_replay_cost(1.,dict(native_hard_cost=float('inf')),{})

    def test_incomplete_physical_check_still_rejected(self):
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            finish_replay(dict(activation_replay=dict(status='failed')),1.)

    def test_controller_accepts_disagreement_with_compatible_pool(self):
        row,center,result=self.saved()
        validate_ap(result,row,0,center,center[None,:],.2,result[2]-.05)
        self.assertGreater(row['replay']['floor_scaled_discrepancy'],1e-4)

    def test_controller_negative_pool_gap_still_rejected(self):
        row,center,result=self.saved()
        with self.assertRaisesRegex(ValueError,'negative bound/pool'):
            validate_ap(result,row,0,center,center[None,:],.2,row['replay']['cost'])

    def test_controller_negative_incumbent_gap_still_rejected(self):
        row,center,result=self.saved();row['bound']=result[2]-.1
        with self.assertRaisesRegex(ValueError,'negative bound/incumbent'):
            validate_ap(result,row,0,center,center[None,:],.2,result[2]-.2)

    def test_solver_status_still_required(self):
        row,center,result=self.saved();row['solver_status']=9
        with self.assertRaisesRegex(ValueError,'Incomplete AP solver'):
            validate_ap(result,row,0,center,center[None,:],.2,result[2]-.05)

    def test_replay_arithmetic_integrity_still_required(self):
        row,center,result=self.saved();row['replay']['floor_scaled_discrepancy']=0.
        with self.assertRaisesRegex(ValueError,'discrepancy'):
            validate_ap(result,row,0,center,center[None,:],.2,result[2]-.05)

    def test_raw_cost_substitution_rejected(self):
        row,center,result=self.saved();row['replay']['cost']=result[2]
        row['replay']['floor_scaled_discrepancy']=0.
        with self.assertRaisesRegex(ValueError,'Changed replay cost evidence'):
            validate_ap(result,row,0,center,center[None,:],.2,result[2]-.05)

    def test_hard_smooth_objectives_unchanged(self):
        class Model:
            def per_sample_cost(self,x1,x2):
                return torch.tensor([1.,3.,2.])
            def hard_shipping_cost(self,x1,x2):
                return torch.tensor([5.,4.,6.])
        pools=[np.zeros((2,6)),np.ones((1,6))]
        self.assertEqual(restricted_objective(pools)(Model()).tolist(),[3.,2.])
        self.assertEqual(hard_objective(pools)(Model()).tolist(),[5.,6.])

    def test_parameters_unchanged(self):
        self.assertEqual(PARAMETERS,previous.adversary.PARAMETERS)


if __name__=='__main__':
    unittest.main()