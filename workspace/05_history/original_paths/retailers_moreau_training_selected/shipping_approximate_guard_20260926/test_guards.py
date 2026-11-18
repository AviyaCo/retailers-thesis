"""Retained safeguards, new tolerance boundaries and advisory signed-gap checks."""
import copy
from fractions import Fraction as F
from types import SimpleNamespace
import unittest
from guard_common import np,torch,read,REPLAY,definitions,parent,PHYSICAL_TOLERANCE
from guard_adversary import (EPSILON,PARAMETERS,classify_support,validate_activation,
    finish_replay,check_replay_cost)
from guard_pool import validate_ap,hard_objective,restricted_objective

previous=parent.prior.previous
definitions(globals(),previous.previous.PREVIOUS/'test_epsilon.py',('weights','evidence'))
definitions(globals(),REPLAY/'test_epsilon.py',('NativeFixture',))
definitions(globals(),parent.OBSERVED/'test_observed.py',('ObservedTests',),changes=((
    'record=evidence(target,ap_q,flags)',
    "record=evidence(target,ap_q,flags)\n        record['incumbent']['cost']=.1*sum(flags)"),(
    '[-6e-7,0.,0.]','[-2e-6,0.,0.]'),(
    "    def test_controller_negative_pool_gap_still_rejected(self):\n"
    "        row,center,result=self.saved()\n"
    "        with self.assertRaisesRegex(ValueError,'negative bound/pool'):\n"
    "            validate_ap(result,row,0,center,center[None,:],.2,row['replay']['cost'])",
    "    def test_controller_negative_pool_gap_recorded_not_rejected(self):\n"
    "        row,center,result=self.saved()\n"
    "        raw_cost=row['replay']['cost']\n"
    "        validate_ap(result,row,0,center,center[None,:],.2,raw_cost)\n"
    "        self.assertTrue(row['material_negative_bound_pool_gap'])\n"
    "        self.assertLess(row['raw_bound_minus_pool'],0.)\n"
    "        self.assertEqual(row['pool_cost'],raw_cost)\n"
    "        self.assertEqual(row['bound_pool_consistency_status'],'recorded_only')")))
definitions(globals(),parent.ROOT/'test_complete.py',('FailureLoggingTests',),changes=((
    '[-6e-7,0.,0.]','[-2e-6,0.,0.]'),))


class GuardBoundaryTests(unittest.TestCase):
    def activation(self,native):
        return ObservedTests().activation([.5,-1.,-1.],[.5,0.,0.],[1.,0.,0.],native)

    def test_observed_negative_accepted_raw_unchanged(self):
        row=self.activation([-6e-7,0.,0.])
        a=row['activation_replay']
        self.assertEqual(a['native_q'],[-6e-7,0.,0.])
        self.assertEqual(a['native_physical_residual'],6e-7)
        self.assertEqual(a['native_physical_tolerance'],1e-6)
        self.assertTrue(a['native_physical_guard_passed'])
        self.assertEqual(row['replay']['cost'],0.)

    def test_negative_boundary_accepted(self):
        self.activation([-PHYSICAL_TOLERANCE,0.,0.])

    def test_just_beyond_negative_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical constraints'):
            self.activation([np.nextafter(-PHYSICAL_TOLERANCE,-np.inf),0.,0.])

    def test_capacity_boundary_accepted_unchanged(self):
        self.activation([1.+PHYSICAL_TOLERANCE,0.,0.])

    def test_just_beyond_capacity_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError,'physical constraints'):
            self.activation([np.nextafter(1.+PHYSICAL_TOLERANCE,np.inf),0.,0.])

    def test_native_tolerance_does_not_change_ap_tolerance(self):
        self.assertEqual(PARAMETERS['FeasibilityTol'],1e-9)
        self.assertEqual(PARAMETERS['IntFeasTol'],1e-9)
        self.assertEqual(EPSILON,1e-6)

    def test_nonfinite_pool_cost_integrity_unchanged(self):
        row,center,result=ObservedTests().saved()
        row['replay']['cost']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            validate_ap(result,row,0,center,center[None,:],.2,1.)


if __name__=='__main__':
    unittest.main()