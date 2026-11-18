"""All prior regressions plus explicit advisory cost evidence on physical failure."""
import copy
from fractions import Fraction as F
from types import SimpleNamespace
import unittest
from common import np,torch,read,REPLAY,OBSERVED,prior as previous,definitions,prior_pool
from adversary import (EPSILON,PARAMETERS,classify_support,validate_activation,
    finish_replay,check_replay_cost)

validate_ap=prior_pool.validate_ap
hard_objective,restricted_objective=prior_pool.hard_objective,prior_pool.restricted_objective
# Resolve the same previous module expected by the frozen parameter assertion.
previous=previous.previous
definitions(globals(),previous.previous.PREVIOUS/'test_epsilon.py',('weights','evidence'))
definitions(globals(),REPLAY/'test_epsilon.py',('NativeFixture',))
definitions(globals(),OBSERVED/'test_observed.py',('ObservedTests',),changes=((
    'record=evidence(target,ap_q,flags)',
    "record=evidence(target,ap_q,flags)\n        record['incumbent']['cost']=.1*sum(flags)"),))


class FailureLoggingTests(unittest.TestCase):
    def test_physical_failure_retains_advisory_cost_without_success(self):
        target=[.5,-1.,-1.]
        record=evidence(target,[.5,0.,0.],[1.,0.,0.])
        record['incumbent']['cost']=.1
        model=NativeFixture([-6e-7,0.,0.])
        with self.assertRaisesRegex(ValueError,'physical constraints'):
            validate_activation(model,weights(target),np.zeros(3),np.zeros(3),record)
        self.assertEqual(model.calls,1)
        self.assertEqual(record['activation_replay']['status'],'failed')
        self.assertEqual(record['native_cost_observation']['status'],'completed')
        self.assertEqual(record['native_cost_observation']['consistency_status'],'recorded_only')
        self.assertEqual(record['native_cost_observation']['raw_native_minus_AP'],-.1)
        self.assertFalse(record['native_cost_observation']['cost_matches_previous_tolerance'])
        self.assertNotIn('replay',record)

    def test_completed_witness_has_identical_raw_observations(self):
        record=ObservedTests().activation([-1.]*3,[0.]*3,[0.]*3,[1e-6,0.,0.])
        for key in ('cost','ap_cost','floor_scaled_discrepancy','raw_native_minus_AP'):
            self.assertEqual(record['replay'][key],record['native_cost_observation'][key])


if __name__=='__main__':
    unittest.main()