"""Training-record recovery tests; no test scenarios or fitted mock results."""
import copy
import unittest
import update_comparison as u


def fixture(costs, pools, radius=.7):
    phases = dict(status='completed', wall_time_s=2., ended_at_utc='fixture')
    iterations=[]
    for i,(cost,pool) in enumerate(zip(costs,pools)):
        a=dict(candidate_id=i, assessed=True, samples=[dict(controller_status='completed') for _ in range(40)],
            ap_incumbent_mean=cost, restricted_mean=pool, checkpoint=f'candidate_{i}',
            checkpoint_sha256=str(i), state_sha256=str(i), separation_status='violating')
        for name in ('evaluation','ap','incoming_pool_evaluation','mp_call','returned_pool_evaluation'):
            a[name]=copy.deepcopy(phases)
        iterations.append(a)
    return dict(radius=radius,iterations=iterations, input_verification_and_loading=copy.deepcopy(phases),started_at_utc='fixture')


class RecoveryTests(unittest.TestCase):
    def test_first_ap_ineligible_but_minimum_incumbent_eligible(self):
        r=u.recover_prefix(fixture([10,12,11],[10,10,10.99]))
        self.assertEqual((r['stop_candidate'],r['selected_candidate']),(2,0))

    def test_no_lookahead_selection_or_next_mp_timing(self):
        d=fixture([15,13,12,1],[10,10,11.95,1])
        r=u.recover_prefix(d)
        self.assertEqual((r['stop_candidate'],r['selected_candidate']),(2,2))
        # load2 + three (pool2+AP2) + two (incoming2+MP2+returned2) =26
        self.assertEqual(r['measured_prefix_phase_sum_s'],26)
        self.assertEqual(r['prefix_mp_s'],4)
        self.assertEqual(r['fresh_short_run_wall_s'],None)

    def test_exact_threshold_strict_absolute_and_earliest_tie(self):
        r=u.recover_prefix(fixture([101,101,101],[100,100,101]))
        self.assertEqual((r['stop_candidate'],r['selected_candidate']),(2,0))
        r=u.recover_prefix(fixture([90,90,100],[100,100,100]))
        self.assertEqual(r['stop_candidate'],2)

    def test_missing_initial_only_is_not_substituted(self):
        r=u.recover_prefix(fixture([10],[10]))
        self.assertFalse(r['recoverable'])
        self.assertNotIn('checkpoint',r)

    def test_zero_is_explicit_nominal_reference(self):
        r=u.recover_prefix(fixture([10],[10],radius=0))
        self.assertEqual(r['reason'],'nominal_reference_not_historical_stop')
        self.assertEqual(r['mp_count'],0)

    def test_incomplete_ap_and_timing_rejected(self):
        for kind in ('ap','timing'):
            d=fixture([12,10],[10,10])
            if kind=='ap':
                d['iterations'][0]['samples'].pop()
            else:
                d['iterations'][0]['mp_call']['status']='failed'
            with self.assertRaises(ValueError):
                u.recover_prefix(d)


if __name__=='__main__':
    unittest.main()