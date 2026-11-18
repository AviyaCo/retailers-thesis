"""Small control-flow fixtures, never used as experiment observations."""
import unittest
import numpy as np
import fill_short_completion as f


class ShortCompletionTests(unittest.TestCase):
    def candidate(self, k, ap=10., pool=10.):
        return dict(candidate_id=k, assessed=True, ap_incumbent_mean=ap, restricted_mean=pool,
                    samples=[dict(controller_status='completed') for _ in range(40)])

    def test_initial_is_ineligible(self):
        self.assertFalse(f.mean_stop(self.candidate(0))[0])
        self.assertTrue(f.mean_stop(self.candidate(1))[0])

    def test_strict_absolute_gap(self):
        self.assertFalse(f.mean_stop(self.candidate(1, 0.01, 0.))[0])
        self.assertFalse(f.mean_stop(self.candidate(1, 98., 100.))[0])
        self.assertFalse(f.mean_stop(self.candidate(1, 101., 100.))[0])
        self.assertTrue(f.mean_stop(self.candidate(1, 100.9, 100.))[0])

    def test_failed_or_nonfinite_assessment_rejected(self):
        a = self.candidate(1)
        a['samples'][0]['controller_status'] = 'failed'
        with self.assertRaises(ValueError):
            f.mean_stop(a)
        with self.assertRaises(ValueError):
            f.mean_stop(self.candidate(1, np.nan))

    def test_same_violating_nonduplicate_insertion(self):
        pools = [np.zeros((1, 6)) for _ in range(40)]
        points = [np.ones(6) for _ in range(40)]
        rows = [dict(incumbent_violating=i < 2, membership=dict(exact_duplicate=i == 1)) for i in range(40)]
        self.assertEqual(f.add_points(pools, points, rows), [0])
        self.assertEqual([len(p) for p in pools], [2]+[1]*39)

    def test_iteration_timing_excludes_poststop_mp(self):
        def phase(t):
            return dict(status='completed', wall_time_s=t, ended_at_utc='fixture')
        candidates=[]
        for k in range(3):
            a=self.candidate(k)
            a.update(checkpoint=f'fixture{k}', checkpoint_sha256=str(k), state_sha256=str(k),
                     separation_status='separation_satisfied', evaluation=phase(1), ap=phase(2),
                     incoming_pool_evaluation=phase(3), mp_call=phase(4), returned_pool_evaluation=phase(5))
            candidates.append(a)
        r=f.u.recover_prefix(dict(iterations=candidates, radius=.001,
            input_verification_and_loading=phase(6), started_at_utc='fixture'))
        self.assertEqual(r['stop_candidate'],1)
        self.assertEqual(r['selected_candidate'],0)
        self.assertEqual(r['measured_prefix_phase_sum_s'],24)


if __name__ == '__main__':
    unittest.main()