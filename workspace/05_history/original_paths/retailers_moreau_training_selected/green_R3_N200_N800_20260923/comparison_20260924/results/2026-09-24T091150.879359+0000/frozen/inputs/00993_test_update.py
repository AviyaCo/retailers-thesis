"""Synthetic control-flow fixtures only; no data, training, or solver runs."""
import copy
import unittest
from reconstruct import reconstruct, choose_radii


def fixture(aps, pools, radius=.7):
    phase = dict(status='completed', wall_time_s=2., ended_at_utc='fixture')
    iterations = []
    for i, (a, p) in enumerate(zip(aps, pools)):
        row = dict(candidate_id=i, assessed=True,
            samples=[dict(sample_id=s, controller_status='completed', incumbent={'cost': a}) for s in range(40)],
            ap_incumbent_mean=a, ap_bound_mean=a, restricted_mean=p,
            checkpoint=f'fixture_{i}', checkpoint_sha256=str(i), state_sha256=str(i),
            separation_status='violating')
        for name in ('evaluation', 'ap', 'incoming_pool_evaluation', 'mp_call', 'returned_pool_evaluation'):
            row[name] = copy.deepcopy(phase)
        row['ap']['candidate_id'] = i  # Real AP metadata already carries this key.
        iterations.append(row)
    return dict(status='completed', seed=0, radius=radius, mp_count=len(aps)-1,
        terminal_status='separation_satisfied', iterations=iterations,
        input_verification_and_loading=phase, summary_path='fixture', started_at_utc='fixture',
        wall_time_s=100., cached_nominal={'training_timing': phase})


class ReconstructionTests(unittest.TestCase):
    def test_zero_mp_eligible_and_no_lookahead(self):
        row = reconstruct(fixture([10, 1], [10, 1]), 'green')
        self.assertEqual((row['stop_candidate'], row['selected_iteration'], row['mp_count']), (0, 0, 0))
        self.assertEqual(row['reconstructed_phase_s'], 6)
        self.assertEqual(row['mp_s'], 0)
        self.assertIsNone(row['fresh_protocol_wall_s'])

    def test_min_ap_not_terminal_or_future(self):
        row = reconstruct(fixture([9, 11, 10, 1], [7, 8, 10, 1]), 'green')
        self.assertEqual((row['stop_candidate'], row['selected_iteration']), (2, 0))
        self.assertEqual(row['reconstructed_phase_s'], 26)
        self.assertEqual(row['mp_s'], 4)

    def test_black_minimum_and_cap_preserved(self):
        d = fixture([10, 9, 11], [7, 8, 11])
        d['terminal_status'] = 'iteration_cap'
        r = reconstruct(d, 'black')
        self.assertEqual((r['stop_candidate'], r['selected_iteration']), (2, 1))
        self.assertEqual(r['terminal_status'], 'iteration_cap')

    def test_strict_threshold_absolute_and_earliest_tie(self):
        r = reconstruct(fixture([101, 101], [100, 101]), 'green')
        self.assertEqual((r['stop_candidate'], r['selected_iteration']), (1, 0))
        self.assertEqual(reconstruct(fixture([90, 100], [100, 100]), 'green')['stop_candidate'], 1)

    def test_missing_stop_no_fallback(self):
        with self.assertRaises(ValueError):
            reconstruct(fixture([12], [10]), 'green')

    def test_incomplete_ap_rejected(self):
        d = fixture([10], [10]); d['iterations'][0]['samples'].pop()
        with self.assertRaises(ValueError):
            reconstruct(d, 'green')

    def test_wrong_phase_candidate_rejected(self):
        d = fixture([10], [10]); d['iterations'][0]['ap']['candidate_id'] = 1
        with self.assertRaises(ValueError):
            reconstruct(d, 'green')

    def test_failed_timing_rejected(self):
        d = fixture([12, 10], [10, 10]); d['iterations'][0]['mp_call']['status'] = 'failed'
        with self.assertRaises(ValueError):
            reconstruct(d, 'green')

    def test_validation_selection_not_test(self):
        rows = [dict(color=c, seed=s, radius=r, validation_mean=1., test_mean=-r)
            for c in ('green', 'black') for s in range(10)
            for r in (0., .001, .003, .01, .03, .2, .4, .7, 1.5)]
        result = choose_radii(rows)
        self.assertTrue(all(r['radius'] == 0 for group in result.values() for r in group))


if __name__ == '__main__':
    unittest.main()