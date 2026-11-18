import time
import unittest
from run_campaign import publish_radius_selections,failed_future,require_complete
from common import core


class CampaignTests(unittest.TestCase):
    def test_complete_barrier_and_ties(self):
        rows=[dict(seed=s,radius=r,status='completed',mean=1.,checkpoint_sha256='c',
                   state_sha256='s',terminal_status='iteration_cap') for s in core.SEEDS for r in core.RADII]
        with self.assertRaises(ValueError):
            publish_radius_selections(rows[:-1])
        self.assertEqual([s['radius'] for s in publish_radius_selections(rows)],[0.]*10)
        rows[-1]['status']='failed'
        with self.assertRaises(ValueError):
            publish_radius_selections(rows)

    def test_exception_timing_not_imputed_as_worker_time(self):
        row=failed_future(RuntimeError('fixture'),1,.2,time.perf_counter())
        self.assertIsNone(row['wall_time_s'])
        self.assertGreaterEqual(row['parent_observed_seconds'],0)
        with self.assertRaises(ValueError):
            require_complete([row],1)


if __name__=='__main__':
    unittest.main()