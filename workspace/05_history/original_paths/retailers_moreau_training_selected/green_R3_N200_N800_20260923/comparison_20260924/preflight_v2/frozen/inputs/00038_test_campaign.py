import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import run_campaign as campaign
from run_campaign import require_grid, choose_workers


class CampaignTests(unittest.TestCase):
    def test_partial_duplicate_failure_and_nonconverged_rejected(self):
        expected={(200,0),(800,0)}
        rows=[dict(Ntotal=n,seed=0,status='completed',complete=True) for n in (200,800)]
        require_grid(rows,expected,('Ntotal','seed'))
        for bad in (rows[:1],rows+[rows[0]],rows[:1]+[dict(rows[1],status='nonconverged',complete=False)],
                    rows[:1]+[dict(rows[1],status='failed')]):
            with self.assertRaises(ValueError):
                require_grid(bad,expected,('Ntotal','seed'))

    def test_worker_budget_obeys_cpu_memory_and_quota(self):
        snapshot=dict(affinity=list(range(72)),load_average=[8,8,8],cpu_quota_cores=None,
                      memory_available_bytes=476*1024**3)
        preflight=dict(sizes=[dict(peak_rss_bytes=1024**3)])
        self.assertEqual(choose_workers(snapshot,preflight)['ccg'],60)
        self.assertEqual(choose_workers(dict(snapshot,memory_available_bytes=32*1024**3),preflight)['ccg'],4)
        with self.assertRaises(ValueError):
            choose_workers(dict(snapshot,cpu_quota_cores=2),preflight)

    def test_failed_initialization_preserves_nested_solver_evidence(self):
        def fail_saa(x1,x2,record):
            with campaign.core.timed(record):
                record.update(solver_status=9,solver_runtime_s=.125,iterations=17)
                raise RuntimeError('injected solver failure, no solver executed')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest=root/'manifest.json'
            campaign.core.write_json(manifest,dict(workers={'ccg':1}))
            with patch.object(campaign,'verify_sources'),patch.object(campaign,'solve_saa',side_effect=fail_saa):
                result=campaign.worker('initialization',dict(n=200,seed=0),root/'job',manifest)
            saved=json.loads((root/'job/result.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertEqual(saved['saa_initialization']['solver_status'],9)
            self.assertEqual(saved['saa_initialization']['status'],'failed')
            self.assertEqual(saved['saa_initialization']['iterations'],17)
            self.assertGreaterEqual(saved['saa_initialization']['wall_time_s'],0.)

    def test_failed_validation_is_registered_before_call(self):
        def fail_evaluation(model,x1,x2,record):
            with campaign.core.timed(record):
                record['sample_count']=40
                raise RuntimeError('injected evaluation failure, no solver executed')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            manifest=root/'manifest.json'
            campaign.core.write_json(manifest,dict(workers={'ccg':1}))
            split=root/'split.json'
            campaign.core.write_json(split,dict(arrays={'val_x1':[[1.,1.,1.]]*40,'val_x2':[[2.,2.,2.]]*40}))
            setup=dict(split_path=str(split),split_sha256=campaign.core.digest(split))
            checkpoint=root/'checkpoint'
            checkpoint.write_text('fixture only, never deserialized')
            arms=[dict(Ntotal=200,seed=0,lr_net=lr,status='completed',complete=True,
                checkpoint=str(checkpoint),checkpoint_sha256=campaign.core.digest(checkpoint),state_sha256='mock')
                for lr in campaign.LR_GRID]
            with patch.object(campaign,'verify_sources'),patch.object(campaign.torch,'load'),\
                 patch.object(campaign.core,'model_from_state'),patch.object(campaign.legacy_ccg,'state_hash',return_value='mock'),\
                 patch.object(campaign,'evaluate',side_effect=fail_evaluation):
                result=campaign.worker('lr_validation',dict(n=200,seed=0,setup=setup,arms=arms),root/'job',manifest)
            self.assertEqual(result['status'],'failed')
            saved=json.loads((root/'job/result.json').read_text())
            self.assertEqual(saved['evaluations'][0]['status'],'failed')
            self.assertEqual(saved['evaluations'][0]['sample_count'],40)
            self.assertIn('wall_time_s',saved['evaluations'][0])


if __name__=='__main__':
    unittest.main()