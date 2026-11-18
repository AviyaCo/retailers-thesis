"""No held-out or solver execution; fixed launch barriers and failure evidence."""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_comparison as r


class RunnerTests(unittest.TestCase):
    def test_fixed_resource_gate_has_no_degraded_retry(self):
        with patch.dict(r._resource_ns, resources=lambda:{'host':'fixture'},
                        choose_workers=lambda s,p:dict(workers=15)):
            with self.assertRaisesRegex(ValueError,'headroom'):
                r.resource_gate(1)
        with patch.dict(r._resource_ns, resources=lambda:{'host':'fixture'},
                        choose_workers=lambda s,p:dict(workers=32)):
            self.assertEqual(r.resource_gate(1)['policy']['workers'],16)

    def test_failed_probe_persists_and_reraises(self):
        spec=dict(id='fixture',method='green',Ntotal=50,seed=0,radius=0.,checkpoint='missing')
        with tempfile.TemporaryDirectory() as root, patch.object(r.c,'load_model',side_effect=ValueError('injected')):
            with self.assertRaisesRegex(ValueError,'injected'):
                r.probe(spec,root)
            row=r.c.read(Path(root)/'fixture/result.json')
            self.assertEqual(row['status'],'failed')
            self.assertGreater(row['wall_time_s'],0)
            self.assertEqual(row['solves_attempted'],0)
            self.assertIn('injected',row['error'])

    def test_data_and_publish_barrier_order(self):
        text=inspect.getsource(r.main)
        anchors=['f.freeze(output)',"'test_freeze'","'training_only_preflight'",'resource_gate(',
                 'p._baselines(', 'c.c.gen(10000, 9999)',"'evaluation'",'Incomplete or failed grid',
                 "'journal_verification'","'final_integrity'",'p.render(']
        positions=[text.index(a) for a in anchors]
        self.assertEqual(positions,sorted(positions))
        self.assertNotIn('select_radius(',text)


if __name__=='__main__':
    unittest.main()