"""Controller integration must not convert a replay exception into a gap waiver."""
import copy
import unittest
from campaign_common import REPLAY,read,np,torch,adversary
from shipping_pool import validate_ap,hard_objective,restricted_objective


class PoolTests(unittest.TestCase):
    def case(self,name='N800_radius0.2.json'):
        row=copy.deepcopy(read(REPLAY/'ap_gate_v1'/name))
        inc=row['incumbent']
        center=np.concatenate((row['center1'],row['center2']))
        result=(np.asarray(inc['worst1']),np.asarray(inc['worst2']),inc['cost'])
        return row,center,result

    def test_exception_accepts_compatible_pool(self):
        row,center,result=self.case()
        validate_ap(result,row,0,center,center[None,:],.2,result[2]-.05)
        self.assertTrue(row['consistency_exception'])
        self.assertEqual(row['replay']['cost'],13.224030014779418)

    def test_exception_cannot_waive_negative_pool_gap(self):
        row,center,result=self.case()
        with self.assertRaisesRegex(ValueError,'negative bound/pool'):
            validate_ap(result,row,0,center,center[None,:],.2,row['replay']['cost'])

    def test_exception_cannot_waive_bound_incumbent(self):
        row,center,result=self.case()
        row['bound']=result[2]-.1
        with self.assertRaisesRegex(ValueError,'negative bound/incumbent'):
            validate_ap(result,row,0,center,center[None,:],.2,result[2]-.2)

    def test_exception_flag_is_insufficient(self):
        row,center,result=self.case()
        row['activation_replay']['native_q'][2]=1e-6
        with self.assertRaises(ValueError):
            validate_ap(result,row,0,center,center[None,:],.2,result[2]-.05)

    def test_raw_cost_cannot_be_substituted(self):
        row,center,result=self.case()
        row['replay']['cost']=result[2]
        row['replay']['floor_scaled_discrepancy']=0.
        with self.assertRaises(ValueError):
            validate_ap(result,row,0,center,center[None,:],.2,result[2]-.05)

    def test_ordinary_replay(self):
        row,center,result=self.case('N50_radius0.json')
        validate_ap(result,row,0,center,center[None,:],0.,result[2])
        self.assertFalse(row['consistency_exception'])

    def test_hard_and_smooth_objectives_are_separate(self):
        class Model:
            def per_sample_cost(self,x1,x2):
                return torch.tensor([1.,3.,2.])
            def hard_shipping_cost(self,x1,x2):
                return torch.tensor([5.,4.,6.])
        pools=[np.zeros((2,6)),np.ones((1,6))]
        self.assertEqual(restricted_objective(pools)(Model()).tolist(),[3.,2.])
        self.assertEqual(hard_objective(pools)(Model()).tolist(),[5.,6.])


if __name__=='__main__':
    unittest.main()