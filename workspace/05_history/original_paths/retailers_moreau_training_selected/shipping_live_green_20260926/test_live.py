"""Analytic fixtures only; the separate preflight exercises native shipping."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace
import shipping_live_common as common
import shipping_ccg as ccg
import run_campaign as campaign

core,np,torch=common.core,common.np,common.torch
training=common.shipping_training
common.definitions(globals(),common.GREEN/'test_green.py',
    ('DecisionTests','Tiny','quadratic','TrainingTests'))


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=torch.nn.Parameter(torch.tensor(20.,dtype=torch.float64))
        self.proj_layer=SimpleNamespace(drain_summary=lambda: {'fixture':True})

    def per_sample_cost(self,x1,x2):
        return x1[:,0].double()+self.bias

    def hard_shipping_cost(self,x1,x2):
        return self.per_sample_cost(x1,x2)+.5


class ControllerTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.ap_calls=self.mp_calls=0
        model=Policy();self.centers=np.ones((40,6));self.ids=list(range(40))
        checkpoint=self.root/'nominal.pth';torch.save(core.weights(model),checkpoint)
        data=self.root/'train.json'
        core.write_json(data,dict(centers=self.centers,original_sample_ids=self.ids,data_sha256='fixture'))
        self.nominal=dict(Ntotal=50,seed=0,lr_net=.01,train_mean=21.,data_sha256='fixture',
            checkpoint=str(checkpoint),checkpoint_sha256=core.digest(checkpoint),
            state_sha256=ccg.state_hash(model),training_data_path=str(data),training_data_sha256=core.digest(data),
            training_timing=dict(status='completed',wall_time_s=10.,epochs_evaluated=2))
        self.manifest=self.root/'manifest.json'
        core.write_json(self.manifest,dict(protocol=common.PROTOCOL,workers={'ccg':1}))
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(ccg,'verify_sources'))
        stack.enter_context(patch.object(core,'model_from_state',side_effect=self.load))
        stack.enter_context(patch.object(torch,'get_num_threads',return_value=1))
        stack.enter_context(patch.object(torch,'get_num_interop_threads',return_value=1))
        stack.enter_context(patch.object(ccg.adversary,'solve_ap',side_effect=self.ap))
        stack.enter_context(patch.object(ccg,'validate_ap',side_effect=self.validate))
        stack.enter_context(patch.object(ccg,'train_model',side_effect=self.train))

    def load(self,state):
        model=Policy();model.load_state_dict(state);return model

    def ap(self,model,x1,x2,radius,row):
        self.ap_calls+=1
        a=x1.copy();a[0]+=radius
        with core.timed(row):
            cost=float(model.hard_shipping_cost(torch.tensor(a[None,:]),torch.tensor(x2[None,:]))[0].detach())
            row.update(bound=cost,solver_status=2,solcount=1,incumbent=dict(cost=cost),
                material_negative_bound_pool_gap=False,activation_replay={'support_matches':True},
                native_cost_observation={'cost_matches_previous_tolerance':True})
        return a,x2.copy(),cost

    def validate(self,result,row,index,center,pool,radius,pool_cost):
        point=np.concatenate(result[:2]);gap=result[2]-pool_cost
        row.update(incumbent_violating=gap>1e-4,residual=max(0.,gap)/max(1.,abs(pool_cost)),
            membership=ccg.point_membership(point,pool))
        return point

    def train(self,model,loss_fn,lr,folder,record,epochs,patience):
        self.mp_calls+=1
        self.assertEqual((epochs,patience),(3000,300))
        with core.timed(record),torch.no_grad():
            first=float(loss_fn(model));self.assertAlmostEqual(first,21.4)
            model.bias.sub_(.01);best=float(loss_fn(model))
            record.update(first_train=first,best_train=best,selected_training_loss=best,
                epochs_evaluated=1,optimizer_steps=1)
            torch.save(core.weights(model),folder/'selected.pth')
        return model

    def run_case(self,radius):
        return ccg.run_configuration(50,0,radius,self.root/'out',self.nominal,self.manifest)

    def test_hard_stopping_separate_from_smooth_training_one_mp(self):
        row=self.run_case(.4)
        self.assertEqual(row['status'],'completed',row['error'])
        self.assertEqual((self.ap_calls,self.mp_calls),(80,1))
        data=json.loads(Path(row['summary_path']).read_text())
        first,second=data['iterations']
        self.assertEqual(first['restricted_mean'],21.5)
        self.assertEqual(first['smooth_training_replay']['mean'],21.)
        self.assertAlmostEqual(first['incoming_pool_evaluation']['mean'],21.4,places=6)
        self.assertEqual(second['candidate_id'],row['stopping_candidate_id'])
        self.assertNotIn('mp',second)
        self.assertEqual(row['selected_candidate_id'],1)

    def test_candidate_zero_stops_no_mp(self):
        row=self.run_case(0.)
        self.assertEqual(row['status'],'completed',row['error'])
        self.assertEqual((self.ap_calls,self.mp_calls),(40,0))

    def test_failed_ap_all_centers_logged_no_selection(self):
        original=self.ap
        def failing(*args):
            if self.ap_calls==3:
                self.ap_calls+=1;raise RuntimeError('fixture AP failure')
            return original(*args)
        with patch.object(ccg.adversary,'solve_ap',side_effect=failing):
            row=self.run_case(.4)
        self.assertEqual(row['status'],'failed')
        self.assertEqual(self.ap_calls,40)
        self.assertIsNone(row['selected_checkpoint'])

    def test_iteration_cap_not_convergence(self):
        with patch.object(ccg,'MAX_MP',0):
            row=self.run_case(.4)
        self.assertEqual(row['status'],'nonconverged')
        self.assertIsNone(row['selected_checkpoint'])


class SelectionTests(unittest.TestCase):
    def test_hard_cost_not_smooth_and_smaller_tie(self):
        rows=[dict(lr_net=lr,status='completed',complete=True,validation_hard_mean=hard,smooth_mean=smooth)
            for lr,hard,smooth in ((.05,2.,0.),(.01,1.,3.),(.005,1.,4.))]
        self.assertEqual(campaign.select_row(rows,'lr_net')['lr_net'],.005)

    def test_incomplete_grid_cannot_select(self):
        with self.assertRaises(ValueError):
            campaign.require_grid([], {(50,0)},('Ntotal','seed'))

    def test_nonconverged_grid_blocks_radius_selection(self):
        rows=[dict(radius=0.,status='completed',complete=True,converged=False)]
        with self.assertRaisesRegex(ValueError,'Nonconverged'):
            campaign.require_grid(rows,{(0.,)},('radius',),converged=True)


if __name__=='__main__':
    unittest.main()