"""Solver-free contract fixtures plus tiny native/spawn regressions; NOT fits."""
from source_adapter import ROOT, sources, verify, digest, check, GREEN, load_into
import shipping_core as core
from shipping_core import torch, np
from shipping_model import Model, make_model, model_from_state, charges
from shipping_common import split_data, metadata, PROTOCOL
from shipping_training import train_model
from shipping_resources import choose_workers
from run_nominal import require_grid, array_hash
from validate_preflight import completed, source_checks, validate
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch
import argparse
import ast
import copy
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest


def spawn_probe(i):
    model = make_model(2.,np.ones(3))
    loss = model.per_sample_cost(torch.ones(2,3),torch.ones(2,3)).mean()
    loss.backward()
    return dict(task=i,pid=__import__('os').getpid(),threads=torch.get_num_threads(),
                interop=torch.get_num_interop_threads(),finite=bool(torch.isfinite(loss)),
                module=train_model.__module__)


class FixtureModel(torch.nn.Module):
    """Only for trainer bookkeeping tests, never used as a preflight policy."""
    def __init__(self):
        super().__init__()
        self.fc_layers = torch.nn.Linear(1,1)
        self.Q10 = torch.nn.Parameter(torch.ones(1))
        self.Q1r = torch.nn.Parameter(torch.ones(3))

    def objective(self):
        return sum(p.square().sum() for p in self.parameters())


class Contracts(unittest.TestCase):
    def test_01_tiny_positive_fee(self):
        for value in (np.nextafter(0.,1.),2.**-100,2.**-30,5e-7):
            q = torch.tensor([[value,0.,0.]],dtype=torch.float64)
            self.assertEqual(float(charges(q)[1]),.1)

    def test_02_exact_zero_fee(self):
        soft, hard = charges(torch.zeros(2,3,dtype=torch.float64))
        self.assertTrue(torch.equal(soft,hard))
        self.assertEqual(float(hard.sum()),0.)

    def test_03_smooth_gradient(self):
        q = torch.tensor([[0.,.2,1.]],dtype=torch.float64,requires_grad=True)
        charges(q)[0].sum().backward()
        self.assertTrue(torch.allclose(q.grad,.5*torch.exp(-5*q)))

    def test_04_zero_capacity_native(self):
        p = core.NativeProjection()
        q, = p(torch.tensor([[2.,1.,-1.]],dtype=torch.float32),torch.zeros(1,1),core._SOURCE_SOLVER_ARGS)
        # Frozen native arithmetic returns a tiny NEGATIVE residual on this
        # fixture; do not clip it or assert exact-real support certification.
        self.assertTrue(bool((q <= 0).all()))
        self.assertEqual(p.drain_summary()['statuses'],{1:1})
        self.assertEqual(float(charges(q)[1]),0.)

    def test_05_batch_and_same_q(self):
        m = make_model(3.,np.ones(3))
        for batch in (1,2,7,2):
            soft, hard, q = m.costs_and_shipments(torch.ones(batch,3),torch.ones(batch,3))
            self.assertEqual(q.shape,(batch,3))
            self.assertEqual(q.dtype,torch.float64)
            self.assertEqual(next(m.fc_layers.parameters()).dtype,torch.float32)
            a,b = charges(q)
            self.assertTrue(torch.allclose(hard-soft,b-a,atol=1e-14,rtol=0))
            m.zero_grad(set_to_none=True)
            soft.mean().backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters()))

    def test_06_projection_bad_dimensions(self):
        with self.assertRaises(ValueError):
            core.NativeProjection()(torch.ones(2,4),torch.ones(2,1),core._SOURCE_SOLVER_ARGS)

    def test_07_projection_nonfinite(self):
        with self.assertRaises(FloatingPointError):
            core.NativeProjection()(torch.full((1,3),float('nan')),torch.ones(1,1),core._SOURCE_SOLVER_ARGS)

    def test_08_projection_negative_capacity(self):
        with self.assertRaises(ValueError):
            core.NativeProjection()(torch.ones(1,3),-torch.ones(1,1),core._SOURCE_SOLVER_ARGS)

    def test_09_zero_initialization_fails(self):
        for q10,q1 in ((0.,np.ones(3)),(1.,np.array([0.,1.,1.]))):
            with self.assertRaises(FloatingPointError):
                make_model(q10,q1)

    def test_10_immutable_initial_weights(self):
        a,b = make_model(1.,np.ones(3)),make_model(1.,np.ones(3))
        self.assertTrue(all(torch.equal(v,core.weights(b)[k]) for k,v in core.weights(a).items()))
        state = core.weights(a)
        c = model_from_state(state)
        with torch.no_grad():
            c.Q10.add_(1)
        self.assertTrue(torch.equal(a.Q10,state['Q10']))

    def test_11_training_preupdate_initial(self):
        m = FixtureModel()
        before = core.weights(m)
        row = metadata(50,0,'fixture')
        with tempfile.TemporaryDirectory() as d:
            train_model(m,lambda m:m.objective(),.01,d,row,epochs=1)
        self.assertEqual(row['best_epoch'],1)
        self.assertEqual(row['optimizer_steps'],1)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in core.weights(m).items()))

    def test_12_training_patience_updates_final(self):
        m = FixtureModel()
        row = metadata(50,0,'fixture')
        with tempfile.TemporaryDirectory() as d:
            train_model(m,lambda m:0*m.objective()+1,.01,d,row,epochs=5,patience=1)
        self.assertEqual((row['epochs_evaluated'],row['optimizer_steps'],row['best_epoch']),(2,2,1))
        self.assertEqual(row['stop_reason'],'training_patience')

    def test_13_training_nan_fails(self):
        row = metadata(50,0,'fixture')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FloatingPointError):
                train_model(FixtureModel(),lambda m:m.objective()*float('nan'),.01,d,row,epochs=2)
            self.assertTrue((Path(d)/'failure_state.pth').exists())
            self.assertEqual(row['status'],'failed')
            self.assertEqual(row['optimizer_steps'],0)

    def test_14_train40_guard_only(self):
        for n in (10,50,200,800):
            with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
                train_model(FixtureModel(),lambda m:m.objective(),.01,d,dict(Ntrain=n),epochs=1)

    def test_15_split_all_seeds(self):
        for n in (50,200,800):
            for seed in range(10):
                d = split_data(n,seed)
                ids = np.random.default_rng(100000+seed).permutation(n)
                self.assertTrue(np.array_equal(ids[:4*n//5],d['train_indices']))
                self.assertFalse(set(d['train_indices']) & set(d['val_indices']))
                self.assertEqual(d['train_x1'].shape,(4*n//5,3))
                self.assertTrue(np.array_equal(d['xi1'][d['train_indices']].astype('float32'),d['train_x1'].numpy()))

    def test_16_grid_barrier(self):
        row = dict(Ntotal=50,seed=0,status='completed',complete=True)
        require_grid([row],{(50,0)},('Ntotal','seed'))
        for rows in ([],[row,row],[{**row,'status':'failed'}]):
            with self.assertRaises(ValueError):
                require_grid(rows,{(50,0)},('Ntotal','seed'))

    def test_17_source_readonly_and_mismatch(self):
        before = sources()
        verify(before)
        self.assertEqual(before,sources())
        broken = dict(before)
        broken[str(ROOT/'shipping_model.py')] = '0'*64
        with self.assertRaises(ValueError):
            verify(broken)

    def test_18_decorator_retained(self):
        row = {}
        with core.timed(row):
            pass
        self.assertEqual(row['status'],'completed')
        self.assertEqual(train_model.__module__,'shipping_training')

    def test_19_real_spawn_two_tasks(self):
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=core.configure_worker,max_tasks_per_child=1) as pool:
            rows = list(pool.map(spawn_probe,(0,1)))
        self.assertEqual({r['task'] for r in rows},{0,1})
        self.assertEqual(len({r['pid'] for r in rows}),2)
        self.assertTrue(all(r['threads']==r['interop']==1 and r['finite'] for r in rows))

    def test_20_resource_caps(self):
        s = dict(affinity=list(range(128)),cpu_quota_cores=None,load_average=[1,2,3],
                 memory_available_bytes=1024*1024**3)
        workers = choose_workers(s,0)
        self.assertEqual((workers['initialization'],workers['nominal']),(30,90))
        s['cpu_quota_cores'] = 8
        self.assertEqual(choose_workers(s,0)['nominal'],1)
        s['cpu_quota_cores'] = 7
        with self.assertRaises(ValueError):
            choose_workers(s,0)

    def test_21_nested_status_rejected(self):
        for row in ({'status':'failed'},dict(status='completed',wall_time_s=float('nan'))):
            with self.assertRaises((ValueError,KeyError)):
                completed(row)
        with self.assertRaises(ValueError):
            source_checks(dict(source_sha256={},source_checks=[]),{})

    def test_22_false_completed_preflight_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'summary.json'
            p.write_text(json.dumps(dict(status='completed',wall_time_s=1,started_at_utc='x',ended_at_utc='y',
                source_sha256=sources(),source_checks=[])))
            for sha in ('0'*64,digest(p)):
                with self.assertRaises(ValueError):
                    validate(p,sha)

    def test_23_no_heldout_worker_access(self):
        from run_nominal import nominal_job
        import inspect
        text = inspect.getsource(nominal_job)
        self.assertNotIn("setup['split_path']",text)
        self.assertNotIn('val_x',text)
        self.assertEqual(PROTOCOL['initialization_barrier'],30)
        self.assertEqual(PROTOCOL['nominal_arms'],90)

    def test_24_projection_bad_status_raises(self):
        class Impl:
            def solve(self):
                return dict(status=torch.tensor([2]),iterations=torch.tensor([1]))
        class FakeSolver:
            def __init__(self,**kwargs):
                self._impl = Impl()
        p = core.NativeProjection()
        with patch.object(core,'Solver',FakeSolver):
            solver = p._new_solver()
        with self.assertRaises(RuntimeError):
            solver._impl.solve()

    def test_25_projection_nonfinite_result_raises(self):
        class Impl:
            def solve(self):
                return dict(status=torch.tensor([1]),iterations=torch.tensor([1]),x=torch.tensor([float('nan')]))
        class FakeSolver:
            def __init__(self,**kwargs):
                self._impl = Impl()
        p = core.NativeProjection()
        with patch.object(core,'Solver',FakeSolver):
            solver = p._new_solver()
        with self.assertRaises(FloatingPointError):
            solver._impl.solve()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    check(not args.output.exists(),'Test output exists')
    core.configure_worker()
    record = dict(method='disposable contract fixtures',workers=2,threads=1)
    with core.timed(record):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
        record.update(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors))
    if not result.wasSuccessful():
        record['status']='failed'
    core.write_json(args.output,record)
    raise SystemExit(0 if result.wasSuccessful() else 1)