"""R10-specific invariants; synthetic fixtures are NOT research measurements."""
import ast
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import green_common as common
import green_ccg as ccg
import green_training as training
import run_campaign as campaign
import r10_strict as strict
import r10_resources as resource_policy
import source_adapter as adapter

core, np, torch = common.core, common.np, common.torch
_spawn_calls = 0

def spawn_probe():
    global _spawn_calls
    _spawn_calls += 1
    # Actual Torch parallel operation; a second inter-op initialization would fail.
    value = float((torch.ones((10, 21)) @ torch.ones((21, 16))).sum())
    return dict(pid=os.getpid(), task=_spawn_calls, value=value,
        intra=torch.get_num_threads(), inter=torch.get_num_interop_threads(),
        env={k: os.environ.get(k) for k in common.THREAD_ENV})

class R10Tests(unittest.TestCase):
    def test_model_projection_and_state_layout(self):
        model = core.make_model(10., np.ones(10))
        self.assertEqual((model.fc_layers[0].in_features, model.fc_layers[0].out_features,
                          model.fc_layers[2].out_features, model.fc_layers[4].out_features),
                         (21, 16, 16, 10))
        self.assertEqual(tuple(model.Q1r.shape), (10,))
        self.assertEqual((model.proj_layer._solver_kwargs['n'], model.proj_layer._solver_kwargs['m']), (10, 11))
        self.assertEqual(model.proj_layer._solver_kwargs['cones'].num_nonneg_cones, 11)
        restored = core.model_from_state(core.weights(model))
        self.assertEqual(ccg.state_hash(model), ccg.state_hash(restored))
        reference = strict.Model()
        reference.load_state_dict(core.weights(model), strict=True)
        self.assertEqual(tuple(reference.targets(np.ones((2, 10)))[0].shape), (2, 10))

    def test_folding_uses_all_ten_inventory_and_demand_coordinates(self):
        model = core.make_model(10., np.arange(1., 11.))
        weights = common.adversary._load_weights(model)
        self.assertEqual(weights['W1'].shape, (16, 10))
        self.assertEqual(weights['W3'].shape, (10, 16))
        x = np.arange(1., 11.)
        first = model.fc_layers[0].weight.detach().numpy().astype(np.float64)
        bias = model.fc_layers[0].bias.detach().numpy().astype(np.float64)
        expected = first @ np.concatenate((x, weights['Q1r'], [weights['Q10']])) + bias
        np.testing.assert_allclose(weights['W1'] @ x + weights['b1'], expected, rtol=1e-14, atol=1e-14)

    def test_cost_three_branches_and_solver_parameters_unchanged(self):
        text = adapter.source(adapter.OLD / 'adversary.py')
        import r10_adversary  # populate the auditable edit ledger
        edits = [r for r in adapter.ADAPTATIONS if r['source'] == str(adapter.OLD / 'adversary.py')]
        for edit in edits:
            self.assertNotIn('for _ in range(3)', edit['before'])
            self.assertNotIn('NumericFocus', edit['before'])
            self.assertNotIn('REPLAY_TOLERANCE', edit['before'])
        changed = text
        for edit in edits:
            changed = changed.replace(edit['before'], edit['after'])
        self.assertIn('for _ in range(3)', changed)
        self.assertIn('addGenConstrMax(v[r], [g1, g2, g3])', changed)
        self.assertNotIn('for r in range(3)', changed)
        self.assertEqual(r10_adversary.PARAMETERS, dict(Threads=1, MIPGap=1e-4,
            FeasibilityTol=1e-9, OptimalityTol=1e-9, IntFeasTol=1e-9, NumericFocus=3, OutputFlag=0))
        self.assertEqual(strict.PHYSICAL_TOL, 1e-4)

    def test_protocol_only_authorized_changes_and_exact_counts(self):
        scope = dict(SIZES=(200, 800), SEEDS=tuple(range(10)), RADII=common.RADII,
                     LR_GRID=common.LR_GRID, MAX_MP=40)
        tree = ast.parse(adapter.source(adapter.GREEN / 'green_common.py'))
        node = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'PROTOCOL' for t in n.targets))
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<frozen-protocol>', 'exec'), scope)
        expected = dict(scope['PROTOCOL'], experiment=common.PROTOCOL['experiment'], R=10, sizes=[50,200,800])
        self.assertEqual(common.PROTOCOL, expected)
        self.assertEqual([common.metadata(n)['Ntrain'] for n in common.SIZES], [40,160,640])
        self.assertEqual([common.metadata(n)['Nval'] for n in common.SIZES], [10,40,160])
        self.assertEqual(len(common.SIZES)*len(common.SEEDS), 30)
        self.assertEqual(len(common.SIZES)*len(common.SEEDS)*len(common.LR_GRID), 90)
        self.assertEqual(len(common.SIZES)*len(common.SEEDS)*len(common.RADII), 270)
        self.assertEqual((core.baseline.cfg.c, core.baseline.cfg.h, core.baseline.cfg.b), (.25,.05,.5))
        historical = {}
        adapter.load_into(historical, adapter.R10_REFERENCE, names=('Cfg',))
        self.assertEqual((historical['Cfg'].R, historical['Cfg'].c, historical['Cfg'].h,
                          historical['Cfg'].b), (10,.25,.05,.5))

    def test_no_legacy_controllers_or_global_numerical_modules_imported(self):
        self.assertIs(core.train_model, adapter.forbidden)
        self.assertIs(common.legacy_ccg.run_configuration, adapter.forbidden)
        self.assertNotIn('train', vars(core.baseline))
        self.assertNotIn('solve_saa', vars(core.baseline))
        for name in ('core', 'ccg', 'adversary', '_retailer_definition_loader'):
            self.assertNotIn(name, sys.modules)
        self.assertFalse(hasattr(strict, 'gen'))
        self.assertFalse(hasattr(strict, 'evaluate_job'))

    def test_source_coverage_and_missing_new_module_rejected(self):
        from test_green import manifest
        bound = manifest()
        common.verify_sources(bound)
        actual = {p.name for p in common.ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.md')}
        self.assertEqual(set(adapter.SOURCE_NAMES), actual)
        for name in adapter.SOURCE_NAMES:
            copy = dict(bound, source_sha256=dict(bound['source_sha256']))
            del copy['source_sha256'][str(common.ROOT/name)]
            with self.assertRaisesRegex(ValueError, 'Missing computational source freeze'):
                common.verify_sources(copy)

    def test_grid_barriers_before_each_heldout_stage(self):
        # The local runner reuses this exact main, changing only dimensions/hash coverage.
        text = adapter.source(adapter.GREEN / 'run_campaign.py')
        self.assertLess(text.index('require_grid(nominal,'), text.index("run_stage('lr_validation'"))
        self.assertLess(text.index('require_grid(jobs,' , text.index('def main()')),
                        text.index("run_stage('radius_validation'"))
        expected = {(n,s,r) for n in common.SIZES for s in common.SEEDS for r in common.RADII}
        rows = [dict(Ntotal=n,seed=s,radius=r,status='completed',complete=True) for n,s,r in expected]
        campaign.require_grid(rows, expected, ('Ntotal','seed','radius'))
        with self.assertRaises(ValueError):
            campaign.require_grid(rows[:-1], expected, ('Ntotal','seed','radius'))

    def test_synthetic_41_pool_dimensions_and_membership(self):
        for n in (40,160,640):
            centers = np.ones((n,20), dtype=np.float64)
            pools = [c[None]+np.linspace(0.,1.5,41)[:,None] for c in centers]
            for c,p in zip(centers,pools):
                self.assertEqual(p.shape, (41,20))
                self.assertEqual(ccg.check_membership(p,c,1.5), 0.)
            # Exercise shape/order without a numerical projection/solver.
            class Fixture:
                def per_sample_cost(self, x1, x2):
                    self_shapes = (tuple(x1.shape),tuple(x2.shape))
                    if self_shapes != ((n*41,10),(n*41,10)):
                        raise AssertionError(self_shapes)
                    return (x1+x2).sum(1)
            np.testing.assert_array_equal(ccg.restricted_objective(pools)(Fixture()).numpy(), np.full(n,50.))

    def test_real_spawn_initializer_reused_worker_single_threads(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=common.configure_worker) as pool:
            first = pool.submit(spawn_probe).result()
            second = pool.submit(spawn_probe).result()
        self.assertEqual(first['pid'], second['pid'])
        self.assertEqual((first['task'], second['task']), (1,2))
        for row in (first,second):
            self.assertEqual((row['intra'],row['inter']), (1,1))
            self.assertEqual(row['env'], common.THREAD_ENV)
            self.assertEqual(row['value'], 3360.)

    def test_nested_cgroup_guard_and_new_job_caps(self):
        fixture = dict(hostname='fixture',platform='fixture',affinity=list(range(100)),
            loadavg=[0,0,0],memory={'MemAvailable_bytes':1000*1024**3},at_utc='fixture',
            cgroup_membership='fixture',cgroup_files={
                '/mount/cpu.max':'max 100000', '/mount/parent/cpu.max':'1200000 100000',
                '/mount/memory.max':str(100*1024**3),'/mount/memory.current':str(10*1024**3),
                '/mount/parent/memory.max':str(40*1024**3),'/mount/parent/memory.current':str(8*1024**3),
                '/mount/parent/cpuset.cpus.effective':'0-15'})
        with patch.object(resource_policy,'environment',return_value=fixture):
            snapshot = resource_policy.resources()
        self.assertEqual(snapshot['cpu_quota_cores'],12)
        self.assertEqual(snapshot['memory_available_bytes'],32*1024**3)
        self.assertEqual(len(snapshot['affinity']),16)
        workers = campaign.choose_workers(snapshot,dict(sizes=[dict(peak_rss_bytes=1)]))
        self.assertEqual(workers['ccg'],4)
        huge = dict(snapshot,affinity=list(range(1000)),cpu_quota_cores=None,
                    memory_available_bytes=10000*1024**3)
        workers = campaign.choose_workers(huge,dict(sizes=[dict(peak_rss_bytes=1)]))
        self.assertEqual(tuple(workers[k] for k in ('initialization','nominal','ccg','validation')), (30,90,270,30))

    def test_N50_controller_all40_no_validation(self):
        from test_green import ControllerTests
        case = ControllerTests()
        case.setUp()
        try:
            case.n_total = 50
            case.prepare_inputs()
            result = case.run_case(0.)
            self.assertEqual(result['status'],'completed',result['error'])
            self.assertEqual((result['Ntrain'],result['Nval'],case.ap_calls,case.train_calls), (40,10,40,0))
        finally:
            case.doCleanups()

if __name__ == '__main__':
    unittest.main()