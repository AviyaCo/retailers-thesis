"""Solver-free regressions and a separately invoked bounded Gurobi fixture.

Never generate seed9999 data. The legacy evaluator is AST-loaded definitions
only; its top-level CLI, test generator, cache writes and campaigns never run.
"""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import compare as c
import numpy as np
import torch


def legacy_definitions(names):
    tree = ast.parse(c.OLD_EVAL.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
             and node.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError('Missing legacy definitions')
    scope = dict(np=np, torch=torch, nn=torch.nn, R=3, c=.25, h=.05, b=.5,
                 truncnorm=c.truncnorm, gp=c.gp, GRB=c.GRB)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(c.OLD_EVAL), 'exec'), scope)
    return scope


def legacy_selection(rho, seed, directories):
    """Execute the original per-seed loop, replacing ALL numerical work by fixtures."""
    tree = ast.parse(c.OLD_EVAL.read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.For) and
                 isinstance(n.target, ast.Name) and n.target.id == 'rho')
    loop = next(n for n in outer.body if isinstance(n, ast.For) and
                isinstance(n.target, ast.Name) and n.target.id == 'seed')
    loaded = []

    class FixtureModel:
        def get_Q2(self, _):
            return np.zeros((1, 3)), 0., np.zeros(3)

        def state_dict(self):
            return {}

    def load(path):
        loaded.append(Path(path))
        return FixtureModel()

    src = ('sweep', str(directories[rho])) if rho in directories else ('saainit',)
    if rho == 0:
        src = ('nominal',)
    scope = dict(SEEDS=[seed], N_TRAIN=50, gen=lambda *_: (np.zeros((1, 3)), np.zeros((1, 3))),
                 torch=torch, os=c.os, json=json, NOMINAL_DIR=str(c.NOMINAL),
                 load_model=load, mean_cost=lambda *_: 0., cvar_cost=lambda *_: 0.,
                 exact_mp_is=lambda *_: 0., xi1_test_t=torch.zeros((1, 3)),
                 xi1_test=np.zeros((1, 3)), xi2_test=np.zeros((1, 3)),
                 rob_is=[], rob_oos=[], rob_cvar=[], nom_is=[], nom_oos=[], n_ok=0,
                 rho=rho, src=src, FINAL_ITER_040=c.FINAL_ITER_040, print=lambda *_args, **_kw: None)
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(c.OLD_EVAL), 'exec'), scope)
    if scope['n_ok'] != 1:
        raise AssertionError('Legacy source skipped a checkpoint')
    return loaded[-1]


class SolverFreeTests(unittest.TestCase):
    def test_original_plotted_domain_and_all_230_mappings(self):
        cache = c.read_json(c.CACHE)
        tree = ast.parse((c.OLD / 'plot_combined_N50_R3.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_curve')
        scope = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<original curve extractor>', 'exec'), scope)
        expected = scope['extract_curve'](cache)[0].tolist()
        self.assertEqual(c.plotted_radii(cache), expected)
        self.assertEqual(len(expected), 23)
        directories = {float(p.name[4:].replace('p', '.')): p for p in c.SWEEP.glob('rho_*') if p.is_dir()}
        for rho in expected:
            for seed in c.SEEDS:
                with self.subTest(rho=rho, seed=seed):
                    self.assertEqual(c.old_choice(rho, seed, directories)[0],
                                     legacy_selection(rho, seed, directories))

    def test_nominal_zero_missing_key_and_special_point_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = root / 'seed_3_summary.json'
            for data, iteration in [({}, 0), ({'final_mp_iter': 0}, 0), ({'final_mp_iter': 7}, 7)]:
                summary.write_text(json.dumps(data))
                path, actual, _ = c.old_choice(.4, 3, {.4: root}, nominal=root)
                self.assertEqual(actual, iteration)
                self.assertEqual(path.name, 'seed_3_nominal.pth' if iteration == 0 else 'seed_3_mp_iter7.pth')
            self.assertEqual(c.old_choice(.4, 3, {}, nominal=root)[1], c.FINAL_ITER_040[3])
            self.assertEqual(c.old_choice(0, 3, {}, nominal=root)[1], 0)
            with self.assertRaises(FileNotFoundError):
                c.old_choice(.2, 3, {}, nominal=root)
            summary.write_text('{"final_mp_iter": -1}')
            with self.assertRaises(ValueError):
                c.old_choice(.4, 3, {.4: root}, nominal=root)

    def test_exact_generator_parity_non_test_seed_only(self):
        legacy = legacy_definitions({'gen'})
        with patch.object(c, 'TEST_SEED', 9999):
            a = c.gen(7, 12345)
            b = legacy['gen'](7, 12345)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
            self.assertEqual(x.dtype, np.float64)

    def test_targets_exact_legacy_float32_and_only_xi1(self):
        original = legacy_definitions({'Model'})
        captured = []
        original['_gurobi_project'] = lambda z, q10: captured.append((z.copy(), q10)) or z
        torch.manual_seed(123)
        model = c.Model()
        old = original['Model']()
        old.load_state_dict(model.state_dict(), strict=True)
        x = np.array([[.123456789, 7., 13.], [4., 5., 6.]], dtype=np.float64)
        expected_z, expected_q10, expected_q1r = old.get_Q2(torch.tensor(x, dtype=torch.float32))
        z, q10, q1r = model.targets(x)
        np.testing.assert_array_equal(z, expected_z)
        np.testing.assert_array_equal(q1r, expected_q1r)
        self.assertEqual(q10, expected_q10)
        self.assertEqual(z.dtype, np.float32)
        self.assertEqual(q1r.dtype, np.float32)
        self.assertEqual(len(captured), 2)
        self.assertEqual(list(__import__('inspect').signature(model.targets).parameters), ['xi1'])

    def test_cost_exact_legacy_dtype_and_regimes(self):
        old = legacy_definitions({'scenario_costs'})['scenario_costs']
        q1r = np.array([1.234567, 10., 2.], dtype=np.float32)
        x1 = np.array([[1., 4., 20.], [0., 30., 1.]], dtype=np.float64)
        x2 = np.array([[20., 2., 0.], [0., 0., 5.]], dtype=np.float64)
        q2 = np.array([[.5, 1., 2.], [0., 3., 4.]], dtype=np.float64)
        actual = c.scenario_costs(9.12345, q1r, q2, x1, x2)
        np.testing.assert_array_equal(actual, old(9.12345, q1r, q2, x1, x2))
        self.assertEqual(actual.dtype, np.float64)
        # Three terms chosen deliberately: demand branch, inventory branch, stage1 branch.
        for x1v, x2v, q1v, q2v, recourse in [(10., 10., 0., 0., 10.),
                                             (0., 0., 10., 0., .5),
                                             (10., 0., 0., 10., 5.)]:
            value = c.scenario_costs(20., np.full(3, q1v, dtype=np.float32),
                                    np.full((1, 3), q2v), np.full((1, 3), x1v), np.full((1, 3), x2v))
            self.assertAlmostEqual(float(value[0]), .25 * (20. + 3 * q1v) + 1. + 3 * recourse)

    def test_strict_loading_missing_extra_dtype_shape_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'checkpoint.pth'
            model = c.Model()
            state = model.state_dict()
            torch.save(state, path)
            self.assertEqual(c.state_hash(c.load_model(path, 'new')), c.state_hash(model))
            variants = [dict(state), dict(state), dict(state), dict(state), dict(state)]
            del variants[0]['Q10']
            variants[1]['unknown'] = torch.zeros(1)
            variants[2]['Q10'] = torch.zeros(1, dtype=torch.float64)
            variants[3]['Q1r'] = torch.zeros(4)
            variants[4]['Q10'] = torch.tensor([float('nan')])
            for broken in variants:
                torch.save(broken, path)
                with self.assertRaises(ValueError):
                    c.load_model(path, 'new')

    def test_projection_failures_no_suboptimal_or_zero_fallback(self):
        for status, solution in [(c.GRB.SUBOPTIMAL, [0., 0., 0.]),
                                 (c.GRB.INFEASIBLE, [0., 0., 0.]),
                                 (c.GRB.OPTIMAL, [np.nan, 0., 0.]),
                                 (c.GRB.OPTIMAL, [2., 2., 2.])]:
            model = MagicMock()
            model.Status, model.IterCount, model.BarIterCount, model.ObjVal = status, 1., 2, 0.
            variables = [MagicMock() for _ in range(3)]
            for v, x in zip(variables, solution):
                v.X = x
            model.addVars.return_value = variables
            wrapper = MagicMock()
            wrapper.__enter__.return_value = model
            diagnostic = {}
            with patch.object(c.gp, 'Model', return_value=wrapper), patch.object(c.gp, 'quicksum', return_value=0.):
                with self.assertRaises((RuntimeError, FloatingPointError)):
                    c.strict_project(np.ones(3), 1., None, diagnostic)
            self.assertFalse(diagnostic['accepted'])
            self.assertEqual(diagnostic['status'], status)
            self.assertGreaterEqual(diagnostic['wall_time_s'], 0)
            self.assertEqual(model.optimize.call_count, 1)

    def test_manifest_gate_before_test_read(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(c, 'RESULTS', Path(tmp) / 'results'):
            with patch.object(c, 'verify_manifest', side_effect=ValueError('bad freeze')), \
                    patch.object(c, 'gen') as gen:
                with self.assertRaisesRegex(ValueError, 'bad freeze'):
                    c.run()
                gen.assert_not_called()
            result = c.read_json(c.RESULTS / 'summary.json')
            self.assertEqual(result['status'], 'failed')
            self.assertGreaterEqual(result['wall_time_s'], 0)

    def test_full_grid_gate_rejects_missing_and_failed(self):
        manifest = {'jobs': [{'id': 'old_r1_s0'}]}
        with self.assertRaises(ValueError):
            c.complete_grid({'jobs': []}, manifest)
        with self.assertRaises(ValueError):
            c.complete_grid({'jobs': [{'id': 'old_r1_s0', 'status': 'failed'}]}, manifest)

    def test_worker_failure_persists_solve_and_no_cost_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_model = c.Model()
            path = root / 'selected.pth'
            torch.save(state_model.state_dict(), path)
            test_info = {}
            for key in ('xi1', 'xi2'):
                path = root / f'{key}.npy'
                np.save(path, np.zeros((2, 3), dtype=np.float64))
                test_info[key] = dict(file=path.name, file_sha256=c.sha(path))
            job = dict(id='fixture', method='new', radius=.2, seed=0,
                       checkpoint_copy='selected.pth', checkpoint_sha256=c.sha(root / 'selected.pth'),
                       state_sha256=c.state_hash(state_model))

            def fail(_z, _capacity, _env, diagnostic):
                diagnostic.update(status=c.GRB.SUBOPTIMAL, iterations=3., barrier_iterations=0,
                                  wall_time_s=.001, accepted=False, error='injected fixture failure')
                raise RuntimeError('injected fixture failure')

            settings = MagicMock()
            settings.getParamInfo.return_value = ('test', float, 1e-6)
            settings.__enter__.return_value = settings
            with patch.object(c, 'FROZEN', root), patch.object(c, 'N_TEST', 2), \
                    patch.object(c.gp, 'Env'), patch.object(c.gp, 'Model', return_value=settings), \
                    patch.object(c, 'strict_project', side_effect=fail) as projection, \
                    patch.object(c, 'scenario_costs') as cost:
                record = c.evaluate_job(job, 'fixture_manifest', test_info, root)
            self.assertEqual(record['status'], 'failed')
            self.assertEqual(record['status_counts'], {str(c.GRB.SUBOPTIMAL): 1})
            self.assertEqual(record['solves_attempted'], 1)
            self.assertEqual(record['solves_accepted'], 0)
            self.assertIsNone(record['mean_oos'])
            projection.assert_called_once()
            cost.assert_not_called()
            self.assertEqual(len((root / 'fixture/solves.jsonl').read_text().splitlines()), 1)
            self.assertEqual(c.read_json(root / 'fixture/result.json')['status'], 'failed')
            self.assertEqual(record['phases']['projection_qp']['status'], 'failed')
            self.assertGreaterEqual(record['wall_time_s'], 0)


class GurobiPreflight(unittest.TestCase):
    diagnostics = []

    def test_five_fresh_qps_exact_legacy_and_known_solutions(self):
        # Hand-chosen numerical fixtures, NOT held-out demand observations.
        fixtures = [([-2., -1., -3.], 4., [0., 0., 0.]),
                    ([1., 2., 3.], 10., [1., 2., 3.]),
                    ([3., 2., 1.], 3., [2., 1., 0.]),
                    ([2., 2., 2.], 0., [0., 0., 0.]),
                    ([4., -3., 2.], 2., [2., 0., 0.])]
        original = legacy_definitions({'_gurobi_project'})
        with c.gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            original['_get_grb_env'] = lambda: env
            for target, capacity, expected in fixtures:
                diagnostic = dict(target=target, capacity=capacity)
                self.diagnostics.append(diagnostic)
                actual = c.strict_project(np.array(target, dtype=np.float32), capacity, env, diagnostic)
                start = c.time.perf_counter()
                legacy = original['_gurobi_project'](np.array(target, dtype=np.float32), capacity)
                diagnostic['legacy_reference_wall_time_s'] = c.time.perf_counter() - start
                diagnostic['exact_legacy_parity'] = bool(np.array_equal(actual, legacy))
                np.testing.assert_array_equal(actual, legacy)
                np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
                self.assertEqual(diagnostic['status'], c.GRB.OPTIMAL)
                self.assertLessEqual(diagnostic['physical_residual'], c.PHYSICAL_TOL)


if __name__ == '__main__':
    unittest.main()