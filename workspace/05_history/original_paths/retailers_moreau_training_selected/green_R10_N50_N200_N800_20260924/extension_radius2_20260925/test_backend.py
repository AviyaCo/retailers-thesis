"""Solver-free fixtures and exact-source AST checks. Parent runs this suite.

No experiment, training, Gurobi environment, Torch import, full freeze or test
data generation is performed by these tests. Reading frozen source is allowed.
"""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('radius2_backend_test_subject',
                                              Path(__file__).with_name('backend.py'))
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


class ProtocolTests(unittest.TestCase):
    def test_only_private_radius_and_experiment_change(self):
        original = b._read(b.ORIGINAL_PROVENANCE)['protocol']
        saved = deepcopy(original)
        private = b._extension_protocol(original)
        self.assertEqual(original, saved)
        self.assertEqual(private.pop('radii'), original['radii'] + [2.])
        self.assertEqual(private.pop('experiment'), b.EXPERIMENT)
        self.assertEqual(private, {k: v for k, v in original.items()
                                   if k not in ('radii', 'experiment')})
        self.assertEqual(original['batch'], 'full training set')
        self.assertEqual((original['max_mp'], original['max_epochs']), (40, 3000))

    def test_private_function_globals_and_original_immutability(self):
        protocol = b._read(b.ORIGINAL_PROVENANCE)['protocol']
        common = types.SimpleNamespace(PROTOCOL=deepcopy(protocol),
            metadata=lambda n, seed, radius, method: dict(Ntotal=n, Ntrain=4*n//5,
                                                         seed=seed, radius=radius, method=method))
        original = types.ModuleType('fixture_original')
        original.PROTOCOL = common.PROTOCOL
        original.check = b.check
        original.np = types.SimpleNamespace(isscalar=lambda x: isinstance(x, (int, float)),
                                            isfinite=math.isfinite)
        before = dict(vars(original))
        private = b._private_green(original, common, b.R3_GREEN.read_text())
        self.assertEqual(vars(original), before)
        self.assertNotIn(2., original.PROTOCOL['radii'])
        self.assertIn(2., private.PROTOCOL['radii'])
        for name in ('run_configuration', 'assess_candidate', '_flush', '_pool_mean',
                     'terminal_decision', 'choose_candidate'):
            self.assertIs(getattr(private, name).__globals__, vars(private))
        for n, count in ((50, 40), (200, 160), (800, 640)):
            self.assertEqual(private.metadata(n, 0, 2., 'CCG')['Ntrain'], count)
        self.assertEqual(private.metadata(50, 0, 2., 'CCG')['experiment'], b.EXPERIMENT)
        candidate = dict(assessed=True, ap_incumbent_mean=100.5, restricted_mean=100.)
        self.assertTrue(private.terminal_decision(candidate)['green_pass'])
        candidate['ap_incumbent_mean'] = 101.
        self.assertFalse(private.terminal_decision(candidate)['green_pass'])
        candidates = [dict(candidate_id=i, assessed=True, ap_incumbent_mean=x)
                      for i, x in enumerate((3., 2., 2., 1.))]
        self.assertEqual(private.choose_candidate(candidates, 2)['candidate_id'], 1)

    def test_green_adaptation_is_exact_and_definition_only(self):
        original = b._definitions(b.R3_GREEN.read_text())
        adapted = b._green_definitions(b.R3_GREEN.read_text())
        for before, after in reversed(b.GREEN_CHANGES):
            self.assertEqual(adapted.count(after), 1)
            adapted = adapted.replace(after, before)
        self.assertEqual(adapted, original)
        self.assertTrue(all(isinstance(n, ast.FunctionDef) for n in ast.parse(adapted).body))
        with self.assertRaises(RuntimeError):
            b._green_definitions(b.R3_GREEN.read_text().replace('paired6', 'unexpected'))

    def test_winner_rejects_warm_start_without_loading_checkpoint(self):
        winner = dict(Ntotal=50, seed=0, checkpoint='/nominal')
        summary = dict(status='completed', phases=dict(lr_validation=dict(
            jobs=[dict(Ntotal=50, seed=0, winner=winner)])))
        with patch.object(b, '_read', return_value=summary), patch.object(b, 'sha') as digest:
            with self.assertRaisesRegex(RuntimeError, 'SAME saved nominal'):
                b._saved_winner(50, 0, dict(winner, checkpoint='/radius_1.5/selected.pth'))
            digest.assert_not_called()

    def test_explicit_sizes_seeds_and_output_boundary(self):
        for n in (50, 200, 800):
            for seed in range(10):
                b._identity(n, seed)
        for n, seed in ((True, 0), (50, True), (51, 0), (50, 10)):
            with self.assertRaises(RuntimeError):
                b._identity(n, seed)
        with self.assertRaises(RuntimeError):
            b._output(b.SOURCE / 'forbidden')


class SourceTests(unittest.TestCase):
    def test_source_inventory_without_hashing_or_freezing(self):
        paths = set(b.source_paths())
        self.assertEqual(len(b.COMPARISON_NAMES), 13)
        self.assertEqual(len(b.TRAINING_NAMES), 16)
        self.assertTrue({b.CAMPAIGN / n for n in b.TRAINING_NAMES} <= paths)
        self.assertTrue({b.CAMPAIGN / 'comparison_20260925' / n
                         for n in b.COMPARISON_NAMES} <= paths)
        self.assertTrue({b.HERE / n for n in b.LOCAL_NAMES} <= paths)
        self.assertTrue({b.ORIGINAL_PROVENANCE, b.ORIGINAL_SUMMARY, b.SRO_SOURCE} <= paths)

    def test_source_manifest_rejects_omission_and_mutation(self):
        original = b._read(b.ORIGINAL_PROVENANCE)
        inventory = {str(p): 'fixture' for p in b.source_paths()}
        inventory.update(original['source_sha256'])
        manifest = dict(source_sha256=inventory, protocol=b._extension_protocol(original['protocol']),
            local_source_paths=[str(b.HERE / n) for n in b.LOCAL_NAMES],
            workers=dict(ccg=2, sro=3, evaluation=4))
        # Hash fixture only: never hash the real inventory in the unit suite.
        with patch.object(b, 'sha', side_effect=lambda p: inventory[str(p)]):
            b.verify_sources(manifest)
            missing = deepcopy(manifest)
            del missing['source_sha256'][str(b.HERE / 'test_backend.py')]
            with self.assertRaisesRegex(RuntimeError, 'Incomplete supplied source'):
                b.verify_sources(missing)
            changed = deepcopy(manifest)
            changed['source_sha256'][str(b.SRO_SOURCE)] = 'changed'
            with self.assertRaisesRegex(RuntimeError, 'Frozen source changed'):
                b.verify_sources(changed)
            changed = deepcopy(manifest)
            changed['protocol']['max_epochs'] = 10
            with self.assertRaisesRegex(RuntimeError, 'protocol mismatch'):
                b.verify_sources(changed)
        self.assertEqual(b._workers(manifest, 'sro'), 3)
        with self.assertRaises(RuntimeError):
            b._workers(dict(workers=True), 'ccg')


class SroSourceTests(unittest.TestCase):
    def setUp(self):
        self.text = b.SRO_SOURCE.read_text()

    def test_whitelist_has_no_top_level_import_or_main(self):
        tree = ast.parse(b._definitions(self.text, b.SRO_NAMES))
        self.assertEqual([n.name for n in tree.body], list(b.SRO_NAMES))
        self.assertTrue(all(isinstance(n, ast.FunctionDef) for n in tree.body))
        constants = [n for n in ast.parse(self.text).body if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) in ('R', '(C, H, B)')]
        self.assertEqual(len(constants), 2)

    def test_instrumentation_keeps_entire_lp_and_extraction(self):
        original = ast.parse(b._definitions(self.text, ('solve_sro_fa',))).body[0]
        changed = b._instrument_solve(self.text).body[0]
        self.assertIsInstance(changed.body[0], ast.Try)
        self.assertIn('_capture_solver', ast.unparse(changed.body[0].finalbody[0]))
        changed.body = changed.body[0].body
        construction = deepcopy(changed.body[0].body)
        for node in construction:
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in ('env', 'model'):
                self.assertEqual(ast.unparse(node.value.func), '_stack.enter_context')
                node.value = node.value.args[0]
        count = len(construction)
        self.assertEqual([ast.dump(n) for n in construction],
                         [ast.dump(n) for n in original.body[:count]])
        self.assertEqual(ast.unparse(original.body[count]), 'solve_start = time.time()')
        optimize = changed.body[1].body[0]
        self.assertIsInstance(optimize, ast.Try)
        self.assertEqual(ast.unparse(optimize.body[0]), 'model.optimize()')
        self.assertIn('_capture_solver', ast.unparse(optimize.finalbody[0]))
        gate = changed.body[3]
        self.assertEqual(ast.unparse(gate.test), 'model.status != GRB.OPTIMAL')
        self.assertIsInstance(gate.body[-1], ast.Raise)
        self.assertEqual([ast.dump(n) for n in changed.body[4].body],
                         [ast.dump(n) for n in original.body[count+4:count+8]])
        self.assertEqual(ast.dump(changed.body[-1]), ast.dump(original.body[-1]))
        code = ast.unparse(changed)
        self.assertNotIn('time.time()', code)
        self.assertNotIn('.dispose()', code)
        self.assertIn("env.setParam('Threads', 1)", code)
        self.assertIn('np.where(np.isinf(part_hi), 1000000000.0, part_hi)', code)

    def test_sro_evaluation_cell_selection_never_uses_xi2(self):
        evaluate = ast.parse(b._definitions(self.text, ('evaluate',))).body[0]
        assignment_index = next(i for i, n in enumerate(evaluate.body)
                                if isinstance(n, ast.Assign)
                                and ast.unparse(n.targets[0]) == 'q2_test')
        prefix = ast.Module(body=evaluate.body[:assignment_index+1], type_ignores=[])
        self.assertNotIn('xi2_test', {n.id for n in ast.walk(prefix) if isinstance(n, ast.Name)})
        self.assertIn('np.argmax(membership, axis=1)', ast.unparse(prefix))

    def test_no_test_loading_in_training_definitions(self):
        tree = ast.parse(Path(b.__file__).read_text())
        for name in ('green_job', 'sro_train'):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            calls = [ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)]
            self.assertFalse(any('evaluate' in n or 'sro_test' in n for n in calls))
        solve_text = self.text.replace('solve_start = time.time()', 'solve_start = 0')
        with self.assertRaises(RuntimeError):
            b._instrument_solve(solve_text)


class TimingTests(unittest.TestCase):
    def test_failed_phase_records_monotonic_duration_and_reraises(self):
        record = {}
        with patch.object(b.time, 'perf_counter', side_effect=[5., 8.]):
            with self.assertRaisesRegex(ValueError, 'solver failed'):
                with b._phase(record, 'optimization'):
                    raise ValueError('solver failed')
        self.assertEqual(record['optimization_time_s'], 3.)
        self.assertEqual(record['phases']['optimization']['status'], 'failed')

    def test_failure_solver_metadata_has_iterations_and_barrier(self):
        model = types.SimpleNamespace(Status=9, Runtime=12., IterCount=4., BarIterCount=2)
        record = {}
        b._capture_solver(model, record, types.SimpleNamespace(GurobiError=RuntimeError))
        self.assertEqual(record, dict(solver_status=9, solver_runtime_s=12.,
                                     iterations=4., barrier_iterations=2))

    def test_missing_solver_metadata_is_not_zero(self):
        record = {}
        b._capture_solver(object(), record, types.SimpleNamespace(GurobiError=RuntimeError))
        self.assertIsNone(record['solver_runtime_s'])
        self.assertIn('barrier_iterations', record['unavailable_solver_attributes'])

    def test_json_rejects_nonfinite_partition_bounds(self):
        # Infinite partitions belong in NPZ, never non-standard JSON literals.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                b._write(Path(directory) / 'record.json', {'part_hi': math.inf})


if __name__ == '__main__':
    unittest.main()