"""Orchestration fixtures: SAA and training are mocked; no experiment launch."""
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import core
import run_lr_search as runner
from test_core import Tiny


class InlinePool:
    """Completed futures for testing parent collection without real workers."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as error:
            future.set_exception(error)
        return future


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.initial_states = []
        self.data = core.split_data(0)

    def fake_model(self, q10, q1r):
        model = Tiny()
        model.proj_layer = torch.nn.Identity()
        model.proj_layer.settings = {'fixture': True}
        return model

    def fake_from_state(self, state):
        self.initial_states.append({key: value.clone() for key, value in state.items()})
        model = Tiny()
        model.load_state_dict(state)
        return model

    def fake_saa(self, x1, x2, record):
        # Exact raw train40 inputs; no validation and no float32 round-trip.
        np.testing.assert_array_equal(x1, self.data['xi1'][self.data['train_indices']])
        np.testing.assert_array_equal(x2, self.data['xi2'][self.data['train_indices']])
        self.assertEqual(x1.dtype, np.float64)
        with core.timed(record):
            record.update(q10=2., q1r=[1., 1., 1.], solver_status=2, iterations=1)
        return 2., np.ones(3)

    def fake_train(self, model, loss, val1, val2, lr, folder, record, epochs, patience):
        self.assertEqual((epochs, patience), (3000, 300))
        torch.testing.assert_close(val1, self.data['val_x1'])
        torch.testing.assert_close(val2, self.data['val_x2'])
        expected = model.per_sample_cost(self.data['train_x1'], self.data['train_x2']).mean()
        torch.testing.assert_close(loss(model), expected)
        with core.timed(record), torch.no_grad():
            # All three fixtures tie on validation; smallest LR must win.
            record.update(best_val=float(model.per_sample_cost(val1, val2).mean()),
                          selected_training_loss=float(expected), best_epoch=1,
                          epochs_evaluated=1, optimizer_steps=1,
                          stop_reason='max_evaluated_epochs')
            torch.save(core.weights(model), folder / 'selected.pth')
        core.write_json(folder / 'training.json', record)
        return model

    def seed_patches(self):
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.object(core, 'solve_saa', side_effect=self.fake_saa))
        stack.enter_context(patch.object(core, 'make_model', side_effect=self.fake_model))
        stack.enter_context(patch.object(core, 'model_from_state', side_effect=self.fake_from_state))
        stack.enter_context(patch.object(core, 'train_model', side_effect=self.fake_train))
        return stack

    def test_seed_train_only_identical_initials_tie_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, self.seed_patches():
            record = runner.run_seed(0, directory, 'fixture-provenance')
            folder = Path(directory) / 'seed_0'
            self.assertTrue(record['complete'])
            self.assertEqual(record['winner']['lr_net'], .005)
            self.assertEqual(len(self.initial_states), 3)
            initial = torch.load(folder / 'initial.pth', weights_only=True)
            for state in self.initial_states:
                self.assertEqual(state.keys(), initial.keys())
                for key in initial:
                    torch.testing.assert_close(state[key], initial[key], rtol=0., atol=0.)
            split = json.loads((folder / 'split.json').read_text())
            self.assertEqual(split['raw_data_sha256'], runner.raw_hash(self.data))
            self.assertEqual(len(split['arrays']['train_indices']), 40)
            self.assertEqual(len(split['arrays']['val_indices']), 10)
            self.assertEqual(record['initial_sha256'], core.digest(folder / 'initial.pth'))
            for row in record['configurations']:
                self.assertEqual(row['method'], 'nominal_moreau')
                self.assertEqual((row['Ntotal'], row['Ntrain'], row['Nval'], row['radius']), (50, 40, 10, 0.))
                self.assertEqual(row['checkpoint_replay_check']['validation_abs_error'], 0.)
                for phase in ('training', 'restored_train_evaluation', 'restored_validation_evaluation'):
                    self.assertEqual(row[phase]['status'], 'completed')
                    self.assertGreaterEqual(row[phase]['wall_time_s'], 0.)
                self.assertGreaterEqual(row['wall_time_s'], row['training']['wall_time_s'])
                self.assertEqual(row['stop_status'], 'max_evaluated_epochs')
            summary = runner.summarize([record])
            self.assertEqual(summary['incomplete_seeds'], [])
            for row in summary['per_lr'].values():
                self.assertEqual(row['completed_count'], 1)
                self.assertEqual(row['runtime_s']['training']['count'], 1)

    def test_replay_failure_is_incomplete_continues_lrs_and_not_aggregated(self):
        original = self.fake_train

        def mismatch(*args, **kwargs):
            model = original(*args, **kwargs)
            if args[4] == .01:
                args[6]['best_val'] += 1e-3
            return model

        with tempfile.TemporaryDirectory() as directory, self.seed_patches(), patch.object(
                core, 'train_model', side_effect=mismatch):
            with self.assertRaisesRegex(RuntimeError, 'incomplete'):
                runner.run_seed(0, directory, 'fixture')
            saved = json.loads((Path(directory) / 'seed_0' / 'summary.json').read_text())
            self.assertEqual(len(self.initial_states), 3)
            self.assertFalse(saved['complete'])
            self.assertIsNone(saved['winner'])
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['configurations'][1]['stop_status'], 'failed')
            self.assertGreaterEqual(saved['configurations'][1]['wall_time_s'], 0.)
            summary = runner.summarize([saved])
            self.assertEqual(summary['per_lr']['0.01']['completed_count'], 0)
            self.assertIsNone(summary['per_lr']['0.01']['runtime_s']['training']['mean'])
            self.assertEqual(summary['per_lr']['0.05']['completed_count'], 1)
            self.assertEqual(len(summary['failed_or_incomplete_configurations']), 1)

    def test_zero_softplus_failure_is_not_repaired_or_retried(self):
        with tempfile.TemporaryDirectory() as directory, self.seed_patches(), patch.object(
                core, 'make_model', side_effect=FloatingPointError('Nonfinite learned state: Q10')) as make:
            with self.assertRaises(FloatingPointError):
                runner.run_seed(0, directory, 'fixture')
            self.assertEqual(make.call_count, 1)
            self.assertEqual(self.initial_states, [])
            saved = json.loads((Path(directory) / 'seed_0' / 'summary.json').read_text())
            self.assertFalse(saved['complete'])
            self.assertEqual(saved['initial_model']['status'], 'failed')
            self.assertEqual([row['status'] for row in saved['configurations']], ['not_run'] * 3)

    def test_parent_collects_all_ten_before_raising_and_exclusive_output(self):
        seen = []

        def seed_result(seed, output, provenance_hash):
            seen.append(seed)
            if seed == 0:
                raise RuntimeError('worker fixture failure')
            return dict(seed=seed, status='completed', complete=True, configurations=[], winner=None)

        with tempfile.TemporaryDirectory() as directory, patch.object(
                runner, 'OUTPUT', Path(directory) / 'results'), patch.object(
                runner, 'ProcessPoolExecutor', side_effect=InlinePool) as pool, patch.object(
                runner, 'provenance', return_value={'fixture': True}), patch.object(
                runner, 'run_seed', side_effect=seed_result):
            with self.assertRaisesRegex(RuntimeError, 'LR search failed'):
                runner.main()
            self.assertEqual(seen, list(range(10)))
            self.assertEqual(pool.call_args.kwargs['max_workers'], 10)
            self.assertEqual(pool.call_args.kwargs['mp_context'].get_start_method(), 'spawn')
            self.assertIs(pool.call_args.kwargs['initializer'], runner.configure_worker)
            path = runner.OUTPUT / 'summary.json'
            saved = json.loads(path.read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(len(saved['seed_results']), 10)
            self.assertEqual(len(saved['errors']), 1)
            self.assertGreaterEqual(saved['wall_time_s'], 0.)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.main()
            self.assertEqual(path.read_bytes(), before)

    def test_provenance_covers_loader_canonical_core_and_native(self):
        record = runner.provenance()
        for path in (Path(core.__file__).resolve(), core._BASELINE_PATH.resolve(),
                     core._loader.SOURCE.resolve()):
            self.assertEqual(record['source_sha256'][str(path)], core.digest(path))
        self.assertEqual(record['native_dependency']['version'], '0.4.0')
        self.assertTrue(any(path.endswith('.so') for path in record['source_sha256']))
        self.assertTrue(all(value == '1' for value in runner.THREAD_ENV.values()))


if __name__ == '__main__':
    unittest.main()