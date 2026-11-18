"""Six solver-free unit tests; importing common still verifies frozen dependencies."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
from common import core, torch
import training


class Tiny(torch.nn.Module):
    def __init__(self, initial=1.):
        super().__init__()
        self.fc_layers = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.fc_layers.weight.fill_(initial)
        self.Q10 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.Q1r = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def per_sample_cost(self, *args, **kwargs):
        raise AssertionError('The trainer must not evaluate held-out costs')


def quadratic(model):
    return (model.fc_layers.weight.square().sum()
            + model.Q10.square().sum() + model.Q1r.square().sum())


class TrainingTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0., rtol=0.)

    def artifacts(self, directory):
        folder = Path(directory)
        record = json.loads((folder / 'training.json').read_text())
        entries = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
        return record, entries

    def test_one_epoch_incoming_selected_and_final_postupdate_state(self):
        model = Tiny()
        initial = core.weights(model)
        record = dict(seed=7, radius=.2, method='analytic fixture', mp_iteration=3,
                      input_hash='fixture-hash', parallel_workers=1)
        metadata = dict(record)
        adam = torch.optim.Adam
        with tempfile.TemporaryDirectory() as directory, patch.object(
                torch.optim, 'Adam', wraps=adam) as factory:
            result = training.train_model(model, quadratic, .1, directory, record, epochs=1)
            self.assertIs(result, model)
            factory.assert_called_once()
            selected = torch.load(Path(directory) / 'selected.pth', weights_only=True)
            final = torch.load(Path(directory) / 'final_state.pth', weights_only=False)
            saved, entries = self.artifacts(directory)
            self.assert_state_equal(selected, initial)
            self.assert_state_equal(core.weights(model), initial)
            self.assert_state_equal(final['candidate_reference_model_state'], initial)
            self.assert_state_equal(final['selected_model_state'], initial)
            self.assertFalse(torch.equal(final['model']['fc_layers.weight'], initial['fc_layers.weight']))
            self.assertEqual([g['lr'] for g in final['optimizer']['param_groups']], [.1, .01])
            self.assertEqual(len(final['optimizer']['state']), 3)
            for state in final['optimizer']['state'].values():
                self.assertEqual(float(state['step']), 1.)
            self.assertEqual(final['scheduler']['last_epoch'], 1)
            self.assertEqual(set(final['rng']), {'torch', 'numpy', 'python'})
            self.assertEqual(final['optimizer_steps'], 1)
            self.assertEqual(saved['best_epoch'], 1)
            for key in ('first_train', 'last_train', 'best_train', 'selected_training_loss'):
                self.assertEqual(saved[key], 1.)
            for key, value in metadata.items():
                self.assertEqual(saved[key], value)
            self.assertEqual(saved['epoch_history'], entries)
            self.assertEqual(entries[0]['preupdate_optimizer_steps'], 0)
            self.assertEqual(entries[0]['optimizer_steps'], 1)
            self.assertGreater(entries[0]['gradient_norm_preclip'], 1.)
            self.assertEqual(saved['status'], 'completed')
            self.assertEqual(saved['train_step_times']['count'], 1)
            self.assertEqual(saved['training_time_s'], saved['wall_time_s'])
            self.assertGreater(saved['training_time_s'], 0.)
            for key in ('forward_time_s', 'backward_time_s', 'update_time_s',
                        'checkpoint_time_s', 'optimizer_setup_time_s'):
                self.assertGreater(saved[key], 0.)
            self.assertIn('started_at_utc', saved)
            self.assertIn('ended_at_utc', saved)
            self.assertFalse(model.training)

    def test_selected_minimum_matches_evaluated_preupdate_epoch(self):
        observed = []

        def objective(model):
            observed.append(core.weights(model))
            return quadratic(model)

        with tempfile.TemporaryDirectory() as directory:
            model = Tiny()
            training.train_model(model, objective, .6, directory, {}, epochs=12)
            saved, entries = self.artifacts(directory)
            values = [float(state['fc_layers.weight'].square().sum()) for state in observed]
            best_index = min(range(len(values)), key=values.__getitem__)
            self.assertGreater(best_index, 0)
            self.assertEqual(saved['best_epoch'], best_index + 1)
            self.assertEqual(saved['best_train'], values[best_index])
            selected = torch.load(Path(directory) / 'selected.pth', weights_only=True)
            self.assert_state_equal(selected, observed[best_index])
            self.assert_state_equal(core.weights(model), selected)
            self.assertEqual(len(observed), 12)  # No objective replay after restoration.
            for index, entry in enumerate(entries):
                self.assertEqual(entry['epoch'], index + 1)
                self.assertEqual(entry['preupdate_optimizer_steps'], index)
                self.assertEqual(entry['optimizer_steps'], index + 1)
                self.assertEqual(entry['train_loss'], values[index])

    def test_incoming_optimum_and_subthreshold_changes_keep_first_checkpoint(self):
        for initial, scale in ((0., 1.), (1., 1e-6)):
            with self.subTest(initial=initial), tempfile.TemporaryDirectory() as directory:
                model = Tiny(initial)
                initial_state = core.weights(model)
                training.train_model(model, lambda candidate: scale * quadratic(candidate),
                                     .1, directory, {}, epochs=4)
                saved, entries = self.artifacts(directory)
                self.assertEqual(saved['best_epoch'], 1)
                self.assertEqual(saved['best_train'], scale * initial ** 2)
                self.assertEqual([entry['improved'] for entry in entries], [True, False, False, False])
                self.assertEqual(saved['no_improvement'], 3)
                self.assert_state_equal(core.weights(model), initial_state)
                if initial:
                    self.assertLess(saved['last_train'], saved['best_train'])
                    self.assertLess(saved['best_train'] - saved['last_train'], 1e-4)

    def test_patience_and_plateau_use_training_metric_including_final_update(self):
        scheduler_type = torch.optim.lr_scheduler.ReduceLROnPlateau
        original_step = scheduler_type.step
        with tempfile.TemporaryDirectory() as directory, patch.object(
                scheduler_type, 'step', autospec=True, side_effect=original_step) as step, patch.object(
                torch.optim.lr_scheduler, 'ReduceLROnPlateau', wraps=scheduler_type) as factory:
            training.train_model(Tiny(), lambda model: 0. * quadratic(model) + 1.,
                                 .1, directory, {}, epochs=305, patience=300)
            saved, entries = self.artifacts(directory)
            self.assertEqual(saved['stop_reason'], 'training_patience')
            self.assertEqual(saved['epochs_evaluated'], 301)
            self.assertEqual(saved['optimizer_steps'], 301)
            self.assertEqual(saved['no_improvement'], 300)
            self.assertEqual(step.call_count, 301)
            self.assertEqual([call.args[1] for call in step.call_args_list],
                             [entry['train_loss'] for entry in entries])
            self.assertEqual(factory.call_args.kwargs,
                             dict(mode='min', factor=.7, patience=150, threshold=1e-3))
            self.assertAlmostEqual(entries[-1]['learning_rates'][0], .07)
            self.assertAlmostEqual(entries[-1]['learning_rates'][1], .007)

    def test_api_and_artifacts_have_no_heldout_metrics_or_calls(self):
        signature = inspect.signature(training.train_model)
        self.assertEqual(list(signature.parameters),
                         ['model', 'loss_fn', 'lr', 'folder', 'record', 'epochs', 'patience'])
        self.assertEqual(signature.parameters['epochs'].default, 3000)
        self.assertEqual(signature.parameters['patience'].default, 300)

        def check_keys(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key != 'validation_time_s':
                        self.assertNotIn('validation', key)
                        self.assertFalse(key.startswith('val_') or key.endswith('_val'))
                    check_keys(item)
            elif isinstance(value, list):
                for item in value:
                    check_keys(item)

        with tempfile.TemporaryDirectory() as directory, patch.object(
                common, 'evaluate', side_effect=AssertionError('evaluation forbidden')) as evaluate, patch.object(
                core, 'train_model', side_effect=AssertionError('legacy trainer forbidden')) as old:
            folder = Path(directory) / 'new' / 'run'
            training.train_model(Tiny(), quadratic, .1, folder, {}, epochs=2)
            saved, entries = self.artifacts(folder)
            check_keys(saved)
            self.assertEqual(saved['validation_time_s'], 0.)
            self.assertTrue(all(entry['projection'] is None for entry in entries))
            self.assertEqual(saved['projection_call_count'], 0)
            evaluate.assert_not_called()
            old.assert_not_called()
            before = {path.name: path.read_bytes() for path in folder.iterdir()}
            with self.assertRaises(FileExistsError):
                training.train_model(Tiny(), quadratic, .1, folder, {}, epochs=1)
            self.assertEqual(before, {path.name: path.read_bytes() for path in folder.iterdir()})

    def test_partial_failures_preserve_state_json_and_raise_without_fallback(self):
        for kind in ('nonfinite', 'forward', 'backward'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                model = Tiny()
                seen = []

                def objective(candidate):
                    seen.append(core.weights(candidate))
                    loss = quadratic(candidate)
                    if len(seen) == 2:
                        if kind == 'nonfinite':
                            return loss * float('nan')
                        if kind == 'forward':
                            raise RuntimeError('forward failure')

                        def fail_backward(gradient):
                            raise RuntimeError('backward failure')

                        loss.register_hook(fail_backward)
                    return loss

                expected = FloatingPointError if kind == 'nonfinite' else RuntimeError
                with self.assertRaises(expected):
                    training.train_model(model, objective, .1, directory, {}, epochs=3)
                saved, entries = self.artifacts(directory)
                failure = torch.load(Path(directory) / 'failure_state.pth', weights_only=False)
                self.assertEqual(saved['status'], 'failed')
                self.assertEqual(saved['error']['type'], expected.__name__)
                self.assertEqual([entry['status'] for entry in entries], ['completed', 'failed'])
                self.assertEqual(saved['epoch_history'], entries)
                self.assertEqual(saved['optimizer_steps'], 1)
                self.assertEqual(saved['epochs_attempted'], 2)
                self.assertEqual(saved['epochs_evaluated'], 2 if kind == 'backward' else 1)
                self.assertEqual(saved['train_step_times']['count'], 1)
                self.assertEqual(saved['validation_time_s'], 0.)
                self.assertGreater(saved['training_time_s'], 0.)
                self.assertEqual(len(seen), 2)
                self.assert_state_equal(failure['model'], seen[1])
                self.assert_state_equal(core.weights(model), seen[1])
                self.assert_state_equal(failure['candidate_reference_model_state'], seen[0])
                self.assertIsNotNone(failure['selected_model_state'])
                self.assertEqual(failure['optimizer_steps'], 1)
                if kind == 'nonfinite':
                    self.assertEqual(entries[1]['train_loss'], 'NaN')
                self.assertFalse((Path(directory) / 'selected.pth').exists())
                self.assertFalse((Path(directory) / 'final_state.pth').exists())


if __name__ == '__main__':
    unittest.main()