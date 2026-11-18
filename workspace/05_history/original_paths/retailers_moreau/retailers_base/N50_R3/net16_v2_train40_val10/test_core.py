"""Unittest fixtures only; no campaign, SAA optimization, or real training runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import core


class Tiny(torch.nn.Module):
    """Fixture with the optimizer's inventory/network interface, not a retailer net."""
    def __init__(self):
        super().__init__()
        self.fc_layers = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.fc_layers.weight.fill_(1.)
        self.Q10 = torch.nn.Parameter(torch.zeros(1))
        self.Q1r = torch.nn.Parameter(torch.zeros(3))

    def per_sample_cost(self, x1, x2):
        return (self.fc_layers(x1[:, :1]).squeeze(1) - x2[:, 0]).square()


class CoreTests(unittest.TestCase):
    def test_split_pair_conservation_and_independent_rng(self):
        original_rng = np.random.get_state()
        try:
            data = core.split_data(3)
            after = np.random.get_state()
            self.assertEqual(original_rng[0], after[0])
            np.testing.assert_array_equal(original_rng[1], after[1])
            self.assertEqual(original_rng[2:], after[2:])
            again = core.split_data(3)
            for key in data:
                np.testing.assert_array_equal(data[key], again[key])
            xi1, xi2 = core.baseline.generate_data(3)
            np.testing.assert_array_equal(data['xi1'], xi1)
            np.testing.assert_array_equal(data['xi2'], xi2)
            indices = np.random.default_rng(100003).permutation(50)
            np.testing.assert_array_equal(data['train_indices'], indices[:40])
            np.testing.assert_array_equal(data['val_indices'], indices[40:])
            self.assertFalse(set(data['train_indices']) & set(data['val_indices']))
            self.assertEqual(set(indices), set(range(50)))
            for split, n in [('train', 40), ('val', 10)]:
                for component in (1, 2):
                    tensor = data[f'{split}_x{component}']
                    self.assertEqual(tensor.dtype, torch.float32)
                    self.assertEqual(tuple(tensor.shape), (n, 3))
                    np.testing.assert_array_equal(tensor.numpy(),
                        data[f'xi{component}'][data[f'{split}_indices']].astype(np.float32))
        finally:
            np.random.set_state(original_rng)

    def test_native_active_inactive_finite_differences(self):
        layer = core.NativeProjection()
        z = torch.tensor([[3., 2., -1.], [.3, -.2, .4]], dtype=torch.float64, requires_grad=True)
        cap = torch.tensor([[2.], [4.]], dtype=torch.float64, requires_grad=True)
        probe = torch.tensor([[.2, .8, .4], [.7, .1, .3]], dtype=torch.float64)
        args = dict(solve_method='SCS', eps=1e-4, max_iters=200000)
        output, = layer(z, cap, args)
        self.assertEqual(output.dtype, torch.float64)
        np.testing.assert_allclose(output.detach().numpy(), [[1.5, .5, 0.], [.3, 0., .4]], atol=1e-8, rtol=0.)
        gradients = torch.autograd.grad((output * probe).sum(), (z, cap))
        h = 1e-5
        for variable, tensor in enumerate((z, cap)):
            numerical = torch.empty_like(tensor)
            for index in range(tensor.numel()):
                plus = [z.detach().clone(), cap.detach().clone()]
                minus = [z.detach().clone(), cap.detach().clone()]
                plus[variable].view(-1)[index] += h
                minus[variable].view(-1)[index] -= h
                with torch.no_grad():
                    numerical.view(-1)[index] = ((layer(*plus, args)[0] - layer(*minus, args)[0]) * probe).sum() / (2 * h)
            torch.testing.assert_close(gradients[variable], numerical, atol=1e-6, rtol=0.)
        summary = layer.drain_summary()
        self.assertEqual(summary['count'], 17)
        self.assertEqual(summary['sample_count'], 34)
        self.assertEqual(summary['statuses'], {1: 34})
        self.assertEqual(len(layer.calls), 0)
        self.assertEqual(layer.drain_summary()['count'], 0)

    def test_exact_capacity_no_buffer_and_explicit_adapter(self):
        layer = core.NativeProjection()
        z = torch.tensor([[3., 2., 1.]], dtype=torch.float64)
        cap = torch.tensor([[2.]], dtype=torch.float64)
        args = dict(solve_method='SCS', eps=1e-4, max_iters=200000)
        with torch.no_grad():
            q, = layer(z, cap, args)
        self.assertLess(abs(float(q.sum()) - 2.), 1e-8)
        self.assertGreater(abs(float(q.sum()) - (2. + 1e-6)), 9e-7)
        with self.assertRaises(ValueError):
            layer(z, cap, {**args, 'eps': 1e-3})

    def test_validation_forward_preserves_pending_native_training_gradients(self):
        """B40 graph survives B10 no-grad solve on the same projection layer."""
        args = dict(solve_method='SCS', eps=1e-4, max_iters=200000)
        reference, interleaved = core.NativeProjection(), core.NativeProjection()
        # Away from active-set boundaries; both target and capacity derivatives
        # are nonzero. Inputs match the float32 neural projection interface.
        targets = torch.tensor([[3., 2., -1.], [.3, -.2, .4]],
                               dtype=torch.float32).repeat(20, 1)
        capacities = torch.tensor([[2.], [4.]], dtype=torch.float32).repeat(20, 1)
        probe = torch.tensor([[.2, .8, .4], [.7, .1, .3]],
                             dtype=torch.float32).repeat(20, 1)
        observed = []
        for layer, insert_validation in ((reference, False), (interleaved, True)):
            z = targets.clone().requires_grad_()
            cap = capacities.clone().requires_grad_()
            output, = layer(z, cap, args)
            loss = (output * probe).sum()
            if insert_validation:
                with torch.no_grad():
                    layer(torch.tensor([[8., -2., 3.]], dtype=torch.float32).repeat(10, 1),
                          torch.full((10, 1), .7, dtype=torch.float32), args)
            observed.append(torch.autograd.grad(loss, (z, cap)))
        for expected, actual in zip(*observed):
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(bool(expected.abs().sum() > 0))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.)

        # Also check the complete identical retailer models' learned gradients.
        data = core.split_data(0)
        reference_model = core.make_model(2., np.ones(3))  # fixed fixture, no SAA
        interleaved_model = core.model_from_state(core.weights(reference_model))
        for candidate, insert_validation in ((reference_model, False), (interleaved_model, True)):
            candidate.train()
            loss = candidate.per_sample_cost(data['train_x1'], data['train_x2']).mean()
            if insert_validation:
                candidate.eval()
                with torch.no_grad():
                    candidate.per_sample_cost(data['val_x1'], data['val_x2'])
                candidate.train()
            loss.backward()
        for (name, expected), (other_name, actual) in zip(
                reference_model.named_parameters(), interleaved_model.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertIsNotNone(expected.grad, name)
            self.assertIsNotNone(actual.grad, name)
            torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=0.)

    def test_repeated_train40_validation10_backward_matches_fresh_reference(self):
        """Both cycles must finish; checking only the first backward missed the bug."""
        args = dict(solve_method='SCS', eps=1e-4, max_iters=200000)
        layer = core.NativeProjection()
        targets = torch.tensor([[3., 2., -1.], [.3, -.2, .4]],
                               dtype=torch.float32).repeat(20, 1)
        capacities = torch.tensor([[2.], [4.]], dtype=torch.float32).repeat(20, 1)
        probe = torch.tensor([[.2, .8, .4], [.7, .1, .3]],
                             dtype=torch.float32).repeat(20, 1)
        for cycle in range(2):
            with self.subTest(cycle=cycle + 1):
                # Change q and b between cycles to rule out cached solutions.
                z = (targets + .05 * cycle).requires_grad_()
                cap = (capacities + .1 * cycle).requires_grad_()
                ref_z = z.detach().clone().requires_grad_()
                ref_cap = cap.detach().clone().requires_grad_()
                reference = core.NativeProjection()
                expected, = reference(ref_z, ref_cap, args)
                expected_grads = torch.autograd.grad((expected * probe).sum(), (ref_z, ref_cap))
                actual, = layer(z, cap, args)
                with torch.no_grad():
                    val_z = torch.tensor([[8., -2., 3.]], dtype=torch.float32).repeat(10, 1)
                    val_cap = torch.full((10, 1), .7 + .1 * cycle, dtype=torch.float32)
                    val, = layer(val_z, val_cap, args)
                    val_reference = core.NativeProjection()
                    expected_val, = val_reference(val_z, val_cap, args)
                actual_grads = torch.autograd.grad((actual * probe).sum(), (z, cap))
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=0.)
                torch.testing.assert_close(val, expected_val, atol=1e-8, rtol=0.)
                for observed, wanted in zip(actual_grads, expected_grads):
                    self.assertEqual(observed.dtype, torch.float32)
                    self.assertTrue(bool(wanted.abs().sum() > 0))
                    torch.testing.assert_close(observed, wanted, atol=1e-6, rtol=0.)
        self.assertEqual(set(layer._solvers), {40, 10})
        self.assertIsNot(layer._solvers[40], layer._solvers[10])
        for batch_size, solver in layer._solvers.items():
            self.assertEqual(solver._impl._current_batch_size, batch_size)
            self.assertEqual(solver._impl._batch_size_change_count, 0)
            self.assertFalse(getattr(solver._impl, '_warned_backward_batch_change', False))
            self.assertEqual(core._json_value(solver._settings.model_dump(mode='python')),
                             layer.settings)
        summary = layer.drain_summary()
        self.assertEqual(summary['count'], 4)
        self.assertEqual(summary['sample_count'], 100)
        self.assertEqual(summary['statuses'], {1: 100})
        self.assertEqual(summary['iterations']['count'], 100)
        self.assertEqual(layer.drain_summary()['count'], 0)

    def test_two_native_training_updates_restore_learned_only_and_count_all_solves(self):
        data = core.split_data(0)
        model = core.make_model(2., np.ones(3))  # fixed fixture, no SAA
        initial_keys = set(core.weights(model))
        record = dict(seed=0, radius=0., method='two-update native lifecycle fixture')
        original_solve = core.Solver.solve
        with tempfile.TemporaryDirectory() as directory, patch.object(
                core.Solver, 'solve', autospec=True, side_effect=original_solve) as solve:
            core.train_model(model, lambda candidate: candidate.per_sample_cost(
                data['train_x1'], data['train_x2']).mean(), data['val_x1'], data['val_x2'],
                .01, directory, record, epochs=2)
            self.assertEqual(record['status'], 'completed')
            self.assertEqual(record['epochs_evaluated'], 2)
            self.assertEqual(record['optimizer_steps'], 2)
            self.assertEqual(record['projection_call_count'], 4)
            self.assertEqual(record['projection_sample_count'], 100)
            self.assertEqual(record['projection_statuses'], {1: 100})
            self.assertEqual(solve.call_count, 4)
            for call in solve.call_args_list:
                self.assertIn('warm_start', call.kwargs)
                self.assertIsNone(call.kwargs['warm_start'])
            selected = torch.load(Path(directory) / 'selected.pth', weights_only=True)
            self.assertEqual(set(selected), initial_keys)
            self.assertEqual(set(model.state_dict()), initial_keys)
            for key, value in core.weights(model).items():
                torch.testing.assert_close(value, selected[key], atol=0., rtol=0.)
            solvers = dict(model.proj_layer._solvers)
            core._load_learned(model, selected)
            self.assertEqual(model.proj_layer._solvers, solvers)
            replay = core.model_from_state(selected)
            self.assertEqual(replay.proj_layer._solvers, {})
            for split in ('train', 'val'):
                with torch.no_grad():
                    observed = model.per_sample_cost(data[f'{split}_x1'], data[f'{split}_x2'])
                    expected = replay.per_sample_cost(data[f'{split}_x1'], data[f'{split}_x2'])
                torch.testing.assert_close(observed, expected, atol=1e-6, rtol=0.)

    def test_one_update_preserves_preupdate_best_and_final_snapshot(self):
        model = Tiny()
        initial = core.weights(model)
        x1 = torch.ones(10, 3)
        x2 = torch.ones(10, 3)
        record = dict(seed=0, radius=0., method='one-update fixture')

        def train_loss(candidate):
            # Different fixture target: the update worsens validation deliberately.
            return candidate.per_sample_cost(x1, torch.zeros_like(x2)).mean()

        with tempfile.TemporaryDirectory() as directory:
            result = core.train_model(model, train_loss, x1, x2, .01, directory, record, epochs=1)
            self.assertIs(result, model)
            selected = torch.load(Path(directory) / 'selected.pth', weights_only=True)
            final = torch.load(Path(directory) / 'final_state.pth', weights_only=False)
            for key, tensor in initial.items():
                torch.testing.assert_close(core.weights(model)[key], tensor, atol=0., rtol=0.)
                torch.testing.assert_close(selected[key], tensor, atol=0., rtol=0.)
                torch.testing.assert_close(final['candidate_reference_model_state'][key], tensor, atol=0., rtol=0.)
            self.assertFalse(torch.equal(final['model']['fc_layers.weight'], initial['fc_layers.weight']))
            self.assertEqual(record['best_epoch'], 1)
            self.assertEqual(record['best_val'], 0.)
            self.assertEqual(record['selected_training_loss'], 1.)
            self.assertEqual(record['last_train'], 1.)
            self.assertEqual(record['optimizer_steps'], 1)
            self.assertEqual(final['optimizer_steps'], 1)
            self.assertEqual(final['scheduler']['last_epoch'], 1)
            self.assertTrue(final['optimizer']['state'])
            self.assertEqual(set(final['rng']), {'torch', 'numpy', 'python'})
            for state in final['optimizer']['state'].values():
                self.assertEqual(float(state['step']), 1.)
            log = [json.loads(line) for line in (Path(directory) / 'epochs.jsonl').read_text().splitlines()]
            self.assertEqual(len(log), 1)
            self.assertEqual(log[0]['validation_mean'], 0.)
            self.assertEqual(log[0]['train_loss'], 1.)
            self.assertGreater(log[0]['gradient_norm_preclip'], 1.)
            saved = json.loads((Path(directory) / 'training.json').read_text())
            self.assertEqual(saved['status'], 'completed')
            self.assertGreater(saved['wall_time_s'], 0.)
            self.assertGreater(saved['validation_time_s'], 0.)
            self.assertEqual(saved['train_step_times']['count'], 1)
            self.assertFalse((Path(directory) / 'failure_state.pth').exists())

    def test_failure_is_raised_and_persisted_as_strict_json(self):
        model = Tiny()
        record = dict(seed=0, radius=0., method='nonfinite fixture')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FloatingPointError):
                core.train_model(model, lambda candidate: candidate.fc_layers.weight.sum() * float('nan'),
                                 torch.ones(10, 3), torch.ones(10, 3), .01,
                                 directory, record, epochs=1)
            saved = json.loads((Path(directory) / 'training.json').read_text())
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['error']['type'], 'FloatingPointError')
            self.assertEqual(saved['optimizer_steps'], 0)
            self.assertGreater(saved['wall_time_s'], 0.)
            entry = json.loads((Path(directory) / 'epochs.jsonl').read_text())
            self.assertEqual(entry['train_loss'], 'NaN')
            self.assertTrue((Path(directory) / 'failure_state.pth').exists())
            self.assertFalse((Path(directory) / 'selected.pth').exists())

    def test_timed_records_and_reraises_base_exception(self):
        record = {}
        with self.assertRaises(KeyboardInterrupt):
            with core.timed(record):
                raise KeyboardInterrupt('fixture interruption')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['error']['type'], 'KeyboardInterrupt')
        self.assertGreaterEqual(record['wall_time_s'], 0.)
        self.assertIn('started_at_utc', record)
        self.assertIn('ended_at_utc', record)


if __name__ == '__main__':
    unittest.main()