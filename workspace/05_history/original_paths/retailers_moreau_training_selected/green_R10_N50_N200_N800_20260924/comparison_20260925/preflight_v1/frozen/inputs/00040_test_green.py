"""Solver-free analytic/mocked tests, not research results or real preflight.

Run with Python -B to avoid writing bytecode into frozen source directories.
Only temporary-directory fixtures are written. Never solve an LP/QP/MIP here.
"""
from contextlib import ExitStack
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import green_common as common
import green_ccg as ccg
import green_training as training

core, np, torch = common.core, common.np, common.torch


def manifest():
    sources = dict(common.FROZEN)
    for name in ('green_common.py', 'green_training.py', 'green_ccg.py', 'test_green.py'):
        path = common.ROOT / name
        sources[str(path)] = core.digest(path)
    return dict(source_sha256=sources, protocol=common.PROTOCOL,
                workers=dict(nominal=2, ccg=3))


class CommonTests(unittest.TestCase):
    def test_split_both_sizes_all_seeds_exact_generator_and_rng_restoration(self):
        before = np.random.get_state()
        for n in common.SIZES:
            for seed in common.SEEDS:
                with self.subTest(n=n, seed=seed):
                    data = common.split_data(n, seed)
                    after = np.random.get_state()
                    self.assertEqual(before[0], after[0])
                    np.testing.assert_array_equal(before[1], after[1])
                    self.assertEqual(before[2:], after[2:])
                    # Compare the unchanged canonical generator itself, not the
                    # historical fixed40 split/controller. Restore both globals.
                    with patch.object(core.baseline.cfg, 'N_SAMPLES', n):
                        try:
                            a, b = core.baseline.generate_data(seed)
                        finally:
                            np.random.set_state(before)
                    np.testing.assert_array_equal(data['xi1'], a)
                    np.testing.assert_array_equal(data['xi2'], b)
                    self.assertEqual(a.dtype, np.float64)
                    self.assertEqual(b.dtype, np.float64)
                    indices = np.random.default_rng(100000 + seed).permutation(n)
                    train, val = data['train_indices'], data['val_indices']
                    np.testing.assert_array_equal(train, indices[:4 * n // 5])
                    np.testing.assert_array_equal(val, indices[4 * n // 5:])
                    self.assertFalse(set(train) & set(val))
                    self.assertEqual(len(set(train) | set(val)), n)
                    for split, ids in (('train', train), ('val', val)):
                        for index in (1, 2):
                            tensor = data[f'{split}_x{index}']
                            self.assertEqual(tensor.dtype, torch.float32)
                            np.testing.assert_array_equal(tensor.numpy(),
                                data[f'xi{index}'][ids].astype(np.float32))
                    self.assertEqual(common.raw_hash(data), common.raw_hash(common.split_data(n, seed)))
        self.assertEqual(core.baseline.cfg.N_SAMPLES, 50)

    def test_metadata_grid_and_worker_setup(self):
        self.assertEqual(common.SIZES, (200, 800))
        self.assertEqual(common.SEEDS, tuple(range(10)))
        self.assertEqual(common.RADII, (0., .001, .003, .01, .03, .2, .4, .7, 1.5))
        for n in common.SIZES:
            info = common.metadata(n, 9, .7, 'fixture')
            self.assertEqual((info['Ntrain'], info['Nval']), (4 * n // 5, n // 5))
            self.assertEqual((info['R'], info['f'], info['network_seed'], info['split_seed']),
                             (3, 0, 0, 100009))
        with patch.object(torch, 'set_num_threads') as intra, patch.object(
                torch, 'set_num_interop_threads') as inter:
            common.configure_worker()
            intra.assert_called_once_with(1)
            inter.assert_called_once_with(1)

    def test_frozen_sources_include_loader_and_definition(self):
        frozen = manifest()
        self.assertIn(str(core._BASELINE_PATH), frozen['source_sha256'])
        self.assertIn(str(core._loader.SOURCE), frozen['source_sha256'])
        common.verify_sources(frozen)
        del frozen['source_sha256'][str(core._loader.SOURCE)]
        with self.assertRaisesRegex(ValueError, 'historical freeze'):
            common.verify_sources(frozen)

    def test_pool_aggregation_both_sizes_and_exact_identity(self):
        model = Policy()
        for n in (160, 640):
            pools = [np.ones((1 + i % 3, 6), dtype=np.float64) for i in range(n)]
            for pool in pools:
                pool[-1, 0] = 2.
            record = {}
            values = ccg._pool_mean(model, pools, record)
            np.testing.assert_array_equal(values, np.full(n, 22.))
            self.assertEqual(record['sample_count'], n)
            self.assertEqual(record['mean'], 22.)
        center = np.ones(6, dtype=np.float64)
        near = center.copy()
        near[0] += 5e-10
        details = ccg.point_membership(near, center[None])
        self.assertFalse(details['exact_duplicate'])
        self.assertTrue(details['near_duplicate'])
        self.assertTrue(details['float32_collision'])
        self.assertNotEqual(ccg.pool_hash([center[None]]),
                            ccg.pool_hash([np.vstack([center, near])]))

    def test_saa_rejects_fixed40_without_invoking_solver(self):
        record = {}
        with patch.object(core.gp, 'Env', side_effect=AssertionError('No solver permitted')):
            with self.assertRaisesRegex(ValueError, 'approved train80'):
                common.solve_saa(np.ones((40, 3)), np.ones((40, 3)), record)
        self.assertEqual(record['status'], 'failed')
        self.assertGreaterEqual(record['wall_time_s'], 0.)


class DecisionTests(unittest.TestCase):
    @staticmethod
    def candidate(index, ap, pool=100.):
        return dict(candidate_id=index, assessed=True, ap_incumbent_mean=ap,
                    restricted_mean=pool)

    def test_initial_eligible_strict_boundary_absolute_and_floor(self):
        self.assertTrue(ccg.terminal_decision(self.candidate(0, 100.9))['green_pass'])
        self.assertFalse(ccg.terminal_decision(self.candidate(0, 101.))['green_pass'])
        self.assertFalse(ccg.terminal_decision(self.candidate(0, 99.))['green_pass'])
        self.assertTrue(ccg.terminal_decision(self.candidate(0, 0., 0.))['green_pass'])
        self.assertFalse(ccg.terminal_decision(self.candidate(0, 2e-11, 0.))['green_pass'])

    def test_inclusive_exact_ties_earliest_and_no_future_read(self):
        class Future:
            def __getitem__(self, key):
                raise AssertionError('Future row accessed')
        rows = [self.candidate(0, 2.), self.candidate(1, 1.), self.candidate(2, 1.), Future()]
        self.assertIs(ccg.choose_candidate(rows, 2), rows[1])
        rows[2]['ap_incumbent_mean'] = .5
        self.assertIs(ccg.choose_candidate(rows, 2), rows[2])
        self.assertIs(ccg.choose_candidate(rows, 0), rows[0])

    def test_incomplete_or_nonfinite_prefix_rejected(self):
        row = self.candidate(0, 100.)
        row['assessed'] = False
        with self.assertRaises(ValueError):
            ccg.terminal_decision(row)
        with self.assertRaises(ValueError):
            ccg.choose_candidate([row], 0)
        row.update(assessed=True, ap_incumbent_mean=float('nan'))
        with self.assertRaises(ValueError):
            ccg.terminal_decision(row)


class Tiny(torch.nn.Module):
    def __init__(self, initial=1.):
        super().__init__()
        self.fc_layers = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.fc_layers.weight.fill_(initial)
        self.Q10 = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.Q1r = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))

    def per_sample_cost(self, *args, **kwargs):
        raise AssertionError('Trainer must not evaluate held-out costs')


def quadratic(model):
    return (model.fc_layers.weight.square().sum()
            + model.Q10.square().sum() + model.Q1r.square().sum())


class TrainingTests(unittest.TestCase):
    def test_incoming_preupdate_checkpoint_fresh_adam_final_update_and_timing(self):
        for n in common.SIZES:
            with self.subTest(n=n), tempfile.TemporaryDirectory() as directory:
                model = Tiny()
                initial = core.weights(model)
                record = common.metadata(n, 0, .2, 'analytic fixture')
                with patch.object(torch.optim, 'Adam', wraps=torch.optim.Adam) as factory:
                    training.train_model(model, quadratic, .1, directory, record, epochs=1)
                    factory.assert_called_once()
                folder = Path(directory)
                final = torch.load(folder / 'final_state.pth', weights_only=False)
                selected = torch.load(folder / 'selected.pth', weights_only=True)
                for key in initial:
                    torch.testing.assert_close(selected[key], initial[key], atol=0., rtol=0.)
                    torch.testing.assert_close(core.weights(model)[key], initial[key], atol=0., rtol=0.)
                self.assertFalse(torch.equal(final['model']['fc_layers.weight'], initial['fc_layers.weight']))
                self.assertEqual([g['lr'] for g in final['optimizer']['param_groups']], [.1, .01])
                self.assertEqual(record['n_train'], 4 * n // 5)
                self.assertEqual(record['best_epoch'], 1)
                self.assertEqual(record['optimizer_steps'], 1)
                self.assertEqual(record['status'], 'completed')
                self.assertGreater(record['training_time_s'], 0.)
                self.assertEqual(record['training_time_s'], record['wall_time_s'])
                self.assertEqual(record['epoch_history'], [json.loads(line) for line in
                    (folder / 'epochs.jsonl').read_text().splitlines()])
                self.assertEqual(final['scheduler']['last_epoch'], 1)
                before = {p.name: p.read_bytes() for p in folder.iterdir()}
                with self.assertRaises(FileExistsError):
                    training.train_model(model, quadratic, .1, directory, record, epochs=1)
                self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})

    def test_threshold_and_preupdate_minimum(self):
        for scale, epochs in ((1e-6, 4), (1., 12)):
            with self.subTest(scale=scale), tempfile.TemporaryDirectory() as directory:
                observed = []
                def objective(model):
                    observed.append(core.weights(model))
                    return scale * quadratic(model)
                model = Tiny()
                record = common.metadata(200)
                training.train_model(model, objective, .6, directory, record, epochs=epochs)
                selected = 0
                values = [float(scale * s['fc_layers.weight'].square().sum()) for s in observed]
                for index, value in enumerate(values):
                    if value < values[selected] - 1e-4:
                        selected = index
                self.assertEqual(record['best_epoch'], selected + 1)
                self.assertEqual(record['best_train'], values[selected])
                self.assertEqual(len(observed), epochs)
                for key, value in core.weights(model).items():
                    torch.testing.assert_close(value, observed[selected][key], atol=0., rtol=0.)

    def test_patience_scheduler_training_only_and_last_epoch_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            record = common.metadata(200)
            training.train_model(Tiny(), lambda m: quadratic(m) * 0. + 1.,
                                 .1, directory, record, epochs=305, patience=300)
            self.assertEqual(record['stop_reason'], 'training_patience')
            self.assertEqual(record['epochs_evaluated'], 301)
            self.assertEqual(record['optimizer_steps'], 301)
            self.assertEqual(record['validation_time_s'], 0.)
            self.assertEqual(record['scheduler_metric'], 'training_loss')
            self.assertAlmostEqual(record['epoch_history'][-1]['learning_rates'][0], .07)
            self.assertAlmostEqual(record['epoch_history'][-1]['learning_rates'][1], .007)

    def test_failures_preserve_partial_journal_and_no_restoration(self):
        for kind in ('nonfinite', 'forward', 'backward'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                seen = []
                def objective(model):
                    seen.append(core.weights(model))
                    value = quadratic(model)
                    if len(seen) == 2:
                        if kind == 'nonfinite':
                            return value * float('nan')
                        if kind == 'forward':
                            raise RuntimeError('fixture forward failure')
                        def fail(gradient):
                            raise RuntimeError('fixture backward failure')
                        value.register_hook(fail)
                    return value
                model, record = Tiny(), common.metadata(200)
                with self.assertRaises((RuntimeError, FloatingPointError)):
                    training.train_model(model, objective, .1, directory, record, epochs=3)
                self.assertEqual(record['status'], 'failed')
                self.assertEqual(record['optimizer_steps'], 1)
                self.assertEqual([e['status'] for e in record['epoch_history']], ['completed', 'failed'])
                for key, value in core.weights(model).items():
                    torch.testing.assert_close(value, seen[1][key], atol=0., rtol=0.)
                self.assertTrue((Path(directory) / 'failure_state.pth').exists())
                self.assertFalse((Path(directory) / 'selected.pth').exists())
                self.assertGreater(record['wall_time_s'], 0.)

    def test_api_has_no_heldout_or_production_budget_cli(self):
        parameters = inspect.signature(training.train_model).parameters
        self.assertEqual(list(parameters), ['model', 'loss_fn', 'lr', 'folder', 'record', 'epochs', 'patience'])
        self.assertEqual(parameters['epochs'].default, 3000)
        self.assertEqual(parameters['patience'].default, 300)
        self.assertEqual(common.MAX_MP, 40)


class Policy(torch.nn.Module):
    def __init__(self, n_val=40):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(20.))
        self.n_val = n_val

    def per_sample_cost(self, x1, x2):
        if len(x1) == self.n_val:
            raise AssertionError('Held-out-sized call forbidden in fixture')
        return x1[:, 0].double() + self.bias.double()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.n_total, self.n_train = 200, 160
        self.ap_calls = self.train_calls = 0
        self.restore = False
        self.steps = None
        self.ap_fault = self.train_fault = None
        self.prepare_inputs()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(torch, 'get_num_threads', return_value=1))
        stack.enter_context(patch.object(torch, 'get_num_interop_threads', return_value=1))
        stack.enter_context(patch.object(core, 'model_from_state', side_effect=self.from_state))
        stack.enter_context(patch.object(ccg, 'train_model', side_effect=self.train))
        stack.enter_context(patch.object(ccg.adversary, 'solve_ap', side_effect=self.ap))
        for target, name in ((core, 'split_data'), (core, 'train_model'), (core, 'solve_saa'),
                (common, 'split_data'), (common, 'evaluate'), (ccg.legacy_ccg, 'assess_candidate'),
                (ccg.legacy_ccg, 'run_configuration'), (ccg.legacy_ccg, 'load_nominal')):
            stack.enter_context(patch.object(target, name, side_effect=AssertionError(f'Forbidden {name}')))
        stack.enter_context(patch.object(core.gp, 'Env', side_effect=AssertionError('Real solver forbidden')))

    def prepare_inputs(self):
        self.n_train = 4 * self.n_total // 5
        self.centers = np.ones((self.n_train, 6), dtype=np.float64)
        # Include IDs beyond 50 and near Ntotal; never regenerate a split here.
        self.ids = list(range(self.n_total - self.n_train, self.n_total))
        checkpoint = self.root / 'nominal.pth'
        model = Policy(self.n_total // 5)
        torch.save(core.weights(model), checkpoint)
        data_path = self.root / 'train.json'
        core.write_json(data_path, dict(centers=self.centers,
            original_sample_ids=self.ids, data_sha256='fixture-parent-data-hash'))
        self.provenance = self.root / 'provenance.json'
        core.write_json(self.provenance, manifest())
        self.nominal = dict(Ntotal=self.n_total, seed=0, lr_net=.005,
            checkpoint=str(checkpoint), checkpoint_sha256=core.digest(checkpoint),
            state_sha256=ccg.state_hash(model), train_mean=21., data_sha256='fixture-parent-data-hash',
            training_data_path=str(data_path), training_data_sha256=core.digest(data_path),
            training_timing=dict(status='completed', wall_time_s=1000., epochs_evaluated=3))
        self.nominal_bytes = checkpoint.read_bytes()

    def from_state(self, state):
        model = Policy(self.n_total // 5)
        model.load_state_dict(state)
        return model

    def train(self, model, loss_fn, lr, folder, record, epochs=3000, patience=300):
        self.assertEqual((lr, epochs, patience), (.005, 3000, 300))
        self.assertEqual(record['Ntrain'], self.n_train)
        self.train_calls += 1
        with core.timed(record), torch.no_grad():
            pools = json.loads((folder / 'pool.json').read_text())['pools']
            expected = torch.stack([model.per_sample_cost(
                torch.tensor(p, dtype=torch.float32)[:, :3],
                torch.tensor(p, dtype=torch.float32)[:, 3:]).max() for p in pools]).mean()
            torch.testing.assert_close(loss_fn(model), expected, atol=0., rtol=0.)
            first = float(loss_fn(model))
            if not self.restore:
                model.bias.sub_(.01)
            best = float(loss_fn(model))
            record.update(first_train=first, best_train=best, selected_training_loss=best,
                          epochs_evaluated=1, optimizer_steps=1)
            torch.save(core.weights(model), folder / 'selected.pth')
            if self.train_fault:
                self.train_fault(model, record, folder)
        return model

    def ap(self, model, center1, center2, radius, row):
        sweep, sample_id = divmod(self.ap_calls, self.n_train)
        self.ap_calls += 1
        self.assertEqual(row['sample_id'], sample_id)
        self.assertEqual(row['original_sample_id'], self.ids[sample_id])
        self.assertEqual(center1.dtype, np.float64)
        np.testing.assert_array_equal(center1, self.centers[sample_id, :3])
        np.testing.assert_array_equal(center2, self.centers[sample_id, 3:])
        x1, x2 = center1.copy(), center2.copy()
        x1[0] += min(radius, self.steps[sweep] if self.steps else .5 * (sweep + 1))
        with core.timed(row), torch.no_grad():
            cost = float(model.per_sample_cost(torch.tensor(x1[None], dtype=torch.float32),
                                               torch.tensor(x2[None], dtype=torch.float32))[0])
            row.update(solver_status=2, solcount=1, bound=cost, iterations=1,
                incumbent=dict(cost=cost, worst1=x1.tolist(), worst2=x2.tolist()),
                replay=dict(status='completed', cost=cost, floor_scaled_discrepancy=0.))
        result = (x1, x2, cost)
        return self.ap_fault(result, row, sweep, sample_id) if self.ap_fault else result

    def run_case(self, radius=1.5):
        result = ccg.run_configuration(self.n_total, 0, radius, self.root / 'out',
                                       self.nominal, self.provenance)
        self.summary = json.loads(Path(result['summary_path']).read_text())
        self.assertEqual(result['summary_sha256'], core.digest(result['summary_path']))
        self.assertNotIn('iterations', result)
        json.dumps(result, allow_nan=False)
        self.assertGreaterEqual(result['wall_time_s'], 0.)
        self.assertIn('started_at_utc', result)
        self.assertIn('ended_at_utc', result)
        return result

    def assert_unpublished(self, result, status):
        self.assertEqual(result['status'], status, result['error'])
        self.assertFalse(result['complete'])
        self.assertIsNone(result['selected_checkpoint'])
        self.assertIsNone(result['selected_candidate_id'])
        self.assertFalse((Path(result['output_directory']) / 'selected.pth').exists())

    def test_initial_eligible_zero_mp_and_private_hashes_exclusive(self):
        result = self.run_case(.03)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual((self.ap_calls, self.train_calls), (160, 0))
        self.assertEqual((result['stopping_candidate_id'], result['selected_candidate_id']), (0, 0))
        self.assertEqual(result['ccg_workers'], 3)
        self.assertEqual(Path(result['selected_checkpoint']).read_bytes(), self.nominal_bytes)
        self.assertEqual(Path(self.nominal['checkpoint']).read_bytes(), self.nominal_bytes)
        self.assertNotIn('added_sample_ids', self.summary['iterations'][0])
        self.assertEqual(self.summary['cached_nominal']['training_timing']['wall_time_s'], 1000.)
        frozen = Path(result['output_directory']) / 'inputs/nominal.pth'
        self.assertEqual(frozen.stat().st_mode & 0o222, 0)
        self.assertNotEqual(frozen.stat().st_ino, Path(self.nominal['checkpoint']).stat().st_ino)
        before = Path(result['summary_path']).read_bytes()
        with self.assertRaises(FileExistsError):
            ccg.run_configuration(200, 0, .03, self.root / 'out', self.nominal, self.provenance)
        self.assertEqual(before, Path(result['summary_path']).read_bytes())

    def test_N800_zero_radius_all640_centers_and_no_heldout_calls(self):
        self.n_total = 800
        self.prepare_inputs()
        result = self.run_case(0.)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual((result['Ntrain'], result['Nval']), (640, 160))
        self.assertEqual((self.ap_calls, self.train_calls), (640, 0))
        self.assertEqual(len(self.summary['iterations'][0]['samples']), 640)

    def test_first_pass_stops_before_future_and_selects_earlier_minimum(self):
        # Later fixture rows do not exist: any future AP sweep fails immediately.
        self.steps = [.5, 1., 1.1]
        class Poison:
            def __float__(self):
                raise AssertionError('Held-out preference accessed')
        self.nominal['validation_mean'] = Poison()
        result = self.run_case()
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual((self.ap_calls, self.train_calls), (480, 2))
        self.assertEqual((result['stopping_candidate_id'], result['selected_candidate_id']), (2, 0))
        self.assertLess(result['stopping_green_gap'], .01)
        self.assertGreater(result['selected_green_gap'], .01)
        self.assertEqual(result['selected_separation_status'], 'violating')
        self.assertEqual(Path(result['selected_checkpoint']).read_bytes(), self.nominal_bytes)
        last = self.summary['iterations'][-1]
        self.assertNotIn('mp', last)
        self.assertNotIn('added_sample_ids', last)
        self.assertFalse((Path(result['output_directory']) / 'candidate_03').exists())

    def test_stopping_candidate_included_and_exact_tie_earliest(self):
        result = self.run_case(.4)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual((result['stopping_candidate_id'], result['selected_candidate_id']), (1, 1))
        self.assertEqual((self.train_calls, self.ap_calls), (1, 320))

    def test_exact_tie_prefers_initial_even_when_repeated_stop(self):
        self.restore = True
        result = self.run_case(.4)
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual((result['stopping_candidate_id'], result['selected_candidate_id']), (1, 0))
        self.assertTrue(result['closure_with_unchangedweights'])
        self.assertEqual(self.summary['iterations'][0]['mp']['same_pool_change'], 0.)

    def test_black_closure_does_not_stop_green(self):
        # .212/21 > .01, but .212/21.212 < .01: different denominators.
        self.steps = [.212, .212]
        result = self.run_case(.4)
        self.assertEqual(result['status'], 'completed', result['error'])
        first = self.summary['iterations'][0]
        self.assertEqual(first['separation_status'], 'separation_satisfied')
        self.assertFalse(first['green_pass'])
        self.assertEqual(result['stopping_candidate_id'], 1)
        self.assertEqual(self.train_calls, 1)

    def test_cap_is_nonconverged_no_selection_final_full_assessment(self):
        with patch.object(ccg, 'MAX_MP', 2):
            result = self.run_case()
        self.assert_unpublished(result, 'nonconverged')
        self.assertEqual(result['computational_job_status'], 'completed')
        self.assertEqual(result['terminal_status'], 'iteration_cap')
        self.assertIsNone(result['error'])
        self.assertEqual((self.train_calls, self.ap_calls), (2, 480))
        self.assertTrue(all(c['assessed'] for c in self.summary['iterations']))
        for candidate in self.summary['iterations']:
            self.assertGreaterEqual(candidate['ap_total_time_s'], 0.)
            self.assertGreaterEqual(candidate['mp_total_time_s'], 0.)
        self.assertEqual(result['ap_total_time_s'], sum(
            c['ap']['wall_time_s'] for c in self.summary['iterations']))

    def test_repetition_is_nonconverged_not_failure(self):
        self.restore = True
        result = self.run_case()
        self.assert_unpublished(result, 'nonconverged')
        self.assertEqual(result['terminal_status'], 'stagnation')
        self.assertEqual((self.train_calls, self.ap_calls), (1, 320))

    def test_no_additions_is_nonconverged(self):
        # Isolate the no-additions branch; production AP/mean-gap gates are not changed.
        with patch.object(ccg, 'terminal_decision', return_value=dict(green_gap=.02, green_pass=False)):
            result = self.run_case(0.)
        self.assert_unpublished(result, 'nonconverged')
        self.assertEqual(self.summary['terminal_reason'], 'no_new_violating_incumbents')
        self.assertEqual(self.train_calls, 0)

    def test_failed_ap_attempts_all_centers_retains_prefix_and_no_selection(self):
        def fault(result, row, sweep, sample_id):
            if sweep == 1 and sample_id == 3:
                row['solver_status'] = 9
                raise RuntimeError('fixture AP failure')
            return result
        self.ap_fault = fault
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        self.assertEqual((self.train_calls, self.ap_calls), (1, 320))
        first, failed = self.summary['iterations']
        self.assertTrue(first['assessed'])
        self.assertFalse(failed['assessed'])
        self.assertEqual(failed['failed_sample_ids'], [3])
        self.assertEqual(len(failed['samples']), 160)
        self.assertEqual(failed['samples'][3]['solver_status'], 9)
        for row in failed['samples']:
            self.assertGreaterEqual(row['controller_wall_time_s'], 0.)
            self.assertTrue((Path(failed['ap_reference']).parent / f"sample_{row['sample_id']:02d}.json").exists())

    def test_failed_mp_preserves_partial_timing(self):
        def fault(model, record, folder):
            raise RuntimeError('fixture training failure')
        self.train_fault = fault
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        candidate = self.summary['iterations'][0]
        self.assertEqual(candidate['mp']['status'], 'failed')
        self.assertGreater(candidate['mp']['wall_time_s'], 0.)
        self.assertGreater(result['mp_total_time_s'], 0.)

    def test_bad_checkpoint_state_rejected_before_AP(self):
        self.nominal['state_sha256'] = 'wrong'
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        self.assertEqual(self.ap_calls, 0)
        self.assertIn('state mismatch', result['error']['message'])

    def test_training_data_tamper_rejected_before_AP(self):
        path = Path(self.nominal['training_data_path'])
        path.write_text(path.read_text() + ' ')
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        self.assertEqual(self.ap_calls, 0)
        self.assertIn('hash mismatch', result['error']['message'])

    def test_extra_heldout_data_rejected(self):
        path = Path(self.nominal['training_data_path'])
        data = json.loads(path.read_text())
        data['val_x1'] = []
        core.write_json(path, data)
        self.nominal['training_data_sha256'] = core.digest(path)
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        self.assertEqual(self.ap_calls, 0)
        self.assertIn('must contain only', result['error']['message'])

    def test_nominal_size_and_fresh_training_timing_required(self):
        self.nominal['Ntotal'] = 50
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        self.assertIn('sample size mismatch', result['error']['message'])
        self.assertEqual(self.ap_calls, 0)

    def test_missing_training_timing_is_failure(self):
        del self.nominal['training_timing']
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        self.assertEqual(self.ap_calls, 0)

    def test_mp_returned_state_must_match_checkpoint(self):
        def fault(model, record, folder):
            torch.save(core.weights(Policy()), folder / 'selected.pth')
        self.train_fault = fault
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        self.assertIn('state mismatch', result['error']['message'])

    def test_nonfinite_bound_retained_not_repaired(self):
        def fault(result, row, sweep, sample_id):
            if sample_id == 5:
                row['bound'] = float('nan')
            return result
        self.ap_fault = fault
        result = self.run_case()
        self.assert_unpublished(result, 'failed')
        self.assertEqual(self.ap_calls, 160)
        self.assertEqual(self.summary['iterations'][0]['samples'][5]['bound'], 'NaN')

    def test_final_source_verification_failure_never_publishes(self):
        with patch.object(ccg, 'verify_sources', side_effect=[None, ValueError('source changed')]):
            result = self.run_case(0.)
        self.assert_unpublished(result, 'failed')
        self.assertIn('source changed', result['error']['traceback'])


if __name__ == '__main__':
    unittest.main()