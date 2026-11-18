"""Runner-only regression tests: no numerical backend or historical test data.

All artifacts are temporary. KFold, NumPy arrays, Torch checkpoint serialization,
and the invalid-action worker subprocess are real; optimization is always fake.
Run with: /data/home/aviyac/ap_env/bin/python -B -m unittest -v test_runner
"""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().with_name('runner.py')
SPEC = importlib.util.spec_from_file_location('_green_fulln_runner_tests', SOURCE)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

import numpy as np
from sklearn.model_selection import KFold
import torch


class FakeModel:
    """Tensor state only; no parameters, optimizer, solver, or real training."""

    def __init__(self, state):
        self.state = {k: v.clone() for k, v in state.items()}

    def eval(self):
        return self

    def per_sample_cost(self, x1, x2):
        return x1[:, 0] * 0 + x2[:, 0] * 0 + 2.


class FakeBackend:
    def __init__(self, kind):
        self.kind = kind
        self.np, self.torch = np, torch
        self.core = SimpleNamespace(write_json=runner.write, timed=runner.timed)
        self.protocol = {'test': 'not accessed'}
        self.verify_sources = Mock()
        self.sources = Mock(return_value={})
        self.train_model = Mock(side_effect=AssertionError('Unexpected trainer access'))
        self.events = []
        self.saa_calls = []
        self.fit_calls = []
        self.evaluations = []
        self.initial_models = []
        self.generate_calls = []

    def metadata(self, n, seed=None, method=None):
        return dict(Ntotal=n, Ntrain=n, Nval=0, seed=seed, method=method,
                    network_seed=0, torch_threads=1, gurobi_threads=1)

    def generate(self, n, seed):
        self.generate_calls.append((n, seed))
        width = 10 if self.kind == 'R10' else 3
        rows = np.arange(n * width, dtype=np.float64).reshape(n, width)
        # Unique, exactly representable row identities in both arrays.
        self.data = dict(xi1=rows, xi2=rows + 10000 + seed * 1000)
        return self.data

    def solve_saa(self, x1, x2, record):
        self.events.append(('saa', len(x1)))
        self.saa_calls.append((x1.copy(), x2.copy(), deepcopy(record)))
        with runner.timed(record):
            record.update(solver_status=2, iterations=0)
        return len(self.saa_calls), np.zeros(x1.shape[1])

    def make_model(self, q10, q1):
        model = FakeModel({'initialization': torch.tensor(q10), 'lr': torch.tensor(0., dtype=torch.float64)})
        self.initial_models.append(model)
        return model

    @staticmethod
    def weights(model):
        return {k: v.clone() for k, v in model.state.items()}

    @staticmethod
    def model_from_state(state):
        return FakeModel(state)

    @staticmethod
    def state_hash(model):
        return hashlib.sha256(b''.join(v.numpy().tobytes() for k, v in sorted(model.state.items()))).hexdigest()

    def fake_fit(self, backend, initial, x1, x2, lr, folder, meta):
        assert backend is self
        self.events.append(('fit', meta.get('fold'), lr))
        self.fit_calls.append(dict(initial=self.weights(FakeModel(initial)),
                                   x1=x1.numpy().copy(), x2=x2.numpy().copy(),
                                   lr=lr, folder=folder, meta=deepcopy(meta)))
        folder.mkdir()
        model = FakeModel(initial)
        model.state['lr'] = torch.tensor(lr, dtype=torch.float64)
        torch.save(self.weights(model), folder / 'selected.pth')
        training = dict(meta)
        with runner.timed(training):
            training.update(epochs_evaluated=0, optimizer_steps=0, selected_training_loss=2.)
        runner.write(folder / 'training.json', training)
        return model, 2., runner.timing_ref(training, folder / 'training.json')

    def evaluate(self, model, x1, x2, record):
        self.events.append(('evaluate', record['fold'], record['lr']))
        self.evaluations.append(dict(x1=x1.numpy().copy(), x2=x2.numpy().copy(), meta=deepcopy(record)))
        # This assertion runs at scoring time, not after the orchestration ends.
        arms = [c for c in self.fit_calls if c['meta'].get('fold') == record['fold']]
        assert len(arms) == 3, 'Held-out scoring before all LR checkpoints were fitted'
        frozen = runner.read(arms[0]['folder'].parent / 'frozen_arms.json')
        assert len(frozen) == 3
        assert all(runner.sha(a['checkpoint']) == a['checkpoint_sha256'] for a in frozen)
        assert float(model.state['lr']) == record['lr']
        with runner.timed(record):
            record.update(mean={.005: 4., .01: 1., .05: 3.}[record['lr']], sample_count=len(x1))
        return record['mean']


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='green_runner_tests_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Never freeze or alter the live sources: the parent may still edit them.
        self.sources = self.root / 'sources'
        self.sources.mkdir()
        (self.sources / 'runner.py').write_text('# isolated source-freeze fixture\n')
        self.manifest = self.root / 'manifest.json'
        with patch.object(runner, 'ROOT', self.sources):
            runner.write(self.manifest, dict(local_sources=runner.local_sources(),
                         source_sha256={}, workers=dict(nominal=1, ccg=1)))

    def assert_timed(self, record, status='completed'):
        self.assertEqual(record['status'], status)
        self.assertGreaterEqual(record['wall_time_s'], 0.)
        self.assertEqual(record['timing_clock'], 'time.perf_counter')
        self.assertEqual(record['timing_unit'], 'seconds')
        self.assertLessEqual(record['started_at_utc'], record['ended_at_utc'])

    @staticmethod
    def cv_rows():
        return [dict(fold=f, lr=lr, status='completed', mean=2.)
                for f in range(5) for lr in runner.LRS]

    def test_five_fold_coverage_all_sizes_and_seeds(self):
        self.assertEqual(runner.SIZES, (50, 200, 800))
        self.assertEqual(runner.SEEDS, tuple(range(10)))
        for n in runner.SIZES:
            for seed in runner.SEEDS:
                with self.subTest(n=n, seed=seed):
                    actual = runner.folds(n, seed)
                    expected = list(KFold(n_splits=5, shuffle=True, random_state=100000 + seed).split(np.arange(n)))
                    self.assertEqual(len(actual), 5)
                    self.assertEqual(sorted(np.concatenate([v for _, v in actual]).tolist()), list(range(n)))
                    for (train, held), (et, ev), (rt, rv) in zip(actual, expected, runner.folds(n, seed)):
                        np.testing.assert_array_equal(train, et)
                        np.testing.assert_array_equal(held, ev)
                        np.testing.assert_array_equal(train, rt)
                        np.testing.assert_array_equal(held, rv)
                        self.assertEqual((len(train), len(held)), (4 * n // 5, n // 5))
                        self.assertFalse(set(train) & set(held))
                        self.assertEqual(sorted([*train, *held]), list(range(n)))
                    self.assertFalse(np.array_equal(actual[0][1], runner.folds(n, seed + 1)[0][1]))

    def test_choose_lr_complete_means_and_order_independence(self):
        rows = self.cv_rows()
        for row in rows:
            row['mean'] = { .005: 5., .01: 3., .05: 4.}[row['lr']] + row['fold']
        before = deepcopy(rows)
        for ordered in (rows, rows[::-1], rows[7:] + rows[:7]):
            chosen, means = runner.choose_lr(ordered)
            self.assertEqual(chosen, .01)
            self.assertEqual(means, {.005: 7., .01: 5., .05: 6.})
        self.assertEqual(rows, before)

    def test_choose_lr_exact_tie_not_tolerance(self):
        rows = self.cv_rows()
        self.assertEqual(runner.choose_lr(rows)[0], .005)
        for row in rows:
            if row['lr'] == .01:
                row['mean'] = np.nextafter(2., 0.)
        self.assertEqual(runner.choose_lr(rows)[0], .01)

    def test_choose_lr_requires_all_fifteen_unique_arms(self):
        rows = self.cv_rows()
        bad_grids = [rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]]
        for field, value in (('fold', 5), ('lr', .1)):
            bad = deepcopy(rows)
            bad[0][field] = value
            bad_grids.append(bad)
        for bad in bad_grids:
            with self.subTest(keys=[(r['fold'], r['lr']) for r in bad]):
                with self.assertRaisesRegex(ValueError, '15 completed finite'):
                    runner.choose_lr(bad)

    def test_choose_lr_rejects_every_nonfinite_or_incomplete_arm(self):
        for index in range(15):
            for field, value in [('mean', float('nan')), ('mean', float('inf')),
                                 ('mean', -float('inf')), ('status', 'failed'),
                                 ('status', 'running'), ('status', 'nonconverged')]:
                with self.subTest(arm=index, field=field, value=value):
                    rows = self.cv_rows()
                    rows[index][field] = value
                    with self.assertRaises(ValueError):
                        runner.choose_lr(rows)

    def test_choose_lr_ignores_test_values_and_does_no_io(self):
        rows = self.cv_rows()
        for row in rows:
            row.update(test_mean=-1e9 if row['lr'] == .05 else 1e9, test=float('nan'))
        with patch.object(runner, 'read', side_effect=AssertionError('Selection read external data')), \
                patch.object(runner, 'backend_for', side_effect=AssertionError('Selection loaded backend')):
            self.assertEqual(runner.choose_lr(rows)[0], .005)
            for row in rows:
                row['test_mean'] *= -1
            self.assertEqual(runner.choose_lr(rows)[0], .005)

    def test_nominal_fold_isolation_same_lr_folds_and_fresh_full_n(self):
        for kind in runner.KINDS:
            for n in runner.SIZES:
                with self.subTest(kind=kind, n=n):
                    b = FakeBackend(kind)
                    folder = self.root / f'{kind}_{n}'
                    with patch.object(runner, 'ROOT', self.sources), \
                            patch.object(runner, 'backend_for', return_value=b) as backend, \
                            patch.object(runner, 'fit', side_effect=b.fake_fit) as fit:
                        result = runner.nominal(kind, n, 7, folder, self.manifest)
                    backend.assert_called_once_with(kind)
                    b.train_model.assert_not_called()
                    self.assertEqual(b.generate_calls, [(n, 7)])
                    self.assertEqual(fit.call_count, 16)
                    self.assertEqual(len(b.saa_calls), 6)
                    self.assertEqual(len(b.evaluations), 15)
                    self.assertEqual(b.verify_sources.call_count, 2)
                    with np.load(folder / 'data.npz') as saved:
                        for key in ('xi1', 'xi2'):
                            np.testing.assert_array_equal(saved[key], b.data[key])
                    saved_folds = runner.read(folder / 'folds.json')
                    expected = runner.folds(n, 7)
                    for f, (train, held) in enumerate(expected):
                        self.assertEqual(saved_folds[f], dict(fold=f, train_ids=train.tolist(), heldout_ids=held.tolist()))
                        saa_x1, saa_x2, saa_meta = b.saa_calls[f]
                        np.testing.assert_array_equal(saa_x1, b.data['xi1'][train])
                        np.testing.assert_array_equal(saa_x2, b.data['xi2'][train])
                        self.assertEqual((saa_meta['Ntrain'], saa_meta['Nval']), (len(train), len(held)))
                        hashes = []
                        for arm, score in zip(b.fit_calls[f * 3:(f + 1) * 3], b.evaluations[f * 3:(f + 1) * 3]):
                            np.testing.assert_array_equal(arm['x1'], b.data['xi1'][train])
                            np.testing.assert_array_equal(arm['x2'], b.data['xi2'][train])
                            np.testing.assert_array_equal(score['x1'], b.data['xi1'][held])
                            np.testing.assert_array_equal(score['x2'], b.data['xi2'][held])
                            self.assertFalse(set(arm['x1'][:, 0]) & set(score['x1'][:, 0]))
                            self.assertEqual(arm['meta']['fold'], f)
                            self.assertEqual(float(arm['initial']['lr']), 0.)
                            hashes.append(b.state_hash(FakeModel(arm['initial'])))
                        self.assertEqual(len(set(hashes)), 1)
                        self.assertEqual([a['lr'] for a in b.fit_calls[f * 3:(f + 1) * 3]], list(runner.LRS))
                        persisted = runner.read(folder / f'fold_{f}/summary.json')
                        self.assert_timed(persisted)
                        self.assertEqual({a['initial_state_sha256'] for a in persisted['arms']}, set(hashes))
                    # The sixth SAA is after every held-out score, and its model is not a fold winner.
                    self.assertEqual(b.events[-2:], [('saa', n), ('fit', None, .01)])
                    final = b.fit_calls[-1]
                    for key, index in (('xi1', 0), ('xi2', 1)):
                        np.testing.assert_array_equal(b.saa_calls[-1][index], b.data[key])
                        np.testing.assert_array_equal(final['x1' if index == 0 else 'x2'], b.data[key])
                    self.assertEqual((final['meta']['Ntrain'], final['meta']['Nval']), (n, 0))
                    self.assertEqual(int(final['initial']['initialization']), 6)
                    self.assertEqual(float(final['initial']['lr']), 0.)
                    self.assertEqual(final['lr'], .01)
                    self.assertEqual(len({b.state_hash(m) for m in b.initial_models}), 6)
                    nominal = runner.read(folder / 'nominal.json')
                    self.assertEqual(nominal, result['nominal'])
                    self.assertEqual(nominal['Ntotal'], n)
                    self.assertEqual(nominal['checkpoint_sha256'], runner.sha(nominal['checkpoint']))
                    training = runner.read(nominal['training_data_path'])
                    self.assertEqual(training['original_sample_ids'], list(range(n)))
                    np.testing.assert_array_equal(training['centers'], np.concatenate([b.data['xi1'], b.data['xi2']], axis=1))
                    self.assertEqual(nominal['training_data_sha256'], runner.sha(nominal['training_data_path']))
                    summary = runner.read(folder / 'summary.json')
                    self.assert_timed(summary)
                    self.assertEqual(summary['test'], 'not accessed')
                    self.assertEqual(len(summary['cv_rows']), 15)
                    self.assertEqual(summary['raw_data_file_sha256'], runner.sha(folder / 'data.npz'))
                    self.assertEqual(summary['folds_sha256'], runner.sha(folder / 'folds.json'))
                    self.assertEqual(result['summary_sha256'], runner.sha(folder / 'summary.json'))

    def test_nominal_fit_failure_persists_and_prevents_full_n(self):
        b = FakeBackend('R3')
        folder = self.root / 'failed_nominal'
        with patch.object(runner, 'ROOT', self.sources), patch.object(runner, 'backend_for', return_value=b), \
                patch.object(runner, 'fit', side_effect=RuntimeError('fake training failure')):
            with self.assertRaisesRegex(RuntimeError, 'fake training failure'):
                runner.nominal('R3', 50, 0, folder, self.manifest)
        self.assert_timed(runner.read(folder / 'summary.json'), 'failed')
        self.assert_timed(runner.read(folder / 'fold_0/summary.json'), 'failed')
        self.assertEqual(len(b.saa_calls), 1)
        self.assertFalse((folder / 'nominal.json').exists())
        self.assertFalse((folder / 'fullN').exists())

    def test_nominal_nonfinite_cv_does_not_refit(self):
        b = FakeBackend('R3')
        evaluate = b.evaluate

        def bad_score(*args):
            evaluate(*args)
            args[-1]['mean'] = float('nan')

        b.evaluate = bad_score
        folder = self.root / 'nonfinite_nominal'
        with patch.object(runner, 'ROOT', self.sources), patch.object(runner, 'backend_for', return_value=b), \
                patch.object(runner, 'fit', side_effect=b.fake_fit):
            with self.assertRaises(ValueError):
                runner.nominal('R3', 50, 0, folder, self.manifest)
        self.assert_timed(runner.read(folder / 'summary.json'), 'failed')
        self.assertEqual(len(b.saa_calls), 1)
        self.assertFalse((folder / 'nominal.json').exists())

    def test_fit_replay_and_checkpoint_with_fake_trainer_only(self):
        b = FakeBackend('R3')
        initial = b.weights(b.make_model(1, np.zeros(3)))
        x = torch.zeros((40, 3))
        folder = self.root / 'fit'

        def train(model, loss, lr, out, record):
            self.assertEqual(float(loss(model)), 2.)
            self.assertEqual(lr, .01)
            out.mkdir()
            with runner.timed(record):
                record.update(selected_training_loss=2., epochs_evaluated=0, optimizer_steps=0)
            torch.save(b.weights(model), out / 'selected.pth')
            runner.write(out / 'training.json', record)
            return model

        b.train_model.side_effect = train
        model, mean, timing = runner.fit(b, initial, x, x, .01, folder, b.metadata(50, 0))
        b.train_model.assert_called_once()
        self.assertEqual(mean, 2.)
        self.assertEqual(timing['sha256'], runner.sha(folder / 'training.json'))
        replay = runner.read(folder / 'training_replay.json')
        self.assert_timed(replay)
        self.assertEqual(replay['sample_count'], 40)
        self.assertEqual(replay['state_sha256'], b.state_hash(model))

    def test_fit_rejects_training_replay_or_checkpoint_mismatch(self):
        for mismatch in ('loss', 'checkpoint'):
            with self.subTest(mismatch=mismatch):
                b = FakeBackend('R3')
                initial = b.weights(b.make_model(1, np.zeros(3)))
                folder = self.root / mismatch

                def train(model, loss, lr, out, record):
                    out.mkdir()
                    record['selected_training_loss'] = 99. if mismatch == 'loss' else 2.
                    saved = b.weights(model)
                    if mismatch == 'checkpoint':
                        saved['initialization'] += 1
                    torch.save(saved, out / 'selected.pth')
                    return model

                b.train_model.side_effect = train
                with self.assertRaisesRegex(ValueError, 'replay mismatch|checkpoint mismatch'):
                    runner.fit(b, initial, torch.zeros(40, 3), torch.zeros(40, 3), .01, folder, {})
                self.assert_timed(runner.read(folder / 'training_replay.json'), 'failed')

    def test_local_source_freeze_detects_edits_additions_and_deletions(self):
        with patch.object(runner, 'ROOT', self.sources):
            for name in ('backend.py', 'reporting.py', 'test_runner.py', 'README.md'):
                (self.sources / name).write_text('fixture\n')
            manifest = dict(local_sources=runner.local_sources())
            runner.verify_local(manifest)
            (self.sources / 'ignored.json').write_text('{}')
            runner.verify_local(manifest)
            for name in ('runner.py', 'backend.py', 'reporting.py', 'test_runner.py', 'README.md'):
                file = self.sources / name
                original = file.read_text()
                file.write_text(original + 'changed\n')
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'sources changed'):
                    runner.verify_local(manifest)
                file.write_text(original)
            extra = self.sources / 'extra.py'
            extra.write_text('# new\n')
            with self.assertRaises(ValueError):
                runner.verify_local(manifest)
            extra.unlink()
            (self.sources / 'backend.py').unlink()
            with self.assertRaises(ValueError):
                runner.verify_local(manifest)

    def test_nominal_checks_freeze_before_generating_or_writing_job(self):
        b = FakeBackend('R3')
        (self.sources / 'runner.py').write_text('# changed\n')
        folder = self.root / 'must_not_exist'
        with patch.object(runner, 'ROOT', self.sources), patch.object(runner, 'backend_for', return_value=b):
            with self.assertRaisesRegex(ValueError, 'sources changed'):
                runner.nominal('R3', 50, 0, folder, self.manifest)
        self.assertFalse(folder.exists())
        self.assertFalse(b.generate_calls)

    def test_nominal_rechecks_freeze_at_completion(self):
        b = FakeBackend('R3')

        def fit(*args):
            result = b.fake_fit(*args)
            if 'fold' not in args[-1]:
                (self.sources / 'runner.py').write_text('# changed during nominal\n')
            return result

        folder = self.root / 'changed_nominal'
        with patch.object(runner, 'ROOT', self.sources), patch.object(runner, 'backend_for', return_value=b), \
                patch.object(runner, 'fit', side_effect=fit):
            with self.assertRaisesRegex(ValueError, 'sources changed'):
                runner.nominal('R3', 50, 0, folder, self.manifest)
        self.assert_timed(runner.read(folder / 'summary.json'), 'failed')

    def test_worker_ccg_dispatch_all_nine_radii_keeps_nonconverged_status(self):
        expected_radii = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
        self.assertEqual(runner.RADII, expected_radii)
        nominal = {'checkpoint': 'fake-fullN-checkpoint'}
        b = SimpleNamespace(run_configuration=Mock(side_effect=lambda *a: dict(status='nonconverged', radius=a[2])))
        with patch.object(runner, 'ROOT', self.sources), patch.object(runner, 'backend_for', return_value=b):
            for radius in runner.RADII:
                result = runner.worker(dict(action='ccg', kind='R3', n=800, seed=9, radius=radius,
                    folder=str(self.root / 'ccg'), manifest=str(self.manifest), nominal=nominal))
                self.assertEqual(result, dict(status='nonconverged', radius=radius, kind='R3'))
        self.assertEqual(b.run_configuration.call_count, 9)
        for radius, call in zip(expected_radii, b.run_configuration.call_args_list):
            self.assertEqual(call.args[:4], (800, 9, radius, self.root / 'ccg'))
            self.assertIs(call.args[4], nominal)
            self.assertEqual(call.args[5], str(self.manifest))

    def test_invoke_real_subprocess_failure_and_persisted_receipts(self):
        (self.root / 'jobs').mkdir()
        spec = dict(id='invalid_action', action='deliberately_invalid', folder=str(self.root / 'unused'))
        with redirect_stdout(io.StringIO()):
            result = runner.invoke(spec, self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(runner.read(self.root / 'jobs/invalid_action.json'), spec)
        receipt = runner.read(self.root / 'jobs/invalid_action.invocation.json')
        self.assert_timed(receipt, 'failed')
        self.assertNotEqual(receipt['returncode'], 0)
        self.assertEqual(receipt['command'][:4], [sys.executable, '-B', '-u', str(SOURCE)])
        self.assertEqual(receipt['cwd'], str(SOURCE.parent))
        self.assertIn('Unrecognized worker action', Path(receipt['log']).read_text())
        self.assert_timed(runner.read(self.root / 'jobs/invalid_action.process.json'), 'failed')
        self.assertFalse((self.root / 'jobs/invalid_action.result.json').exists())

    def test_invoke_rejects_nonconverged_and_preserves_worker_result(self):
        (self.root / 'jobs').mkdir()
        spec = dict(id='nonconverged', action='ccg', folder=str(self.root / 'unused'))
        worker_result = dict(status='nonconverged', iterations=7, wall_time_s=.25)
        runner.write(self.root / 'jobs/nonconverged.result.json', worker_result)
        with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as process, \
                redirect_stdout(io.StringIO()):
            result = runner.invoke(spec, self.root)
        self.assertEqual(result['status'], 'nonconverged')
        self.assertEqual(result['wall_time_s'], .25)
        self.assertIn('nonconverged', result['invocation_error']['message'])
        receipt = runner.read(self.root / 'jobs/nonconverged.invocation.json')
        self.assert_timed(receipt, 'failed')
        self.assertEqual(receipt['returncode'], 0)
        self.assertEqual(runner.read(self.root / 'jobs/nonconverged.result.json'), worker_result)
        process.assert_called_once()
        self.assertFalse(process.call_args.kwargs['check'])

    def test_invoke_success_and_missing_result_failure(self):
        (self.root / 'jobs').mkdir()
        for name, result_exists in [('success', True), ('missing', False)]:
            with self.subTest(name=name):
                value = dict(status='completed', marker=name)
                if result_exists:
                    runner.write(self.root / f'jobs/{name}.result.json', value)
                with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                        redirect_stdout(io.StringIO()):
                    result = runner.invoke(dict(id=name, action='fake', folder=str(self.root / 'unused')), self.root)
                if result_exists:
                    self.assertEqual(result, value)
                else:
                    self.assertEqual(result['status'], 'failed')
                self.assert_timed(runner.read(self.root / f'jobs/{name}.invocation.json'),
                                  'completed' if result_exists else 'failed')

    def test_main_nonconverged_worker_returns_failure_and_saves_raw_status(self):
        spec = self.root / 'job.json'
        runner.write(spec, dict(action='ccg'))
        raw = dict(status='nonconverged', iterations=12)
        with patch.object(sys, 'argv', ['runner.py', '--worker', str(spec)]), \
                patch.object(runner, 'worker', return_value=raw):
            self.assertEqual(runner.main(), 1)
        self.assertEqual(runner.read(spec.with_suffix('.result.json')), raw)
        process = runner.read(spec.with_suffix('.process.json'))
        self.assertGreaterEqual(process['wall_time_s'], 0.)
        self.assertIn('ended_at_utc', process)
        # The process record measures successful dispatch; result/exit carry convergence.

    def test_stage_failure_is_persisted_and_propagated(self):
        summary = dict(phases={})
        specs = [dict(id=str(i)) for i in range(3)]
        results = [dict(status='completed'), dict(status='nonconverged'), dict(status='failed')]
        with patch.object(runner, 'invoke', side_effect=lambda spec, output: results[int(spec['id'])]):
            with self.assertRaisesRegex(RuntimeError, 'next stage not permitted'):
                runner.stage('ccg', specs, self.root, summary, 2)
        saved = runner.read(self.root / 'summary.json')['phases']['ccg']
        self.assert_timed(saved, 'failed')
        self.assertEqual(saved['planned'], 3)
        self.assertEqual(saved['workers'], 2)
        self.assertCountEqual(saved['results'], results)

    def test_run_builds_full_grid_before_separate_all_radius_evaluation(self):
        captured = {}
        checkpoint = self.root / 'fake_checkpoint.pth'
        torch.save({'fake': torch.tensor(1)}, checkpoint)
        sealed_summary = self.root / 'fake_ccg_summary.json'
        runner.write(sealed_summary, dict(selected_candidate_id=2, stopping_candidate_id=2))
        report = ModuleType('reporting')
        report._grid = Mock()
        report.source_bindings = Mock(return_value={})
        report.publish = Mock(return_value={'artifacts': []})

        def stage(name, specs, output, summary, workers):
            captured[name] = deepcopy(specs)
            if name == 'prepare':
                return [dict(status='completed') for _ in specs]
            if name == 'nominal':
                return [dict(status='completed', kind=s['kind'], Ntotal=s['n'], seed=s['seed'],
                             nominal=dict(marker=(s['kind'], s['n'], s['seed']))) for s in specs]
            if name == 'ccg':
                self.assertFalse((output / 'training_seal.json').exists())
                return [dict(status='completed', kind=s['kind'], Ntotal=s['n'], seed=s['seed'], radius=s['radius'],
                             summary_path=str(sealed_summary), summary_sha256=runner.sha(sealed_summary),
                             selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=runner.sha(checkpoint)) for s in specs]
            self.assertEqual(name, 'evaluation')
            gate = runner.read(output / 'training_seal.json')
            self.assertEqual(gate['status'], 'sealed')
            self.assertEqual(len(gate['configurations']), 810)
            self.assertTrue(summary['test_accessed'])
            return [dict(status='completed') for _ in specs]

        with patch.object(runner, 'stage', side_effect=stage), \
                patch.object(runner, 'runtime_report', return_value={}), \
                patch.object(runner, 'ROOT', self.root), \
                patch.dict(sys.modules, {'reporting': report}), \
                patch.object(runner, 'backend_for', side_effect=AssertionError('Real backend forbidden')):
            result = runner.run(self.root / 'runs/grid', 1)
        self.assertEqual(list(captured), ['prepare', 'nominal', 'ccg', 'evaluation'])
        self.assertEqual([len(captured[k]) for k in captured], [3, 90, 810, 810])
        for stage_name in ('nominal', 'ccg', 'evaluation'):
            specs = captured[stage_name]
            self.assertEqual(len({s['id'] for s in specs}), len(specs))
            self.assertTrue(all(Path(s['folder']).is_relative_to(self.root / 'runs/grid') for s in specs))
        expected = {(k, n, s, r) for k in ('R3', 'R10', 'shipping_R3')
                    for n in (50, 200, 800) for s in range(10)
                    for r in (0., .001, .003, .01, .03, .2, .4, .7, 1.5)}
        self.assertEqual({(s['kind'], s['n'], s['seed'], s['radius']) for s in captured['ccg']}, expected)
        self.assertEqual({(s['kind'], s['compact']['Ntotal'], s['compact']['seed'], s['compact']['radius'])
                          for s in captured['evaluation']}, expected)
        self.assertEqual(len({s['folder'] for s in captured['evaluation']}), 810)
        for spec in captured['ccg']:
            self.assertEqual(spec['nominal']['marker'], (spec['kind'], spec['n'], spec['seed']))
        self.assert_timed(result)
        report._grid.assert_called_once()
        report.publish.assert_called_once()

    def test_run_failed_training_never_reaches_evaluation_or_publication(self):
        seen = []

        def stage(name, specs, output, summary, workers):
            seen.append(name)
            if name == 'nominal':
                raise RuntimeError('fake nominal stage failed')
            return []

        with patch.object(runner, 'stage', side_effect=stage), \
                patch.object(runner, 'ROOT', self.root), \
                patch.object(runner, 'runtime_report', return_value={}):
            with self.assertRaisesRegex(RuntimeError, 'fake nominal stage failed'):
                runner.run(self.root / 'runs/blocked', 1)
        self.assertEqual(seen, ['prepare', 'nominal'])
        saved = runner.read(self.root / 'runs/blocked/summary.json')
        self.assert_timed(saved, 'failed')
        self.assertFalse(saved['test_accessed'])
        self.assertFalse((self.root / 'runs/blocked/training_seal.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)