"""Isolated controller fixtures: no production imports, research data, or solves."""
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


def configure_fixture_worker():
    pass


def fixture_worker(kind, params, folder, provenance_path):
    """Picklable tiny worker used only by the actual spawn-pool fixture."""
    return dict(Ntotal=params['n_total'], seed=params['seed'], status='completed',
                complete=True, wall_time_s=0., result_path=str(Path(folder) / 'result.json'))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, default=lambda item: item.tolist()))


@contextmanager
def timed(record):
    start = time.perf_counter()
    record.update(status='running', started_at_utc='fixture-start')
    try:
        yield
    except BaseException:
        record['status'] = 'failed'
        raise
    else:
        record['status'] = 'completed'
    finally:
        record.update(wall_time_s=time.perf_counter()-start, ended_at_utc='fixture-end')


def verify_sources(manifest):
    for mapping in ('source_sha256', 'input_sha256'):
        for path, expected in manifest.get(mapping, {}).items():
            check(digest(path) == expected, 'Changed fixture binding')


def isolated_campaign():
    """Load real runner code against fixture primitives, not import-time archives."""
    common = ModuleType('shipping_live_common')
    common.ROOT = Path(__file__).resolve().parent
    common.PRODUCTION = Path('/unused-fixture-production')
    common.PROTOCOL = dict(full_CCG_launched=True, fixture={'unchanged': True})
    common.SIZES, common.SEEDS = (50,), (0,)
    common.LR_GRID, common.RADII = (.005, .01, .05), (0., .2)
    common.THREAD_ENV = {}
    common.check, common.digest, common.read = check, digest, read
    common.np = np
    common.core = SimpleNamespace(timed=timed, utc_now=lambda: 'fixture-utc',
        write_json=write_json, configure_worker=configure_fixture_worker)
    common.legacy_ccg = SimpleNamespace(state_hash=lambda model: 'fixture-state')
    common.nominal_common = SimpleNamespace(split_data=Mock())
    common.shipping_model = SimpleNamespace()
    common.shipping_resources = SimpleNamespace(resources=Mock(return_value={}),
        choose_workers=Mock(return_value={'nominal': 2}))
    common.torch = SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1)
    common.metadata = lambda n, seed, radius, method: dict(
        Ntotal=n, seed=seed, radius=radius, method=method)
    common.verify_sources = verify_sources
    common.validate_integration = Mock()
    spec = importlib.util.spec_from_file_location('_campaign_fixture', common.ROOT / 'run_campaign.py')
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get('shipping_live_common')
    sys.modules['shipping_live_common'] = common
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            del sys.modules['shipping_live_common']
        else:
            sys.modules['shipping_live_common'] = previous
    return module


class CampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.c = isolated_campaign()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(self.c, 'persist', side_effect=self.persist))

    def persist(self, output, sweep):
        sweep['wall_time_human'] = 'fixture timing'

    def gate(self, peak=800000000):
        summary = self.root / 'summary.json'
        write_json(summary, {'fixture': True})
        frozen = {str(self.c.ROOT / 'shipping_ccg.py'): digest(self.c.ROOT / 'shipping_ccg.py')}
        manifest = self.root / 'provenance.json'
        write_json(manifest, dict(source_sha256=frozen, protocol=self.c.PROTOCOL))
        self.c.common.validate_integration.return_value = dict(peak_rss_bytes=peak)
        argv = ['run_campaign', '--output', str(self.root / 'out'),
                '--preflight', str(summary), '--preflight-sha256', digest(summary)]
        self.stack.enter_context(patch.object(self.c.sys, 'argv', argv))
        return summary, manifest, frozen

    def test_cli_requires_preflight_and_hash(self):
        for extras in ([], ['--preflight', str(self.root / 'summary.json')]):
            with self.subTest(extras=extras), patch.object(self.c.sys, 'argv',
                    ['run_campaign', '--output', str(self.root / 'out')] + extras), \
                    patch.object(self.c, 'load_cached_grid') as load:
                with self.assertRaises(SystemExit):
                    self.c.main()
                load.assert_not_called()

    def test_bad_gate_refused_before_cached_grid(self):
        summary, _, _ = self.gate()
        self.c.common.validate_integration.side_effect = ValueError('bad integration gate')
        with patch.object(self.c, 'load_cached_grid') as load:
            with self.assertRaisesRegex(ValueError, 'bad integration gate'):
                self.c.main()
            load.assert_not_called()
        self.c.common.validate_integration.assert_called_once_with(summary, digest(summary))

    def test_wrong_summary_hash_refused_before_cached_grid(self):
        summary, _, _ = self.gate()
        summary.write_text('changed')
        with patch.object(self.c, 'load_cached_grid') as load:
            with self.assertRaisesRegex(ValueError, 'Changed input'):
                self.c.main()
            load.assert_not_called()
        self.c.common.validate_integration.assert_not_called()

    def test_manifest_change_during_gate_refused_before_load(self):
        _, manifest, _ = self.gate()
        def validate(*args):
            row = read(manifest)
            row['protocol']['fixture']['unchanged'] = False
            write_json(manifest, row)
            return dict(peak_rss_bytes=800000000)
        self.c.common.validate_integration.side_effect = validate
        with patch.object(self.c, 'load_cached_grid') as load:
            with self.assertRaisesRegex(ValueError, 'Changed fixture binding'):
                self.c.main()
            load.assert_not_called()

    def test_gate_bindings_frozen_sources_and_peak_before_load(self):
        summary, manifest, frozen = self.gate()
        before = copy.deepcopy(self.c.PROTOCOL)
        def load(output, sweep, inputs):
            self.assertEqual(inputs[str(summary)], digest(summary))
            self.assertEqual(inputs[str(manifest)], digest(manifest))
            self.assertEqual(read(manifest)['source_sha256'], frozen)
            self.assertEqual(sweep['resource_peak_evidence']['peak_rss_bytes'], 800000000)
            self.assertIn('not memory certification', sweep['resource_peak_evidence']['scope'])
            raise RuntimeError('fixture stops before data')
        with patch.object(self.c, 'load_cached_grid', side_effect=load):
            with self.assertRaisesRegex(RuntimeError, 'fixture stops'):
                self.c.main()
        self.c.shipping_resources.choose_workers.assert_called_once()
        self.assertEqual(self.c.shipping_resources.choose_workers.call_args.args[1], 800000000)
        self.assertEqual(self.c.PROTOCOL, before)

    def test_readiness_peak_is_lower_bound(self):
        self.gate(1)
        with patch.object(self.c, 'load_cached_grid', side_effect=RuntimeError('fixture stop')):
            with self.assertRaisesRegex(RuntimeError, 'fixture stop'):
                self.c.main()
        self.assertEqual(self.c.shipping_resources.choose_workers.call_args.args[1], 713433088)

    def test_manifest_copies_frozen_mapping_and_protocol(self):
        _, _, frozen = self.gate()
        before = copy.deepcopy(self.c.PROTOCOL)
        captured = []
        def write(path, value):
            captured.append(value)
            write_json(path, value)
        with patch.object(self.c.core, 'write_json', side_effect=write), \
                patch.object(self.c.importlib.metadata, 'version', return_value='fixture'):
            self.c._manifest(self.root / 'nominal.json', {}, {}, {}, frozen)
            self.c._manifest(self.root / 'ccg.json', {}, {}, {}, frozen, ccg=True)
        self.assertEqual(captured[0]['source_sha256'], frozen)
        self.assertIsNot(captured[0]['source_sha256'], frozen)
        self.assertFalse(captured[0]['protocol']['full_CCG_launched'])
        self.assertTrue(captured[1]['protocol']['full_CCG_launched'])
        self.assertEqual(self.c.PROTOCOL, before)
        captured[0]['source_sha256']['fixture-mutation'] = 'not-in-original'
        self.assertNotIn('fixture-mutation', frozen)

    def test_both_main_barriers_block_later_phases(self):
        self.gate()
        source_map = read(self.root / 'provenance.json')['source_sha256']
        setup = dict(Ntotal=50, seed=0)
        selected = dict(Ntotal=50, seed=0, status='completed', complete=True,
            nominal_record=dict(training_data_path='fixture-train', training_data_sha256='fixture'),
            result_path='fixture-result', result_sha256='fixture')
        cases = [dict(Ntotal=50, seed=0, radius=r, status='completed', complete=True,
                      converged=False) for r in self.c.RADII]
        real_bind = self.c._bind
        def bind(inputs, path, expected=None):
            if Path(path).is_file():
                return real_bind(inputs, path, expected)
            inputs[str(path)] = 'fixture'
            return str(path)
        def manifest(path, inputs, workers, environment, source_hashes, **kwargs):
            self.assertEqual(source_hashes, source_map)
            return path
        for first, later, message in (([], [], 'Incomplete/duplicate'),
                                      ([selected], cases, 'Nonconverged')):
            with self.subTest(message=message), \
                    patch.object(self.c, 'load_cached_grid', return_value=([setup], [])), \
                    patch.object(self.c, '_bind', side_effect=bind), \
                    patch.object(self.c, '_manifest', side_effect=manifest), \
                    patch.object(self.c, 'run_stage', side_effect=[first, later]) as stage, \
                    patch.object(Path, 'mkdir'):
                with self.assertRaisesRegex(ValueError, message):
                    self.c.main()
                self.assertEqual([call.args[0] for call in stage.call_args_list],
                    ['nominal_validation'] if not first else ['nominal_validation', 'ccg'])

    def test_worker_rejects_changed_preflight_before_selection(self):
        summary, manifest, frozen = self.gate()
        child = self.root / 'child.json'
        write_json(child, dict(source_sha256=frozen, workers={},
            input_sha256={str(summary): digest(summary), str(manifest): digest(manifest)}))
        for path in (summary, manifest):
            with self.subTest(path=path):
                old = path.read_text()
                path.write_text(old + ' ')
                with patch.object(self.c, 'select_nominal') as select:
                    row = self.c.worker('nominal_validation', dict(n_total=50, seed=0),
                        self.root / ('worker-' + path.stem), child)
                self.assertEqual(row['status'], 'failed')
                self.assertFalse(row['complete'])
                self.assertIn('Changed fixture binding', row['error']['message'])
                select.assert_not_called()
                path.write_text(old)

    def split_fixture(self):
        x1 = np.arange(150, dtype=float).reshape(50, 3)
        x2 = x1 + .25
        train_ids, val_ids = np.arange(40), np.arange(40, 50)
        pair_hash = self.c.array_hash(x1, x2)
        train_hash = self.c.array_hash(x1[train_ids], x2[train_ids])
        shared = dict(Ntotal=50, seed=0, split_seed=100000, paired_data_sha256=pair_hash)
        train = dict(shared, x1=x1[train_ids], x2=x2[train_ids],
                     original_sample_ids=train_ids, training_array_sha256=train_hash)
        split = dict(shared, train_ids=train_ids, holdout_ids=val_ids,
                     train_array_sha256=train_hash,
                     holdout_array_sha256=self.c.array_hash(x1[val_ids], x2[val_ids]))
        setup = dict(Ntotal=50, seed=0, paired_data_sha256=pair_hash)
        for key, value in (('training_data', train), ('split', split)):
            path = self.root / (key + '.json')
            write_json(path, value)
            setup[key + '_path'], setup[key + '_sha256'] = str(path), digest(path)
        data = dict(xi1=x1, xi2=x2, train_indices=train_ids, val_indices=val_ids)
        self.c.nominal_common.split_data.return_value = data
        return setup, data

    def test_verified_split_checks_exact_arrays_ids_and_hashes(self):
        setup, data = self.split_fixture()
        self.assertIs(self.c.verified_split(setup), data)
        self.c.nominal_common.split_data.assert_called_once_with(50, 0)
        data['xi1'][49, 0] += .001
        with self.assertRaisesRegex(ValueError, 'paired hash'):
            self.c.verified_split(setup)

    def test_verified_split_rejects_changed_file_and_holdout_hash(self):
        setup, _ = self.split_fixture()
        path = Path(setup['split_path'])
        split = read(path)
        split['holdout_array_sha256'] = 'wrong'
        write_json(path, split)
        with self.assertRaisesRegex(ValueError, 'Changed split'):
            self.c.verified_split(setup)
        setup['split_sha256'] = digest(path)
        with self.assertRaisesRegex(ValueError, 'holdout hash'):
            self.c.verified_split(setup)

    def test_select_nominal_uses_hard_cost_and_exact_tie_not_smooth(self):
        setup, data = self.split_fixture()
        folder = self.root / 'selection'
        folder.mkdir()
        scores = {.005: (1., 9.), .01: (1., 3.), .05: (2., 0.)}
        fits = [dict(lr_net=lr, checkpoint=lr, checkpoint_sha256='fixture',
                     selected_training_loss=7., training_timing={}) for lr in self.c.LR_GRID]
        def evaluate(model, x1, x2, record):
            if len(x1) == 10:
                np.testing.assert_array_equal(x1, data['xi1'][data['val_indices']])
                hard, smooth = scores[model]
            else:
                np.testing.assert_array_equal(x1, data['xi1'][data['train_indices']])
                hard, smooth = 8., 7.
            record.update(hard_mean=hard, smooth_mean=smooth)
        record = {}
        with patch.object(self.c, '_load_model', side_effect=lambda path, sha: path), \
                patch.object(self.c, 'native_evaluate', side_effect=evaluate) as evaluate_mock:
            self.c.select_nominal(dict(setup=setup, fits=fits), folder, record)
        self.assertEqual(record['selected_lr'], .005)
        self.assertEqual(record['validation_hard_mean'], 1.)
        self.assertEqual(evaluate_mock.call_count, 4)
        self.assertTrue(record['complete'])

    def stage_fixture(self, rows):
        futures = []
        for row in rows:
            future = Future()
            if isinstance(row, BaseException):
                future.set_exception(row)
            else:
                future.set_result(row)
            future.cancel = Mock(wraps=future.cancel)
            futures.append(future)
        pool = Mock()
        pool.submit.side_effect = futures
        self.stack.enter_context(patch.object(self.c, 'ProcessPoolExecutor', return_value=pool))
        self.stack.enter_context(patch.object(self.c, 'as_completed', return_value=iter(futures)))
        tasks = [(dict(n_total=50, seed=i), self.root / f'job{i}') for i in range(len(rows))]
        sweep = dict(phases={}, protocol={})
        return pool, futures, tasks, sweep

    def test_ordinary_failed_job_does_not_cancel_others(self):
        rows = [dict(Ntotal=50, seed=i, status=status, complete=status == 'completed')
                for i, status in enumerate(('failed', 'completed'))]
        pool, futures, tasks, sweep = self.stage_fixture(rows)
        result = self.c.run_stage('nominal_validation', tasks, 2, self.root, 'fixture', sweep)
        self.assertEqual(result, rows)
        for future in futures:
            future.cancel.assert_not_called()
        pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        with self.assertRaisesRegex(ValueError, 'Failed/incomplete'):
            self.c.require_grid(result, {(50, 0), (50, 1)}, ('Ntotal', 'seed'))

    def test_exceptional_future_keeps_failure_despite_persisted_success(self):
        pool, _, tasks, sweep = self.stage_fixture([RuntimeError('fixture transport failure')])
        folder = tasks[0][1]
        folder.mkdir()
        write_json(folder / 'result.json', dict(Ntotal=50, seed=0, status='completed', complete=True))
        with self.assertRaisesRegex(RuntimeError, 'fixture transport failure'):
            self.c.run_stage('nominal_validation', tasks, 1, self.root, 'fixture', sweep)
        phase = sweep['phases']['nominal_validation']
        row = phase['jobs'][0]
        self.assertEqual(phase['status'], 'failed')
        self.assertEqual(row['status'], 'future_failed')
        self.assertFalse(row['complete'])
        self.assertEqual(row['future_error']['message'], 'fixture transport failure')
        self.assertEqual(row['raw_persisted_result_path'], str(folder / 'result.json'))
        self.assertIn('Unverified', row['raw_persisted_result_scope'])
        pool.shutdown.assert_called_once()
        self.assertEqual(pool.submit.call_count, 1)
        with self.assertRaisesRegex(ValueError, 'Failed/incomplete'):
            self.c.require_grid(phase['jobs'], {(50, 0)}, ('Ntotal', 'seed'))

    def test_persist_failure_still_shuts_down_pool(self):
        pool, _, tasks, sweep = self.stage_fixture([])
        with patch.object(self.c, 'persist', side_effect=OSError('fixture disk failure')):
            with self.assertRaisesRegex(OSError, 'fixture disk failure'):
                self.c.run_stage('nominal_validation', tasks, 1, self.root, 'fixture', sweep)
        pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_actual_spawn_pool_serializes_tiny_worker_result(self):
        tasks = [(dict(n_total=50, seed=0), self.root / 'spawn-job')]
        sweep = dict(phases={}, protocol={})
        with patch.object(self.c, 'worker', fixture_worker):
            rows = self.c.run_stage('nominal_validation', tasks, 1, self.root, 'fixture', sweep)
        self.c.require_grid(rows, {(50, 0)}, ('Ntotal', 'seed'))
        self.assertEqual(sweep['phases']['nominal_validation']['status'], 'completed')


if __name__ == '__main__':
    unittest.main()