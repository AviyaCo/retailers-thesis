"""Solver-free unit fixtures ONLY; never experimental observations.

Parent runs these tests. Importing this file does not import a training backend,
generate the production test set, or start a solver. Worker numerical calls are
mocked; the unchanged comparison evaluator has its own strict-QP tests.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

_path = Path(__file__).with_name('replay.py')
_spec = importlib.util.spec_from_file_location('_radius2_replay_tests_subject', _path)
assert _spec is not None and _spec.loader is not None
r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r)


def bind(path, obj=None, sealed=False):
    if obj is not None:
        path.write_text(json.dumps(obj, allow_nan=False))
    result = dict(file=str(path.resolve()), sha256=r._sha(path))
    if sealed:
        path.with_suffix('.sha256').write_text(result['sha256'])
    return result


class ReplayTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.p = r._plot_module()
        # Reuse the existing complete 810-row artificial validation fixture in
        # a unique namespace, without publishing a bare plot_comparison alias.
        path = r.COMPARISON / 'test_plot.py'
        text = path.read_text()
        self.assertEqual(text.count('import plot_comparison as p'), 1)
        fixture = r._private_module(path, text.replace('import plot_comparison as p', 'p = _plot'),
                                    {'_plot': self.p})
        self.base_manifest, self.old_rows = fixture.fixture(self.root)
        self.oracle = fixture.oracle_fixture(self.base_manifest)
        self.base = self.p.build_plot_data(self.base_manifest, self.old_rows, self.oracle)
        self.green, self.sro = [], []
        for n in r.SIZES:
            for seed in r.SEEDS:
                old = next(row for row in self.old_rows if row['method'] == 'green'
                           and row['Ntotal'] == n and row['seed'] == seed and row['radius'] == 1.5)
                green = deepcopy(old)
                green.update(id=f'green_N{n}_r2_s{seed}', radius=2.,
                    pool_path=old['checkpoint'], pool_file_sha256=old['checkpoint_sha256'],
                    qp_journals={}, timing_clock='time.perf_counter', timing_unit='seconds',
                    start_utc='2026-09-25T00:00:00+00:00', end_utc='2026-09-25T00:00:01+00:00',
                    iterations=0, phases={'evaluation': {'status': 'completed', 'wall_time_s': .125}},
                    environment={'fixture': 'ARTIFICIAL'}, solver_threads=1, worker_count=2)
                sro = {k: deepcopy(green[k]) for k in ('Ntotal', 'seed', 'radius', 'status', 'test',
                    'cached', 'wall_time_s', 'manifest_sha256', 'timing_clock', 'timing_unit',
                    'start_utc', 'end_utc', 'phases', 'environment', 'solver_threads', 'worker_count',
                    'iterations', 'training_runtime_reference', 'input_files')}
                sro.update(id=f'sro_fa_N{n}_r2_s{seed}', method='sro_fa', Ntrain=n,
                           oos=float(n+seed+2), policy_path=old['checkpoint'],
                           policy_sha256=old['checkpoint_sha256'])
                self.green.append(green)
                self.sro.append(sro)
        self.jobs = [{k: deepcopy(v) for k, v in row.items() if k in
            ('id', 'method', 'Ntotal', 'Ntrain', 'seed', 'radius', 'input_files', 'checkpoint',
             'checkpoint_sha256', 'state_sha256', 'policy_path', 'policy_sha256', 'pool_path',
             'pool_sha256', 'pool_file_sha256', 'sizes')}
            for row in self.green + self.sro]
        self.manifest = dict(status='frozen', R=10, radius=2., new_selection=False,
            policies_frozen_before_test_eval=True, n_test=10000, test_seed=9999,
            threads=1, workers=2, membership_tolerance=1e-9, physical_tolerance=1e-4,
            jobs=self.jobs, files=deepcopy(self.base_manifest['files']), base_test=deepcopy(self.base['test']),
            base_manifest=bind(self.root / 'base_manifest.json', self.base_manifest, sealed=True))
        self.manifest_binding = bind(self.root / 'manifest.json', self.manifest, sealed=True)
        for row in self.green + self.sro:
            row['manifest_sha256'] = self.manifest_binding['sha256']
        self.bindings = dict(manifest=self.manifest_binding, results={
            row['id']: bind(self.root / (row['id']+'.json'), row) for row in self.green+self.sro})

    def indexed(self, rows):
        return {r._identity(row): row for row in rows}

    def append(self):
        return r._append(self.base, self.indexed(self.green), self.indexed(self.sro), self.p, self.bindings)

    def test_append_preserves_every_old_plot_value_evidence_and_selection(self):
        before = deepcopy((self.base, self.green, self.sro))
        data = self.append()
        self.assertEqual((self.base, self.green, self.sro), before)
        self.assertEqual(data['validation_selected_oos'], self.base['validation_selected_oos'])
        self.assertEqual(data['provenance'], self.base['provenance'])
        for old, new in zip(self.base['panels'], data['panels']):
            for key in ('scs', 'saa_knn', 'horizontal', 'historical_saa_reference', 'cache_protocol'):
                self.assertEqual(old[key], new[key])
            for method in ('green', 'sro_fa'):
                self.assertEqual(new[method]['radius'][:-1], old[method]['radius'])
                self.assertEqual(new[method]['radius'][-1], 2.)
                self.assertEqual(new[method]['provenance']['per_seed'][:-1], old[method]['provenance']['per_seed'])
                for field in (('oos', 'finite_pool') if method == 'green' else ('oos',)):
                    for stat in ('per_seed', 'mean', 'p20', 'p80'):
                        self.assertEqual(new[method][field][stat][:-1], old[method][field][stat])
            self.assertEqual(new['green']['provenance']['current_evaluation_time_s'][:-1],
                             old['green']['provenance']['current_evaluation_time_s'])
        account = data['replay_accounting']
        self.assertEqual((account['cached_neural_rows'], account['fresh_green_rows'], account['new_sro_rows']),
                         (810, 30, 30))
        self.assertFalse(account['new_selection'])
        self.assertIn('SCS stops at', data['footnote'])
        self.assertEqual(account['panels']['50']['green']['current_evaluation_time_s'][0], [None]*10)
        r._renderer(self.p)._validate_plot_data(data)

    def test_shuffle_does_not_change_seed_order_or_percentiles(self):
        original = self.append()
        self.green.reverse()
        self.sro.reverse()
        self.assertEqual(self.append(), original)
        for panel in original['panels']:
            values = panel['sro_fa']['oos']['per_seed'][-1]
            self.assertEqual(panel['sro_fa']['oos']['p20'][-1], float(np.percentile(values, 20)))
            self.assertEqual(panel['sro_fa']['oos']['p80'][-1], float(np.percentile(values, 80)))

    def test_private_renderer_keeps_original_constants_and_all_validation(self):
        private = r._renderer(self.p)
        self.assertEqual(self.p.GREEN_RADII[-1], 1.5)
        self.assertEqual(private.GREEN_RADII[-1], 2.)
        self.assertEqual(private.SCS_RADII, self.p.SCS_RADII)
        self.assertEqual(private.SCS_COUNTS, {50: 18, 200: 18, 800: 18})
        for change in (lambda d: d['panels'][0]['green']['oos']['mean'].__setitem__(-1, -99),
                       lambda d: d['panels'][0]['scs']['radius'].append(2.),
                       lambda d: d['panels'][0]['green']['provenance']['per_seed'][-1][0].update(status='failed'),
                       lambda d: d['panels'][0]['green']['finite_pool']['per_seed'][-1].reverse()):
            data = self.append()
            change(data)
            with self.assertRaises(ValueError):
                private._validate_plot_data(data)

    def test_namespace_isolation_without_bare_imports(self):
        sentinel = types.ModuleType('plot_comparison')
        green_sentinel = types.ModuleType('green_common')
        with patch.dict(sys.modules, {'plot_comparison': sentinel, 'green_common': green_sentinel}):
            module = r._plot_module()
            self.assertIsNot(module, sentinel)
            self.assertIs(sys.modules['plot_comparison'], sentinel)
            self.assertIs(sys.modules['green_common'], green_sentinel)

    def test_render_emits_only_radius2_filenames_without_mutating_data(self):
        data = self.append()
        before = deepcopy(data)
        private = r._renderer(self.p)
        output = self.root / 'render_only'
        private.render(data, output)
        self.assertEqual(data, before)
        self.assertEqual({path.name for path in output.iterdir()},
                         {'comparison_R10_radius2.png', 'comparison_R10_radius2.pdf'})
        self.assertTrue(all(path.stat().st_size > 0 for path in output.iterdir()))

    def test_manifest_requires_all_sixty_selected_policies_before_evaluation(self):
        self.assertEqual(r._manifest(self.manifest_binding), self.manifest)
        changes = (lambda m: m['jobs'].pop(), lambda m: m['jobs'].__setitem__(1, m['jobs'][0]),
                   lambda m: m.update(policies_frozen_before_test_eval=False),
                   lambda m: m.update(new_selection=True),
                   lambda m: m['jobs'][0].update(Ntrain=50),
                   lambda m: m['jobs'][0].update(radius=1.5),
                   lambda m: m['jobs'][0].update(seed=True),
                   lambda m: m['base_test']['arrays']['xi1'].update(shape=[10000, 3]),
                   lambda m: m['jobs'][-1].update(policy_sha256='0'*64))
        for change in changes:
            m = deepcopy(self.manifest)
            change(m)
            binding = bind(self.root / 'bad_manifest.json', m, sealed=True)
            with self.subTest(change=change), self.assertRaises(ValueError):
                r._manifest(binding)

    def test_manifest_sidecar_and_frozen_copy_tampering_rejected(self):
        path = Path(self.manifest_binding['file'])
        path.with_suffix('.sha256').write_text('0'*64)
        with self.assertRaisesRegex(ValueError, 'sidecar'):
            r._manifest(self.manifest_binding)
        path.with_suffix('.sha256').write_text(self.manifest_binding['sha256'])
        source = self.jobs[0]['checkpoint']
        Path(self.manifest['files'][source]['copy']).write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            r._manifest(self.manifest_binding)

    def validate(self, rows, method):
        return r._new_rows(rows, method, self.manifest, self.manifest_binding['sha256'], self.bindings, self.p)

    def test_new_rows_reject_missing_duplicate_failure_nonfinite_and_bad_bindings(self):
        with patch.object(r, '_journals'):
            self.assertEqual(len(self.validate(self.green, 'green')), 30)
            self.assertEqual(len(self.validate(self.sro, 'sro_fa')), 30)
            for change in (lambda rows: rows.pop(), lambda rows: rows.__setitem__(1, rows[0]),
                           lambda rows: rows[0].update(status='failed'),
                           lambda rows: rows[0].update(cached=True),
                           lambda rows: rows[0].update(mean_oos=float('nan')),
                           lambda rows: rows[0].update(manifest_sha256='0'*64),
                           lambda rows: rows[0].update(solves_accepted=10000),
                           lambda rows: rows[0].update(Ntrain=50),
                           lambda rows: rows[0].update(wall_time_s=-1)):
                rows = deepcopy(self.green)
                change(rows)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.validate(rows, 'green')
        # Rebind deliberately invalid ARTIFICIAL rows so semantic validation,
        # not just persisted-result equality, is exercised.
        for method, rows, field, value in (('green', self.green, 'status_counts', {'9': 10080}),
                ('green', self.green, 'finite_pool_mean', -999.),
                ('sro_fa', self.sro, 'wall_time_s', -1.),
                ('sro_fa', self.sro, 'status', 'failed'),
                ('sro_fa', self.sro, 'worker_count', 1)):
            altered = deepcopy(rows)
            altered[0][field] = value
            binding = self.bindings['results'][altered[0]['id']]
            self.bindings['results'][altered[0]['id']] = bind(Path(binding['file']), altered[0])
            with self.subTest(field=field), patch.object(r, '_journals'), self.assertRaises(ValueError):
                self.validate(altered, method)
            self.bindings['results'][rows[0]['id']] = bind(Path(binding['file']), rows[0])

    def test_raw_qp_journals_require_every_optimal_row_and_center(self):
        row = deepcopy(self.green[0])
        row['qp_journals'] = {}
        for name, count in (('oos', 10000), ('is', sum(row['sizes']))):
            path = self.root / f'{name}.jsonl'
            centers = np.repeat(np.arange(row['Ntrain']), row['sizes'])
            row['original_sample_ids'] = list(range(row['Ntrain']))
            with path.open('w') as stream:
                for i in range(count):
                    entry = dict(row=i, phase=name, status=2, accepted=True, controller_wall_time_s=.001)
                    if name == 'is':
                        entry.update(center_id=int(centers[i]), original_sample_id=int(centers[i]))
                    stream.write(json.dumps(entry)+'\n')
            row['qp_journals'][name] = bind(path)
        r._journals(row)
        path = Path(row['qp_journals']['is']['file'])
        lines = path.read_text().splitlines()
        lines[-1] = json.dumps(dict(json.loads(lines[-1]), status=9, accepted=False))
        path.write_text('\n'.join(lines)+'\n')
        row['qp_journals']['is'] = bind(path)
        with self.assertRaisesRegex(ValueError, 'non-OPTIMAL'):
            r._journals(row)

    def test_worker_failure_is_persisted_no_hidden_fallback_or_overwrite(self):
        with patch.object(r, '_load_green_inputs', side_effect=ValueError('ARTIFICIAL input failure')):
            row = r.evaluate_green(self.green[0], self.root / 'manifest.json', self.root / 'worker')
        self.assertEqual(row['status'], 'failed')
        self.assertIn('ARTIFICIAL input failure', row['error'])
        self.assertEqual(row['solves_attempted'], 0)
        self.assertIsNone(row['mean_oos'])
        self.assertGreaterEqual(row['wall_time_s'], 0.)
        self.assertEqual(row['phases']['input_loading']['status'], 'failed')
        self.assertEqual(r._json(row['result_path']), row)
        with self.assertRaises(FileExistsError):
            r.evaluate_green(self.green[0], self.root / 'manifest.json', self.root / 'worker')

    def test_worker_calls_only_two_r10_evaluations_and_preserves_own_pool(self):
        job = deepcopy(self.jobs[0])
        job['original_sample_ids'] = list(range(job['Ntrain']))
        model = object()
        # Artificial 20-coordinate pools, not production/test draws.
        pools = [np.full((size, 20), float(i)) for i, size in enumerate(job['sizes'])]
        arrays = [object(), object()]
        reference = types.SimpleNamespace(state_hash=lambda unused: job['state_sha256'],
            torch=types.SimpleNamespace(get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1))
        calls = []

        def evaluate(actual_model, xi1, xi2, name, record, folder, sizes=None):
            self.assertIs(actual_model, model)
            calls.append(name)
            if name == 'oos':
                self.assertIs(xi1, arrays[0])
                self.assertIs(xi2, arrays[1])
                record['mean_oos'] = 3.
                count = 10000
            else:
                self.assertEqual(sizes, job['sizes'])
                self.assertEqual(xi1.shape, (sum(sizes), 10))
                self.assertEqual(xi2.shape, (sum(sizes), 10))
                np.testing.assert_array_equal(xi1, np.concatenate(pools)[:, :10])
                np.testing.assert_array_equal(xi2, np.concatenate(pools)[:, 10:])
                record.update(per_center_maxima=[4.]*job['Ntrain'], finite_pool_mean=4.)
                count = sum(sizes)
            record['solves_attempted'] += count
            record['solves_accepted'] += count
            record['status_counts']['2'] = record['solves_accepted']
            (folder / f'{name}_solves.jsonl').write_text('ARTIFICIAL MOCK JOURNAL\n')

        evaluator = types.SimpleNamespace(c=reference, _evaluate=evaluate)
        result = (evaluator, self.manifest, self.manifest_binding['sha256'], model, arrays, pools)
        with patch.object(r, '_load_green_inputs', return_value=result), \
             patch.object(r, '_sources'), patch.object(r, '_bound'):
            row = r.evaluate_green(job, self.manifest_binding['file'], self.root / 'happy_worker')
        self.assertEqual(row['status'], 'completed', row.get('error'))
        self.assertEqual(calls, ['oos', 'is'])
        self.assertEqual(row['test'], self.manifest['base_test'])
        self.assertEqual(row['per_center_maxima'], [4.]*job['Ntrain'])
        self.assertEqual(set(row['qp_journals']), {'oos', 'is'})
        self.assertEqual(r._json(row['result_path']), row)

    def test_merge_never_stages_a_partial_cohort(self):
        output = self.root / 'stage'
        with patch.object(r, '_manifest', return_value=self.manifest), \
             patch.object(r, '_sources'), patch.object(r, '_base', return_value=self.base), \
             self.assertRaises(ValueError):
            r.merge_and_render(self.root / 'base.json', self.green[:-1], self.sro, output, self.bindings)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()