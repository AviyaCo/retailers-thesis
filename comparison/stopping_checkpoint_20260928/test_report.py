"""Generated, solver-free publisher fixtures; never evaluate or train a policy."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


spec = importlib.util.spec_from_file_location('stopping_report', Path(__file__).with_name('report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(root):
    """All 1,110 identities, 11 caps, matched seeds, and real hash-bound JSON files."""
    tests = {str(r): dict(n_test=10000, seed=9999, arrays={name: dict(
        shape=[10000, r], dtype='float64', array_sha256=digest(f'{r}/{name}/array'),
        file_sha256=digest(f'{r}/{name}/file'), file=f'/synthetic/R{r}/{name}.npy')
        for name in ('xi1', 'xi2')}) for r in (3, 10)}
    phase = dict(status='completed', wall_time_s=2., start_utc='2026-09-28T00:00:00+00:00',
                 end_utc='2026-09-28T00:00:02+00:00', timing_clock='time.perf_counter')
    jobs, rows, capped = [], [], 0
    for method in ('green', 'black'):
        for r in (3, 10):
            for n in report.SIZES:
                for seed in report.SEEDS:
                    for rho in report._grid(method, r):
                        cap = method == 'black' and capped < 11
                        capped += cap
                        stop = 40 if cap else (2 if method == 'black' else 1)
                        identifier = f'{method}_R{r}_N{n}_s{seed}_rho{rho:g}'
                        status = 'iteration_cap' if cap else (
                            'green_gap_satisfied' if method == 'green' else 'converged')
                        hist = dict(status='persisted_not_current',
                            nominal=dict(training_timing=dict(phase, wall_time_s=10.+seed,
                                                              epochs_attempted=12, seed=seed)),
                            iterations=[dict(candidate_id=0, ap=phase, mp=phase, epochs=12)],
                            green_through_stop=dict(active_phase_time_s=2., iterations=[dict(ap=phase)],
                                                    timing_scope='persisted phase sum; not elapsed'),
                            black_after_green_stop=dict(active_phase_time_s=2., iterations=[dict(mp=phase)],
                                                        timing_scope='persisted phase sum; not elapsed'),
                            total_wall_time_s=None if cap else 6., original_prefix={'wall_time_s': 4.},
                            continuation={'wall_time_s': 2.} if method == 'black' else None,
                            current_training_performed=False)
                        job = dict(id=identifier, method=method, R=r, Ntotal=n, Ntrain=4*n//5,
                            seed=seed, radius=rho, checkpoint_sha256=digest(identifier+'/checkpoint'),
                            state_sha256=digest(identifier+'/state'), pool_sha256=digest(identifier+'/pool'),
                            pool_file_sha256=digest(identifier+'/poolfile'),
                            training_data_sha256=digest(f'{r}/{n}/{seed}/training'),
                            sizes=[1]*(4*n//5), selected_candidate_id=stop, stopping_candidate_id=stop,
                            converged=not cap, terminal_status=status, historical_training=hist,
                            input_sha256={'/synthetic/large-map': digest('inputs')})
                        row = {k: deepcopy(job[k]) for k in report.IDENTITY}
                        value = r + n / 100. + seed + rho + (3. if method == 'black' else 0.)
                        maxima = np.arange(job['Ntrain'], dtype=float) + seed
                        row.update(status='completed', cached=False, mean_oos=value,
                            finite_pool_mean=float(maxima.mean()), per_center_maxima=maxima.tolist(),
                            sizes=job['sizes'].copy(), test=deepcopy(tests[str(r)]), wall_time_s=2.+seed,
                            phases={'evaluation': deepcopy(phase)}, start_utc=phase['start_utc'],
                            end_utc=phase['end_utc'], solver_threads=1, worker_count=16,
                            iterations=20, timing_clock='time.perf_counter')
                        path = root / (identifier + '.json')
                        sha = write(path, row)
                        row.update(result_path=str(path), result_sha256=sha, journal_verified=True)
                        jobs.append(job)
                        rows.append(row)
    bases, bindings = {}, {}
    for r in (3, 10):
        panels = []
        for n in report.SIZES:
            panels.append(dict(Ntotal=n, label=f'N = {n}', training_counts={'green': 4*n//5, 'scs': n},
                extra_immutable={'nested': [False, None, {'claim': 'original qualification'}]},
                green={'historical_selection': 'minimum AP'}, black={'historical_selection': 'minimum AP'},
                scs=dict(label='SCS', radius=[0., .01, 1.5],
                    oos=dict(mean=[10., 12., 13.], p20=[8., 10., 11.], p80=[12., 14., 15.]),
                    finite_pool={'mean': [11., 13., 14.]}, provenance={'cached': True, 'preserve': [1, 2]}),
                sro_fa=dict(label='SRO-FA', radius=[0., .2, 1.5],
                    oos=dict(mean=[12., 13., 14.], p20=[10., 11., 12.], p80=[14., 15., 16.])),
                saa_knn=dict(oos=dict(mean=11., p20=10., p80=12.), original_provenance=['untouched']),
                horizontal=[dict(kind='conditional' if r == 3 else 'empirical', value=9.125,
                                 label='Historical reference', interpretation='not a population bound')]))
        base = dict(schema_version=1, status='completed', panels=panels, seeds=report.SEEDS.copy(),
            scs_radii=[0., .01, 1.5], test=deepcopy(tests[str(r)]),
            band_definition='old bands', is_definition='old IS', caption='old minimum AP claim',
            footnote='old validation-selected footnote', provenance={'old': 'minimum AP'},
            validation_selected_oos={'do_not_present_as_current': [1, 2]},
            selection_interpretation='validation winners', black_overlay={'bindings': {'huge_map': 'old'}})
        path = root / f'base_R{r}.json'
        bindings[str(r)] = dict(path=str(path), sha256=write(path, base))
        bases[r] = base
    manifest = dict(jobs=jobs, tests=tests, base_plot_data=bindings,
                    source_sha256={'/synthetic/source.py': digest('source')},
                    selection_rule='stopping_checkpoint', return_rule='stopping_checkpoint')
    return manifest, rows, bases


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = tempfile.TemporaryDirectory()
        cls.original_manifest, cls.original_rows, cls.bases = fixture(Path(cls.inputs.name))

    @classmethod
    def tearDownClass(cls):
        cls.inputs.cleanup()

    def setUp(self):
        self.manifest = deepcopy(self.original_manifest)
        self.rows = deepcopy(self.original_rows)
        self.staging = tempfile.TemporaryDirectory()
        self.addCleanup(self.staging.cleanup)
        self.output = Path(self.staging.name)

    def rejected(self, message):
        with patch.object(report, '_render') as render:
            with self.assertRaisesRegex((ValueError, KeyError), message):
                report.publish_data(self.manifest, self.rows, self.output)
            render.assert_not_called()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_complete_publication_actual_render_and_seals(self):
        before_manifest, before_rows = deepcopy(self.manifest), deepcopy(self.rows)
        solvers_before = {k for k in sys.modules if k.startswith(('gurobi', 'torch', 'cvxpy'))}
        # Deliberately reorder input: per-seed curves must still use seeds 0..9.
        paths = report.publish_data(self.manifest, list(reversed(self.rows)), self.output)
        self.assertEqual(set(paths), set(self.output.iterdir()))
        self.assertEqual(len(paths), 15)
        self.assertEqual(self.manifest, before_manifest)
        self.assertEqual(self.rows, before_rows)
        self.assertEqual(solvers_before, {k for k in sys.modules if k.startswith(('gurobi', 'torch', 'cvxpy'))})
        for path in paths:
            if path.suffix == '.json':
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 path.with_suffix('.sha256').read_text().strip())
        for r in (3, 10):
            data = json.loads((self.output / f'plot_data_R{r}.json').read_text())
            base = self.bases[r]
            self.assertNotIn('validation_selected_oos', data)
            self.assertNotIn('black_overlay', data)
            self.assertNotIn('selection_interpretation', data)
            history = data['historical_base_metadata']
            self.assertEqual(history['validation_selected_oos'], base['validation_selected_oos'])
            self.assertEqual(history['caption'], base['caption'])
            self.assertEqual(history['provenance'], base['provenance'])
            self.assertEqual(history['black_overlay']['base_plot_data'], self.manifest['base_plot_data'][str(r)])
            for actual, original in zip(data['panels'], base['panels']):
                self.assertEqual({k: v for k, v in actual.items() if k not in ('green', 'black')},
                                 {k: v for k, v in original.items() if k not in ('green', 'black')})
                for method in ('green', 'black'):
                    curve = actual[method]
                    self.assertEqual(curve['radius'], list(report._grid(method, r)))
                    expected = np.array([r + actual['Ntotal']/100. + seed + (3. if method == 'black' else 0.)
                                         for seed in report.SEEDS])
                    self.assertEqual(curve['oos']['per_seed'][0], expected.tolist())
                    self.assertEqual(curve['oos']['p20'][0], float(np.percentile(expected, 20)))
                    self.assertEqual(curve['oos']['p80'][0], float(np.percentile(expected, 80)))
                    self.assertNotIn('input_sha256', curve['provenance']['per_seed'][0][0])
            self.assertGreater((self.output / f'comparison_R{r}_stopping.png').stat().st_size, 1000)
            self.assertTrue((self.output / f'comparison_R{r}_stopping.pdf').read_bytes().startswith(b'%PDF'))
            binding = self.manifest['base_plot_data'][str(r)]
            self.assertEqual(hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest(), binding['sha256'])
        runtime = json.loads((self.output / 'runtime_report.json').read_text())
        self.assertEqual(len(runtime['nominal_selected_arms']), 60)
        self.assertTrue(all(a['training_wall_time_s']['count'] == 10 for a in runtime['nominal_aggregates']))
        caps = [a for a in runtime['aggregates'] if a['training_status'] == 'capped']
        self.assertEqual(sum(a['count'] for a in caps), 11)
        self.assertTrue(all(a['persisted_training_total_wall_time_s']['count'] == 0 for a in caps))
        self.assertTrue(all(a['persisted_training_total_wall_time_s']['mean'] is None for a in caps))
        self.assertTrue(all(a['persisted_training_active_phase_time_s']['mean'] == 4. for a in caps))
        self.assertIsNone(runtime['whole_pipeline_elapsed_s'])
        self.assertIsNone(runtime['sro_fa']['total_wall_time_s'])
        self.assertEqual(runtime['records'][0]['historical_training']['iterations'],
                         self.manifest['jobs'][0]['historical_training']['iterations'])
        self.assertEqual(runtime['records'][0]['current_evaluation']['worker_count'], 16)
        costs = json.loads((self.output / 'fixed_radius_comparison.json').read_text())
        self.assertEqual(len(costs), 57)
        for c in costs:
            if c['radius'] == 2.:
                self.assertIsNone(c['green_oos'])
                self.assertIsNone(c['black_minus_green'])
                self.assertEqual(c['paired_seed_count'], 0)
            else:
                self.assertAlmostEqual(c['black_minus_green'], 3.)
                self.assertEqual(c['paired_seed_count'], 10)
        summary = json.loads((self.output / 'report.json').read_text())
        for name, sha in summary['artifacts'].items():
            self.assertEqual(sha, hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        text = (self.output / 'RESULTS.md').read_text()
        self.assertIn('559 converged + 11 capped', text)
        self.assertIn('not confidence intervals', text)
        self.assertIn('pending parent completion', text)
        self.assertFalse((self.output / 'RESULTS.sha256').exists())

    def test_missing_seed(self):
        self.rows.pop()
        self.rejected('Incomplete')

    def test_renderer_actual_values_symlog_and_black_only_radius_two(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure

        indexed = {report._key(row): row for row in self.rows}
        original_subplots, figures = plt.subplots, []
        def capture(*args, **kwargs):
            fig, axes = original_subplots(*args, **kwargs)
            figures.append((fig, axes))
            return fig, axes
        for r in (3, 10):
            data = report._plot_data(self.bases[r], self.manifest['base_plot_data'][str(r)],
                                     self.manifest, r, indexed)
            with patch.object(plt, 'subplots', side_effect=capture), patch.object(Figure, 'savefig'):
                report._render(data, r, self.output)
            for ax, panel in zip(figures[-1][1], data['panels']):
                self.assertEqual(ax.get_xscale(), 'symlog')
                self.assertEqual(ax.get_xlim(), (0., 2. if r == 10 else 1.5))
                self.assertEqual(2. in ax.get_xticks(), r == 10)
                lines = {line.get_label(): line for line in ax.lines}
                for method in ('green', 'black', 'scs', 'sro_fa'):
                    curve = panel[method]
                    line = lines[curve['label'] + ' OOS']
                    np.testing.assert_array_equal(line.get_xdata(), curve['radius'])
                    np.testing.assert_array_equal(line.get_ydata(), curve['oos']['mean'])
                for reference in panel['horizontal']:
                    self.assertEqual(list(lines[reference['label']].get_ydata()), [reference['value']]*2)

    def test_duplicate_seed(self):
        self.rows[-1] = deepcopy(self.rows[-2])
        self.rejected('duplicate')

    def test_manifest_missing_seed_even_with_matching_rows(self):
        self.manifest['jobs'][-1] = deepcopy(self.manifest['jobs'][-2])
        self.rows[-1] = deepcopy(self.rows[-2])
        self.rejected('Incomplete/duplicate')

    def test_row_changed_stop_or_checkpoint_or_method(self):
        for key, value in [('selected_candidate_id', 0), ('stopping_candidate_id', 0),
                           ('checkpoint_sha256', digest('wrong')), ('state_sha256', digest('wrong')),
                           ('pool_sha256', digest('wrong')), ('method', 'black'), ('converged', 1)]:
            with self.subTest(field=key):
                self.rows = deepcopy(self.original_rows)
                self.rows[0][key] = value
                self.rejected('identity')

    def test_manifest_min_ap_not_stop(self):
        self.manifest['jobs'][0]['selected_candidate_id'] = 0
        self.rejected('selected stop')

    def test_cap_deleted_or_mislabeled(self):
        job = next(j for j in self.manifest['jobs'] if not j['converged'])
        job.update(converged=True, terminal_status='converged')
        self.rejected('11 BLACK caps')

    def test_cap_not_at_40(self):
        job = next(j for j in self.manifest['jobs'] if not j['converged'])
        job.update(selected_candidate_id=39, stopping_candidate_id=39)
        self.rejected('convergence/cap')

    def test_cap_row_convergence_changed(self):
        row = next(r for r in self.rows if not r['converged'])
        row.update(converged=True, terminal_status='converged')
        self.rejected('identity')

    def test_unverified_cached_or_failed_result(self):
        for field, value in [('journal_verified', False), ('cached', True), ('status', 'failed')]:
            with self.subTest(field=field):
                self.rows = deepcopy(self.original_rows)
                self.rows[0][field] = value
                self.rejected('journal-verified')

    def test_result_hash_tamper(self):
        self.rows[0]['result_sha256'] = digest('tampered')
        self.rejected('Hash mismatch')

    def test_result_value_tamper(self):
        self.rows[0]['mean_oos'] += 1.
        self.rejected('Raw result/row mismatch')

    def test_test_arrays_changed(self):
        self.rows[0]['test']['arrays']['xi1']['array_sha256'] = digest('different array')
        self.rejected('test arrays differ')

    def test_pool_sizes_or_mean_changed(self):
        self.rows[0]['sizes'][0] += 1
        self.rejected('pool sizes')
        self.rows = deepcopy(self.original_rows)
        self.rows[0]['finite_pool_mean'] += 1.
        self.rejected('pool mean')

    def test_base_hash_tamper(self):
        self.manifest['base_plot_data']['3']['sha256'] = digest('tampered')
        self.rejected('Hash mismatch')

    def test_base_test_identity_changed_even_if_rehashed(self):
        base = deepcopy(self.bases[3])
        base['test']['arrays']['xi1']['array_sha256'] = digest('changed')
        path = Path(self.inputs.name) / 'altered_base.json'
        self.manifest['base_plot_data']['3'] = dict(path=str(path), sha256=write(path, base))
        self.rejected('Base test arrays differ')

    def test_nominal_not_silently_deduplicated_if_different(self):
        self.manifest['jobs'][1]['historical_training']['nominal']['training_timing']['wall_time_s'] += 1.
        self.rejected('Nominal selected-arm timing differs')

    def test_bad_timing_or_nonfinite_cost(self):
        for key, value in [('wall_time_s', -1.), ('mean_oos', float('nan'))]:
            with self.subTest(field=key):
                self.rows = deepcopy(self.original_rows)
                self.rows[0][key] = value
                self.rejected('Invalid')

    def test_nonempty_staging_is_not_overwritten(self):
        path = self.output / 'parent-owned.txt'
        path.write_text('keep')
        with self.assertRaisesRegex(ValueError, 'empty staging'):
            report.publish_data(self.manifest, self.rows, self.output)
        self.assertEqual(path.read_text(), 'keep')

    def test_missing_staging_not_created(self):
        target = self.output / 'missing'
        with self.assertRaisesRegex(ValueError, 'empty staging'):
            report.publish_data(self.manifest, self.rows, target)
        self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()