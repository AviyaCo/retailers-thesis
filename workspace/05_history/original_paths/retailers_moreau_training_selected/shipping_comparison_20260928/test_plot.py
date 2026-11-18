"""Synthetic presentation fixtures only; never model, solver or real OOS calls."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import plot_comparison as p


def fixture(root):
    """Artificial observations, with the real baseline JSON field conventions."""
    m = dict(status='frozen', n_test=10000, test_seed=9999,
             test=dict(n_test=10000, seed=9999, arrays={'fixture': 'not real test data'}),
             files={}, jobs=[], baseline_paths={})

    def freeze(name, obj):
        original = '/nonexistent/original/' + name
        dest = root / name
        dest.write_text(json.dumps(obj, allow_nan=False))
        m['files'][original] = dict(copy=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        return original

    rows = []
    for n in p.SIZES:
        for method, grid in (('green', p.GREEN_RADII), ('legacy', [0., .005, .15, 1.5])):
            for rho in grid:
                for seed in p.SEEDS:
                    identity = f'{method}-{n}-{rho}-{seed}'
                    cp = '/nonexistent/checkpoints/' + identity
                    # Plotting binds identity but never opens/deserializes models.
                    digest = hashlib.sha256(identity.encode()).hexdigest()
                    m['files'][cp] = dict(copy=str(root / identity), sha256=digest)
                    train = 4*n//5 if method == 'green' else n
                    spec = dict(id=identity, method=method, Ntotal=n, Ntrain=train, Nval=n-train,
                        seed=seed, radius=rho, checkpoint=cp, checkpoint_sha256=digest,
                        state_sha256=digest, sizes=[1]*train, pools=[[[0.]*6]]*train,
                        training_runtime_reference=dict(total_time_s=12., status='persisted_not_current'))
                    m['jobs'].append(spec)
                    count = 10000+train
                    rows.append(dict(deepcopy(spec), status='completed', mean_oos=n/100+seed+rho,
                        finite_pool_mean=n/100+seed+rho+2, test=deepcopy(m['test']),
                        solves_attempted=count, solves_accepted=count, status_counts={'2': count},
                        wall_time_s=1.5, phases={'oos': {'wall_time_s': 1.}, 'is': {'wall_time_s': .5}},
                        worker_count=2, solver_threads=1))
        knn = dict(oos=[float(n+s) for s in p.SEEDS], selected_k=None,
                   selected_k_by_seed=[1, 3]*5, selection_metric='per_seed_loo_validation_cost')
        cache = dict(meta=dict(N_train=n, N_test=10000, seed_test=9999, seeds=p.SEEDS.copy(),
                              shipping_evaluation='hard charge 0.1 when Q2 > 1e-6'),
            runtime=dict(status='completed', current_invocation_elapsed_s=123.),
            benchmarks=dict(saa_knn=knn, _knn_k_selection='shipping_per_seed_loo_grid_v1',
                            knn_k_grid=[1, 3]),
            sweep={'not_used': {'rob_oos': 'must not plot historical neural cache'}})
        if n == 800:
            cache['meta'].update(experiment='fixed_shipping_N800_corrected_minibatch',
                                 sweep_source='N800/rho_sweep_minibatch')
        sro = {str(r): {str(s): dict(oos=float(n+s+r), time_partition=.1, time_lp=3.,
                   time_eval=.2, time_total=3.5) for s in p.SEEDS} for r in (0., .07, 1.5, 2.)}
        sro['0.07']['0'].update(solver_status=9, sol_count=1, mip_gap=.02, obj_bound=10.)
        sro['metadata'] = dict(solver='Gurobi', note='Synthetic mixed-key metadata')
        conditional = dict(N=5000, M=200, seed=42, scenarios=1000000, relaxed=False,
            exact_mip=True, saa_cond_mip_1m_shipping_exact=19.5, status='OPTIMAL',
            proven_optimal=True, best_bound=19.4, mip_gap=.004, sol_count=3,
            solve_time_s=1762., source_file='/historical/shared/reference',
            best_nominal_neural_seed=2, best_nominal_neural_cost=20.)
        m['baseline_paths'][str(n)] = dict(cache=freeze(f'cache{n}.json', cache),
            sro=freeze(f'sro{n}.json', sro), conditional=freeze(f'conditional{n}.json', conditional))
    return m, rows


def rewrite(m, source, change):
    """Reseal synthetic fixtures, never real frozen artifacts."""
    info = m['files'][source]
    path = Path(info['copy'])
    obj = json.loads(path.read_text())
    change(obj)
    path.write_text(json.dumps(obj))
    info['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m, self.rows = fixture(self.root)

    def build(self):
        return p.build_plot_data(self.m, self.rows)

    def test_complete_grid_alignment_percentiles_and_no_mutation(self):
        original = deepcopy((self.m, self.rows))
        data = self.build()
        self.assertEqual((self.m, self.rows), original)
        self.assertEqual(data, p.build_plot_data(self.m, list(reversed(self.rows))))
        self.assertEqual(sum(j['method'] == 'green' for j in self.m['jobs']), 270)
        for panel in data['panels']:
            n = panel['Ntotal']
            self.assertEqual(panel['training_counts'], dict(green=4*n//5, legacy=n))
            self.assertEqual(panel['green']['radius'], p.GREEN_RADII)
            self.assertEqual(panel['legacy']['radius'], [0., .005, .15, 1.5])
            self.assertEqual(panel['sro_fa']['radius'], [0., .07, 1.5])
            for method in ('green', 'legacy'):
                values = np.asarray(panel[method]['oos']['per_seed'])
                np.testing.assert_array_equal(values[0], np.arange(10)+n/100)
                np.testing.assert_array_equal(panel[method]['oos']['p20'], np.percentile(values, 20, axis=1))
                np.testing.assert_array_equal(panel[method]['oos']['p80'], np.percentile(values, 80, axis=1))
        self.assertIn('not confidence intervals', data['band_definition'])
        self.assertNotIn('validation_selected_oos', data)
        json.dumps(data, allow_nan=False)
        p._validate_plot_data(data)

    def test_missing_duplicate_failed_nonfinite_and_wrong_identity(self):
        for change in (lambda r: r.pop(), lambda r: r.append(deepcopy(r[0])),
                       lambda r: r[0].update(status='failed'),
                       lambda r: r[0].update(status='running'),
                       lambda r: r[0].update(mean_oos=float('nan')),
                       lambda r: r[0].update(finite_pool_mean=float('inf')),
                       lambda r: r[0].update(seed=10),
                       lambda r: r[0].update(Ntrain=999),
                       lambda r: r[0].update(checkpoint_sha256='wrong'),
                       lambda r: r[0].update(state_sha256='wrong'),
                       lambda r: r[0].update(checkpoint='/wrong'),
                       lambda r: r[0].update(id=r[1]['id'])):
            rows = deepcopy(self.rows)
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.build_plot_data(self.m, rows)

    def test_grid_holes_and_optional_legacy_radii(self):
        m = deepcopy(self.m)
        m['legacy_radii'] = {str(n): [0., .005, .15, 1.5] for n in p.SIZES}
        self.assertEqual(p.build_plot_data(m, self.rows), self.build())
        m['legacy_radii']['50'].pop()
        with self.assertRaises(ValueError):
            p.build_plot_data(m, self.rows)
        for method in ('green', 'legacy'):
            m, rows = deepcopy(self.m), deepcopy(self.rows)
            key = next(i for i, r in enumerate(rows) if r['method'] == method)
            rows.pop(key)
            m['jobs'].pop(key)
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'fixed policy grid'):
                p.build_plot_data(m, rows)
        m = deepcopy(self.m)
        m['jobs'].append(deepcopy(m['jobs'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            p.build_plot_data(m, self.rows)

    def test_test_protocol_and_checkpoint_binding(self):
        for change in (lambda m: m.update(n_test=9999),
                       lambda m: m['test'].update(seed=7),
                       lambda m: m.update(test_seed=8),
                       lambda m: m.update(status='running'),
                       lambda m: m['files'][m['jobs'][0]['checkpoint']].update(sha256='wrong')):
            m = deepcopy(self.m)
            change(m)
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.build_plot_data(m, self.rows)
        for change in (lambda r: r['test'].update(seed=3),
                       lambda r: r['test']['arrays'].update(fixture='wrong'),
                       lambda r: r.update(n_test=2), lambda r: r.update(test_seed=8)):
            rows = deepcopy(self.rows)
            change(rows[0])
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'test'):
                p.build_plot_data(self.m, rows)

    def test_strict_counts_cannot_hide_missing_or_nonoptimal_solves(self):
        for change in (lambda r: r.update(solves_accepted=10000),
                       lambda r: r.update(solves_attempted=10000),
                       lambda r: r.update(status_counts={'2': 10039, '9': 1}),
                       lambda r: r.update(status_counts={'2': 10040, '9': 0}),
                       lambda r: r.update(solver_status_counts={'9': 10040}),
                       lambda r: r.pop('status_counts')):
            rows = deepcopy(self.rows)
            change(rows[0])
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.build_plot_data(self.m, rows)
        for r in self.rows:
            r['solver_status_counts'] = r.pop('status_counts')
        self.build()

    def test_fee_solver_and_timing_provenance_preserved(self):
        data = self.build()
        for panel in data['panels']:
            for method in ('green', 'legacy'):
                provenance = panel[method]['provenance']
                self.assertFalse(provenance['cached'])
                self.assertEqual(provenance['fee_rule'], p.NEURAL_FEE)
                row = provenance['per_seed'][0][0]
                self.assertEqual(row['wall_time_s'], 1.5)
                self.assertEqual(row['phases'], self.rows[0]['phases'])
                self.assertEqual(row['training_runtime_reference']['total_time_s'], 12.)
                self.assertNotIn('pools', row)
            self.assertEqual(panel['legacy']['provenance']['training_solver'], 'Clarabel')
            for method, threshold in (('saa_knn', 1e-6), ('sro_fa', 1e-10)):
                provenance = panel[method]['provenance']
                self.assertEqual(provenance['fee_rule']['threshold'], threshold)
                self.assertFalse(provenance['strict_certification'])
                self.assertIsNone(provenance['current_evaluation_time_s'])
            sro = panel['sro_fa']['provenance']['per_seed'][1][0]
            self.assertEqual((sro['solver_status'], sro['sol_count'], sro['mip_gap']), (9, 1, .02))
            self.assertEqual(sro['time_lp'], 3.)
            ref = panel['horizontal'][0]
            self.assertEqual(ref['metadata']['mip_gap'], .004)
            self.assertIn('not an exact population bound', ref['interpretation'])
        self.rows[0]['fee_rule'] = dict(p.NEURAL_FEE, threshold=1e-6)
        with self.assertRaisesRegex(ValueError, 'fee provenance'):
            self.build()

    def test_historical_cache_failures(self):
        changes = [lambda d: d['meta']['seeds'].reverse(),
                   lambda d: d['meta'].update(seed_test=0),
                   lambda d: d['meta'].update(shipping_evaluation='q2 > 0'),
                   lambda d: d['benchmarks']['saa_knn']['selected_k_by_seed'].pop(),
                   lambda d: d['benchmarks']['saa_knn'].update(selection_metric='test_best'),
                   lambda d: d['benchmarks']['saa_knn']['selected_k_by_seed'].__setitem__(0, 999),
                   lambda d: d['benchmarks']['saa_knn']['oos'].pop(),
                   lambda d: d['benchmarks']['saa_knn']['oos'].__setitem__(0, float('nan'))]
        for change in changes:
            self.m, self.rows = fixture(self.root)
            rewrite(self.m, self.m['baseline_paths']['50']['cache'], change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build()

    def test_wrong_n800_cache_and_tampered_fee_rejected(self):
        rewrite(self.m, self.m['baseline_paths']['800']['cache'],
                lambda d: d['meta'].update(sweep_source='N800/rho_sweep'))
        with self.assertRaisesRegex(ValueError, 'corrected minibatch'):
            self.build()
        self.m, self.rows = fixture(self.root)
        data = self.build()
        data['panels'][0]['saa_knn']['provenance']['fee_rule']['threshold'] = 0.
        with self.assertRaisesRegex(ValueError, 'fee/certification'):
            p._validate_plot_data(data)

    def test_sro_mixed_keys_missing_seeds_duplicates_and_failed_rows(self):
        for change in (lambda d: d['0.0'].pop('9'),
                       lambda d: d['0.0']['0'].update(status='failed'),
                       lambda d: d['0.0']['0'].update(oos=float('inf')),
                       lambda d: d['0.0']['0'].update(solver_status=3),
                       lambda d: d['0.07']['0'].update(sol_count=0),
                       lambda d: d.update({'0': deepcopy(d['0.0'])}),
                       lambda d: d.update({'unknown': {'oos': 1}})):
            self.m, self.rows = fixture(self.root)
            rewrite(self.m, self.m['baseline_paths']['50']['sro'], change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build()

    def test_frozen_only_hash_and_conditional_validation(self):
        for change in (lambda d: d.update(scenarios=1000),
                       lambda d: d.update(saa_cond_mip_1m_shipping_exact=float('nan')),
                       lambda d: d.update(status='TIME_LIMIT')):
            self.m, self.rows = fixture(self.root)
            rewrite(self.m, self.m['baseline_paths']['50']['conditional'], change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build()
        self.m, self.rows = fixture(self.root)
        source = self.m['baseline_paths']['50']['cache']
        Path(self.m['files'][source]['copy']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()
        del self.m['files'][source]
        with self.assertRaisesRegex(ValueError, 'Unbound'):
            self.build()

    def test_render_rejects_tampered_aggregates_before_writing(self):
        data = self.build()
        data['panels'][0]['green']['oos']['mean'][0] += 1
        output = self.root / 'must_not_exist'
        with self.assertRaisesRegex(ValueError, 'Aggregates'):
            p.render(data, output)
        self.assertFalse(output.exists())

    def test_render_styles_outputs_and_visible_caption(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure

        captured = []
        original = Figure.savefig

        def save(fig, *args, **kwargs):
            captured.append(fig)
            return original(fig, *args, **kwargs)

        output = self.root / 'figures'
        with patch.object(Figure, 'savefig', save):
            self.assertIsNone(p.render(self.build(), output))
        self.assertEqual({f.name for f in output.iterdir()},
                         {'comparison_shipping.png', 'comparison_shipping.pdf'})
        self.assertTrue((output / 'comparison_shipping.png').read_bytes().startswith(b'\x89PNG'))
        self.assertTrue((output / 'comparison_shipping.pdf').read_bytes().startswith(b'%PDF'))
        fig = captured[0]
        self.assertEqual(len(fig.axes), 3)
        for ax in fig.axes:
            self.assertEqual(ax.get_xscale(), 'symlog')
            self.assertEqual(ax.get_xlim(), (0., 1.5))
            self.assertIn(0., ax.get_xticks())
            self.assertEqual(len(ax.lines), 7)
            self.assertEqual([line.get_color() for line in ax.lines],
                [p.COLORS[k] for k in ('green', 'green_is', 'legacy', 'legacy_is', 'sro_fa', 'saa_knn')]
                + ['#888888'])
            self.assertFalse(ax.texts)  # No test-best annotations.
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        self.assertEqual(len(labels), 7)
        self.assertTrue(any('Clarabel' in s for s in labels))
        self.assertFalse(any('SCS' in s or 'black' in s or 'best' in s for s in labels))
        fig.canvas.draw()
        bounds = fig.bbox
        for text in fig.texts:
            box = text.get_window_extent(fig.canvas.get_renderer())
            self.assertGreaterEqual(box.x0, bounds.x0)
            self.assertLessEqual(box.x1, bounds.x1)
            self.assertGreaterEqual(box.y0, bounds.y0)
            self.assertLessEqual(box.y1, bounds.y1)

    def test_presentation_module_has_no_evaluator_or_solver_imports(self):
        tree = ast.parse(Path(p.__file__).read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module.split('.')[0])
        self.assertLessEqual(set(imports), {'__future__', 'copy', 'hashlib', 'json', 'pathlib',
                                          'numpy', 'matplotlib'})


if __name__ == '__main__':
    unittest.main()