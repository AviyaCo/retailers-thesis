"""Solver-free synthetic fixtures, NEVER research observations or model runs."""
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
    """Deliberately artificial costs and grids exercising the frozen schema."""
    arrays = {name: dict(array_sha256=name+'-fixture', dtype='float64', shape=[10000, 3])
              for name in ('xi1', 'xi2')}
    m = dict(status='frozen', n_test=10000, test_seed=9999, files={}, jobs=[],
             test=dict(n_test=10000, seed=9999, arrays=arrays),
             cached_green_N50=dict(jobs=[], test_arrays=deepcopy(arrays)),
             cached_baselines={}, scs_radii={})

    def freeze(name, obj):
        original = '/nonexistent/original/' + name
        dest = root / name
        dest.write_text(json.dumps(obj, allow_nan=False))
        m['files'][original] = dict(copy=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        return original

    rows = []
    for n in p.SIZES:
        grid = [0.] + np.linspace(.001, 1.5, p.SCS_COUNTS[n]-1).tolist()
        m['scs_radii'][str(n)] = grid
        for method, radii in [('green', p.GREEN_RADII), ('scs', grid)]:
            for rho in radii:
                for seed in p.SEEDS:
                    cost = n/100 + seed + rho
                    if method == 'green' and n == 50:
                        m['cached_green_N50']['jobs'].append(dict(radius=rho, seed=seed,
                            status='completed', test_mean=cost, finite_pool_mean=cost+2))
                    else:
                        spec = dict(id=f'{method}-{n}-{rho}-{seed}', Ntotal=n, method=method,
                                    radius=rho, seed=seed, Ntrain=4*n//5 if method == 'green' else n)
                        m['jobs'].append(spec)
                        rows.append(dict(spec, status='completed', mean_oos=cost, finite_pool_mean=cost+2))
        knn = dict(oos=[float(s+n) for s in p.SEEDS])
        sro = {str(rho): {str(s): dict(oos=s+n+rho, **{'is': s+n}, K=3, time_total=.5)
                         for s in p.SEEDS} for rho in (.001, .07, 1.5, 2.)}
        cache = dict(meta=dict(N_train=n, N_test=10000, seed_test=9999, seeds=p.SEEDS.copy()),
                     sweep={str(r): {} for r in grid},
                     benchmarks=dict(saa_knn=knn, saa_lb=dict(oos=[float(n+s+4) for s in p.SEEDS])))
        m['cached_baselines'][str(n)] = dict(saa_knn=knn, sro_fa=sro,
            cache=freeze(f'cache{n}.json', cache), sro_source=freeze(f'sro{n}.json', sro))
    old = dict(test_arrays=deepcopy(arrays), horizontal=[
        ['SAA-kNN (historical)', -100., '#000', '-'],
        ['SAA-LP LB (cached 2k, not matched 10k)', -2., '#000', '--'],
        ['Conditional LB (separate 1M; seed 42)', -3., '#000', '-.']])
    m['cached_green_N50']['plot_data'] = freeze('old_plot.json', old)
    return m, rows


def rewrite(m, source, change):
    """Reseal a deliberately changed fixture, not a real frozen artifact."""
    info = m['files'][source]
    path = Path(info['copy'])
    d = json.loads(path.read_text())
    change(d)
    path.write_text(json.dumps(d))
    info['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m, self.rows = fixture(self.root)

    def build(self):
        return p.build_plot_data(self.m, self.rows)

    def test_complete_shapes_seed_alignment_percentiles_and_no_mutation(self):
        original = deepcopy((self.m, self.rows))
        data = self.build()
        self.assertEqual((self.m, self.rows), original)
        self.assertEqual(data, p.build_plot_data(self.m, list(reversed(self.rows))))
        self.assertEqual(len(data['panels']), 3)
        for panel in data['panels']:
            n = panel['Ntotal']
            self.assertEqual(panel['training_counts'], dict(green=4*n//5, scs=n))
            self.assertEqual(panel['green']['radius'], p.GREEN_RADII)
            self.assertEqual(len(panel['scs']['radius']), p.SCS_COUNTS[n])
            self.assertEqual(panel['sro_fa']['radius'], [.001, .07, 1.5])
            values = np.asarray(panel['green']['oos']['per_seed'])
            np.testing.assert_array_equal(values[0], np.arange(10)+n/100)
            np.testing.assert_array_equal(panel['green']['oos']['p20'], np.percentile(values, 20, axis=1))
            np.testing.assert_array_equal(panel['green']['oos']['p80'], np.percentile(values, 80, axis=1))
            self.assertEqual(panel['sro_fa']['provenance']['strict_test_status'], 'not_available')
        self.assertIn('not confidence intervals', data['band_definition'])
        self.assertIn('not box-certified', data['caption'])
        self.assertEqual(data['validation_selected_oos'], [])
        json.dumps(data, allow_nan=False)

    def test_missing_duplicate_failed_and_nonfinite_rows_rejected(self):
        for change in (lambda r: r.pop(), lambda r: r.append(deepcopy(r[0])),
                       lambda r: r[0].update(status='failed'),
                       lambda r: r[0].update(status='running'),
                       lambda r: r[0].update(mean_oos=float('nan')),
                       lambda r: r[0].update(finite_pool_mean=float('inf')),
                       lambda r: r[0].update(seed=10),
                       lambda r: r[0].update(Ntrain=999)):
            rows = deepcopy(self.rows)
            change(rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                p.build_plot_data(self.m, rows)

    def test_wrong_or_incomplete_scs_grid_rejected_even_if_rows_match_manifest(self):
        m = deepcopy(self.m)
        m['scs_radii']['50'].pop()
        with self.assertRaisesRegex(ValueError, 'Grid count'):
            p.build_plot_data(m, self.rows)
        old = self.m['scs_radii']['50'][1]
        self.m['scs_radii']['50'][1] = .0005
        for row in self.m['jobs'] + self.rows:
            if row['method'] == 'scs' and row['Ntotal'] == 50 and row['radius'] == old:
                row['radius'] = .0005
        with self.assertRaisesRegex(ValueError, 'frozen cache'):
            self.build()

    def test_missing_cached_green_and_wrong_test_identity_rejected(self):
        self.m['cached_green_N50']['jobs'].pop()
        with self.assertRaisesRegex(ValueError, 'cached green'):
            self.build()
        self.m, self.rows = fixture(self.root)
        self.m['cached_green_N50']['test_arrays']['xi1']['array_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'array identity'):
            self.build()
        self.m, self.rows = fixture(self.root)
        self.m['test']['seed'] = 1
        with self.assertRaisesRegex(ValueError, 'Test protocol'):
            self.build()

    def test_historical_missing_nonfinite_and_seed_order_rejected(self):
        for change in (lambda b: b['saa_knn']['oos'].pop(),
                       lambda b: b['saa_knn']['oos'].__setitem__(0, float('nan')),
                       lambda b: b['sro_fa']['0.001'].pop('9'),
                       lambda b: b['sro_fa']['0.001']['0'].update(oos=float('inf'))):
            m = deepcopy(self.m)
            change(m['cached_baselines']['50'])
            with self.assertRaises(ValueError):
                p.build_plot_data(m, self.rows)
        src = self.m['cached_baselines']['50']['cache']
        rewrite(self.m, src, lambda d: d['meta']['seeds'].reverse())
        with self.assertRaisesRegex(ValueError, 'seed order'):
            self.build()

    def test_sro_incomplete_or_failed_source_is_not_a_certified_success(self):
        for change in (lambda d: d['0.001'].pop('9'),
                       lambda d: d['0.001']['0'].update(status='failed'),
                       lambda d: d['0.001']['0'].update(oos=float('nan'))):
            self.m, self.rows = fixture(self.root)
            baseline = self.m['cached_baselines']['50']
            rewrite(self.m, baseline['sro_source'], change)
            baseline['sro_fa'] = json.loads(Path(self.m['files'][baseline['sro_source']]['copy']).read_text())
            with self.assertRaises(ValueError):
                self.build()

    def test_unbound_plot_and_worker_projection_failure_rejected(self):
        m = deepcopy(self.m)
        del m['files'][m['cached_green_N50']['plot_data']]
        with self.assertRaisesRegex(ValueError, 'Unbound'):
            p.build_plot_data(m, self.rows)
        self.rows[0].update(solves_attempted=10001, solves_accepted=10000,
                            status_counts={'2': 10000, '9': 1}, sizes=[1])
        with self.assertRaisesRegex(ValueError, 'strict solves'):
            self.build()

    def test_frozen_only_references_n_specific_metadata_and_tamper_guard(self):
        data = self.build()
        for panel in data['panels']:
            refs = panel['horizontal']
            self.assertEqual([r['kind'] for r in refs], ['saa_lp', 'conditional'])
            self.assertEqual(refs[0]['value'], panel['Ntotal']+8.5)
            self.assertIn('10,000', refs[0]['source_label'])
            self.assertNotIn('2k', refs[0]['source_label'])
            self.assertEqual(refs[1]['value'], -3.)
            self.assertIn('separate 1M', refs[1]['source_label'])
        source = self.m['cached_green_N50']['plot_data']
        Path(self.m['files'][source]['copy']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()

    def test_optional_references_not_invented_and_shared_origin_disclosed(self):
        for baseline in self.m['cached_baselines'].values():
            rewrite(self.m, baseline['cache'], lambda d: d['benchmarks'].pop('saa_lb'))
        data = self.build()
        self.assertTrue(all(panel['horizontal'][0]['value'] == -2. for panel in data['panels']))
        self.assertIn('not an N-specific', data['panels'][1]['horizontal'][0]['interpretation'])
        self.m['cached_green_N50'].pop('plot_data')
        self.assertTrue(all(not panel['horizontal'] for panel in self.build()['panels']))

    def test_saved_validation_choices_preserved_without_test_argmin(self):
        winners = [dict(Ntotal=n, seed=s, radius=1.5, criterion='fixture validation')
                   for n in (200, 800) for s in p.SEEDS]
        self.m['saved_green_selections'] = winners
        result = self.build()['validation_selected_oos']
        self.assertEqual([r['selection'] for r in result], winners)
        self.assertTrue(all(r['mean_oos'] == r['selection']['Ntotal']/100+r['selection']['seed']+1.5
                            for r in result))
        self.m['saved_green_selections'].pop()
        with self.assertRaisesRegex(ValueError, 'validation selections'):
            self.build()

    def test_render_styles_six_curves_references_ticks_and_outputs(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from PIL import Image
        captured = []
        original = Figure.savefig

        def save(fig, path, **kwargs):
            captured.append(fig)
            return original(fig, path, **kwargs)

        out = self.root / 'figures'
        with patch.object(Figure, 'savefig', save):
            p.render(self.build(), out)
        self.assertEqual(sorted(f.name for f in out.iterdir()), ['comparison_R3.pdf', 'comparison_R3.png'])
        with Image.open(out / 'comparison_R3.png') as image:
            self.assertAlmostEqual(image.info['dpi'][0], 300, places=1)
        pdf = (out / 'comparison_R3.pdf').read_bytes()
        self.assertTrue(pdf.startswith(b'%PDF'))
        self.assertNotIn(b'/Subtype /Image', pdf)
        fig = captured[0]
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual(len(fig.legends[0].get_texts()), 8)
        for ax in fig.axes:
            self.assertEqual(ax.get_xscale(), 'symlog')
            np.testing.assert_array_equal(ax.get_xticks(), p.TICKS)
            self.assertEqual(len(ax.lines), 8)
            self.assertEqual([line.get_linestyle() for line in ax.lines[:6]], ['-', '--', '-', '--', '-', '-'])
            self.assertEqual([line.get_color() for line in ax.lines[:6]],
                             [p.COLORS[k] for k in ('green', 'green_is', 'scs', 'scs_is', 'sro_fa', 'saa_knn')])

    def test_render_rejects_tampered_or_partial_data_before_writing(self):
        for change in (lambda d: d['panels'].pop(),
                       lambda d: d['panels'][0]['green']['seeds'].reverse(),
                       lambda d: d['panels'][0]['scs']['radius'].pop(),
                       lambda d: d['panels'][0]['green']['oos']['mean'].__setitem__(0, float('nan')),
                       lambda d: d['panels'][0]['green']['oos']['per_seed'][0].reverse()):
            data = self.build()
            change(data)
            out = self.root / 'must_not_exist'
            with self.assertRaises(ValueError):
                p.render(data, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()