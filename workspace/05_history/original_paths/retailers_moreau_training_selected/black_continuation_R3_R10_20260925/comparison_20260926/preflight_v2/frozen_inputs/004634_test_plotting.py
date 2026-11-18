"""Display-contract tests only; artificial black fixtures are NOT observations.

Reads both actual frozen base schemas. No policy evaluation, solver, training,
test-data generation, or policy selection. Rendering uses temporary directories.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import plotting as p


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(bases):
    """570 explicitly artificial rows, with 11 caps and no missing seeds."""
    bindings = dict(input_sha256={'/artificial/input': digest('input')},
        source_sha256={}, renderers={}, base_sha256={}, jobs={}, results={}, test={},
        selection_rule=p.SELECTION_RULE)
    rows = {3: [], 10: []}
    for R in (3, 10):
        source = p.RENDERER_PATHS[R]
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        bindings['renderers'][str(R)] = dict(path=str(source), sha256=sha)
        bindings['source_sha256'][str(source)] = sha
        bindings['base_sha256'][str(R)] = p.canonical_sha256(bases[R])
        bindings['test'][str(R)] = deepcopy(bases[R]['test'])
        for n in p.SIZES:
            for rho in p.RADII[R]:
                for seed in p.SEEDS:
                    identifier = f'ARTIFICIAL-R{R}-N{n}-rho{rho}-seed{seed}'
                    capped = ((R, n, rho, seed) == (3, 50, .7, 9)
                              or (R == 10 and n == 200 and rho == 2.))
                    job = dict(id=identifier, R=R, Ntotal=n, Ntrain=4*n//5, seed=seed,
                        radius=rho, checkpoint_sha256=digest(identifier+'checkpoint'),
                        state_sha256=digest(identifier+'state'), pool_sha256=digest(identifier+'pool'),
                        pool_file_sha256=digest(identifier+'pool_file'), converged=not capped,
                        terminal_status='iteration_cap' if capped else 'separation_satisfied',
                        selected_candidate_id=4, stopping_candidate_id=40 if capped else 7)
                    bindings['jobs'][identifier] = job
                    ref = dict(status='completed', journal_verified=True,
                        result_sha256=digest(identifier+'raw_result'),
                        raw_proofs={'/artificial/'+identifier+'/proof.json': digest(identifier+'proof')})
                    bindings['results'][identifier] = ref
                    cost = float(seed + rho + n)
                    maxima = [cost+2]*job['Ntrain']
                    expected = 10000 + 2*job['Ntrain']
                    rows[R].append(dict(deepcopy(job), status='completed', cached=False,
                        mean_oos=cost, finite_pool_mean=float(np.mean(maxima)),
                        per_center_maxima=maxima, sizes=[2]*job['Ntrain'],
                        result_sha256=ref['result_sha256'], test=deepcopy(bases[R]['test']),
                        solves_attempted=expected, solves_accepted=expected,
                        solves_physical_accepted=expected, status_counts={'2': expected},
                        wall_time_s=.125, phases={'oos_s': .1, 'finite_pool_s': .025},
                        source_prefix_timings={'status': 'unavailable; rerun required'},
                        fixture_only='ARTIFICIAL; never measured'))
    return bindings, rows


class OverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Actual large schemas are deliberately read, not approximated or skipped.
        cls.bases = {R: json.loads(path.read_bytes()) for R, path in p.BASE_PATHS.items()}
        cls.bindings, cls.rows = fixture(cls.bases)
        cls.overlays = {R: p.build_overlay(cls.bases[R], list(reversed(cls.rows[R])), R,
                                         cls.bindings) for R in (3, 10)}

    def test_actual_bases_preserved_and_grids_not_extended(self):
        for R, overlay in self.overlays.items():
            base = self.bases[R]
            self.assertEqual(p.canonical_sha256(base), self.bindings['base_sha256'][str(R)])
            for key, value in base.items():
                if key not in ('panels', 'caption', 'footnote'):
                    self.assertEqual(overlay[key], value)
            self.assertEqual(overlay['caption'], base['caption'] + p.CAPTION_SUFFIX)
            self.assertEqual(overlay['footnote'], base['footnote'] + p.FOOTNOTE_SUFFIX)
            for before, after in zip(base['panels'], overlay['panels']):
                self.assertEqual({k: v for k, v in after.items() if k != 'black'}, before)
                self.assertEqual(after['black']['radius'], p.RADII[R])
                self.assertEqual(after['black']['seeds'], list(range(10)))
                for method in ('green', 'scs', 'sro_fa'):
                    self.assertLessEqual(max(after[method]['radius']), 1.5)
            self.assertIsNot(overlay['provenance'], base['provenance'])

    def test_statistics_and_all_nonconverged_seeds_retained(self):
        for R, overlay in self.overlays.items():
            meta = overlay['black_overlay']
            self.assertEqual(len(meta['black_nonconverged']), 11)
            self.assertEqual(len(meta['cohort_nonconverged']), 1 if R == 3 else 10)
            self.assertEqual(meta['cohort_policy_count'], 270 if R == 3 else 300)
            for panel in overlay['panels']:
                for field in ('oos', 'finite_pool'):
                    stats = panel['black'][field]
                    a = np.asarray(stats['per_seed'])
                    self.assertEqual(a.shape, (len(p.RADII[R]), 10))
                    self.assertEqual(stats['mean'], a.mean(axis=-1).tolist())
                    self.assertEqual(stats['p20'], np.percentile(a, 20, axis=-1).tolist())
                    self.assertEqual(stats['p80'], np.percentile(a, 80, axis=-1).tolist())
        capped = self.overlays[10]['panels'][1]['black']['provenance']['per_seed'][-1]
        self.assertEqual([r['seed'] for r in capped], list(range(10)))
        self.assertTrue(all(not r['converged'] and r['status'] == 'completed' for r in capped))

    def test_row_rejections(self):
        bad_fields = {
            'mean_oos': float('nan'), 'finite_pool_mean': float('inf'),
            'sizes': [2], 'per_center_maxima': [1.], 'solves_attempted': 0,
            'solves_accepted': 0, 'solves_physical_accepted': 0,
            'status_counts': {'9': 10080}, 'result_sha256': 'a'*64,
            'checkpoint_sha256': 'b'*64, 'pool_sha256': 'c'*64,
            'pool_file_sha256': 'd'*64, 'state_sha256': 'e'*64,
            'status': 'failed', 'cached': True, 'wall_time_s': -1.,
            'phases': {}, 'source_prefix_timings': {}, 'converged': False,
            'selected_candidate_id': 5,
        }
        for field, value in bad_fields.items():
            with self.subTest(field=field):
                row = deepcopy(self.rows[3][0])
                row[field] = value
                with self.assertRaises(ValueError):
                    p._validate_row(row, 3, self.bindings)
        row = deepcopy(self.rows[3][0])
        row['test']['arrays']['xi1']['array_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'different test'):
            p._validate_row(row, 3, self.bindings)

    def test_global_barrier_and_nonconvergence_cannot_be_masked(self):
        bindings = deepcopy(self.bindings)
        identifier = self.rows[10][-1]['id']
        bindings['results'][identifier]['journal_verified'] = False
        with self.assertRaisesRegex(ValueError, 'barrier'):
            p._validate_bindings(bindings)
        bindings = deepcopy(self.bindings)
        for job in bindings['jobs'].values():
            job.update(converged=True, terminal_status='converged')
        with self.assertRaisesRegex(ValueError, 'eleven'):
            p._validate_bindings(bindings)
        bindings = deepcopy(self.bindings)
        capped = next(j for j in bindings['jobs'].values() if not j['converged'])
        capped['stopping_candidate_id'] = 39
        with self.assertRaisesRegex(ValueError, 'terminal 40'):
            p._validate_bindings(bindings)
        bindings = deepcopy(self.bindings)
        del bindings['results'][identifier]
        with self.assertRaisesRegex(ValueError, 'barrier'):
            p._validate_bindings(bindings)

    def test_missing_and_duplicate_cohorts(self):
        for rows in (self.rows[3][:-1], self.rows[3] + self.rows[3][:1]):
            with self.assertRaisesRegex(ValueError, 'Incomplete black|Duplicate black'):
                p.build_overlay(self.bases[3], rows, 3, self.bindings)

    def test_frozen_source_change_rejected(self):
        bindings = deepcopy(self.bindings)
        path = str(p.RENDERER_PATHS[3])
        bindings['renderers']['3']['sha256'] = '0'*64
        bindings['source_sha256'][path] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source changed'):
            p._load_original(3, bindings)

    def test_raw_proofs_and_base_binding_required(self):
        bindings = deepcopy(self.bindings)
        bindings['results'][self.rows[3][0]['id']]['raw_proofs'] = {}
        with self.assertRaisesRegex(ValueError, 'proof references'):
            p._validate_bindings(bindings)
        bindings = deepcopy(self.bindings)
        bindings['base_sha256']['3'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'Base snapshot mismatch'):
            p.build_overlay(self.bases[3], self.rows[3], 3, bindings)

    def test_timing_and_raw_evidence_not_blended(self):
        for R, overlay in self.overlays.items():
            indexed = {row['id']: row for row in self.rows[R]}
            for panel in overlay['panels']:
                for group in panel['black']['provenance']['per_seed']:
                    for row in group:
                        self.assertEqual(row, indexed[row['id']])
                        self.assertEqual(row['wall_time_s'], .125)
                        self.assertEqual(row['source_prefix_timings'],
                                         {'status': 'unavailable; rerun required'})

    def test_private_renderer_geometry_and_original_artists(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes

        real_plot, real_xlim = Axes.plot, Axes.set_xlim
        for R in (3, 10):
            artists, limits, saved = [], [], []

            def plot(ax, *args, **kwargs):
                if 'label' in kwargs:
                    artists.append((deepcopy(args), deepcopy(kwargs)))
                return real_plot(ax, *args, **kwargs)

            def xlim(ax, *args, **kwargs):
                if args == (0, 1.5) or args == (0, 2.):
                    limits.append(args)
                return real_xlim(ax, *args, **kwargs)

            with tempfile.TemporaryDirectory() as tmp, patch.object(Axes, 'plot', plot), \
                    patch.object(Axes, 'set_xlim', xlim), \
                    patch.object(Figure, 'savefig', lambda fig, path, **kw: saved.append(Path(path).name)):
                p._load_original(R, self.bindings)['render'](self.bases[R], Path(tmp))
                original_artists = deepcopy(artists)
                artists.clear()
                limits.clear()
                saved.clear()
                p._load_original(R, self.bindings, overlay=True)['render'](self.overlays[R], Path(tmp))
            nonblack = [a for a in artists if not a[1]['label'].startswith('Black Moreau')]
            self.assertEqual(nonblack, original_artists)
            self.assertEqual(limits, [(0, 1.5 if R == 3 else 2.)]*3)
            self.assertEqual(saved, [f'comparison_R{R}_with_black.png', f'comparison_R{R}_with_black.pdf'])
            black = [a for a in artists if a[1]['label'] == 'Black Moreau OOS']
            self.assertEqual(len(black), 3)
            self.assertTrue(all(a[0][0] == p.RADII[R] and a[1]['color'] == '#000000' for a in black))
            inside = [a for a in artists if a[1]['label'] == 'Black Moreau IS']
            self.assertTrue(all(a[0][2] == '--' and a[1]['color'] == '#999999' for a in inside))

    def test_render_refuses_tampered_overlay_before_writes(self):
        data = deepcopy(self.overlays[3])
        data['panels'][0]['black']['oos']['mean'][0] += 1.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Overlay data/evidence changed'):
                p.render(data, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()