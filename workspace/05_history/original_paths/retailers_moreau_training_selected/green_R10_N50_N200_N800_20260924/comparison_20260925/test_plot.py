"""Solver-free artificial fixtures ONLY; never observations or model runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import plot_comparison as p


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture(root):
    """810 labelled artificial policies with checksummed stand-in input bytes."""
    arrays = {name: dict(array_sha256=digest(name), dtype='float64', shape=[10000, 10],
                        file=f'/nonexistent/artificial/{name}.npy', file_sha256=digest('file-'+name))
              for name in ('xi1', 'xi2')}
    m = dict(status='frozen', R=10, n_test=10000, test_seed=9999, files={}, jobs=[],
             job_count=810, test=dict(n_test=10000, seed=9999, arrays=arrays),
             cached_baselines={}, scs_radii={}, saved_green_selections=[])

    def freeze(name, obj):
        source = '/nonexistent/original/' + name
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(obj, allow_nan=False))
        m['files'][source] = dict(copy=str(dest), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        return source

    rows = []
    for n in p.SIZES:
        grid = p.SCS_RADII.copy()
        m['scs_radii'][str(n)] = grid
        for method, radii in [('green', p.GREEN_RADII), ('scs', grid)]:
            for rho in radii:
                for seed in p.SEEDS:
                    identity = f'{method}-{n}-{rho}-{seed}'
                    checkpoint = freeze(identity+'.json', dict(artificial_checkpoint=identity))
                    ntrain = 4*n//5 if method == 'green' else n
                    cost = float(n/100 + seed + rho)
                    spec = dict(id=identity, Ntotal=n, method=method, radius=rho, seed=seed,
                        Ntrain=ntrain, checkpoint=checkpoint,
                        checkpoint_sha256=m['files'][checkpoint]['sha256'], state_sha256=digest(identity),
                        pool_sha256=digest('pool-'+identity), sizes=[2]*ntrain, input_files=[checkpoint],
                        selected_candidate_id=4, stopping_candidate_id=5,
                        training_runtime_reference=dict(status='persisted_not_current'),
                        scs_historical_membership=dict(box_residual=7.91e-6, certified=False))
                    if method == 'scs' and rho == 0. and seed == 0:
                        spec['centers'] = [[1.]*20 for _ in range(n)]  # Artificial training centers.
                    m['jobs'].append(spec)
                    maxima = [cost+2]*ntrain
                    rows.append(dict(deepcopy(spec), status='completed', cached=False,
                        mean_oos=cost, finite_pool_mean=float(np.mean(maxima)), per_center_maxima=maxima,
                        manifest_sha256=digest('manifest'), test=deepcopy(m['test']),
                        solves_attempted=10000+2*ntrain, solves_accepted=10000+2*ntrain,
                        status_counts={'2': 10000+2*ntrain}, wall_time_s=.125))
                    if method == 'green' and rho == 1.5:
                        m['saved_green_selections'].append(dict(Ntotal=n, seed=seed, radius=rho,
                            method='green_validation', status='completed', checkpoint=checkpoint,
                            checkpoint_sha256=spec['checkpoint_sha256'], state_sha256=spec['state_sha256'],
                            candidate_id=4, stopping_candidate_id=5))
        knn = dict(oos=[float(s+n) for s in p.SEEDS], selection_metric='validation',
                   selected_k_by_seed=[3]*10)
        sro = {str(rho): {str(s): dict(oos=s+n+rho, **{'is': s+n}, K=3, time_total=.5)
                         for s in p.SEEDS} for rho in (.001, .07, 1.5, 2.)}
        cache = dict(meta=dict(N_train=n, R=10, N_test=10000, seed_test=9999,
                              seeds=p.SEEDS.copy(), projection_method='gurobi_qp'),
                     sweep={str(r): {} for r in grid},
                     benchmarks=dict(saa_knn=knn, saa_lb=dict(oos=[float(n+s+4) for s in p.SEEDS])))
        if n == 800:
            del cache['meta']['R']  # Preserve the actual legacy omission in every fixture.
        m['cached_baselines'][str(n)] = dict(saa_knn=knn, sro_fa=sro,
            cache=freeze(f'cache{n}.json', cache), sro_source=freeze(f'sro{n}.json', sro))
        baseline = m['cached_baselines'][str(n)]
        job = next(j for j in m['jobs'] if j['method'] == 'scs' and j['Ntotal'] == n
                   and j['radius'] == 0. and j['seed'] == 0)
        sources = [freeze(f'dimension{n}/{name}.py', dict(artificial_source=name))
                   for name in ('evaluation', 'reference', 'compare', 'historical_evaluator')]
        baseline['source_references'] = sources[-1:]
        bindings = [baseline['cache'], job['checkpoint'], *sources]
        baseline['dimension_evidence'] = dict(schema='scs_checkpoint_centers_R10_v1',
            Ntotal=n, verified_R=10, validator='c.load_model(old)', job_id=job['id'],
            **{k: job[k] for k in ('checkpoint', 'checkpoint_sha256', 'state_sha256')},
            centers_shape=[n, 20],
            state_shapes={'Q10': [1], 'Q1r': [10], 'fc_layers.0.weight': [16, 21],
                          'fc_layers.0.bias': [16], 'fc_layers.2.weight': [16, 16],
                          'fc_layers.2.bias': [16], 'fc_layers.4.weight': [10, 16],
                          'fc_layers.4.bias': [10], 'proj_layer.q': [12, 12], 'proj_layer.A': [45, 12]},
            source_sha256={s: m['files'][s]['sha256'] for s in bindings},
            scope='Artificial dimension evidence for solver-free unit tests only.')
    return m, rows


def oracle_fixture(m):
    """Clearly FAKE solver-free oracle; stand-in source bytes, no LP/data generation.

    Keep fixture(root)'s two-value return API for the separate runner tests.
    """
    root = Path(next(iter(m['files'].values()))['copy']).parent
    source = '/nonexistent/artificial/oracle.py'
    dest = root / 'oracle.py'
    dest.write_text('# Artificial oracle source ONLY; not an LP implementation.\n')
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    m['files'][source] = dict(copy=str(dest), sha256=sha)
    return dict(method='empirical_clairvoyant_reference', status='completed', R=10,
                n_test=10000, test_seed=9999, test=deepcopy(m['test']), solver_status=2,
                independent_solves=1, solves_attempted=1, solves_accepted=1,
                status_counts={'2': 1}, mean_cost=12.5, wall_time_s=.25,
                source_sha256={source: sha}, fixture_only='FAKE; never measured')


def rewrite(m, source, change):
    """Reseal a deliberately modified test fixture, not a research artifact."""
    info = m['files'][source]
    path = Path(info['copy'])
    obj = json.loads(path.read_text())
    change(obj)
    path.write_text(json.dumps(obj))
    info['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for baseline in m['cached_baselines'].values():
        bindings = baseline['dimension_evidence']['source_sha256']
        if source in bindings:
            bindings[source] = info['sha256']
    return obj


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.m, self.rows = fixture(self.root)
        self.oracle = oracle_fixture(self.m)

    def build(self):
        return p.build_plot_data(self.m, self.rows, self.oracle)

    def test_complete_fresh_shapes_seed_alignment_percentiles_and_no_mutation(self):
        original = deepcopy((self.m, self.rows, self.oracle))
        data = self.build()
        self.assertEqual((self.m, self.rows, self.oracle), original)
        self.assertEqual(data, p.build_plot_data(self.m, list(reversed(self.rows)), self.oracle))
        self.assertEqual(len(self.rows), 810)
        self.assertEqual(sum(r['method'] == 'green' for r in self.rows), 270)
        self.assertEqual(sum(r['method'] == 'scs' for r in self.rows), 540)
        self.assertEqual(p.SCS_COUNTS, {50: 18, 200: 18, 800: 18})
        for panel in data['panels']:
            n = panel['Ntotal']
            self.assertEqual(panel['training_counts'], dict(green=4*n//5, scs=n))
            self.assertEqual(panel['green']['radius'], p.GREEN_RADII)
            self.assertEqual(panel['scs']['radius'], p.SCS_RADII)
            self.assertEqual(panel['sro_fa']['radius'], [.001, .07, 1.5])
            for method in ('green', 'scs'):
                curve = panel[method]
                self.assertFalse(curve['provenance']['cached'])
                self.assertEqual(curve['provenance']['current_evaluation_time_s'][0], [.125]*10)
                for field in ('oos', 'finite_pool'):
                    values = np.asarray(curve[field]['per_seed'])
                    np.testing.assert_array_equal(curve[field]['p20'], np.percentile(values, 20, axis=1))
                    np.testing.assert_array_equal(curve[field]['p80'], np.percentile(values, 80, axis=1))
                np.testing.assert_array_equal(curve['oos']['per_seed'][0], np.arange(10)+n/100)
            self.assertEqual(panel['sro_fa']['provenance']['strict_test_status'], 'not_available')
        self.assertEqual(len(data['validation_selected_oos']), 30)
        self.assertIn('not confidence intervals', data['band_definition'])
        for text in ('810', '270', '540', '9999', '7.91e-6', 'nonconverged', '40/160/640',
                     '50/200/800', 'visual only', 'not box-certified', 'descriptive'):
            self.assertIn(text, data['caption'])
        json.dumps(data, allow_nan=False)

    def test_missing_duplicate_failed_nonfinite_and_identity_rows_rejected(self):
        changes = [lambda r: r.pop(), lambda r: r.append(deepcopy(r[0])),
            lambda r: r[1].update(id=r[0]['id']), lambda r: r[0].update(status='failed'),
            lambda r: r[0].update(status='running'), lambda r: r[0].update(mean_oos=float('nan')),
            lambda r: r[0].update(finite_pool_mean=float('inf')), lambda r: r[0].update(seed=10),
            lambda r: r[0].update(seed=True), lambda r: r[0].update(Ntrain=999),
            lambda r: r[0].update(checkpoint_sha256=digest('wrong')),
            lambda r: r[0].update(pool_sha256=digest('wrong')),
            lambda r: r[0].update(state_sha256=digest('wrong')),
            lambda r: r[0].update(selected_candidate_id=99),
            lambda r: r[0].update(manifest_sha256=digest('other manifest'))]
        for i, change in enumerate(changes):
            rows = deepcopy(self.rows)
            change(rows)
            with self.subTest(case=i), self.assertRaises(ValueError):
                p.build_plot_data(self.m, rows, self.oracle)

    def test_no_fake_cached_n50_or_missing_n50(self):
        self.m['cached_green_N50'] = dict(jobs=[])
        with self.assertRaisesRegex(ValueError, 'cached green N50'):
            self.build()
        del self.m['cached_green_N50']
        self.rows[0]['cached'] = True
        with self.assertRaisesRegex(ValueError, 'fresh replay'):
            self.build()
        self.rows[0]['cached'] = False
        self.rows = [r for r in self.rows if not (r['method'] == 'green' and r['Ntotal'] == 50)]
        with self.assertRaisesRegex(ValueError, '810 fresh'):
            self.build()

    def test_wrong_or_incomplete_grid_rejected_even_if_rows_match(self):
        m = deepcopy(self.m)
        m['scs_radii']['50'].pop()
        with self.assertRaisesRegex(ValueError, 'Grid count'):
            p.build_plot_data(m, self.rows, self.oracle)
        self.m['scs_radii']['50'][1] = .0005
        for row in self.m['jobs'] + self.rows:
            if row['method'] == 'scs' and row['Ntotal'] == 50 and row['radius'] == .001:
                row['radius'] = .0005
        with self.assertRaisesRegex(ValueError, 'saved radius grid'):
            self.build()

    def test_wrong_test_shape_dtype_seed_and_worker_hash_rejected(self):
        for change in (lambda m: m['test'].update(seed=1),
                       lambda m: m['test']['arrays']['xi1'].update(shape=[10000, 3]),
                       lambda m: m['test']['arrays']['xi2'].update(dtype='float32'),
                       lambda m: m['test']['arrays']['xi1'].update(array_sha256=''),
                       lambda m: m['test']['arrays']['xi2'].update(array_sha256=digest('other'))):
            m = deepcopy(self.m)
            change(m)
            with self.assertRaises(ValueError):
                p.build_plot_data(m, self.rows, self.oracle)
        self.rows[0]['test']['arrays']['xi1']['array_sha256'] = digest('wrong')
        with self.assertRaisesRegex(ValueError, 'Worker test identity'):
            self.build()

    def test_pool_maxima_mean_and_complete_strict_solves_required(self):
        for change in (lambda r: r[0]['per_center_maxima'].pop(),
                       lambda r: r[0]['per_center_maxima'].__setitem__(0, float('nan')),
                       lambda r: r[0]['per_center_maxima'].__setitem__(0, -999.),
                       lambda r: r[0].update(finite_pool_mean=-10.),
                       lambda r: r[0].update(solves_accepted=10000),
                       lambda r: r[0].update(status_counts={'2': 10079, '9': 1})):
            rows = deepcopy(self.rows)
            change(rows)
            with self.assertRaises(ValueError):
                p.build_plot_data(self.m, rows, self.oracle)
        for row in (self.m['jobs'][0], self.rows[0]):
            row['sizes'][0] = 0
        with self.assertRaisesRegex(ValueError, 'pool sizes'):
            self.build()

    def test_unbound_tampered_inputs_and_checkpoint_hash_rejected(self):
        source = self.rows[0]['checkpoint']
        info = self.m['files'].pop(source)
        with self.assertRaisesRegex(ValueError, 'Unbound'):
            self.build()
        self.m['files'][source] = info
        Path(info['copy']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()
        info['sha256'] = hashlib.sha256(b'{}').hexdigest()
        with self.assertRaisesRegex(ValueError, 'Checkpoint hash mismatch'):
            self.build()

    def test_required_hashes_and_shared_manifest_digest(self):
        for field in ('state_sha256', 'pool_sha256', 'checkpoint_sha256'):
            m, rows = deepcopy((self.m, self.rows))
            del m['jobs'][0][field]
            del rows[0][field]
            with self.assertRaisesRegex(ValueError, 'Missing/invalid'):
                p.build_plot_data(m, rows, self.oracle)
        self.m['manifest_sha256'] = digest('wrong')
        with self.assertRaisesRegex(ValueError, 'Manifest hash mismatch'):
            self.build()

    def test_historical_baseline_binding_seed_order_and_protocol(self):
        for change in (lambda b: b['saa_knn']['oos'].pop(),
                       lambda b: b['saa_knn']['oos'].__setitem__(0, float('nan')),
                       lambda b: b['sro_fa']['0.001'].pop('9'),
                       lambda b: b.pop('sro_source'), lambda b: b.pop('cache')):
            m = deepcopy(self.m)
            change(m['cached_baselines']['50'])
            with self.assertRaises(ValueError):
                p.build_plot_data(m, self.rows, self.oracle)
        source = self.m['cached_baselines']['50']['cache']
        original = Path(self.m['files'][source]['copy']).read_text()
        for change in (lambda d: d['meta']['seeds'].reverse(),
                       lambda d: d['meta'].update(R=3),
                       lambda d: d['meta'].update(seed_test=42)):
            Path(self.m['files'][source]['copy']).write_text(original)
            rewrite(self.m, source, change)
            with self.assertRaisesRegex(ValueError, 'protocol/seed order'):
                self.build()

    def test_legacy_missing_R_is_explicit_pure_and_preserved_in_plot(self):
        baseline = self.m['cached_baselines']['800']
        cache = p._frozen_json(self.m, baseline['cache'])
        before = deepcopy((self.m, cache))
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('Validator must be pure')):
            protocol = p.validate_cache_protocol(cache, 800, self.m)
        self.assertEqual((self.m, cache), before)
        self.assertEqual(protocol['schema'], 'legacy_N800_R10_10k_missing_R')
        self.assertIsNone(protocol['recorded_R'])
        self.assertEqual(protocol['independently_verified_R'], 10)
        panel = self.build()['panels'][2]
        self.assertEqual(panel['cache_protocol'], protocol)
        self.assertNotIn('R', panel['historical_saa_reference']['metadata'])
        self.assertNotIn('R', p._frozen_json(self.m, baseline['cache'])['meta'])

    def test_missing_R_other_sizes_wrong_R_and_unknown_schema_rejected(self):
        for n in p.SIZES:
            baseline = self.m['cached_baselines'][str(n)]
            cache = p._frozen_json(self.m, baseline['cache'])
            for value in (3, None, '10', 10., True):
                bad = deepcopy(cache)
                bad['meta']['R'] = value
                with self.subTest(n=n, R=value), self.assertRaisesRegex(ValueError, 'recorded R'):
                    p.validate_cache_protocol(bad, n, self.m)
            if n != 800:
                del cache['meta']['R']
                with self.assertRaisesRegex(ValueError, 'missing R only allowed for N800'):
                    p.validate_cache_protocol(cache, n, self.m)
            else:
                cache['meta']['unexpected_dimension'] = 10
                with self.assertRaisesRegex(ValueError, 'unsupported meta schema'):
                    p.validate_cache_protocol(cache, n, self.m)
        for change in (lambda m: m.pop('R'), lambda m: m.update(R=3)):
            m = deepcopy(self.m)
            change(m)
            with self.assertRaisesRegex(ValueError, 'manifest R'):
                p.build_plot_data(m, self.rows, self.oracle)

    def test_legacy_requires_bound_checkpoint_center_and_source_dimension_evidence(self):
        source = self.m['cached_baselines']['800']['cache']
        cache = p._frozen_json(self.m, source)
        changes = [lambda b: b.pop('dimension_evidence'),
            lambda b: b['dimension_evidence'].update(verified_R=3),
            lambda b: b['dimension_evidence'].update(validator='unchecked'),
            lambda b: b['dimension_evidence'].update(job_id='unbound'),
            lambda b: b['dimension_evidence'].update(checkpoint_sha256=digest('wrong')),
            lambda b: b['dimension_evidence'].update(state_sha256=digest('wrong')),
            lambda b: b['dimension_evidence'].update(centers_shape=[800, 6]),
            lambda b: b['dimension_evidence']['state_shapes'].update(Q1r=[3]),
            lambda b: b['dimension_evidence'].update(source_sha256={}),
            lambda b: b['dimension_evidence']['source_sha256'].pop(source),
            lambda b: b['dimension_evidence']['source_sha256'].__setitem__(source, digest('wrong')),
            lambda b: b.pop('source_references')]
        for i, change in enumerate(changes):
            m = deepcopy(self.m)
            change(m['cached_baselines']['800'])
            with self.subTest(case=i), self.assertRaises(ValueError):
                p.validate_cache_protocol(cache, 800, m)
        evidence = self.m['cached_baselines']['800']['dimension_evidence']
        job = next(j for j in self.m['jobs'] if j['id'] == evidence['job_id'])
        job['centers'] = [[1.]*6 for _ in range(800)]
        with self.assertRaisesRegex(ValueError, 'dimension centers'):
            p.validate_cache_protocol(cache, 800, self.m)

    def test_actual_three_cache_schemas_with_synthetic_replay_rows(self):
        """Real cache bytes, FAKE oracle and neural rows; never a model evaluation."""
        original_hashes = {}
        for n in p.SIZES:
            base = Path('/data/home/aviyac/retailers_problem/R10')
            suffix = '_10k' if n == 800 else ''
            source = base / f'N{n}/rho_sweep_net16_v2/sweep_eval_net16_v2_N{n}_R10{suffix}.json'
            payload = source.read_bytes()
            original_hashes[source] = hashlib.sha256(payload).hexdigest()
            cache = json.loads(payload)
            baseline = self.m['cached_baselines'][str(n)]
            bound_cache = baseline['cache']
            dest = Path(self.m['files'][bound_cache]['copy'])
            dest.write_bytes(payload)  # Copy byte-for-byte; do not insert R.
            self.m['files'][bound_cache]['sha256'] = original_hashes[source]
            baseline['dimension_evidence']['source_sha256'][bound_cache] = original_hashes[source]
            baseline['saa_knn'] = deepcopy(cache['benchmarks']['saa_knn'])
            sro_source = base / f'sro_fa/sro_fa_results_N{n}_R10.json'
            sro_bytes = sro_source.read_bytes()
            original_hashes[sro_source] = hashlib.sha256(sro_bytes).hexdigest()
            baseline['sro_fa'] = json.loads(sro_bytes)
            info = self.m['files'][baseline['sro_source']]
            Path(info['copy']).write_bytes(sro_bytes)
            info['sha256'] = original_hashes[sro_source]
            p.validate_cache_protocol(cache, n, self.m)
            self.assertEqual('R' in cache['meta'], n != 800)
        data = self.build()
        self.assertEqual([panel['cache_protocol']['recorded_R'] for panel in data['panels']], [10, 10, None])
        self.assertEqual([panel['cache_protocol']['independently_verified_R'] for panel in data['panels']], [10]*3)
        self.assertEqual(len(data['panels'][2]['historical_saa_reference']['original']['oos']), 2)
        self.assertEqual([panel['horizontal'][0]['value'] for panel in data['panels']], [12.5]*3)
        for source, original in original_hashes.items():
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)

    def test_sro_incomplete_failed_nonfinite_duplicate_radius_and_gaps(self):
        baseline = self.m['cached_baselines']['50']
        original = deepcopy(baseline['sro_fa'])
        for change in (lambda d: d['0.001'].pop('9'),
                       lambda d: d['0.001']['0'].update(status='failed'),
                       lambda d: d['0.001']['0'].update(oos=float('nan')),
                       lambda d: d.update({'0.0010': deepcopy(d['0.001'])})):
            path = Path(self.m['files'][baseline['sro_source']]['copy'])
            path.write_text(json.dumps(original))
            baseline['sro_fa'] = rewrite(self.m, baseline['sro_source'], change)
            with self.assertRaises(ValueError):
                self.build()
        path.write_text(json.dumps(original))
        baseline['sro_fa'] = rewrite(self.m, baseline['sro_source'], lambda d: d.pop('0.07'))
        self.assertEqual(self.build()['panels'][0]['sro_fa']['radius'], [.001, 1.5])

    def test_common_oracle_scalar_not_cached_seed_aggregates(self):
        for panel in self.build()['panels']:
            refs = panel['horizontal']
            self.assertEqual([r['kind'] for r in refs], ['empirical_clairvoyant_reference'])
            self.assertEqual(refs[0]['label'], 'Empirical clairvoyant reference')
            self.assertEqual(refs[0]['value'], 12.5)
            self.assertEqual(refs[0]['independent_solves'], 1)
            self.assertEqual(refs[0]['provenance'], self.oracle)
            self.assertTrue({'seeds', 'oos', 'p20', 'p80', 'per_seed'}.isdisjoint(refs[0]))
            raw = p._frozen_json(self.m, self.m['cached_baselines'][str(panel['Ntotal'])]['cache'])
            self.assertEqual(panel['historical_saa_reference']['original'], raw['benchmarks']['saa_lb'])
        for baseline in self.m['cached_baselines'].values():
            rewrite(self.m, baseline['cache'], lambda d: d['benchmarks'].pop('saa_lb'))
        self.assertTrue(all(panel['horizontal'][0]['value'] == 12.5 for panel in self.build()['panels']))

    def test_required_oracle_fields_status_counts_test_sources_and_timing(self):
        with self.assertRaises(TypeError):
            p.build_plot_data(self.m, self.rows)  # Third argument is deliberately required.
        for field in ('method', 'status', 'R', 'n_test', 'test_seed', 'test', 'solver_status',
                      'independent_solves', 'solves_attempted', 'solves_accepted', 'status_counts',
                      'mean_cost', 'wall_time_s', 'source_sha256'):
            bad = deepcopy(self.oracle)
            del bad[field]
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, 'oracle record fields'):
                p.build_plot_data(self.m, self.rows, bad)
        source = next(iter(self.oracle['source_sha256']))
        changes = [lambda o: o.update(status='failed'), lambda o: o.update(solver_status=9),
            lambda o: o.update(method='saa_lb'), lambda o: o.update(R=3),
            lambda o: o.update(n_test=50), lambda o: o.update(test_seed=0),
            lambda o: o.update(independent_solves=10), lambda o: o.update(solves_attempted=2),
            lambda o: o.update(solves_accepted=True), lambda o: o.update(status_counts={'2': 10}),
            lambda o: o.update(mean_cost=float('nan')), lambda o: o.update(mean_cost=float('inf')),
            lambda o: o.update(wall_time_s=float('nan')), lambda o: o.update(wall_time_s=-1.),
            lambda o: o['test']['arrays']['xi1'].update(file='/other/xi1.npy'),
            lambda o: o['test']['arrays']['xi2'].update(file_sha256=digest('wrong')),
            lambda o: o['test']['arrays']['xi1'].update(array_sha256=digest('wrong')),
            lambda o: o.update(source_sha256={}),
            lambda o: o['source_sha256'].update({source: digest('wrong')}),
            lambda o: o['source_sha256'].update({'/unbound/lp.py': digest('missing')})]
        for i, change in enumerate(changes):
            bad = deepcopy(self.oracle)
            change(bad)
            with self.subTest(case=i), self.assertRaises(ValueError):
                p.build_plot_data(self.m, self.rows, bad)
        Path(self.m['files'][source]['copy']).write_text('# Tampered artificial source\n')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()

    def test_cached_source_tamper_is_not_replaced_by_live_or_other_data(self):
        source = self.m['cached_baselines']['50']['cache']
        Path(self.m['files'][source]['copy']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build()

    def test_saved_validation_choices_preserved_not_test_argmin(self):
        winners = self.m['saved_green_selections']
        result = self.build()['validation_selected_oos']
        self.assertEqual([r['selection'] for r in result], winners)
        self.assertTrue(all(r['mean_oos'] == r['selection']['Ntotal']/100+r['selection']['seed']+1.5
                            for r in result))
        self.assertTrue(all(r['selection']['radius'] == 1.5 for r in result))
        for change in (lambda w: w.pop(), lambda w: w.__setitem__(1, deepcopy(w[0])),
                       lambda w: w[0].update(radius=.005),
                       lambda w: w[0].update(checkpoint_sha256=digest('wrong')),
                       lambda w: w[0].update(candidate_id=99),
                       lambda w: w[0].update(method='green_test'),
                       lambda w: w[0].update(status='failed')):
            m = deepcopy(self.m)
            change(m['saved_green_selections'])
            with self.assertRaises(ValueError):
                p.build_plot_data(m, self.rows, self.oracle)
        del self.m['saved_green_selections']
        with self.assertRaisesRegex(ValueError, 'validation selections'):
            self.build()

    def test_three_nonconverged_historical_n50_and_observational_pools_retained(self):
        keys = [i for i, r in enumerate(self.rows) if r['method'] == 'scs' and r['Ntotal'] == 50][:3]
        for i in keys:
            for row in (self.m['jobs'][i], self.rows[i]):
                row['training_runtime_reference']['ccg'] = dict(converged=False, status='max_iterations')
        curve = self.build()['panels'][0]['scs']
        self.assertEqual(len(curve['radius']), 18)
        evidence = curve['provenance']['per_seed'][0][:3]
        self.assertTrue(all(not r['training_runtime_reference']['ccg']['converged'] for r in evidence))
        self.assertTrue(all(r['scs_historical_membership']['box_residual'] == 7.91e-6 for r in evidence))

    def test_render_styles_nodes_zero_vector_pdf_and_300dpi(self):
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
        data = self.build()
        with patch.object(Figure, 'savefig', save):
            p.render(data, out)
        self.assertEqual(sorted(f.name for f in out.iterdir()), ['comparison_R10.pdf', 'comparison_R10.png'])
        with Image.open(out / 'comparison_R10.png') as image:
            self.assertAlmostEqual(image.info['dpi'][0], 300, places=1)
        pdf = (out / 'comparison_R10.pdf').read_bytes()
        self.assertTrue(pdf.startswith(b'%PDF'))
        self.assertNotIn(b'/Subtype /Image', pdf)
        fig = captured[0]
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.legends), 1)
        self.assertEqual(len(fig.legends[0].get_texts()), 7)
        self.assertIn('R=10', fig._suptitle.get_text())
        for ax, panel in zip(fig.axes, data['panels']):
            self.assertEqual(ax.get_xscale(), 'symlog')
            np.testing.assert_array_equal(ax.get_xticks(), p.TICKS)
            self.assertEqual(ax.get_xlim(), (0., 1.5))
            self.assertEqual(len(ax.lines), 7)
            self.assertEqual(ax.lines[-1].get_label(), 'Empirical clairvoyant reference')
            np.testing.assert_array_equal(ax.lines[-1].get_ydata(), [12.5, 12.5])
            self.assertEqual([line.get_linestyle() for line in ax.lines[:6]], ['-', '--', '-', '--', '-', '-'])
            self.assertEqual([line.get_color() for line in ax.lines[:6]],
                             [p.COLORS[k] for k in ('green', 'green_is', 'scs', 'scs_is', 'sro_fa', 'saa_knn')])
            for line, method, field in zip(ax.lines[:5], ('green', 'green', 'scs', 'scs', 'sro_fa'),
                                          ('oos', 'finite_pool', 'oos', 'finite_pool', 'oos')):
                np.testing.assert_array_equal(line.get_xdata(), panel[method]['radius'])
                np.testing.assert_array_equal(line.get_ydata(), panel[method][field]['mean'])
                self.assertFalse(line.get_rasterized())

    def test_render_rejects_partial_tampered_cached_or_conditional_data_before_writing(self):
        data = self.build()
        changes = [lambda d: d['panels'].pop(),
            lambda d: d['panels'][0]['green']['seeds'].reverse(),
            lambda d: d['panels'][0]['scs']['radius'].pop(),
            lambda d: d['panels'][0]['green']['oos']['mean'].__setitem__(0, float('nan')),
            lambda d: d['panels'][0]['green']['oos']['per_seed'][0].reverse(),
            lambda d: d['panels'][0]['green']['provenance'].update(cached=True),
            lambda d: d['panels'][0]['green']['provenance']['per_seed'][0][0]['per_center_maxima'].pop(),
            lambda d: d['panels'][0]['horizontal'][0].update(kind='conditional'),
            lambda d: d['panels'][1]['horizontal'].clear(),
            lambda d: d['panels'][2]['horizontal'][0].update(p20=0., p80=100.),
            lambda d: d['panels'][0]['horizontal'][0].update(value=-999.)]
        for i, change in enumerate(changes):
            altered = deepcopy(data)
            change(altered)
            out = self.root / f'must_not_exist_{i}'
            with self.subTest(case=i), self.assertRaises(ValueError):
                p.render(altered, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()