"""Deterministic solver-free tests. No experiments or historical test-data reads.

These tests are supplied for the parent; importing them does not run them.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

_spec = importlib.util.spec_from_file_location('_black_backend_under_test', Path(__file__).with_name('backend.py'))
assert _spec is not None and _spec.loader is not None
b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b)


def candidates(means):
    return [dict(candidate_id=i, assessed=True, ap_incumbent_mean=v,
                 checkpoint=f'candidate_{i}', checkpoint_sha256=f'bytes{i}',
                 state_sha256=f'state{i}', pool_sha256=f'pool{i}',
                 ap_reference=f'ap{i}') for i, v in enumerate(means)]


def grid():
    return [dict(id=f'R{r}_N{n}_s{s}_rho{rho:g}', R=r, Ntotal=n, seed=s, radius=rho)
            for r in (3, 10) for n in (50, 200, 800) for s in range(10)
            for rho in b.RADII + ((2.,) if r == 10 else ())]


class SelectionTests(unittest.TestCase):
    def test_terminal40_is_eligible_and_earliest_ties(self):
        prefix = candidates([20.] * 40 + [10.])
        self.assertEqual(b.choose(prefix)['candidate_id'], 40)
        prefix[7]['ap_incumbent_mean'] = 10.
        self.assertEqual(b.choose(prefix)['candidate_id'], 7)
        self.assertEqual(b.choose(candidates([1., 1.]))['candidate_id'], 0)

    def test_prefix_hole_unassessed_and_nan_rejected(self):
        for field, value in [('candidate_id', 8), ('assessed', False), ('ap_incumbent_mean', float('nan'))]:
            prefix = candidates([2., 1.])
            prefix[1][field] = value
            with self.assertRaises(ValueError):
                b.choose(prefix)

    def test_retained_minimum_not_terminal_pool(self):
        original = dict(iterations=candidates([10., 5.]), mp_count=1, stopping_candidate_id=1)
        boundary = dict(original['iterations'][-1], mp={'outgoing': True}, mp_call={},
                        added_sample_ids=[0], incoming_pool_evaluation={}, returned_pool_evaluation={})
        new = candidates([0., 0., 8.])[2:]
        record = dict(extension_boundary=boundary, new_iterations=new, mp_count=2,
                      new_mp_count=1, stopping_candidate_id=2)
        before = deepcopy(original)
        prefix, chain = b.joined_prefix(original, record)
        selected = b.choose(prefix)
        self.assertEqual(selected['pool_sha256'], 'pool1')
        self.assertEqual(selected['ap_reference'], 'ap1')
        self.assertNotIn('mp', selected)
        self.assertIn('mp', chain[1])
        self.assertEqual(original, before)

    def test_boundary_cannot_change_original_assessment(self):
        original = dict(iterations=candidates([1.]), mp_count=0, stopping_candidate_id=0)
        boundary = dict(original['iterations'][0], pool_sha256='different')
        with self.assertRaises(ValueError):
            b.joined_prefix(original, dict(extension_boundary=boundary, new_iterations=[],
                mp_count=0, new_mp_count=0, stopping_candidate_id=0))

    def test_cached_boundary_and_counts_are_exact(self):
        original = dict(iterations=candidates([1.]), mp_count=0, stopping_candidate_id=0)
        record = dict(extension_boundary=deepcopy(original['iterations'][0]), new_iterations=[],
                      mp_count=0, new_mp_count=0, stopping_candidate_id=0)
        prefix, chain = b.joined_prefix(original, record)
        self.assertEqual(prefix, chain)
        for key, value in [('mp_count', 1), ('new_mp_count', 1), ('stopping_candidate_id', 1)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.joined_prefix(original, dict(record, **{key: value}))
        record['extension_boundary']['unexpected'] = True
        with self.assertRaises(ValueError):
            b.joined_prefix(original, record)

    def test_radius2_starts_at_candidate0(self):
        record = dict(new_iterations=candidates([4., 2.]), mp_count=1,
                      new_mp_count=1, stopping_candidate_id=1)
        prefix, chain = b.joined_prefix(None, record)
        self.assertEqual(prefix, chain)
        self.assertEqual(b.choose(prefix)['candidate_id'], 1)

    def test_published_identity_not_test_cost(self):
        selected = candidates([5.])[0]
        published = dict(selected_iteration=0, checkpoint='candidate_0',
                         checkpoint_sha256='bytes0', state_sha256='state0', mean_oos=-1e10)
        b._published_identity(selected, published, legacy=True)
        published['state_sha256'] = 'other'
        with self.assertRaises(ValueError):
            b._published_identity(selected, published, legacy=True)

    def test_full_cohort_and_no_seed_filtering(self):
        jobs = grid()
        b.cohort(jobs)
        self.assertEqual(len(jobs), 570)
        for invalid in (jobs[:-1], jobs + jobs[:1], jobs[:-1] + jobs[:1]):
            with self.assertRaises(ValueError):
                b.cohort(invalid)
        self.assertEqual(sum(j['R'] == 10 and j['radius'] == 2. for j in jobs), 30)
        self.assertEqual(sum(j['R'] == 3 and j['Ntotal'] == 50 for j in jobs), 90)


class PoolTests(unittest.TestCase):
    def fixture(self):
        centers = np.asarray([[1.] * 6])
        pools = [centers.copy()]
        point = [1.1] * 6
        row = dict(sample_id=0, original_sample_id=4, candidate_id=40,
            candidate_sha256='state', checkpoint_sha256='bytes', pool_sha256=b.pool_digest(pools),
            seed=9, radius=.7, center1=[1.] * 3, center2=[1.] * 3,
            status='completed', controller_status='completed', solver_status=2, solcount=1,
            bound=12., pool_cost=10., residual=2./12., membership_violation=0.,
            membership={'exact_duplicate': False}, incumbent_violating=True,
            incumbent=dict(cost=11., worst1=point[:3], worst2=point[3:]))
        a = dict(candidate_id=40, assessed=True, failed_sample_ids=[], seed=9, radius=.7,
            samples=[row], pool_costs=[10.], state_sha256='state', checkpoint_sha256='bytes',
            pool_sha256=b.pool_digest(pools), ap_incumbent_mean=11., ap_bound_mean=12.,
            restricted_mean=10., separation_status='violating')
        job = dict(R=3, Ntotal=50, Ntrain=1, seed=9, radius=.7)
        return a, job, centers, [4], pools

    def test_terminal40_checks_every_raw_row(self):
        a, job, centers, ids, pools = self.fixture()
        closed, points = b.candidate_rows(a, job, centers, ids, pools)
        self.assertFalse(closed)
        self.assertEqual(points[0].dtype, np.float64)
        for key, bad in [('solver_status', 9), ('residual', 0.), ('candidate_sha256', 'other'),
                         ('original_sample_id', 5), ('bound', float('nan'))]:
            broken = deepcopy(a)
            broken['samples'][0][key] = bad
            with self.assertRaises(ValueError):
                b.candidate_rows(broken, job, centers, ids, pools)

    def test_no_saved_mean_substitution(self):
        a, job, centers, ids, pools = self.fixture()
        a['ap_incumbent_mean'] = 0.
        with self.assertRaises(ValueError):
            b.candidate_rows(a, job, centers, ids, pools)

    def test_box_upper_bound_preserves_producer_arithmetic(self):
        a, job, centers, ids, pools = self.fixture()
        centers[:] = 64.
        pools = [centers.copy()]
        radius = .001
        point = centers[0] + radius
        self.assertGreater(float((point - centers[0] - radius).max()), 0.)
        job['radius'] = a['radius'] = radius
        a['pool_sha256'] = b.pool_digest(pools)
        row = a['samples'][0]
        row.update(radius=radius, center1=centers[0, :3].tolist(),
                   center2=centers[0, 3:].tolist(), pool_sha256=a['pool_sha256'])
        row['incumbent'].update(worst1=point[:3].tolist(), worst2=point[3:].tolist())
        self.assertEqual(row['membership_violation'], 0.)
        b.candidate_rows(a, job, centers, ids, pools)

    def test_original_green_without_R_and_new_candidate_with_R(self):
        for r, n in ((3, 200), (10, 50)):
            with self.subTest(R=r):
                a, job, _, ids, _ = self.fixture()
                job.update(R=r, Ntotal=n)
                centers = np.ones((1, 2 * r))
                pools = [centers.copy()]
                a.update(Ntotal=n, Ntrain=1, ap={'R': r}, pool_sha256=b.pool_digest(pools))
                row = a['samples'][0]
                row.update(Ntotal=n, Ntrain=1, center1=[1.] * r, center2=[1.] * r,
                           pool_sha256=a['pool_sha256'])
                row['incumbent'].update(worst1=[1.1] * r, worst2=[1.1] * r)
                self.assertNotIn('R', a)
                b.candidate_rows(a, job, centers, ids, pools)
                a['R'] = r
                b.candidate_rows(a, job, centers, ids, pools)
                a['R'] = 10 if r == 3 else 3
                with self.assertRaises(ValueError):
                    b.candidate_rows(a, job, centers, ids, pools)
                del a['R']
                a['ap']['R'] = 10 if r == 3 else 3
                with self.assertRaises(ValueError):
                    b.candidate_rows(a, job, centers, ids, pools)
                a['ap']['R'] = r
                # Same concatenated center, but incorrect stage boundary.
                row['center1'].append(row['center2'].pop())
                with self.assertRaises(ValueError):
                    b.candidate_rows(a, job, centers, ids, pools)

    def test_exact_duplicate_insertion_and_raw_precision(self):
        pools = [np.asarray([[1., 2.]])] * 3
        point = np.asarray([1. + 1e-10, 2.])
        rows = [dict(incumbent_violating=True, membership={'exact_duplicate': False}),
                dict(incumbent_violating=True, membership={'exact_duplicate': True}),
                dict(incumbent_violating=False, membership={'exact_duplicate': False})]
        result, added = b.transition(pools, [point] * 3, rows)
        self.assertEqual(added, [0])
        self.assertEqual([len(p) for p in result], [2, 1, 1])
        self.assertEqual(result[0][-1, 0], 1. + 1e-10)
        self.assertNotEqual(b.pool_digest(result), b.pool_digest(pools))
        self.assertEqual([len(p) for p in pools], [1, 1, 1])

    def test_membership_failure_never_clipped(self):
        centers = np.asarray([[1., 2.]])
        pool = [[[1., 2.], [100., 2.]]]
        before = deepcopy(pool)
        with self.assertRaises(ValueError):
            b.validate_pools(pool, centers, .1)
        self.assertEqual(pool, before)

    def test_assessment_allows_only_documented_timing_lag(self):
        a = dict(candidate_id=0, mp_call={'wall_time_s': 2.}, mp_total_time_s=2., samples=[])
        disk = dict(a, mp_total_time_s=0.)
        b.assessment_equal(a, disk)
        self.assertEqual(disk['mp_total_time_s'], 0.)
        b.assessment_equal(a, dict(a))
        for bad in (-1., 1., 3., float('nan')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                b.assessment_equal(a, dict(disk, mp_total_time_s=bad))
        with self.assertRaises(ValueError):
            b.assessment_equal(dict(a, mp_total_time_s=0.), disk)
        disk['candidate_id'] = 1
        with self.assertRaises(ValueError):
            b.assessment_equal(a, disk)

    def test_continuation_assessment_omits_derived_time_on_both_sides(self):
        a = dict(candidate_id=1, mp_call={'wall_time_s': 2.}, samples=[])
        b.assessment_equal(a, deepcopy(a))
        for left, right in ((a, dict(a, mp_total_time_s=0.)),
                            (dict(a, mp_total_time_s=2.), a),
                            (a, dict(a, mp_call={'wall_time_s': 3.}))):
            with self.assertRaises(ValueError):
                b.assessment_equal(left, right)
        terminal = dict(candidate_id=2, mp_total_time_s=0., samples=[])
        b.assessment_equal(terminal, deepcopy(terminal))


class IntegrityTests(unittest.TestCase):
    def test_materialize_preserves_original_and_checks_both_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.json'
            source.write_text('{"training": true}')
            original = source.read_bytes()
            digest = b.sha(source)
            files = b.materialize(dict(input_sha256={str(source): digest}, source_sha256={}), root / 'copies')
            manifest = dict(files=files)
            copy = b._bound(manifest, source, digest)
            self.assertEqual(copy.read_bytes(), original)
            self.assertEqual(source.read_bytes(), original)
            source.write_text('changed')
            with self.assertRaises(ValueError):
                b._bound(manifest, source, digest)
            source.write_bytes(original)
            copy.chmod(0o644)
            copy.write_text('changed copy')
            with self.assertRaises(ValueError):
                b._bound(manifest, source, digest)

    def test_no_copy_of_changed_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            source.write_text('old')
            digest = b.sha(source)
            source.write_text('new')
            with self.assertRaises(ValueError):
                b.materialize(dict(input_sha256={str(source): digest}, source_sha256={}), Path(tmp) / 'copies')

    def test_manifest_requires_full_cohort_and_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            manifest = dict(status='frozen', policies_frozen_before_test=True,
                threads=1, physical_tolerance=1e-4, membership_tolerance=1e-9, workers=16, jobs=grid())
            path.write_text(json.dumps(manifest))
            path.with_suffix('.sha256').write_text(b.sha(path))
            parsed, _ = b._sealed(path)
            self.assertEqual(len(parsed['jobs']), 570)
            path.write_text(json.dumps(dict(manifest, workers=32)))
            with self.assertRaises(ValueError):
                b._sealed(path)

    def test_private_full_namespace_does_not_touch_bare_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'evaluation.py'
            path.write_text('from __future__ import annotations\n'
                'from plot_comparison import validate_cache_protocol\n'
                'VALUE = 7\n'
                'def evaluate():\n    return validate_cache_protocol(VALUE)\n')
            sentinel = object()
            old = sys.modules.get('plot_comparison', sentinel)
            plot = SimpleNamespace(validate_cache_protocol=lambda value: value + 1)
            module = b._private_module(path, '_private_test', plot)
            self.assertEqual(module.evaluate(), 8)
            self.assertIs(sys.modules.get('plot_comparison', sentinel), old)
            self.assertNotIn('_private_test', sys.modules)

    def test_initialization_wrapper_once_and_rejects_dimension_switch(self):
        calls = []
        e = SimpleNamespace(initialize_worker=lambda: calls.append('init'), versions=lambda: {'test': 1},
            c=SimpleNamespace(PHYSICAL_TOL=1e-4, torch=SimpleNamespace(
                get_num_threads=lambda: 1, get_num_interop_threads=lambda: 1)))
        manifest = {'base_comparisons': {'3': {'versions': {'test': 1},
            'manifest_path': str(b.BASE_MANIFESTS[3]), 'manifest_sha256': b.LOCKS[b.BASE_MANIFESTS[3]]}}}
        with patch.object(b, '_CONTEXT', None), patch.object(b, '_sealed', return_value=(manifest, 'seal')), \
             patch.object(b, '_verify_sources'), patch.object(b, '_private_module', return_value=e), \
                         patch.object(b, '_bound', return_value=Path('/unused')) as bound, \
                         patch.object(b, 'read', return_value={'versions': {'test': 1}}) as read, \
             patch.dict(b.os.environ, {}), patch.object(sys, 'dont_write_bytecode', False):
            self.assertIs(b.initialize_worker(3, '/unused'), e)
            self.assertIs(b.initialize_worker(3, '/unused'), e)
            self.assertEqual(calls, ['init'])
            bound.assert_called_with(manifest, str(b.BASE_MANIFESTS[3]), b.LOCKS[b.BASE_MANIFESTS[3]])
            read.assert_called_with(Path('/unused'))
            with patch.object(b, 'read', return_value={'versions': {'test': 2}}):
                with self.assertRaises(ValueError):
                    b.initialize_worker(3, '/unused')
            with self.assertRaises(ValueError):
                b.initialize_worker(10, '/unused')

    def test_collection_has_no_evaluator_or_test_array_call(self):
        tree = ast.parse(Path(b.__file__).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_policies')
        names = {n.func.id for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertTrue(names.isdisjoint({'_test_arrays', 'worker', '_execute', 'initialize_worker', '_private_module'}))
        attributes = {n.func.attr for n in ast.walk(function) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertTrue(attributes.isdisjoint({'load_model', '_evaluate', 'gen', 'load'}))

    def test_mp_paths_original_boundary_later_and_fresh_radius2(self):
        tree = ast.parse(Path(b.__file__).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_prefix')
        routes = [n for n in ast.walk(function) if isinstance(n, ast.If)
                  and isinstance(n.test, ast.BoolOp)
                  and 'continuation_path is not None' in ast.unparse(n.test)]
        self.assertEqual(len(routes), 1)
        code = compile(ast.Module(body=routes, type_ignores=[]), '<MP path branch>', 'exec')
        for source, retained, index, expected in (
            (None, 3, 0, '/original/candidate_00/mp'),
            ('/new/summary.json', 3, 1, '/original/candidate_01/mp'),
            ('/new/summary.json', 3, 2, '/new/extension_boundary/candidate_02/mp'),
            ('/new/summary.json', 3, 3, '/new/extension_boundary/candidate_03/mp'),
            ('/new/summary.json', 0, 0, '/new/extension_boundary/candidate_00/mp')):
            with self.subTest(source=source, retained=retained, index=index):
                namespace = dict(Path=Path, continuation_path=source, original_count=retained, i=index,
                                 a={'ap_reference': f'/original/candidate_{index:02d}/assessment.json'})
                exec(code, namespace)
                self.assertEqual(namespace['folder'], Path(expected))


class WorkerTests(unittest.TestCase):
    def test_smoke_uses_exact_two_training_centers_without_tests(self):
        from contextlib import contextmanager
        @contextmanager
        def phase(phases, name):
            phases[name] = {'status': 'completed', 'wall_time_s': 0.}
            yield phases[name]
        calls = []
        def evaluate(model, xi1, xi2, name, record, folder, sizes=None):
            calls.append((xi1.copy(), xi2.copy(), name, sizes))
            record.update(mean_oos=4., solves_attempted=2, solves_accepted=2, status_counts={'2': 2})
        e = SimpleNamespace(_evaluate=evaluate, c=SimpleNamespace(phase=phase, state_hash=lambda m: 'state'))
        centers = np.arange(24, dtype=float).reshape(4, 6)
        job = dict(id='probe', R=3, state_sha256='state', input_files=[], input_sha256={},
                   original_sample_ids=[8, 4, 3, 1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text('{}')
            digest = b.sha(path)
            path.with_suffix('.sha256').write_text(digest)
            with patch.object(b, '_sealed', return_value=({'workers': 16}, digest)), \
                 patch.object(b, 'initialize_worker', return_value=e), patch.object(b, '_verify_sources'), \
                 patch.object(b, '_own_inputs', return_value=(object(), centers, [])), \
                 patch.object(b, '_test_arrays', side_effect=AssertionError('test access forbidden')):
                result = b.training_smoke(job, path, Path(tmp) / 'out')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['training_probe_original_sample_ids'], [8, 4])
            self.assertEqual(result['training_probe_mean'], 4.)
            self.assertIsNone(result['mean_oos'])
            self.assertFalse(result['test_evaluated'])
            self.assertEqual(calls[0][2], 'training_probe')
            self.assertIsNone(calls[0][3])
            np.testing.assert_array_equal(calls[0][0], centers[:2, :3])
            np.testing.assert_array_equal(calls[0][1], centers[:2, 3:])
            self.assertEqual(result['summary_sha256'], hashlib.sha256(Path(result['result_path']).read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()