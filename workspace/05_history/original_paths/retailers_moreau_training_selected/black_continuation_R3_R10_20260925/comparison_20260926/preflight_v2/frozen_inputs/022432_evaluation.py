"""R10 fixed-policy replay, using exact-source R3 evaluation definitions.

Parent API is freeze, initialize_worker, worker, _evaluate, versions and c.
freeze captures 270 fresh green + 540 fresh SCS policies BEFORE test generation.
The parent then generates c.gen(10000, 9999), freezes float64 arrays of shape
(10000, 10), binds runner sources, workers and test metadata, and seals manifest
and its .sha256 sidecar as in R3. There is NO cached_green_N50 field or path.
No preflight/test generator/solver is invoked on import or during green freeze.

Historical SCS coordinates are observational, not box certified. Cached kNN and
SRO-FA retain original evidence/timings; adjacent scripts are not certified cache
producers. All fresh neural projection rows require OPTIMAL and are journalled.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

from plot_comparison import validate_cache_protocol

HERE = Path(__file__).resolve().parent
HOME = Path('/data/home/aviyac')
CAMPAIGN = HERE.parent
SOURCE = CAMPAIGN / 'results/20260924T131056.490113+0000'
REFERENCE = HERE / 'reference.py'
_loader = importlib.util.spec_from_file_location('_r10_comparison_reference', REFERENCE)
assert _loader is not None and _loader.loader is not None
c = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(c)
COMPARE = c.COMPARE
np = c.np
read_json, write_json = c.read_json, c.write_json
MEMBERSHIP_TOL = 1e-9
SCS_MEMBERSHIP_POLICY = 'historical_saved_coordinates_observational_not_box_certified'
RADII = c.RADII
SCS_RADII = (0., .001, .005, .01, .02, .05, .1, .15, .2, .3, .4, .45,
             .5, .6, .75, .8, 1., 1.5)
SOURCE_FILES = (Path(__file__).resolve(), REFERENCE, HERE / 'plot_comparison.py', HERE / 'test_evaluation.py',
                c.R3_EVALUATION, COMPARE)
EXPERIMENT = 'R10_fixed_policy_comparison_20260925'
# Predetermined artifact schemas, indexed by training seed; never an existence fallback.
NOMINAL_METADATA_KINDS = {
    50: ('json',) * 10,
    200: ('json',) * 10,
    800: ('json', 'json', 'checkpoint_metadata', 'checkpoint_metadata',
          'checkpoint_metadata', 'checkpoint_metadata', 'checkpoint_metadata',
          'checkpoint_metadata', 'checkpoint_metadata', 'checkpoint_metadata'),
}


# Reuse the complete R3 worker/failure/journal implementation. Each replacement
# is an exact fragment against a locked file, not a numeric/global substitution.
c.load_definitions(globals(), c.R3_EVALUATION,
    ('initialize_worker', 'check', 'stats', 'versions', 'pool_digest', 'finite_pool_cost',
     'validate_pools', 'validate_assessment', 'scs_pools', '_Snapshot', '_freeze_green',
     '_bound_file', '_load_inputs', '_evaluate', 'worker'),
    (('centers.shape[1] == 6', 'centers.shape[1] == 20', 1),
     ('a.shape[1] == 6', 'a.shape[1] == 20', 1),
     ('len(point) == 6', 'len(point) == 20', 1),
     ('for n in (200, 800)', 'for n in (50, 200, 800)', 1),
     ('len(jobs) == len(fixed) == 180', 'len(jobs) == len(fixed) == 270', 1),
     ('len(setups) == 20', 'len(setups) == 30', 1),
     ("len(summary['selections']) == 20", "len(summary['selections']) == 30", 1),
     ('raw.shape == (n, 6)', 'raw.shape == (n, 20)', 1),
     ("for path, digest in provenance['source_sha256'].items():",
      "check(len(provenance['source_sha256']) == 54, 'Expected 54 training source bindings')\n    for path, digest in provenance['source_sha256'].items():", 1),
     ('a, pool_path, pools = selected_green_pool(d, training, snap.read)',
      'a, pool_path, pools = selected_green_pool(d, training, snap.read)\n        prefix_inputs = capture_green_prefix(snap, d)', 1),
     ("a['checkpoint'], a['ap_reference'], d['selected_checkpoint']]",
      "a['checkpoint'], a['ap_reference'], d['selected_checkpoint']] + prefix_inputs", 1),
     ("selected_candidate_id=d['selected_candidate_id'], stopping_candidate_id=d['stopping_candidate_id'],",
      "selected_candidate_id=d['selected_candidate_id'], stopping_candidate_id=d['stopping_candidate_id'],\n            selected_green_gap=d['selected_green_gap'], stopping_green_gap=d['stopping_green_gap'],\n            selection_rule=d['selection_rule'], selected_own_gap_below_one_percent=d['selected_green_gap'] < .01,", 1),
     ("[10000, 3]", "[10000, 10]", 1),
     ("array_hash == info['array_sha256'] == m['cached_green_N50']['test_arrays'][key]['array_sha256']",
      "array_hash == info['array_sha256']", 1),
     ('Unmatched historical seed9999 test array', 'Unmatched frozen seed9999 test array', 1),
     ('np.empty((len(xi1), 3)', 'np.empty((len(xi1), 10)', 1),
     ("experiment='R3_fixed_policy_comparison_20260924'", 'experiment=EXPERIMENT', 1),
     ("points = np.concatenate(pools)", "points = np.concatenate(pools).reshape(-1, 20)", 1),
     ('points[:, :3], points[:, 3:]', 'points[:, :10], points[:, 10:]', 1)))


def selected_green_pool(record, training, read=read_json):
    """Verify the complete incoming prefix, never reselect or require selected gap <1%."""
    selected, stop = record['selected_candidate_id'], record['stopping_candidate_id']
    check(type(selected) is int and type(stop) is int and 0 <= selected <= stop,
          'Invalid selected/stopping candidate IDs')
    candidates = record['iterations']
    check(len(candidates) == stop + 1, 'Incomplete or future candidate prefix')
    check(record['selection_rule'] ==
          'earliest minimum mean AP incumbent over complete executed prefix including stop',
          'Changed green selection rule')
    ids = training['original_sample_ids']
    selected_result = None
    for i, a in enumerate(candidates):
        check(a['candidate_id'] == i and a['assessed'] and not a['failed_sample_ids'],
              f'Incomplete green candidate {i}')
        for key in ('Ntotal', 'Ntrain', 'seed', 'radius'):
            check(a[key] == record[key], f'Green candidate identity mismatch: {key}')
        check(np.isfinite([a['ap_incumbent_mean'], a['restricted_mean'], a['green_gap']]).all(),
              'Nonfinite green candidate statistics')
        samples = a['samples']
        check([s['sample_id'] for s in samples] == list(range(record['Ntrain']))
              and [s['original_sample_id'] for s in samples] == ids, 'Green AP ball coverage mismatch')
        for s, center in zip(samples, training['centers']):
            check(s['controller_status'] == 'completed', 'Failed AP sample')
            check(s['candidate_id'] == i and s['candidate_sha256'] == a['state_sha256']
                  and s['checkpoint_sha256'] == a['checkpoint_sha256']
                  and s['pool_sha256'] == a['pool_sha256'], 'AP policy/pool mismatch')
            check(all(s[k] == record[k] for k in ('Ntotal', 'Ntrain', 'seed', 'radius'))
                  and np.array_equal(s['center1'] + s['center2'], center), 'AP center/identity mismatch')
        path = Path(a['checkpoint']).parent / 'pool.json'
        check(c.sha(path) == a['pool_file_sha256'], f'Prefix pool file changed: {path}')
        pool = read(path)
        check(pool['original_sample_ids'] == ids, 'Prefix pool training IDs changed')
        pools = validate_pools(pool['pools'], training['centers'], record['radius'])
        check(pool_digest(pools) == pool['pool_sha256'] == a['pool_sha256'], 'Prefix pool digest changed')
        validate_assessment(read(a['ap_reference']), a)
        if i == selected:
            selected_result = a, path, pools
    # Audit the saved rule; this does not replace the incoming choice.
    check(selected == min(range(len(candidates)), key=lambda i: candidates[i]['ap_incumbent_mean']),
          'Incoming selection is not earliest minimum AP in its prefix')
    a = candidates[selected]
    for key in ('checkpoint_sha256', 'state_sha256'):
        check(a[key] == record['selected_' + key], f'Selected identity mismatch: {key}')
    check(a['ap_reference'] == record['selected_ap_reference'], 'Selected AP reference mismatch')
    check(a['green_gap'] == record['selected_green_gap'] and
          candidates[stop]['green_gap'] == record['stopping_green_gap'], 'Saved gap identity mismatch')
    check(selected_result is not None, 'Missing selected candidate')
    return selected_result


def capture_green_prefix(snap, record):
    inputs = []
    for a in record['iterations']:
        snap.model(a['checkpoint'], 'green', a['checkpoint_sha256'], a['state_sha256'])
        path = Path(a['checkpoint']).parent / 'pool.json'
        snap.capture(path, a['pool_file_sha256'])
        snap.capture(a['ap_reference'])
        inputs.extend([a['checkpoint'], str(path), a['ap_reference']])
    return inputs


def scs_choice(n, rho, seed, directories, nominal):
    check(n in (50, 200, 800), 'Invalid R10 sample size')
    return c.old_choice(rho, seed, directories, nominal)


def validate_knn(benchmarks):
    """Check per-seed LOO costs/tie breaking; NOT full-pipeline LOO."""
    check(benchmarks['_knn_k_selection'] == 'per_seed_loo_grid_v3', 'Unvalidated kNN selector')
    grid, knn, by_k = benchmarks['knn_k_grid'], benchmarks['saa_knn'], benchmarks['saa_knn_by_k']
    check(len(grid) > 0 and len(set(grid)) == len(grid) and
          all(type(k) is int and k > 0 for k in grid), 'Invalid kNN grid')
    for entry in [knn] + [by_k[str(k)] for k in grid]:
        for field in ('is', 'val', 'oos'):
            check(np.asarray(entry[field]).shape == (10,) and np.isfinite(entry[field]).all(),
                  f'Incomplete/nonfinite kNN {field}')
    check(len(knn['selected_k_by_seed']) == 10, 'Incomplete kNN selections')
    for seed in range(10):
        best = min(grid, key=lambda k: (by_k[str(k)]['val'][seed], k))
        check(knn['selected_k_by_seed'][seed] == best, 'kNN per-seed LOO selection mismatch')
        check(all(knn[field][seed] == by_k[str(best)][field][seed] for field in ('is', 'val', 'oos')),
              'kNN selected cache values mismatch')
    return knn


def nominal_metadata(snap, n, seed, nominal):
    """Bind original nominal artifacts, independently of the selected replay policy."""
    check(type(n) is int and n in NOMINAL_METADATA_KINDS
          and type(seed) is int and 0 <= seed < 10, 'Invalid nominal metadata identity')
    kind = NOMINAL_METADATA_KINDS[n][seed]
    summary = Path(nominal) / f'seed_{seed}_summary.json'
    if kind == 'checkpoint_metadata':
        check(not summary.exists(), f'Unexpected nominal JSON for fixed checkpoint mapping: {summary}')
    # Validate all learned tensors and legacy buffer shapes before trusting metadata.
    model = snap.model(Path(nominal) / f'seed_{seed}_nominal.pth', 'scs')
    payload = c.torch.load(_bound_file(snap.manifest, model['checkpoint']),
                           map_location='cpu', weights_only=False)
    check(type(payload['seed']) is int and payload['seed'] == seed,
          'Nominal checkpoint seed mismatch')
    check(type(payload['n_samples']) is int and payload['n_samples'] == n,
          'Nominal checkpoint n_samples mismatch')
    fields = ('seed', 'saa_obj', 'saa_Q10', 'saa_Q1r')
    for key in ('saa_obj', 'saa_Q10'):
        check(type(payload[key]) in (int, float) and np.isfinite(payload[key]),
              f'Invalid nominal primitive: {key}')
    q1r = payload['saa_Q1r']
    check(type(q1r) is list and len(q1r) == 10
          and all(type(v) in (int, float) and np.isfinite(v) for v in q1r),
          'Invalid nominal primitive: saa_Q1r')
    if kind == 'json':
        source = snap.capture(summary)
        record = snap.read(summary)
        check(type(record['seed']) is int and all(record[k] == payload[k] for k in fields),
              'Nominal JSON/checkpoint metadata mismatch')
    else:
        source = model['checkpoint']
        # Never put a state_dict, sparse tensor, or optimizer object in the manifest.
        record = {key: payload[key] for key in ('seed', 'n_samples', 'saa_obj', 'saa_Q10', 'saa_Q1r')}
    return dict(source=source, source_kind=kind, source_sha256=snap.manifest['files'][source]['sha256'],
        summary_path=str(summary.resolve()), summary_status='present' if kind == 'json' else 'missing',
        metadata=record, **model, verified_n_samples=n, verified_R=10,
        validator='c.load_model(old)',
        state_shapes={k: list(v.shape) for k, v in payload['model_state_dict'].items()},
        full_training_time_s=None, full_training_time_status='unavailable; rerun required')


def _freeze_scs(snap, manifest):
    manifest['cached_baselines'], manifest['scs_radii'] = {}, {}
    manifest['nominal_metadata_sources'] = {}
    manifest['historically_nonconverged_scs'] = []
    for n in (50, 200, 800):
        base = c.OLD / f'R10/N{n}'
        sweep, nominal = base / 'rho_sweep_net16_v2', base / 'saainit_net16_v2'
        suffix = '_10k' if n == 800 else ''
        cache_path = sweep / f'sweep_eval_net16_v2_N{n}_R10{suffix}.json'
        cache = snap.read(cache_path)
        sources = sorted(sweep.glob('eval_sweep*.py'))
        check(bool(sources), 'Missing historical evaluation source reference')
        sources += [sweep / f'run_rho_seed_net16_v2_N{n}_R10.py',
                    nominal / f'run_all_net16_v2_N{n}_R10.py']
        for path in sources:
            snap.capture(path)
        knn = validate_knn(cache['benchmarks'])
        sro_path = c.OLD / f'R10/sro_fa/sro_fa_results_N{n}_R10.json'
        sro = snap.read(sro_path)
        check(len(sro) == 17 and sorted(map(float, sro)) == list(SCS_RADII[1:]), 'SRO-FA radius grid mismatch')
        for rows in sro.values():
            check(set(rows) == set(map(str, range(10))), 'SRO-FA seed coverage mismatch')
            check(all(np.isfinite(row['oos']) and len(row['Q1r']) == 10 for row in rows.values()),
                  'Invalid R10 SRO-FA cache')
        manifest['cached_baselines'][str(n)] = dict(cache=str(cache_path), saa_knn=knn,
            knn_k_grid=cache['benchmarks']['knn_k_grid'], saa_knn_by_k=cache['benchmarks']['saa_knn_by_k'],
            knn_selection='per_seed_loo_grid_v3; not full-pipeline LOO',
            historical_runtime=cache.get('runtime'), benchmark_runtime=cache['benchmarks'].get('benchmark_time_s'),
            sro_source=str(sro_path), sro_fa=sro, sro_radii=sorted(map(float, sro)),
            cached=True, strict_certification=False, current_evaluation_time_s=None,
            source_references=[str(p) for p in sources],
            provenance_caveat='Adjacent historical scripts are source references, not verified exact producers '
                'of canonical caches; some use analytical projection or fallbacks. Fresh neural replay '
                'instead requires strict OPTIMAL. Historical caches have no checkpoint-hash certification.',
            timing_note='Original fields preserved; absent historical timings unavailable, rerun required.')
        directories = {}
        for path in sorted(sweep.glob('rho_*')):
            if path.is_dir():
                rho = float(path.name[4:].replace('p', '.'))
                check(rho not in directories, f'Ambiguous SCS radius directory: {rho}')
                directories[rho] = path
        radii = sorted(float(k) for k in cache['sweep'] if 0 <= float(k) <= 1.5)
        check(radii == list(SCS_RADII), 'SCS actual radius grid mismatch')
        manifest['scs_radii'][str(n)] = radii
        nominal_sources = {str(seed): nominal_metadata(snap, n, seed, nominal) for seed in range(10)}
        manifest['nominal_metadata_sources'][str(n)] = nominal_sources
        for rho in radii:
            entry = cache['sweep'][str(rho)]
            check(entry['n_seeds'] == 10 and entry['projection_method'] == 'gurobi_qp'
                  and entry['evaluator_version'] == 'gurobi_qp_final_mp_is_v2'
                  and all(len(entry[k]) == 10 and np.isfinite(entry[k]).all() for k in ('rob_is', 'rob_oos')),
                  'Incomplete historical grid/evaluator metadata')
            for seed in range(10):
                checkpoint, iteration, selector = scs_choice(n, rho, seed, directories, nominal)
                nominal_info = nominal_sources[str(seed)]
                selection = snap.read(selector) if selector else None
                if selection is not None:
                    check(selection['final_mp_iter'] == iteration and selection['seed'] == seed
                          and selection['rho'] == rho and selection['completed'] is True,
                          'SCS selector changed during freeze')
                paths = [checkpoint.parent / f'seed_{seed}_ap_iter{i}.json' for i in range(1, iteration + 1)]
                rng = np.random.get_state()
                try:
                    centers = np.concatenate(c.gen(n, seed), axis=1)  # Training only.
                finally:
                    np.random.set_state(rng)
                pools, membership = scs_pools(centers, rho, paths, snap.read)
                model = snap.model(checkpoint, 'scs')
                payload = c.torch.load(manifest['files'][model['checkpoint']]['copy'],
                                       map_location='cpu', weights_only=False)
                check(payload['seed'] == seed and (iteration == 0 or payload['ccg_iter'] == iteration),
                      f'SCS checkpoint identity mismatch: {checkpoint}')
                if 'rho' in payload:
                    check(payload['rho'] == rho, 'SCS checkpoint radius mismatch')
                native = float(payload['best_loss'])
                check(np.isfinite(native), 'Nonfinite saved SCS training loss')
                inputs = list(dict.fromkeys([model['checkpoint'], nominal_info['source'],
                    nominal_info['checkpoint']] + [str(p) for p in paths]))
                if selector:
                    inputs.append(str(selector))
                historical_converged = selection['converged'] if selection is not None else None
                job_id = f'scs_N{n}_r{rho:g}_s{seed}'
                if historical_converged is False:
                    manifest['historically_nonconverged_scs'].append(dict(id=job_id, Ntotal=n,
                        radius=rho, seed=seed, selector=str(selector), retained=True))
                manifest['jobs'].append(dict(id=job_id, method='scs', Ntotal=n, Ntrain=n, Nval=0,
                    seed=seed, radius=rho, **model, selected_candidate_id=iteration, stopping_candidate_id=None,
                    selector=str(selector) if selector else 'nominal', historically_converged=historical_converged,
                    historical_status='nominal_reference' if selection is None else
                        ('completed_converged' if historical_converged else 'completed_nonconverged_retained'),
                    ap_paths=[str(p) for p in paths], pools=[p.tolist() for p in pools],
                    scs_historical_membership=membership, pool_sha256=pool_digest(pools),
                    sizes=[len(p) for p in pools], centers=centers.tolist(), original_sample_ids=list(range(n)),
                    input_files=inputs, saved_native_finite_pool_mean=native,
                    training_runtime_reference=dict(status='historical_fields_not_certified_full_runtime',
                        nominal_source=nominal_info['source'], nominal=nominal_info['metadata'],
                        nominal_source_kind=nominal_info['source_kind'],
                        nominal_source_sha256=nominal_info['source_sha256'],
                        nominal_summary_path=nominal_info['summary_path'],
                        nominal_summary_status=nominal_info['summary_status'],
                        nominal_full_training_time_s=nominal_info['full_training_time_s'],
                        nominal_full_training_time_status=nominal_info['full_training_time_status'],
                        selector_source=str(selector) if selector else None, ccg=selection,
                        checkpoint_training_time_s=payload.get('training_time_s'),
                        note='Missing full-model times unavailable; rerun required. Cache/resume overhead is not training.')))
                if rho == 0. and seed == 0:
                    # snap.model has already passed c.load_model's exact R10 old
                    # tensor/buffer schema. Bind observed shapes, not a meta default.
                    bound_sources = [cache_path, checkpoint, *sources, REFERENCE, COMPARE,
                                     Path(__file__).resolve(), HERE / 'plot_comparison.py']
                    evidence = dict(schema='scs_checkpoint_centers_R10_v1', Ntotal=n, verified_R=10,
                        validator='c.load_model(old)', job_id=job_id, **model,
                        centers_shape=list(centers.shape),
                        state_shapes={k: list(v.shape) for k, v in payload['model_state_dict'].items()},
                        source_sha256={str(p): manifest['files'][str(p)]['sha256'] for p in bound_sources},
                        scope='Bound nominal SCS checkpoint and training centers; not cache-producer certification.')
                    manifest['cached_baselines'][str(n)]['dimension_evidence'] = evidence
                    manifest['cached_baselines'][str(n)]['cache_protocol'] = validate_cache_protocol(cache, n, manifest)


def freeze(output: Path) -> dict:
    """Freeze all policies before the parent may generate its one shared test."""
    start = time.perf_counter()
    folder = Path(output).resolve() / 'frozen'
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(experiment=EXPERIMENT, status='freezing', start_utc=c.utc(), files={}, jobs=[],
        versions=versions(), membership_tolerance=MEMBERSHIP_TOL, physical_tolerance=c.PHYSICAL_TOL,
        scs_membership_policy=SCS_MEMBERSHIP_POLICY, n_test=10000, test_seed=9999,
        workers=None, threads=1, R=10, green_radii=list(RADII), new_training=False, new_selection=False,
        timing_clock='time.perf_counter', timing_unit='seconds', source_adaptations=c.ADAPTATIONS)
    try:
        snap = _Snapshot(folder, manifest)
        for path in SOURCE_FILES:
            snap.capture(path, c.LOCKS.get(path))
        _freeze_green(snap, manifest)
        _freeze_scs(snap, manifest)
        check(len(manifest['jobs']) == len({j['id'] for j in manifest['jobs']}) == 810,
              'Expected 270 green + 540 SCS fixed-grid policies')
        check(sum(j['method'] == 'green' for j in manifest['jobs']) == 270, 'Green count mismatch')
        observed = {(j['Ntotal'], j['radius'], j['seed']) for j in manifest['historically_nonconverged_scs']}
        check(observed == {(50, .15, 4), (50, .6, 5), (50, .8, 1)}, 'Historical convergence disclosure changed')
        for path, info in manifest['files'].items():
            check(c.sha(path) == c.sha(info['copy']) == info['sha256'], f'Freeze input changed: {path}')
        for sources in manifest['nominal_metadata_sources'].values():
            for info in sources.values():
                if info['source_kind'] == 'checkpoint_metadata':
                    check(not Path(info['summary_path']).exists(),
                          f'Unexpected nominal JSON during freeze: {info["summary_path"]}')
        manifest.update(status='frozen', job_count=len(manifest['jobs']), end_utc=c.utc(),
                        wall_time_s=time.perf_counter() - start)
        return manifest
    except BaseException:
        write_json(folder / 'freeze_failure.json', dict(status='failed', error=traceback.format_exc(),
            start_utc=manifest['start_utc'], end_utc=c.utc(), wall_time_s=time.perf_counter() - start,
            jobs_validated=len(manifest['jobs']), test_generated=False))
        raise