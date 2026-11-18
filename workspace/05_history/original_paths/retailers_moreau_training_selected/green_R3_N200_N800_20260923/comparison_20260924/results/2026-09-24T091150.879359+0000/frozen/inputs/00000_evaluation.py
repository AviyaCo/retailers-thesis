"""Frozen fixed-policy evaluation; no training, selection, or test generation.

Parent API:
  manifest = freeze(output)  # creates output/frozen; never overwrites
  # Bind runner/plot files in manifest['files'] (absolute original/copy, sha256).
  # Only now call c.gen(10000, 9999), save arrays, and set:
  # manifest['test'] = {'n_test': 10000, 'seed': 9999, 'arrays': {
  #   'xi1': {'file': ABS_PATH, 'file_sha256': ..., 'array_sha256': ...,
  #           'dtype': 'float64', 'shape': [10000, 3]}, 'xi2': {...}}}
  # Set manifest['workers'] to the actual spawn concurrency (1..32).
  # Write immutable manifest.json and manifest.sha256 beside it.
  # Spawn initialize_worker, then worker(spec, manifest_path, results_dir).

worker returns and persists a failed record on failure; the parent MUST reject
any status other than completed. Raw solves are journalled even on failure.
freeze failures propagate and persist freeze_failure.json; no test may follow.
Green membership uses its existing strict 1e-9 gate. Historical SCS finite pools
retain exact saved coordinates and disclose every positive box residual; their
membership is observational, not box certification or a relaxed tolerance.
Cached comparator records retain their original certification/timing semantics.
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

HOME = Path('/data/home/aviyac')
HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
SOURCE = CAMPAIGN / 'results/20260923T092253.404783+0000'
COMPARE = HOME / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison/compare.py'
N50 = HOME / 'retailers_moreau_training_selected/N50_R3_train40_val10/stopping_only_min_ap/paper_figure/results'
_loader = importlib.util.spec_from_file_location('_comparison_20260924_reference', COMPARE)
assert _loader is not None and _loader.loader is not None
c = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(c)
np = c.np
read_json, write_json = c.read_json, c.write_json
initialize_worker = c.initialize_worker
MEMBERSHIP_TOL = 1e-9
SCS_MEMBERSHIP_POLICY = 'historical_saved_coordinates_observational_not_box_certified'
RADII = c.RADII


def check(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    a = np.asarray(list(values), dtype=float)
    check(a.ndim == 1 and len(a) > 0 and np.isfinite(a).all(), 'Invalid statistics')
    return dict(n=len(a), count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()))


def versions():
    return dict(python=sys.version, numpy=np.__version__, scipy=c.scipy.__version__,
                torch=c.torch.__version__, gurobi=list(c.gp.gurobi.version()))


def pool_digest(pools):
    """Exact saved controller contract (group identity and order included)."""
    h = hashlib.sha256()
    for i, pool in enumerate(pools):
        h.update(np.asarray([i, len(pool)], dtype='<i8').tobytes())
        h.update(np.asarray(pool, dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def finite_pool_cost(costs, sizes):
    a = np.asarray(costs, dtype=np.float64)
    check(len(sizes) > 0 and all(type(n) is int and n > 0 for n in sizes), 'Empty/invalid pools')
    check(a.ndim == 1 and len(a) == sum(sizes) and np.isfinite(a).all(), 'Invalid pool costs')
    maxima = np.maximum.reduceat(a, np.r_[0, np.cumsum(sizes)[:-1]])
    return float(maxima.mean()), maxima.tolist()


def validate_pools(pools, centers, radius, *, membership_policy='strict'):
    check(membership_policy in ('strict', SCS_MEMBERSHIP_POLICY), 'Unknown membership policy')
    centers = np.asarray(centers, dtype=np.float64)
    check(centers.ndim == 2 and centers.shape[1] == 6 and len(centers) > 0
          and np.isfinite(centers).all() and (centers >= 0).all(), 'Invalid training centers')
    check(np.isfinite(radius) and radius >= 0 and len(pools) == len(centers), 'Pool count/radius mismatch')
    arrays = []
    for i, (pool, center) in enumerate(zip(pools, centers)):
        a = np.asarray(pool, dtype=np.float64)
        check(a.ndim == 2 and a.shape[1] == 6 and len(a) > 0 and np.isfinite(a).all(),
              f'Invalid pool at center {i}')
        check(np.array_equal(a[0], center), f'Pool lacks its original center: {i}')
        violations = np.maximum(np.maximum(np.maximum(0, center - radius) - a,
                                            a - center - radius), 0).max(axis=1)
        bad = np.flatnonzero(violations > MEMBERSHIP_TOL)
        if membership_policy == 'strict':
            check(not len(bad), f'Pool membership failure center={i}, rows={bad.tolist()}, '
                  f'residuals={violations[bad].tolist()}, tolerance={MEMBERSHIP_TOL}')
        arrays.append(a)
    return arrays


def selected_green_pool(record, training, read=read_json):
    """Use selected candidate's pool, NOT the later stopping candidate's pool."""
    selected, stop = record['selected_candidate_id'], record['stopping_candidate_id']
    check(type(selected) is int and type(stop) is int and 0 <= selected <= stop,
          'Invalid selected/stopping candidate IDs')
    candidates = record['iterations']
    check(len(candidates) == stop + 1, 'Incomplete or future candidate prefix')
    ids = training['original_sample_ids']
    for i, a in enumerate(candidates):
        check(a['candidate_id'] == i and a['assessed'] and not a['failed_sample_ids'],
              f'Incomplete green candidate {i}')
        samples = a['samples']
        check([r['sample_id'] for r in samples] == list(range(record['Ntrain']))
              and [r['original_sample_id'] for r in samples] == ids,
              f'Green AP ball coverage mismatch: candidate {i}')
        check(all(r['controller_status'] == 'completed' for r in samples), 'Failed AP sample')
        check(all(r['candidate_sha256'] == a['state_sha256']
                  and r['pool_sha256'] == a['pool_sha256'] for r in samples), 'AP policy/pool mismatch')
    a = candidates[selected]
    for key in ('checkpoint_sha256', 'state_sha256'):
        check(a[key] == record['selected_' + key], f'Selected identity mismatch: {key}')
    path = Path(a['checkpoint']).parent / 'pool.json'
    check(c.sha(path) == a['pool_file_sha256'], f'Selected pool file changed: {path}')
    pool = read(path)
    check(pool['original_sample_ids'] == ids, 'Selected pool training IDs changed')
    pools = validate_pools(pool['pools'], training['centers'], record['radius'])
    check(pool_digest(pools) == pool['pool_sha256'] == a['pool_sha256'], 'Selected pool digest changed')
    return a, path, pools


def validate_assessment(assessment, candidate):
    # The controller writes assessment before _flush refreshes the derived MP
    # total. This affects exactly the two returned-before-stop candidates; the
    # authoritative MP call time remains recorded in both artifacts.
    check(set(assessment) == set(candidate), 'Selected assessment schema mismatch')
    for key in candidate:
        if key == 'mp_total_time_s' and assessment[key] != candidate[key]:
            check(assessment[key] == 0 and candidate[key] == candidate['mp_call']['wall_time_s'],
                  'Unexplained assessment MP timing difference')
        else:
            check(assessment[key] == candidate[key], f'Selected assessment mismatch: {key}')


def scs_choice(n, rho, seed, directories, nominal):
    """Legacy precedence, but missing summaries/selector fields are never defaults."""
    check(rho == 0 or rho in directories or (n == 50 and rho == .4), 'Missing SCS radius directory')
    if rho != 0 and rho in directories:
        summary = read_json(directories[rho] / f'seed_{seed}_summary.json')
        check('final_mp_iter' in summary and type(summary['final_mp_iter']) is int
              and summary['final_mp_iter'] >= 0, 'Missing/invalid SCS final_mp_iter')
        check(summary['seed'] == seed and summary['rho'] == rho and summary['completed'],
              'SCS summary identity/completion mismatch')
    return c.old_choice(rho, seed, directories, nominal)


def scs_pools(centers, radius, paths, read=read_json):
    """Return (exact historical pools, membership disclosure), never certify a box.

    Only AP files 1..selected_MP enter the pool; no post-MP separation. All
    positive residuals are reported, including those below the green threshold.
    The threshold is a reporting comparison, not an SCS acceptance tolerance.
    """
    centers = np.asarray(centers, dtype=np.float64)
    pools = [[row.tolist()] for row in centers]
    validate_pools(pools, centers, radius)
    membership = dict(policy=SCS_MEMBERSHIP_POLICY, box_certified=False,
        coordinates_unchanged=True, reporting_threshold=MEMBERSHIP_TOL,
        saved_points_checked=0, positive_residual_count=0,
        count_above_reporting_threshold=0, max_positive_residual=0., offenders=[])
    for path in paths:
        rows = read(path)  # Missing files are fatal, including holes in a prefix.
        ids = [r['ball_id'] for r in rows]
        missing = sorted(set(range(len(centers))) - set(ids))
        check(all(type(i) is int for i in ids) and sorted(ids) == list(range(len(centers))),
              f'SCS AP coverage failure: {path}; missing={missing}, count={len(ids)}')
        for row in rows:
            i = row['ball_id']
            check(np.array_equal(row['xi1_hat'] + row['xi2_hat'], centers[i]),
                  f'SCS AP center mismatch: {path}, ball={i}')
            check(np.isfinite(row['max_cost']), f'Nonfinite saved AP cost: {path}, ball={i}')
            point = row['worst_xi1'] + row['worst_xi2']
            check(len(point) == 6 and np.isfinite(point).all(), f'Invalid AP point: {path}, ball={i}')
            residual = float(np.maximum(np.maximum(np.maximum(0, centers[i] - radius) - point,
                                                   np.asarray(point) - centers[i] - radius), 0).max())
            membership['saved_points_checked'] += 1
            if residual > 0:
                membership['positive_residual_count'] += 1
                membership['count_above_reporting_threshold'] += int(residual > MEMBERSHIP_TOL)
                membership['max_positive_residual'] = max(membership['max_positive_residual'], residual)
                membership['offenders'].append(dict(path=str(path), ball_id=i,
                    pool_row=len(pools[i]), positive_residual=residual))
            pools[i].append(point)
    return validate_pools(pools, centers, radius, membership_policy=SCS_MEMBERSHIP_POLICY), membership


class _Snapshot:
    def __init__(self, folder, manifest):
        self.folder, self.manifest = folder, manifest
        (folder / 'inputs').mkdir(parents=True, exist_ok=False)

    def capture(self, path, expected=None):
        path = Path(path).resolve()
        key, digest = str(path), c.sha(path)
        check(expected is None or digest == expected, f'Source hash mismatch: {path}')
        files = self.manifest['files']
        if key in files:
            check(files[key]['sha256'] == digest, f'Source changed during freeze: {path}')
        else:
            dest = self.folder / 'inputs' / f'{len(files):05d}_{path.name}'
            with path.open('rb') as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            check(c.sha(dest) == digest == c.sha(path), f'Input changed during copy: {path}')
            dest.chmod(0o444)
            files[key] = dict(sha256=digest, copy=str(dest))
        return key

    def read(self, path, expected=None):
        key = self.capture(path, expected)
        return read_json(self.manifest['files'][key]['copy'])

    def model(self, path, method, expected=None, state=None):
        key = self.capture(path, expected)
        model = c.load_model(self.manifest['files'][key]['copy'], {'green': 'new', 'scs': 'old'}[method])
        digest = c.state_hash(model)
        check(state is None or digest == state, f'Learned state mismatch: {path}')
        return dict(checkpoint=key, checkpoint_sha256=self.manifest['files'][key]['sha256'], state_sha256=digest)


def _freeze_green(snap, manifest):
    summary = snap.read(SOURCE / 'summary.json')
    check(summary['status'] == 'completed' and summary['inputs_unchanged']
          and summary['final_selection_published'] and not summary['test_evaluated'], 'Green sweep incomplete')
    provenance = snap.read(SOURCE / 'provenance.json', summary['provenance_sha256'])
    for path, digest in provenance['source_sha256'].items():
        snap.capture(path, digest)
    fixed = snap.read(SOURCE / 'green_checkpoints_before_validation.json')['policies']
    jobs = summary['phases']['ccg']['jobs']
    key = lambda j: (j['Ntotal'], j['seed'], j['radius'])
    expected = {(n, s, r) for n in (200, 800) for s in range(10) for r in RADII}
    check(len(jobs) == len(fixed) == 180 and {key(j) for j in jobs} == {key(j) for j in fixed} == expected,
          'Green fixed-grid coverage mismatch')
    frozen = {key(j): j for j in fixed}
    setups = {(j['Ntotal'], j['seed']): j for j in summary['phases']['initialization']['jobs']}
    check(len(setups) == 20, 'Missing initialization jobs')
    manifest['saved_green_selections'] = summary['selections']
    check(len(summary['selections']) == 20 and
          {(s['Ntotal'], s['seed']) for s in summary['selections']} == set(setups), 'Missing saved winners')
    for winner in summary['selections']:
        row = frozen[key(winner)]
        for a, b in [('checkpoint', 'selected_checkpoint'), ('checkpoint_sha256', 'selected_checkpoint_sha256'),
                     ('state_sha256', 'selected_state_sha256'), ('candidate_id', 'selected_candidate_id'),
                     ('stopping_candidate_id', 'stopping_candidate_id')]:
            check(winner[a] == row[b], f'Saved winner identity mismatch: {a}')
    for job in sorted(jobs, key=key):
        d = snap.read(job['summary_path'], job['summary_sha256'])
        check(d['complete'] and d['converged'] and d['status'] == 'completed', 'Incomplete green policy')
        for name, value in frozen[key(job)].items():
            check(d[name] == value == job[name], f'Prevalidation identity changed: {name}')
        n, seed, rho = key(job)
        setup = setups[n, seed]
        training = snap.read(setup['training_data_path'], setup['training_data_sha256'])
        split = snap.read(setup['split_path'], setup['split_sha256'])
        ids = training['original_sample_ids']
        arrays = split['arrays']
        check(d['Ntrain'] == 4*n//5 and d['Nval'] == n//5 and len(ids) == d['Ntrain']
              and all(type(i) is int and 0 <= i < n for i in ids) and len(set(ids)) == len(ids),
              'Green training IDs/count mismatch')
        raw = np.concatenate([np.asarray(arrays[k], dtype=np.float64) for k in ('xi1', 'xi2')], axis=1)
        raw_hash = hashlib.sha256(b''.join(np.asarray(arrays[k], dtype='<f8').tobytes()
                                         for k in ('xi1', 'xi2'))).hexdigest()
        check(raw.shape == (n, 6) and raw_hash == training['data_sha256'] == split['raw_data_sha256']
              == setup['data_sha256'] == d['data_sha256'], 'Green raw data hash mismatch')
        check(ids == arrays['train_indices'] and set(ids).isdisjoint(arrays['val_indices'])
              and sorted(ids + arrays['val_indices']) == list(range(n))
              and np.array_equal(raw[ids], training['centers']), 'Green training center/split mismatch')
        a, pool_path, pools = selected_green_pool(d, training, snap.read)
        inputs = [job['summary_path'], setup['training_data_path'], setup['split_path'], str(pool_path),
                  a['checkpoint'], a['ap_reference'], d['selected_checkpoint']]
        for path in inputs:
            snap.capture(path)
        snap.capture(a['checkpoint'], a['checkpoint_sha256'])
        validate_assessment(snap.read(a['ap_reference']), a)
        model = snap.model(d['selected_checkpoint'], 'green', d['selected_checkpoint_sha256'], d['selected_state_sha256'])
        manifest['jobs'].append(dict(id=f'green_N{n}_r{rho:g}_s{seed}', method='green',
            Ntotal=n, Ntrain=d['Ntrain'], Nval=d['Nval'], seed=seed, radius=rho, **model,
            selected_candidate_id=d['selected_candidate_id'], stopping_candidate_id=d['stopping_candidate_id'],
            pool_path=str(pool_path), pool_sha256=a['pool_sha256'], pool_file_sha256=a['pool_file_sha256'],
            sizes=[len(p) for p in pools], centers=training['centers'], original_sample_ids=ids,
            source_summary_path=job['summary_path'], input_files=inputs,
            saved_native_finite_pool_mean=a['restricted_mean'],
            training_runtime_reference=dict(source=job['summary_path'], status='persisted_not_current',
                ccg=job, nominal=d['cached_nominal'], phases='iterations[].ap and iterations[].mp_call')))
    # Keep a source reference, not a sum of concurrent model times.
    manifest['green_training_reference'] = dict(summary=str(SOURCE / 'summary.json'),
        provenance=str(SOURCE / 'provenance.json'), phases=list(summary['phases']))


def _freeze_cached(snap, manifest):
    summary = snap.read(N50 / 'summary.json')
    plot = snap.read(N50 / 'plot_data.json')
    snap.read(N50 / 'frozen_jobs.json', summary['frozen_jobs_sha256'])
    check(summary['status'] == 'completed' and summary['inputs_unchanged'], 'N50 cache incomplete')
    for path, digest in summary['input_sha256'].items():
        snap.capture(path, digest)
    jobs = summary['jobs']
    check(len(jobs) == 90 and {(j['seed'], j['radius']) for j in jobs} ==
          {(s, r) for s in range(10) for r in RADII}, 'N50 cached grid mismatch')
    for row in jobs:
        count = sum(row['sizes'])
        check(row['status'] == 'completed' and row['solves_accepted'] == row['solves_attempted'] == count
              and row['solver_status_counts'] == {'2': count}, 'Incomplete N50 cached IS evaluation')
        for name in ('test_mean', 'finite_pool_mean'):
            check(np.isfinite(row[name]), 'Nonfinite N50 cache')
        idx = plot['green']['radius'].index(row['radius'])
        check(plot['green']['oos_per_seed'][idx][row['seed']] == row['test_mean']
              and plot['green']['is_per_seed'][idx][row['seed']] == row['finite_pool_mean'], 'N50 plot/cache mismatch')
    manifest['cached_green_N50'] = dict(summary=str(N50 / 'summary.json'), plot_data=str(N50 / 'plot_data.json'),
        jobs=jobs, test_arrays=plot['test_arrays'], cached=True, current_evaluation_time_s=None)


def _freeze_scs(snap, manifest):
    manifest['cached_baselines'] = {}
    manifest['scs_radii'] = {}
    for n in (50, 200, 800):
        base = c.OLD / f'R3/N{n}'
        sweep, nominal = base / 'rho_sweep_net16_v2', base / 'saainit_net16_v2'
        suffix = '' if n == 50 else f'_N{n}'
        cache_path = sweep / f'sweep_eval_net16_v2{suffix}_10k.json'
        cache = snap.read(cache_path)
        for path in (sweep / f'eval_sweep_net16_v2{suffix}_10k.py',
                     sweep / f'run_rho_seed_net16_v2{suffix}.py', nominal / f'run_all_net16_v2{suffix}.py'):
            snap.capture(path)
        check(cache['meta']['N_train'] == n and cache['meta']['N_test'] == 10000
              and cache['meta']['seed_test'] == 9999 and cache['meta']['seeds'] == list(range(10)), 'Cache protocol mismatch')
        benchmarks = cache['benchmarks']
        knn = benchmarks['saa_knn']
        check(benchmarks['_knn_k_selection'] == 'per_seed_loo_grid_v3'
              and len(knn['selected_k_by_seed']) == 10
              and all(k in benchmarks['knn_k_grid'] for k in knn['selected_k_by_seed']), 'Unvalidated kNN selector')
        check(len(knn['oos']) == 10 and np.isfinite(knn['oos']).all(), 'Incomplete kNN cache')
        sro_path = base / 'sro_fa' / f'sro_fa_results_N{n}_R3.json'
        sro = snap.read(sro_path)
        manifest['cached_baselines'][str(n)] = dict(cache=str(cache_path), saa_knn=knn,
            historical_runtime=cache.get('runtime'), sro_source=str(sro_path), sro_fa=sro,
            cached=True, strict_certification=False, current_evaluation_time_s=None,
            timing_note='Original fields preserved; absent historical timings unavailable, rerun required.')
        directories = {}
        for p in sorted(sweep.glob('rho_*')):
            if p.is_dir():
                rho = float(p.name[4:].replace('p', '.'))
                check(rho not in directories, f'Ambiguous SCS radius directory: {rho}')
                directories[rho] = p
        radii = sorted(float(k) for k in cache['sweep'] if 0 <= float(k) <= 1.5)
        manifest['scs_radii'][str(n)] = radii
        for rho in radii:
            entry = cache['sweep'][str(rho)]
            check(entry['n_seeds'] == 10 and len(entry['rob_oos']) == 10, 'Incomplete historical grid')
            for seed in range(10):
                checkpoint, iteration, selector = scs_choice(n, rho, seed, directories, nominal)
                nominal_summary = nominal / f'seed_{seed}_summary.json'
                nominal_record = snap.read(nominal_summary)
                check(nominal_record['seed'] == seed, 'Nominal summary seed mismatch')
                selection = snap.read(selector) if selector else None
                paths = [checkpoint.parent / f'seed_{seed}_ap_iter{i}.json' for i in range(1, iteration + 1)]
                rng = np.random.get_state()
                try:
                    centers = np.concatenate(c.gen(n, seed), axis=1)  # Training only; never seed9999.
                finally:
                    np.random.set_state(rng)
                pools, membership = scs_pools(centers, rho, paths, snap.read)
                model = snap.model(checkpoint, 'scs')
                payload = c.torch.load(manifest['files'][model['checkpoint']]['copy'], map_location='cpu', weights_only=False)
                check(payload['seed'] == seed and (iteration == 0 or payload['ccg_iter'] == iteration),
                      f'SCS checkpoint identity mismatch: {checkpoint}')
                if 'rho' in payload:
                    check(payload['rho'] == rho, 'SCS checkpoint radius mismatch')
                native = float(payload['best_loss'])
                check(np.isfinite(native), 'Nonfinite saved SCS training loss')
                inputs = [model['checkpoint'], str(nominal_summary)] + [str(p) for p in paths]
                if selector:
                    inputs.append(str(selector))
                manifest['jobs'].append(dict(id=f'scs_N{n}_r{rho:g}_s{seed}', method='scs',
                    Ntotal=n, Ntrain=n, Nval=0, seed=seed, radius=rho, **model,
                    selected_candidate_id=iteration, stopping_candidate_id=None,
                    selector=str(selector) if selector else ('nominal' if rho == 0 else 'FINAL_ITER_040'),
                    ap_paths=[str(p) for p in paths], pools=[p.tolist() for p in pools],
                    scs_historical_membership=membership,
                    pool_sha256=pool_digest(pools), sizes=[len(p) for p in pools], centers=centers.tolist(),
                    original_sample_ids=list(range(n)), input_files=inputs,
                    saved_native_finite_pool_mean=native,
                    training_runtime_reference=dict(status='historical_fields_not_certified_full_runtime',
                        nominal_source=str(nominal_summary), nominal=nominal_record,
                        selector_source=str(selector) if selector else None, ccg=selection,
                        checkpoint_training_time_s=payload.get('training_time_s'),
                        note='Missing full-model times unavailable; rerun required. Cache/resume overhead is not training.')))


def freeze(output: Path) -> dict:
    """Capture all required artifacts before parent generates the matched test."""
    start = time.perf_counter()
    folder = Path(output).resolve() / 'frozen'
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(experiment='R3_fixed_policy_comparison_20260924', status='freezing',
        start_utc=c.utc(), files={}, jobs=[], versions=versions(), membership_tolerance=MEMBERSHIP_TOL,
        scs_membership_policy=SCS_MEMBERSHIP_POLICY,
        n_test=10000, test_seed=9999, workers=None, threads=1,
        new_training=False, new_selection=False, timing_clock='time.perf_counter', timing_unit='seconds')
    try:
        snap = _Snapshot(folder, manifest)
        for path in (Path(__file__), HERE / 'test_evaluation.py', COMPARE):
            snap.capture(path)
        _freeze_green(snap, manifest)
        _freeze_cached(snap, manifest)
        _freeze_scs(snap, manifest)
        check(len(manifest['jobs']) == 780 and len({j['id'] for j in manifest['jobs']}) == 780,
              'Expected 180 green + 600 SCS fixed-grid policies')
        for path, info in manifest['files'].items():
            check(c.sha(path) == c.sha(info['copy']) == info['sha256'], f'Freeze input changed: {path}')
        manifest.update(status='frozen', job_count=len(manifest['jobs']), end_utc=c.utc(),
                        wall_time_s=time.perf_counter() - start)
        return manifest
    except BaseException:
        write_json(folder / 'freeze_failure.json', dict(status='failed', error=traceback.format_exc(),
            start_utc=manifest['start_utc'], end_utc=c.utc(), wall_time_s=time.perf_counter() - start,
            jobs_validated=len(manifest['jobs']), test_generated=False))
        raise


def _bound_file(manifest, path):
    info = manifest['files'][str(Path(path).resolve())]
    check(c.sha(path) == c.sha(info['copy']) == info['sha256'], f'Frozen input changed: {path}')
    return Path(info['copy'])


def _load_inputs(spec, manifest_path):
    path = Path(manifest_path)
    digest = c.sha(path)
    check(digest == path.with_suffix('.sha256').read_text().strip(), 'Immutable manifest digest changed')
    m = read_json(path)
    check(m['status'] == 'frozen' and m['versions'] == versions(), 'Manifest status/environment changed')
    check(m['membership_tolerance'] == MEMBERSHIP_TOL and m['threads'] == 1, 'Validation contract changed')
    check(type(m['workers']) is int and 1 <= m['workers'] <= 32, 'Invalid worker concurrency')
    matches = [j for j in m['jobs'] if j['id'] == spec['id']]
    check(len(matches) == 1 and matches[0] == spec, 'Per-policy identity differs from manifest')
    # Parent-bound sources are checked as well; unrelated cached data are not reread per policy.
    for source in m['files']:
        if Path(source).suffix == '.py':
            _bound_file(m, source)
    for source in spec['input_files']:
        _bound_file(m, source)
    checkpoint = _bound_file(m, spec['checkpoint'])
    check(c.sha(checkpoint) == spec['checkpoint_sha256'], 'Checkpoint hash changed')
    model = c.load_model(checkpoint, {'green': 'new', 'scs': 'old'}[spec['method']])
    check(c.state_hash(model) == spec['state_sha256'], 'Learned state changed')
    if spec['method'] == 'green':
        pool_file = _bound_file(m, spec['pool_path'])
        check(c.sha(pool_file) == spec['pool_file_sha256'], 'Pool file changed')
        pool = read_json(pool_file)
        check(pool['original_sample_ids'] == spec['original_sample_ids'], 'Pool IDs changed')
        pools = pool['pools']
    else:
        check(spec['method'] == 'scs' and m['scs_membership_policy'] == SCS_MEMBERSHIP_POLICY,
              'Historical SCS membership contract changed')
        pools, membership = scs_pools(spec['centers'], spec['radius'], spec['ap_paths'],
                                     lambda p: read_json(_bound_file(m, p)))
        check(membership == spec['scs_historical_membership'], 'SCS membership disclosure changed')
        check(pool_digest(pools) == pool_digest(spec['pools']), 'SCS reconstructed pool changed')
    pools = validate_pools(pools, spec['centers'], spec['radius'],
        membership_policy='strict' if spec['method'] == 'green' else SCS_MEMBERSHIP_POLICY)
    check(len(pools) == spec['Ntrain'] and [len(p) for p in pools] == spec['sizes']
          and pool_digest(pools) == spec['pool_sha256'], 'Pool size/digest changed')
    test = m['test']
    check(test['n_test'] == m['n_test'] == 10000 and test['seed'] == m['test_seed'] == 9999, 'Unmatched test protocol')
    arrays = []
    for key in ('xi1', 'xi2'):
        info = test['arrays'][key]
        file = Path(info['file'])
        check(file.is_absolute() and c.sha(file) == info['file_sha256'], f'Test file changed: {key}')
        a = np.load(file, mmap_mode='r', allow_pickle=False)
        check(list(a.shape) == info['shape'] == [10000, 3] and str(a.dtype) == info['dtype'] == 'float64'
              and np.isfinite(a).all() and (a >= 0).all(), f'Invalid test array: {key}')
        array_hash = hashlib.sha256(a.tobytes(order='C')).hexdigest()
        check(array_hash == info['array_sha256'] == m['cached_green_N50']['test_arrays'][key]['array_sha256'],
              f'Unmatched historical seed9999 test array: {key}')
        arrays.append(a)
    return m, digest, model, arrays, pools


def _evaluate(model, xi1, xi2, name, record, folder, sizes=None):
    phases = record['phases']
    with c.phase(phases, name + '_targets'):
        z, q10, q1r = model.targets(xi1)
    counts = record['status_counts']
    with c.phase(phases, name + '_projection') as projection:
        projection.update(sample_count=len(xi1), solves_attempted=0, solves_accepted=0,
                          iterations=0., barrier_iterations=0, solver_threads=1, status_counts={})
        q2 = np.empty((len(xi1), 3), dtype=np.float64)
        with (folder / f'{name}_solves.jsonl').open('x', buffering=1) as journal, c.gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            with c.gp.Model(env=env) as settings:
                projection['solver_settings'] = {k: settings.getParamInfo(k)[2] for k in
                    ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'Method', 'Threads')}
            center_ids = np.repeat(np.arange(len(sizes)), sizes) if sizes is not None else None
            for i in range(len(xi1)):
                diag = dict(row=i, phase=name, start_utc=c.utc(), status=None, iterations=None,
                            barrier_iterations=None, accepted=False)
                if center_ids is not None:
                    diag.update(center_id=int(center_ids[i]), original_sample_id=record['original_sample_ids'][center_ids[i]])
                t = time.perf_counter()
                record['solves_attempted'] += 1
                projection['solves_attempted'] += 1
                try:
                    q2[i] = c.strict_project(z[i], q10, env, diag)
                    check(diag['status'] == c.GRB.OPTIMAL and diag['accepted'] and np.isfinite(q2[i]).all(),
                          'Strict projection did not return a finite OPTIMAL result')
                    projection['solves_accepted'] += 1
                    record['solves_accepted'] += 1
                except BaseException as exc:
                    diag.update(accepted=False, error=f'{type(exc).__name__}: {exc}')
                    record['failed_projection'] = diag
                    raise
                finally:
                    diag.update(end_utc=c.utc(), controller_wall_time_s=time.perf_counter() - t)
                    counts[str(diag['status'])] = counts.get(str(diag['status']), 0) + 1
                    local_counts = projection['status_counts']
                    local_counts[str(diag['status'])] = local_counts.get(str(diag['status']), 0) + 1
                    for key in ('iterations', 'barrier_iterations'):
                        projection[key] += diag[key] or 0
                        record[key] += diag[key] or 0
                    journal.write(json.dumps(diag, allow_nan=False) + '\n')
    with c.phase(phases, name + '_cost'):
        costs = c.scenario_costs(q10, q1r, q2, xi1, xi2)
        check(costs.shape == (len(xi1),) and np.isfinite(costs).all(), 'Nonfinite/invalid scenario costs')
        if sizes is None:
            record['mean_oos'] = float(costs.mean())
        else:
            record['finite_pool_mean'], record['per_center_maxima'] = finite_pool_cost(costs, sizes)
            record['difference_from_saved_native'] = record['finite_pool_mean'] - record['saved_native_finite_pool_mean']


def worker(spec: dict, manifest_path: str, output: str) -> dict:
    start = time.perf_counter()
    check(Path(spec['id']).name == spec['id'], 'Invalid policy ID')
    folder = Path(output) / spec['id']
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(**spec, experiment='R3_fixed_policy_comparison_20260924', status='running', start_utc=c.utc(),
        phases={}, status_counts={}, solves_attempted=0, solves_accepted=0, iterations=0., barrier_iterations=0,
        mean_oos=None, finite_pool_mean=None, cached=False, solver_threads=1,
        torch_threads=c.torch.get_num_threads(), torch_interop_threads=c.torch.get_num_interop_threads(),
        pid=os.getpid(), environment=dict(host=platform.node(), platform=platform.platform()),
        timing_clock='time.perf_counter', timing_unit='seconds')
    write_json(folder / 'result.json', record)
    try:
        with c.phase(record['phases'], 'input_loading'):
            check(record['torch_threads'] == record['torch_interop_threads'] == 1, 'Worker not initialized')
            manifest, digest, model, arrays, pools = _load_inputs(spec, manifest_path)
            record.update(manifest_sha256=digest, test=manifest['test'], worker_count=manifest['workers'])
        _evaluate(model, *arrays, 'oos', record, folder)
        points = np.concatenate(pools)
        _evaluate(model, points[:, :3], points[:, 3:], 'is', record, folder, spec['sizes'])
        check(c.state_hash(model) == spec['state_sha256'], 'Evaluation changed learned state')
        check(c.sha(manifest_path) == digest, 'Manifest changed during evaluation')
        for source in spec['input_files']:
            _bound_file(manifest, source)
        for info in manifest['test']['arrays'].values():
            check(c.sha(info['file']) == info['file_sha256'], 'Test file changed during evaluation')
        check(record['solves_accepted'] == record['solves_attempted'] == 10000 + sum(spec['sizes'])
              and record['status_counts'] == {'2': record['solves_accepted']}, 'Incomplete strict evaluation')
        record['status'] = 'completed'
    except BaseException:
        record.update(status='failed', error=traceback.format_exc(), mean_oos=None, finite_pool_mean=None)
    finally:
        record.update(end_utc=c.utc(), wall_time_s=time.perf_counter() - start)
        record['wall_time_human'] = c.human(record['wall_time_s'])
        write_json(folder / 'result.json', record)
    return record