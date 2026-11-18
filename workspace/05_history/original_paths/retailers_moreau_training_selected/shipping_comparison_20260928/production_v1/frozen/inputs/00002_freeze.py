"""Read-only source validation and immutable input capture; no evaluation/solves.

Parent contract: freeze(output), run unit tests, verify(manifest), then (and only
then) preflight/test generation. Set workers in the parent before sealing its
manifest. All local *.py and README.md must exist before freeze. This module
does not run freeze on import or implement a campaign runner.

R3 approval: retain unchanged historical pools and disclose ALL positive box
residuals. Green remains strict at 1e-9. Hash-only historical dependencies are
verified separately, not copied en masse; use this module's verify as well as
common.verify before proceeding. No held-out costs enter policy/grid selection.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import time
import traceback

import common as c

HOME, ROOT = c.HOME, c.ROOT
LIVE = HOME / 'retailers_moreau_training_selected/shipping_live_green_20260926/production_v1'
CONTINUED = HOME / 'retailers_moreau_training_selected/shipping_advisory_continue_20260928/continuation_v1'
LEGACY = HOME / 'retailers_with_shipping'
MEMBERSHIP_TOL = 1e-9
LEGACY_POLICY = 'historical_saved_coordinates_observational_not_box_certified'
REPLACEMENTS = {(800, 7, .2), (800, 9, .4), (200, 5, 1.5), (800, 6, .7), (50, 7, 1.5)}
check, np = c.check, c.np

# Reuse the already tested, exact numerical pool contract, without importing its
# evaluator or starting any historical experiment. Both references are captured.
_namespace = dict(np=np, Path=Path, check=check, c=c.c, read_json=c.read,
                  MEMBERSHIP_TOL=MEMBERSHIP_TOL, SCS_MEMBERSHIP_POLICY=LEGACY_POLICY,
                  pool_digest=c.pool_digest)
for _name in ('validate_pools', 'validate_assessment', 'selected_green_pool', 'scs_pools'):
    _namespace[_name] = c.definition(c.POOL_REFERENCE, _name, _namespace)
validate_pools = _namespace['validate_pools']
validate_assessment = _namespace['validate_assessment']
selected_green_pool = _namespace['selected_green_pool']
legacy_pools = _namespace['scs_pools']


class Snapshot:
    """Exclusive, read-only copies; never change originals or replace a freeze."""

    def __init__(self, folder, manifest):
        self.folder, self.manifest = Path(folder).resolve(), manifest
        (self.folder / 'inputs').mkdir(exist_ok=False)

    def capture(self, path, expected=None):
        path = Path(path).resolve()
        key, digest = str(path), c.sha(path)
        check(expected is None or digest == expected, f'Source hash mismatch: {path}')
        files = self.manifest['files']
        if key in files:
            info = files[key]
            check(info['sha256'] == digest == c.sha(info['copy']), f'Changed captured input: {path}')
        else:
            dest = self.folder / 'inputs' / f'{len(files):05d}_{path.name}'
            with path.open('rb') as src, dest.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            check(c.sha(dest) == digest == c.sha(path), f'Changed input during copy: {path}')
            dest.chmod(0o444)
            files[key] = dict(copy=str(dest), sha256=digest)
        return key

    def read(self, path, expected=None):
        key = self.capture(path, expected)
        return c.read(self.manifest['files'][key]['copy'])

    def model(self, path, method, expected=None, state=None):
        key = self.capture(path, expected)
        model = c.load_model(self.manifest['files'][key]['copy'], method)
        digest = c.c.state_hash(model)
        check(state is None or digest == state, f'Learned state hash mismatch: {path}')
        return dict(checkpoint=key, checkpoint_sha256=self.manifest['files'][key]['sha256'],
                    state_sha256=digest)


def _key(row):
    return row['Ntotal'], row['seed'], row['radius']


def green_cohort(live, continued):
    """Replace exactly the approved five; never filter failures into a cohort."""
    jobs, replacements = live['phases']['ccg']['jobs'], continued['results']
    expected = {(n, s, r) for n in c.SIZES for s in c.SEEDS for r in c.RADII}
    check(len(jobs) == 270 and {_key(j) for j in jobs} == expected, 'Incomplete green grid')
    check(all(j['status'] in ('completed', 'failed') for j in jobs), 'Unknown green status')
    failed = {_key(j) for j in jobs if j['status'] == 'failed'}
    check(failed == REPLACEMENTS and sum(j['status'] == 'completed' for j in jobs) == 265,
          'Original green failure identities changed')
    check(continued['status'] == 'completed' and continued['all_five_converged']
          and len(replacements) == 5 and {_key(j) for j in replacements} == failed
          and all(j['status'] == 'completed' for j in replacements), 'Invalid five-job continuation')
    check(live['test_evaluated'] is False and continued['test_evaluated'] is False,
          'Unexpected prior test evaluation')
    mapping = {_key(j): j for j in replacements}
    return [mapping[_key(j)] if _key(j) in failed else j for j in sorted(jobs, key=_key)]


def validate_prefix(record, training):
    """Validate the assessed prefix and training-only earliest-minimum selector."""
    selected, stop = record['selected_candidate_id'], record['stopping_candidate_id']
    rows = record['iterations']
    check(type(selected) is int and type(stop) is int and 0 <= selected <= stop
          and len(rows) == stop + 1, 'Invalid complete candidate prefix')
    ids, centers = training['original_sample_ids'], np.asarray(training['centers'], dtype=float)
    scores = []
    for i, a in enumerate(rows):
        check(type(a['candidate_id']) is int and a['candidate_id'] == i and a['assessed'] is True
              and not a['failed_sample_ids'], f'Unassessed candidate {i}')
        check(all(a[k] == record[k] for k in ('Ntotal', 'Ntrain', 'seed', 'radius')),
              'Candidate configuration mismatch')
        samples = a['samples']
        check([s['sample_id'] for s in samples] == list(range(len(ids)))
              and [s['original_sample_id'] for s in samples] == ids, 'Incomplete AP ball coverage')
        costs = []
        for k, s in enumerate(samples):
            check(s['controller_status'] == 'completed' and s['candidate_id'] == i
                  and s['candidate_sha256'] == a['state_sha256']
                  and s['checkpoint_sha256'] == a['checkpoint_sha256']
                  and s['pool_sha256'] == a['pool_sha256'], 'AP candidate binding mismatch')
            check(np.array_equal(s['center1'] + s['center2'], centers[k]), 'Green AP center mismatch')
            point = s['incumbent']['worst1'] + s['incumbent']['worst2']
            validate_pools([[centers[k].tolist(), point]], centers[k:k+1], record['radius'])
            costs.append(s['incumbent']['cost'])
        score, restricted = a['ap_incumbent_mean'], a['restricted_mean']
        check(np.isfinite(costs).all() and np.isfinite([score, restricted]).all()
              and float(np.mean(costs)) == score, 'Invalid AP mean')
        gap = abs(score - restricted) / max(abs(restricted), 1e-9)
        check(a['green_gap'] == gap and a['green_pass'] == (gap < .01)
              and a['green_pass'] == (i == stop), 'Not the first strict green stop')
        scores.append(score)
    check(selected == min(range(len(scores)), key=scores.__getitem__), 'Not earliest minimum mean AP')
    check(record['selected_ap_incumbent_mean'] == scores[selected]
          and record['selected_restricted_mean'] == rows[selected]['restricted_mean']
          and record['selected_ap_reference'] == rows[selected]['ap_reference'], 'Selected summary mismatch')


_TIMING_KEYS = ('status', 'started_at_utc', 'ended_at_utc', 'wall_time_s', 'total_time_s',
    'current_invocation_elapsed_time_s', 'training_time_s', 'training_time', 'seed_time_s',
    'epochs_attempted', 'epochs_evaluated', 'optimizer_steps', 'mp_count', 'ccg_iters',
    'ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s', 'timing_scope',
    'ccg_workers', 'ap_workers', 'ap_workers_per_job', 'torch_threads', 'gurobi_threads',
    'torch_interop_threads', 'workers', 'environment', 'iter', 'ap_time_s', 'mp_time_s',
    'ap_status', 'mp_status', 'ap_time_current_invocation_s', 'mp_time_current_invocation_s',
    'ap_model_time_s', 'mp_model_time_s', 'n_samples', 'batch_size', 'mp_batch_size')


def compact_timing(row):
    return {k: row[k] for k in _TIMING_KEYS if k in row}


def _green_runtime(record, source):
    return dict(source=str(source), status='persisted_not_current',
        current_invocation=compact_timing(record), nominal=record['cached_nominal'],
        cached={k: record[k] for k in ('cached_prefix_timing', 'cached_prior_job_timing',
            'cached_failed_assessment_timing', 'logical_prefix_totals', 'cached_timing_scope') if k in record},
        iterations=[dict(candidate_id=a['candidate_id'], ap=compact_timing(a['ap']),
            mp_call=compact_timing(a.get('mp_call', {})), mp=compact_timing(a.get('mp', {})),
            evaluation=compact_timing(a['evaluation'])) for a in record['iterations']],
        note='Persisted times only; cached phase sums are not current elapsed time. '
             'Missing historical timings unavailable; rerun required.')


def _historical_bindings(snap, live, continued):
    bindings = snap.manifest['historical_bindings']
    for folder, summary in ((LIVE, live), (CONTINUED, continued)):
        source = snap.read(folder / 'provenance.json', summary.get('provenance_sha256'))
        for kind in ('source_sha256', 'input_sha256'):
            check(isinstance(source[kind], dict) and source[kind], 'Empty historical provenance')
            for name, digest in source[kind].items():
                check(Path(name).is_absolute(), 'Historical binding must be absolute')
                name = str(Path(name).resolve())
                if name in bindings:
                    check(bindings[name] == digest, f'Conflicting historical binding: {name}')
                else:
                    check(c.sha(name) == digest, f'Changed historical binding: {name}')
                    bindings[name] = digest
    snap.manifest['historical_binding_note'] = (
        'Every source/input hash revalidated read-only; only consumed artifacts copied. '
        'These bindings do not certify absent historical cache checkpoint hashes.')


def _freeze_green(snap, live, continued):
    for job in green_cohort(live, continued):
        source = job['summary_path']
        record = snap.read(source, job['summary_sha256'])
        check(record['status'] == 'completed' and record['complete'] and record['converged'],
              'Incomplete green policy')
        for key in ('Ntotal', 'Ntrain', 'Nval', 'seed', 'radius', 'selected_candidate_id',
                    'stopping_candidate_id', 'selected_checkpoint', 'selected_checkpoint_sha256',
                    'selected_state_sha256', 'selected_ap_reference'):
            check(record[key] == job[key], f'Green job/summary mismatch: {key}')
        n, seed, rho = _key(record)
        inputs = [snap.capture(source)]
        training_path = Path(source).parent / 'inputs/training_data.json'
        training = snap.read(training_path, record['training_data_sha256'])
        inputs.append(str(training_path.resolve()))
        ids = training['original_sample_ids']
        check(record['Ntrain'] == 4*n//5 and record['Nval'] == n//5 and len(ids) == 4*n//5
              and len(set(ids)) == len(ids) and all(type(i) is int and 0 <= i < n for i in ids)
              and training['data_sha256'] == record['data_sha256'], 'Green training identity mismatch')
        raw1, raw2 = _training_arrays(n, seed)
        raw = np.concatenate([raw1, raw2], axis=1)
        raw_hash = hashlib.sha256(b''.join(np.asarray(x, dtype='<f8').tobytes()
                                         for x in (raw1, raw2))).hexdigest()
        check(raw_hash == training['data_sha256'] and np.array_equal(raw[ids], training['centers']),
              'Green original training centers mismatch')
        validate_prefix(record, training)
        for a in record['iterations']:
            validate_assessment(snap.read(a['ap_reference']), a)
            inputs.append(snap.capture(a['ap_reference']))
        selected, pool_path, pools = selected_green_pool(record, training, snap.read)
        inputs += [snap.capture(pool_path, selected['pool_file_sha256']),
                   snap.capture(selected['checkpoint'], selected['checkpoint_sha256'])]
        # Both candidate and published copy must have identical checkpoint bytes.
        model = snap.model(record['selected_checkpoint'], 'green',
                           record['selected_checkpoint_sha256'], record['selected_state_sha256'])
        inputs.append(model['checkpoint'])
        snap.manifest['jobs'].append(dict(id=_id('green', n, rho, seed), method='green',
            Ntotal=n, Ntrain=record['Ntrain'], Nval=record['Nval'], seed=seed, radius=rho, **model,
            pools=[p.tolist() for p in pools], centers=training['centers'], sizes=[len(p) for p in pools],
            original_sample_ids=ids, pool_sha256=c.pool_digest(pools), pool_path=str(pool_path.resolve()),
            selected_candidate_id=record['selected_candidate_id'],
            stopping_candidate_id=record['stopping_candidate_id'], input_files=sorted(set(inputs)),
            saved_native_finite_pool_mean=selected['restricted_mean'], legacy_membership=None,
            source_summary_path=source, training_runtime_reference=_green_runtime(record, source)))


def _training_arrays(n, seed):
    check(n in c.SIZES and seed in c.SEEDS, 'Only original training samples may be generated')
    state = np.random.get_state()
    try:
        return c.c.gen(n, seed)
    finally:
        np.random.set_state(state)


def _id(method, n, radius, seed):
    return f'{method}_N{n}_r{radius:g}_s{seed}'.replace('.', 'p')


def legacy_grid(sweep, cache):
    directories = {}
    for path in sorted(Path(sweep).glob('rho_*')):
        if not path.is_dir():
            continue
        rho = float(path.name[4:].replace('p', '.'))
        check(np.isfinite(rho) and rho > 0 and rho not in directories, 'Invalid/duplicate legacy radius')
        directories[rho] = path
    radii = sorted(r for r in directories if r <= 1.5)
    cache_radii = [float(k) for k in cache['sweep']]
    check(len(cache_radii) == len(set(cache_radii)) and np.isfinite(cache_radii).all(),
          'Invalid/duplicate cached radii')
    check(len(radii) == 15 and radii[-1] == 1.5
          and radii == sorted(r for r in cache_radii if 0 < r <= 1.5), 'Historical radius grid mismatch')
    for rho in radii:
        entry = next(v for k, v in cache['sweep'].items() if float(k) == rho)
        check(entry['n_seeds'] == 10, 'Incomplete historical cache metadata')
        names = {p.name for p in directories[rho].glob('seed_*_summary.json')}
        check(names == {f'seed_{s}_summary.json' for s in c.SEEDS}, 'Legacy seed coverage mismatch')
    return [0.] + radii, directories


def legacy_choice(snap, n, seed, rho, base, directories):
    nominal = Path(base) / f'seed_{seed}_nominal.pth'
    if rho == 0:
        return nominal, 0, None, None
    path = directories[rho] / f'seed_{seed}_summary.json'
    record = snap.read(path)
    check(record['seed'] == seed and record['rho'] == rho and record['completed'] is True,
          'Incomplete/mismatched historical summary')
    iteration = record['final_mp_iter']  # Mandatory selector, never best test loss or a default.
    check(type(iteration) is int and iteration >= 0, 'Invalid final_mp_iter')
    if 'n_samples' in record:
        check(record['n_samples'] == n, 'Historical sample count mismatch')
    checkpoint = nominal if iteration == 0 else directories[rho] / f'seed_{seed}_mp_iter{iteration}.pth'
    return checkpoint, iteration, path, record


def _cache_checkpoint(cache, rho, seed, checkpoint, iteration, record, digest):
    provenance = cache.get('checkpoint_provenance')
    if provenance is None:
        return dict(status='historical_byte_identity_unavailable', checkpoint_hash_persisted=False)
    if rho == 0:  # Corrected cache has positive-radius provenance only.
        return dict(status='nominal_not_in_positive_radius_cache', checkpoint_hash_persisted=False)
    rows = next(v for k, v in provenance.items() if float(k) == rho)
    check(set(rows) == {str(s) for s in c.SEEDS}, 'Incomplete cache checkpoint provenance')
    row = rows[str(seed)]
    path = Path(row['checkpoint'])
    path = path if path.is_absolute() else LEGACY / path
    check(path.resolve() == checkpoint.resolve() and row['final_mp_iter'] == iteration
          and row['best_ap_iter'] == record['best_ap_iter'], 'Cache checkpoint selector mismatch')
    saved_hash = row.get('checkpoint_sha256')
    check(saved_hash is None or saved_hash == digest, 'Cache checkpoint bytes changed')
    return dict(status='selector_metadata_verified' if saved_hash is None else 'checkpoint_hash_verified',
                checkpoint_hash_persisted=saved_hash is not None,
                note='Path/iteration agreement alone is not historical byte identity.')


def _freeze_legacy(snap):
    manifest = snap.manifest
    for n in c.SIZES:
        base = LEGACY / f'N{n}'
        sweep = base / ('rho_sweep_minibatch' if n == 800 else 'rho_sweep')
        paths = dict(cache=sweep / 'sweep_eval_shipping_10k.json',
                     sro=base / 'sro_fa' / f'sro_fa_shipping_results_N{n}_R3.json',
                     conditional=base / 'saa_cond_mip_1m_shipping_exact.json')
        manifest['baseline_paths'][str(n)] = {k: snap.capture(p) for k, p in paths.items()}
        cache = snap.read(paths['cache'])
        meta = cache['meta']
        check(meta['N_train'] == n and meta['N_test'] == 10000 and meta['seed_test'] == 9999
              and meta['seeds'] == c.SEEDS, 'Historical cache protocol mismatch')
        if n == 800:
            check(meta['experiment'] == 'fixed_shipping_N800_corrected_minibatch'
                  and meta['sweep_source'] == 'N800/rho_sweep_minibatch', 'Wrong N800 cache')
        radii, directories = legacy_grid(sweep, cache)
        manifest['legacy_radii'][str(n)] = radii
        for seed in c.SEEDS:
            centers = np.concatenate(_training_arrays(n, seed), axis=1)
            nominal_source = base / f'seed_{seed}_summary.json'
            nominal_record = snap.read(nominal_source)
            for rho in radii:
                checkpoint, iteration, selector, record = legacy_choice(
                    snap, n, seed, rho, base, directories)
                ap_paths = [directories[rho] / f'seed_{seed}_ap_iter{i}.json'
                            for i in range(1, iteration + 1)]
                pools, membership = legacy_pools(centers, rho, ap_paths, snap.read)
                model = snap.model(checkpoint, 'legacy')
                payload = c.torch.load(manifest['files'][model['checkpoint']]['copy'],
                                       map_location='cpu', weights_only=False)
                check(payload['seed'] == seed, 'Legacy checkpoint seed mismatch')
                if iteration:
                    check(payload['ccg_iter'] == iteration and payload['rho'] == rho,
                          'Legacy checkpoint iteration/radius mismatch')
                else:
                    check(payload['n_samples'] == n, 'Nominal checkpoint sample count mismatch')
                cache_binding = _cache_checkpoint(cache, rho, seed, checkpoint, iteration,
                                                   record, model['checkpoint_sha256'])
                inputs = [checkpoint, nominal_source, paths['cache'], *ap_paths]
                if selector is not None:
                    inputs.append(selector)
                # best_loss is smooth shipping loss, not native hard finite-pool loss.
                native = payload.get('hard_loss') if iteration == 0 else None
                check(native is None or np.isfinite(native), 'Nonfinite saved native hard loss')
                runtime = dict(status='historical_fields_not_certified_full_runtime',
                    nominal_source=str(nominal_source), nominal=compact_timing(nominal_record),
                    selector_source=str(selector) if selector else None,
                    ccg=compact_timing(record or {}), checkpoint=compact_timing(payload),
                    iterations=[compact_timing(a) for a in (record or {}).get('ccg_trajectory', [])],
                    note='Saved zeros retained, not proof of zero work. Missing full-model/phase times '
                         'unavailable; rerun required. Cache/resume times are not full training times.')
                manifest['jobs'].append(dict(id=_id('legacy', n, rho, seed), method='legacy',
                    Ntotal=n, Ntrain=n, Nval=0, seed=seed, radius=rho, **model,
                    pools=[p.tolist() for p in pools], centers=centers.tolist(), sizes=[len(p) for p in pools],
                    pool_sha256=c.pool_digest(pools), original_sample_ids=list(range(n)),
                    input_files=sorted({snap.capture(p) for p in inputs}), selected_candidate_id=iteration,
                    stopping_candidate_id=None, saved_native_finite_pool_mean=native,
                    training_runtime_reference=runtime, legacy_membership=membership,
                    historical_cache_binding=cache_binding))


def verify(manifest):
    """Recheck both copied inputs and hash-only historical dependencies."""
    c.verify(manifest)
    for source, digest in manifest['historical_bindings'].items():
        check(c.sha(source) == digest, f'Changed historical binding: {source}')


def freeze(output) -> dict:
    """Return the 750-job manifest; exclusively create output/frozen, no test data."""
    start, started = time.perf_counter(), c.utc()
    folder = Path(output).resolve() / 'frozen'
    folder.mkdir(parents=True, exist_ok=False)
    manifest = dict(experiment='shipping_comparison_20260928_R3', status='freezing', files={}, jobs=[],
        historical_bindings={}, baseline_paths={}, legacy_radii={},
        green_radii=list(c.RADII), seeds=list(c.SEEDS), n_test=10000, test_seed=9999, threads=1,
        workers=None, membership_tolerance=MEMBERSHIP_TOL, legacy_membership_policy=LEGACY_POLICY,
        legacy_membership_approval='R3: retain unchanged historical pools and disclose residuals',
        new_training=False, new_selection=False, test_generated=False,
        timing=dict(start_utc=started, clock='time.perf_counter', unit='seconds', status='running'))
    try:
        manifest['versions'] = c.versions()
        snap = Snapshot(folder, manifest)
        for path in sorted(ROOT.glob('*.py')) + [ROOT / 'README.md', c.REFERENCE, c.POOL_REFERENCE, c.RESOURCE_REFERENCE]:
            snap.capture(path)
        live, continued = snap.read(LIVE / 'summary.json'), snap.read(CONTINUED / 'summary.json')
        _historical_bindings(snap, live, continued)
        _freeze_green(snap, live, continued)
        _freeze_legacy(snap)
        expected = {(m, n, s, r) for n in c.SIZES for s in c.SEEDS for m in ('green', 'legacy')
                    for r in (c.RADII if m == 'green' else manifest['legacy_radii'][str(n)])}
        jobs = manifest['jobs']
        check(len(jobs) == len({j['id'] for j in jobs}) == len(expected) == 750
              and {(j['method'], j['Ntotal'], j['seed'], j['radius']) for j in jobs} == expected,
              'Expected exactly 270 green + 480 historical jobs')
        verify(manifest)
        manifest.update(status='frozen', job_count=len(jobs))
        manifest['timing'].update(status='completed', end_utc=c.utc(), wall_time_s=time.perf_counter()-start)
        # Raw freeze timing and jobs are durable even before the parent seals its
        # own test/concurrency manifest; this file is not a test-generation permit.
        c.write(folder / 'freeze_manifest.json', manifest)
        (folder / 'freeze_manifest.json').chmod(0o444)
        return manifest
    except BaseException:
        manifest['timing'].update(status='failed', end_utc=c.utc(), wall_time_s=time.perf_counter()-start)
        c.write(folder / 'freeze_failure.json', dict(status='failed', error=traceback.format_exc(),
            timing=manifest['timing'], jobs_validated=len(manifest['jobs']), test_generated=False))
        raise