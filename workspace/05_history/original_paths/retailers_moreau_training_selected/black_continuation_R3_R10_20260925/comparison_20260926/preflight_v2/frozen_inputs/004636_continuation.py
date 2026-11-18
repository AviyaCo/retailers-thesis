"""Read-only GREEN-prefix continuation; no held-out access or nominal training.

Public API: initialize(R, manifest), load_job(job, manifest),
verify_job(job, manifest, output_folder), preflight_job(job, manifest, folder),
run_job(job, manifest_path, output_folder). Manifest arguments accept a path or
a decoded mapping except run_job, whose manifest_path must be a file path.
Use a fresh process per R (and keep unit tests in separate processes).

Manifest schema: source_sha256 maps absolute paths to hashes (including this
controller); sourceconfig[str(R)] contains the ORIGINAL campaign's protocol and
source_sha256. Every source map supplied anywhere in the manifest is enforced.
workers is a positive integer or {'ccg': positive integer}. CLI: job --manifest
PATH --jobid ID; manifest.jobs is a list of jobs with jobid and output_folder.
Job fields: R, Ntotal, seed, radius, nominal_record (actual winner, with Ntotal),
source_summary_path/source_summary_sha256 (both None only for new R10 rho=2),
input_sha256 (absolute ORIGINAL paths, complete before worker launch).

Required inputs are train JSON, nominal checkpoint, source summary, every prefix
checkpoint, assessment, pool, MP controller_training.json, training.json,
selected.pth and MP pool.json. Samples are read from the summary, not per-center
files. Assessment/summary equality permits ONLY mp_total_time_s normalization:
the original _flush derives it from mp_call.wall_time_s *after* assessment writes.
No original _flush or controller is called. Historical AP/MP/evaluation/wall
times remain separate from this invocation and nominal historical time.

Preflight is explicitly disposable: one real center AP at the original radius,
then two epochs of the native MP on a private checkpoint/pool. It never produces
a selected production policy. These functions do no work merely on import.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
CAMPAIGNS = {
    3: ROOT.parent / 'green_R3_N200_N800_20260923',
    10: ROOT.parent / 'green_R10_N50_N200_N800_20260924',
}
PROTOCOL = dict(
    experiment='black_continuation_R3_R10_20260925',
    stop='all max(0,bound-pool)/max(1,abs(bound),abs(pool)) <= .01',
    selection='earliest minimum AP incumbent mean in complete prefix including stop; only on convergence',
    max_mp=40, max_epochs=3000, patience=300, separation_tolerance=.01,
    insertion_tolerance=1e-4, pool_replay_tolerance=1e-6,
    resume='saved last assessed STOP, not returned minimum-AP checkpoint',
    nominal_retrain=False, heldout_access=False, solver_fallback=False,
)
THREAD_ENV = {key: '1' for key in (
    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLIS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS')}
_CONTEXT = None


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    _require(type(value) in (int, float) and math.isfinite(value),
             f'Expected finite number, got {value!r}')
    return float(value)


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _manifest(value):
    return _read(value) if isinstance(value, (str, Path)) else deepcopy(value)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _verify_map(mapping):
    _require(isinstance(mapping, dict) and mapping, 'Empty/malformed hash map')
    for name, expected in mapping.items():
        path = Path(name)
        _require(path.is_absolute(), f'Hash path must be absolute: {path}')
        _require(_digest(path) == expected, f'Hash mismatch: {path}')


def _verify_source_maps(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'source_sha256':
                _verify_map(child)
            else:
                _verify_source_maps(child)
    elif isinstance(value, list):
        for child in value:
            _verify_source_maps(child)


def initialize(R, manifest):
    """Load frozen helpers in a private module, never alter their PROTOCOL."""
    global _CONTEXT
    manifest = _manifest(manifest)
    _require(type(R) is int and R in CAMPAIGNS, 'Require R=3 or R=10')
    _verify_source_maps(manifest)
    _require(manifest['source_sha256'].get(str(Path(__file__).resolve())) ==
             _digest(__file__), 'Controller source must be frozen in manifest')
    config = manifest['sourceconfig'][str(R)]
    for name in ('green_common.py', 'green_training.py', 'green_ccg.py'):
        path = str(CAMPAIGNS[R] / name)
        _require(config['source_sha256'].get(path) == _digest(path),
                 f'Missing original computational freeze: {path}')
    if _CONTEXT is not None:
        _require(_CONTEXT.R == R, 'Different R requires a fresh worker process')
        _CONTEXT.common.verify_sources(config)
        _require(config['protocol'] == _CONTEXT.source_protocol,
                 'Original source protocol mismatch')
        _require(_CONTEXT.common.PROTOCOL == _CONTEXT.source_protocol and
                 _CONTEXT.torch.get_num_threads() == _CONTEXT.torch.get_num_interop_threads() == 1,
                 'Worker numerical protocol or thread settings changed')
        return _CONTEXT
    # Reject preexisting unqualified campaign modules instead of replacing them.
    for name in ('green_common', 'green_training', 'green_ccg', 'source_adapter',
                 'core', 'ccg', 'adversary', 'r10_core', 'r10_pool', 'r10_adversary'):
        _require(name not in sys.modules, f'Fresh worker required: {name} already loaded')
    os.environ.update(THREAD_ENV)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(CAMPAIGNS[R]))
    path = CAMPAIGNS[R] / 'green_ccg.py'
    spec = importlib.util.spec_from_file_location(f'_black_frozen_green_R{R}', path)
    _require(spec is not None and spec.loader is not None, 'Cannot load green helpers')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = sys.modules['green_common']
    common.verify_sources(config)
    _require(config['protocol'] == common.PROTOCOL, 'Original source protocol mismatch')
    common.configure_worker()
    _require(common.torch.get_num_threads() == common.torch.get_num_interop_threads() == 1,
             'Require one Torch intra/inter-op thread')
    expected = dict(max_mp=40, max_epochs=3000, patience=300,
                    separation_tolerance=.01, improvement_threshold=1e-4,
                    membership_tolerance=1e-9, ap_replay_tolerance=1e-4)
    _require(all(common.PROTOCOL[k] == v for k, v in expected.items()),
             'Frozen numerical settings differ')
    _require(common.metadata(200)['R'] == R, 'Loaded numerical metadata R mismatch')
    _CONTEXT = SimpleNamespace(R=R, source=module, common=common,
        source_protocol=deepcopy(common.PROTOCOL), np=common.np, torch=common.torch,
        core=common.core)
    return _CONTEXT


def _decision(candidate):
    """Full-coverage, recomputed bound residual; never use a mean-gap stop."""
    _require(candidate.get('assessed') is True, 'Incomplete candidate')
    rows = candidate['samples']
    n = candidate['Ntrain']
    _require(type(n) is int and n > 0 and len(rows) == n and
             [r['sample_id'] for r in rows] == list(range(n)), 'Incomplete AP coverage')
    _require(len(candidate['pool_costs']) == n, 'Incomplete pool costs')
    passed = True
    for i, row in enumerate(rows):
        _require(row.get('status') == 'completed' and row.get('controller_status') == 'completed'
                 and row.get('solver_status') == 2 and row.get('solcount', 0) >= 1,
                 'Incomplete AP solver result')
        b, p = _finite(row['bound']), _finite(row['pool_cost'])
        _require(p == _finite(candidate['pool_costs'][i]), 'Pool cost mismatch')
        residual = max(0., b - p) / max(1., abs(b), abs(p))
        _require(abs(residual - _finite(row['residual'])) <= 1e-12,
                 'Saved residual does not match bound formula')
        passed = passed and residual <= PROTOCOL['separation_tolerance']
    return passed


def _next_action(candidate, seen):
    if _decision(candidate):
        return 'converged'
    if candidate['state_sha256'] in seen:
        return 'stagnation'
    _require(type(candidate['candidate_id']) is int and
             0 <= candidate['candidate_id'] <= PROTOCOL['max_mp'], 'Invalid total MP count')
    if candidate['candidate_id'] == PROTOCOL['max_mp']:
        return 'iteration_cap'
    return 'mp'


def _choose(prefix, stopping_id):
    _require(type(stopping_id) is int and 0 <= stopping_id < len(prefix), 'Invalid stop index')
    eligible = prefix[:stopping_id + 1]
    for i, candidate in enumerate(eligible):
        _require(candidate['candidate_id'] == i and candidate.get('assessed') is True,
                 'Selection requires complete contiguous prefix')
        _decision(candidate)
        _finite(candidate['ap_incumbent_mean'])
    return min(eligible, key=lambda c: c['ap_incumbent_mean'])


def _same_assessment(summary_row, disk_row):
    """Only original _flush's derived MP duration may lag in assessment.json."""
    a, b = deepcopy(summary_row), deepcopy(disk_row)
    expected = a.get('mp_call', {}).get('wall_time_s', 0.)
    _require(a.get('mp_total_time_s', expected) == expected,
             'Summary MP duration differs from mp_call')
    # On disk this can still be zero from the preceding _flush, or current.
    _require(b.get('mp_total_time_s', 0.) in (0., expected),
             'Unexplained assessment MP duration')
    a['mp_total_time_s'] = b['mp_total_time_s'] = expected
    _require(a == b, 'Assessment differs from source summary beyond derived MP timing')


def _bound_input(job, path, expected=None):
    path = Path(path)
    _require(path.is_absolute(), f'Input path must be absolute: {path}')
    saved = job['input_sha256'].get(str(path))
    _require(saved is not None and (expected is None or saved == expected),
             f'Missing/inconsistent input freeze: {path}')
    _require(_digest(path) == saved, f'Changed input: {path}')
    return path


def _load_pool(ctx, job, path, byte_hash, pool_hash, centers, ids, radius):
    data = _read(_bound_input(job, path, byte_hash))
    pools = [ctx.np.asarray(p, dtype=ctx.np.float64) for p in data['pools']]
    _require(data['original_sample_ids'] == ids and len(pools) == len(centers),
             'Pool sample identity mismatch')
    _require(data['pool_sha256'] == pool_hash == ctx.source.pool_hash(pools), 'Pool hash mismatch')
    for pool, center in zip(pools, centers):
        ctx.source.check_membership(pool, center, radius)
    return pools


def _points(ctx, candidate, pools, centers, ids, radius):
    """Revalidate saved AP gates without a solver call; retain raw float64 points."""
    _decision(candidate)
    points = []
    for i, saved in enumerate(candidate['samples']):
        _require(saved['original_sample_id'] == ids[i] and
                 saved['candidate_id'] == candidate['candidate_id'] and
                 saved['candidate_sha256'] == candidate['state_sha256'] and
                 saved['checkpoint_sha256'] == candidate['checkpoint_sha256'] and
                 saved['pool_sha256'] == candidate['pool_sha256'] and
                 saved['Ntotal'] == candidate['Ntotal'] and
                 saved['Ntrain'] == candidate['Ntrain'] and
                 saved['seed'] == candidate['seed'] and saved['radius'] == radius,
                 'Saved AP identity mismatch')
        _require(ctx.np.array_equal(saved['center1'] + saved['center2'], centers[i]),
                 'AP center differs from train-only JSON')
        row = deepcopy(saved)
        inc = row['incumbent']
        result = (ctx.np.asarray(inc['worst1'], dtype=ctx.np.float64),
                  ctx.np.asarray(inc['worst2'], dtype=ctx.np.float64), inc['cost'])
        points.append(ctx.source.validate_ap(result, row, i, centers[i], pools[i],
                                             radius, candidate['pool_costs'][i]))
        for key in ('residual', 'incumbent_violating', 'membership', 'membership_violation',
                    'raw_bound_minus_pool', 'signed_floor_scaled_gap',
                    'raw_bound_minus_incumbent', 'raw_incumbent_minus_pool'):
            _require(row[key] == saved[key], f'Saved AP gate mismatch: {key}')
    _require(abs(sum(r['incumbent']['cost'] for r in candidate['samples']) / len(ids)
                 - _finite(candidate['ap_incumbent_mean'])) <= 1e-10, 'AP mean mismatch')
    _require(abs(sum(r['bound'] for r in candidate['samples']) / len(ids)
                 - _finite(candidate['ap_bound_mean'])) <= 1e-10, 'AP bound mean mismatch')
    _require(abs(sum(candidate['pool_costs']) / len(ids)
                 - _finite(candidate['restricted_mean'])) <= 1e-10, 'Restricted mean mismatch')
    return points


def _insert(ctx, pools, points, rows, centers, radius):
    _require(len(pools) == len(points) == len(rows) == len(centers), 'Insertion coverage mismatch')
    additions = []
    for i, (point, row) in enumerate(zip(points, rows)):
        _require(point.dtype == ctx.np.float64, 'Insertion must preserve raw float64')
        ctx.source.check_membership(point[None, :], centers[i], radius)
        member = ctx.source.point_membership(point, pools[i])
        a, p = _finite(row['incumbent']['cost']), _finite(row['pool_cost'])
        violating = a - p > PROTOCOL['insertion_tolerance'] * max(1., abs(a), abs(p))
        _require(violating == row['incumbent_violating'] and member == row['membership'],
                 'Insertion gate mismatch')
        if violating and not member['exact_duplicate']:
            pools[i] = ctx.np.vstack((pools[i], point))
            additions.append(i)
    return additions


def _mp_checks(training, incoming, returned):
    _require(training.get('status') == 'completed', 'Incomplete native MP')
    first, best, selected = (_finite(training[k]) for k in
                             ('first_train', 'best_train', 'selected_training_loss'))
    _require(abs(first - incoming['mean']) <= 1e-6, 'MP first/incoming loss mismatch')
    _require(best <= first and selected <= first and
             (best == first or best < first - 1e-4), 'MP training-best threshold mismatch')
    _require(abs(best - selected) <= 1e-6 and abs(returned['mean'] - selected) <= 1e-6,
             'MP best/selected/returned loss mismatch')


def load_job(job, manifest):
    """Verify complete prefix and inputs, returning numerical state; no AP/MP."""
    ctx = initialize(job['R'], manifest)
    manifest = _manifest(manifest)
    n, seed, radius = job['Ntotal'], job['seed'], job['radius']
    _require(type(n) is int and n in ((200, 800) if ctx.R == 3 else (50, 200, 800)),
             'R3 N50 full-black cache belongs to parent reuse, not this controller')
    _require(type(seed) is int and seed in range(10), 'Unapproved seed')
    _require(_finite(radius) in ((0., .001, .003, .01, .03, .2, .4, .7, 1.5) +
                                 ((2.,) if ctx.R == 10 else ())), 'Unapproved radius')
    _verify_map(job['input_sha256'])
    _verify_source_maps(job)
    nominal = job['nominal_record']
    _require(nominal['Ntotal'] == n and nominal['seed'] == seed and
             nominal['lr_net'] in ctx.common.PROTOCOL['lr_grid'], 'Nominal winner mismatch')
    _require(nominal['training_timing']['status'] == 'completed' and
             _finite(nominal['training_timing']['wall_time_s']) >= 0, 'Missing nominal timing')
    data = _read(_bound_input(job, nominal['training_data_path'], nominal['training_data_sha256']))
    _require(set(data) == {'centers', 'original_sample_ids', 'data_sha256'} and
             data['data_sha256'] == nominal['data_sha256'], 'Invalid train-only JSON')
    centers = ctx.np.asarray(data['centers'], dtype=ctx.np.float64)
    ids = data['original_sample_ids']
    _require(centers.shape == (4 * n // 5, 2 * ctx.R) and ctx.np.isfinite(centers).all()
             and (centers >= 0).all(), 'Malformed training centers')
    _require(len(ids) == len(centers) and len(set(ids)) == len(ids) and
             all(type(i) is int and 0 <= i < n for i in ids), 'Malformed training IDs')
    nominal_path = _bound_input(job, nominal['checkpoint'], nominal['checkpoint_sha256'])
    model = ctx.source._verify_checkpoint(nominal_path, nominal['checkpoint_sha256'],
                                          nominal['state_sha256'])
    prefix, source_summary = [], None
    pools = [c[None, :].copy() for c in centers]
    source_path = job['source_summary_path']
    if source_path is None:
        _require(ctx.R == 10 and radius == 2. and job['source_summary_sha256'] is None,
                 'Only R10 rho=2 may start from saved nominal')
    else:
        source_summary = _read(_bound_input(job, source_path, job['source_summary_sha256']))
        for key, expected in dict(R=ctx.R, Ntotal=n, Ntrain=len(ids), seed=seed, radius=radius,
                                  lr_net=nominal['lr_net'], data_sha256=nominal['data_sha256'],
                                  training_data_sha256=nominal['training_data_sha256'],
                                  nominal_checkpoint_sha256=nominal['checkpoint_sha256'],
                                  nominal_state_sha256=nominal['state_sha256']).items():
            _require(source_summary[key] == expected, f'Source summary mismatch: {key}')
        _require(source_summary['status'] == 'completed' and source_summary['converged'] is True
                 and source_summary['terminal_status'] == 'green_gap_satisfied',
                 'Require completed saved GREEN stopping prefix')
        prefix = deepcopy(source_summary['iterations'])
        k = source_summary['stopping_candidate_id']
        _require(type(k) is int and 0 <= k <= 40 and len(prefix) == k + 1 and
                 source_summary['mp_count'] == k, 'Incomplete source stopping prefix')
        previous = None
        for i, candidate in enumerate(prefix):
            _require(candidate['candidate_id'] == i and candidate['Ntotal'] == n and
                     candidate['Ntrain'] == len(ids) and candidate['seed'] == seed and
                     candidate['radius'] == radius, 'Candidate identity mismatch')
            path = _bound_input(job, candidate['checkpoint'], candidate['checkpoint_sha256'])
            model = ctx.source._verify_checkpoint(path, candidate['checkpoint_sha256'],
                                                  candidate['state_sha256'])
            assessment = _bound_input(job, candidate['ap_reference'])
            _same_assessment(candidate, _read(assessment))
            pools = _load_pool(ctx, job, assessment.parent / 'pool.json',
                candidate['pool_file_sha256'], candidate['pool_sha256'], centers, ids, radius)
            points = _points(ctx, candidate, pools, centers, ids, radius)
            if i == 0:
                _require(candidate['state_sha256'] == nominal['state_sha256'] and
                         candidate['checkpoint_sha256'] == nominal['checkpoint_sha256'] and
                         all(ctx.np.array_equal(p, c[None, :]) for p, c in zip(pools, centers)),
                         'Candidate0 is not saved nominal on center-only pools')
                _require(abs(candidate['restricted_mean'] - nominal['train_mean']) <= 1e-6,
                         'Saved nominal training loss mismatch')
            else:
                _require(previous['returned_state_sha256'] == candidate['state_sha256'] and
                         previous['selected_checkpoint_sha256'] == candidate['checkpoint_sha256'] and
                         previous['pool_sha256'] == candidate['pool_sha256'] and
                         abs(previous['selected_training_loss'] - candidate['restricted_mean']) <= 1e-6,
                         'Incomplete MP-to-next-candidate chain')
            if i < k:
                mp = candidate['mp']
                mp_folder = assessment.parent / 'mp'
                _require(_read(_bound_input(job, mp_folder / 'controller_training.json')) == mp,
                         'MP controller record mismatch')
                raw_training = _read(_bound_input(job, mp_folder / 'training.json'))
                _require(all(key in mp and mp[key] == value for key, value in raw_training.items()),
                         'Native training record differs from MP controller record')
                _require(mp['mp_iteration'] == i + 1 and mp['output_candidate_id'] == i + 1
                         and mp['incoming_state_sha256'] == candidate['state_sha256'],
                         'MP iteration/state chain mismatch')
                _require(all(mp[key] == value for key, value in dict(
                    R=ctx.R, Ntotal=n, Ntrain=len(ids), seed=seed, radius=radius,
                    max_epochs=3000, patience=300, lr_net=nominal['lr_net'],
                    torch_threads=1, torch_interop_threads=1, gurobi_threads=1).items()),
                    'Historical MP configuration mismatch')
                _mp_checks(mp, candidate['incoming_pool_evaluation'], candidate['returned_pool_evaluation'])
                _require(_finite(candidate['mp_call']['wall_time_s']) >= 0, 'Invalid historical MP time')
                inserted = [p.copy() for p in pools]
                added = _insert(ctx, inserted, points, candidate['samples'], centers, radius)
                _require(added == candidate['added_sample_ids'] and added and
                         ctx.source.pool_hash(inserted) == mp['pool_sha256'], 'MP insertion history mismatch')
                _load_pool(ctx, job, mp_folder / 'pool.json', mp['pool_file_sha256'],
                           mp['pool_sha256'], centers, ids, radius)
                selected_path = _bound_input(job, mp_folder / 'selected.pth', mp['selected_checkpoint_sha256'])
                ctx.source._verify_checkpoint(selected_path, mp['selected_checkpoint_sha256'],
                                               mp['returned_state_sha256'])
                previous = mp
            else:
                _require('mp' not in candidate and 'mp_call' not in candidate,
                         'Saved STOP already contains an outgoing MP')
        for key, total in (
            ('ap_total_time_s', sum(c['ap']['wall_time_s'] for c in prefix)),
            ('mp_total_time_s', sum(c.get('mp_call', {}).get('wall_time_s', 0.) for c in prefix)),
            ('evaluation_time_s', sum(c.get(phase, {}).get('wall_time_s', 0.)
                for c in prefix for phase in ('evaluation', 'incoming_pool_evaluation',
                                              'returned_pool_evaluation')))):
            _require(_finite(source_summary[key]) >= 0 and
                     abs(source_summary[key] - total) <= 1e-6,
                     f'Historical aggregate timing mismatch: {key}')
        _require(_finite(source_summary['wall_time_s']) >= 0, 'Invalid historical wall time')
        # The model/pool here are the LAST assessed STOP, never source selected.*.
    workers = manifest['workers']
    workers = workers['ccg'] if isinstance(workers, dict) else workers
    _require(type(workers) is int and workers > 0, 'Invalid worker count')
    return SimpleNamespace(ctx=ctx, job=deepcopy(job), manifest=manifest, model=model,
        pools=pools, centers=centers, ids=ids, prefix=prefix, source_summary=source_summary,
        workers=workers, nominal=nominal)


@contextmanager
def _timed(row):
    start = time.perf_counter()
    row.update(started_at_utc=_utc(), status='running')
    try:
        yield row
    except BaseException:
        row['status'] = 'failed'
        raise
    else:
        row['status'] = 'completed'
    finally:
        row.update(ended_at_utc=_utc(), wall_time_s=time.perf_counter() - start)


def _metadata(state, method):
    job = state.job
    result = state.ctx.common.metadata(job['Ntotal'], job['seed'], job['radius'], method)
    _require(result['R'] == job['R'], 'Numerical metadata R mismatch')
    result.update(experiment=PROTOCOL['experiment'], ccg_workers=state.workers,
                  torch_interop_threads=1, ap_workers=1)
    return result


def _candidate(state, folder, index, checkpoint, byte_hash, pools):
    folder.mkdir(parents=True, exist_ok=False)
    src = state.ctx.source
    c = _metadata(state, 'fixed_policy_AP_sweep')
    c.update(candidate_id=index, checkpoint=str(folder / 'candidate.pth'),
        checkpoint_sha256=byte_hash, state_sha256=src.state_hash(state.model),
        pool_sha256=src.pool_hash(pools), ap_reference=str(folder / 'assessment.json'), assessed=False)
    src._copy(checkpoint, c['checkpoint'], byte_hash)
    src._verify_checkpoint(c['checkpoint'], byte_hash, c['state_sha256'])
    c['pool_file_sha256'] = src._save_pool(folder / 'pool.json', pools, state.ids)
    return c


def _train(state, candidate, pools, folder, save, epochs=3000):
    src = state.ctx.source
    folder.mkdir(parents=True, exist_ok=False)
    training = candidate['mp'] = _metadata(state, 'robust_neural_MP')
    training.update(mp_iteration=candidate['candidate_id'] + 1,
        output_candidate_id=candidate['candidate_id'] + 1, sample_count=len(pools),
        incoming_state_sha256=src.state_hash(state.model), pool_sha256=src.pool_hash(pools),
        pool_file_sha256=src._save_pool(folder / 'pool.json', pools, state.ids),
        preflight_only=epochs == 2)
    try:
        incoming = candidate['incoming_pool_evaluation'] = {}
        src._pool_mean(state.model, pools, incoming)
        training['incoming_pool_mean'] = incoming['mean']
        per_ball = src.restricted_objective(pools)
        save()
        with _timed(candidate.setdefault('mp_call', {})):
            state.model = src.train_model(state.model, lambda current: per_ball(current).mean(),
                state.nominal['lr_net'], folder, training, epochs=epochs, patience=300)
        returned = candidate['returned_pool_evaluation'] = {}
        src._pool_mean(state.model, pools, returned)
        _mp_checks(training, incoming, returned)
        _require(src.pool_hash(pools) == training['pool_sha256'] and
                 _digest(folder / 'pool.json') == training['pool_file_sha256'], 'MP changed pools')
        selected = folder / 'selected.pth'
        byte_hash, learned = _digest(selected), src.state_hash(state.model)
        src._verify_checkpoint(selected, byte_hash, learned)
        selected.chmod(0o444)
        training.update(returned_state_sha256=learned, selected_checkpoint_sha256=byte_hash,
            returned_pool_mean=returned['mean'], same_pool_change=returned['mean'] - incoming['mean'],
            unchanged_weights=learned == training['incoming_state_sha256'], controller_status='completed')
        return selected, byte_hash
    except BaseException as error:
        training.update(controller_status='failed', controller_error=dict(
            type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        state.ctx.core.write_json(folder / 'controller_training.json', training)
        save()


def _reverify(state):
    _verify_map(state.job['input_sha256'])
    _verify_source_maps(state.job)
    _verify_source_maps(state.manifest)
    state.ctx.common.verify_sources(state.manifest['sourceconfig'][str(state.ctx.R)])
    _require(state.ctx.common.PROTOCOL == state.ctx.source_protocol,
             'Frozen numerical PROTOCOL mutated')
    _require(state.ctx.torch.get_num_threads() == state.ctx.torch.get_num_interop_threads() == 1,
             'Worker thread settings changed')


def _continue(state, folder, record, save):
    ctx, src = state.ctx, state.ctx.source
    prefix = state.prefix
    seen = {c['state_sha256'] for c in prefix[:-1]}
    if prefix:
        candidate = deepcopy(prefix[-1])
        record['extension_boundary'] = candidate
        # Do not change original candidate paths or write its assessment.
    else:
        candidate = _candidate(state, folder / 'candidate_00', 0,
            state.nominal['checkpoint'], state.nominal['checkpoint_sha256'], state.pools)
        record['new_iterations'].append(candidate)
        src.assess_candidate(state.model, state.pools, state.centers, state.ids,
            state.job['radius'], candidate, folder / 'candidate_00', save)
        _require(abs(candidate['restricted_mean'] - state.nominal['train_mean']) <= 1e-6,
                 'Fresh radius2 nominal pool replay mismatch')
    while True:
        action = _next_action(candidate, seen)
        record.update(stopping_candidate_id=candidate['candidate_id'], terminal_status=action,
                      stopping_separation_status='separation_satisfied' if action == 'converged' else 'violating')
        if action != 'mp':
            record['converged'] = action == 'converged'
            break
        points = _points(ctx, candidate, state.pools, state.centers, state.ids, state.job['radius'])
        pools = [p.copy() for p in state.pools]
        added = _insert(ctx, pools, points, candidate['samples'], state.centers, state.job['radius'])
        candidate['added_sample_ids'] = added
        if not added:
            record.update(terminal_status='stagnation', terminal_reason='no_new_violating_incumbents')
            break
        seen.add(candidate['state_sha256'])
        index = candidate['candidate_id']
        # Even the first outgoing MP lives exclusively under the new boundary.
        mp_folder = folder / 'extension_boundary' / f'candidate_{index:02d}' / 'mp'
        checkpoint, byte_hash = _train(state, candidate, pools, mp_folder, save)
        if candidate is not record.get('extension_boundary'):
            ctx.core.write_json(candidate['ap_reference'], candidate)
        record['mp_count'] += 1
        record['new_mp_count'] += 1
        state.pools = pools
        previous = candidate
        candidate = _candidate(state, folder / f'candidate_{index + 1:02d}', index + 1,
                               checkpoint, byte_hash, pools)
        record['new_iterations'].append(candidate)
        src.assess_candidate(state.model, pools, state.centers, state.ids, state.job['radius'],
                             candidate, Path(candidate['ap_reference']).parent, save)
        _require(abs(candidate['restricted_mean'] - previous['mp']['selected_training_loss']) <= 1e-6,
                 'Next assessed candidate differs from returned MP loss')
        candidate['unchanged_from_previous'] = candidate['state_sha256'] == previous['state_sha256']
        candidate['closure_with_unchangedweights'] = candidate['unchanged_from_previous'] and _decision(candidate)
        ctx.core.write_json(candidate['ap_reference'], candidate)
        save()
    record['closure_with_unchangedweights'] = bool(candidate.get('unchanged_from_previous', False)
                                                  and record['converged'])
    if record['converged']:
        selected = _choose(prefix + record['new_iterations'], candidate['candidate_id'])
        _reverify(state)
        src._verify_checkpoint(selected['checkpoint'], selected['checkpoint_sha256'], selected['state_sha256'])
        # Selection is a verified pointer, not a second training/held-out phase.
        record.update(selected_candidate_id=selected['candidate_id'],
            selected_checkpoint=selected['checkpoint'], selected_checkpoint_sha256=selected['checkpoint_sha256'],
            selected_state_sha256=selected['state_sha256'], selected_ap_reference=selected['ap_reference'],
            selected_ap_incumbent_mean=selected['ap_incumbent_mean'],
            selected_separation_status='separation_satisfied' if _decision(selected) else 'violating')


def _preflight(state, folder, record, save):
    ctx, src = state.ctx, state.ctx.source
    current = state.prefix[-1] if state.prefix else state.nominal
    candidate = _candidate(state, folder / 'disposable',
        len(state.prefix) - 1 if state.prefix else 0,
        current['checkpoint'], current['checkpoint_sha256'], state.pools)
    record['disposable'] = candidate
    row = candidate['single_center_ap'] = dict(sample_id=0)
    frozen = src.state_hash(state.model)
    with _timed(candidate.setdefault('ap', {})):
        result = src.adversary.solve_ap(state.model, state.centers[0, :ctx.R],
            state.centers[0, ctx.R:], state.job['radius'], row)
        src.validate_ap(result, row, 0, state.centers[0], state.pools[0],
                        state.job['radius'], record['native_last_pool_replay']['per_ball_costs'][0])
        _require(src.state_hash(state.model) == frozen, 'Preflight AP changed model')
    # Disposable native MP on a copied checkpoint and raw pool; no policy reuse.
    state.model = src._verify_checkpoint(candidate['checkpoint'], candidate['checkpoint_sha256'], frozen)
    pools = [p.copy() for p in state.pools]
    _train(state, candidate, pools, folder / 'disposable' / 'mp', save, epochs=2)
    record.update(solver_status=row['solver_status'], preflight_only=True,
                  production_reuse_forbidden=True, terminal_status='preflight_completed')


def _execute(job, manifest_arg, output_folder, mode):
    start = time.perf_counter()
    folder = Path(output_folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    summary = folder / 'summary.json'
    record = dict(experiment=PROTOCOL['experiment'], method='robust_neural_CCG_continuation',
        **{k: job[k] for k in ('R', 'Ntotal', 'seed', 'radius')},
        protocol=deepcopy(PROTOCOL), mode=mode, status='running', complete=False,
        converged=False, terminal_status='running', new_iterations=[], new_mp_count=0,
        mp_count=0, selected_candidate_id=None, selected_checkpoint=None,
        selected_checkpoint_sha256=None, selected_state_sha256=None,
        selected_ap_reference=None, selected_ap_incumbent_mean=None,
        selected_separation_status=None, started_at_utc=_utc(), summary_path=str(summary),
        timing_unit='seconds', timing_clock='time.perf_counter',
        timing_scope='current invocation including verification; historical prefix/nominal excluded; final write excluded',
        torch_threads=1, torch_interop_threads=1, gurobi_threads=1, ap_workers=1,
        global_robust_optimality_certified=False, heldout_access=False,
        environment=dict(platform=platform.platform(), hostname=platform.node(),
                         python=sys.version, executable=sys.executable), error=None)
    state = None
    manifest_hash = None

    def save():
        if state is None:
            summary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        else:
            state.ctx.core.write_json(summary, record)

    try:
        save()
        manifest = _manifest(manifest_arg)
        if isinstance(manifest_arg, (str, Path)):
            manifest_hash = _digest(manifest_arg)
            record.update(manifest_path=str(Path(manifest_arg).resolve()), manifest_sha256=manifest_hash)
        with _timed(record.setdefault('input_verification', {})):
            state = load_job(job, manifest)
        record.update(Ntrain=len(state.ids), Nval=job['Ntotal'] // 5,
            network_seed=0, split_seed=100000 + job['seed'], ccg_workers=state.workers,
            lr_net=state.nominal['lr_net'], mp_count=len(state.prefix) - 1 if state.prefix else 0,
            historical_nominal=dict(reused=True, retrained=False,
                training_timing=deepcopy(state.nominal['training_timing']),
                scope='persisted nominal winner time, not current invocation'),
            source_prefix=dict(summary_path=job['source_summary_path'],
                summary_sha256=job['source_summary_sha256'], candidates=len(state.prefix),
                immutable=True, timing_scope='original saved complete GREEN prefix',
                assessment_timing_normalization='Only mp_total_time_s derived from mp_call.wall_time_s; disk may lag at zero',
                timings={k: state.source_summary.get(k) for k in
                    ('wall_time_s', 'ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s',
                     'started_at_utc', 'ended_at_utc', 'ccg_workers')}
                    if state.source_summary is not None else None))
        replay = record['native_last_pool_replay'] = {}
        state.ctx.source._pool_mean(state.model, state.pools, replay)
        expected = state.prefix[-1]['restricted_mean'] if state.prefix else state.nominal['train_mean']
        _require(abs(replay['mean'] - expected) <= 1e-6, 'Native last-pool replay mismatch')
        if state.prefix:
            _require(len(replay['per_ball_costs']) == len(state.prefix[-1]['pool_costs']) and
                     all(abs(a - b) <= 1e-6 for a, b in
                         zip(replay['per_ball_costs'], state.prefix[-1]['pool_costs'])),
                     'Native last-pool per-center replay mismatch')
        save()
        if mode == 'job':
            _continue(state, folder, record, save)
            record.update(status='completed' if record['converged'] else 'nonconverged',
                          complete=record['converged'])
        elif mode == 'preflight':
            _preflight(state, folder, record, save)
            record.update(status='completed', complete=True)
        else:
            record.update(status='completed', complete=True, terminal_status='verified')
    except BaseException as error:
        record.update(status='failed', complete=False, converged=False, terminal_status='failure',
            error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        if not isinstance(error, Exception):
            raise
    finally:
        try:
            if state is not None:
                _reverify(state)
            elif 'manifest' in locals():
                _verify_source_maps(manifest)
                _verify_map(job['input_sha256'])
            if manifest_hash is not None:
                _require(_digest(manifest_arg) == manifest_hash, 'Manifest changed during invocation')
        except BaseException as error:
            record.update(integrity_error=dict(type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc()), status='failed', complete=False,
                converged=False, terminal_status='failure')
        if record['status'] == 'failed':
            for key in list(record):
                if key.startswith('selected_'):
                    record[key] = None
        rows = list(record['new_iterations'])
        if 'extension_boundary' in record:
            # Boundary AP/evaluation are historical, only its new MP phases count.
            boundary = record['extension_boundary']
            rows.append({k: boundary[k] for k in ('mp_call', 'incoming_pool_evaluation',
                         'returned_pool_evaluation', 'mp') if k in boundary})
        if 'disposable' in record:
            rows.append(record['disposable'])
            record['solver_status'] = record['disposable'].get('single_center_ap', {}).get('solver_status')
            record.update(preflight_only=True, production_reuse_forbidden=True)
        record['current_timings'] = dict(
            ap_total_time_s=sum(c.get('ap', {}).get('wall_time_s', 0.) for c in rows),
            mp_total_time_s=sum(c.get('mp_call', {}).get('wall_time_s', 0.) for c in rows),
            evaluation_time_s=record.get('native_last_pool_replay', {}).get('wall_time_s', 0.) +
                sum(c.get(k, {}).get('wall_time_s', 0.) for c in rows for k in
                    ('evaluation', 'incoming_pool_evaluation', 'returned_pool_evaluation')),
            wall_time_s=time.perf_counter() - start,
            nominal_training_time_s=0., heldout_evaluation_time_s=0.,
            newly_trained_runtime_available=any('mp_call' in c for c in rows))
        record.update(ended_at_utc=_utc(), peak_rss_bytes=int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            cached_only=mode == 'job' and not record['new_iterations'] and not any('mp_call' in c for c in rows))
        record['wall_time_s'] = record['current_timings']['wall_time_s']
        record['wall_time_human'] = f"{record['wall_time_s']:.3f} s / {record['wall_time_s'] / 60:.3f} min"
        save()
    excluded = {'new_iterations', 'extension_boundary', 'disposable', 'native_last_pool_replay', 'protocol'}
    result = {k: v for k, v in record.items() if k not in excluded}
    result['summary_sha256'] = _digest(summary)
    return result


def run_job(job, manifest_path, output_folder):
    """Continue the saved last STOP, return compact JSON-safe evidence."""
    _require(isinstance(manifest_path, (str, Path)), 'run_job requires manifest_path')
    return _execute(job, manifest_path, output_folder, 'job')


def verify_job(job, manifest, output):
    """Read-only prefix/checkpoint verification plus native terminal-pool replay."""
    return _execute(job, manifest, output, 'verify')


def preflight_job(job, manifest, folder):
    """One center AP and disposable two-epoch native MP; no production selection."""
    return _execute(job, manifest, folder, 'preflight')


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('job', 'verify', 'preflight'))
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--jobid', required=True)
    args = parser.parse_args()
    manifest = _read(args.manifest)
    matches = [j for j in manifest['jobs'] if str(j['jobid']) == args.jobid]
    _require(len(matches) == 1, 'Expected one matching jobid')
    job = matches[0]
    result = _execute(job, args.manifest, job['output_folder'], args.command)
    print(json.dumps(result, allow_nan=False))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(_main())