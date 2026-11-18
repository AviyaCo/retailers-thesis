"""Fixed-policy CCG controller; no launcher, test data, or nominal retraining.

API: run_configuration(seed, radius, output_root, provenance_hash=None).
Creates output_root/seed_{seed}/radius_{radius:g} exclusively (no resume).
Ordinary failures return status='failed' and persist summary.json; directory
collisions raise FileExistsError without touching existing artifacts. Workers
must set Torch intra/inter-op threads to one BEFORE calling this function.

AP is a float64 real-arithmetic folded policy approximation to native float32
inference. REPLAY_TOLERANCE checks this discrepancy and numerical ordering;
SEPARATION_TOLERANCE is a distinct fixed-policy closure criterion, NOT a global
optimality gap. Neither Adam objectives nor candidate AP values are monotone.
"""
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

import adversary
import core

N_TRAIN = 40
MAX_MP = 40
SEPARATION_TOLERANCE = .01
MEMBERSHIP_TOLERANCE = 1e-9
LR_SEARCH_ROOT = Path(__file__).resolve().parent / 'lr_search_results_v2'


def configuration_directory(output_root, seed, radius):
    return Path(output_root).resolve() / f'seed_{seed}' / f'radius_{radius:g}'


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(core.weights(model).items()):
        h.update(name.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def raw_data_hash(data):
    h = hashlib.sha256()
    for key in ('xi1', 'xi2'):
        h.update(np.asarray(data[key], dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def pool_hash(pools):
    h = hashlib.sha256()
    for sample_id, pool in enumerate(pools):
        h.update(np.asarray([sample_id, len(pool)], dtype='<i8').tobytes())
        h.update(np.asarray(pool, dtype='<f8').tobytes(order='C'))
    return h.hexdigest()


def _copy(source, destination, expected=None):
    """Independent, read-only snapshot, checked after copy (never a hard link)."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)
    actual = core.digest(destination)
    if expected is not None and actual != expected:
        raise ValueError(f'Input hash mismatch: {source}')
    destination.chmod(0o444)
    return actual


def load_nominal(seed, folder, record):
    """Verify completed v2 winner, exact regenerated split, and source manifest."""
    inputs = folder / 'inputs'
    inputs.mkdir()
    source = LR_SEARCH_ROOT / f'seed_{seed}'
    copies = record['input_sha256'] = {}
    for name, path in [('lr_summary.json', LR_SEARCH_ROOT / 'summary.json'),
                       ('seed_summary.json', source / 'summary.json'),
                       ('lr_provenance.json', LR_SEARCH_ROOT / 'provenance.json')]:
        copies[name] = _copy(path, inputs / name)
    sweep = json.loads((inputs / 'lr_summary.json').read_text())
    nominal = json.loads((inputs / 'seed_summary.json').read_text())
    if sweep['status'] != 'completed' or nominal['status'] != 'completed' or not nominal['complete']:
        raise ValueError('Require completed v2 LR search and completed seed')
    if nominal['seed'] != seed or nominal['split_seed'] != 100000 + seed:
        raise ValueError('Nominal seed/split seed mismatch')
    if copies['lr_provenance.json'] != sweep['provenance_sha256'] or nominal['provenance_sha256'] != sweep['provenance_sha256']:
        raise ValueError('LR provenance hash mismatch')
    embedded = [row for row in sweep['seed_results'] if row['seed'] == seed]
    if len(embedded) != 1 or embedded[0] != nominal:
        raise ValueError('Sweep and seed summaries disagree')
    manifest = json.loads((inputs / 'lr_provenance.json').read_text())
    for path, expected in manifest['source_sha256'].items():
        if core.digest(Path(path)) != expected:
            raise ValueError(f'LR source changed: {path}')
    copies['split.json'] = _copy(source / 'split.json', inputs / 'split.json', nominal['split_artifact_sha256'])
    split = json.loads((inputs / 'split.json').read_text())
    data = core.split_data(seed)
    for key, value in data.items():
        if not np.array_equal(np.asarray(value), np.asarray(split['arrays'][key])):
            raise ValueError(f'Regenerated split differs: {key}')
    raw_hash = raw_data_hash(data)
    if raw_hash != nominal['raw_data_sha256'] or raw_hash != split['raw_data_sha256']:
        raise ValueError('Regenerated raw float64 data hash mismatch')
    rows = nominal['configurations']
    if len(rows) != len(core.LR_GRID) or {r['lr_net'] for r in rows} != set(core.LR_GRID):
        raise ValueError('Incomplete LR grid')
    for row in rows:
        if row['status'] != 'completed' or not np.isfinite(row['restored_validation_evaluation']['mean']):
            raise ValueError('Failed/nonfinite LR configuration')
    winner = nominal['winner']
    best = min(rows, key=lambda row: (row['restored_validation_evaluation']['mean'], row['lr_net']))
    if (winner['lr_net'] != best['lr_net'] or winner['selected_sha256'] != best['selected_sha256']
            or winner['validation_mean'] != best['restored_validation_evaluation']['mean']):
        raise ValueError('Winner is not the exact validation-selected LR checkpoint')
    checkpoint = source / f"lr_{winner['lr_net']:g}" / 'selected.pth'
    if checkpoint.resolve() != Path(winner['checkpoint']).resolve():
        raise ValueError('Winner checkpoint path mismatch')
    copies['nominal.pth'] = _copy(checkpoint, inputs / 'nominal.pth', winner['selected_sha256'])
    # Freeze the controller and numerical source files as well as all input data.
    for path in (Path(__file__), Path(core.__file__), Path(adversary.__file__),
                 core._BASELINE_PATH, core._loader.SOURCE):
        name = f'source_{len(copies)}_{Path(path).name}'
        copies[name] = _copy(path, inputs / name)
    record.update(lr_net=winner['lr_net'], raw_data_sha256=raw_hash,
                  nominal_checkpoint_sha256=winner['selected_sha256'],
                  cached_nominal=dict(reused=True, retrained=False,
                      timing_scope='persisted earlier invocation; NOT current job elapsed',
                      selected_training=best['training'],
                      selected_configuration_wall_time_s=best['wall_time_s'],
                      selected_train_evaluation=best['restored_train_evaluation'],
                      selected_validation_evaluation=best['restored_validation_evaluation'],
                      saa_initialization=nominal['saa_initialization'],
                      seed_lr_search_wall_time_s=nominal['wall_time_s'],
                      whole_lr_search_wall_time_s=sweep['wall_time_s'],
                      lr_selection=nominal['lr_selection'], seed_workers=sweep['seed_workers']),
                  nominal_expected_validation=winner['validation_mean'],
                  nominal_expected_train=winner['train_mean'])
    state = torch.load(inputs / 'nominal.pth', map_location='cpu', weights_only=True)
    return core.model_from_state(state), data


def check_membership(points, center, radius):
    points = np.asarray(points)
    if points.dtype != np.float64 or points.ndim != 2 or points.shape[1] != 6 or not len(points):
        raise ValueError('Pool/AP points must be nonempty raw float64 paired6 vectors')
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite pool/AP point')
    lower, upper = np.maximum(0., center - radius), center + radius
    residual = max(0., float((lower - points).max()), float((points - upper).max()))
    if residual > MEMBERSHIP_TOLERANCE:
        raise ValueError(f'Box membership violation: {residual}')
    return residual


def point_membership(point, pool):
    """Exact per-ball identity; near/collision diagnostics never merge raw points."""
    distances = np.max(np.abs(pool - point), axis=1)
    exact = np.all(pool == point, axis=1)
    collisions = np.all(pool.astype(np.float32) == point.astype(np.float32), axis=1)
    return dict(exact_duplicate=bool(exact.any()),
                nearest_linf_distance=float(distances.min()),
                near_duplicate=bool(((distances <= MEMBERSHIP_TOLERANCE) & ~exact).any()),
                float32_collision=bool((collisions & ~exact).any()),
                float32_matches=np.flatnonzero(collisions).tolist())


def restricted_objective(pools):
    """Flatten ragged pools ONCE per MP; no padding and differentiable group max."""
    sizes = [len(pool) for pool in pools]
    points = np.concatenate(pools, axis=0)
    x1 = torch.tensor(points[:, :3], dtype=torch.float32)
    x2 = torch.tensor(points[:, 3:], dtype=torch.float32)

    def per_ball(model):
        costs = model.per_sample_cost(x1, x2)
        if costs.shape != (len(points),) or not bool(torch.isfinite(costs).all()):
            raise ValueError('Nonfinite/malformed restricted-pool costs')
        return torch.stack([torch.max(group) for group in torch.split(costs, sizes)])

    return per_ball


def _finite(value, name):
    if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(f'Nonfinite/malformed {name}: {value}')
    return float(value)


def validate_ap(result, row, sample_id, center, pool, radius, pool_cost):
    if row.get('sample_id') != sample_id:
        raise ValueError('Missing/duplicate AP sample_id')
    if row.get('status') != 'completed' or row.get('solver_status') != 2 or row.get('solcount', 0) < 1:
        raise ValueError('Incomplete AP solver result')
    if not isinstance(result, tuple) or len(result) != 3:
        raise ValueError('Missing/malformed AP result')
    x1, x2, returned_cost = result
    if np.shape(x1) != (3,) or np.shape(x2) != (3,):
        raise ValueError('Malformed AP paired point')
    point = np.concatenate((x1, x2))
    row['membership_violation'] = check_membership(point[None, :], center, radius)
    row['membership'] = point_membership(point, pool)
    a = _finite(row['incumbent']['cost'], 'AP incumbent')
    b = _finite(row['bound'], 'AP bound')
    native = _finite(row['replay']['cost'], 'AP replay cost')
    if row['replay'].get('status') != 'completed':
        raise ValueError('Incomplete AP replay')
    if _finite(returned_cost, 'returned AP cost') != a:
        raise ValueError('Returned AP incumbent mismatch')
    for name, values in [('worst1', x1), ('worst2', x2)]:
        if not np.array_equal(np.asarray(row['incumbent'][name]), values):
            raise ValueError('Returned AP point mismatch')
    tol = adversary.REPLAY_TOLERANCE
    discrepancy = abs(a - native) / max(1., abs(a), abs(native))
    reported = _finite(row['replay']['floor_scaled_discrepancy'], 'reported replay discrepancy')
    if discrepancy > tol or abs(discrepancy - reported) > 1e-12:
        raise ValueError('Invalid AP/native replay discrepancy')
    raw_gap = b - pool_cost
    scale = max(1., abs(b), abs(pool_cost))
    incumbent_gap = a - pool_cost
    row.update(pool_cost=pool_cost, raw_bound_minus_pool=raw_gap,
               signed_floor_scaled_gap=raw_gap / scale,
               raw_bound_minus_incumbent=b - a,
               raw_incumbent_minus_pool=incumbent_gap,
               residual=max(0., raw_gap) / scale,
               incumbent_violating=incumbent_gap > tol * max(1., abs(a), abs(pool_cost)))
    if b - a < -tol * max(1., abs(b), abs(a)):
        raise ValueError('Material negative bound/incumbent ordering')
    if raw_gap < -tol * scale:
        raise ValueError('Material negative bound/pool gap')
    if row['residual'] > SEPARATION_TOLERANCE:
        if row['membership']['exact_duplicate'] or row['membership']['float32_collision']:
            raise ValueError('Duplicate/colliding point with material separation residual')
        if not row['incumbent_violating']:
            raise ValueError('Unresolved AP bound: no numerically violating incumbent')
    return point


def assess_candidate(model, pools, centers, data, radius, candidate, folder):
    """Assess the immutable candidate against its pre-insertion pool; all40 APs."""
    frozen = candidate['state_sha256']
    candidate.update(assessed=False, separation_status='unassessed', samples=[],
                     ap_total_time_s=0., mp_total_time_s=0.)
    objective = restricted_objective(pools)
    evaluation = candidate['evaluation'] = {}
    with core.timed(evaluation), torch.no_grad():
        model.eval()
        maxima = objective(model).detach().cpu().numpy()
        validation = model.per_sample_cost(data['val_x1'], data['val_x2'])
        if validation.shape != (10,) or not bool(torch.isfinite(validation).all()):
            raise ValueError('Nonfinite/malformed candidate validation')
        candidate.update(pool_costs=maxima.tolist(), restricted_mean=float(maxima.mean()),
                         validation_mean=_finite(float(validation.mean()), 'validation mean'))
    if state_hash(model) != frozen:
        raise ValueError('Candidate changed during evaluation')
    ap_record = candidate['ap'] = {}
    points, failures = [], []
    try:
        with core.timed(ap_record):
            for sample_id in range(N_TRAIN):
                row = dict(seed=candidate['seed'], radius=radius, sample_id=sample_id,
                           original_sample_id=int(data['train_indices'][sample_id]),
                           candidate_id=candidate['candidate_id'], candidate_sha256=frozen,
                           pool_sha256=candidate['pool_sha256'])
                start = time.perf_counter()
                row['controller_started_at_utc'] = core.utc_now()
                try:
                    if state_hash(model) != frozen:
                        raise ValueError('Frozen AP candidate changed')
                    result = adversary.solve_ap(model, centers[sample_id, :3],
                                                centers[sample_id, 3:], radius, row)
                    point = validate_ap(result, row, sample_id, centers[sample_id],
                                        pools[sample_id], radius, float(maxima[sample_id]))
                    if state_hash(model) != frozen:
                        raise ValueError('Frozen AP candidate changed')
                    points.append(point)
                    row['controller_status'] = 'completed'
                except Exception as error:
                    failures.append(sample_id)
                    row.update(controller_status='failed', controller_error=dict(
                        type=type(error).__name__, message=str(error)))
                finally:
                    row.update(controller_wall_time_s=time.perf_counter() - start,
                               controller_ended_at_utc=core.utc_now())
                    candidate['samples'].append(row)
                    core.write_json(folder / f'sample_{sample_id:02d}.json', row)
                    core.write_json(folder / 'assessment.json', candidate)
            if failures:
                raise ValueError(f'Failed/incomplete AP samples: {failures}')
            ids = [row['sample_id'] for row in candidate['samples']]
            if len(ids) != N_TRAIN or set(ids) != set(range(N_TRAIN)):
                raise ValueError('Missing/duplicate AP coverage')
    finally:
        candidate['ap_total_time_s'] = ap_record.get('wall_time_s', 0.)
        core.write_json(folder / 'assessment.json', candidate)
    separated = all(row['residual'] <= SEPARATION_TOLERANCE for row in candidate['samples'])
    candidate.update(assessed=True, separation_status='separation_satisfied' if separated else 'violating',
                     ap_incumbent_mean=float(np.mean([r['incumbent']['cost'] for r in candidate['samples']])),
                     ap_bound_mean=float(np.mean([r['bound'] for r in candidate['samples']])))
    return points


def run_configuration(seed, radius, output_root, provenance_hash=None):
    """Return strict-JSON-safe summary, including failure and provisional selection.

    Selection fields: selected_candidate_id, selected_checkpoint,
    selected_validation_mean, selected_state_sha256, selected_ap_reference,
    selected_separation_status. On failure these remain None; provisional_best
    refers only to a fully assessed candidate, never an incomplete AP sweep.
    status/complete describe execution; terminal_status describes trajectory.
    wall_time_s excludes all cached nominal/LR timings. iterations contains one
    record per candidate, including nominal0 and the final MP's own assessment.
    """
    if isinstance(seed, bool) or seed not in core.SEEDS or not np.isfinite(radius) or radius < 0:
        raise ValueError('Require approved data seed and nonnegative finite radius')
    seed, radius = int(seed), float(radius)
    folder = configuration_directory(output_root, seed, radius)
    folder.mkdir(parents=True, exist_ok=False)
    job_start = time.perf_counter()
    record = dict(experiment='retailer_R3_N50_net16_v2_train40_val10_CCG',
                  method='robust_neural_CCG', seed=seed, radius=radius, Ntotal=50,
                  Ntrain=40, Nval=10, split_seed=100000 + seed, network_seed=0,
                  max_mp=MAX_MP, max_epochs=3000, patience=300, lr_inventory=.01,
                  separation_tolerance=SEPARATION_TOLERANCE,
                  numerical_consistency_tolerance=adversary.REPLAY_TOLERANCE,
                  membership_tolerance=MEMBERSHIP_TOLERANCE,
                  output_directory=str(folder), summary_path=str(folder / 'summary.json'),
                  provenance_sha256=provenance_hash, iterations=[], mp_count=0,
                  complete=False, terminal_status='running', configuration_status='running',
                  selected_candidate_id=None, selected_checkpoint=None,
                  selected_validation_mean=None, selected_state_sha256=None,
                  selected_ap_reference=None, selected_separation_status=None,
                  provisional_best=None, timing_unit='seconds', timing_clock='time.perf_counter',
                  torch_threads=torch.get_num_threads(),
                  torch_interop_threads=torch.get_num_interop_threads(),
                  gurobi_threads=1, ap_workers_per_job=1, job_workers='external orchestrator',
                  environment=dict(platform=platform.platform(), hostname=platform.node()),
                  approximation='float64 folded AP vs float32 native policy; replay checked; no global optimality claim')
    best, seen = None, set()
    try:
        with core.timed(record):
            core.write_json(folder / 'summary.json', record)
            if record['torch_threads'] != 1 or record['torch_interop_threads'] != 1:
                raise ValueError('External worker initializer must set Torch intra/inter-op threads to 1')
            loading = record['input_verification_and_loading'] = {}
            with core.timed(loading):
                model, data = load_nominal(seed, folder, record)
            centers = np.concatenate([data['xi1'][data['train_indices']],
                                      data['xi2'][data['train_indices']]], axis=1)
            if centers.shape != (N_TRAIN, 6) or centers.dtype != np.float64:
                raise ValueError('Expected 40 raw float64 train centers')
            pools = [center[None, :].copy() for center in centers]
            for candidate_id in range(MAX_MP + 1):
                iteration_folder = folder / f'candidate_{candidate_id:02d}'
                iteration_folder.mkdir()
                for pool, center in zip(pools, centers):
                    check_membership(pool, center, radius)
                candidate = dict(candidate_id=candidate_id, seed=seed, radius=radius,
                                 state_sha256=state_hash(model), pool_sha256=pool_hash(pools),
                                 checkpoint=str(iteration_folder / 'candidate.pth'),
                                 ap_reference=str(iteration_folder / 'assessment.json'))
                record['iterations'].append(candidate)
                torch.save(core.weights(model), candidate['checkpoint'])
                Path(candidate['checkpoint']).chmod(0o444)
                candidate['checkpoint_sha256'] = core.digest(Path(candidate['checkpoint']))
                core.write_json(iteration_folder / 'pool.json', dict(
                    pool_sha256=candidate['pool_sha256'], pools=pools,
                    original_sample_ids=data['train_indices']))
                (iteration_folder / 'pool.json').chmod(0o444)
                points = assess_candidate(model, pools, centers, data, radius, candidate, iteration_folder)
                if candidate_id == 0:
                    if (abs(candidate['validation_mean'] - record['nominal_expected_validation']) > 1e-6
                            or abs(candidate['restricted_mean'] - record['nominal_expected_train']) > 1e-6):
                        raise ValueError('Cached nominal objective replay mismatch')
                if best is None or candidate['validation_mean'] < best['validation_mean']:
                    best = candidate
                core.write_json(iteration_folder / 'assessment.json', candidate)
                if candidate['separation_status'] == 'separation_satisfied':
                    record['terminal_status'] = 'separation_satisfied'
                    break
                if candidate['state_sha256'] in seen:
                    record.update(terminal_status='stagnation', terminal_reason='assessed_candidate_cycle_or_validation_restoration')
                    break
                seen.add(candidate['state_sha256'])
                if candidate_id == MAX_MP:
                    record['terminal_status'] = 'iteration_cap'
                    break
                additions = []
                for sample_id, (point, row) in enumerate(zip(points, candidate['samples'])):
                    if row['incumbent_violating'] and not row['membership']['exact_duplicate']:
                        pools[sample_id] = np.vstack([pools[sample_id], point])
                        additions.append(sample_id)
                candidate['added_sample_ids'] = additions
                if not additions:
                    record.update(terminal_status='stagnation', terminal_reason='no_new_violating_incumbents_unresolved_AP')
                    break
                training = candidate['mp'] = dict(seed=seed, radius=radius, method='robust_neural_MP',
                    mp_iteration=candidate_id + 1, incoming_state_sha256=candidate['state_sha256'],
                    pool_sha256=pool_hash(pools), output_candidate_id=candidate_id + 1)
                mp_folder = iteration_folder / 'mp'
                mp_folder.mkdir()
                core.write_json(mp_folder / 'pool.json', dict(pool_sha256=pool_hash(pools), pools=pools))
                (mp_folder / 'pool.json').chmod(0o444)
                per_ball = restricted_objective(pools)
                start = time.perf_counter()
                try:
                    model = core.train_model(model, lambda current: per_ball(current).mean(),
                        data['val_x1'], data['val_x2'], record['lr_net'], mp_folder, training,
                        epochs=3000, patience=300)
                    if training.get('status') != 'completed':
                        raise ValueError('Incomplete MP training')
                    saved = torch.load(mp_folder / 'selected.pth', map_location='cpu', weights_only=True)
                    returned = core.weights(model)
                    if saved.keys() != returned.keys() or any(
                            not torch.equal(saved[key], returned[key]) for key in saved):
                        raise ValueError('MP returned state differs from selected checkpoint')
                    record['mp_count'] += 1
                    training['returned_state_sha256'] = state_hash(model)
                finally:
                    candidate['mp_total_time_s'] = time.perf_counter() - start
                    core.write_json(iteration_folder / 'assessment.json', candidate)
                    core.write_json(folder / 'summary.json', record)
            record.update(complete=True, configuration_status='completed')
    except Exception as error:
        record['error'] = dict(type=type(error).__name__, message=str(error))
    finally:
        if not record['complete']:
            record.update(status='failed', configuration_status='failed', terminal_status='failure')
        selection = record['selection_and_checkpoint'] = {}
        try:
            with core.timed(selection):
                if best is not None:
                    selected = dict(candidate_id=best['candidate_id'], validation_mean=best['validation_mean'],
                                    state_sha256=best['state_sha256'], ap_reference=best['ap_reference'],
                                    separation_status=best['separation_status'])
                    name = 'selected.pth' if record['complete'] else 'provisional_selected.pth'
                    _copy(best['checkpoint'], folder / name, best['checkpoint_sha256'])
                    selected['checkpoint'] = str(folder / name)
                    if record['complete']:
                        record.update(selected_candidate_id=selected['candidate_id'],
                                      selected_checkpoint=selected['checkpoint'],
                                      selected_validation_mean=selected['validation_mean'],
                                      selected_state_sha256=selected['state_sha256'],
                                      selected_ap_reference=selected['ap_reference'],
                                      selected_separation_status=selected['separation_status'])
                    else:
                        record['provisional_best'] = selected
        except Exception as error:
            record.update(status='failed', complete=False, configuration_status='failed',
                          terminal_status='failure', finalization_error=dict(
                              type=type(error).__name__, message=str(error)))
        record.update(ap_sweeps_attempted=len(record['iterations']),
                      ap_sweeps_completed=sum(row.get('assessed', False) for row in record['iterations']),
                      ap_total_time_s=sum(row.get('ap_total_time_s', 0.) for row in record['iterations']),
                      mp_total_time_s=sum(row.get('mp_total_time_s', 0.) for row in record['iterations']),
                      evaluation_time_s=sum(row.get('evaluation', {}).get('wall_time_s', 0.) for row in record['iterations']))
        record.update(wall_time_s=time.perf_counter() - job_start, ended_at_utc=core.utc_now())
        record['wall_time_human'] = f"{record['wall_time_s']:.3f} s / {record['wall_time_s'] / 60:.3f} min"
        core.write_json(folder / 'summary.json', record)
    return core._json_value(record)