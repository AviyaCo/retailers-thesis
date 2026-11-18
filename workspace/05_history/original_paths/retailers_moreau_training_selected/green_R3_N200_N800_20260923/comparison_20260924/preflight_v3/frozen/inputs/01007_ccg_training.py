"""Training-only CCG; terminal policy selection, never intermediate held-out selection.

run_configuration(seed, radius, output_root, nominal_record, provenance_path)
creates output_root/seed_{seed}/radius_{radius:g} exclusively. Existing jobs raise
FileExistsError, without modifying them. Other ordinary execution failures are
persisted and returned with no published selection. The return value is compact;
summary_path and summary_sha256 identify the complete, streamed job record.

The parent must supply a freshly trained per-seed LR winner, not a historical
winner. In addition to its checkpoint/state/data hashes, lr_net, train_mean and
training_timing, nominal_record must contain training_data_path and
training_data_sha256. That JSON has exactly centers (40 raw float64 paired
six-vectors), original_sample_ids (40 distinct integers in [0,50)), and
data_sha256 (the parent's original-data hash). Its byte hash authenticates this
train-only subset; the original 50-row hash cannot be recomputed from 40 rows.
No split regeneration or held-out data loading takes place here.

The parent owns cross-job concurrency and later evaluation. Workers must already
have Torch intra/inter-op threads set to one. MAX_MP is fixed in production;
tests may patch the module constant, not expose a different experiment budget.
"""
import json
from pathlib import Path
import platform
import shutil
import time
import traceback

from common import (PROTOCOL, adversary, check, core, legacy_ccg, metadata,
                    np, torch, verify_sources)
from training import train_model

# Reuse only the frozen numerical/pool helpers, never the old controller/trainer.
restricted_objective = legacy_ccg.restricted_objective
validate_ap = legacy_ccg.validate_ap
check_membership = legacy_ccg.check_membership
point_membership = legacy_ccg.point_membership
pool_hash = legacy_ccg.pool_hash
state_hash = legacy_ccg.state_hash
N_TRAIN = 40
MAX_MP = 40


def _error(error):
    return dict(type=type(error).__name__, message=str(error),
                traceback=traceback.format_exc())


def _finite(value, name):
    check(not isinstance(value, (bool, str)) and np.isscalar(value)
          and np.isfinite(value), f'Nonfinite/malformed {name}: {value}')
    return float(value)


def _copy(source, destination, expected):
    """Byte-identical private snapshot; never link or overwrite an artifact."""
    source, destination = Path(source), Path(destination)
    check(core.digest(source) == expected, f'Input hash mismatch: {source}')
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    check(core.digest(destination) == expected, f'Copy hash mismatch: {source}')
    destination.chmod(0o444)


def _verify_checkpoint(path, byte_hash, learned_hash):
    check(core.digest(path) == byte_hash, f'Checkpoint hash mismatch: {path}')
    model = core.model_from_state(torch.load(path, map_location='cpu', weights_only=True))
    check(state_hash(model) == learned_hash, f'Checkpoint state mismatch: {path}')
    return model


def _save_pool(path, pools, ids):
    core.write_json(path, dict(pool_sha256=pool_hash(pools), pools=pools,
                               original_sample_ids=ids))
    path.chmod(0o444)
    return core.digest(path)


def _pool_mean(model, pools, record):
    frozen = state_hash(model)
    with core.timed(record), torch.no_grad():
        model.eval()
        values = restricted_objective(pools)(model).detach().cpu().numpy()
        check(values.shape == (N_TRAIN,) and np.isfinite(values).all(),
              'Expected forty finite restricted-pool maxima')
        record.update(mean=_finite(float(values.mean()), 'restricted mean'),
                      per_ball_costs=values.tolist(), sample_count=N_TRAIN,
                      state_sha256=frozen)
        check(state_hash(model) == frozen, 'Pool evaluation changed policy')
    return values


def _flush(record):
    """Durations are non-overlapping controller phases, not a sweep elapsed sum."""
    candidates = record['iterations']
    record.update(ap_sweeps_attempted=len(candidates),
                  ap_sweeps_completed=sum(c.get('assessed', False) for c in candidates),
                  ap_total_time_s=sum(c.get('ap', {}).get('wall_time_s', 0.) for c in candidates),
                  mp_total_time_s=sum(c.get('mp_call', {}).get('wall_time_s', 0.) for c in candidates),
                  evaluation_time_s=sum(
                      c.get('evaluation', {}).get('wall_time_s', 0.)
                      + c.get('incoming_pool_evaluation', {}).get('wall_time_s', 0.)
                      + c.get('returned_pool_evaluation', {}).get('wall_time_s', 0.)
                      for c in candidates))
    for candidate in candidates:
        candidate['ap_total_time_s'] = candidate.get('ap', {}).get('wall_time_s', 0.)
        candidate['mp_total_time_s'] = candidate.get('mp_call', {}).get('wall_time_s', 0.)
    core.write_json(record['summary_path'], record)


def assess_candidate(model, pools, centers, original_sample_ids, radius,
                     candidate, folder, publish):
    """Frozen train40 pool evaluation plus all40 fresh APs, even after AP failure.

    solve_ap rebuilds fixed-weight coefficients/bounds and performs native replay
    on every call. validate_ap retains the exact frozen strict-status, finite,
    membership, replay, ordering, duplicate and separation gates.
    """
    frozen = candidate['state_sha256']
    candidate.update(assessed=False, separation_status='unassessed', samples=[],
                     failed_sample_ids=[])
    evaluation = candidate['evaluation'] = {}
    maxima = _pool_mean(model, pools, evaluation)
    check(state_hash(model) == frozen, 'Candidate changed before AP')
    candidate.update(pool_costs=maxima.tolist(), restricted_mean=evaluation['mean'])
    ap_record = candidate['ap'] = metadata(candidate['seed'], radius, 'fixed_policy_AP_sweep')
    ap_record.update(candidate_id=candidate['candidate_id'], sample_count=N_TRAIN,
                     ap_workers=1)
    points = []
    ap_start = time.perf_counter()
    try:
        with core.timed(ap_record):
            for sample_id in range(N_TRAIN):
                row = dict(seed=candidate['seed'], radius=radius, sample_id=sample_id,
                           original_sample_id=int(original_sample_ids[sample_id]),
                           candidate_id=candidate['candidate_id'], candidate_sha256=frozen,
                           checkpoint_sha256=candidate['checkpoint_sha256'],
                           pool_sha256=candidate['pool_sha256'],
                           controller_started_at_utc=core.utc_now())
                start = time.perf_counter()
                try:
                    check(state_hash(model) == frozen, 'Frozen AP candidate changed')
                    result = adversary.solve_ap(model, centers[sample_id, :3],
                                                centers[sample_id, 3:], radius, row)
                    point = validate_ap(result, row, sample_id, centers[sample_id],
                                        pools[sample_id], radius, float(maxima[sample_id]))
                    check(state_hash(model) == frozen, 'Frozen AP candidate changed')
                    points.append(point)
                    row['controller_status'] = 'completed'
                except Exception as error:
                    candidate['failed_sample_ids'].append(sample_id)
                    row.update(controller_status='failed', controller_error=_error(error))
                finally:
                    row.update(controller_wall_time_s=time.perf_counter() - start,
                               controller_ended_at_utc=core.utc_now())
                    candidate['samples'].append(row)
                    core.write_json(folder / f'sample_{sample_id:02d}.json', row)
                    ap_record['wall_time_s'] = time.perf_counter() - ap_start
                    core.write_json(folder / 'assessment.json', candidate)
                    publish()
            check(not candidate['failed_sample_ids'],
                  f"Failed/incomplete AP samples: {candidate['failed_sample_ids']}")
            check([row['sample_id'] for row in candidate['samples']] == list(range(N_TRAIN)),
                  'Missing/duplicate AP coverage')
            check(pool_hash(pools) == candidate['pool_sha256'], 'AP mutated scenario pool')
            check(core.digest(folder / 'pool.json') == candidate['pool_file_sha256'],
                  'AP pool snapshot changed')
            _verify_checkpoint(candidate['checkpoint'], candidate['checkpoint_sha256'], frozen)
            check(state_hash(model) == frozen, 'Frozen AP candidate changed')
            separated = all(row['residual'] <= PROTOCOL['separation_tolerance']
                            for row in candidate['samples'])
            candidate.update(assessed=True,
                separation_status='separation_satisfied' if separated else 'violating',
                ap_incumbent_mean=float(np.mean([r['incumbent']['cost'] for r in candidate['samples']])),
                ap_bound_mean=float(np.mean([r['bound'] for r in candidate['samples']])))
    finally:
        candidate['ap_total_time_s'] = ap_record.get('wall_time_s', 0.)
        core.write_json(folder / 'assessment.json', candidate)
        publish()
    return points


def run_configuration(seed, radius, output_root, nominal_record, provenance_path):
    """Return a compact JSON-safe job summary and the full summary's byte hash."""
    job_start = time.perf_counter()
    started = core.utc_now()
    folder = Path(output_root).resolve() / f'seed_{seed}' / f'radius_{radius:g}'
    folder.mkdir(parents=True, exist_ok=False)
    record = metadata(seed, radius, 'robust_neural_CCG')
    record.update(status='running', configuration_status='running', complete=False,
        converged=False, terminal_status='running', started_at_utc=started,
        output_directory=str(folder), summary_path=str(folder / 'summary.json'),
        iterations=[], mp_count=0, max_mp=MAX_MP, max_epochs=3000, patience=300,
        selected_candidate_id=None, selected_checkpoint=None, selected_checkpoint_sha256=None,
        selected_state_sha256=None, selected_ap_reference=None, selected_separation_status=None,
        closure_with_unchangedweights=False, global_robust_optimality_certified=False,
        data_access='hashed train-only JSON; no split regeneration or held-out loading',
        selection_rule='terminal fully AP-assessed candidate',
        nominal_origin_contract='parent supplies fresh per-seed training-only LR-search winner',
        timing_scope='current job invocation including loading; prior nominal training excluded',
        ap_timing_scope='AP sweep including native replay, integrity gates and sample persistence',
        mp_timing_scope='train_model call only; pool checks counted separately as evaluation',
        ccg_workers=PROTOCOL['ccg_workers'], ap_workers_per_job=1,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        environment=dict(platform=platform.platform(), hostname=platform.node()), error=None)
    try:
        _flush(record)
        with core.timed(record.setdefault('input_verification_and_loading', {})):
            check(not isinstance(seed, bool) and seed in PROTOCOL['data_seeds'], 'Unapproved seed')
            check(not isinstance(radius, bool) and np.isfinite(radius)
                  and radius in PROTOCOL['radii'], 'Unapproved radius')
            check(record['torch_threads'] == record['torch_interop_threads'] == 1,
                  'Worker must set Torch intra/inter-op threads to one')
            check(PROTOCOL['max_mp'] == 40 and PROTOCOL['max_epochs'] == 3000
                  and PROTOCOL['patience'] == 300, 'Unexpected training/CCG protocol')
            check(PROTOCOL['separation_tolerance'] == legacy_ccg.SEPARATION_TOLERANCE
                  and PROTOCOL['membership_tolerance'] == legacy_ccg.MEMBERSHIP_TOLERANCE
                  and PROTOCOL['ap_replay_tolerance'] == adversary.REPLAY_TOLERANCE,
                  'Frozen AP gates differ from declared protocol')
            inputs = folder / 'inputs'
            inputs.mkdir()
            provenance_path = Path(provenance_path).resolve()
            record['provenance_sha256'] = core.digest(provenance_path)
            _copy(provenance_path, inputs / 'provenance.json', record['provenance_sha256'])
            manifest = json.loads((inputs / 'provenance.json').read_text())
            verify_sources(manifest)
            check(nominal_record['seed'] == seed, 'Nominal seed mismatch')
            check(nominal_record['lr_net'] in PROTOCOL['lr_grid'], 'Unapproved nominal LR')
            record.update(lr_net=nominal_record['lr_net'], data_sha256=nominal_record['data_sha256'],
                nominal_expected_train=_finite(nominal_record['train_mean'], 'nominal train mean'),
                nominal_checkpoint_sha256=nominal_record['checkpoint_sha256'],
                nominal_state_sha256=nominal_record['state_sha256'],
                training_data_sha256=nominal_record['training_data_sha256'])
            timing = nominal_record['training_timing']
            check(timing['status'] == 'completed'
                  and _finite(timing['wall_time_s'], 'nominal training time') >= 0,
                  'Require completed measured nominal training')
            record['cached_nominal'] = dict(training_timing=timing, reused=True, retrained=False,
                timing_scope='persisted parent nominal phase; excluded from current CCG time')
            _copy(nominal_record['training_data_path'], inputs / 'training_data.json',
                  nominal_record['training_data_sha256'])
            data = json.loads((inputs / 'training_data.json').read_text())
            check(set(data) == {'centers', 'original_sample_ids', 'data_sha256'},
                  'Training JSON must contain only centers, original_sample_ids, data_sha256')
            check(data['data_sha256'] == nominal_record['data_sha256'], 'Original-data hash mismatch')
            centers = np.asarray(data['centers'], dtype=np.float64)
            ids = data['original_sample_ids']
            check(centers.shape == (N_TRAIN, 6) and np.isfinite(centers).all()
                  and (centers >= 0).all(), 'Expected forty finite nonnegative paired6 centers')
            check(len(ids) == N_TRAIN and all(type(i) is int and 0 <= i < 50 for i in ids)
                  and len(set(ids)) == N_TRAIN, 'Malformed original training sample IDs')
            del data
            _copy(nominal_record['checkpoint'], inputs / 'nominal.pth',
                  nominal_record['checkpoint_sha256'])
            model = _verify_checkpoint(inputs / 'nominal.pth', nominal_record['checkpoint_sha256'],
                                       nominal_record['state_sha256'])
        pools = [center[None, :].copy() for center in centers]
        checkpoint_source = inputs / 'nominal.pth'
        checkpoint_hash = nominal_record['checkpoint_sha256']
        seen = set()
        previous_state = None
        previous_training = None
        for candidate_id in range(MAX_MP + 1):
            iteration_folder = folder / f'candidate_{candidate_id:02d}'
            iteration_folder.mkdir()
            for pool, center in zip(pools, centers):
                check_membership(pool, center, radius)
            frozen = state_hash(model)
            candidate = dict(candidate_id=candidate_id, seed=seed, radius=radius,
                state_sha256=frozen, pool_sha256=pool_hash(pools),
                checkpoint=str(iteration_folder / 'candidate.pth'),
                checkpoint_sha256=checkpoint_hash,
                ap_reference=str(iteration_folder / 'assessment.json'),
                repeated_state=frozen in seen, unchanged_from_previous=previous_state == frozen,
                closure_with_unchangedweights=False, assessed=False)
            record['iterations'].append(candidate)
            _copy(checkpoint_source, candidate['checkpoint'], checkpoint_hash)
            _verify_checkpoint(candidate['checkpoint'], checkpoint_hash, frozen)
            candidate['pool_file_sha256'] = _save_pool(iteration_folder / 'pool.json', pools, ids)
            try:
                points = assess_candidate(model, pools, centers, ids, radius, candidate,
                                          iteration_folder, lambda: _flush(record))
            except Exception as error:
                candidate['controller_error'] = _error(error)
                raise
            finally:
                core.write_json(iteration_folder / 'assessment.json', candidate)
                _flush(record)
            if candidate_id == 0:
                check(abs(candidate['restricted_mean'] - record['nominal_expected_train']) <= 1e-6,
                      'Nominal training-pool replay mismatch')
            else:
                check(abs(candidate['restricted_mean'] - previous_training['selected_training_loss']) <= 1e-6,
                      'Next candidate differs from selected MP training loss')
            separated = candidate['separation_status'] == 'separation_satisfied'
            candidate['closure_with_unchangedweights'] = separated and candidate['unchanged_from_previous']
            record['closure_with_unchangedweights'] = candidate['closure_with_unchangedweights']
            core.write_json(iteration_folder / 'assessment.json', candidate)
            _flush(record)
            # Preserve the original ordering, but record repetition before closure.
            if separated:
                record.update(terminal_status='separation_satisfied', converged=True,
                              terminal_reason='fixed-policy pool closure, not global MP optimality')
                break
            if candidate['repeated_state']:
                record.update(terminal_status='stagnation', terminal_reason='repeated_assessed_state_without_separation')
                break
            seen.add(frozen)
            if candidate_id == MAX_MP:
                record.update(terminal_status='iteration_cap', terminal_reason='forty_MP_budget_exhausted')
                break
            additions = []
            for sample_id, (point, row) in enumerate(zip(points, candidate['samples'])):
                if row['incumbent_violating'] and not row['membership']['exact_duplicate']:
                    pools[sample_id] = np.vstack([pools[sample_id], point])
                    additions.append(sample_id)
            candidate['added_sample_ids'] = additions
            if not additions:
                record.update(terminal_status='stagnation', terminal_reason='no_new_violating_incumbents')
                break
            mp_folder = iteration_folder / 'mp'
            mp_folder.mkdir()
            training = candidate['mp'] = metadata(seed, radius, 'robust_neural_MP')
            training.update(mp_iteration=candidate_id + 1, sample_count=N_TRAIN,
                incoming_state_sha256=frozen, output_candidate_id=candidate_id + 1,
                pool_sha256=pool_hash(pools),
                pool_file_sha256=_save_pool(mp_folder / 'pool.json', pools, ids))
            try:
                incoming = candidate['incoming_pool_evaluation'] = {}
                _pool_mean(model, pools, incoming)
                training['incoming_pool_mean'] = incoming['mean']
                per_ball = restricted_objective(pools)
                _flush(record)
                with core.timed(candidate.setdefault('mp_call', {})):
                    model = train_model(model, lambda current: per_ball(current).mean(),
                                        record['lr_net'], mp_folder, training, epochs=3000, patience=300)
                check(training.get('status') == 'completed', 'Incomplete MP training')
                record['mp_count'] += 1
                first = _finite(training['first_train'], 'MP first train')
                best = _finite(training['best_train'], 'MP best train')
                selected = _finite(training['selected_training_loss'], 'MP selected training loss')
                check(abs(first - incoming['mean']) <= 1e-6, 'MP incoming/first training loss mismatch')
                check(best <= first and selected <= first, 'MP selected training loss worsened incoming state')
                check(best == first or best < first - PROTOCOL['improvement_threshold'],
                        'MP selected improvement violates training checkpoint threshold')
                check(abs(best - selected) <= 1e-6, 'MP best/selected training loss mismatch')
                returned = candidate['returned_pool_evaluation'] = {}
                _pool_mean(model, pools, returned)
                check(abs(returned['mean'] - selected) <= 1e-6, 'Returned MP pool loss differs from selected loss')
                check(pool_hash(pools) == training['pool_sha256']
                      and core.digest(mp_folder / 'pool.json') == training['pool_file_sha256'],
                      'MP scenario pool changed')
                checkpoint_source = mp_folder / 'selected.pth'
                checkpoint_hash = core.digest(checkpoint_source)
                returned_hash = state_hash(model)
                _verify_checkpoint(checkpoint_source, checkpoint_hash, returned_hash)
                checkpoint_source.chmod(0o444)
                training.update(returned_state_sha256=returned_hash,
                    selected_checkpoint_sha256=checkpoint_hash,
                    returned_pool_mean=returned['mean'],
                    same_pool_change=returned['mean'] - incoming['mean'],
                    unchanged_weights=returned_hash == frozen,
                    controller_status='completed')
                previous_state, previous_training = frozen, training
            except Exception as error:
                training.update(controller_status='failed', controller_error=_error(error))
                raise
            finally:
                core.write_json(mp_folder / 'controller_training.json', training)
                core.write_json(iteration_folder / 'assessment.json', candidate)
                _flush(record)
        with core.timed(record.setdefault('selection_and_checkpoint', {})):
            terminal = record['iterations'][-1]
            check(terminal['assessed'], 'Terminal policy lacks full AP coverage')
            verify_sources(manifest)
            check(core.digest(provenance_path) == record['provenance_sha256'], 'Provenance changed')
            for path, expected in ((inputs / 'provenance.json', record['provenance_sha256']),
                                   (inputs / 'training_data.json', nominal_record['training_data_sha256']),
                                   (inputs / 'nominal.pth', nominal_record['checkpoint_sha256']),
                                   (nominal_record['checkpoint'], nominal_record['checkpoint_sha256']),
                                   (nominal_record['training_data_path'], nominal_record['training_data_sha256'])):
                check(core.digest(path) == expected, f'Frozen input changed: {path}')
            _verify_checkpoint(terminal['checkpoint'], terminal['checkpoint_sha256'], terminal['state_sha256'])
            selected_path = folder / 'selected.pth'
            # Verify the private copy before atomically publishing the selection.
            pending = folder / 'unpublished_selected.pth'
            _copy(terminal['checkpoint'], pending, terminal['checkpoint_sha256'])
            _verify_checkpoint(pending, terminal['checkpoint_sha256'], terminal['state_sha256'])
            pending.rename(selected_path)
            record.update(selected_candidate_id=terminal['candidate_id'],
                selected_checkpoint=str(selected_path), selected_checkpoint_sha256=terminal['checkpoint_sha256'],
                selected_state_sha256=terminal['state_sha256'], selected_ap_reference=terminal['ap_reference'],
                selected_separation_status=terminal['separation_status'],
                selected_restricted_mean=terminal['restricted_mean'])
        record.update(status='completed', configuration_status='completed', complete=True)
    except Exception as error:
        record.update(status='failed', configuration_status='failed', complete=False,
            converged=False, terminal_status='failure', error=_error(error),
            selected_candidate_id=None, selected_checkpoint=None, selected_checkpoint_sha256=None,
            selected_state_sha256=None, selected_ap_reference=None, selected_separation_status=None)
    finally:
        record.update(wall_time_s=time.perf_counter() - job_start, ended_at_utc=core.utc_now())
        record['wall_time_human'] = f"{record['wall_time_s']:.3f} s / {record['wall_time_s'] / 60:.3f} min"
        _flush(record)
    keys = ('experiment', 'method', 'seed', 'radius', 'status', 'configuration_status',
            'complete', 'converged', 'terminal_status', 'mp_count', 'ap_sweeps_attempted',
            'ap_sweeps_completed', 'selected_candidate_id', 'selected_checkpoint',
            'selected_checkpoint_sha256', 'selected_state_sha256', 'selected_ap_reference',
            'selected_separation_status', 'closure_with_unchangedweights',
            'global_robust_optimality_certified', 'summary_path', 'output_directory',
            'started_at_utc', 'ended_at_utc', 'wall_time_s', 'wall_time_human',
            'ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s', 'error')
    result = {key: record[key] for key in keys}
    result['summary_sha256'] = core.digest(record['summary_path'])
    return core._json_value(result)