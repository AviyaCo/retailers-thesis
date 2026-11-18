"""Hash-bound continuation at the failed AP boundary, not a new CCG run.

The original assessed prefix remains historical evidence. Only its failed AP
sweep is redone under the advisory policy; all forty MPs share one budget.
"""
import ast
import copy
import json
from pathlib import Path
import platform
import shutil
import time
import traceback

import advisory as adversary
from continue_common import (PARENT, GREEN, MAX_MP, PROTOCOL, parent, check, core, digest,
                    read, metadata, np, torch, verify_sources, shipping_training,
                    legacy_ccg, definitions as _definitions)

train_model = shipping_training.train_model
validate_ap = adversary.validate_ap
hard_objective = adversary.hard_objective
restricted_objective = adversary.restricted_objective
state_hash, pool_hash = adversary.state_hash, adversary.pool_hash
check_membership, point_membership = adversary.check_membership, adversary.point_membership

EXPECTED_BOUNDARIES = {(50, 7, 1.5): 9, (200, 5, 1.5): 1,
                       (800, 6, .7): 3, (800, 7, .2): 1, (800, 9, .4): 1}
EVALUATION_PHASES = ('evaluation', 'smooth_training_replay',
                     'incoming_pool_evaluation', 'returned_pool_evaluation')


def _phase_totals(candidates):
    def duration(candidate, phase):
        item = candidate.get(phase)
        if item is None:
            return 0.0
        elapsed = item.get('wall_time_s')
        if elapsed is None:
            check(item.get('status') == 'running', 'Missing completed phase duration')
            return 0.0
        value = _finite(elapsed, f'{phase} wall time')
        check(value >= 0, 'Negative historical duration')
        return value

    return dict(ap_sweeps_attempted=len(candidates),
                ap_sweeps_completed=sum(c['assessed'] for c in candidates),
                ap_total_time_s=sum(duration(c, 'ap') for c in candidates),
                mp_total_time_s=sum(duration(c, 'mp_call') for c in candidates),
                evaluation_time_s=sum(duration(c, p) for c in candidates
                                      for p in EVALUATION_PHASES))


def _restore_boundary(record, nominal_record, manifest, centers, ids):
    """Validate every used artifact before accepting any cached candidate."""
    bindings = manifest['input_sha256']

    def bound(path, expected=None):
        path = Path(path).resolve()
        check(str(path) in bindings, f'Unbound resume input: {path}')
        expected = bindings[str(path)] if expected is None else expected
        check(bindings[str(path)] == expected and digest(path) == expected,
              f'Resume input hash mismatch: {path}')
        return path

    def pool_file(path, file_hash, numerical_hash):
        data = read(bound(path, file_hash))
        check(data['original_sample_ids'] == ids, 'Pool sample IDs differ')
        pools = [np.asarray(p, dtype=np.float64) for p in data['pools']]
        check(len(pools) == len(centers), 'Pool sample count differs')
        for pool, center in zip(pools, centers):
            check(pool.ndim == 2 and pool.shape[1] == 6 and len(pool) > 0
                  and np.isfinite(pool).all(), 'Malformed cached pool')
            check_membership(pool, center, record['radius'])
        check(pool_hash(pools) == data['pool_sha256'] == numerical_hash,
              'Numerical pool hash mismatch')
        return pools

    summary_path = bound(nominal_record['resume_summary_path'],
                         nominal_record['resume_summary_sha256'])
    prior = read(summary_path)
    root = summary_path.parent
    check(Path(prior['summary_path']).resolve() == summary_path
          and Path(prior['output_directory']).resolve() == root, 'Parent job path mismatch')
    check(prior['status'] == prior['configuration_status'] == 'failed'
          and prior['computational_job_status'] == 'failed'
          and not prior['complete'] and not prior['converged']
          and prior['terminal_status'] == 'failure'
          and prior['selected_checkpoint'] is None
          and prior['stopping_candidate_id'] is None, 'Parent must be a failed unselected job')
    for key in ('Ntotal', 'Ntrain', 'seed', 'radius'):
        check(prior[key] == record[key], f'Parent configuration mismatch: {key}')
    for key in ('lr_net', 'data_sha256', 'training_data_sha256'):
        check(prior[key] == nominal_record[key], f'Parent nominal mismatch: {key}')
    check(prior['nominal_checkpoint_sha256'] == nominal_record['checkpoint_sha256']
          and prior['nominal_state_sha256'] == nominal_record['state_sha256']
          and prior['nominal_expected_train'] == nominal_record['train_mean'],
          'Parent nominal checkpoint/loss differs')
    check(prior['max_mp'] == 40, 'Parent MP budget differs')
    candidates = prior['iterations']
    k = len(candidates) - 1
    check(type(prior['mp_count']) is int and prior['mp_count'] == k
          and 1 <= k <= MAX_MP, 'Parent boundary/MP count mismatch')
    check(EXPECTED_BOUNDARIES.get((record['Ntotal'], record['seed'], record['radius'])) == k,
          'Unapproved failed configuration/boundary')
    check(prior['ap_sweeps_attempted'] == k + 1 and prior['ap_sweeps_completed'] == k,
          'Parent AP coverage counts differ')

    prior_manifest = read(bound(root / 'inputs/provenance.json', prior['provenance_sha256']))
    check(prior_manifest['protocol'] == parent.PROTOCOL, 'Parent protocol differs')
    for key, hash_key, snapshot in (
            ('checkpoint', 'checkpoint_sha256', 'nominal.pth'),
            ('training_data_path', 'training_data_sha256', 'training_data.json')):
        original = bound(nominal_record[key], nominal_record[hash_key])
        check(prior_manifest['input_sha256'].get(str(original)) == nominal_record[hash_key],
              'Nominal input missing from parent manifest')
        bound(root / 'inputs' / snapshot, nominal_record[hash_key])
    data = read(root / 'inputs/training_data.json')
    check(data['data_sha256'] == record['data_sha256']
          and data['original_sample_ids'] == ids
          and np.array_equal(np.asarray(data['centers'], dtype=np.float64), centers),
          'Parent training input differs')
    _verify_checkpoint(root / 'inputs/nominal.pth', nominal_record['checkpoint_sha256'],
                       nominal_record['state_sha256'])

    expected_pool = pool_hash([c[None, :].copy() for c in centers])
    expected_checkpoint = nominal_record['checkpoint_sha256']
    expected_state = nominal_record['state_sha256']
    seen = set()
    previous_state = previous_training = None
    for index, candidate in enumerate(candidates):
        directory = root / f'candidate_{index:02d}'
        check(candidate['candidate_id'] == index, 'Noncontiguous parent prefix')
        for key in ('Ntotal', 'Ntrain', 'seed', 'radius'):
            check(candidate[key] == record[key], f'Cached candidate mismatch: {key}')
        check(Path(candidate['checkpoint']).resolve() == directory / 'candidate.pth'
              and Path(candidate['ap_reference']).resolve() == directory / 'assessment.json',
              'Cached candidate artifact path mismatch')
        bound(candidate['checkpoint'], candidate['checkpoint_sha256'])
        model = _verify_checkpoint(candidate['checkpoint'], candidate['checkpoint_sha256'],
                                   candidate['state_sha256'])
        check(candidate['checkpoint_sha256'] == expected_checkpoint
              and candidate['state_sha256'] == expected_state
              and candidate['pool_sha256'] == expected_pool,
              'Selected MP to next candidate chain mismatch')
        if previous_training is not None:
            check(candidate['pool_file_sha256'] == previous_training['pool_file_sha256'],
                  'Selected MP to next pool file hash mismatch')
        pools = pool_file(directory / 'pool.json', candidate['pool_file_sha256'], expected_pool)
        check(candidate['repeated_state'] == (expected_state in seen)
              and candidate['unchanged_from_previous'] == (expected_state == previous_state),
              'Cached repetition history mismatch')
        mirror = read(bound(candidate['ap_reference']))
        # The frozen controller writes assessment before _flush refreshes the
        # derived MP total. Accept only that documented zero-to-measured change.
        mp_time = candidate.get('mp_call', {}).get('wall_time_s', 0.)
        check(mirror.get('mp_total_time_s', 0.) in (0., mp_time)
              and candidate['mp_total_time_s'] == mp_time, 'Assessment MP timing mismatch')
        mirror['mp_total_time_s'] = mp_time
        check(mirror == candidate, 'Assessment mirror differs from parent summary')
        check(candidate['ap_total_time_s'] == candidate['ap']['wall_time_s'],
              'Assessment AP timing mismatch')
        samples = candidate['samples']
        check(len(samples) == len(ids), 'Incomplete cached sample coverage')
        for sample_id, sample in enumerate(samples):
            check(sample['sample_id'] == sample_id and sample['original_sample_id'] == ids[sample_id]
                  and sample['candidate_id'] == index and sample['candidate_sha256'] == expected_state
                  and sample['checkpoint_sha256'] == expected_checkpoint
                  and sample['pool_sha256'] == expected_pool, 'Cached sample identity mismatch')
            check(read(bound(directory / f'sample_{sample_id:02d}.json')) == sample,
                  'Sample mirror differs from assessment')
        failed = [i for i, s in enumerate(samples) if s['controller_status'] != 'completed']
        check(candidate['failed_sample_ids'] == failed, 'Failed sample IDs differ')
        if index == k:
            check(candidate['assessed'] is False and failed and candidate['ap']['status'] == 'failed',
                  'Boundary is not a failed AP sweep')
            check(not any(key in candidate for key in ('mp', 'mp_call', 'added_sample_ids',
                      'incoming_pool_evaluation', 'returned_pool_evaluation', 'green_pass'))
                  and not (directory / 'mp').exists(), 'Failed boundary has outgoing MP activity')
            break

        check(candidate['assessed'] is True and not failed and candidate['ap']['status'] == 'completed',
              'Cached prefix contains incomplete AP')
        decision = terminal_decision(candidate)
        check(candidate['green_gap'] == decision['green_gap']
              and candidate['green_pass'] is False and not decision['green_pass']
              and not candidate['repeated_state'], 'Cached prefix already terminal')
        check(abs(candidate['ap_incumbent_mean'] - float(np.mean([
                  s['incumbent']['cost'] for s in samples]))) <= 1e-12,
              'Cached AP mean differs from samples')
        smooth = candidate['smooth_training_replay']
        expected_loss = (nominal_record['train_mean'] if index == 0
                         else previous_training['selected_training_loss'])
        check(smooth['status'] == 'completed' and smooth['state_sha256'] == expected_state
              and abs(smooth['mean'] - expected_loss) <= 1e-6, 'Cached smooth replay mismatch')
        training = candidate['mp']
        mp_directory = directory / 'mp'
        check(read(bound(mp_directory / 'controller_training.json')) == training,
              'MP training mirror differs from assessment')
        check(training['status'] == training['controller_status'] == 'completed'
              and candidate['mp_call']['status'] == 'completed'
              and training['mp_iteration'] == training['output_candidate_id'] == index + 1
              and training['incoming_state_sha256'] == expected_state,
              'Incomplete or mislinked cached MP')
        outgoing = pool_file(mp_directory / 'pool.json', training['pool_file_sha256'],
                             training['pool_sha256'])
        reconstructed = [p.copy() for p in pools]
        additions = []
        for sample_id, sample in enumerate(samples):
            point = np.concatenate((sample['incumbent']['worst1'], sample['incumbent']['worst2']))
            membership = point_membership(point, pools[sample_id])
            check(sample['membership']['exact_duplicate'] == membership['exact_duplicate'],
                  'Cached AP membership differs')
            if sample['incumbent_violating'] and not membership['exact_duplicate']:
                reconstructed[sample_id] = np.vstack((pools[sample_id], point))
                additions.append(sample_id)
        check(additions and candidate['added_sample_ids'] == additions
              and pool_hash(reconstructed) == pool_hash(outgoing), 'Cached MP pool transition differs')
        incoming = candidate['incoming_pool_evaluation']
        returned = candidate['returned_pool_evaluation']
        first, best, selected = (_finite(training[key], key) for key in
                                 ('first_train', 'best_train', 'selected_training_loss'))
        check(incoming['status'] == returned['status'] == 'completed'
              and incoming['state_sha256'] == expected_state
              and returned['state_sha256'] == training['returned_state_sha256']
              and abs(first - incoming['mean']) <= 1e-6
              and best <= first and selected <= first
              and (best == first or best < first - PROTOCOL['improvement_threshold'])
              and abs(best - selected) <= 1e-6 and abs(returned['mean'] - selected) <= 1e-6,
              'Cached MP smooth best/unchanged guard differs')
        check(training['unchanged_weights'] == (training['returned_state_sha256'] == expected_state),
              'Cached MP unchanged-state flag differs')
        selected_path = bound(mp_directory / 'selected.pth', training['selected_checkpoint_sha256'])
        _verify_checkpoint(selected_path, training['selected_checkpoint_sha256'],
                           training['returned_state_sha256'])
        seen.add(expected_state)
        previous_state, previous_training = expected_state, training
        expected_pool = training['pool_sha256']
        expected_checkpoint = training['selected_checkpoint_sha256']
        expected_state = training['returned_state_sha256']

    prefix = copy.deepcopy(candidates[:k])
    prefix_timing = _phase_totals(prefix)
    failed_timing = _phase_totals([candidates[k]])
    all_timing = _phase_totals(candidates)
    for key in ('ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s'):
        check(abs(_finite(prior[key], key) - all_timing[key]) <= 1e-9,
              f'Parent aggregate timing differs: {key}')
    check(_finite(prior['wall_time_s'], 'parent job wall time') >= 0, 'Negative parent job time')
    prior_timing_keys = ('wall_time_s', 'wall_time_human', 'started_at_utc', 'ended_at_utc',
                        'status', 'mp_count', 'ap_total_time_s', 'mp_total_time_s',
                        'evaluation_time_s', 'workers', 'ccg_workers', 'torch_threads',
                        'gurobi_threads', 'environment', 'Ntotal', 'Ntrain', 'seed', 'radius')
    record.update(iterations=prefix, mp_count=k, cached_mp_count=k,
                  cached_prefix_candidate_count=k, resume_candidate_id=k,
                  resume_summary_path=str(summary_path), resume_summary_sha256=digest(summary_path),
                  cached_prefix_timing=prefix_timing, cached_failed_assessment_timing=failed_timing,
                  cached_prior_job_timing={key: copy.deepcopy(prior[key]) for key in prior_timing_keys},
                  historical_failed_assessment=dict(path=candidates[k]['ap_reference'],
                      sha256=bindings[str(Path(candidates[k]['ap_reference']).resolve())],
                      candidate_id=k, status='failed', reused_as_assessment=False),
                  cached_timing_scope='persisted parent only; excluded from current invocation elapsed',
                  timing_scope='current invocation including resume verification and failed AP redo; cached work excluded')
    return (model, pools, Path(candidates[k]['checkpoint']), candidates[k]['checkpoint_sha256'],
            seen, previous_state, copy.deepcopy(previous_training), k)


_RESUME_CHANGES = {
    '_flush': (("    candidates = record['iterations']",
                "    candidates = record['iterations'][record.get('cached_prefix_candidate_count', 0):]"),),
    'run_configuration': ((
        "        pools = [center[None, :].copy() for center in centers]\n"
        "        checkpoint_source = inputs / 'nominal.pth'\n"
        "        checkpoint_hash = nominal_record['checkpoint_sha256']\n"
        "        seen = set()\n"
        "        previous_state = previous_training = None\n"
        "        for candidate_id in range(MAX_MP + 1):",
        "        with core.timed(record.setdefault('boundary_verification', {})):\n"
        "            (model, pools, checkpoint_source, checkpoint_hash, seen,\n"
        "             previous_state, previous_training, start_candidate) = _restore_boundary(\n"
        "                record, nominal_record, manifest, centers, ids)\n"
        "        for candidate_id in range(start_candidate, MAX_MP + 1):"),),
}


def _load_shipping_definitions(namespace, adaptations):
    """Execute only frozen definition calls, never the old shipping imports.

Each original shipping hard/smooth adaptation is retained, then the additional
exact-match boundary/timing anchors are applied by common.definitions.
"""
    source = PARENT / 'shipping_ccg.py'
    _definitions(namespace, source, ('_smooth_pool_mean',))

    def adapted_definitions(target, path, names, changes=()):
        extra = tuple(change for name in names for change in adaptations.get(name, ()))
        _definitions(target, path, names, changes=tuple(changes) + extra)

    namespace['definitions'] = adapted_definitions
    calls = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name) and node.value.func.id == 'definitions']
    check(len(calls) == 5, 'Frozen shipping definition-call inventory changed')
    try:
        for node in calls:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    finally:
        namespace['definitions'] = _definitions


_load_shipping_definitions(globals(), _RESUME_CHANGES)
_controller_flush = _flush
_controller_run = run_configuration


def _flush(record):
    """Raw candidate timings stay historical; public phase totals are current."""
    k = record.setdefault('cached_prefix_candidate_count', 0)
    record.setdefault('cached_mp_count', 0)
    record['new_mp_count'] = record['mp_count'] - record['cached_mp_count']
    current = record['iterations'][k:]
    cached = record.get('cached_prefix_timing', {})
    totals = _phase_totals(current)
    record['logical_prefix_totals'] = {
        key: cached.get(key, 0) + value for key, value in totals.items()}
    record['logical_prefix_totals'].update(mp_count=record['mp_count'],
        timing_scope='sum of cached prefix and current phases; NOT elapsed wall time')
    for key in ('ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s'):
        record[f'cached_prefix_{key}'] = cached.get(key, 0.)
    if 'wall_time_s' in record:
        record['current_invocation_elapsed_time_s'] = record['wall_time_s']
    _controller_flush(record)


def run_configuration(n_total, seed, radius, output_root, nominal_record, provenance_path):
    """The frozen public API, including current elapsed and cached timing fields."""
    result = _controller_run(n_total, seed, radius, output_root, nominal_record, provenance_path)
    record = read(result['summary_path'])
    keys = ('cached_mp_count', 'new_mp_count', 'cached_prefix_candidate_count',
            'resume_candidate_id', 'resume_summary_path', 'resume_summary_sha256',
            'cached_prefix_timing', 'cached_prior_job_timing', 'cached_failed_assessment_timing',
            'historical_failed_assessment', 'boundary_verification', 'cached_timing_scope',
            'timing_scope', 'current_invocation_elapsed_time_s', 'logical_prefix_totals',
            'cached_prefix_ap_total_time_s', 'cached_prefix_mp_total_time_s',
            'cached_prefix_evaluation_time_s', 'cached_nominal')
    result.update({key: record[key] for key in keys if key in record})
    return core._json_value(result)