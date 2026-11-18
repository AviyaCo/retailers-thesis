"""Complete only the 31 missing mean-rule prefixes, without modifying old runs."""
from common import (core, np, torch, configure_worker, check, metadata, PROTOCOL)
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import argparse
import json
import multiprocessing
from pathlib import Path
import platform
import shlex
import sys
import time
import traceback

import ccg_training as ccg
from training import train_model
import update_comparison as u

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'short_rule_completion'
FROZEN = OUTPUT / 'frozen'
WORKERS = 16


def verify():
    path = FROZEN / 'manifest.json'
    check(core.digest(path) == (FROZEN / 'manifest.sha256').read_text().strip(), 'Manifest changed')
    m = json.loads(path.read_text())
    for source, info in m['files'].items():
        check(core.digest(source) == core.digest(FROZEN / info['copy']) == info['sha256'],
              f'Frozen input changed: {source}')
    return m


def freeze():
    start = time.perf_counter()
    OUTPUT.mkdir(exist_ok=False)
    FROZEN.mkdir()
    m = dict(status='freezing', started_at_utc=core.utc_now(), files={}, jobs=[],
             experiment='retailer_missing_mean_rule_completion_20260916',
             workers=WORKERS, evaluation_workers=8, threads=1, max_mp=40,
             protocol='SHORT_COMPLETION_PROTOCOL.md', new_nominal_training=False,
             selection='first eligible mean gap < .01; earliest minimum AP through stop',
             timing_clock='time.perf_counter', timing_unit='seconds')

    def capture(path, expected=None):
        path = Path(path).resolve()
        digest = core.digest(path)
        check(expected is None or digest == expected, f'Input mismatch: {path}')
        if str(path) not in m['files']:
            dest = FROZEN / f'{len(m["files"]):04d}_{path.name}'
            ccg._copy(path, dest, digest)
            m['files'][str(path)] = dict(copy=dest.name, sha256=digest)
        return m['files'][str(path)]

    try:
        previous = u.verify()
        for source, info in previous['files'].items():
            capture(source, info['sha256'])
        for name in ('fill_short_completion.py', 'test_short_completion.py',
                     'report_short_completion.py', 'SHORT_COMPLETION_PROTOCOL.md', 'launch_short_completion.sh'):
            capture(HERE / name)
        for name in ('plot_data.json', 'summary.json', 'frozen/manifest.json'):
            capture(u.OUTPUT / name)
        train = json.loads((u.TRAIN / 'summary.json').read_text())
        missing = {(r['seed'], r['radius']) for r in previous['recovery'] if not r['recoverable']}
        expected = {(s, r) for s in range(10) for r in (.001, .003, .01)} | {(0, .03)}
        check(missing == expected, 'Missing cohort changed')
        for job in sorted(train['ccg']['jobs'], key=lambda j: (j['radius'], j['seed'])):
            if (job['seed'], job['radius']) not in missing:
                continue
            record = json.loads(Path(job['summary_path']).read_text())
            check(record['status'] == 'completed' and record['mp_count'] == 0
                  and len(record['iterations']) == 1, 'Only initial-AP continuations authorized')
            a = record['iterations'][0]
            paths = dict(summary=job['summary_path'], checkpoint=a['checkpoint'],
                         pool=str(Path(a['checkpoint']).parent / 'pool.json'),
                         training_data=str(Path(job['summary_path']).parent / 'inputs/training_data.json'))
            hashes = dict(summary=job['summary_sha256'], checkpoint=a['checkpoint_sha256'],
                          pool=a['pool_file_sha256'], training_data=record['training_data_sha256'])
            inputs = {name: dict(source=path, **capture(path, hashes[name])) for name, path in paths.items()}
            m['jobs'].append(dict(seed=job['seed'], radius=job['radius'], inputs=inputs))
        # Timing references are frozen, but never enter continuation workers.
        import report_short_completion as reporting
        for base, pattern in ((reporting.SCS_SELECTED, 'rho_*/seed_*_summary.json'),
                              (reporting.SCS_NOMINAL, 'seed_*_summary.json'),
                              (u.c.SWEEP, 'rho_*/seed_*_summary.json')):
            for path in sorted(base.glob(pattern)):
                capture(path)
        m.update(status='frozen', ended_at_utc=core.utc_now(), wall_time_s=time.perf_counter()-start)
        core.write_json(FROZEN / 'manifest.json', m)
        (FROZEN / 'manifest.json').chmod(0o444)
        (FROZEN / 'manifest.sha256').write_text(core.digest(FROZEN / 'manifest.json') + '\n')
        verify()
        print(f'Frozen {len(m["jobs"])} continuations', flush=True)
    except BaseException:
        core.write_json(OUTPUT / 'freeze_failure.json', dict(status='failed', error=traceback.format_exc(),
                        wall_time_s=time.perf_counter()-start))
        raise


def mean_stop(candidate):
    check(candidate['assessed'] and len(candidate['samples']) == 40
          and all(r['controller_status'] == 'completed' for r in candidate['samples']),
          'Mean stopping requires all forty validated APs')
    ap, pool = candidate['ap_incumbent_mean'], candidate['restricted_mean']
    check(np.isfinite([ap, pool]).all(), 'Nonfinite mean diagnostic')
    gap = abs(ap-pool) / max(abs(pool), 1e-9)
    return candidate['candidate_id'] >= 1 and gap < .01, gap


def add_points(pools, points, samples):
    check(len(points) == len(samples) == len(pools) == 40, 'Incomplete insertion coverage')
    added = []
    for i, (point, row) in enumerate(zip(points, samples)):
        if row['incumbent_violating'] and not row['membership']['exact_duplicate']:
            pools[i] = np.vstack([pools[i], point])
            added.append(i)
    return added


def continue_job(job, output=OUTPUT):
    """Resume after saved initial AP; fresh Adam per MP, no held-out loading."""
    start = time.perf_counter()
    seed, radius = job['seed'], job['radius']
    folder = Path(output) / 'continuations' / f'seed_{seed}' / f'radius_{radius:g}'
    folder.mkdir(parents=True, exist_ok=False)
    record = metadata(seed, radius, 'continued_mean_rule_CCG')
    record.update(experiment='retailer_missing_mean_rule_completion_20260916',
        status='running', terminal_status='running', started_at_utc=core.utc_now(),
        summary_path=str(folder / 'summary.json'), iterations=[], mp_count=0, max_mp=40,
        workers=WORKERS, threads=1, fresh_short_run_wall_s=None,
        timing_scope='current resume invocation; cached nominal/initial AP excluded',
        selected_candidate_id=None, selected_checkpoint=None, selected_checkpoint_sha256=None,
        selected_state_sha256=None, selected_separation_status=None,
        data_access='frozen train40 data only; no validation/test or new nominal selection')

    def publish():
        record['current_ap_total_time_s'] = sum(a.get('ap', {}).get('wall_time_s', 0.)
                                               for a in record['iterations'][1:])
        record['current_mp_total_time_s'] = sum(a.get('mp_call', {}).get('wall_time_s', 0.)
                                               for a in record['iterations'])
        core.write_json(record['summary_path'], record)

    try:
        with core.timed(record.setdefault('input_verification_and_loading', {})):
            check(torch.get_num_threads() == torch.get_num_interop_threads() == 1, 'Thread contract')
            inputs = folder / 'inputs'
            inputs.mkdir()
            local = {}
            for name, info in job['inputs'].items():
                local[name] = inputs / (name + ('.pth' if name == 'checkpoint' else '.json'))
                ccg._copy(FROZEN / info['copy'], local[name], info['sha256'])
            original = json.loads(local['summary'].read_text())
            check(original['seed'] == seed and original['radius'] == radius
                  and original['mp_count'] == 0 and len(original['iterations']) == 1, 'Resume identity')
            data = json.loads(local['training_data'].read_text())
            check(set(data) == {'centers', 'original_sample_ids', 'data_sha256'}, 'Train-only data contract')
            centers = np.asarray(data['centers'], dtype=np.float64)
            ids = data['original_sample_ids']
            check(centers.shape == (40, 6) and len(set(ids)) == 40, 'Training dimensions')
            saved_pool = json.loads(local['pool'].read_text())
            pools = [np.asarray(p, dtype=np.float64) for p in saved_pool['pools']]
            a = deepcopy(original['iterations'][0])
            check(ccg.pool_hash(pools) == a['pool_sha256'], 'Saved pool identity')
            check(all(np.array_equal(p, c[None, :]) for p, c in zip(pools, centers)), 'Initial pool changed')
            model = ccg._verify_checkpoint(local['checkpoint'], a['checkpoint_sha256'], a['state_sha256'])
            initial_folder = folder / 'candidate_00'
            initial_folder.mkdir()
            ccg._copy(local['checkpoint'], initial_folder / 'candidate.pth', a['checkpoint_sha256'])
            a['checkpoint'] = str(initial_folder / 'candidate.pth')
            a['pool_file_sha256'] = ccg._save_pool(initial_folder / 'pool.json', pools, ids)
            record.update(lr_net=original['lr_net'], nominal_state_sha256=original['nominal_state_sha256'],
                          training_data_sha256=original['training_data_sha256'],
                          cached_nominal=original['cached_nominal'])
            record['cached_prefix'] = dict(source_summary=job['inputs']['summary'],
                original_input_phase=original['input_verification_and_loading'],
                initial_evaluation=a['evaluation'], initial_ap=a['ap'],
                reused=True, timing_scope='persisted previous invocation, not recomputed')
            a.update(cached_assessment=True, source_ap_reference=a['ap_reference'])
            a['ap_reference'] = str(initial_folder / 'assessment.json')
            record['iterations'].append(a)
            # Validate archived points/gates again without rerunning their solves.
            points = []
            for i, row in enumerate(a['samples']):
                incumbent = row['incumbent']
                result = (np.asarray(incumbent['worst1']), np.asarray(incumbent['worst2']), incumbent['cost'])
                points.append(ccg.validate_ap(result, deepcopy(row), i, centers[i], pools[i], radius,
                                             a['pool_costs'][i]))
            check(not mean_stop(a)[0], 'Initial AP cannot stop mean-rule run')
            core.write_json(a['ap_reference'], a)
        seen = set()
        while True:
            a = record['iterations'][-1]
            passed, gap = mean_stop(a)
            a['mean_stop_gap'] = gap
            if passed:
                record['terminal_status'] = 'mean_gap_satisfied'
                break
            if a['candidate_id'] == 40:
                record['terminal_status'] = 'iteration_cap'
                break
            if a['state_sha256'] in seen:
                record['terminal_status'] = 'repeated_state_without_mean_stop'
                break
            seen.add(a['state_sha256'])
            a['added_sample_ids'] = add_points(pools, points, a['samples'])
            if not a['added_sample_ids']:
                record['terminal_status'] = 'no_new_violating_incumbents'
                break
            mp_folder = Path(a['checkpoint']).parent / 'mp'
            mp_folder.mkdir()
            training = a['mp'] = metadata(seed, radius, 'robust_neural_MP')
            training.update(mp_iteration=a['candidate_id']+1, incoming_state_sha256=a['state_sha256'],
                pool_sha256=ccg.pool_hash(pools),
                pool_file_sha256=ccg._save_pool(mp_folder / 'pool.json', pools, ids))
            incoming = a['incoming_pool_evaluation'] = {}
            ccg._pool_mean(model, pools, incoming)
            objective = ccg.restricted_objective(pools)
            publish()
            with core.timed(a.setdefault('mp_call', {})):
                model = train_model(model, lambda m: objective(m).mean(), record['lr_net'],
                                    mp_folder, training, epochs=3000, patience=300)
            check(training['status'] == 'completed', 'Incomplete MP')
            record['mp_count'] += 1
            check(abs(training['first_train']-incoming['mean']) <= 1e-6
                  and training['selected_training_loss'] <= incoming['mean']+1e-6, 'MP incoming alignment')
            returned = a['returned_pool_evaluation'] = {}
            ccg._pool_mean(model, pools, returned)
            check(abs(returned['mean']-training['selected_training_loss']) <= 1e-6, 'MP returned alignment')
            check(ccg.pool_hash(pools) == training['pool_sha256']
                  and core.digest(mp_folder / 'pool.json') == training['pool_file_sha256'], 'Pool changed')
            checkpoint = mp_folder / 'selected.pth'
            digest, learned = core.digest(checkpoint), ccg.state_hash(model)
            ccg._verify_checkpoint(checkpoint, digest, learned)
            checkpoint.chmod(0o444)
            core.write_json(Path(a['checkpoint']).parent / 'assessment.json', a)
            iteration = a['candidate_id']+1
            candidate_folder = folder / f'candidate_{iteration:02d}'
            candidate_folder.mkdir()
            ccg._copy(checkpoint, candidate_folder / 'candidate.pth', digest)
            candidate = dict(candidate_id=iteration, seed=seed, radius=radius,
                checkpoint=str(candidate_folder / 'candidate.pth'), checkpoint_sha256=digest,
                state_sha256=learned, pool_sha256=ccg.pool_hash(pools),
                pool_file_sha256=ccg._save_pool(candidate_folder / 'pool.json', pools, ids),
                ap_reference=str(candidate_folder / 'assessment.json'), cached_assessment=False)
            record['iterations'].append(candidate)
            points = ccg.assess_candidate(model, pools, centers, ids, radius, candidate, candidate_folder, publish)
            check(abs(candidate['restricted_mean']-training['selected_training_loss']) <= 1e-6,
                  'Next candidate differs from selected MP objective')
        if record['terminal_status'] != 'mean_gap_satisfied':
            record['status'] = 'incomplete'
        else:
            with core.timed(record.setdefault('selection_and_checkpoint', {})):
                # Original loading is used only for the comparable saved+new phase sum.
                recovered = u.recover_prefix(dict(record,
                    input_verification_and_loading=original['input_verification_and_loading']))
                check(recovered['recoverable'] and recovered['stop_candidate'] == record['mp_count'],
                      'Recovery and online stopping disagree')
                selected = record['iterations'][recovered['selected_candidate']]
                path = folder / 'selected.pth'
                ccg._copy(selected['checkpoint'], path, selected['checkpoint_sha256'])
                ccg._verify_checkpoint(path, selected['checkpoint_sha256'], selected['state_sha256'])
                record.update(selected_candidate_id=selected['candidate_id'], selected_checkpoint=str(path),
                    selected_checkpoint_sha256=selected['checkpoint_sha256'], selected_state_sha256=selected['state_sha256'],
                    selected_separation_status=selected['separation_status'],
                    measured_prefix_phase_sum_s=recovered['measured_prefix_phase_sum_s'],
                    prefix_ap_s=recovered['prefix_ap_s'], prefix_mp_s=recovered['prefix_mp_s'],
                    prefix_phases=recovered['phases'], stop_mean_gap=recovered['stop_mean_gap'])
            record['status'] = 'completed'
    except BaseException:
        record.update(status='failed', terminal_status='failure', error=traceback.format_exc())
    finally:
        record.update(ended_at_utc=core.utc_now(), wall_time_s=time.perf_counter()-start)
        record['wall_time_human'] = u.c.human(record['wall_time_s'])
        publish()
    return record


def evaluation_worker():
    u.c.FROZEN = OUTPUT / 'evaluation_frozen'
    u.c.initialize_worker()


def run():
    check(not (OUTPUT / 'summary.json').exists(), 'No overwrite or silent resume')
    start = time.perf_counter()
    summary = dict(status='running', started_at_utc=core.utc_now(), phases={},
        command=shlex.join(sys.orig_argv), cwd=str(Path.cwd()), workers=WORKERS,
        environment=dict(host=platform.node(), platform=platform.platform(),
                         cpu='Intel Xeon Gold 6354; shared server'),
        timing_clock='time.perf_counter', timing_unit='seconds',
        continuations=dict(jobs=[]), evaluation=dict(jobs=[]))
    def publish():
        core.write_json(OUTPUT / 'summary.json', summary)
    try:
        with u.c.phase(summary['phases'], 'verify_inputs'):
            m = verify()
            summary['manifest_sha256'] = core.digest(FROZEN / 'manifest.json')
        context = multiprocessing.get_context('spawn')
        phase = summary['continuations']
        with core.timed(phase):
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=context, initializer=configure_worker) as pool:
                futures = {pool.submit(continue_job, j): j for j in m['jobs']}
                for future in as_completed(futures):
                    try:
                        item = future.result()
                    except BaseException:
                        item = dict(seed=futures[future]['seed'], radius=futures[future]['radius'],
                                    status='worker_failed', wall_time_s=None, error=traceback.format_exc())
                    phase['jobs'].append(item)
                    print(f'Continuation {len(phase["jobs"])}/31 seed={item["seed"]} rho={item["radius"]}: {item["status"]}', flush=True)
                    publish()
        check(len(phase['jobs']) == 31 and all(j['status'] == 'completed' for j in phase['jobs']),
              'Missing/failed/nonconverged continuation; no full graph or test evaluation')
        verify()
        # All31 policies are frozen before opening test arrays or prior OOS costs.
        with u.c.phase(summary['phases'], 'freeze_selected_policies'):
            ef = OUTPUT / 'evaluation_frozen'
            ef.mkdir()
            evaluation_jobs = []
            for j in sorted(phase['jobs'], key=lambda j: (j['radius'], j['seed'])):
                copy = f's{j["seed"]}_r{j["radius"]:g}.pth'
                ccg._copy(j['selected_checkpoint'], ef / copy, j['selected_checkpoint_sha256'])
                evaluation_jobs.append(dict(id=f'mean_gap_r{j["radius"]:g}_s{j["seed"]}', method='new',
                    comparison_method='mean_gap', seed=j['seed'], radius=j['radius'], n_train=40, n_val=10,
                    network_seed=0, split_seed=100000+j['seed'], training_backend='Moreau',
                    checkpoint=j['selected_checkpoint'], checkpoint_copy=copy,
                    checkpoint_sha256=j['selected_checkpoint_sha256'], state_sha256=j['selected_state_sha256'],
                    selected_iteration=j['selected_candidate_id'], selected_separation_status=j['selected_separation_status'],
                    terminal_status=j['terminal_status']))
            fixed = dict(jobs=evaluation_jobs, fixed_at_utc=core.utc_now(), source_manifest_sha256=summary['manifest_sha256'])
            core.write_json(ef / 'manifest.json', fixed)
            (ef / 'manifest.json').chmod(0o444)
        evaluation = summary['evaluation']
        prior = json.loads((u.OUTPUT / 'frozen/manifest.json').read_text())
        evaluation.update(manifest_sha256=core.digest(ef / 'manifest.json'), test_arrays=prior['test_arrays'])
        with u.c.phase(summary['phases'], 'copy_test_arrays'):
            for info in prior['test_arrays'].values():
                ccg._copy(u.OUTPUT / info['file'], OUTPUT / info['file'], info['file_sha256'])
        with core.timed(evaluation):
            with ProcessPoolExecutor(max_workers=8, mp_context=context, initializer=evaluation_worker) as pool:
                futures = {pool.submit(u.c.evaluate_job, j, evaluation['manifest_sha256'], evaluation['test_arrays'], OUTPUT): j
                           for j in evaluation_jobs}
                for future in as_completed(futures):
                    try:
                        item = future.result()
                    except BaseException:
                        item = dict(**futures[future], status='worker_failed', wall_time_s=None, error=traceback.format_exc())
                    evaluation['jobs'].append(item)
                    print(f'Evaluation {len(evaluation["jobs"])}/31 {item["id"]}: {item["status"]}', flush=True)
                    publish()
        u.c.complete_grid(evaluation, fixed)
        verify()
        summary['status'] = 'completed'
        import report_short_completion as reporting
        summary['reporting'] = reporting.build_report(OUTPUT, summary)
        verify()
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        summary.update(ended_at_utc=core.utc_now(), wall_time_s=time.perf_counter()-start)
        summary['wall_time_human'] = u.c.human(summary['wall_time_s'])
        publish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run'))
    args = parser.parse_args()
    if args.command == 'freeze':
        configure_worker()
        freeze()
    else:
        run()