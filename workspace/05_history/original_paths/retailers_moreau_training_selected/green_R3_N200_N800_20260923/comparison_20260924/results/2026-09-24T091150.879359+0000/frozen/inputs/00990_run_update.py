"""Isolated retrospective rule update; three missing evaluations, no optimizers."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'

import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN = ROOT / 'results/20260916T080636.766867Z'
OLD = Path('/data/home/aviyac/retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10')
OUT = HERE / 'results'
sys.path.insert(0, str(ROOT))
from common import core, torch, np, legacy_ccg, evaluate, configure_worker
sys.path.insert(0, str(OLD / 'matched_oos_comparison'))
import compare as c
from reconstruct import reconstruct, identity, choose_radii, require
from report import build_report


def compact(row):
    return {k: row[k] for k in ('id', 'color', 'seed', 'radius', 'checkpoint_sha256',
        'state_sha256', 'selected_iteration', 'stop_candidate', 'terminal_status',
        'validation_mean')}


def check_test(row, arrays):
    require(row['status'] == 'completed' and row['n_test'] == 10000 and
            row['test_seed'] == 9999 and row['solves_accepted'] == row['solves_attempted'] == 10000
            and row['status_counts'] == {'2': 10000} and row['test_arrays'] == arrays
            and np.isfinite(row['mean_oos']) and row['solver_threads'] == 1,
            'Cached/current evaluation contract failed')


def cache_insert(cache, row, metric):
    old = cache.get(identity(row))
    require(old is None or old[metric] == row[metric],
            f'Duplicate policy has conflicting cached {metric}')
    cache.setdefault(identity(row), row)


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=False)
    frozen = OUT / 'frozen'
    frozen.mkdir()
    summary = dict(experiment='initial_eligible_mean_stop_matched_min_ap_20260923',
        status='running', started_at_utc=c.utc(), command=sys.orig_argv, cwd=str(Path.cwd()),
        timing_clock='time.perf_counter', timing_unit='seconds', workers=1, threads_per_worker=1,
        environment=dict(host=platform.node(), platform=platform.platform(),
            processor=platform.processor(), cpu_count=os.cpu_count(), python=sys.version),
        phases={}, choices=[], fresh_validation=[], fresh_test=[], input_sha256={},
        new_training=False, new_ap=False, new_mp=False, new_cv=False,
        test_caveat='previously inspected seed9999; descriptive, not confirmatory')

    def capture(path, expected=None):
        path = Path(path).resolve()
        digest = c.sha(path)
        if expected is not None:
            require(digest == expected, f'Historical hash mismatch: {path}')
        prior = summary['input_sha256'].get(str(path))
        require(prior is None or prior == digest, f'Input changed: {path}')
        summary['input_sha256'][str(path)] = digest
        return digest

    def read(path, expected=None):
        before = capture(path, expected)
        result = c.read_json(path)
        require(c.sha(path) == before, f'Changed during read: {path}')
        return result

    def save():
        c.write_json(OUT / 'summary.json', summary)

    try:
        configure_worker()
        c.WORKERS = 1
        c.FROZEN = frozen
        with c.phase(summary['phases'], 'reconstruct_and_freeze'):
            for name in ('run_update.py', 'reconstruct.py', 'report.py', 'test_update.py',
                         'PROTOCOL.md', 'launch.sh', 'preflight.py', 'preflight.json'):
                capture(HERE / name)
            preflight = read(HERE / 'preflight.json')
            require(preflight['status'] == 'completed', 'Preflight not passed')
            for path, digest in preflight['source_sha256'].items():
                capture(path, digest)
            m = read(ROOT / 'matched_oos_update/frozen/manifest.json',
                     (ROOT / 'matched_oos_update/frozen/manifest.sha256').read_text().strip())
            versions = dict(python=sys.version, numpy=np.__version__, scipy=c.scipy.__version__,
                            torch=torch.__version__, gurobi=list(c.gp.gurobi.version()))
            require(versions == m['versions'], 'Evaluator environment differs from archived runs')
            summary['versions'] = versions
            for path, info in m['files'].items():
                capture(path, info['sha256'])
            train = read(TRAIN / 'summary.json', m['files'][str(TRAIN / 'summary.json')]['sha256'])
            require(train['status'] == 'completed' and train['inputs_unchanged'], 'Training incomplete')
            provenance = read(TRAIN / 'provenance.json')
            summary['historical_environment'] = provenance['environment']
            for path, digest in provenance['source_sha256'].items():
                capture(path, digest)
            summary['source_campaign'] = dict(nominal_lr_search_s=train['nominal']['wall_time_s'],
                ccg_sweep_s=train['ccg']['wall_time_s'], validation_s=train['validation']['wall_time_s'],
                overall_s=train['wall_time_s'])
            for job in sorted(train['ccg']['jobs'], key=lambda j: (j['seed'], j['radius'])):
                record = read(job['summary_path'], job['summary_sha256'])
                for color in ('green', 'black'):
                    row = reconstruct(record, color)
                    row['id'] = f'{color}_r{row["radius"]:g}_s{row["seed"]}'
                    row['source_summary_sha256'] = job['summary_sha256']
                    capture(row['checkpoint'], row['checkpoint_sha256'])
                    model = c.load_model(row['checkpoint'], 'new')
                    require(c.state_hash(model) == row['state_sha256'], 'Learned state mismatch')
                    row['checkpoint_copy'] = row['checkpoint_sha256'] + '.pth'
                    target = frozen / row['checkpoint_copy']
                    if not target.exists():
                        shutil.copyfile(row['checkpoint'], target)
                        target.chmod(0o444)
                    require(c.sha(target) == row['checkpoint_sha256'], 'Snapshot mismatch')
                    summary['choices'].append(row)
            require(len(summary['choices']) == 180 and
                {(r['color'], r['seed'], r['radius']) for r in summary['choices']} ==
                {(c0, s, r) for c0 in ('green', 'black') for s in range(10) for r in c.RADII},
                'Incomplete selection grid')
            # No held-out score was consulted in constructing the selected states.
            c.write_json(OUT / 'training_only_choices.json', dict(frozen_at_utc=c.utc(),
                choices=summary['choices'], validation_used_for_candidate_selection=False,
                test_used_for_candidate_selection=False))
            summary['training_choices_sha256'] = c.sha(OUT / 'training_only_choices.json')
        print('All180 minimum-AP checkpoint identities frozen from training records.', flush=True)
        save()

        with c.phase(summary['phases'], 'validation_and_radius_selection'):
            val_protocol = read(ROOT / 'supervisor_comparison/evidence_complete/protocol.json')
            for path, digest in val_protocol['split_hashes'].items():
                capture(path, digest)
            old_validation = read(ROOT / 'supervisor_comparison/evidence_complete/green_validation_progress.json')
            old_jobs = {identity(j): j for j in val_protocol['jobs']}
            validation_cache = {}
            for row in train['validation']['evaluations'] + old_validation:
                require(row['status'] == 'completed' and row['sample_count'] == 10
                        and np.isfinite(row['mean']), 'Invalid cached validation')
                if row in old_validation:
                    require(identity(row) in old_jobs, 'Unfrozen cached validation checkpoint')
                cache_insert(validation_cache, row, 'mean')
            missing = [r for r in summary['choices'] if identity(r) not in validation_cache]
            require({(r['color'], r['seed'], r['radius'], r['selected_iteration']) for r in missing}
                    == {('black', 1, .7, 11), ('black', 7, .7, 12), ('black', 9, .7, 39)},
                    'Missing validation inventory differs from approved three checkpoints')
            for choice in missing:
                split_path = TRAIN / f'nominal/seed_{choice["seed"]}/split.json'
                split = read(split_path, val_protocol['split_hashes'][str(split_path)])['arrays']
                require(len(split['train_indices']) == 40 and len(split['val_indices']) == 10 and
                        not set(split['train_indices']) & set(split['val_indices']), 'Invalid split')
                v = {k: choice[k] for k in ('id', 'seed', 'radius', 'checkpoint_sha256', 'state_sha256')}
                v.update(method='native_Moreau_validation', sample_count=10, workers=1, threads=1)
                summary['fresh_validation'].append(v)
                with core.timed(v.setdefault('load_and_evaluation', {})):
                    model = core.model_from_state(torch.load(frozen / choice['checkpoint_copy'],
                                                             map_location='cpu', weights_only=True))
                    require(legacy_ccg.state_hash(model) == choice['state_sha256'], 'Validation state mismatch')
                    x1 = torch.tensor(split['val_x1'], dtype=torch.float32)
                    x2 = torch.tensor(split['val_x2'], dtype=torch.float32)
                    evaluate(model, x1, x2, v)
                require(v['projection']['sample_count'] == 10 and v['projection']['statuses'] == {1: 10},
                        'Native validation projection failure')
                validation_cache[identity(choice)] = v
                save()
            current_val_ids = {identity(v) for v in summary['fresh_validation']}
            for row in summary['choices']:
                v = validation_cache[identity(row)]
                row.update(validation_mean=v['mean'], validation_wall_s=v['wall_time_s'],
                    validation_origin='current' if identity(row) in current_val_ids else 'cached')
            with c.phase(summary['phases'], 'radius_selection'):
                summary['selected_radii'] = {color: [compact(r) for r in rows]
                    for color, rows in choose_radii(summary['choices']).items()}
                c.write_json(OUT / 'radius_choices_before_test.json', dict(frozen_at_utc=c.utc(),
                    selection='minimum val10 per seed; ties smaller radius; nominal eligible',
                    choices=summary['selected_radii'], test_scores_read_in_this_invocation=False))
                summary['radius_choices_sha256'] = c.sha(OUT / 'radius_choices_before_test.json')
        print('Validation complete; all20 radius choices frozen before test lookup.', flush=True)
        save()

        with c.phase(summary['phases'], 'test_lookup_and_evaluation'):
            prior = read(ROOT / 'matched_oos_update/summary.json')
            require(prior['status'] == 'completed', 'Prior evaluation incomplete')
            require(prior['manifest_sha256'] == c.sha(ROOT / 'matched_oos_update/frozen/manifest.json'),
                    'Prior evaluation manifest mismatch')
            arrays = prior['test_arrays']
            require(arrays == m['test_arrays'], 'Test array contract changed')
            evaluation_cache = {}
            completion_manifest = read(ROOT / 'short_rule_completion/evaluation_frozen/manifest.json')
            completion_sha = c.sha(ROOT / 'short_rule_completion/evaluation_frozen/manifest.json')
            historical_jobs = m['jobs'] + completion_manifest['jobs']
            frozen_jobs = {j['id']: j for j in historical_jobs}
            rows = prior['jobs'] + [read(ROOT / 'short_rule_completion' / j['id'] / 'result.json')
                                  for j in completion_manifest['jobs']]
            for row in rows:
                check_test(row, arrays)
                require(identity(row) == identity(frozen_jobs[row['id']]), 'Cached identity mismatch')
                origin_sha = (prior['manifest_sha256'] if row in prior['jobs'] else completion_sha)
                require(row['manifest_sha256'] == origin_sha, 'Cached evaluation origin mismatch')
                capture(row['checkpoint'], row['checkpoint_sha256'])
                cache_insert(evaluation_cache, row, 'mean_oos')
            pending = [r for r in summary['choices'] if identity(r) not in evaluation_cache]
            require({(r['seed'], r['radius'], r['selected_iteration']) for r in pending}
                    == {(1, .7, 11), (7, .7, 12), (9, .7, 39)}, 'Unexpected missing test inventory')
            summary['test_arrays'] = arrays
            for key, info in arrays.items():
                source = OLD / 'matched_oos_comparison/results' / info['file']
                capture(source, info['file_sha256'])
                shutil.copyfile(source, OUT / info['file'])
                require(c.sha(OUT / info['file']) == info['file_sha256'], 'Copied test data mismatch')
                (OUT / info['file']).chmod(0o444)
            with c.phase(summary['phases'], 'fresh_test_evaluation'):
                for choice in pending:
                    keys = ('id', 'seed', 'radius', 'method', 'n_train', 'n_val', 'network_seed',
                            'split_seed', 'training_backend', 'checkpoint', 'checkpoint_copy',
                            'checkpoint_sha256', 'state_sha256', 'selected_iteration',
                            'selected_separation_status', 'terminal_status')
                    job = {k: choice[k] for k in keys}
                    job['comparison_method'] = 'black_min_ap'
                    result = c.evaluate_job(job, summary['training_choices_sha256'], arrays, output=OUT)
                    summary['fresh_test'].append(result)
                    save()
                    check_test(result, arrays)
                    evaluation_cache[identity(choice)] = result
                    print(f'{job["id"]}: {result["status"]}, {result["solves_accepted"]} OPTIMAL QPs, '
                          f'{result["wall_time_s"]:.3f}s', flush=True)
            current_test_ids = {identity(r) for r in summary['fresh_test']}
            for row in summary['choices']:
                test = evaluation_cache[identity(row)]
                row.update(test_mean=test['mean_oos'], test_wall_s=test['wall_time_s'],
                    test_origin='current' if identity(row) in current_test_ids else 'cached',
                    test_source_id=test['id'], status='completed')
            by_id = {r['id']: r for r in summary['choices']}
            for selected in summary['selected_radii'].values():
                for row in selected:
                    row['test_mean'] = by_id[row['id']]['test_mean']

        with c.phase(summary['phases'], 'end_verification'):
            old_plot = read(ROOT / 'short_rule_completion/plot_data.json')
            require(old_plot['test_arrays'] == arrays, 'Reference figure test identity mismatch')
            for path, digest in summary['input_sha256'].items():
                require(c.sha(path) == digest, f'Protected input changed: {path}')
            for row in summary['choices']:
                require(c.sha(frozen / row['checkpoint_copy']) == row['checkpoint_sha256'], 'Snapshot changed')
            require(c.sha(OUT / 'training_only_choices.json') == summary['training_choices_sha256']
                    and c.sha(OUT / 'radius_choices_before_test.json') == summary['radius_choices_sha256'],
                    'Frozen decisions changed')
            summary['inputs_unchanged'] = True
        summary['status'] = 'completed'
        with c.phase(summary['phases'], 'reporting'):
            summary['report'] = build_report(OUT, summary, old_plot)
        print('Completed: no training/MP/AP; 3 validation and 3 test evaluations.', flush=True)
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        summary.update(ended_at_utc=c.utc(), wall_time_s=time.perf_counter()-start)
        summary['wall_time_human'] = c.human(summary['wall_time_s'])
        save()


if __name__ == '__main__':
    main()