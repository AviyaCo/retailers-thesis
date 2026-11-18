"""Frozen-policy supervisor comparison: validation-only selection, no training.

Green uses the existing native val10 protocol. All green radius choices are
written before any OOS result file is read. Blue/purple/black choices are saved
historical choices, never reselected here. Test data have already been viewed;
this is a retrospective descriptive comparison, not a sealed test experiment.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[name] = '1'

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BASE = NEW.parents[1]
OLD = BASE / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10'
RUN = NEW / 'results/20260916T080636.766867Z'
OUT = HERE / 'evidence_complete'
sys.path.insert(0, str(NEW))
from common import core, evaluate, legacy_ccg, configure_worker, torch, np


def utc():
    return datetime.now(timezone.utc).isoformat()


def stats(values):
    values = list(values)
    assert values and all(np.isfinite(x) for x in values)
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values), min=min(values), max=max(values))


def main():
    OUT.mkdir(exist_ok=False)
    start = time.perf_counter()
    result = dict(experiment='four_color_supervisor_comparison', status='running',
        started_at_utc=utc(), command=sys.orig_argv, cwd=str(Path.cwd()),
        host=platform.node(), python=sys.version, workers=1, torch_threads=1,
        timing_clock='time.perf_counter', timing_unit='seconds',
        training='not run; all policies fixed', input_sha256={}, phases={},
        new_validation=[], selections={}, runtime_rows=[])

    def load(path):
        path = Path(path)
        result['input_sha256'][str(path)] = core.digest(path)
        return json.loads(path.read_text())

    try:
        configure_worker()
        result['input_sha256'][str(Path(__file__))] = core.digest(__file__)
        for path, expected in json.loads((OLD / 'ccg_results/provenance.json').read_text())['source_sha256'].items():
            assert core.digest(path) == expected
            result['input_sha256'][path] = expected
        result['input_sha256'][str(NEW / 'common.py')] = core.digest(NEW / 'common.py')
        prior_manifest = load(NEW / 'matched_oos_update/frozen/manifest.json')
        continuation_manifest = load(NEW / 'short_rule_completion/evaluation_frozen/manifest.json')
        green = [r for r in prior_manifest['jobs'] if r['comparison_method'] == 'mean_gap']
        green += continuation_manifest['jobs']
        green.sort(key=lambda r: (r['seed'], r['radius']))
        assert len(green) == 90 and {(r['seed'], r['radius']) for r in green} == {
            (s, r) for s in range(10) for r in core.RADII}
        # Freeze source identities and split inputs before evaluating val10.
        splits = {}
        for seed in range(10):
            split = load(RUN / f'nominal/seed_{seed}/split.json')
            old_split = load(OLD / f'lr_search_results_v2/seed_{seed}/split.json')
            assert split['arrays'] == old_split['arrays']
            arr = split['arrays']
            assert len(arr['train_indices']) == 40 and len(arr['val_indices']) == 10
            assert not set(arr['train_indices']) & set(arr['val_indices'])
            splits[seed] = (torch.tensor(arr['val_x1'], dtype=torch.float32),
                            torch.tensor(arr['val_x2'], dtype=torch.float32))
        for job in green:
            assert core.digest(job['checkpoint']) == job['checkpoint_sha256']
            result['input_sha256'][job['checkpoint']] = job['checkpoint_sha256']
        core.write_json(OUT / 'protocol.json', dict(frozen_at_utc=utc(), jobs=green,
            split_hashes={p: h for p, h in result['input_sha256'].items() if p.endswith('split.json')},
            rule='per-seed minimum native Moreau val10 cost; exact ties smaller radius; all nine including nominal',
            refit=False, training=False, radius_grid=list(core.RADII), seeds=list(range(10)),
            projection='existing unbuffered native Moreau, same validation arithmetic as black/purple',
            test='previously inspected seed9999; do not read costs until choices persisted'))
        with core.timed(result['phases'].setdefault('green_validation', {})):
            for job in green:
                row = {k: job[k] for k in ('seed', 'radius', 'checkpoint', 'checkpoint_sha256',
                                          'state_sha256', 'selected_iteration', 'selected_separation_status')}
                row.update(Ntrain=40, Nval=10, network_seed=0, split_seed=100000+job['seed'], workers=1)
                result['new_validation'].append(row)
                with core.timed(row.setdefault('load_and_evaluation', {})):
                    model = core.model_from_state(torch.load(job['checkpoint'], map_location='cpu', weights_only=True))
                    assert legacy_ccg.state_hash(model) == job['state_sha256']
                    evaluate(model, *splits[job['seed']], row)
                assert row['projection']['sample_count'] == 10 and row['projection']['statuses'] == {1: 10}
                core.write_json(OUT / 'green_validation_progress.json', result['new_validation'])
        with core.timed(result['phases'].setdefault('green_radius_selection', {})):
            result['selections']['green'] = [dict(min(
                (r for r in result['new_validation'] if r['seed'] == seed),
                key=lambda r: (r['mean'], r['radius']))) for seed in range(10)]
            core.write_json(OUT / 'green_selections_before_test_lookup.json', dict(
                selected_at_utc=utc(), selections=result['selections']['green'],
                test_read_in_this_invocation=False))
        # Existing choices: no reoptimization and no new candidate selection.
        historical = load(BASE / 'retailers_problem/R3/rho_validation_results_net16_v2.json')['by_N']['50']
        blue = []
        for seed in range(10):
            row = historical['per_seed'][str(seed)]
            chosen = min(row['validation_costs'], key=lambda r: (row['validation_costs'][r], float(r)))
            assert float(chosen) == row['selected_rho']
            blue.append(dict(seed=seed, radius=row['selected_rho'], mean=row['selected_validation_cost'],
                             historical_test=row['selected_10k_oos']))
        result['selections']['blue'] = blue
        purple = load(OLD / 'ccg_results/summary.json')
        black = load(RUN / 'summary.json')
        result['selections']['purple'] = [dict(seed=r['seed'], radius=r['final_selection']['radius'],
            mean=r['final_selection']['selected_validation_mean'],
            state_sha256=r['final_selection']['selected_state_sha256'],
            selected_separation_status=r['final_selection']['selected_separation_status']) for r in purple['selections']]
        result['selections']['black'] = [dict(seed=r['seed'], radius=r['radius'], mean=r['validation_mean'],
            state_sha256=r['state_sha256'], checkpoint_sha256=r['checkpoint_sha256'],
            terminal_status=r['terminal_status']) for r in black['selections']]
        with core.timed(result['phases'].setdefault('saved_test_lookup', {})):
            old_eval = load(OLD / 'matched_oos_comparison/results/summary.json')
            new_eval = load(NEW / 'matched_oos_update/summary.json')
            complete = load(NEW / 'short_rule_completion/summary.json')
            groups = dict(blue=[r for r in old_eval['jobs'] if r['method'] == 'old'],
                purple=[r for r in old_eval['jobs'] if r['method'] == 'new'],
                black=[r for r in new_eval['jobs'] if r['comparison_method'] == 'terminal'],
                green=[r for r in new_eval['jobs'] if r['comparison_method'] == 'mean_gap'] + complete['evaluation']['jobs'])
            result['test_array_hashes'] = old_eval['test_arrays']
            result['fixed_radius_costs'] = {}
            for color, jobs in groups.items():
                assert len(jobs) == (230 if color == 'blue' else 90)
                lookup = {(r['seed'], r['radius']): r for r in jobs}
                assert len(lookup) == len(jobs)
                for job in jobs:
                    assert job['status'] == 'completed' and job['solves_accepted'] == job['solves_attempted'] == 10000
                    assert job['status_counts'] == {'2': 10000}
                    assert job['test_arrays'] == old_eval['test_arrays']
                for row in result['selections'][color]:
                    test = lookup[row['seed'], row['radius']]
                    if 'state_sha256' in row:
                        assert row['state_sha256'] == test['state_sha256']
                    if 'checkpoint_sha256' in row:
                        assert row['checkpoint_sha256'] == test['checkpoint_sha256']
                    assert core.digest(test['checkpoint']) == test['checkpoint_sha256']
                    row.update(test_mean=test['mean_oos'], test_job_id=test['id'],
                        checkpoint=test['checkpoint'], checkpoint_sha256=test['checkpoint_sha256'],
                        state_sha256=test['state_sha256'], selected_iteration=test['selected_iteration'])
                result['fixed_radius_costs'][color] = [dict(radius=r, cost=stats(
                    j['mean_oos'] for j in jobs if j['radius']==r)) for r in sorted({j['radius'] for j in jobs})]
                result['runtime_rows'].extend(dict(color=color, phase='OOS_evaluation', seed=j['seed'], radius=j['radius'],
                    status=j['status'], wall_s=j['wall_time_s'], workers=8, threads=1,
                    started_at_utc=j['start_utc'], ended_at_utc=j['end_utc']) for j in jobs)
        result['selected_cost_statistics'] = {color: dict(validation=stats(r['mean'] for r in rows),
            test=stats(r['test_mean'] for r in rows), radii=[r['radius'] for r in sorted(rows,key=lambda r:r['seed'])])
            for color, rows in result['selections'].items()}

        # Runtime evidence is descriptive, preserving status and original scope.
        purple_lr = load(OLD / 'lr_search_results_v2/summary.json')
        timing = load(NEW / 'short_rule_completion/runtime_comparison.json')
        for j in purple['job_results']:
            result['runtime_rows'].append(dict(color='purple', phase='CCG_wall', seed=j['seed'], radius=j['radius'],
                status=j['terminal_status'], wall_s=j['wall_time_s'], ap_s=j['ap_total_time_s'], mp_s=j['mp_total_time_s'],
                mp_count=j['mp_count'], ap_count=j['ap_sweeps_completed'], workers=16, threads=1,
                started_at_utc=j['started_at_utc'], ended_at_utc=j['ended_at_utc']))
        for seed in purple_lr['seed_results']:
            winner = next(r for r in seed['configurations'] if r['lr_net']==seed['winner']['lr_net'])
            t = winner['training']
            result['runtime_rows'].append(dict(color='purple', phase='nominal_winner', seed=seed['seed'], radius=0,
                status=t['status'], wall_s=t['wall_time_s'], epochs=t['epochs_evaluated'], workers=10, threads=1,
                started_at_utc=t['started_at_utc'], ended_at_utc=t['ended_at_utc']))
        for j in timing['models']:
            color = {'Moreau every-center':'black', 'Moreau mean-gap':'green'}[j['method']]
            for phase, key in [('CCG_phase_sum','phase_sum_s'), ('CCG_wall','wall_s')]:
                result['runtime_rows'].append(dict(color=color, phase=phase, seed=j['seed'], radius=j['radius'],
                    status=j['terminal_status'], wall_s=j[key], mp_count=j['mp_count'], ap_count=j['ap_count'],
                    origin=j['origin'], workers=16, threads=1,
                    started_at_utc=j.get('started_at_utc'), ended_at_utc=j.get('ended_at_utc'),
                    timestamp_scope='unavailable when absent in recovered timing; not reconstructed'))
        for color, key in [('black','nominal_moreau'), ('blue','nominal_scs')]:
            for j in timing[key]:
                result['runtime_rows'].append(dict(color=color, phase='nominal_winner' if color=='black' else 'nominal_rerun',
                    seed=j['seed'], radius=0, status=j['status'], wall_s=j['wall_s'], epochs=j['epochs'],
                    workers=j['workers'], threads=j['threads']))
        for path in sorted((BASE/'retailers_problem/R3/N50/timing_test').glob('rho_*/seed_*_timing.json')):
            j = load(path)
            assert j['converged']
            result['runtime_rows'].append(dict(color='blue', phase='dedicated_SCS_timing', seed=j['seed'], radius=j['rho'],
                status='recorded_converged', wall_s=j['total_time_s'], nominal_s=j['nominal_time_s'], ccg_s=j['ccg_time_s'],
                mp_count=sum(r['mp_time_s']>0 for r in j['ccg_trajectory']), ap_count=j['ccg_iters'],
                workers=9, threads=None, clock='time.time rounded 0.01s', raw=j))
        result['scs_selected_radius_rerun'] = timing['scs_unmatched_selected']['overall_statistics']
        result['runtime_statistics'] = []
        keys = sorted({(r['color'],r['phase'],r['radius'],r['status']) for r in result['runtime_rows']})
        for color,phase,radius,status in keys:
            group=[r for r in result['runtime_rows'] if (r['color'],r['phase'],r['radius'],r['status'])==(color,phase,radius,status)]
            entry=dict(color=color,phase=phase,radius=radius,status=status,n=len(group))
            for metric in ('wall_s','nominal_s','ccg_s','epochs','mp_count','ap_count'):
                values=[r[metric] for r in group if r.get(metric) is not None]
                if values:
                    entry[metric]=stats(values)
            result['runtime_statistics'].append(entry)
        result['campaign_elapsed_s'] = dict(purple_nominal_LR_search=purple_lr['wall_time_s'],
            purple_CCG=purple['wall_time_s'], black_nominal_LR_search=black['nominal']['wall_time_s'],
            black_CCG=black['ccg']['wall_time_s'], black_validation=black['validation']['wall_time_s'],
            black_total=black['wall_time_s'], blue_purple_OOS=old_eval['wall_time_s'],
            black_initial_green_OOS=new_eval['wall_time_s'], green_completion_total=complete['wall_time_s'],
            green_continuations=complete['continuations']['wall_time_s'], green_completion_OOS=complete['evaluation']['wall_time_s'])
        result['purple_execution'] = dict(mp_total=sum(j['mp_count'] for j in purple['job_results']),
            terminal_status_counts={s:sum(j['terminal_status']==s for j in purple['job_results']) for s in sorted({j['terminal_status'] for j in purple['job_results']})},
            selected_violating_count=sum(j['selected_separation_status']=='violating' for j in purple['job_results']))
        result['nominal_LRs'] = dict(purple=[s['winner']['lr_net'] for s in sorted(purple_lr['seed_results'],key=lambda r:r['seed'])],
            black_green=[s['winner']['lr_net'] for s in sorted(black['nominal']['jobs'],key=lambda r:r['seed'])])
        with (OUT/'selected_policies.csv').open('x') as stream:
            writer=csv.writer(stream)
            writer.writerow(['color','seed','radius','validation_cost','test_cost','selected_iteration','checkpoint','checkpoint_sha256'])
            for color in ('blue','purple','green','black'):
                for r in sorted(result['selections'][color],key=lambda r:r['seed']):
                    writer.writerow([color,r['seed'],r['radius'],r['mean'],r['test_mean'],r['selected_iteration'],r['checkpoint'],r['checkpoint_sha256']])
        for path, expected in result['input_sha256'].items():
            assert core.digest(path)==expected, path
        result['status']='completed'
    except BaseException as error:
        result.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        result.update(ended_at_utc=utc(), wall_time_s=time.perf_counter()-start)
        core.write_json(OUT/'summary.json',result)
    print(json.dumps(dict(status=result['status'], selected=result['selected_cost_statistics'],
                         phases=result['phases'], purple=result['purple_execution']),indent=2))


if __name__ == '__main__':
    main()