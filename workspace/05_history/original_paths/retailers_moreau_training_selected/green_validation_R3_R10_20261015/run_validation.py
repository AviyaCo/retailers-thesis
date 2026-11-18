"""Validation-selected radius for the 540 existing GREEN STOP policies (R3/R10, no shipping).

No training. Each saved STOP checkpoint (stopping_checkpoint_20260928 selection
manifest) is replayed on its own held-out rows (Ntotal/5) with the existing strict
common Gurobi projection evaluator (src.evaluate.evaluate_policy, reached through
the replay module run.py). Per (R, N, seed) the radius with minimum validation mean
is chosen (exact tie -> smaller radius); all 60 choices are SHA256-sealed before any
test cost is opened. Test costs are then looked up from the existing seed-9999,
10,000-row STOP replay and re-verified against its raw arrays/journals.
"""
import importlib
import importlib.util
from pathlib import Path

STOP_DIR = Path('/data/home/aviyac/retailers_experiment/comparison/stopping_checkpoint_20260928')
_spec = importlib.util.spec_from_file_location('_stop_run', STOP_DIR / 'run.py')
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)  # sets single-thread BLAS env before any numerics are imported

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
import platform
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
EXECUTION = STOP_DIR / 'production_v1/execution_manifest.json'
EXECUTION_SHA = 'bb586892ff4bcc811dec8255310470641a3b4fdefee768c99c1642532c22a78a'
ENVELOPES = STOP_DIR / 'production_v1/worker_envelopes.json'
PLOT_DATA = {r: STOP_DIR / f'production_v1/published/plot_data_R{r}.json' for r in (3, 10)}
RADII = [0., .001, .003, .01, .03, .2, .4, .7, 1.5]
VALIDATION_TOL = 1e-6  # physical residual gate requested for this run (evaluator itself gates at 1e-4)
require, sha, utc, phase = run.require, run.sha, run.utc, run.phase


def write(path, value):
    """Overwritable JSON (progress summaries); sealed artifacts use run.seal."""
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, allow_nan=False, indent=1) + '\n')
    os.replace(tmp, path)


def centers_hash(a):
    return hashlib.sha256(run.numerical().np.asarray(a, dtype='<f8').tobytes()).hexdigest()


def validation_rows(jobs, bindings):
    """Rebuild the canonical split, verify training rows against saved artifacts, return holdout."""
    np = run.numerical().np
    config = importlib.import_module('src.config')  # importable after run.numerical() adds the repo root
    first = jobs[0]
    s = config.split_data(config.Config(first['R'], first['Ntotal'], first['seed']))
    ti, vi = s['train_indices'], s['val_indices']
    train = np.concatenate((s['xi1'][ti], s['xi2'][ti]), axis=1)
    for j in jobs:
        td = run.bound_json(j['training_data_path'], j['training_data_sha256'])
        bindings[j['training_data_path']] = j['training_data_sha256']
        require(s['data_sha256'] == j['data_sha256'] == td['data_sha256']
                and [int(i) for i in ti] == td['original_sample_ids'] == j['original_sample_ids']
                and np.array_equal(train, np.asarray(td['centers'], dtype=np.float64))
                and centers_hash(train) == j['training_centers_sha256']
                and j['Ntrain'] == len(ti) == 4 * j['Ntotal'] // 5,
                f"Reconstructed training rows differ from saved policy data: {j['id']}")
    split_path = Path(first['training_data_path']).parent / 'split.json'
    split_checked = split_path.exists()
    if split_checked:  # R3 N200/N800 and R10 campaigns saved the full split; R3 N50 did not
        a = run.read(split_path)['arrays']
        bindings[str(split_path)] = sha(split_path)
        require(np.array_equal(np.asarray(a['xi1']), s['xi1']) and np.array_equal(np.asarray(a['xi2']), s['xi2'])
                and a['train_indices'] == ti.tolist() and a['val_indices'] == vi.tolist()
                and np.array_equal(np.asarray(a['val_x1'], dtype=np.float32), s['val_x1'].numpy())
                and np.array_equal(np.asarray(a['val_x2'], dtype=np.float32), s['val_x2'].numpy()),
                f'Saved split.json differs from reconstruction: {split_path}')
    v1, v2 = s['xi1'][vi], s['xi2'][vi]
    require(len(vi) == first['Ntotal'] // 5 and not set(ti.tolist()) & set(vi.tolist())
            and sorted([*ti.tolist(), *vi.tolist()]) == list(range(first['Ntotal'])), 'Bad partition')
    require(not {tuple(r) for r in train} & {tuple(r) for r in np.concatenate((v1, v2), axis=1)},
            'Exact train/validation scenario overlap')
    return v1, v2, dict(split_seed=100000 + first['seed'], validation_ids=vi.tolist(),
                        data_sha256=s['data_sha256'], split_json_checked=split_checked,
                        validation_previously_used_for_lr=True)


def verify_validation(folder, model, x1, x2):
    """Re-read every raw array/journal row; recompute targets, costs and residuals (gate 1e-6)."""
    e = run.numerical()
    np = e.np
    raw = run.read(folder / 'result.json')
    n = len(x1)
    require(raw['status'] == 'completed' and raw['role'] == 'validation' and raw['sample_count'] == n
            and raw['solves_attempted'] == raw['solves_accepted'] == n and raw['status_counts'] == {'2': n}
            and raw['state_sha256'] == e.state_hash(model), 'Incomplete validation replay')
    z, q10, q1r = e.targets(model, x1)
    targets, q, costs = (np.load(folder / f, allow_pickle=False) for f in ('targets.npy', 'shipments.npy', 'costs.npy'))
    require(np.array_equal(targets, z) and raw['q10'] == q10 and q.dtype == costs.dtype == np.float64
            and np.array_equal(costs, e.scenario_costs(q10, q1r, q, x1, x2))
            and raw['mean'] == float(costs.mean()), 'Validation arrays/costs differ')
    residual = np.maximum(np.maximum(0., -q.min(axis=1)), q.sum(axis=1) - q10)
    rows = [json.loads(line) for line in (folder / 'solves.jsonl').read_text().splitlines()]
    require(len(rows) == n and all(d['row'] == i and d['status'] == 2 and d['accepted'] is True
            and np.array_equal(q[i], d['shipments']) and d['physical_residual'] == residual[i]
            for i, d in enumerate(rows)), 'Validation journal inconsistent')
    require(float(residual.max()) <= VALIDATION_TOL, f'Physical residual {residual.max()} > {VALIDATION_TOL}')
    return dict(max_physical_residual=float(residual.max()), iterations=raw['iterations'],
                barrier_iterations=raw['barrier_iterations'], solves=n,
                result_sha256=sha(folder / 'result.json'), costs_sha256=sha(folder / 'costs.npy'))


def init_worker():
    run.numerical()


def evaluate_group(group, output, workers):
    """One (R, N, seed) grid: replay all nine radii on the same validation rows."""
    e = run.numerical()
    start = time.perf_counter()
    folder = Path(output) / group['id']
    folder.mkdir()
    require(sha(group['data_path']) == group['data_sha256'], 'Validation input changed')
    with e.np.load(group['data_path'], allow_pickle=False) as d:
        x1, x2 = d['x1'], d['x2']
    rec = {k: v for k, v in group.items() if k != 'arms'}
    rec.update(start_utc=utc(), arms=[], pid=os.getpid(), workers=workers, solver_threads=1,
               torch_threads=e.torch.get_num_threads())
    for spec in group['arms']:
        t0, arm_start = time.perf_counter(), utc()
        arm = dict(spec, folder=str(folder / spec['id']))
        try:
            require(sha(spec['checkpoint']) == spec['checkpoint_sha256'], 'Changed checkpoint')
            model = e.load_model(spec['checkpoint'], spec['R'])
            require(e.state_hash(model) == spec['state_sha256'], 'Changed learned state')
            raw = e.evaluate_policy(model, x1, x2, folder / spec['id'], dict(role='validation', worker_count=workers))
            arm.update(verify_validation(folder / spec['id'], model, x1, x2), mean_validation=raw['mean'],
                       replay_wall_time_s=raw['wall_time_s'], status='completed')
        except BaseException:
            arm.update(status='failed', error=traceback.format_exc())
        arm.update(start_utc=arm_start, end_utc=utc(), wall_time_s=time.perf_counter() - t0,
                   timing_scope='checkpoint load + strict replay + raw re-verification')
        rec['arms'].append(arm)
    rec['status'] = 'completed' if all(a['status'] == 'completed' for a in rec['arms']) else 'failed'
    rec.update(end_utc=utc(), wall_time_s=time.perf_counter() - start)
    run.seal(folder / 'grid.json', rec)
    return rec


def prepare(output, manifest, bindings):
    green = [j for j in manifest['jobs'] if j['method'] == 'green']
    require(len(green) == 540 and all(j['selected_candidate_id'] == j['stopping_candidate_id']
            and j['terminal_status'] == 'green_gap_satisfied' for j in green), 'Wrong GREEN STOP cohort')
    groups = []
    for r in (3, 10):
        for n in (800, 200, 50):  # longest grids first
            for seed in range(10):
                jobs = sorted((j for j in green if (j['R'], j['Ntotal'], j['seed']) == (r, n, seed)),
                              key=lambda j: j['radius'])
                require([j['radius'] for j in jobs] == RADII, 'Incomplete radius grid')
                v1, v2, protocol = validation_rows(jobs, bindings)
                gid = f'R{r}_N{n}_s{seed}'
                path = output / 'inputs' / f'{gid}.npz'
                run.numerical().np.savez(path, x1=v1, x2=v2)
                arms = []
                for j in jobs:
                    require(sha(j['checkpoint']) == j['checkpoint_sha256'], 'Changed checkpoint')
                    bindings[j['checkpoint']] = j['checkpoint_sha256']
                    arms.append({k: j[k] for k in ('id', 'R', 'Ntotal', 'Ntrain', 'seed', 'radius', 'checkpoint',
                                 'checkpoint_sha256', 'state_sha256', 'stopping_candidate_id', 'mp_count')})
                groups.append(dict(id=gid, R=r, Ntotal=n, seed=seed, validation_size=len(v1),
                                   protocol=protocol, data_path=str(path), data_sha256=sha(path), arms=arms))
    groups.sort(key=lambda g: -g['Ntotal'])  # N800 grids (largest validation sets) start first
    return groups


def choose(arms):
    require(sorted(a['radius'] for a in arms) == RADII and all(a['status'] == 'completed'
            and math.isfinite(a['mean_validation']) for a in arms), 'Incomplete/failed grid; no selection')
    return min(arms, key=lambda a: (a['mean_validation'], a['radius']))


def seal_selection(output, grids):
    choices = []
    for g in sorted(grids, key=lambda g: (g['R'], g['Ntotal'], g['seed'])):
        require(run.sealed(output / 'validation' / g['id'] / 'grid.json') == g, 'Grid record changed')
        for a in g['arms']:
            require(sha(Path(a['folder']) / 'result.json') == a['result_sha256']
                    and sha(Path(a['folder']) / 'costs.npy') == a['costs_sha256'], 'Raw validation changed')
        w = choose(g['arms'])
        choices.append(dict(R=g['R'], Ntotal=g['Ntotal'], seed=g['seed'], selected_radius=w['radius'],
                            selected_id=w['id'], validation_mean=w['mean_validation'],
                            state_sha256=w['state_sha256'], checkpoint_sha256=w['checkpoint_sha256'],
                            validation_means={str(a['radius']): a['mean_validation'] for a in g['arms']},
                            rule='minimum validation mean; exact tie -> smaller radius'))
    require(len(choices) == 60, 'Need 60 choices')
    return run.seal(output / 'selections.json', dict(status='sealed', sealed_at_utc=utc(),
        test_costs_accessed=False, inputs_sha256=sha(output / 'inputs.json'), choices=choices))


def attach_tests(output, digest, manifest):
    """First code path that opens test costs; only after the selection seal exists."""
    e = run.numerical()
    np = e.np
    sealed = run.sealed(output / 'selections.json', digest)
    require(sealed['test_costs_accessed'] is False, 'Seal invalid')
    envelopes = {r['id']: r for r in run.sealed(ENVELOPES)}
    plots = {r: run.sealed(PLOT_DATA[r]) for r in (3, 10)}
    tests = {r: run.test_arrays(manifest, r) for r in (3, 10)}
    jobs = {j['id']: j for j in manifest['jobs']}
    rows = []
    for ch in sealed['choices']:
        t0 = time.perf_counter()
        job, env = jobs[ch['selected_id']], envelopes[ch['selected_id']]
        result = run.sealed(env['result_path'], env['result_sha256'])
        require(result['status'] == 'completed' and result['state_sha256'] == ch['state_sha256']
                == job['state_sha256'] and result['checkpoint_sha256'] == ch['checkpoint_sha256'], 'Identity mismatch')
        model = e.load_model(job['checkpoint'], job['R'])
        x1, x2 = tests[job['R']]
        folder = Path(env['result_path']).parent / 'oos'
        oos, _ = run.verify_replay(folder, model, x1, x2)
        q = np.load(folder / 'shipments.npy', allow_pickle=False)
        residual = float(np.maximum(np.maximum(0., -q.min(axis=1)), q.sum(axis=1) - oos['q10']).max())
        panel = next(p for p in plots[job['R']]['panels'] if p['Ntotal'] == job['Ntotal'])['green']
        require(panel['radius'] == RADII and panel['seeds'] == list(range(10)), 'Plot layout changed')
        plotted = panel['oos']['per_seed'][RADII.index(ch['selected_radius'])][ch['seed']]
        require(oos['mean_oos'] == result['mean_oos'] == plotted, 'Test cost mismatch raw/result/plot')
        h = job['historical_training']
        rows.append(dict(ch, test_mean=result['mean_oos'], test_result_path=env['result_path'],
            test_max_physical_residual=residual, test_evaluation_cached=True,
            historical_test_oos_wall_time_s=result['phases']['oos_evaluation']['wall_time_s'],
            historical_ccg_through_stop_active_time_s=h['green_through_stop']['active_phase_time_s'],
            historical_total_wall_time_s=h.get('total_wall_time_s'),
            lookup_and_raw_verification_wall_time_s=time.perf_counter() - t0))
    require(run.sealed(output / 'selections.json', digest) == sealed, 'Selection changed during lookup')
    return rows


def stats(values):
    np = run.numerical().np
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a) > 1 else None,
                median=float(np.median(a)), min=float(a.min()), max=float(a.max()))


def report(output, rows, grids, summary):
    by_rn, radius_times = [], []
    lines = ['# GREEN STOP (no shipping): validation-selected radius, R3/R10', '',
             "Saved STOP policies only; no retraining. Validation = each seed's own held-out Ntotal/5 rows "
             '(split seed 100000+s), which were already used for nominal LR selection (not untouched). '
             'Strict common Gurobi projection (OPTIMAL, residual <= 1e-6). Minimum validation mean, '
             'exact tie -> smaller radius. 60 choices sealed (SHA256) before test lookup. Test = existing '
             'seed-9999 10,000-row replay (historically inspected; descriptive, not a fresh holdout).', '',
             '| R | Ntotal | Seeds | Mean selected test cost | Sample SD | Selected-radius counts |',
             '|---:|---:|---:|---:|---:|---|']
    for r in (3, 10):
        for n in (50, 200, 800):
            sel = sorted((x for x in rows if (x['R'], x['Ntotal']) == (r, n)), key=lambda x: x['seed'])
            require([x['seed'] for x in sel] == list(range(10)), 'Incomplete aggregate')
            counts = dict(sorted(Counter(x['selected_radius'] for x in sel).items()))
            s = stats([x['test_mean'] for x in sel])
            gw = stats([g['wall_time_s'] for g in grids if (g['R'], g['Ntotal']) == (r, n)])
            by_rn.append(dict(R=r, Ntotal=n, test=s, radius_counts={str(k): v for k, v in counts.items()},
                              selected_radii=[x['selected_radius'] for x in sel], grid_wall_time_s=gw))
            lines.append(f"| {r} | {n} | 10 | {s['mean']:.6f} | {s['sd']:.6f} | {counts} |")
            for rho in RADII:
                radius_times.append(dict(R=r, Ntotal=n, radius=rho, **stats([a['wall_time_s'] for g in grids
                    if (g['R'], g['Ntotal']) == (r, n) for a in g['arms'] if a['radius'] == rho])))
    lines += ['', '## Per-seed choices', '', '| R | N | Seed | Radius | Validation mean | Test cost |',
              '|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {x['R']} | {x['Ntotal']} | {x['seed']} | {x['selected_radius']:g} | {x['validation_mean']:.6f} | "
              f"{x['test_mean']:.6f} |" for x in sorted(rows, key=lambda x: (x['R'], x['Ntotal'], x['seed']))]
    sweep = summary['phases']['parallel_validation']['wall_time_s']
    lines += ['', '## Runtime (seconds; time.perf_counter)', '',
              '| Component | Status |', '|---|---|',
              '| Nominal neural training | Not rerun; persisted historical timings in the stopping selection manifest. |',
              '| Robust neural / CCG AP+MP | Not rerun; persisted CCG-through-STOP active time per selected policy in results.json. |',
              '| SRO-FA | Not run. |',
              f"| Validation replay (current) | {len(grids)*9} arms, {summary['workers']} workers x 1 thread; "
              f"parallel sweep elapsed {sweep:.1f} s ({sweep/60:.2f} min). |",
              '| Test evaluation | Cached historical replay; current lookup + raw re-verification timed separately. |', '',
              '| R | N | Grids | Grid mean s | Median | Min | Max |', '|---:|---:|---:|---:|---:|---:|---:|']
    lines += [f"| {g['R']} | {g['Ntotal']} | {g['grid_wall_time_s']['count']} | {g['grid_wall_time_s']['mean']:.3f} | "
              f"{g['grid_wall_time_s']['median']:.3f} | {g['grid_wall_time_s']['min']:.3f} | "
              f"{g['grid_wall_time_s']['max']:.3f} |" for g in by_rn]
    lines += ['', 'Sweep phases (nested; concurrent worker times are not summed):', '']
    lines += [f"- {k}: {v['wall_time_s']:.3f} s ({v['wall_time_s']/60:.3f} min)"
              for k, v in summary['phases'].items() if 'wall_time_s' in v]
    lines += ['', f"Environment: {summary['environment']['host']}, {summary['environment']['versions']}.",
              'Per-arm times (load + replay + verification), per-radius aggregates and raw artifacts: '
              'results.json, validation/. Overall elapsed: summary.json.', '']
    write(output / 'results.json', dict(unit='seconds', by_R_N=by_rn, per_seed=rows, per_radius_arm_time=radius_times,
        per_arm=[{k: a[k] for k in ('id', 'R', 'Ntotal', 'seed', 'radius', 'status', 'mean_validation',
                  'wall_time_s', 'replay_wall_time_s', 'max_physical_residual', 'iterations', 'barrier_iterations',
                  'start_utc', 'end_utc')} for g in grids for a in g['arms']],
        selection_sha256=sha(output / 'selections.json'), phases=summary['phases'],
        workers=summary['workers'], threads=1, environment=summary['environment']))
    (output / 'RESULTS.md').write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--workers', required=True, type=int)
    p.add_argument('--session', required=True, help='tmux session name (recorded only)')
    p.add_argument('--log', required=True, help='persistent log path (recorded only)')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    summary = dict(experiment=HERE.name, status='running', start_utc=utc(), phases={}, workers=args.workers,
                   threads=1, training_run=False, invocation=dict(argv=[sys.executable, *sys.argv], cwd=os.getcwd(),
                   tmux_session=args.session, inside_tmux='TMUX' in os.environ, log=args.log), grids=[],
                   environment=dict(host=platform.node(), affinity_cpus=len(os.sched_getaffinity(0)),
                                    loadavg_at_start=os.getloadavg()))
    write(output / 'summary.json', summary)
    try:
        require(len(os.sched_getaffinity(0)) >= args.workers, 'Insufficient CPU affinity')
        for name in ('inputs', 'validation'):
            (output / name).mkdir()
        bindings = {}
        with phase(summary['phases'], 'input_freeze_and_split_verification'):
            execution = run.sealed(EXECUTION, EXECUTION_SHA)
            manifest = run.selection_bundle(execution['selection_path'], execution['selection_sha256'],
                execution['selection_verification_path'], execution['selection_verification_sha256'])
            summary['environment']['versions'] = run.runtime_environment(manifest)['versions']
            groups = prepare(output, manifest, bindings)
            bindings[str(Path(__file__).resolve())] = sha(__file__)
            run.seal(output / 'inputs.json', dict(selection_sha256=execution['selection_sha256'], groups=groups,
                     bindings=bindings, test_costs_accessed=False, created_at_utc=utc()))
        print(f"Frozen {sum(len(g['arms']) for g in groups)} arms / {len(groups)} grids; training rows verified.",
              flush=True)
        with phase(summary['phases'], 'parallel_validation'):
            with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=init_worker) as pool:
                futures = [pool.submit(evaluate_group, g, str(output / 'validation'), args.workers) for g in groups]
                for f in as_completed(futures):
                    g = f.result()
                    summary['grids'].append(g)
                    write(output / 'summary.json', summary)
                    print(f"{len(summary['grids'])}/60 {g['id']}: {g['status']} {g['wall_time_s']:.2f}s", flush=True)
        failed = [a for g in summary['grids'] for a in g['arms'] if a['status'] != 'completed']
        summary['failed_arms'] = [dict(id=a['id'], error=a['error']) for a in failed]
        require(not failed, f'{len(failed)} validation arms failed; no selection')
        with phase(summary['phases'], 'selection_seal'):
            digest = seal_selection(output, summary['grids'])
        print('60 choices sealed; opening cached test costs.', flush=True)
        with phase(summary['phases'], 'cached_test_lookup_and_raw_verification'):
            rows = attach_tests(output, digest, manifest)
        with phase(summary['phases'], 'final_integrity'):
            for path, h in bindings.items():
                require(sha(path) == h, f'Changed input: {path}')
        with phase(summary['phases'], 'reporting'):
            report(output, rows, summary['grids'], summary)
        summary['status'] = 'completed'
        print('COMPLETED', flush=True)
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        summary.update(end_utc=utc(), wall_time_s=time.perf_counter() - start)
        write(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
