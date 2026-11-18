"""Fixed-policy shipping validation. No training, test-driven selection or fallback."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PARENT = BASE / 'shipping_comparison_20260928'
STOP = BASE / 'shipping_stopping_20261005/production_v2'
MANIFEST_SHA = 'a0fc73c005be77c512a33ae7b6b842ba8c5ea8745a92dccaab1845754d26c619'
STOP_SUMMARY_SHA = '5ac86a50693e90e84ee14f1eac8ff3a8f7b0f4eeb48591fefc655a5a5fc81cd7'
PARENT_SUMMARY_SHA = 'd6d92fc3c852bcc06b7481cd2f59e5ed6e511f3e4ef91f7381078d96a5c9e313'
WORKERS = 8
sys.path.insert(0, str(PARENT))
c = importlib.import_module('common')
replay = importlib.import_module('evaluation')
NEURAL_FEE = importlib.import_module('plot_comparison').NEURAL_FEE
c.check(Path(c.__file__).resolve() == PARENT / 'common.py'
    and Path(replay.__file__).resolve() == PARENT / 'evaluation.py', 'Wrong historical imports')


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(c.np.ascontiguousarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def _rule():
    """Shared radius-selection rule (retailers_experiment/src/config.py), 2026-11-18."""
    shared = '/data/home/aviyac/retailers_experiment'
    if shared not in sys.path:
        sys.path.insert(0, shared)
    return importlib.import_module('src.config')


def choose(rows, radii):
    c.check(len(rows) == len(radii) and sorted(r['radius'] for r in rows) == sorted(radii),
            'Incomplete or duplicate validation grid')
    c.check(all(r['status'] == 'completed' and math.isfinite(r['mean_validation']) for r in rows),
            'Failed/nonfinite validation arm; no partial selection')
    chosen = _rule().select_radius({r['radius']: r['mean_validation'] for r in rows})
    return next(r for r in rows if r['radius'] == chosen)


def checked_read(path, expected):
    c.check(c.sha(path) == expected, f'Changed input: {path}')
    return c.read(path)


def validation_data(method, n, seed, historical, bindings):
    """Only saved train/validation provenance; no test artifact is accessed."""
    x1, x2 = c.c.gen(n, seed)
    if method == 'green':
        folder = BASE / f'shipping_nominal_R3_N50_N200_N800_20260924/production_v1/N{n}/initialization/seed{seed}'
        objects = []
        for name in ('split.json', 'training_data.json'):
            path = folder / name
            expected = historical[str(path)]
            objects.append(checked_read(path, expected))
            bindings[str(path)] = expected
        split, training = objects
        ti, vi = c.np.asarray(split['train_ids']), c.np.asarray(split['holdout_ids'])
        c.check(len(ti) == 4*n//5 and len(vi) == n//5
                and len(set(ti) & set(vi)) == 0 and sorted([*ti, *vi]) == list(range(n)),
                'Train/validation indices are not a disjoint exhaustive partition')
        for data in objects:
            c.check((data['Ntotal'], data['seed'], data['split_seed']) == (n, seed, 100000+seed),
                    'Wrong saved split identity')
            c.check(data['paired_data_sha256'] == array_hash(x1, x2), 'Changed paired data')
        c.check(array_hash(x1[ti], x2[ti]) == split['train_array_sha256']
                == training['training_array_sha256'], 'Changed training arrays')
        c.check(c.np.array_equal(ti, training['original_sample_ids'])
                and c.np.array_equal(x1[ti], training['x1'])
                and c.np.array_equal(x2[ti], training['x2']), 'Training identity mismatch')
        c.check(array_hash(x1[vi], x2[vi]) == split['holdout_array_sha256'], 'Changed holdout arrays')
        tx1, tx2 = x1[ti], x2[ti]
        v1, v2 = x1[vi], x2[vi]
        protocol = dict(split_seed=100000+seed, validation_seed=None,
                        validation_ids=vi.tolist(), training_ids=ti.tolist(),
                        validation_previously_used_for_lr=True)
    else:
        tx1, tx2 = x1, x2
        v1, v2 = c.c.gen(math.ceil(n/4), 88880+seed)
        protocol = dict(split_seed=None, validation_seed=88880+seed,
                        validation_ids=None, training_ids=list(range(n)),
                        validation_previously_used_for_lr=False)
    train_rows = {tuple(a) for a in c.np.concatenate((tx1, tx2), axis=1)}
    val_rows = {tuple(a) for a in c.np.concatenate((v1, v2), axis=1)}
    c.check(not train_rows.intersection(val_rows), 'Exact train/validation scenario overlap')
    c.check(c.np.isfinite(v1).all() and c.np.isfinite(v2).all(), 'Nonfinite validation data')
    return v1, v2, protocol, tx1, tx2


def verify_raw(record, folder, x1, x2):
    phase = record['phases']['validation']
    c.check(phase['status'] == 'completed' and phase['fee_rule'] == NEURAL_FEE, 'Wrong evaluation protocol')
    c.check(c.sha(phase['raw_path']) == phase['raw_sha256'], 'Changed validation raw archive')
    with c.np.load(phase['raw_path'], allow_pickle=False) as archive:
        a = {k: archive[k] for k in archive.files}
    c.check(c.np.array_equal(a['x1'], x1) and c.np.array_equal(a['x2'], x2), 'Wrong validation rows')
    c.check(all(c.np.isfinite(v).all() for v in a.values()), 'Nonfinite validation evidence')
    q = a['shipments']
    c.check(q.shape == (len(x1), 3) and q.dtype == c.np.float64, 'Wrong shipment shape/type')
    residual = c.np.maximum(c.np.maximum(0., -q.min(axis=1)), q.sum(axis=1)-a['q10'].item())
    c.check((residual <= c.PHYSICAL_TOL).all(), 'Validation physical violation')
    costs = c.scenario_costs(a['q10'].item(), a['q1r'], q, x1, x2)
    c.check(c.np.array_equal(costs, a['costs']) and float(costs.mean()) == record['mean_validation'],
            'Validation cost arithmetic mismatch')
    journal = Path(folder) / 'validation_solves.jsonl'
    count = 0
    with journal.open() as stream:
        for i, line in enumerate(stream):
            d = json.loads(line)
            c.check(i < len(x1) and d['row'] == i and d['accepted'] and d['status'] == 2,
                    'Invalid validation solver journal')
            c.check(c.np.array_equal(d['q2'], q[i]) and d['positive_support'] == (q[i] > 0).tolist()
                    and d['physical_residual'] == residual[i], 'Journal/raw inconsistency')
            count += 1
    c.check(count == record['solves_attempted'] == record['solves_accepted'] == len(x1)
            and record['status_counts'] == {'2': count}, 'Incomplete validation solves')
    c.check(phase['solves_attempted'] == phase['solves_accepted'] == count, 'Incomplete validation phase')


def evaluate_arm(spec, x1, x2, folder):
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    record = dict(spec, status='running', start_utc=c.utc(), phases={},
                  status_counts={}, solves_attempted=0, solves_accepted=0,
                  workers=WORKERS, torch_threads=1, solver_threads=1, pid=os.getpid())
    try:
        with c.phase(record['phases'], 'model_loading'):
            c.check(c.sha(spec['checkpoint_copy']) == spec['checkpoint_sha256'], 'Changed checkpoint')
            model = c.load_model(spec['checkpoint_copy'], spec['method'])
            c.check(c.c.state_hash(model) == spec['state_sha256'], 'Changed learned state')
        replay.evaluate(model, x1, x2, 'validation', record, folder)
        record['mean_validation'] = record.pop('mean_oos')
        c.check(c.c.state_hash(model) == spec['state_sha256'], 'Evaluation changed model')
        record['status'] = 'completed'
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record.update(end_utc=c.utc(), wall_time_s=time.perf_counter()-start,
                      timing_scope='Current arm: loading, target evaluation, QPs and cost/raw persistence; excludes final result write')
        c.write(folder / 'result.json', record)
    return record


def worker(group, output):
    start = time.perf_counter()
    folder = Path(output) / group['id']
    folder.mkdir(exist_ok=False)
    record = {k: v for k, v in group.items() if k != 'arms'}
    record.update(start_utc=c.utc(), status='running', arms=[], workers=WORKERS,
                  torch_threads=c.torch.get_num_threads(), solver_threads=1, pid=os.getpid())
    try:
        c.check(c.sha(group['data_path']) == group['data_sha256'], 'Changed validation data')
        with c.np.load(group['data_path'], allow_pickle=False) as data:
            x1, x2 = data['x1'], data['x2']
        for spec in group['arms']:
            arm_folder = folder / spec['id']
            arm = evaluate_arm(spec, x1, x2, arm_folder)
            record['arms'].append(dict(id=spec['id'], radius=spec['radius'], status=arm['status'],
                mean_validation=arm['mean_validation'], wall_time_s=arm['wall_time_s'],
                result_path=str(arm_folder / 'result.json'), result_sha256=c.sha(arm_folder / 'result.json')))
        choose(record['arms'], group['radii'])
        record['status'] = 'completed'
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
    finally:
        record.update(end_utc=c.utc(), wall_time_s=time.perf_counter()-start,
                      timing_scope='Current full per-seed radius grid, excludes prior training and cached test evaluation')
        c.write(folder / 'result.json', record)
    return dict(id=group['id'], status=record['status'], result_path=str(folder / 'result.json'),
                result_sha256=c.sha(folder / 'result.json'), wall_time_s=record['wall_time_s'])


def prepare(output, m):
    bindings = {}
    for source in m['files']:
        if Path(source).suffix == '.py':
            c.bound_file(m, source)
            bindings[source] = m['files'][source]['sha256']
    for source in (Path(__file__), ROOT / 'test_validation.py', ROOT / 'launch.py', ROOT / 'README.md'):
        target = output / 'sources' / source.name
        shutil.copyfile(source, target)
        bindings[str(source)] = c.sha(source)
        c.check(c.sha(target) == bindings[str(source)], 'Source snapshot mismatch')
    groups, probes = [], []
    specs = [j for j in m['jobs'] if j['method'] == 'green' or j['radius'] > 0]
    c.check(len(specs) == 720 and len({j['id'] for j in specs}) == 720, 'Wrong candidate cohort')
    for method in ('green', 'legacy'):
        for n in (50, 200, 800):
            radii = m['green_radii'] if method == 'green' else [r for r in m['legacy_radii'][str(n)] if r > 0]
            for seed in range(10):
                arms = []
                jobs = sorted((j for j in specs if (j['method'], j['Ntotal'], j['seed']) == (method, n, seed)),
                              key=lambda j: j['radius'])
                c.check([j['radius'] for j in jobs] == radii, 'Wrong per-seed candidate grid')
                v1, v2, protocol, t1, t2 = validation_data(method, n, seed, m['historical_bindings'], bindings)
                for j in jobs:
                    if method == 'green':
                        c.check(j['green_pass'] and j['selected_candidate_id'] == j['stopping_candidate_id'],
                                'Green policy is not its passing STOP checkpoint')
                        c.check(c.np.array_equal(c.np.asarray(j['centers']), c.np.concatenate((t1, t2), axis=1)),
                                'Green model training rows differ from saved split')
                    for source in j['input_files']:
                        c.bound_file(m, source)
                        bindings[source] = m['files'][source]['sha256']
                    spec = {k: j[k] for k in ('id', 'method', 'Ntotal', 'Ntrain', 'seed', 'radius',
                        'checkpoint', 'checkpoint_sha256', 'state_sha256', 'selected_candidate_id',
                        'stopping_candidate_id', 'training_runtime_reference')}
                    spec['checkpoint_copy'] = str(c.bound_file(m, j['checkpoint']))
                    bindings[spec['checkpoint_copy']] = j['checkpoint_sha256']
                    spec['validation_size'] = len(v1)
                    arms.append(spec)
                    if seed == 0 and j['radius'] == .2:
                        probes.append((spec, t1[:2], t2[:2]))
                group_id = f'{method}_N{n}_s{seed}'
                data_path = output / 'inputs' / f'{group_id}.npz'
                c.np.savez(data_path, x1=v1, x2=v2)
                bindings[str(data_path)] = c.sha(data_path)
                groups.append(dict(id=group_id, method=method, Ntotal=n, seed=seed, radii=radii,
                    validation_size=len(v1), protocol=protocol, data_path=str(data_path),
                    data_sha256=c.sha(data_path), array_sha256=array_hash(v1, v2), arms=arms))
    return groups, probes, bindings


def seal_selection(output, groups, completed):
    choices = []
    by_id = {g['id']: g for g in groups}
    c.check(len(completed) == 60 and {r['id'] for r in completed} == set(by_id), 'Incomplete group results')
    for compact in sorted(completed, key=lambda r: r['id']):
        c.check(compact['status'] == 'completed', 'Failed group; no selection or test lookup')
        result = checked_read(compact['result_path'], compact['result_sha256'])
        group = by_id[compact['id']]
        with c.np.load(group['data_path'], allow_pickle=False) as data:
            for a in result['arms']:
                record = checked_read(a['result_path'], a['result_sha256'])
                spec = next(s for s in group['arms'] if s['id'] == a['id'])
                c.check(all(record[k] == v for k, v in spec.items()), 'Arm/model identity mismatch')
                c.check(record['mean_validation'] == a['mean_validation'], 'Changed selection metric')
                verify_raw(record, Path(a['result_path']).parent, data['x1'], data['x2'])
        winner = choose(result['arms'], group['radii'])
        spec = next(a for a in group['arms'] if a['id'] == winner['id'])
        choices.append(dict(method=group['method'], Ntotal=group['Ntotal'], seed=group['seed'],
            selected_radius=winner['radius'], selected_id=winner['id'],
            validation_mean=winner['mean_validation'], state_sha256=spec['state_sha256'],
            checkpoint_sha256=spec['checkpoint_sha256'], validation_result=compact,
            selection_rule=_rule().RADIUS_SELECTION_RULE,
            validation_wall_time_s=result['wall_time_s']))
    path = output / 'selections.json'
    c.write(path, dict(status='sealed', sealed_at_utc=c.utc(), test_costs_accessed=False,
                      input_manifest_sha256=c.sha(output / 'inputs.json'), choices=choices))
    path.with_suffix('.sha256').write_text(c.sha(path) + '\n')
    path.chmod(0o444)
    return choices


def require_seal(output):
    path = output / 'selections.json'
    s = checked_read(path, path.with_suffix('.sha256').read_text().strip())
    c.check(s['status'] == 'sealed' and s['test_costs_accessed'] is False and len(s['choices']) == 60,
            'No complete pre-test selection seal')
    c.check(s['input_manifest_sha256'] == c.sha(output / 'inputs.json'), 'Changed validation input manifest')
    return s


def attach_tests(output, stop_manifest):
    """First code path allowed to open held-out costs, after all choices are sealed."""
    sealed = require_seal(output)
    summaries = {
        'green': checked_read(STOP / 'summary.json', STOP_SUMMARY_SHA),
        'legacy': checked_read(PARENT / 'production_v1/summary.json', PARENT_SUMMARY_SHA)}
    legacy_manifest = checked_read(PARENT / 'production_v1/manifest.json', stop_manifest['parent_manifest_sha256'])
    manifests = dict(green={**stop_manifest, 'seal': MANIFEST_SHA},
                     legacy={**legacy_manifest, 'seal': stop_manifest['parent_manifest_sha256']})
    attached = []
    for choice in sealed['choices']:
        start = time.perf_counter()
        row = dict(choice, lookup_start_utc=c.utc())
        method = choice['method']
        compact = next(r for r in summaries[method]['results'] if r['id'] == choice['selected_id'])
        manifest = manifests[method]
        spec = next(j for j in manifest['jobs'] if j['id'] == choice['selected_id'])
        c.check(spec['state_sha256'] == choice['state_sha256']
                and spec['checkpoint_sha256'] == choice['checkpoint_sha256'], 'Selected/cached model mismatch')
        result = replay.verify_result(spec, compact, manifest)
        c.check(result['fee_rule'] == NEURAL_FEE and manifest['test_seed'] == 9999
                and manifest['n_test'] == 10000, 'Wrong cached test protocol')
        row.update(test_mean=result['mean_oos'], test_result_path=compact['result_path'],
                   test_result_sha256=compact['result_sha256'], test_evaluation_cached=True,
                   historical_evaluation_wall_time_s=result['wall_time_s'],
                   historical_oos_wall_time_s=result['phases']['oos']['wall_time_s'],
                   historical_is_wall_time_s=result['phases']['is']['wall_time_s'],
                   historical_training_runtime_reference=spec['training_runtime_reference'],
                   lookup_end_utc=c.utc(), lookup_and_raw_verification_wall_time_s=time.perf_counter()-start)
        attached.append(row)
    c.check(require_seal(output) == sealed, 'Test attachment altered validation choices')
    c.write(output / 'selected_test_results.json', dict(selection_sha256=c.sha(output / 'selections.json'),
            fee_rule=NEURAL_FEE, test_seed=9999, test_samples=10000, rows=attached))
    return attached


def report(output, results, groups, completed, summary):
    score_groups, runtime_groups, arm_groups = [], [], []
    text = ['# Shipping: validation-selected radius, raw positive shipping fees', '',
        'Both colors use the unchanged strict common Gurobi projection and charge 0.1 per raw q>0 shipment. '
        'All 60 choices were sealed before test-cost lookup. No retraining, clipping, fallback or test-driven selection.', '',
        'Green: saved STOP policies, existing 10/40/160-row holdout (already used for LR selection), '
        'nine radii including zero. Blue: saved historical policies, independent 13/50/200-row validation '
        'sets with seed 88880+s, original 15 positive radii. Seeds 0–9; largest radius within 1e-6 of the minimum validation mean; no refit.', '',
        'The inspected historical 10,000-row seed-9999 test and test-informed historical grids are descriptive, '
        'not fresh confirmatory evidence. Green retains the mixed 265-original/5-advisory cohort. '
        'Blue retains historical non-converged/capped policies where present; no post-hoc filtering. '
        'Training sizes and validation protocols differ, so this is not a matched-protocol solver comparison.', '',
        '## Selected test costs (lower is better)', '',
        '| Method | Ntotal | Seeds | Mean | Sample SD | Selected radius counts |',
        '|---|---:|---:|---:|---:|---|']
    for method in ('green', 'legacy'):
        for n in (50, 200, 800):
            rows = sorted((r for r in results if r['method'] == method and r['Ntotal'] == n), key=lambda r:r['seed'])
            c.check([r['seed'] for r in rows] == list(range(10)), 'Incomplete score aggregate')
            values = [r['test_mean'] for r in rows]
            counts = dict(sorted(Counter(r['selected_radius'] for r in rows).items()))
            stats = dict(method=method, Ntotal=n, **c.stats(values), sd=float(c.np.std(values, ddof=1)),
                         selected_radii=[r['selected_radius'] for r in rows], radius_counts=counts)
            score_groups.append(stats)
            text.append(f"| {method} | {n} | 10 | {stats['mean']:.6f} | {stats['sd']:.6f} | {counts} |")
            runtime_groups.append(dict(method=method, Ntotal=n,
                validation_grid=c.stats([r['validation_wall_time_s'] for r in rows]),
                cached_oos_evaluation=c.stats([r['historical_oos_wall_time_s'] for r in rows]),
                current_lookup_and_raw_verification=c.stats([r['lookup_and_raw_verification_wall_time_s'] for r in rows])))
    for group in groups:
        compact = next(r for r in completed if r['id'] == group['id'])
        record = checked_read(compact['result_path'], compact['result_sha256'])
        for arm in record['arms']:
            arm_groups.append(dict(method=group['method'], Ntotal=group['Ntotal'], seed=group['seed'],
                                   radius=arm['radius'], wall_time_s=arm['wall_time_s']))
    radius_times = []
    for key in sorted({(a['method'], a['Ntotal'], a['radius']) for a in arm_groups}):
        values = [a['wall_time_s'] for a in arm_groups if (a['method'],a['Ntotal'],a['radius']) == key]
        c.check(len(values) == 10, 'Incomplete radius timing aggregate')
        radius_times.append(dict(method=key[0], Ntotal=key[1], radius=key[2], **c.stats(values)))
    text += ['', '## Per-seed choices', '', '| Method | N | Seed | Radius | Validation cost | Test cost |',
             '|---|---:|---:|---:|---:|---:|']
    for r in sorted(results, key=lambda r:(r['method'],r['Ntotal'],r['seed'])):
        text.append(f"| {r['method']} | {r['Ntotal']} | {r['seed']} | {r['selected_radius']:g} | "
                    f"{r['validation_mean']:.6f} | {r['test_mean']:.6f} |")
    text += ['', '## Runtime coverage', '', '| Component | Status / scope |', '|---|---|',
        '| Nominal neural training | Not rerun. Persisted selected-model timings retained; missing historical isolated fit time remains unavailable. |',
        '| Robust neural / CCG AP and MP | Not rerun. Historical per-iteration timing references retained in selected_test_results.json. |',
        '| SRO-FA partition / optimization / evaluation / total | Not run; no new comparable timing. |',
        '| Validation / selection | Newly measured, perf_counter seconds, full grid per seed and each candidate below. No cross-validation or refit. |',
        '| Final test | Cached strict-q>0 replay, with original OOS time retained. Current lookup/raw verification is NOT test evaluation time. |', '',
        f"Environment: {summary['environment']}. Eight validation workers; one Torch/Gurobi thread each. "
        f'Concurrent job times are not summed as sweep elapsed. Full child time, command, log and cwd: ../{output.name}.invocation.json.', '',
        '| Method | N | Runs | Grid mean s (min) | Median s | Min s | Max s | Historical selected OOS mean s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runtime_groups:
        s = r['validation_grid']
        text.append(f"| {r['method']} | {r['Ntotal']} | {s['count']} | {s['mean']:.3f} ({s['mean']/60:.3f}) | "
                    f"{s['median']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {r['cached_oos_evaluation']['mean']:.3f} |")
    text += ['', '| Method | N | Radius | Runs | Arm mean s | Median s | Min s | Max s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in radius_times:
        text.append(f"| {r['method']} | {r['Ntotal']} | {r['radius']:g} | {r['count']} | "
                    f"{r['mean']:.3f} | {r['median']:.3f} | {r['min']:.3f} | {r['max']:.3f} |")
    text += ['', '## Current sweep phases (nested; not summed worker walls)', '']
    for k, v in summary['phases'].items():
        if 'wall_time_s' in v:
            text.append(f"- {k}: {v['wall_time_s']:.3f} s ({v['wall_time_s']/60:.3f} min), {v['status']}.")
    text += ['', 'Raw per-arm arrays, solve journals, status/iteration counts and timing are under validation/. '
             'Input hashes and split identities: inputs.json. Pre-test choices: selections.json and its SHA256 seal. '
             'Selected test identities, raw verification time and prior training/evaluation times: selected_test_results.json.', '']
    c.write(output / 'runtime_report.json', dict(unit='seconds', per_seed_groups=runtime_groups,
        per_radius_groups=radius_times, sweep_phases=summary['phases'], historical_training_scope='persisted, not rerun',
        environment=summary['environment'], workers=WORKERS, threads=1))
    c.write(output / 'score_summary.json', dict(groups=score_groups, test_seed=9999, n_test=10000,
        selection_sha256=c.sha(output / 'selections.json'), fee_rule=NEURAL_FEE))
    (output / 'RESULTS.md').write_text('\n'.join(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    c.check(output.is_dir() and not any(output.iterdir()), 'Require fresh empty output directory')
    started = time.perf_counter()
    summary = dict(experiment=ROOT.name, status='running', start_utc=c.utc(), phases={}, results=[],
                   workers=WORKERS, threads=1, training_run=False, radius_selection_run=True,
                   fee_rule=NEURAL_FEE, environment=dict(host=platform.node(), platform=platform.platform(),
                   affinity_cpus=len(os.sched_getaffinity(0)), versions=c.versions()))
    c.write(output / 'summary.json', summary)
    try:
        c.check(len(os.sched_getaffinity(0)) >= WORKERS, 'Insufficient CPU affinity for requested workers')
        for name in ('sources', 'inputs', 'probes', 'validation'):
            (output / name).mkdir()
        with c.phase(summary['phases'], 'regression_tests'):
            with (output / 'tests.log').open('x') as log:
                subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', 'test_validation'],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with c.phase(summary['phases'], 'input_freeze_and_split_verification'):
            m = checked_read(STOP / 'manifest.json', MANIFEST_SHA)
            c.check(m['versions'] == c.versions(), 'Changed numerical environment')
            groups, probes, bindings = prepare(output, m)
            c.write(output / 'inputs.json', dict(parent_manifest_sha256=MANIFEST_SHA, groups=groups,
                bindings=bindings, workers=WORKERS, threads=1, versions=c.versions(), fee_rule=NEURAL_FEE,
                test_costs_accessed=False, created_at_utc=c.utc()))
        c.initialize_worker()
        with c.phase(summary['phases'], 'training_only_preflight'):
            for spec, x1, x2 in probes:
                folder = output / 'probes' / spec['id']
                r = evaluate_arm(spec, x1, x2, folder)
                verify_raw(r, folder, x1, x2)
            c.check(len(probes) == 6, 'Incomplete training-only preflight')
        print('Frozen 720 candidates / 60 validation grids; six training-only probes passed.', flush=True)
        with c.phase(summary['phases'], 'parallel_validation'):
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                                     initializer=c.initialize_worker) as pool:
                futures = [pool.submit(worker, g, str(output / 'validation')) for g in groups]
                for future in as_completed(futures):
                    result = future.result()
                    summary['results'].append(result)
                    c.write(output / 'summary.json', summary)
                    print(f"{len(summary['results'])}/60 {result['id']}: {result['status']} {result['wall_time_s']:.3f}s", flush=True)
        with c.phase(summary['phases'], 'raw_validation_verification_and_selection'):
            seal_selection(output, groups, summary['results'])
        print('All 60 validation choices sealed; opening cached test costs now.', flush=True)
        with c.phase(summary['phases'], 'cached_test_lookup_and_raw_verification'):
            results = attach_tests(output, m)
        with c.phase(summary['phases'], 'final_integrity'):
            for path, expected in bindings.items():
                c.check(c.sha(path) == expected, f'Changed source/input: {path}')
            c.check(c.sha(STOP / 'manifest.json') == MANIFEST_SHA, 'Changed parent manifest')
            require_seal(output)
        with c.phase(summary['phases'], 'reporting'):
            report(output, results, groups, summary['results'], summary)
        runtime = c.read(output / 'runtime_report.json')
        runtime['sweep_phases'] = summary['phases']
        c.write(output / 'runtime_report.json', runtime)
        with (output / 'RESULTS.md').open('a') as stream:
            elapsed = summary['phases']['reporting']['wall_time_s']
            stream.write(f'\nReporting: {elapsed:.3f} s ({elapsed/60:.3f} min); completed, excludes final timing snapshot writes.\n')
        summary.update(status='completed', candidate_count=720, selection_count=60,
            validation_projections=sum(g['validation_size']*len(g['arms']) for g in groups),
            artifacts={name:c.sha(output/name) for name in
            ('RESULTS.md','score_summary.json','runtime_report.json','selected_test_results.json','selections.json')})
        print('COMPLETED: 60 selections; strict q>0 test results published.', flush=True)
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        summary.update(end_utc=c.utc(), wall_time_s=time.perf_counter()-started)
        c.write(output / 'summary.json', summary)


if __name__ == '__main__':
    main()