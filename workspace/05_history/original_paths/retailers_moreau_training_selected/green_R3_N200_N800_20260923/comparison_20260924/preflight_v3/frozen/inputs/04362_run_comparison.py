"""Exclusive, preflight-bound fixed-policy replay; never train or select on test."""
from __future__ import annotations

import os

THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS')
for _name in THREAD_ENV:
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import math
import multiprocessing
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback

import evaluation as e
import plot_comparison as plot

ROOT = Path(__file__).resolve().parent
GIB = 1024**3
EXPECTED_LOCAL = {'evaluation.py', 'test_evaluation.py', 'plot_comparison.py',
                  'test_plot.py', 'run_comparison.py', 'test_runner.py',
                  'preflight.py', 'launch.sh', 'README.md'}
ENTRYPOINT = e.HOME / 'retailers_problem/plot_combined_R3_net16v2.py'


def rss_bytes():
    """Linux lifetime high-water RSS, not a per-phase allocation estimate."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def resources():
    mem = {s.split(':')[0]: int(s.split()[1])*1024
           for s in Path('/proc/meminfo').read_text().splitlines()
           if s.startswith(('MemAvailable:', 'MemTotal:'))}
    quotas, headrooms = [], []
    groups = Path('/proc/self/cgroup').read_text()
    # Walk the process cgroup and its ancestors (v1 and v2), including mount root.
    for line in groups.splitlines():
        _, controllers, group = line.split(':', 2)
        for kind, base in (('v2', '/sys/fs/cgroup'), ('cpu', '/sys/fs/cgroup/cpu'),
                           ('memory', '/sys/fs/cgroup/memory')):
            if (kind == 'v2' and controllers) or (kind != 'v2' and kind not in controllers.split(',')):
                continue
            root = Path(base)
            location = root / group.lstrip('/')
            while location.is_relative_to(root):
                if kind == 'v2':
                    cpu, limit, usage = location/'cpu.max', location/'memory.max', location/'memory.current'
                    if cpu.exists():
                        q, p = cpu.read_text().split()
                        if q != 'max':
                            quotas.append(int(q)/int(p))
                elif kind == 'cpu':
                    q, p = location/'cpu.cfs_quota_us', location/'cpu.cfs_period_us'
                    if q.exists() and int(q.read_text()) > 0:
                        quotas.append(int(q.read_text())/int(p.read_text()))
                    limit, usage = location/'memory.limit_in_bytes', location/'memory.usage_in_bytes'
                else:
                    limit, usage = location/'memory.limit_in_bytes', location/'memory.usage_in_bytes'
                if limit.exists() and usage.exists() and limit.read_text().strip() != 'max':
                    headrooms.append(max(0, int(limit.read_text())-int(usage.read_text())))
                if location == root:
                    break
                location = location.parent
    return dict(host=platform.node(), platform=platform.platform(), executable=sys.executable,
        python=sys.version, affinity=sorted(os.sched_getaffinity(0)), logical_cpus=os.cpu_count(),
        cpu_quota_cores=min(quotas) if quotas else None, load_average=list(os.getloadavg()),
        memory_available_bytes=min([mem['MemAvailable'], *headrooms]), host_memory=mem,
        cgroup_memory_headrooms_bytes=headrooms, cgroup=groups, captured_at_utc=e.c.utc(),
        thread_environment={k: os.environ.get(k) for k in THREAD_ENV}, peak_rss_bytes=rss_bytes())


def choose_workers(snapshot, peak_rss):
    effective = len(snapshot['affinity'])
    if snapshot['cpu_quota_cores'] is not None:
        effective = min(effective, math.floor(snapshot['cpu_quota_cores']))
    background = math.ceil(max(snapshot['load_average']))
    cpu_slots = effective - background - 4
    per_worker = max(2*GIB, 2*peak_rss)
    memory_slots = (snapshot['memory_available_bytes']-16*GIB)//per_worker
    workers = min(32, cpu_slots, memory_slots)
    e.check(workers >= 1, 'Insufficient CPU/memory headroom; no launch or resource fallback')
    return dict(workers=int(workers), threads=1, start_method='spawn', cpu_reserve=4,
        background_cpu_slots=background, memory_reserve_bytes=16*GIB,
        memory_budget_bytes_per_worker=per_worker, preflight_peak_rss_bytes=peak_rss,
        note='Sampled headroom, not a guarantee against future contention or allocation growth.')


def local_sources():
    paths = {p.resolve() for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py', '.sh', '.md')}
    e.check(EXPECTED_LOCAL <= {p.name for p in paths}, 'Finalize all nine local source/doc files before preflight')
    return sorted(paths | {ENTRYPOINT.resolve(), (ROOT.parent/'README.md').resolve()} |
                  {e.c.OLD/f'R3/N{n}/sro_fa/sro_fa_sweep.py' for n in (50, 200, 800)})


def capture_sources(manifest, folder, paths):
    """Extend the existing backend snapshot without recreating its inputs directory."""
    for source in paths:
        source = Path(source).resolve()
        key, digest = str(source), e.c.sha(source)
        if key in manifest['files']:
            e.check(manifest['files'][key]['sha256'] == digest, f'Source changed: {source}')
            continue
        dest = Path(folder)/'inputs'/f'{len(manifest["files"]):05d}_{source.name}'
        with source.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        e.check(e.c.sha(dest) == digest == e.c.sha(source), f'Source changed during copy: {source}')
        dest.chmod(0o444)
        manifest['files'][key] = dict(copy=str(dest), sha256=digest)


def hashset(manifest):
    return {p: info['sha256'] for p, info in manifest['files'].items()}


def verify_files(manifest):
    for source, info in manifest['files'].items():
        e.check(e.c.sha(source) == e.c.sha(info['copy']) == info['sha256'], f'Frozen file changed: {source}')


def seal(path, value):
    """Write once, including sidecar, with no overwriting an earlier manifest."""
    import json
    path = Path(path)
    sidecar = path.with_suffix('.sha256')
    e.check(not path.exists() and not sidecar.exists(), 'Immutable artifact already exists')
    with path.open('x') as f:
        json.dump(value, f, allow_nan=False, separators=(',', ':'))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    digest = e.c.sha(path)
    with sidecar.open('x') as f:
        f.write(digest+'\n')
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return digest


def read_sealed(path):
    path = Path(path).resolve()
    e.check(e.c.sha(path) == path.with_suffix('.sha256').read_text().strip(), f'Immutable digest changed: {path}')
    return e.read_json(path)


def require_grid(manifest):
    grids = manifest['scs_radii']
    for n, count in plot.SCS_COUNTS.items():
        r = grids[str(n)]
        e.check(len(r) == count and r == sorted(set(r)) and r[0] == 0 and r[-1] <= 1.5,
                f'Invalid SCS radius cohort N={n}')
    expected = {(method, n, rho, s) for method in ('green', 'scs') for n in (50, 200, 800)
                if method == 'scs' or n != 50
                for rho in (e.RADII if method == 'green' else grids[str(n)]) for s in range(10)}
    actual = [(j['method'], j['Ntotal'], j['radius'], j['seed']) for j in manifest['jobs']]
    e.check(len(actual) == len(expected) == 780 and set(actual) == expected
            and len({j['id'] for j in manifest['jobs']}) == 780, 'Incomplete/duplicate 780-policy grid')


def membership_report(manifest):
    reports = [dict(id=j['id'], Ntotal=j['Ntotal'], radius=j['radius'], seed=j['seed'],
                    **{k: v for k, v in j['scs_historical_membership'].items() if k != 'offenders'})
               for j in manifest['jobs'] if j['method'] == 'scs']
    return dict(policy=e.SCS_MEMBERSHIP_POLICY, box_certified=False, coordinates_unchanged=True,
        policy_count=len(reports), saved_points_checked=sum(r['saved_points_checked'] for r in reports),
        positive_residual_count=sum(r['positive_residual_count'] for r in reports),
        count_above_reporting_threshold=sum(r['count_above_reporting_threshold'] for r in reports),
        max_positive_residual=max(r['max_positive_residual'] for r in reports), per_policy=reports,
        note='Counts include shared historical AP points in multiple selected pools; full offenders in manifest.')


def disk_gate(folder, required_bytes, scope):
    free = shutil.disk_usage(folder).free
    report = dict(scope=scope, required_bytes=int(required_bytes), free_bytes=free,
                  required_gib=required_bytes/GIB, free_gib=free/GIB,
                  note='Conservative reservation estimate, not a hard output-size cap.')
    print('Disk reservation:', report, flush=True)
    e.check(free >= required_bytes, f'Insufficient disk for {scope}: {report}')
    return report


def freeze_all(output, expected=None):
    sources = local_sources()
    before = {str(p): e.c.sha(p) for p in sources}
    manifest = e.freeze(Path(output))
    capture_sources(manifest, Path(output)/'frozen', sources)
    e.check(before == {str(p): e.c.sha(p) for p in sources}, 'Local source changed during freeze')
    manifest['runner_sources'] = before
    require_grid(manifest)
    if expected is not None:
        e.check(hashset(manifest) == hashset(expected), 'Full freeze source hashset differs from preflight')
        e.check(manifest['runner_sources'] == expected['runner_sources'], 'Source discovery differs from preflight')
    verify_files(manifest)
    return manifest


def validate_preflight(path):
    pref = read_sealed(path)
    e.check(pref['status'] == 'completed' and pref['test_generated'] is False
            and pref['job_count'] == 780, 'Preflight incomplete or accessed test')
    e.check(all(pref['phases'][name]['status'] == 'completed' for name in
                ('syntax', 'unit_tests', 'resources', 'full_freeze', 'numerical_training_only', 'final_verification')),
            'Incomplete preflight phases')
    manifest_path = Path(pref['manifest'])
    e.check(e.c.sha(manifest_path) == pref['manifest_sha256'], 'Preflight manifest binding changed')
    manifest = read_sealed(manifest_path)
    e.check(manifest['status'] == 'frozen' and 'test' not in manifest and manifest['versions'] == e.versions(),
            'Preflight environment/protocol changed')
    require_grid(manifest)
    e.check(pref['source_sha256'] == hashset(manifest), 'Preflight full hashset mismatch')
    e.check(manifest['runner_sources'] == {str(p): e.c.sha(p) for p in local_sources()},
            'Current exact source/doc set differs from preflight; rerun preflight')
    verify_files(manifest)
    return pref, manifest


def freeze_test(manifest, folder):
    """Called only after the complete production freeze and preflight binding."""
    require_grid(manifest)
    e.check(manifest['status'] == 'frozen', 'Cannot generate test before full freeze')
    arrays = e.c.gen(10000, 9999)
    test = dict(n_test=10000, seed=9999, arrays={})
    for name, a in zip(('xi1', 'xi2'), arrays):
        e.check(a.shape == (10000, 3) and a.dtype == e.np.float64 and e.np.isfinite(a).all()
                and (a >= 0).all(), 'Invalid generated matched test')
        digest = hashlib.sha256(a.tobytes(order='C')).hexdigest()
        old = manifest['cached_green_N50']['test_arrays'][name]
        e.check(digest == old['array_sha256'] and old['dtype'] == 'float64' and old['shape'] == [10000, 3],
                'Generated test differs from cached N50 test')
        path = Path(folder)/f'{name}.npy'
        with path.open('xb') as f:
            e.np.save(f, a, allow_pickle=False)
        path.chmod(0o444)
        test['arrays'][name] = dict(file=str(path.resolve()), file_sha256=e.c.sha(path),
            array_sha256=digest, dtype='float64', shape=[10000, 3])
    manifest['test'] = test


def compact(row, result_path):
    fields = ('id', 'method', 'Ntotal', 'Ntrain', 'Nval', 'seed', 'radius', 'status',
              'mean_oos', 'finite_pool_mean', 'start_utc', 'end_utc', 'wall_time_s',
              'iterations', 'barrier_iterations', 'solves_attempted', 'solves_accepted',
              'status_counts', 'worker_count', 'solver_threads', 'pid', 'phases', 'error',
              'parent_queue_elapsed_s', 'manifest_sha256', 'peak_rss_bytes')
    return dict({k: row[k] for k in fields if k in row}, result_path=str(result_path))


def run_worker(spec, manifest_path, output):
    output = Path(output)
    with (output/'logs'/f'{spec["id"]}.log').open('x', buffering=1) as log, \
            redirect_stdout(log), redirect_stderr(log):
        row = e.worker(spec, manifest_path, str(output/'policies'))
    row['peak_rss_bytes'] = rss_bytes()
    path = output/'policies'/spec['id']/'result.json'
    e.write_json(path, row)
    return compact(row, path)


def process_failure(spec, output, error, queue_elapsed):
    path = Path(output)/'process_failures'/f'{spec["id"]}.json'
    row = dict(id=spec['id'], method=spec['method'], Ntotal=spec['Ntotal'], radius=spec['radius'],
        seed=spec['seed'], status='process_failed', wall_time_s=None, start_utc=None, end_utc=e.c.utc(),
        error=error, parent_queue_elapsed_s=queue_elapsed,
        timing_note='Worker model wall time unavailable; parent queue elapsed is not evaluation time.')
    e.write_json(path, row)
    return compact(row, path)


def execute(manifest, manifest_path, output, summary):
    """Bound in-flight payloads; persist only compact progress, never repeated pools."""
    jobs = iter(manifest['jobs'])
    with ProcessPoolExecutor(max_workers=manifest['workers'], mp_context=multiprocessing.get_context('spawn'),
                             initializer=e.initialize_worker) as pool:
        pending = {}

        def submit():
            spec = next(jobs, None)
            if spec is None:
                return
            start = time.perf_counter()
            future = pool.submit(run_worker, spec, str(manifest_path), str(output))
            pending[future] = (spec, start)

        for _ in range(manifest['workers']):
            submit()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            failed = False
            for future in done:
                spec, queued = pending.pop(future)
                try:
                    row = future.result()
                except Exception:
                    row = process_failure(spec, output, traceback.format_exc(), time.perf_counter()-queued)
                summary['jobs'].append(row)
                failed |= row['status'] != 'completed'
            e.write_json(output/'summary.json', summary)
            if failed:
                # Do not silently retry a broken pool; retain already running policy records.
                for future in pending:
                    future.cancel()
                for future, (spec, queued) in pending.items():
                    if future.cancelled():
                        continue
                    try:
                        summary['jobs'].append(future.result())
                    except Exception:
                        summary['jobs'].append(process_failure(spec, output, traceback.format_exc(),
                                                              time.perf_counter()-queued))
                e.write_json(output/'summary.json', summary)
                raise RuntimeError('Policy/process failure; no partial plot or substituted result')
            for _ in done:
                submit()


def require_completed(manifest, rows, manifest_sha):
    require_grid(manifest)
    indexed = {r['id']: r for r in rows}
    e.check(len(indexed) == len(rows) == 780 and set(indexed) == {s['id'] for s in manifest['jobs']},
            'Missing or duplicate completed policies')
    for spec in manifest['jobs']:
        row = indexed[spec['id']]
        count = 10000 + sum(spec['sizes'])
        e.check(row['status'] == 'completed' and row['manifest_sha256'] == manifest_sha
                and all(row.get(k) == v for k, v in spec.items()), 'Changed/failed policy identity')
        e.check(row['solves_accepted'] == row['solves_attempted'] == count
                and row['status_counts'] == {'2': count}, 'Incomplete strict projection counts')
        e.check(e.np.isfinite([row['mean_oos'], row['finite_pool_mean'], row['wall_time_s']]).all()
                and row['wall_time_s'] >= 0, 'Nonfinite cost/runtime')


def final_verify(manifest, path, digest):
    e.check(e.c.sha(path) == digest == path.with_suffix('.sha256').read_text().strip(), 'Manifest changed')
    e.check(manifest['runner_sources'] == {str(p): e.c.sha(p) for p in local_sources()}, 'Source set changed')
    verify_files(manifest)
    for info in manifest['test']['arrays'].values():
        e.check(e.c.sha(info['file']) == info['file_sha256'], 'Test file changed')
        a = e.np.load(info['file'], mmap_mode='r', allow_pickle=False)
        e.check(str(a.dtype) == info['dtype'] and list(a.shape) == info['shape']
                and hashlib.sha256(a.tobytes()).hexdigest() == info['array_sha256'], 'Test array changed')


def runtime_aggregates(rows):
    groups = {}
    for row in rows:
        for scope in ('total', 'oos', 'is'):
            value = row.get('wall_time_s') if scope == 'total' else None
            if scope != 'total':
                phases = [row.get('phases', {}).get(scope+'_'+s) for s in ('targets', 'projection', 'cost')]
                if all(p and p['status'] == 'completed' for p in phases):
                    value = sum(p['wall_time_s'] for p in phases)
            key = (row['method'], row['Ntotal'], row['radius'], row['status'], scope)
            group = groups.setdefault(key, dict(method=key[0], Ntotal=key[1], radius=key[2],
                status=key[3], scope=scope, n_records=0, unavailable=0, seconds=[]))
            group['n_records'] += 1
            if value is None:
                group['unavailable'] += 1
            else:
                e.check(math.isfinite(value) and value >= 0, 'Invalid measured runtime')
                group['seconds'].append(value)
    for group in groups.values():
        values = group.pop('seconds')
        group['wall_time_s'] = e.stats(values) if values else None
        group['mean_human'] = e.c.human(group['wall_time_s']['mean']) if values else 'unavailable'
    return dict(timing_unit='seconds', groups=list(groups.values()),
                note='Status-separated per-model wall times; concurrent times are not sweep elapsed.')


def write_report(output, manifest, summary, aggregates):
    lines = ['# Fixed-policy R3 comparison', '', f"Status: **{summary['status']}**.",
        'No new training, CV, or test-based selection. Green training counts 40/160/640; SCS 50/200/800.',
        'Fresh strict green N200/N800 (180 policies) and SCS N50/N200/N800 (600 policies).',
        'Green N50 (90 policies), SAA-kNN, and SRO-FA are cached historical references.',
        'OOS: fixed seed 9999, 10,000 float64 samples. Previously inspected test: descriptive comparison.',
        'IS: mean of per-center maxima on the returned checkpoint finite pool, not an adversarial supremum.',
        'Green membership gate remains 1e-9. SCS saved coordinates are unchanged and not box-certified.',
        '', f"Host: {summary['resources']['host']}; affinity CPUs: {len(summary['resources']['affinity'])}; "
        f"spawn workers: {summary['workers']}; one numerical/solver thread each.",
        f"Preflight: {summary['preflight']}; SHA256: {summary['preflight_sha256']}.",
        f"Manifest: frozen/manifest.json; SHA256: {summary.get('manifest_sha256', 'unavailable')}.",
        f"Final source hashes verified: {summary.get('source_hash_final', False)}.",
        f"Current pipeline elapsed before reporting: {e.c.human(summary['wall_time_s'])} ({summary['wall_time_s']:.6f} s).",
        'Raw current times: policies/*/result.json; journals: policies/*/{oos,is}_solves.jsonl.',
        'Per-model backend wall times exclude final result serialization; overall sweep includes dispatch and worker I/O.',
        'Training references and AP/MP iteration timings remain in the checksummed frozen original summaries.',
        '', '## Current evaluation wall times (seconds)', '',
        '| Method | N | Radius | Status | Scope | n | Mean | Median | Min | Max |',
        '|---|---:|---:|---|---|---:|---:|---:|---:|---:|']
    for g in sorted(aggregates['groups'], key=lambda r:(r['method'],r['Ntotal'],r['radius'],r['status'],r['scope'])):
        st = g['wall_time_s']
        values = ' | '.join(f'{st[k]:.4f}' for k in ('mean','median','min','max')) if st else 'unavailable | unavailable | unavailable | unavailable'
        lines.append(f"| {g['method']} | {g['Ntotal']} | {g['radius']:g} | {g['status']} | {g['scope']} | {st['n'] if st else 0} | {values} |")
    lines += ['', '## Persisted historical training / reference timings', '',
        'These are old-run times, never current cache-loading times. Historical hardware/concurrency is only',
        'known where recorded in the frozen source; otherwise unavailable. Missing times require a rerun.',
        '', '| Method / scope | N | Radius / LR | n | Mean s | Median s | Min s | Max s |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    historical = []

    def add(label, n, parameter, values):
        values = [v for v in values if v is not None]
        st = e.stats(values) if values else None
        historical.append(dict(method=label, Ntotal=n, parameter=parameter, wall_time_s=st,
                               cached=True, current_wall_time_s=None,
                               mean_human=e.c.human(st['mean']) if st else 'unavailable'))
        text = ' | '.join(f'{st[k]:.4f}' for k in ('mean','median','min','max')) if st else 'unavailable | unavailable | unavailable | unavailable'
        lines.append(f"| {label} | {n} | {parameter} | {len(values)} | {text} |")

    source = manifest['green_training_reference']['summary']
    training = e.read_json(manifest['files'][source]['copy'])
    for n in (200, 800):
        for lr in (.005, .01, .05):
            arms = [r for r in training['phases']['nominal']['jobs'] if r['Ntotal'] == n and r['lr_net'] == lr and r['status'] == 'completed']
            add('Green nominal training', n, f'LR={lr}', [r['training'].get('wall_time_s') for r in arms])
            add('Green nominal train evaluation', n, f'LR={lr}', [r['train_evaluation'].get('wall_time_s') for r in arms])
        for rho in e.RADII:
            group = [r for r in training['phases']['ccg']['jobs'] if r['Ntotal'] == n and r['radius'] == rho and r['status'] == 'completed']
            add('Green CCG total (excludes shared nominal)', n, rho, [r.get('wall_time_s') for r in group])
        for phase in ('initialization', 'lr_validation', 'radius_validation'):
            add('Green '+phase, n, 'all', [r.get('wall_time_s') for r in training['phases'][phase]['jobs'] if r['Ntotal'] == n and r['status'] == 'completed'])
    for n in (50, 200, 800):
        add('SCS nominal/full CCG (not certified)', n, 'all', [])
        baseline = manifest['cached_baselines'][str(n)]
        for rho, group in sorted(baseline['sro_fa'].items(), key=lambda item:float(item[0])):
            if float(rho) > 1.5:
                continue
            for label, field in [('partition','time_partition'), ('solver','time_lp'), ('OOS','time_eval'), ('total','time_total')]:
                add('SRO-FA historical '+label, n, rho, [r.get(field) for r in group.values()])
    add('Green N50 nominal/CCG (not established here)', 50, 'all', [])
    add('SAA-kNN historical selection/CV (not established here)', '50/200/800', 'all', [])
    aggregates['historical'] = historical
    aggregates['historical_green_phase_elapsed'] = {name:{k:v for k,v in phase.items() if k != 'jobs'}
                                                   for name,phase in training['phases'].items()}
    lines += ['', 'SRO-FA totals/partition/evaluation use historical time.time; time_lp is saved solver time,',
        'not full optimization wrapper elapsed. No historical strict-status certificate is invented.',
        'Original cache runtime fields, SRO seed records, nominal epochs, and AP/MP iteration times are retained',
        'in the manifest and its frozen inputs. No new training, benchmark construction, or CV was performed.',
        '', '## Outputs', '', f"Published figures: {summary['figures']}; plot data: {summary['plot_data']}.",
        'runtime_aggregates.json contains current and persisted historical timings separately.',
        'summary.json contains compact per-policy result paths and phase clocks; full policies are not rewritten per result.']
    if 'scs_membership' in summary:
        m = summary['scs_membership']
        lines += ['', f"SCS membership: {m['positive_residual_count']} positive residual observations; "
                  f"{m['count_above_reporting_threshold']} above 1e-9; maximum {m['max_positive_residual']:.12g}."]
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    preflight_start = e.c.utc()
    pref, previous = validate_preflight(args.preflight)
    preflight_phase = dict(start_utc=preflight_start, end_utc=e.c.utc(), status='completed',
                           wall_time_s=time.perf_counter()-start)
    snapshot = resources()
    budget = choose_workers(snapshot, pref['peak_rss_bytes'])
    output = (args.output or ROOT/'results'/e.c.utc().replace(':','')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'policies', 'process_failures'):
        (output/name).mkdir()
    summary = dict(experiment='R3_fixed_policy_comparison_20260924', status='running', start_utc=preflight_start,
        output=str(output), preflight=str(args.preflight.resolve()), preflight_sha256=e.c.sha(args.preflight),
        resources=snapshot, resource_policy=budget, workers=budget['workers'], threads=1, versions=e.versions(),
        phases={'preflight_validation': preflight_phase}, jobs=[], figures=[], plot_data=None,
        source_hash_final=False, timing_unit='seconds',
        timing_clock='time.perf_counter', command=shlex.join([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
            '--preflight', str(args.preflight.resolve()), '--output', str(output)]),
        launcher={k:os.environ.get(k) for k in ('TMUX','COMPARISON_TMUX_SESSION','COMPARISON_LOG','COMPARISON_EXIT')},
        working_directory=os.getcwd(), invocation_argv=sys.orig_argv if argv is None else list(argv),
        new_training=False, new_selection=False)
    e.write_json(ROOT/'launch.json', summary)
    e.write_json(output/'launch.json', summary)
    e.write_json(output/'summary.json', summary)
    manifest = None
    try:
        with e.c.phase(summary['phases'], 'full_freeze') as phase:
            copies = sum(Path(v['copy']).stat().st_size for v in previous['files'].values())
            solves = 780*10000 + sum(sum(j['sizes']) for j in previous['jobs'])
            phase['disk'] = disk_gate(output, 2*copies + 4096*solves + 2*GIB, 'freeze, raw journals and reports')
            manifest = freeze_all(output, previous)
            manifest.update(workers=budget['workers'], resource_policy=budget, resources=snapshot,
                            preflight=summary['preflight'], preflight_sha256=summary['preflight_sha256'])
            del previous
            phase['peak_rss_bytes'] = rss_bytes()
        with e.c.phase(summary['phases'], 'matched_test_generation'):
            freeze_test(manifest, output/'frozen')
        manifest_path = output/'frozen/manifest.json'
        with e.c.phase(summary['phases'], 'seal_manifest'):
            digest = seal(manifest_path, manifest)
        summary.update(manifest=str(manifest_path), manifest_sha256=digest, scs_membership=membership_report(manifest))
        with e.c.phase(summary['phases'], 'evaluation_sweep'):
            execute(manifest, manifest_path, output, summary)
        with e.c.phase(summary['phases'], 'final_verification'):
            rows = [e.read_json(output/'policies'/spec['id']/'result.json') for spec in manifest['jobs']]
            require_completed(manifest, rows, digest)
            final_verify(manifest, manifest_path, digest)
            summary['source_hash_final'] = True
        with e.c.phase(summary['phases'], 'figure_and_plot_data'):
            data = plot.build_plot_data(manifest, rows)
            stage = output/'publication_staging'
            stage.mkdir()
            e.write_json(stage/'plot_data.json', data)
            plot.render(data, stage)
            # No public artifacts until both figures and all validations succeed.
            for name in ('plot_data.json', 'comparison_R3.png', 'comparison_R3.pdf'):
                e.check((stage/name).stat().st_size > 0, f'Missing render output: {name}')
            for name in ('plot_data.json', 'comparison_R3.png', 'comparison_R3.pdf'):
                (stage/name).rename(output/name)
            summary['figures'] = [str(output/name) for name in ('comparison_R3.png','comparison_R3.pdf')]
            summary['plot_data'] = str(output/'plot_data.json')
        summary['status'] = 'completed'
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        # Include in-flight policies that finished while the executor shut down after failure.
        indexed = {r['id']:r for r in summary['jobs']}
        for path in sorted((output/'policies').glob('*/result.json')):
            row = e.read_json(path)
            if row['id'] not in indexed:
                indexed[row['id']] = compact(row, path)
        summary['jobs'] = sorted(indexed.values(), key=lambda r:r['id'])
        summary['expected_jobs'] = 780
        summary['completed_jobs'] = sum(r['status'] == 'completed' for r in summary['jobs'])
        if manifest is not None:
            summary['missing_job_ids'] = [s['id'] for s in manifest['jobs'] if s['id'] not in indexed]
        summary.update(end_utc=e.c.utc(), wall_time_s=time.perf_counter()-start, peak_rss_bytes=rss_bytes())
        summary['wall_time_human'] = e.c.human(summary['wall_time_s'])
        aggregates = runtime_aggregates(summary['jobs'])
        aggregates.update(sweep=summary['phases'].get('evaluation_sweep'), end_to_end_wall_time_s=summary['wall_time_s'],
                          workers=summary['workers'], environment=snapshot)
        if manifest is not None:
            try:
                with e.c.phase(summary['phases'], 'reporting'):
                    write_report(output, manifest, summary, aggregates)
            except Exception:
                summary.update(status='failed', report_error=traceback.format_exc())
        summary.update(end_utc=e.c.utc(), wall_time_s=time.perf_counter()-start, peak_rss_bytes=rss_bytes())
        summary['wall_time_human'] = e.c.human(summary['wall_time_s'])
        summary['wall_time_scope'] = 'Entry through reporting; final timing footer and JSON serialization excluded.'
        aggregates['end_to_end_wall_time_s'] = summary['wall_time_s']
        if (output/'RESULTS.md').exists():
            with (output/'RESULTS.md').open('a') as report:
                report.write(f"\nEnd-to-end including reporting: {summary['wall_time_s']:.6f} s "
                             f"({summary['wall_time_human']}); final timing/artifact serialization excluded.\n")
        e.write_json(output/'runtime_aggregates.json', aggregates)
        e.write_json(output/'summary.json', summary)
        e.check(summary['status'] == 'completed', f'Comparison failed; inspect {output}/summary.json')
    return output


if __name__ == '__main__':
    main()