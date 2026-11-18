"""Preflight-bound BLACK replay. Importing this file does not import a solver.

Public CLI: preflight OUTPUT; production OUTPUT --preflight SUMMARY --sha SHA256.
Private _worker and _unit modes write envelopes, never full records to stdout.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import unittest

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
WORKERS = 16
GIB = 1024**3
LOCAL = {'backend.py', 'test_backend.py', 'plotting.py', 'test_plotting.py',
         'run.py', 'test_run.py', 'launch.py', 'README.md'}
UNIT_MODULES = ('test_backend', 'test_plotting', 'test_run')
SMOKES = ((3, 50, 9, .7), (10, 50, 8, .2), (10, 200, 6, .7),
          (3, 200, 0, .4), (3, 800, 0, .4), (10, 800, 0, 2.))
RESOURCE_SOURCES = (
    (BASE / 'green_R3_N200_N800_20260923/preflight.py', ('environment',)),
    (BASE / 'green_R10_N50_N200_N800_20260924/r10_resources.py', ('_cpus', 'resources')))
R10_SUMMARY = BASE / 'green_R10_N50_N200_N800_20260924/comparison_20260925/production_v1/summary.json'
HISTORICAL_RUNTIME = ROOT.parent / 'production_v1.launch/child_runtime.json'
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
PREFLIGHT_PHASES = ('syntax', 'unit_tests', 'collection', 'resources', 'source_copy',
                    'policy_seal', 'training_probes', 'journal_verification', 'final_verification')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def backend():
    module = importlib.import_module('backend')
    require(Path(module.__file__).resolve() == ROOT / 'backend.py', 'Wrong backend module')
    return module


def plotting():
    module = importlib.import_module('plotting')
    require(Path(module.__file__).resolve() == ROOT / 'plotting.py', 'Wrong plotting module')
    return module


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
    """Exclusive streaming JSON; never build a hundreds-of-MB JSON string."""
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, separators=(',', ':'))
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def seal(path, value):
    path = Path(path)
    write(path, value)
    digest = sha(path)
    with path.with_suffix('.sha256').open('x') as stream:
        stream.write(digest + '\n')
    path.chmod(0o444)
    path.with_suffix('.sha256').chmod(0o444)
    return digest


def sealed(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, 'Explicit SHA256 mismatch')
    require(path.with_suffix('.sha256').read_text().strip() == digest, 'Seal mismatch')
    value = read(path)
    require(sha(path) == digest, 'Sealed file changed while reading')
    return value


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid duration/count')
    return value


def timing(row):
    seconds(row['wall_time_s'])
    for key in ('start_utc', 'end_utc'):
        require(datetime.fromisoformat(row[key]).utcoffset() is not None, 'Missing timezone')
    # UTC can be adjusted by NTP; durations are monotonic, not UTC differences.


def human(value):
    return f'{value:.6f} s / {value/60:.6f} min / {value/3600:.6f} h'


@contextmanager
def phase(phases, name):
    start = time.perf_counter()
    row = phases[name] = dict(status='running', start_utc=utc(), clock='time.perf_counter')
    try:
        yield row
    except BaseException:
        row['status'] = 'failed'
        raise
    else:
        row['status'] = 'completed'
    finally:
        row.update(end_utc=utc(), wall_time_s=time.perf_counter()-start)


def local_sources():
    actual = {p.name for p in ROOT.iterdir() if p.is_file() and
              (p.suffix == '.py' or p.name == 'README.md')}
    require(actual == LOCAL, f'Local source set changed: {actual ^ LOCAL}')
    return sorted(ROOT / name for name in LOCAL)


def source_hashes():
    paths = set(local_sources()) | set(backend().source_paths())
    paths.update(path for path, _ in RESOURCE_SOURCES)
    paths.update(plotting().RENDERER_PATHS.values())
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def verify_files(manifest):
    bindings = dict(manifest['input_sha256'])
    for source, digest in manifest['source_sha256'].items():
        require(source not in bindings or bindings[source] == digest, 'Conflicting source binding')
        bindings[source] = digest
    require(set(bindings) <= set(manifest['files']), 'Missing copied inputs')
    for source, info in manifest['files'].items():
        copy = Path(info['copy'])
        require(Path(source).is_absolute() and copy.is_absolute() and copy.resolve() != Path(source).resolve(),
                'Require private absolute copies')
        require(bindings.get(source, info['sha256']) == info['sha256'], 'Copy binding mismatch')
        require(sha(source) == sha(copy) == info['sha256'], f'Original/copy changed: {source}')
        require(copy.stat().st_mode & 0o222 == 0, f'Copy is writable: {copy}')
    for path in local_sources():
        require(manifest['source_sha256'][str(path)] == sha(path), 'Executed local source changed')


def resource_snapshot(hashes):
    """Reuse the existing cgroup-v1/v2 ancestor audit, not host-only estimates."""
    namespace = dict(Path=Path, os=os, sys=sys, platform=platform, utc=utc,
                     THREAD_ENV=THREAD_ENV, peak_rss_bytes=lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    for path, names in RESOURCE_SOURCES:
        require(sha(path) == hashes[str(path)], 'Resource auditor changed')
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        require({n.name for n in nodes} == set(names), 'Resource auditor definitions missing')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['resources']()


def resource_gate(snapshot, peaks, manifest_bytes=0):
    require(peaks and all(seconds(p) > 0 for p in peaks), 'Measured RSS required')
    effective = len(snapshot['affinity'])
    if snapshot['cpu_quota_cores'] is not None:
        effective = min(effective, math.floor(snapshot['cpu_quota_cores']))
    reserve_cpu = max(4, math.ceil(.1*effective))
    load = math.ceil(max(snapshot['load_average']))
    available = snapshot['memory_available_bytes']
    reserve_memory = max(8*GIB, math.ceil(.1*available))
    per_worker = max(4*GIB, math.ceil(1.5*max(peaks)), 8*manifest_bytes)
    require(effective-reserve_cpu-load >= WORKERS and
            available-reserve_memory >= WORKERS*per_worker,
            'Insufficient reserved resources for 16 workers; no reduced-worker fallback')
    return dict(workers=WORKERS, threads=1, effective_cpu_slots=effective,
                reserved_cpu_slots=reserve_cpu, background_load_slots=load,
                reserved_memory_bytes=reserve_memory, per_worker_memory_bytes=per_worker,
                measured_peak_rss_bytes=max(peaks), manifest_bytes=manifest_bytes,
                scope='one joint pool; RSS estimates are not guarantees')


def disk_gate(output, required):
    free = shutil.disk_usage(output).free
    reserve = max(8*GIB, math.ceil(.1*free))
    require(free-reserve >= required, 'Insufficient disk for copies/journals/large metadata')
    return dict(free_bytes=free, reserved_bytes=reserve, required_bytes=required)


def cohort(jobs):
    backend().cohort(jobs)
    require(all(Path(j['id']).name == j['id'] and j['id'] not in ('.', '..') for j in jobs),
            'Invalid policy path identity')
    caps = [j for j in jobs if j['capped_nonconverged']]
    require(len(caps) == 11 and sum(j['comparison_derived_cap_selection'] for j in jobs) == 10,
            'All eleven caps must be retained')
    require([(j['R'], j['Ntotal'], j['seed'], j['radius']) for j in caps if j['R'] == 3]
            == [(3, 50, 9, .7)], 'Cached-source cap identity changed')
    for job in jobs:
        require(job['retained'] is True and job['converged'] == (not job['capped_nonconverged']), 'Cap masked')
        require(job['selection_rule'] == backend().SELECTION, 'Selection rule changed')
        evidence = job['selection_evidence']
        require(len(evidence) == job['stopping_candidate_id']+1 == job['assessed_candidate_count'] and
                [e['candidate_id'] for e in evidence] == list(range(len(evidence))), 'Incomplete full prefix')
        for e in evidence:
            require(type(e['ap_incumbent_mean']) in (int, float) and math.isfinite(e['ap_incumbent_mean']),
                    'Nonfinite AP incumbent')
        winner = min(evidence, key=lambda e: e['ap_incumbent_mean'])
        require(winner['candidate_id'] == job['selected_candidate_id'], 'Not earliest full-prefix minimum')
        for key in ('checkpoint_sha256', 'state_sha256', 'pool_sha256', 'pool_file_sha256'):
            require(winner[key] == job[key], f'Selected own-policy identity mismatch: {key}')
        if job['capped_nonconverged']:
            require(job['terminal_status'] == 'iteration_cap' and job['stopping_candidate_id'] == 40,
                    'Cap lacks terminal40')


def smoke_jobs(jobs):
    index = {(j['R'], j['Ntotal'], j['seed'], j['radius']): j for j in jobs}
    require(len(index) == len(jobs), 'Duplicate policy identity')
    chosen = [index[key] for key in SMOKES]
    require(all(j['capped_nonconverged'] for j in chosen[:3]), 'Required capped probes changed')
    return chosen


def compact_bases(collection):
    """Keep the original manifests verbatim on disk; do not duplicate their huge file maps."""
    collection['base_comparisons'] = {
        r: {**{k: v for k, v in base.items() if k != 'files'},
            'files_metadata_reference': base['manifest_path']}
        for r, base in collection['base_comparisons'].items()}


def unit_command():
    return [sys.executable, '-B', '-u', str(ROOT / 'run.py'), '_unit']


def run_units(output):
    command = unit_command() + [str(output / 'unit_evidence.json')]
    with (output / 'unit_tests.log').open('x') as log:
        code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               env={**os.environ, **THREAD_ENV, 'PYTHONDONTWRITEBYTECODE': '1'})
    evidence = read(output / 'unit_evidence.json')
    evidence.update(returncode=code, command=command, path=str(output / 'unit_evidence.json'),
                    sha256=sha(output / 'unit_evidence.json'), log=str(output / 'unit_tests.log'),
                    log_sha256=sha(output / 'unit_tests.log'))
    return evidence


def validate_units(evidence):
    require(evidence['returncode'] == 0 and evidence['status'] == 'completed' and
            evidence['modules'] == list(UNIT_MODULES), 'Unit suites failed/incomplete')
    require(evidence['command'] == unit_command() + [evidence['path']], 'Unit invocation changed')
    require(type(evidence['tests_run']) is int and evidence['tests_run'] == evidence['tests_discovered'] > 0,
            'Unit test counts incomplete')
    require(evidence['skipped'] == evidence['errors'] == evidence['failures'] ==
            evidence['expected_failures'] == evidence['unexpected_successes'] == 0, 'Unit suites did not all pass')
    require(len(evidence['test_ids']) == len(set(evidence['test_ids'])) == evidence['tests_run'] and
            all(any(i.startswith(m+'.') for i in evidence['test_ids']) for m in UNIT_MODULES), 'Unit suite coverage missing')
    for key, hash_key in (('path', 'sha256'), ('log', 'log_sha256')):
        require(sha(evidence[key]) == evidence[hash_key], 'Unit evidence changed')
    raw = read(evidence['path'])
    require(all(evidence[k] == v for k, v in raw.items()), 'Unit evidence envelope mismatch')
    timing(evidence)


def _unit(output):
    start, started = time.perf_counter(), utc()
    suite = unittest.defaultTestLoader.loadTestsFromNames(UNIT_MODULES)
    def identifiers(node):
        if isinstance(node, unittest.TestSuite):
            return [i for child in node for i in identifiers(child)]
        return [node.id()]
    ids = identifiers(suite)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    ok = result.wasSuccessful() and not result.skipped and not result.expectedFailures
    write(output, dict(status='completed' if ok else 'failed', modules=list(UNIT_MODULES),
        tests_discovered=len(ids), tests_run=result.testsRun, test_ids=ids,
        skipped=len(result.skipped), failures=len(result.failures), errors=len(result.errors),
        expected_failures=len(result.expectedFailures), unexpected_successes=len(result.unexpectedSuccesses),
        start_utc=started, end_utc=utc(), wall_time_s=time.perf_counter()-start))
    return 0 if ok else 1


def _worker(manifest_path, identifier, output, smoke):
    """One backend context per fresh process, one policy, compact IPC only."""
    os.environ.update(THREAD_ENV)
    sys.dont_write_bytecode = True
    start, started = time.perf_counter(), utc()
    envelope = dict(id=identifier, status='failed', start_utc=started, pid=os.getpid())
    try:
        b = backend()
        manifest, _ = b._sealed(manifest_path)
        matches = [j for j in manifest['jobs'] if j['id'] == identifier]
        require(len(matches) == 1, 'Unknown worker policy')
        job = matches[0]
        del manifest
        row = (b.training_smoke if smoke else b.worker)(job, manifest_path, output / 'policies')
        path = Path(row['result_path'])
        digest = sha(path)
        require(row['result_sha256'] == row['summary_sha256'] == digest, 'Backend returned a wrong result hash')
        raw = read(path)
        # Backend hashes are added after serialization; compare exactly apart from them.
        require({k: v for k, v in row.items() if k not in ('result_sha256', 'summary_sha256')} == raw,
                'Backend returned/persisted result mismatch')
        envelope.update(status=row['status'], result_path=str(path), result_sha256=digest,
                        model_wall_time_s=row['wall_time_s'])
    except BaseException:
        envelope['error'] = traceback.format_exc()
    finally:
        envelope.update(end_utc=utc(), wall_time_s=time.perf_counter()-start,
                        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                        timing_clock='time.perf_counter', timing_unit='seconds')
        write(output / 'envelopes' / f'{identifier}.json', envelope)
    return 0 if envelope['status'] == 'completed' else 1


def execute(jobs, manifest_path, output, smoke=False):
    for name in ('logs', 'envelopes', 'policies'):
        (output / name).mkdir(exist_ok=False)
    def one(job):
        start, started = time.perf_counter(), utc()
        command = [sys.executable, '-B', '-u', str(ROOT / 'run.py'), '_worker', str(output),
                   '--manifest', str(manifest_path), '--job', job['id']]
        if smoke:
            command.append('--smoke')
        result = dict(id=job['id'], status='process_failed', start_utc=started, command=command)
        try:
            with (output / 'logs' / f'{job["id"]}.log').open('x') as log:
                code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, **THREAD_ENV, 'PYTHONDONTWRITEBYTECODE': '1',
                         'GRB_LICENSE_FILE': '/data/home/aviyac/gurobi.lic'})
            result['returncode'] = code
            path = output / 'envelopes' / f'{job["id"]}.json'
            result.update(envelope_path=str(path), envelope_sha256=sha(path), envelope=read(path))
            if code == 0 and result['envelope']['status'] == 'completed':
                result['status'] = 'completed'
        except Exception:
            result['error'] = traceback.format_exc()
        finally:
            result.update(end_utc=utc(), wall_time_s=time.perf_counter()-start)
        return result
    rows = []
    # Threads only supervise subprocesses: R3/R10 numerical globals never coexist.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(one, job) for job in sorted(jobs, key=lambda j:
                   (j['radius'], j['seed'], j['Ntotal'], j['R']))]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f'{len(rows)}/{len(jobs)} {row["id"]}: {row["status"]}', flush=True)
    return sorted(rows, key=lambda r: r['id'])


def journal_verify(row, job, output, smoke=False):
    """Verify real raw acceptance/physical checks; journals do NOT contain costs."""
    names = [('training_probe', 2)] if smoke else [('oos', 10000), ('is', sum(job['sizes']))]
    require(set(row['journals']) == {name+'_solves' for name, _ in names}, 'Unexpected/missing journals')
    total, iterations, barrier, proofs = 0, 0., 0, {}
    for name, expected in names:
        info = row['journals'][name+'_solves']
        path = output / 'policies' / job['id'] / f'{name}_solves.jsonl'
        require(Path(info['path']) == path and sha(path) == info['sha256'], 'Journal path/hash mismatch')
        count, center, offset, it, bi = 0, 0, 0, 0., 0
        with path.open() as stream:
            for count, line in enumerate(stream, 1):
                d = json.loads(line)
                require(type(d['row']) is int and d['row'] == count-1 and d['phase'] == name and
                        type(d['status']) is int and d['status'] == 2 and d['accepted'] is True,
                        'Failed/misordered strict journal row')
                timing(d)
                seconds(d['controller_wall_time_s'])
                require(seconds(d['physical_residual']) <= 1e-4, 'Physical residual exceeds 1e-4')
                it += seconds(d['iterations'])
                require(type(d['barrier_iterations']) is int, 'Invalid barrier iteration count')
                bi += seconds(d['barrier_iterations'])
                require('error' not in d, 'Journal has a solve error')
                if name == 'is':
                    require(center < len(job['sizes']) and d['center_id'] == center and
                            d['original_sample_id'] == job['original_sample_ids'][center], 'IS center mapping mismatch')
                    offset += 1
                    if offset == job['sizes'][center]:
                        center, offset = center+1, 0
        require(count == expected, 'Incomplete strict journal count')
        if name == 'is':
            require(center == len(job['sizes']) and offset == 0, 'Incomplete own-pool coverage')
        projection = row['phases'][name+'_projection']
        timing(projection)
        require(projection['status'] == 'completed' and
                projection['solves_attempted'] == projection['solves_accepted'] == projection['sample_count'] == count and
                projection['status_counts'] == {'2': count} and projection['iterations'] == it and
                projection['barrier_iterations'] == bi and projection['solver_threads'] == 1 and
                projection['solver_settings']['Threads'] == 1, 'Journal/projection counters differ')
        require(sha(path) == info['sha256'], 'Journal changed during verification')
        proofs[str(path)] = info['sha256']
        total, iterations, barrier = total+count, iterations+it, barrier+bi
    require(type(row['solves_attempted']) is int and type(row['solves_accepted']) is int and
            row['solves_attempted'] == row['solves_accepted'] == total and row['status_counts'] == {'2': total} and
            row['iterations'] == iterations and row['barrier_iterations'] == barrier,
            'Result counters differ from actual journals')
    return total, proofs


def normalize_result(job, process, manifest_path, output, smoke=False):
    require(process['id'] == job['id'] and process['status'] == 'completed' and process['returncode'] == 0,
            'Worker process failed')
    timing(process)
    require(sha(process['envelope_path']) == process['envelope_sha256'] and
            read(process['envelope_path']) == process['envelope'], 'Worker envelope changed')
    envelope = process['envelope']
    timing(envelope)
    require(envelope['id'] == job['id'] and envelope['status'] == 'completed' and
            seconds(envelope['peak_rss_bytes']) > 0, 'Invalid worker envelope')
    path = output / 'policies' / job['id'] / 'result.json'
    digest = sha(path)
    require(envelope['result_path'] == str(path) and envelope['result_sha256'] == digest, 'Raw result binding mismatch')
    row = read(path)
    require(all(row[k] == v for k, v in job.items()), 'Raw result differs from frozen job')
    require(row['status'] == 'completed' and row['inputs_unchanged'] is True and row['cached'] is False and
            row['manifest_sha256'] == sha(manifest_path) and row['manifest_path'] == str(manifest_path) and
            row['preflight_only'] is smoke and row['test_evaluated'] is (not smoke) and
            row['solver_threads'] == row['torch_threads'] == row['torch_interop_threads'] == 1 and
            row['worker_count'] == WORKERS and row['timing_clock'] == 'time.perf_counter' and
            row['timing_unit'] == 'seconds', 'Result protocol mismatch')
    timing(row)
    require(envelope['model_wall_time_s'] == row['wall_time_s'], 'Worker/model timing mismatch')
    for entry in row['phases'].values():
        timing(entry)
        require(entry['status'] == 'completed', 'Incomplete model phase')
    if smoke:
        require(row['training_probe_original_sample_ids'] == job['original_sample_ids'][:2] and
                row['mean_oos'] is None and row['finite_pool_mean'] is None and
                math.isfinite(row['training_probe_mean']) and 'test' not in row, 'Not a training-only probe')
    else:
        import numpy as np
        maxima = np.asarray(row['per_center_maxima'], dtype=np.float64)
        require(maxima.shape == (job['Ntrain'],) and np.isfinite(maxima).all() and
                row['finite_pool_mean'] == float(maxima.mean()) and math.isfinite(row['mean_oos']),
                'Invalid saved own-pool maxima/mean')
    total, proofs = journal_verify(row, job, output, smoke)
    require(sha(path) == digest, 'Raw result changed during verification')
    # Enrich only the display row. Never rewrite the raw, hashed backend result.
    row.update(result_sha256=digest, summary_sha256=digest,
               source_prefix_timings=job['training_runtime_reference'], solves_physical_accepted=total)
    reference = dict(result_path=str(path), result_sha256=digest, status='completed',
        journal_verified=True, raw_proofs=proofs,
        scope='Actual strict/physical journal acceptance; costs/maxima are bound evaluator outputs, not journal-recomputed')
    return row, reference


def verify_all_results(jobs, processes, manifest_path, output, smoke=False):
    require(len(processes) == len(jobs) == len({p['id'] for p in processes}) and
            {p['id'] for p in processes} == {j['id'] for j in jobs}, 'Missing/duplicate worker results')
    indexed = {p['id']: p for p in processes}
    rows, references, failures = [], {}, []
    for job in jobs:
        try:
            row, ref = normalize_result(job, indexed[job['id']], manifest_path, output, smoke)
            rows.append(row)
            references[job['id']] = ref
        except Exception:
            failures.append(dict(id=job['id'], error=traceback.format_exc()))
    write(output / 'journal_verification.json', dict(status='failed' if failures else 'completed',
          verified=len(rows), expected=len(jobs), results=references, failures=failures))
    require(not failures, f'{len(failures)} unverified results; no incomplete publication')
    return rows, references


def validate_preflight_evidence(pref, manifest):
    require(pref['status'] == 'completed' and pref['mode'] == 'preflight' and
            pref['test_bytes_accessed'] is False and pref['job_count'] == 570 and
            pref['workers'] == WORKERS and manifest['workers'] == WORKERS and
            manifest['status'] == 'frozen' and manifest['policies_frozen_before_test'] is True and
            manifest['threads'] == 1 and manifest['physical_tolerance'] == 1e-4 and
            manifest['membership_tolerance'] == 1e-9 and manifest['test_bytes_accessed'] is False and
            manifest['test_evaluated'] is False and 'tests' not in manifest, 'Incomplete/non-training preflight')
    timing(pref)
    for name in PREFLIGHT_PHASES:
        entry = pref['phases'][name]
        require(entry['status'] == 'completed', f'Incomplete preflight phase: {name}')
        timing(entry)
    require(set(pref['phases']['syntax']['files']) == {str(p) for p in local_sources() if p.suffix == '.py'},
            'Incomplete syntax gate')
    validate_units(pref['unit_tests'])
    cohort(manifest['jobs'])
    require(len(pref['processes']) == 6 and len({p['id'] for p in pref['processes']}) == 6 and
            {p['id'] for p in pref['processes']} == {j['id'] for j in smoke_jobs(manifest['jobs'])},
            'Require all six real two-center probes')
    require(pref['solves_attempted'] == pref['solves_accepted'] == 12 and pref['status_counts'] == {'2': 12},
            'Require twelve accepted real training QPs')
    require(pref['source_sha256'] == manifest['source_sha256'], 'Preflight source set mismatch')
    forbidden = {str(path) for path in plotting().BASE_PATHS.values()}
    forbidden.update(info['file'] for base in manifest['base_comparisons'].values()
                     for info in base['test']['arrays'].values())
    require(forbidden.isdisjoint(manifest['files']), 'Policy freeze included test/base plot bytes')


def load_preflight(path, expected):
    require(isinstance(expected, str) and len(expected) == 64, 'Production requires preflight SHA256')
    path = Path(path).resolve()
    pref = sealed(path, expected)
    manifest_path = Path(pref['policies_manifest'])
    manifest = sealed(manifest_path, pref['policies_manifest_sha256'])
    validate_preflight_evidence(pref, manifest)
    verify_files(manifest)
    # Re-read the real journals; a completed label is not a preflight certificate.
    indexed = {p['id']: p for p in pref['processes']}
    for job in smoke_jobs(manifest['jobs']):
        normalize_result(job, indexed[job['id']], manifest_path, path.parent, True)
    return pref, manifest


def bind_after_seal(manifest, policy_path, policy_sha, output):
    """Only called after all 570 policy choices have a persistent immutable seal."""
    require(sha(policy_path) == policy_sha and sealed(policy_path, policy_sha) == manifest,
            'Policies must be sealed before test binding')
    require('tests' not in manifest and manifest['test_bytes_accessed'] is False, 'Already bound tests')
    b, p = backend(), plotting()
    final = {**manifest, 'files': dict(manifest['files']), 'input_sha256': dict(manifest['input_sha256']),
             'tests': {}, 'policies_manifest': str(policy_path), 'policies_manifest_sha256': policy_sha,
             'base_plot_data': {}, 'test_bytes_accessed': True}
    additions = {}
    for r in (3, 10):
        test = manifest['base_comparisons'][str(r)]['test']
        require(test['seed'] == 9999 and test['n_test'] == 10000, 'Wrong common test protocol')
        final['tests'][str(r)] = test
        for name in ('xi1', 'xi2'):
            info = test['arrays'][name]
            require(info['shape'] == [10000, r] and info['dtype'] == 'float64', 'Wrong common test dimensions')
            additions[info['file']] = info['file_sha256']
        additions[str(p.BASE_PATHS[r])] = sha(p.BASE_PATHS[r])
    additions[str(R10_SUMMARY)] = sha(R10_SUMMARY)
    copied = b.materialize(dict(input_sha256=additions, source_sha256={}), output / 'test_and_base_copies')
    for source, info in copied.items():
        require(source not in final['files'], 'Post-seal input was already read/copied during selection')
        final['files'][source] = info
        final['input_sha256'][source] = info['sha256']
    import numpy as np
    for r, test in final['tests'].items():
        for info in test['arrays'].values():
            array = np.load(final['files'][info['file']]['copy'], mmap_mode='r', allow_pickle=False)
            require(list(array.shape) == info['shape'] == [10000, int(r)] and
                    str(array.dtype) == info['dtype'] == 'float64' and np.isfinite(array).all() and
                    (array >= 0).all() and hashlib.sha256(array.tobytes(order='C')).hexdigest() == info['array_sha256'],
                    'Original matched test array bytes/shape/value changed')
    original = read(copied[str(R10_SUMMARY)]['copy'])
    require(original['status'] == 'completed' and original['source_hash_final'] is True and
            original['plot_data'] == str(p.BASE_PATHS[10]) and
            original['manifest'] == str(b.BASE_MANIFESTS[10]) and
            original['manifest_sha256'] == b.LOCKS[b.BASE_MANIFESTS[10]], 'Original R10 plot pointer/manifest mismatch')
    final['original_R10_summary'] = dict(path=str(R10_SUMMARY), sha256=additions[str(R10_SUMMARY)])
    for r in (3, 10):
        base = read(copied[str(p.BASE_PATHS[r])]['copy'])
        require(base['test'] == final['tests'][str(r)], 'Base plot/test metadata mismatch')
        if r == 10:
            require(base['provenance']['manifest_sha256'] == b.LOCKS[b.BASE_MANIFESTS[10]], 'R10 plot provenance changed')
        final['base_plot_data'][str(r)] = dict(path=str(p.BASE_PATHS[r]),
            sha256=additions[str(p.BASE_PATHS[r])], canonical_sha256=p.canonical_sha256(base))
    return final


def result_bindings_unchanged(references):
    for ref in references.values():
        require(sha(ref['result_path']) == ref['result_sha256'], 'Result changed before publication')
        for path, digest in ref['raw_proofs'].items():
            require(sha(path) == digest, 'Journal changed before publication')


def render_both(manifest, rows, references, stage):
    p = plotting()
    # Compact global identities, not 570 copies of full source-prefix metadata.
    fields = p.IDENTITY_FIELDS + p.SELECTION_FIELDS
    bindings = dict(input_sha256=manifest['input_sha256'], source_sha256=manifest['source_sha256'],
        base_sha256={r: info['canonical_sha256'] for r, info in manifest['base_plot_data'].items()},
        renderers={str(r): dict(path=str(path), sha256=manifest['source_sha256'][str(path)])
                   for r, path in p.RENDERER_PATHS.items()},
        test=manifest['tests'], jobs={j['id']: {k: j[k] for k in fields} for j in manifest['jobs']},
        results=references, selection_rule=p.SELECTION_RULE)
    comparisons, paths = [], []
    for r in (3, 10):
        source = manifest['base_plot_data'][str(r)]['path']
        base = read(manifest['files'][source]['copy'])
        data = p.build_overlay(base, [row for row in rows if row['R'] == r], r, bindings)
        paths.extend(p.render(data, stage))
        for panel in data['panels']:
            green, black = panel['green'], panel['black']
            for i, rho in enumerate(black['radius']):
                if rho in green['radius']:
                    gi = green['radius'].index(rho)
                    comparisons.append(dict(R=r, Ntotal=panel['Ntotal'], radius=rho,
                        black_oos=black['oos']['mean'][i], green_oos=green['oos']['mean'][gi],
                        black_minus_green=black['oos']['mean'][i]-green['oos']['mean'][gi],
                        black_own_pool_is=black['finite_pool']['mean'][i],
                        green_own_pool_is=green['finite_pool']['mean'][gi]))
    require(len(paths) == 6 and all(path.is_file() and path.stat().st_size > 0 for path in paths),
            'Both complete R3/R10 figures and plot data required')
    return comparisons, {path.name: sha(path) for path in paths}


def stats(values):
    require(values and all(seconds(v) >= 0 for v in values), 'Invalid runtime group')
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values))


def runtime_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['status'] == 'completed' and row['cached'] is False:
            groups[row['R'], row['Ntotal'], row['radius']].append(row['wall_time_s'])
    return [dict(method='black_fresh_evaluation', R=r, Ntotal=n, radius=rho, **stats(values))
            for (r, n, rho), values in sorted(groups.items())]


def report(output, manifest, summary, rows, comparisons):
    aggregates = runtime_groups(rows)
    historical = read(manifest['files'][str(HISTORICAL_RUNTIME)]['copy'])
    require(historical['timing_clock'] == 'time.perf_counter' and historical['timing_unit'] == 'seconds' and
            historical['status'] == 'nonconverged' and historical['exit_code'] == 2 and
            Path(historical['command'][3]) == ROOT.parent / 'campaign.py', 'Historical continuation timing identity changed')
    seconds(historical['wall_time_s'])
    original_manifest_source = str(backend().PRODUCTION / 'manifest.json')
    original_manifest = read(manifest['files'][original_manifest_source]['copy'])
    historical_concurrency = dict(workers=original_manifest['workers'], protocol=original_manifest['protocol'])
    require(historical_concurrency['workers']['ccg'] == 16 and
        historical_concurrency['protocol']['torch_threads'] == 1 and
        historical_concurrency['protocol']['gurobi_threads'] == 1, 'Historical continuation concurrency changed')
    del original_manifest
    nominal = {}
    for job in manifest['jobs']:
        key = (job['R'], job['Ntotal'], job['seed'])
        item = job['training_runtime_reference']['nominal']
        require(key not in nominal or nominal[key] == item, 'Conflicting persisted nominal timing')
        nominal[key] = item
    nominal_groups = defaultdict(list)
    for (r, n, seed), item in nominal.items():
        require(item['training_timing']['status'] == 'completed', 'Incomplete historical nominal training')
        nominal_groups[r, n].append(seconds(item['training_timing']['wall_time_s']))
    nominal_aggregates = [dict(R=r, Ntotal=n, **stats(values))
                          for (r, n), values in sorted(nominal_groups.items())]
    raw = dict(status=summary['status'], timing_unit='seconds', current_evaluation_groups=aggregates,
        current_phases=summary['phases'], workers=WORKERS, threads_per_worker=1,
        environment=summary.get('resources'), historical_continuation_invocation=historical,
        historical_continuation_concurrency=historical_concurrency,
        historical_nominal=[dict(R=k[0], Ntotal=k[1], seed=k[2], evidence=v) for k, v in sorted(nominal.items())],
        historical_nominal_aggregates=nominal_aggregates,
        final_pipeline_summary=str(Path(summary['output']) / 'summary.json'),
        final_pipeline_note='Authoritative final status and complete report/final-verification durations are in the sealed pipeline summary.',
        source_prefix_timings=[dict(id=j['id'], R=j['R'], Ntotal=j['Ntotal'], radius=j['radius'], seed=j['seed'],
            terminal_status=j['terminal_status'], evidence=j['training_runtime_reference']) for j in manifest['jobs']],
        note='Historical nonconverged whole sweep is separate; per-iteration AP/MP retained, never summed as parallel elapsed.',
        current_training='not run', current_sro_fa='not run', current_cv='not run',
        excluded_processes=[p for p in summary['processes'] if p['status'] != 'completed'])
    write(output / 'runtime_report.json', raw)
    write(output / 'fixed_radius_comparison.json', comparisons)
    lines = ['# BLACK fixed-policy comparison', '',
        'Fresh replay only; warm-started historical continuation is not training from scratch. '
        'All 570 policies and all 11 capped/nonconverged training policies are retained. '
        'The historically known 10,000-row seed9999 tests are reused, not a newly untouched holdout. '
        'No OOS-based checkpoint/radius selection or interpolation is performed.', '',
        'Own-pool IS is a mean of per-center maxima, not an adversarial supremum. '
        'Original green/SCS/SRO-FA/SAA-kNN numbers, grids, gray references and caveats remain unchanged. '
        'R10 black alone extends to radius 2; other curves are not extrapolated.', '',
        '## Fixed common radii (cost differences, not selected winners)', '',
        '| R | N | Radius | Black OOS | Green OOS | Black − green | Black own-pool IS | Green own-pool IS |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append('| '+ ' | '.join(f'{row[k]:.8g}' for k in ('R', 'Ntotal', 'radius', 'black_oos',
            'green_oos', 'black_minus_green', 'black_own_pool_is', 'green_own_pool_is'))+' |')
    lines += ['', '## Current replay wall times (completed, journal-verified models only)', '',
        '| R | N | Radius | Runs | Mean s | Median s | Min s | Max s |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in aggregates:
        lines.append('| '+' | '.join(f'{row[k]:.8g}' for k in
                     ('R', 'Ntotal', 'radius', 'count', 'mean', 'median', 'min', 'max'))+' |')
    lines += ['', '## Persisted nominal training (once per R/N/seed, not repeated per radius)', '',
        '| R | N | Seeds | Mean s | Median s | Min s | Max s |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for row in nominal_aggregates:
        lines.append('| '+' | '.join(f'{row[k]:.8g}' for k in
                     ('R', 'Ntotal', 'count', 'mean', 'median', 'min', 'max'))+' |')
    lines += ['', '| Runtime category | Evidence / time |', '|---|---|',
        '| Nominal neural training | No new training; persisted per-seed evidence in runtime_report.json; missing fields unavailable, rerun required |',
        '| Robust neural/CCG | No new AP/MP; persisted per-iteration AP/MP and nominal initialization kept separately |',
        f'| Historical BLACK continuation whole invocation (nonconverged) | {human(historical["wall_time_s"])}; 16 workers, one solver/Torch thread each; not a completed-model aggregate or current timing |',
        '| SRO-FA partition / optimization / OOS / total | No new run; original base provenance unchanged; missing historical fields unavailable, rerun required |',
        '| Hyperparameter selection / CV / benchmark construction | No new run; cached original evidence only |',
        '| Final neural test and own-pool evaluation | Per-model table; separate oos/is targets, projection and cost phases in raw results |']
    for name, item in summary['phases'].items():
        if 'wall_time_s' in item:
            lines.append(f'| Current {name} ({item["status"]}) | {human(item["wall_time_s"])} |')
    lines += ['', 'Preflight unit/probe time is not model training or production evaluation time. '
        'Concurrent model times are not added to obtain sweep elapsed. The launcher records whole-child '
        'wall time including final serialization; raw model totals exclude final result serialization. '
        'Failed, interrupted and unverified jobs are excluded from completed aggregates.', '',
        f'Concurrency: {WORKERS} joint subprocess slots, one numerical thread per process. '
        f'Environment: {json.dumps(summary.get("resources"), sort_keys=True)}', '',
        'Raw journals certify recorded strict OPTIMAL/physical acceptance and center mapping, '
        'not independent cost recomputation: original evaluators do not persist scenario costs or QP solutions.', '',
        'Publication status and final report/final-verification wall times are recorded in the sealed '
        'pipeline summary one directory above this report. This report is constructed before atomic publication.']
    with (output / 'RESULTS.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


def preflight(output, summary):
    with phase(summary['phases'], 'syntax') as entry:
        paths = [p for p in local_sources() if p.suffix == '.py']
        entry['files'] = [str(p) for p in paths]
        for path in paths:
            compile(path.read_text(), str(path), 'exec')
    before = source_hashes()
    with phase(summary['phases'], 'unit_tests'):
        summary['unit_tests'] = run_units(output)
        validate_units(summary['unit_tests'])
    with phase(summary['phases'], 'collection'):
        collection = backend().collect_policies()
        cohort(collection['jobs'])
        require(collection['test_bytes_accessed'] is False and collection['test_evaluated'] is False,
                'Collection accessed test data')
        backend().merge(collection['source_sha256'], before)
        require(before == source_hashes(), 'Sources changed during preflight')
        # Historical timing is input evidence, never used for choosing a policy.
        backend().merge(collection['input_sha256'], {str(HISTORICAL_RUNTIME): sha(HISTORICAL_RUNTIME)})
        compact_bases(collection)
        summary.update(job_count=len(collection['jobs']), source_sha256=collection['source_sha256'])
    with phase(summary['phases'], 'resources'):
        snapshot = resource_snapshot(collection['source_sha256'])
        summary.update(resources=snapshot, resource_policy=resource_gate(snapshot,
            [resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024]))
        bindings = {**collection['input_sha256'], **collection['source_sha256']}
        source_bytes = sum(Path(path).stat().st_size for path in bindings)
        summary['disk'] = disk_gate(output, source_bytes + 16*GIB)
    with phase(summary['phases'], 'source_copy'):
        files = backend().materialize(collection, output / 'frozen_inputs')
    manifest = {**collection, 'status': 'frozen', 'workers': WORKERS, 'files': files,
                'frozen_at_utc': utc(), 'n_test': 10000, 'test_seed': 9999}
    path = output / 'policies_manifest.json'
    with phase(summary['phases'], 'policy_seal'):
        verify_files(manifest)
        summary['policies_manifest_sha256'] = seal(path, manifest)
        summary['policies_manifest'] = str(path)
    with phase(summary['phases'], 'training_probes'):
        summary['processes'] = execute(smoke_jobs(manifest['jobs']), path, output, True)
    with phase(summary['phases'], 'journal_verification'):
        rows, _ = verify_all_results(smoke_jobs(manifest['jobs']), summary['processes'], path, output, True)
        summary.update(solves_attempted=sum(r['solves_attempted'] for r in rows),
                       solves_accepted=sum(r['solves_accepted'] for r in rows), status_counts={'2': 12})
    with phase(summary['phases'], 'final_verification'):
        require(sealed(path, summary['policies_manifest_sha256']) == manifest, 'Policy manifest changed')
        verify_files(manifest)
    summary['status'] = 'completed'


def production(output, summary, preflight_path, preflight_sha):
    with phase(summary['phases'], 'preflight_validation'):
        pref, policy = load_preflight(preflight_path, preflight_sha)
        policy_path = Path(pref['policies_manifest'])
        summary.update(preflight=str(preflight_path.resolve()), preflight_sha256=preflight_sha,
                       policies_manifest=str(policy_path), policies_manifest_sha256=pref['policies_manifest_sha256'])
    with phase(summary['phases'], 'resources'):
        snapshot = resource_snapshot(policy['source_sha256'])
        summary.update(resources=snapshot, resource_policy=resource_gate(snapshot,
            [p['envelope']['peak_rss_bytes'] for p in pref['processes']], policy_path.stat().st_size))
        solves = 570*10000 + sum(sum(j['sizes']) for j in policy['jobs'])
        summary['disk'] = disk_gate(output, 4096*solves + 32*policy_path.stat().st_size + 16*GIB)
    with phase(summary['phases'], 'test_and_base_binding'):
        manifest = bind_after_seal(policy, policy_path, pref['policies_manifest_sha256'], output)
        summary['test_bytes_accessed'] = True
    manifest_path = output / 'manifest.json'
    with phase(summary['phases'], 'final_manifest_seal'):
        digest = seal(manifest_path, manifest)
        summary.update(manifest=str(manifest_path), manifest_sha256=digest, job_count=570)
    # No collect_policies/materialize(training) call here: preflight copies are retained verbatim.
    del policy
    with phase(summary['phases'], 'neural_cohorts'):
        summary['processes'] = execute(manifest['jobs'], manifest_path, output)
    with phase(summary['phases'], 'journal_verification'):
        rows, references = verify_all_results(manifest['jobs'], summary['processes'], manifest_path, output)
        for row in rows:
            require(row['test'] == manifest['tests'][str(row['R'])], 'Worker common-test binding mismatch')
        summary['solves_accepted'] = sum(r['solves_accepted'] for r in rows)
        summary['solves_attempted'] = sum(r['solves_attempted'] for r in rows)
    stage = output / 'publication_staging'
    stage.mkdir(exist_ok=False)
    with phase(summary['phases'], 'render'):
        comparisons, artifacts = render_both(manifest, rows, references, stage)
    with phase(summary['phases'], 'report'):
        summary['status'] = 'verified_pending_publication'
        report(stage, manifest, summary, rows, comparisons)
        artifacts.update({path.name: sha(path) for path in stage.iterdir() if path.is_file()})
    with phase(summary['phases'], 'final_verification'):
        require(sha(preflight_path) == preflight_sha and
                sha(policy_path) == pref['policies_manifest_sha256'], 'Preflight binding changed')
        require(policy_path.with_suffix('.sha256').read_text().strip() == pref['policies_manifest_sha256'] and
            preflight_path.with_suffix('.sha256').read_text().strip() == preflight_sha,
            'Preflight seal sidecar changed')
        require(sealed(manifest_path, digest) == manifest, 'Final manifest changed')
        verify_files(manifest)
        result_bindings_unchanged(references)
        for name, digest in artifacts.items():
            require(sha(stage / name) == digest, 'Staged artifact changed')
        require(not (output / 'published').exists(), 'Publication exists')
    # A single directory rename publishes BOTH R figures only after the global barrier.
    stage.rename(output / 'published')
    summary.update(status='completed', published=str(output / 'published'), artifacts=artifacts,
                   completed_runtime_aggregates=runtime_groups(rows))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('preflight', 'production', '_worker', '_unit'))
    p.add_argument('output', type=Path)
    p.add_argument('--preflight', type=Path)
    p.add_argument('--sha')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--job')
    p.add_argument('--smoke', action='store_true')
    return p


def invocation_report(output, summary):
    """A runtime table even for syntax failures, interrupted sweeps and preflights."""
    lines = ['# Invocation runtime (not historical training)', '',
        f'Status: {summary["status"]}; mode: {summary["mode"]}.', '',
        '| Category | Measured time / status |', '|---|---|',
        '| Nominal neural training | Not run; persisted history separate; missing fields unavailable, rerun required |',
        '| Robust neural/CCG AP / MP / total | Not run; persisted history separate |',
        '| SRO-FA partition / optimization / evaluation / total | Not run; original references unchanged |',
        '| Hyperparameter selection / CV / benchmark construction | Not run |',
        '| Final neural evaluation | Production raw per-model results only; preflight probes are not model training |']
    for name, item in summary['phases'].items():
        if 'wall_time_s' in item:
            lines.append(f'| {name} ({item["status"]}) | {human(item["wall_time_s"])} |')
    lines += ['', '| Policy | Process status | Process wall time | Backend model wall time |',
              '|---|---|---|---|']
    for process in summary['processes']:
        value = process.get('envelope', {}).get('model_wall_time_s')
        model = 'Unavailable: process failed before returning a measured model record' if value is None else human(value)
        lines.append(f'| {process["id"]} | {process["status"]} | {human(process["wall_time_s"])} | {model} |')
    lines += ['', 'These per-process statuses do not replace the global journal/publication barrier. '
        'No failed or unverified result is a certified completed-model aggregate. '
        'Raw worker records retain separate OOS/IS phases and iterations. '
        'Final elapsed time and this reporting phase are in the sealed summary; '
        'whole-child time including serialization is in the launcher runtime artifact. '
        'Concurrent wall times are not summed. Missing unreturned timings are unavailable, not zero.', '',
        f'Workers: {WORKERS}; numerical threads per worker: 1. '
        f'Environment: {json.dumps(summary.get("resources"), sort_keys=True)}']
    with (output / 'RUNTIME.md').open('x') as stream:
        stream.write('\n'.join(lines)+'\n')


def main(argv=None):
    a = parser().parse_args(argv)
    os.environ.update(THREAD_ENV)
    sys.dont_write_bytecode = True
    output = a.output.resolve()
    if a.mode == '_unit':
        return _unit(output)
    if a.mode == '_worker':
        require(a.manifest is not None and a.job, 'Worker requires --manifest and --job')
        return _worker(a.manifest.resolve(), a.job, output, a.smoke)
    require(a.mode != 'production' or a.preflight is not None and a.sha, 'Production requires --preflight and --sha')
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    summary = dict(experiment='black_fixed_policy_comparison_20260926', mode=a.mode, status='running',
        start_utc=utc(), output=str(output), phases={}, processes=[], workers=WORKERS, threads=1,
        timing_clock='time.perf_counter', timing_unit='seconds', test_bytes_accessed=False,
        new_training=False, new_cv=False, command=sys.argv, cwd=str(Path.cwd()))
    code = 1
    try:
        if a.mode == 'preflight':
            preflight(output, summary)
        else:
            production(output, summary, a.preflight.resolve(), a.sha)
        code = 0
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc())
    finally:
        try:
            with phase(summary['phases'], 'invocation_report'):
                invocation_report(output, summary)
        except BaseException:
            # Preserve failure evidence rather than replacing it with a reporting exception.
            summary.update(status='failed', report_error=traceback.format_exc())
            code = 1
        summary.update(end_utc=utc(), wall_time_s=time.perf_counter()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            timing_scope='current invocation through publication; final summary serialization excluded; launcher includes it')
        summary['wall_time_human'] = human(summary['wall_time_s'])
        # Failed runs still preserve per-process/model times and incomplete outcomes.
        write(output / 'invocation_runtime.json', dict(status=summary['status'], mode=a.mode,
            phases=summary['phases'], processes=summary['processes'], workers=WORKERS, threads=1,
            start_utc=summary['start_utc'], end_utc=summary['end_utc'], wall_time_s=summary['wall_time_s'],
            wall_time_human=summary['wall_time_human'], timing_clock='time.perf_counter', timing_unit='seconds',
            completed_runtime_aggregates=summary.get('completed_runtime_aggregates', []),
            note='No aggregate certifies failed/unverified models; raw process envelopes retain attempted runtimes. '
                 'Final serialization excluded; launcher records whole child.'))
        # Always leave timing/status evidence, including syntax/unit/collection failures.
        digest = seal(output / 'summary.json', summary)
        print(json.dumps(dict(status=summary['status'], summary=str(output / 'summary.json'),
                              sha256=digest, wall_time_s=summary['wall_time_s'])), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())