"""Sealed STOP replay: preflight first, then an explicit 1,110-policy production.

No historical evaluator imports, optimization of policies, test generation, cache
reuse or automatic production launch. Import configures threads before numerics.
"""
from __future__ import annotations

import os
import sys

THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREAD_ENV)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import multiprocessing
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / 'archive'
WORKERS, GIB = 16, 1024**3
COMPARE = ARCHIVE / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison/compare.py'
REFERENCE = ARCHIVE / 'retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/comparison_20260925/reference.py'
COMPARE_SHA = '8285365c2913687a6c8769fee720b56237c785cb9f6d37d81b3a82f8c9d1b366'
REFERENCE_SHA = '3762749519a43e9a6d42e0bb569f04253d8ea8b8da3bcd64da9f3d7dd592105f'
RESOURCE_DEFS = (
    (ARCHIVE / 'retailers_moreau_training_selected/green_R3_N200_N800_20260923/preflight.py', ('environment',)),
    (ARCHIVE / 'retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924/r10_resources.py', ('_cpus', 'resources')))
PROBES = tuple((m, r, n, 0, .4) for r in (3, 10) for n in (50, 200, 800)
               for m in ('green', 'black'))
PREFLIGHT_PHASES = ('unit_tests', 'selection', 'environment', 'resources',
                    'training_parity', 'final_verification')
COUNTS = dict(green=540, black=570, black_converged=559, black_capped=11)
_NUMERICAL = None
_SELECTION = None
_CONTEXT = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write(path, value):
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


def sealed(path, digest=None):
    path = Path(path)
    actual = sha(path)
    require(digest is None or actual == digest, f'Explicit SHA256 mismatch: {path}')
    require(path.with_suffix('.sha256').read_text().strip() == actual, f'Seal mismatch: {path}')
    value = read(path)
    require(sha(path) == actual, f'Changed while reading: {path}')
    return value


def bound_json(path, digest):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, f'Input changed: {path}')
    return json.loads(payload)


def seconds(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'Invalid nonnegative timing/count')
    return value


def human(value):
    return f'{value:.6f} s / {value/60:.6f} min / {value/3600:.6f} h'


def timing(row):
    seconds(row['wall_time_s'])
    require(row['timing_clock'] == 'time.perf_counter' and row['timing_unit'] == 'seconds',
            'Wrong timing clock/unit')
    for key in ('start_utc', 'end_utc'):
        require(datetime.fromisoformat(row[key]).utcoffset() is not None, 'Timezone required')


@contextmanager
def phase(phases, name):
    start = time.perf_counter()
    row = phases[name] = dict(status='running', start_utc=utc(),
                             timing_clock='time.perf_counter', timing_unit='seconds')
    try:
        yield row
    except BaseException as exc:
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    else:
        row['status'] = 'completed'
    finally:
        row.update(end_utc=utc(), wall_time_s=time.perf_counter() - start)


def selection():
    global _SELECTION
    if _SELECTION is None:
        spec = importlib.util.spec_from_file_location('_stop_selection', HERE / 'selection.py')
        require(spec is not None and spec.loader is not None, 'Selection import unavailable')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SELECTION = module
    return _SELECTION


def numerical():
    global _NUMERICAL
    if _NUMERICAL is None:
        sys.path.insert(0, str(ROOT))
        evaluator = importlib.import_module('src.evaluate')
        require(Path(evaluator.__file__).resolve() == ROOT / 'src/evaluate.py', 'Wrong evaluator')
        require(Path(importlib.import_module('src.artifacts').__file__).resolve() ==
                ROOT / 'src/artifacts.py', 'Wrong artifacts module')
        evaluator.torch.set_num_threads(1)
        evaluator.torch.set_num_interop_threads(1)
        _NUMERICAL = evaluator
    require(_NUMERICAL.torch.get_num_threads() == _NUMERICAL.torch.get_num_interop_threads() == 1
            and all(os.environ[k] == '1' for k in THREAD_ENV), 'Thread contract changed')
    return _NUMERICAL


def local_sources():
    paths = set(HERE.glob('*.py')) | {HERE / 'README.md'}
    paths.update(ROOT / p for p in ('src/evaluate.py', 'src/artifacts.py', 'src/__init__.py',
                                   'archive_paths.py', 'archive_paths.json'))
    paths.update((COMPARE, REFERENCE))
    paths.update(p for p, _ in RESOURCE_DEFS)
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def merge_sources(*maps):
    result = {}
    for mapping in maps:
        for path, digest in mapping.items():
            require(path not in result or result[path] == digest, f'Conflicting source: {path}')
            result[path] = digest
    return result


def verify_sources(sources):
    current = local_sources()
    # Detect additions/removals as well as changed existing files.
    require({p for p in sources if Path(p).parent == HERE} ==
            {p for p in current if Path(p).parent == HERE}, 'Local source set changed')
    require(all(sources.get(p) == h for p, h in current.items()), 'Local/shared source changed')
    for path, digest in sources.items():
        require(sha(path) == digest, f'Source changed: {path}')


def verify_inputs(manifest):
    for path, digest in manifest['input_sha256'].items():
        require(sha(path) == digest, f'Selected input changed: {path}')


def cohort(manifest):
    selection()._cohort(manifest['jobs'], selected=True)
    require(manifest['counts'] == COUNTS and manifest['status'] == 'frozen'
            and manifest['return_rule'] == manifest['selection_rule'] == 'stopping_checkpoint'
            and manifest['policies_frozen_before_test'] is True
            and manifest['test_evaluated'] is False and manifest['new_training'] is False,
            'Wrong selection protocol/counts')
    for job in manifest['jobs']:
        require(Path(job['id']).name == job['id'] and job['id'] not in ('.', '..'), 'Unsafe job ID')


def selection_bundle(path, digest, receipt_path, receipt_sha):
    manifest = sealed(path, digest)
    receipt = sealed(receipt_path, receipt_sha)
    cohort(manifest)
    require(Path(manifest['verification_receipt']).resolve() == Path(receipt_path).resolve()
            and receipt['selection_manifest_sha256'] == digest
            and receipt['status'] == 'completed' and receipt['all_old_bindings_verified'] is True
            and receipt['test_array_hashes_checked_after_selection_seal'] is True
            and receipt['test_arrays_loaded'] is False
            and receipt['verified_file_count'] >= len(manifest['input_sha256']),
            'Selection collection did not complete')
    timing(receipt)
    return manifest


def runtime_environment(manifest):
    e = numerical()
    import scipy
    import gurobipy as gp
    versions = dict(python=sys.version, numpy=e.np.__version__, scipy=scipy.__version__,
                    torch=e.torch.__version__, gurobi=list(gp.gurobi.version()))
    old = manifest['source_old_manifest']
    evidence = sealed(old['path'], old['sha256'])
    require(all(evidence['base_comparisons'][str(r)]['versions'] == versions for r in (3, 10)),
            'Python/NumPy/SciPy/Torch/Gurobi differs from historical native evaluator')
    return dict(versions=versions, executable=sys.executable, platform=platform.platform(),
                hostname=platform.node(), machine=platform.machine(), workers=WORKERS,
                threads=1, torch_threads=e.torch.get_num_threads(),
                torch_interop_threads=e.torch.get_num_interop_threads(), thread_environment=THREAD_ENV)


def resource_snapshot(sources):
    namespace = dict(Path=Path, os=os, sys=sys, platform=platform, utc=utc,
        THREAD_ENV=THREAD_ENV, peak_rss_bytes=lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    for path, names in RESOURCE_DEFS:
        require(sha(path) == sources[str(path)], 'Resource auditor changed')
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        require(len(nodes) == len(names), 'Resource auditor definition coverage')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    snapshot = namespace['resources']()
    return snapshot


def resource_gate(snapshot, peak_rss_bytes=0):
    effective = len(snapshot['affinity'])
    if snapshot['cpu_quota_cores'] is not None:
        effective = min(effective, math.floor(snapshot['cpu_quota_cores']))
    load = math.ceil(max(snapshot['load_average']))
    require(effective - 8 - load >= WORKERS and
            snapshot['memory_available_bytes'] - 50*GIB >= WORKERS*4*GIB and
            seconds(peak_rss_bytes) <= 4*GIB,
            'Insufficient resources for 16 single-thread workers; no automatic degradation')
    return dict(workers=WORKERS, threads=1, reserved_cpu_slots=8,
        background_load_slots=load, effective_cpu_slots=effective,
        reserved_memory_bytes=50*GIB, per_worker_memory_bytes=4*GIB,
        measured_preflight_peak_rss_bytes=peak_rss_bytes,
        scope='cgroup/affinity-aware admission snapshot, not a future memory guarantee')


def pure_backend(manifest):
    path = selection()._paths.resolve_archive_path('comparison/black_green_evaluation/backend.py', root=ROOT)
    require(sha(path) == manifest['source_sha256'][str(path)], 'Unbound pool backend')
    return selection()._load(path, '_stop_pure_pool_backend')


def own_inputs(job, backend, e):
    for field, hash_field in (('checkpoint', 'checkpoint_sha256'),
            ('training_data_path', 'training_data_sha256'), ('pool_path', 'pool_file_sha256')):
        require(sha(job[field]) == job[hash_field], f'Changed own input: {field}')
    model = e.load_model(job['checkpoint'], job['R'])
    require(e.state_hash(model) == job['state_sha256'], 'Learned state hash differs')
    training = bound_json(job['training_data_path'], job['training_data_sha256'])
    centers = e.np.asarray(training['centers'], dtype=e.np.float64)
    require(centers.shape == (job['Ntrain'], 2*job['R']) and e.np.isfinite(centers).all()
            and (centers >= 0).all() and training['original_sample_ids'] == job['original_sample_ids']
            and training['data_sha256'] == job['data_sha256']
            and hashlib.sha256(e.np.asarray(centers, dtype='<f8').tobytes(order='C')).hexdigest() ==
                job['training_centers_sha256'], 'Training centers/IDs differ')
    saved = bound_json(job['pool_path'], job['pool_file_sha256'])
    require(saved['original_sample_ids'] == job['original_sample_ids'], 'Pool IDs differ')
    pools = backend.validate_pools(saved['pools'], centers, job['radius'])
    require(backend.pool_digest(pools) == saved['pool_sha256'] == job['pool_sha256']
            and [len(p) for p in pools] == job['sizes'], 'Own-pool digest/sizes differ')
    return model, centers, pools


def reference(r):
    """Only five numerical definitions, with literal historical R10 substitutions."""
    require(r in (3, 10) and sha(COMPARE) == COMPARE_SHA and sha(REFERENCE) == REFERENCE_SHA,
            'Historical numerical reference lock mismatch')
    names = ('Model', 'load_model', 'state_hash', 'scenario_costs', 'strict_project')
    text = COMPARE.read_text()
    lines = text.splitlines(keepends=True)
    nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in names]
    require(len(nodes) == len(names) and {n.name for n in nodes} == set(names), 'Reference coverage')
    fragment = '\n\n'.join(''.join(lines[min([n.lineno] +
        [d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
    changes = []
    if r == 10:
        calls = [n.value for n in ast.parse(REFERENCE.read_text()).body
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name) and n.value.func.id == 'load_definitions']
        require(len(calls) == 1, 'R10 adaptation coverage')
        historical = ast.literal_eval(calls[0].args[3])
        # gen is deliberately NOT executed; its sole dimension replacement is omitted.
        changes = [c for c in historical if c[0] != 'size=(n, 3)']
        require(len(changes) == 8 and len(historical) == 9, 'R10 adaptation count')
        for before, after, count in changes:
            require(fragment.count(before) == count, f'Exact fragment count mismatch: {before}')
            fragment = fragment.replace(before, after)
    e = numerical()
    import gurobipy as gp
    namespace = dict(np=e.np, torch=e.torch, nn=e.nn, gp=gp, GRB=gp.GRB,
                     PHYSICAL_TOL=1e-4, time=time, hashlib=hashlib)
    exec(compile(fragment, str(COMPARE), 'exec'), namespace)
    return namespace, changes


def probe(job, backend, output):
    e = numerical()
    start = time.perf_counter()
    row = dict(id=job['id'], method=job['method'], R=job['R'], Ntotal=job['Ntotal'],
        Ntrain=job['Ntrain'], radius=job['radius'], seed=job['seed'],
        checkpoint_sha256=job['checkpoint_sha256'], state_sha256=job['state_sha256'],
        selected_candidate_id=job['selected_candidate_id'], start_utc=utc(), status='running',
        timing_clock='time.perf_counter', timing_unit='seconds', workers=1, threads=1,
        test_evaluated=False, training_only=True, phases={}, rows=[],
        compare_sha256=COMPARE_SHA, reference_sha256=REFERENCE_SHA)
    try:
        with phase(row['phases'], 'input_loading'):
            model, centers, _ = own_inputs(job, backend, e)
            old, changes = reference(job['R'])
            original = old['load_model'](job['checkpoint'], 'new')
            require(old['state_hash'](original) == e.state_hash(model) == job['state_sha256'],
                    'Reference learned state differs')
            row['adaptations'] = [list(change) for change in changes]
        with phase(row['phases'], 'full_center_targets'):
            x1, x2 = centers[:, :job['R']], centers[:, job['R']:]
            z, q10, q1r = e.targets(model, x1)
            oz, oq10, oq1r = original.targets(x1)
            require(z.dtype == oz.dtype == e.np.float32 and q1r.dtype == oq1r.dtype == e.np.float32
                    and e.np.array_equal(z, oz) and q10 == oq10 and e.np.array_equal(q1r, oq1r),
                    'Full training-batch targets/inventories fail exact parity')
            row.update(full_batch_size=len(x1), targets_exact=True, inventories_exact=True,
                       targets_sha256=hashlib.sha256(z.tobytes(order='C')).hexdigest(),
                       q10=q10, q1r=q1r.tolist())
        with phase(row['phases'], 'strict_training_rows'):
            import gurobipy as gp
            with gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.setParam('Threads', 1)
                env.start()
                for i in range(2):
                    item = dict(row=i, active={}, reference={})
                    row['rows'].append(item)
                    q = e.strict_project(z[i], q10, env, item['active'])
                    oq = old['strict_project'](oz[i], oq10, env, item['reference'])
                    require(e.np.array_equal(q, oq), 'Strict QP shipment parity failed')
                    cost = e.scenario_costs(q10, q1r, q[None], x1[i:i+1], x2[i:i+1])
                    ocost = old['scenario_costs'](oq10, oq1r, oq[None], x1[i:i+1], x2[i:i+1])
                    require(cost.dtype == ocost.dtype == e.np.float64
                            and e.np.isfinite(cost).all() and e.np.array_equal(cost, ocost),
                            'Scenario cost parity failed')
                    for diagnostic in (item['active'], item['reference']):
                        validate_diagnostic(diagnostic)
                    item.update(shipments_exact=True, cost_exact=True, cost=float(cost[0]),
                                shipments=q.tolist())
            require(e.state_hash(model) == old['state_hash'](original) == job['state_sha256'],
                    'Parity evaluation changed learned state')
        row['status'] = 'completed'
    except BaseException:
        row.update(status='failed', error=traceback.format_exc())
    finally:
        row.update(end_utc=utc(), wall_time_s=time.perf_counter()-start,
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        row['wall_time_human'] = human(row['wall_time_s'])
    path = output / (job['id'] + '.json')
    digest = seal(path, row)
    return dict(path=str(path), sha256=digest, **row)


def validate_diagnostic(diagnostic):
    require(type(diagnostic['status']) is int and diagnostic['status'] == 2
            and diagnostic['accepted'] is True, 'Non-OPTIMAL/unaccepted raw solve')
    require(seconds(diagnostic['physical_residual']) <= 1e-4, 'Raw physical residual exceeds gate')
    for field in ('wall_time_s', 'iterations', 'barrier_iterations'):
        seconds(diagnostic[field])


def validate_probes(probes, jobs):
    indexed = {(j['method'], j['R'], j['Ntotal'], j['seed'], j['radius']): j for j in jobs}
    require(len(probes) == len(PROBES), 'Missing parity probes')
    for binding, key in zip(probes, PROBES):
        row = sealed(binding['path'], binding['sha256'])
        require(row == {k: v for k, v in binding.items() if k not in ('path', 'sha256')}, 'Probe binding differs')
        job = indexed[key]
        require(all(row[k] == job[k] for k in ('id', 'method', 'R', 'Ntotal', 'Ntrain', 'seed',
                    'radius', 'checkpoint_sha256', 'state_sha256', 'selected_candidate_id'))
                and row['status'] == 'completed' and row['training_only'] is True
                and row['test_evaluated'] is False and row['targets_exact'] is True
                and row['inventories_exact'] is True and row['full_batch_size'] == job['Ntrain']
                and row['compare_sha256'] == COMPARE_SHA and row['reference_sha256'] == REFERENCE_SHA
                and row['workers'] == row['threads'] == 1 and len(row['rows']) == 2
                and set(row['phases']) == {'input_loading', 'full_center_targets', 'strict_training_rows'},
                'Incomplete/failed real parity probe')
        timing(row)
        for i, item in enumerate(row['rows']):
            require(item['row'] == i and item['shipments_exact'] is True and item['cost_exact'] is True,
                    'Training projection parity incomplete')
            validate_diagnostic(item['active'])
            validate_diagnostic(item['reference'])
        for p in row['phases'].values():
            require(p['status'] == 'completed', 'Failed probe phase')
            timing(p)


def unit_mode(output):
    start = time.perf_counter()
    row = dict(start_utc=utc(), timing_clock='time.perf_counter', timing_unit='seconds')
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern='test*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    ok = result.wasSuccessful() and result.testsRun >= 39 and not result.skipped and not result.expectedFailures
    row.update(status='completed' if ok else 'failed', tests_run=result.testsRun,
               failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
               expected_failures=len(result.expectedFailures), unexpected_successes=len(result.unexpectedSuccesses),
               end_utc=utc(), wall_time_s=time.perf_counter()-start)
    seal(output, row)
    return 0 if ok else 1


def run_units(output):
    evidence = output / 'unit_evidence.json'
    command = [sys.executable, '-B', '-u', str(HERE / 'run.py'), '_unit', str(evidence)]
    log = output / 'unit_tests.log'
    with log.open('x') as stream:
        child = subprocess.run(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, **THREAD_ENV, 'PYTHONDONTWRITEBYTECODE': '1'}, check=False)
    require(child.returncode == 0, f'Local unit tests failed: {log}')
    row = sealed(evidence)
    require(row['status'] == 'completed', 'Incomplete unit evidence')
    return dict(path=str(evidence), sha256=sha(evidence), command=command, returncode=child.returncode,
                log=str(log), log_sha256=sha(log), **row)


def fresh_directory(output):
    output = Path(output).resolve()
    require(not output.is_relative_to(ARCHIVE), 'Cannot write inside archive')
    output.mkdir(parents=True, exist_ok=False)
    return output


def preflight(output):
    output = fresh_directory(output)
    start = time.perf_counter()
    row = dict(mode='preflight', experiment='stopping_checkpoint_20260928', status='running',
        start_utc=utc(), timing_clock='time.perf_counter', timing_unit='seconds',
        workers=1, production_workers=WORKERS, threads=1, phases={}, probes=[], test_evaluated=False)
    try:
        sources = local_sources()
        with phase(row['phases'], 'unit_tests'):
            row['unit_tests'] = run_units(output)
        with phase(row['phases'], 'selection'):
            manifest = selection().collect(output)
            row.update(selection_path=str(output / 'selection_manifest.json'),
                selection_sha256=sha(output / 'selection_manifest.json'),
                selection_verification_path=str(output / 'selection_verification.json'),
                selection_verification_sha256=sha(output / 'selection_verification.json'), counts=manifest['counts'])
            selection_bundle(row['selection_path'], row['selection_sha256'],
                             row['selection_verification_path'], row['selection_verification_sha256'])
            sources = merge_sources(sources, manifest['source_sha256'])
            row['source_sha256'] = sources
            verify_sources(sources)
        with phase(row['phases'], 'environment'):
            row['environment'] = runtime_environment(manifest)
        with phase(row['phases'], 'resources'):
            row['resource_snapshot'] = resource_snapshot(sources)
            row['resource_gate'] = resource_gate(row['resource_snapshot'])
        with phase(row['phases'], 'training_parity'):
            folder = output / 'probes'
            folder.mkdir()
            backend = pure_backend(manifest)
            indexed = {(j['method'], j['R'], j['Ntotal'], j['seed'], j['radius']): j for j in manifest['jobs']}
            # Retain every failed probe, and attempt all prescribed training-only probes.
            row['probes'] = [probe(indexed[key], backend, folder) for key in PROBES]
            validate_probes(row['probes'], manifest['jobs'])
        with phase(row['phases'], 'final_verification'):
            verify_sources(sources)
            row['resource_gate'] = resource_gate(resource_snapshot(sources),
                max(p['peak_rss_bytes'] for p in row['probes']))
            selection_bundle(row['selection_path'], row['selection_sha256'],
                             row['selection_verification_path'], row['selection_verification_sha256'])
        row['status'] = 'completed'
    except BaseException:
        row.update(status='failed', error=traceback.format_exc())
    row.update(end_utc=utc(), wall_time_s=time.perf_counter()-start)
    row['wall_time_human'] = human(row['wall_time_s'])
    row['timing_scope'] = 'preflight invocation excluding imports/final summary serialization; launch child includes both'
    digest = seal(output / 'summary.json', row)
    print(json.dumps(dict(status=row['status'], summary=str(output / 'summary.json'), sha256=digest)), flush=True)
    return 0 if row['status'] == 'completed' else 1


def validate_preflight(path, digest):
    require(isinstance(digest, str) and len(digest) == 64, 'Explicit preflight SHA256 required')
    row = sealed(path, digest)
    require(row['status'] == 'completed' and row['mode'] == 'preflight' and row['counts'] == COUNTS
            and row['test_evaluated'] is False and row['production_workers'] == WORKERS
            and row['threads'] == 1 and set(row['phases']) == set(PREFLIGHT_PHASES), 'Preflight not passed')
    timing(row)
    for p in row['phases'].values():
        require(p['status'] == 'completed', 'Failed preflight phase')
        timing(p)
    units = row['unit_tests']
    evidence = sealed(units['path'], units['sha256'])
    require(evidence['status'] == 'completed' and evidence['tests_run'] >= 39
            and all(evidence[k] == 0 for k in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes'))
            and units['returncode'] == 0 and sha(units['log']) == units['log_sha256'], 'Unverified unit tests')
    manifest = selection_bundle(row['selection_path'], row['selection_sha256'],
                               row['selection_verification_path'], row['selection_verification_sha256'])
    require(merge_sources(row['source_sha256'], manifest['source_sha256']) == row['source_sha256'],
            'Missing selection source bindings')
    verify_sources(row['source_sha256'])
    validate_probes(row['probes'], manifest['jobs'])
    require(runtime_environment(manifest) == row['environment'], 'Preflight execution environment changed')
    return row, manifest


def test_arrays(manifest, r):
    e = numerical()
    test = manifest['tests'][str(r)]
    require(test['seed'] == 9999 and test['n_test'] == 10000, 'Wrong historical test protocol')
    arrays = []
    for name in ('xi1', 'xi2'):
        info = test['arrays'][name]
        require(sha(info['file']) == info['file_sha256'], 'Test file bytes changed')
        a = e.np.load(info['file'], allow_pickle=False)
        require(a.shape == (10000, r) and list(a.shape) == info['shape']
                and a.dtype == e.np.float64 and str(a.dtype) == info['dtype']
                and e.np.isfinite(a).all() and (a >= 0).all()
                and hashlib.sha256(a.tobytes(order='C')).hexdigest() == info['array_sha256']
                and sha(info['file']) == info['file_sha256'], 'Test array content changed')
        a.flags.writeable = False
        arrays.append(a)
    return tuple(arrays)


def initialize_worker(execution_path, execution_sha):
    global _CONTEXT
    execution = sealed(execution_path, execution_sha)
    verify_sources(execution['source_sha256'])
    manifest = selection_bundle(execution['selection_path'], execution['selection_sha256'],
        execution['selection_verification_path'], execution['selection_verification_sha256'])
    e = numerical()
    require(execution['workers'] == WORKERS and execution['threads'] == 1 and
            runtime_environment(manifest) == execution['environment'], 'Worker runtime contract changed')
    arrays = {r: test_arrays(manifest, r) for r in (3, 10)}
    _CONTEXT = (execution, manifest, e, pure_backend(manifest), arrays)


def verify_replay(folder, model, x1, x2, sizes=None):
    """Read EVERY solve and array; recompute costs, targets and physical residuals."""
    e = numerical()
    folder = Path(folder)
    files = {name: sha(folder / name) for name in
             ('result.json', 'targets.npy', 'shipments.npy', 'costs.npy', 'solves.jsonl')}
    raw = bound_json(folder / 'result.json', files['result.json'])
    n, r = x1.shape
    require(raw['status'] == 'completed' and raw['cached'] is False and raw['sample_count'] == n
            and raw['R'] == r and raw['state_sha256'] == e.state_hash(model)
            and raw['solver_threads'] == raw['torch_threads'] == 1 and raw['worker_count'] == WORKERS
            and raw['physical_tolerance'] == 1e-4 and raw['solver_settings']['Threads'] == 1
            and raw['solves_attempted'] == raw['solves_accepted'] == raw['solves_physical_accepted'] == n
            and raw['status_counts'] == {'2': n}, 'Incomplete raw replay counts/status')
    require(raw['timing_clock'] == 'time.perf_counter' and raw['timing_unit'] == 'seconds', 'Raw clock mismatch')
    seconds(raw['wall_time_s'])
    for name in ('started_at_utc', 'ended_at_utc'):
        require(datetime.fromisoformat(raw[name]).utcoffset() is not None, 'Raw timestamps missing')
    require(set(raw['phases']) == {'input_validation', 'target_evaluation', 'projection_qp', 'cost_evaluation'},
            'Incomplete raw phases')
    for value in raw['phases'].values():
        seconds(value)
    z, q10, q1r = e.targets(model, x1)
    targets, shipments, costs = (e.np.load(folder / name, allow_pickle=False)
                              for name in ('targets.npy', 'shipments.npy', 'costs.npy'))
    require(targets.shape == shipments.shape == (n, r) and costs.shape == (n,)
            and targets.dtype == e.np.float32 and shipments.dtype == costs.dtype == e.np.float64
            and all(e.np.isfinite(a).all() for a in (targets, shipments, costs))
            and e.np.array_equal(targets, z) and raw['q10'] == q10 and raw['q1r'] == q1r.tolist()
            and raw['inventory_dtype'] == 'float32', 'Replay arrays/targets/inventories differ')
    attempted = 0
    iterations, barrier = 0., 0
    with (folder / 'solves.jsonl').open() as stream:
        for i, line in enumerate(stream):
            d = json.loads(line)
            require(i < n and type(d['row']) is int and d['row'] == i, 'Wrong solve ordinal/count')
            validate_diagnostic(d)
            seconds(d['solver_runtime_s'])
            require(type(d['objective']) in (int, float) and math.isfinite(d['objective']), 'Invalid objective')
            require(e.np.array_equal(shipments[i], d['shipments']), 'Journal shipments differ')
            residual = float(max(0., -shipments[i].min(), shipments[i].sum()-q10))
            require(residual == d['physical_residual'] and residual <= 1e-4, 'Physical residual differs')
            iterations += d['iterations']
            barrier += d['barrier_iterations']
            attempted += 1
    require(attempted == n and iterations == raw['iterations'] and barrier == raw['barrier_iterations'],
            'Journal count/iteration mismatch')
    require(e.np.array_equal(costs, e.scenario_costs(q10, q1r, shipments, x1, x2)), 'Recomputed costs differ')
    if sizes is None:
        require(raw['mean'] == raw['mean_oos'] == float(costs.mean()) and raw['role'] == 'oos', 'OOS mean differs')
    else:
        mean, maxima = e.finite_pool_cost(costs, sizes)
        require(raw['sizes'] == sizes and raw['mean'] == raw['finite_pool_mean'] == mean
                and raw['per_center_maxima'] == maxima and raw['role'] == 'own_pool', 'Own-pool statistics differ')
    require(all(sha(folder / name) == h for name, h in files.items()), 'Replay files changed during verification')
    return raw, files


def worker(index):
    require(_CONTEXT is not None, 'Worker not initialized')
    execution, manifest, e, backend, tests = _CONTEXT
    job = manifest['jobs'][index]
    start = time.perf_counter()
    folder = Path(execution['output']) / 'jobs' / job['id']
    folder.mkdir(exist_ok=False)
    report = selection()._load(HERE / 'report.py', '_stop_report_contract')
    row = {k: job[k] for k in report.IDENTITY}
    row.update(status='running', cached=False, checkpoint=job['checkpoint'], pool_path=job['pool_path'],
        training_data_sha256=job['training_data_sha256'], training_centers_sha256=job['training_centers_sha256'],
        return_rule='stopping_checkpoint', mp_count=job['mp_count'], sizes=job['sizes'],
        test=manifest['tests'][str(job['R'])], start_utc=utc(), phases={}, worker_count=WORKERS,
        solver_threads=1, torch_threads=1, torch_interop_threads=1, pid=os.getpid(),
        timing_clock='time.perf_counter', timing_unit='seconds', selection_sha256=execution['selection_sha256'],
        historical_training_reference=execution['selection_path'], inputs_unchanged=False)
    try:
        with phase(row['phases'], 'input_loading'):
            model, _, pools = own_inputs(job, backend, e)
            own = e.np.concatenate(pools)
            r = job['R']
            x1, x2 = tests[r]
            input_path = folder / 'input_seal.json'
            input_digest = seal(input_path, dict(checkpoint=job['checkpoint'],
                checkpoint_sha256=job['checkpoint_sha256'], state_sha256=job['state_sha256'],
                test=row['test'], selection_sha256=execution['selection_sha256']))
        with phase(row['phases'], 'oos_evaluation'):
            e.evaluate_policy(model, x1, x2, folder / 'oos',
                dict(role='oos', worker_count=WORKERS, input_seal=dict(path=str(input_path), sha256=input_digest)))
        with phase(row['phases'], 'own_pool_evaluation'):
            e.evaluate_policy(model, own[:, :r], own[:, r:], folder / 'own_pool',
                              dict(role='own_pool', worker_count=WORKERS), sizes=job['sizes'])
        with phase(row['phases'], 'raw_verification'):
            oos, op = verify_replay(folder / 'oos', model, x1, x2)
            ins, ip = verify_replay(folder / 'own_pool', model, own[:, :r], own[:, r:], job['sizes'])
            require(e.state_hash(model) == job['state_sha256'], 'Worker changed learned state')
            for field, hf in (('checkpoint', 'checkpoint_sha256'), ('pool_path', 'pool_file_sha256'),
                              ('training_data_path', 'training_data_sha256')):
                require(sha(job[field]) == job[hf], 'Worker inputs changed')
            row.update(mean_oos=oos['mean_oos'], finite_pool_mean=ins['finite_pool_mean'],
                per_center_maxima=ins['per_center_maxima'], raw_files=dict(oos=op, own_pool=ip),
                input_seal=dict(path=str(input_path), sha256=input_digest),
                solves_attempted=oos['solves_attempted']+ins['solves_attempted'],
                solves_accepted=oos['solves_accepted']+ins['solves_accepted'],
                solves_physical_accepted=oos['solves_physical_accepted']+ins['solves_physical_accepted'],
                status_counts={'2': oos['solves_accepted']+ins['solves_accepted']},
                iterations=oos['iterations']+ins['iterations'],
                barrier_iterations=oos['barrier_iterations']+ins['barrier_iterations'],
                inputs_unchanged=True, cost_recomputed_exact=True)
        row['status'] = 'completed'
    except BaseException:
        row.update(status='failed', error=traceback.format_exc())
        for key in ('mean_oos', 'finite_pool_mean', 'per_center_maxima'):
            row.pop(key, None)
    row.update(end_utc=utc(), wall_time_s=time.perf_counter()-start,
               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    row['wall_time_human'] = human(row['wall_time_s'])
    result = folder / 'result.json'
    digest = seal(result, row)
    elapsed = time.perf_counter()-start
    return dict(id=job['id'], status=row['status'], result_path=str(result), result_sha256=digest,
        start_utc=row['start_utc'], end_utc=utc(), wall_time_s=elapsed,
        timing_clock='time.perf_counter', timing_unit='seconds',
        scope='whole worker including result serialization; shared initializer accounted in sweep elapsed')


def verify_job(job, envelope, execution, manifest, backend, tests):
    require(envelope['id'] == job['id'] and envelope['status'] == 'completed', 'Failed worker envelope')
    timing(envelope)
    folder = Path(execution['output']) / 'jobs' / job['id']
    require(Path(envelope['result_path']) == folder / 'result.json', 'Wrong result path')
    row = sealed(envelope['result_path'], envelope['result_sha256'])
    report = selection()._load(HERE / 'report.py', '_stop_report_verification')
    require(all(row[k] == job[k] and type(row[k]) is type(job[k]) for k in report.IDENTITY)
            and row['status'] == 'completed' and row['cached'] is False and row['inputs_unchanged'] is True
            and row['worker_count'] == WORKERS and row['solver_threads'] == row['torch_threads'] == row['torch_interop_threads'] == 1
            and row['test'] == manifest['tests'][str(job['R'])] and row['sizes'] == job['sizes']
            and row['selection_sha256'] == execution['selection_sha256'], 'Raw job identity/status differs')
    timing(row)
    require(envelope['wall_time_s'] >= row['wall_time_s'], 'Whole worker timing excludes evaluation')
    require(set(row['phases']) == {'input_loading', 'oos_evaluation', 'own_pool_evaluation', 'raw_verification'},
            'Missing worker phase')
    for p in row['phases'].values():
        require(p['status'] == 'completed', 'Failed worker phase')
        timing(p)
    e = numerical()
    model, _, pools = own_inputs(job, backend, e)
    r = job['R']
    own = e.np.concatenate(pools)
    binding = row['input_seal']
    require(Path(binding['path']) == folder / 'input_seal.json', 'Wrong per-policy input seal path')
    inputs = sealed(binding['path'], binding['sha256'])
    require(inputs == dict(checkpoint=job['checkpoint'], checkpoint_sha256=job['checkpoint_sha256'],
            state_sha256=job['state_sha256'], test=row['test'], selection_sha256=execution['selection_sha256']),
            'Per-policy OOS seal differs')
    oos, op = verify_replay(folder / 'oos', model, *tests[r])
    ins, ip = verify_replay(folder / 'own_pool', model, own[:, :r], own[:, r:], job['sizes'])
    require(oos['input_seal'] == binding and row['raw_files'] == dict(oos=op, own_pool=ip)
            and row['mean_oos'] == oos['mean_oos'] and row['finite_pool_mean'] == ins['finite_pool_mean']
            and row['per_center_maxima'] == ins['per_center_maxima'], 'Raw job/replay proof mismatch')
    total = 10000 + sum(job['sizes'])
    require(row['solves_attempted'] == row['solves_accepted'] == row['solves_physical_accepted'] == total
            and row['status_counts'] == {'2': total}
            and row['iterations'] == oos['iterations']+ins['iterations']
            and row['barrier_iterations'] == oos['barrier_iterations']+ins['barrier_iterations'], 'Job solve totals differ')
    # Added ONLY after independent parent verification, never to raw result bytes.
    return dict(row, journal_verified=True, result_path=envelope['result_path'],
                result_sha256=envelope['result_sha256'], worker_envelope=envelope)


def production(output, preflight_path, preflight_sha):
    output = fresh_directory(output)
    start = time.perf_counter()
    summary = dict(mode='production', experiment='stopping_checkpoint_20260928', status='running',
        start_utc=utc(), timing_clock='time.perf_counter', timing_unit='seconds', phases={},
        workers=WORKERS, threads=1, preflight_path=str(Path(preflight_path).resolve()),
        preflight_sha256=preflight_sha, current_training_performed=False)
    published = output / 'published'
    staging = output / '.staging'
    try:
        with phase(summary['phases'], 'preflight_validation'):
            pre, manifest = validate_preflight(preflight_path, preflight_sha)
            verify_inputs(manifest)
        with phase(summary['phases'], 'resources'):
            summary['resource_snapshot'] = resource_snapshot(pre['source_sha256'])
            summary['resource_gate'] = resource_gate(summary['resource_snapshot'],
                max(p['peak_rss_bytes'] for p in pre['probes']))
        with phase(summary['phases'], 'test_binding'):
            tests = {r: test_arrays(manifest, r) for r in (3, 10)}
        execution = {k: pre[k] for k in ('selection_path', 'selection_sha256',
            'selection_verification_path', 'selection_verification_sha256', 'source_sha256', 'environment')}
        execution.update(preflight_path=summary['preflight_path'], preflight_sha256=preflight_sha,
            status='frozen', output=str(output), workers=WORKERS, threads=1, tests=manifest['tests'],
            test_files_and_array_content_verified_after_selection=True, counts=COUNTS,
            fresh_jobs=1110, cache_reuse=False)
        execution_path = output / 'execution_manifest.json'
        execution_sha = seal(execution_path, execution)
        summary.update(execution_manifest=str(execution_path), execution_manifest_sha256=execution_sha,
                       environment=pre['environment'], counts=COUNTS)
        (output / 'jobs').mkdir()
        envelopes = []
        with phase(summary['phases'], 'parallel_sweep'):
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                    initializer=initialize_worker, initargs=(str(execution_path), execution_sha)) as pool:
                futures = {pool.submit(worker, i): j for i, j in enumerate(manifest['jobs'])}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        envelope = future.result()
                    except BaseException:
                        # A crashed worker has no measurable per-job duration. Never fabricate one.
                        envelope = dict(id=job['id'], status='failed', wall_time_s=None,
                            error=traceback.format_exc(), observed_at_utc=utc(), workers=WORKERS, threads=1,
                            timing_scope='worker crashed; per-job timing unavailable')
                        failure_folder = output / 'jobs' / job['id']
                        failure_folder.mkdir(exist_ok=True)
                        failure_path = failure_folder / 'worker_failure.json'
                        envelope['failure_path'] = str(failure_path)
                        envelope['failure_sha256'] = seal(failure_path, dict(envelope))
                    envelopes.append(envelope)
            seal(output / 'worker_envelopes.json', envelopes)
            summary['completed_jobs'] = sum(e['status'] == 'completed' for e in envelopes)
            summary['failed_jobs'] = [e['id'] for e in envelopes if e['status'] != 'completed']
            require(len(envelopes) == summary['completed_jobs'] == 1110, 'Incomplete sweep; no publication')
        with phase(summary['phases'], 'parent_raw_verification'):
            by_id = {e['id']: e for e in envelopes}
            require(len(by_id) == 1110, 'Duplicate worker envelopes')
            backend = pure_backend(manifest)
            rows = [verify_job(j, by_id[j['id']], execution, manifest, backend, tests) for j in manifest['jobs']]
            summary['solves_physical_accepted'] = sum(r['solves_physical_accepted'] for r in rows)
        with phase(summary['phases'], 'final_input_verification'):
            verify_inputs(manifest)
            verify_sources(pre['source_sha256'])
            selection_bundle(pre['selection_path'], pre['selection_sha256'],
                             pre['selection_verification_path'], pre['selection_verification_sha256'])
            sealed(preflight_path, preflight_sha)
            sealed(execution_path, execution_sha)
        with phase(summary['phases'], 'publication'):
            require(len(rows) == 1110 and all(r['journal_verified'] for r in rows), 'Global journal barrier failed')
            staging.mkdir(exist_ok=False)
            publisher = selection()._load(HERE / 'report.py', '_stop_publisher')
            publisher.publish_data(dict(manifest, environment=pre['environment'],
                                        source_sha256=pre['source_sha256']), rows, staging)
            elapsed = time.perf_counter()-start
            with (staging / 'RESULTS.md').open('a') as stream:
                stream.write('\nParent elapsed through rendering/report creation: ' + human(elapsed) + '.\n'
                    'Authoritative parent phases and sweep elapsed: ../pipeline_summary.json. '
                    'The sealed runtime_report.json whole_pipeline_elapsed_s remains null intentionally. '
                    'Full process time INCLUDING imports and final serialization/publication is in the sibling '
                    'launch directory child_runtime.json; unavailable until that wrapper exits. '
                    'No sum of concurrent worker times is an elapsed sweep time.\n')
            verify_sources(pre['source_sha256'])
            summary['artifacts'] = {str(Path('published') / p.name): sha(p) for p in sorted(staging.iterdir())}
            staging.rename(published)
        summary['status'] = 'completed'
    except BaseException:
        # Keep diagnostic staging, but never expose failed output as a publication.
        if published.exists():
            published.rename(staging)
        summary.update(status='failed', error=traceback.format_exc())
    summary.update(end_utc=utc(), wall_time_s=time.perf_counter()-start)
    summary['wall_time_human'] = human(summary['wall_time_s'])
    summary['timing_scope'] = 'current parent invocation excluding imports/final summary serialization; full process in launch child_runtime.json'
    try:
        digest = seal(output / 'pipeline_summary.json', summary)
    except BaseException:
        if published.exists():
            published.rename(staging)
        raise
    print(json.dumps(dict(status=summary['status'], summary=str(output / 'pipeline_summary.json'), sha256=digest)), flush=True)
    return 0 if summary['status'] == 'completed' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'production', '_unit'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--sha')
    args = parser.parse_args(argv)
    if args.mode == 'production':
        require(args.preflight is not None and args.sha is not None, 'Production requires --preflight and --sha')
        return production(args.output, args.preflight, args.sha)
    require(args.preflight is None and args.sha is None, 'Preflight bindings only accepted for production')
    return unit_mode(args.output) if args.mode == '_unit' else preflight(args.output)


if __name__ == '__main__':
    sys.exit(main())