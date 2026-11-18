"""Exclusive R10 replay: freeze 810 policies, then generate one shared test."""
from __future__ import annotations

import os

THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'RAYON_NUM_THREADS')
for _name in THREAD_ENV:
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
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
import oracle
import plot_comparison as plot

ROOT = Path(__file__).resolve().parent
GIB = 1024**3
EXPECTED_LOCAL = {'evaluation.py', 'reference.py', 'test_evaluation.py', 'plot_comparison.py',
                  'test_plot.py', 'run_comparison.py', 'test_runner.py', 'preflight.py',
                  'launch.py', 'launch.sh', 'README.md', 'oracle.py', 'test_oracle.py'}
R3_RUNNER = e.c.R3_EVALUATION.parent / 'run_comparison.py'
R3_PREFLIGHT = e.c.R3_EVALUATION.parent / 'preflight.py'
LOCKS = {R3_RUNNER: '2c0b3e60a43edc34adb001da508949d8fa16bea02f367bcb1e180eb49e08bb55',
         R3_PREFLIGHT: '913b7d1515219087bc48a7b4dc86848b69c95baa65773029bc5a7aee867b3b0c'}
e.c.LOCKS.update(LOCKS)

# Definitions execute in THIS namespace: spawn resolves this module, never the
# historical runner. No historical top-level code or entry point is executed.
e.c.load_definitions(globals(), R3_RUNNER,
    ('rss_bytes', 'resources', 'choose_workers', 'capture_sources', 'hashset',
     'verify_files', 'seal', 'read_sealed', 'membership_report', 'disk_gate',
     'compact', 'run_worker', 'process_failure', 'execute', 'require_completed',
     'final_verify', 'runtime_aggregates', 'write_report'),
    (('len(rows) == 780', 'len(rows) == 810', 1),
     ('# Fixed-policy R3 comparison', '# Fixed-policy R10 comparison', 1),
     ('Fresh strict green N200/N800 (180 policies) and SCS N50/N200/N800 (600 policies).',
      'Fresh strict green N50/N200/N800 (270 policies) and SCS N50/N200/N800 (540 policies).', 1),
     ('Green N50 (90 policies), SAA-kNN, and SRO-FA are cached historical references.',
      'Only SAA-kNN and SRO-FA are cached historical references; green N50 is fresh.', 1),
     ('for n in (200, 800):', 'for n in (50, 200, 800):', 1),
     ("    add('Green N50 nominal/CCG (not established here)', 50, 'all', [])\n", '', 1)))
_r3_write_report = write_report


def local_sources():
    # Explicit local set: output reports/launch metadata are never source bindings.
    paths = {ROOT / name for name in EXPECTED_LOCAL}
    paths |= set(e.SOURCE_FILES) | set(oracle.SOURCE_FILES) | set(LOCKS)
    paths.add(e.c.OLD / 'R10/sro_fa/sro_fa_sweep.py')
    e.check(all(p.is_file() for p in paths), 'Finalize all local sources/docs before preflight')
    for path, digest in LOCKS.items():
        e.check(e.c.sha(path) == digest, f'Locked R3 source changed: {path}')
    return sorted(p.resolve() for p in paths)


def require_grid(manifest):
    e.check('cached_green_N50' not in manifest, 'Cached green N50 forbidden')
    e.check(manifest['scs_radii'] == {str(n): list(e.SCS_RADII) for n in plot.SIZES},
            'Changed R10 SCS radius grid')
    expected = {(method, n, rho, seed) for method in ('green', 'scs') for n in plot.SIZES
                for rho in (e.RADII if method == 'green' else e.SCS_RADII) for seed in range(10)}
    actual = [(j['method'], j['Ntotal'], j['radius'], j['seed']) for j in manifest['jobs']]
    e.check(len(actual) == len(expected) == len({j['id'] for j in manifest['jobs']}) == 810
            and set(actual) == expected and manifest['job_count'] == 810,
            'Incomplete/duplicate 810-policy grid')


def freeze_all(output, expected=None):
    sources = local_sources()
    before = {str(p): e.c.sha(p) for p in sources}
    manifest = e.freeze(Path(output))
    capture_sources(manifest, Path(output) / 'frozen', sources)
    e.check(before == {str(p): e.c.sha(p) for p in local_sources()}, 'Source changed during freeze')
    manifest['runner_sources'] = before
    require_grid(manifest)
    training = e.read_json(manifest['files'][manifest['green_training_reference']['summary']]['copy'])
    for name, count in [('initialization', 30), ('nominal', 90), ('ccg', 270),
                        ('lr_validation', 30), ('radius_validation', 30)]:
        phase = training['phases'][name]
        e.check(phase['status'] == 'completed' and len(phase['jobs']) == count
                and all(j['status'] == 'completed' for j in phase['jobs']), f'Incomplete {name}')
    e.check(sum(j['selected_green_gap'] >= .01 for j in manifest['jobs'] if j['method'] == 'green') == 11,
            'Changed returned-own-gap disclosure')
    if expected is not None:
        e.check(hashset(manifest) == hashset(expected), 'Full source hashset differs from preflight')
        for key in ('runner_sources', 'jobs', 'versions', 'saved_green_selections', 'cached_baselines'):
            e.check(manifest[key] == expected[key], f'Preflight identity changed: {key}')
    verify_files(manifest)
    return manifest


def valid_duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def require_timing(row):
    e.check(valid_duration(row['wall_time_s']), 'Invalid monotonic wall time')
    start, end = (datetime.fromisoformat(row[k]) for k in ('start_utc', 'end_utc'))
    e.check(start.utcoffset() is not None and end.utcoffset() is not None and end >= start,
            'Invalid UTC timing boundaries')


def require_oracle_solve(row, points):
    """Validate real LP diagnostics and re-audit the checksummed raw solution."""
    n = len(points)
    e.check(points.shape == (n, 20) and e.np.isfinite(points).all(), 'Invalid oracle audit rows')
    e.check(row['status'] == 'completed' and row['accepted'] is True
            and type(row['solver_status']) is int and row['solver_status'] == 2
            and row['method'] == 'empirical_clairvoyant_reference'
            and row['learned_policy'] is False and row['radius'] is None
            and row['N'] == n and row['retailers'] == 10
            and row['worker_count'] == row['solver_threads'] == 1,
            'Unaccepted oracle LP')
    e.check(row['source'] == str(oracle.SOURCE) and row['source_sha256'] == oracle.SOURCE_SHA256
            and row['solver_parameters'] == {'Threads': 1, 'OutputFlag': 0}
            and row['dimensions'] == dict(variables=11+20*n, constraints=31*n), 'Oracle LP protocol changed')
    e.check(row['timing_clock'] == 'time.perf_counter' and row['timing_unit'] == 'seconds'
            and valid_duration(row['iterations']) and type(row['barrier_iterations']) is int
            and row['barrier_iterations'] >= 0, 'Missing oracle timing/iterations')
    require_timing(row)
    names = ('construction', 'optimization', 'cost_replay_and_solution_serialization')
    e.check(set(row['phases']) == set(names), 'Incomplete oracle phases')
    previous = datetime.fromisoformat(row['start_utc'])
    for name in names:
        phase = row['phases'][name]
        require_timing(phase)
        e.check(phase['status'] == 'completed'
            and previous <= datetime.fromisoformat(phase['start_utc'])
            and datetime.fromisoformat(phase['end_utc']) <= datetime.fromisoformat(row['end_utc']),
            'Incomplete/overlapping oracle phase')
        previous = datetime.fromisoformat(phase['end_utc'])
    e.check(sum(row['phases'][name]['wall_time_s'] for name in names) <= row['wall_time_s'],
            'Oracle phase time exceeds total')
    solution = row['solution']
    e.check(solution['raw'] is True and solution['arrays'] == ['Q10', 'Q1r', 'Q2']
            and e.c.sha(solution['file']) == solution['sha256'], 'Oracle solution hash changed')
    with e.np.load(solution['file'], allow_pickle=False) as data:
        e.check(set(data.files) == {'Q10', 'Q1r', 'Q2'} and data['Q10'].shape == ()
            and data['Q1r'].shape == (10,) and data['Q2'].shape == (n, 10), 'Oracle solution shape changed')
        _, costs = oracle._definitions()  # Locked algebra only; no Env/Model or solve.
        audit = oracle._audit(float(data['Q10']), data['Q1r'], data['Q2'],
                      points[:, :10], points[:, 10:], row['objective'], costs)
    e.check(all(row[k] == value for k, value in audit.items())
            and audit['physical_residual'] <= oracle.PHYSICAL_TOL
            and audit['objective_abs_error'] <= oracle.OBJECTIVE_ABS_TOL, 'Oracle raw audit mismatch')


def require_training_oracle(pref, manifest):
    row = pref['oracle_training_only']
    chosen = [j for j in manifest['jobs'] if (j['method'], j['Ntotal'], j['radius'], j['seed'])
              == ('green', 50, 0., 0)]
    e.check(len(chosen) == 1, 'Missing oracle training cohort')
    spec = chosen[0]
    points = e.np.asarray(spec['centers'], dtype=e.np.float64)[:2]
    e.check(row['scope'] == 'training_only' and row['id'] == spec['id']
            and row['Ntotal'] == 50 and row['Ntrain'] == 40 and row['seed'] == 0
            and row['original_sample_ids'] == spec['original_sample_ids'][:2]
            and row['training_points_sha256'] == hashlib.sha256(points.tobytes(order='C')).hexdigest()
            and Path(row['solution']['file']).resolve() == Path(pref['output']).resolve() / 'oracle_training_solution.npz'
            and row['manifest_sha256'] == pref['manifest_sha256'], 'Oracle training identity mismatch')
    require_oracle_solve(row, points)


def oracle_plot_record(manifest, output, binding):
    """Explicit plot-schema adapter; counts derive from validated LP diagnostics.

    The raw oracle result is immutable and remains in its original schema.
    """
    path = Path(output) / 'oracle/result.json'
    e.check(binding['file'] == str(path.resolve()) and e.c.sha(path) == binding['sha256'],
            'Oracle result hash changed')
    row = e.read_json(path)
    e.check(row['solution'] == binding['solution']
            and row['test'] == manifest['test'] and row['N'] == 10000
            and row['seed'] == row['test_seed'] == 9999 and row['training_seed'] is None
            and row['invocation'] == 'fresh' and row['independent_replicates'] == 1
            and row['manifest_sha256'] == e.c.sha(Path(output) / 'frozen/manifest.json')
            and Path(row['solution']['file']).resolve() == Path(output).resolve() / 'oracle/solution.npz',
            'Oracle common-test identity changed')
    points = e.np.concatenate([e.np.load(manifest['test']['arrays'][k]['file'], allow_pickle=False)
                       for k in ('xi1', 'xi2')], axis=1)
    require_oracle_solve(row, points)
    require_timing(row)
    e.check(valid_duration(row['solve_wall_time_s']) and row['solve_wall_time_s'] <= row['wall_time_s'],
            'Invalid oracle solve/total timing')
    sources = {str(p): e.c.sha(p) for p in oracle.SOURCE_FILES}
    for source, digest in sources.items():
        e.check(manifest['runner_sources'][source] == manifest['files'][source]['sha256'] == digest
            and e.c.sha(manifest['files'][source]['copy']) == digest, 'Oracle source binding changed')
    attempted = int(row['solver_status'] is not None)
    accepted = int(row['accepted'])
    adapted = dict(row, R=row['retailers'], n_test=row['N'], independent_solves=attempted,
               solves_attempted=attempted, solves_accepted=accepted,
               status_counts={str(row['solver_status']): attempted}, source_sha256=sources,
               historical_source_sha256=row['source_sha256'], result_binding=binding)
    plot._oracle_reference(manifest, adapted)
    return adapted


def verify_result_hashes(bindings):
    for path, digest in bindings.items():
        e.check(e.c.sha(path) == digest, f'Policy result hash changed: {path}')


def validate_preflight_evidence(pref, manifest):
    """Nested evidence, not a completed label; independently exercised by fixtures."""
    e.check(pref['status'] == 'completed' and pref['test_generated'] is False
            and pref['job_count'] == 810 and pref['versions'] == e.versions(), 'Incomplete preflight')
    e.check(pref['timing_clock'] == 'time.perf_counter' and pref['timing_unit'] == 'seconds',
            'Invalid preflight clock')
    require_timing(pref)
    for name in ('syntax', 'unit_tests', 'resources', 'full_freeze', 'numerical_training_only',
                 'oracle_training_only', 'final_verification'):
        phase = pref['phases'][name]
        e.check(phase['status'] == 'completed', f'Incomplete preflight phase: {name}')
        require_timing(phase)
    unit = pref['phases']['unit_tests']
    e.check(type(unit['returncode']) is int and unit['returncode'] == 0
            and unit['command'] == [sys.executable, '-B', '-m', 'unittest', '-v',
                                    'test_evaluation', 'test_plot', 'test_runner', 'test_oracle'], 'Invalid unit test evidence')
    syntax = pref['phases']['syntax']
    e.check(syntax['bash_returncode'] == 0 and set(syntax['python_files']) ==
            {str(p) for p in local_sources() if p.suffix == '.py'}, 'Incomplete syntax evidence')
    expected = {(method, n, rho) for method in ('green', 'scs') for n in plot.SIZES for rho in (0., .2)}
    rows = pref['checks']
    e.check(len(rows) == 12 and {(s['method'], s['Ntotal'], s['radius']) for s in rows} == expected,
            'Require all 12 training cohorts')
    specs = {s['id']: s for s in manifest['jobs']}
    accepted = 0
    for row in rows:
        spec = specs[row['id']]
        e.check(all(row[k] == spec[k] for k in ('method', 'Ntotal', 'Ntrain', 'radius', 'seed'))
                and row['seed'] == 0 and row['status'] == 'completed'
                and row['worker_count'] == row['solver_threads'] == 1
                and row['original_sample_ids'] == spec['original_sample_ids'][:2], 'Smoke identity mismatch')
        phase = row['phases']['training_only_replay']
        e.check(phase['status'] == 'completed', 'Incomplete smoke phase')
        require_timing(phase)
        e.check(len(row['checks']) == 2 and e.np.asarray(row['costs']).shape == (2,)
                and e.np.isfinite(row['costs']).all(), 'Incomplete smoke costs/solves')
        e.check(row['training_subset_mean'] == float(e.np.mean(row['costs'])), 'Smoke mean mismatch')
        for diagnostic in row['checks']:
            require_timing(diagnostic)
            e.check(diagnostic['status'] == 2 and diagnostic['accepted'] is True
                    and valid_duration(diagnostic['iterations'])
                    and type(diagnostic['barrier_iterations']) is int and diagnostic['barrier_iterations'] >= 0
                    and valid_duration(diagnostic['physical_residual'])
                    and diagnostic['physical_residual'] <= e.c.PHYSICAL_TOL, 'Unaccepted strict smoke solve')
            accepted += 1
    e.check(accepted == pref['solves_accepted'] == pref['solves_attempted'] == 24
            and pref['status_counts'] == {'2': 24}, 'Require 24 accepted real QPs')
    e.check(pref['solve_counts_scope'] == 'strict_training_QP_only; oracle LP recorded separately',
            'Ambiguous preflight solve counts')
    require_training_oracle(pref, manifest)
    e.check(type(pref['peak_rss_bytes']) is int and pref['peak_rss_bytes'] > 0, 'Missing measured RSS')


def validate_preflight(path, expected_sha256):
    path = Path(path).resolve()
    e.check(isinstance(expected_sha256, str) and len(expected_sha256) == 64
            and e.c.sha(path) == expected_sha256, 'Exact preflight SHA256 mismatch')
    pref = read_sealed(path)
    manifest_path = Path(pref['manifest'])
    e.check(e.c.sha(manifest_path) == pref['manifest_sha256'], 'Preflight manifest binding changed')
    manifest = read_sealed(manifest_path)
    e.check(manifest['status'] == 'frozen' and 'test' not in manifest and manifest['versions'] == e.versions()
            and manifest['workers'] == manifest['threads'] == 1, 'Preflight environment/protocol changed')
    require_grid(manifest)
    validate_preflight_evidence(pref, manifest)
    e.check(pref['source_sha256'] == hashset(manifest), 'Preflight full hashset mismatch')
    e.check(manifest['runner_sources'] == {str(p): e.c.sha(p) for p in local_sources()}, 'Source/doc set changed')
    log = path.parent / 'unit_tests.log'
    e.check(e.c.sha(log) == pref['phases']['unit_tests']['log_sha256'], 'Unit log changed')
    verify_files(manifest)
    return pref, manifest


def freeze_test(manifest, folder):
    """The policy-only manifest must already be sealed before seed 9999 is drawn."""
    require_grid(manifest)
    folder = Path(folder)
    e.check(read_sealed(folder / 'policies_manifest.json') == manifest, 'Policies must be sealed before test')
    e.check(manifest['status'] == 'frozen' and 'test' not in manifest, 'Test already generated')
    verify_files(manifest)
    arrays = e.c.gen(10000, 9999)
    e.check(len(arrays) == 2, 'Generator must return two arrays')
    test = dict(n_test=10000, seed=9999, arrays={},
                protocol='Exact R3 generator definition with R10 dimension; legacy RNG and float64 draws.',
                historical_byte_certification=False,
                caveat='Protocol matches historical R10 code; not byte-certified against historical SRO-FA tests.')
    for name, array in zip(('xi1', 'xi2'), arrays):
        e.check(array.shape == (10000, 10) and array.dtype == e.np.float64
                and e.np.isfinite(array).all() and (array >= 0).all(), 'Invalid R10 test array')
        path = folder / f'{name}.npy'
        with path.open('xb') as file:
            e.np.save(file, array, allow_pickle=False)
        path.chmod(0o444)
        test['arrays'][name] = dict(file=str(path.resolve()), file_sha256=e.c.sha(path),
            array_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest(), dtype='float64', shape=[10000, 10])
    manifest['test'] = test


def verify_journals(manifest, output):
    """Stream actual journals, independently of per-policy summary counters."""
    counts = dict(policies=0, solves_attempted=0, solves_accepted=0, status_counts={})
    for spec in manifest['jobs']:
        for name, expected in [('oos', 10000), ('is', sum(spec['sizes']))]:
            count = 0
            with (Path(output) / 'policies' / spec['id'] / f'{name}_solves.jsonl').open() as file:
                for count, line in enumerate(file, 1):
                    d = json.loads(line)
                    e.check(d['row'] == count - 1 and d['phase'] == name and d['status'] == 2
                            and d['accepted'] is True, 'Failed/misordered journal solve')
                    require_timing(d)
                    e.check(valid_duration(d['controller_wall_time_s']) and
                            valid_duration(d['physical_residual']) and d['physical_residual'] <= e.c.PHYSICAL_TOL,
                            'Invalid journal timing/residual')
            e.check(count == expected, 'Incomplete strict journal count')
            counts['solves_attempted'] += count
            counts['solves_accepted'] += count
        counts['policies'] += 1
    counts['status_counts'] = {'2': counts['solves_accepted']}
    return counts


def write_report(output, manifest, summary, aggregates):
    _r3_write_report(output, manifest, summary, aggregates)
    # The inherited historical report predates the authorized fresh gray LP.
    oracle_path = output / 'oracle/result.json'
    raw = e.read_json(oracle_path) if oracle_path.exists() else None
    aggregates['common_oracle'] = dict(result=raw, binding=summary.get('oracle_binding'),
        controller_phase=summary.get('phases', {}).get('common_oracle'),
        independent_solves=None if raw is None else int(raw.get('solver_status') is not None),
        separate_replay_time_s=None, separate_serialization_time_s=None,
        note='One common test-fitted LP, not policy training, not three/eight/ten independent observations. '
             'Oracle currently measures replay and NPZ serialization jointly; separate timings unavailable. '
             'run total excludes final result JSON write; parent common_oracle phase includes it and audit/binding.')
    aggregates['historically_nonconverged_scs'] = manifest['historically_nonconverged_scs']
    aggregates['green_returned_own_gap_ge_one_percent'] = [dict(id=j['id'], gap=j['selected_green_gap'])
        for j in manifest['jobs'] if j['method'] == 'green' and j['selected_green_gap'] >= .01]
    with (output / 'RESULTS.md').open('a') as report:
        report.write('\nThree historically nonconverged SCS N50 policies are included unchanged: '
                     '(rho, seed) = (.15, 4), (.6, 5), (.8, 1).\n'
                     'Eleven green returned checkpoints have their own gap >=1%; the stopping candidate '
                     'passed, and the saved minimum-AP selection is retained without reselection.\n'
                     'Test arrays have shape 10000 x 10. Generator output follows the original code protocol, '
                     'but is not byte-certified against historical SRO-FA tests. No cached R3 test hashes are used.\n'
                     'Missing historical kNN selection, benchmark construction and evaluation timings are '
                     'unavailable; reruns would be required. Historical SRO-FA partition, time_lp solver, '
                     'evaluation and total fields remain separate; time_lp is not optimization wrapper time.\n')
        report.write('\n## Fresh common gray reference: correction to inherited report\n\n'
                     'Any inherited statement that no benchmark construction occurred does not apply to '
                     'this R10 gray reference. It is ONE fresh joint LP fitted to the common 10,000-row '
                     'test (seed 9999), shown identically in all three panels, not eight independent '
                     'results, policy training, deployable OOS performance, or a population lower bound. '
                     'Historical gray entries are not plotted, filled, or overwritten. '
                     'No oracle solution enters neural replay, policy selection, or training.\n\n'
                     '| Common oracle scope | Status | Wall seconds | Human time |\n'
                     '|---|---|---:|---|\n')
        phases = {} if raw is None else raw.get('phases', {})
        scopes = [(name, phases.get(name)) for name in
                  ('construction', 'optimization', 'cost_replay_and_solution_serialization')]
        scopes += [('replay separately (unmeasured)', None), ('serialization separately (unmeasured)', None),
                   ('oracle.run total (excludes final result JSON)', raw),
                   ('controller total (includes result JSON, audit/binding)',
                    summary.get('phases', {}).get('common_oracle'))]
        for label, row in scopes:
            value = None if row is None else row.get('wall_time_s')
            status = 'unavailable' if row is None else row.get('status', 'unavailable')
            seconds = 'unavailable' if value is None else f'{value:.6f}'
            human = 'unavailable; rerun with instrumentation required' if value is None else e.c.human(value)
            report.write(f'| {label} | {status} | {seconds} | {human} |\n')
        report.write('\nOracle: one worker, one solver thread; environment and solver iterations/status '
                     'are recorded in its raw result when reached. Oracle phases are separate from fresh '
                     'neural evaluation and persisted nominal/CCG/CV/SRO-FA tables above. '
                     'Failed or unattempted oracle timings are not completed-run aggregates.\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--preflight-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    start, started = time.perf_counter(), e.c.utc()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('logs', 'policies', 'process_failures'):
        (output / name).mkdir()
    summary = dict(experiment=e.EXPERIMENT, status='running', start_utc=started, jobs=[], phases={},
        expected_jobs=810, output=str(output), figures=[], plot_data=None, source_hash_final=False,
        preflight=str(args.preflight.resolve()), preflight_sha256=args.preflight_sha256,
        timing_clock='time.perf_counter', timing_unit='seconds', versions=e.versions(),
        working_directory=os.getcwd(), invocation_argv=sys.orig_argv if argv is None else list(argv),
        new_training=False, new_selection=False, test_generated=False, workers=None)
    manifest = None
    try:
        with e.c.phase(summary['phases'], 'preflight_validation'):
            pref, previous = validate_preflight(args.preflight, args.preflight_sha256)
        with e.c.phase(summary['phases'], 'resources'):
            snapshot = resources()
            budget = choose_workers(snapshot, pref['peak_rss_bytes'])
            summary.update(resources=snapshot, resource_policy=budget, workers=budget['workers'], threads=1)
            copies = sum(Path(v['copy']).stat().st_size for v in previous['files'].values())
            solves = 810*10000 + sum(sum(j['sizes']) for j in previous['jobs'])
            summary['disk'] = disk_gate(output, 2*copies + 4096*solves + 2*GIB, 'freeze, journals and reports')
        with e.c.phase(summary['phases'], 'full_freeze'):
            manifest = freeze_all(output, previous)
            manifest.update(workers=budget['workers'], resources=snapshot, resource_policy=budget,
                            preflight=summary['preflight'], preflight_sha256=args.preflight_sha256)
            del previous
            summary['policies_manifest_sha256'] = seal(output / 'frozen/policies_manifest.json', manifest)
        with e.c.phase(summary['phases'], 'test_generation'):
            freeze_test(manifest, output / 'frozen')
            summary['test_generated'] = True
        manifest_path = output / 'frozen/manifest.json'
        with e.c.phase(summary['phases'], 'seal_manifest'):
            digest = seal(manifest_path, manifest)
        summary.update(manifest=str(manifest_path), manifest_sha256=digest, scs_membership=membership_report(manifest))
        with e.c.phase(summary['phases'], 'common_oracle'):
            raw_oracle = oracle.run(manifest, output)  # Exactly once, before neural replay; never passed to workers.
            oracle_path = output / 'oracle/result.json'
            binding = dict(file=str(oracle_path.resolve()), sha256=e.c.sha(oracle_path),
                           solution=raw_oracle['solution'])
            e.check(e.read_json(oracle_path) == raw_oracle, 'Oracle persisted result differs')
            summary['oracle_binding'] = binding
            oracle_data = oracle_plot_record(manifest, output, binding)
            summary['oracle_binding_sha256'] = seal(output / 'oracle_binding.json', binding)
            oracle_path.chmod(0o444)
            Path(raw_oracle['solution']['file']).chmod(0o444)
        with e.c.phase(summary['phases'], 'evaluation_sweep'):
            execute(manifest, manifest_path, output, summary)
        with e.c.phase(summary['phases'], 'final_verification'):
            summary['policy_result_sha256'] = {str(output / 'policies' / spec['id'] / 'result.json'):
                e.c.sha(output / 'policies' / spec['id'] / 'result.json') for spec in manifest['jobs']}
            rows = [e.read_json(output / 'policies' / spec['id'] / 'result.json') for spec in manifest['jobs']]
            verify_result_hashes(summary['policy_result_sha256'])
            require_completed(manifest, rows, digest)
            summary['journal_counts'] = verify_journals(manifest, output)
            final_verify(manifest, manifest_path, digest)
            e.check(oracle_plot_record(manifest, output, binding) == oracle_data, 'Oracle changed before plotting')
            summary['source_hash_final'] = True
        with e.c.phase(summary['phases'], 'figure_and_plot_data'):
            stage = output / 'publication_staging'
            stage.mkdir()
            data = plot.build_plot_data(manifest, rows, oracle_data)
            e.write_json(stage / 'plot_data.json', data)
            plot.render(data, stage)
            names = ('plot_data.json', 'comparison_R10.png', 'comparison_R10.pdf')
            e.check(all((stage / name).stat().st_size > 0 for name in names), 'Missing staged publication')
        summary['status'] = 'completed'
        summary['figures'] = [str(output / 'published' / name) for name in names[1:]]
        summary['plot_data'] = str(output / 'published/plot_data.json')
        summary['wall_time_s'] = time.perf_counter() - start
        aggregates = runtime_aggregates(summary['jobs'])
        with e.c.phase(summary['phases'], 'reporting'):
            write_report(output, manifest, summary, aggregates)
        # Recheck after rendering/reporting; publish the complete trio atomically.
        final_verify(manifest, manifest_path, digest)
        verify_result_hashes(summary['policy_result_sha256'])
        e.check(read_sealed(output / 'oracle_binding.json') == binding
            and e.c.sha(output / 'oracle_binding.json') == summary['oracle_binding_sha256']
            and oracle_plot_record(manifest, output, binding) == oracle_data,
            'Oracle binding/result changed before publication')
        e.check(not (output / 'published').exists(), 'Publication already exists')
        stage.rename(output / 'published')
    except BaseException:
        summary.update(status='failed', error=traceback.format_exc(), figures=[], plot_data=None)
        raise
    finally:
        indexed = {r['id']: r for r in summary['jobs']}
        for path in sorted((output / 'policies').glob('*/result.json')):
            row = e.read_json(path)
            indexed[row['id']] = compact(row, path)
        summary['jobs'] = sorted(indexed.values(), key=lambda r: r['id'])
        summary['completed_jobs'] = sum(r['status'] == 'completed' for r in summary['jobs'])
        summary['failed_jobs'] = sum(r['status'] != 'completed' for r in summary['jobs'])
        summary['unrecorded_jobs'] = 810 - len(indexed)
        summary['solves_attempted'] = sum(r.get('solves_attempted', 0) for r in summary['jobs'])
        summary['solves_accepted'] = sum(r.get('solves_accepted', 0) for r in summary['jobs'])
        summary['solve_counts_scope'] = 'Recorded policy counters; missing/process-failed workers are not certified zero work.'
        if manifest is not None:
            summary['missing_job_ids'] = [s['id'] for s in manifest['jobs'] if s['id'] not in indexed]
        summary.update(end_utc=e.c.utc(), wall_time_s=time.perf_counter()-start, peak_rss_bytes=rss_bytes())
        summary['wall_time_human'] = e.c.human(summary['wall_time_s'])
        summary['wall_time_scope'] = 'Entry through report/publication; final JSON serialization excluded; launcher records whole child.'
        current = runtime_aggregates(summary['jobs'])
        if 'aggregates' in locals():
            current.update({k: v for k, v in aggregates.items() if k not in ('groups', 'note', 'timing_unit')})
        if manifest is not None and 'reporting' not in summary['phases'] and not (output / 'RESULTS.md').exists():
            # A failed sweep still reports existing evidence; never resumes computation.
            try:
                with e.c.phase(summary['phases'], 'reporting'):
                    write_report(output, manifest, summary, current)
            except Exception:
                summary.update(status='failed', report_error=traceback.format_exc())
        if (output / 'oracle/result.json').exists():
            current['common_oracle'] = dict(current.get('common_oracle', {}),
                result=e.read_json(output / 'oracle/result.json'),
                controller_phase=summary['phases'].get('common_oracle'), binding=summary.get('oracle_binding'))
        summary.update(end_utc=e.c.utc(), wall_time_s=time.perf_counter()-start, peak_rss_bytes=rss_bytes())
        summary['wall_time_human'] = e.c.human(summary['wall_time_s'])
        current.update(sweep=summary['phases'].get('evaluation_sweep'),
                       end_to_end_wall_time_s=summary['wall_time_s'], workers=summary['workers'],
                       environment=summary.get('resources'), status=summary['status'])
        e.write_json(output / 'runtime_aggregates.json', current)
        e.write_json(output / 'summary.json', summary)
        if (output / 'RESULTS.md').exists():
            with (output / 'RESULTS.md').open('a') as report:
                report.write(f"\nFinal status: {summary['status']}; whole pipeline {summary['wall_time_s']:.6f} s "
                             f"({summary['wall_time_human']}); final JSON serialization excluded.\n")
    return output


if __name__ == '__main__':
    main()