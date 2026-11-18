"""Training-only continuation inventory and isolated-process orchestration.

Importing this module does not import a numerical backend or launch work.
See README.md for the continuation adapter's deliberately explicit contract.
"""
import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
R3 = BASE / 'green_R3_N200_N800_20260923'
R10 = BASE / 'green_R10_N50_N200_N800_20260924'
BLACK = BASE / 'N50_R3_train40_val10'
CAMPAIGNS = {
    3: R3 / 'results/20260923T092253.404783+0000',
    10: R10 / 'results/20260924T131056.490113+0000',
}
BLACK_RUN = BLACK / 'results/20260916T080636.766867Z'
PROVENANCE_LOCKS = {
    3: '7eb7938a5d9d505ece1761e3aad5102cabd1d097bb19428e1e177b3fb16d3e00',
    10: 'f432d043ffbf779b7efb1355429e7f87bb496cc1bf537291e3e26f793134bfd5',
    'black': '0a1e289e3a75d62ec35ea8ab14d7b7649d1cc5f22c2fd4fbb5a929c1cf131dcd',
}
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
GROUPS = ((3, 200), (10, 50), (3, 800), (10, 200), (10, 800))
EXPECTED_PASS = {(3, 200): 38, (3, 800): 39, (10, 50): 42,
                 (10, 200): 45, (10, 800): 45}
SMOKES = ((3, 200, 1.5), (3, 800, 1.5), (10, 50, 1.5),
          (10, 200, 1.5), (10, 800, 1.5), (10, 50, 2.))
THREAD_ENV = {key: '1' for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
PROTOCOL = dict(experiment='black_continuation_R3_R10_20260925', max_mp=40,
    separation_tolerance=.01, resume_from='last green STOP and its own pool',
    selection='earliest minimum AP mean in entire original-plus-continuation prefix',
    seeds=list(range(10)), network_seed=0, split_seed='100000 + seed',
    train_fraction=.8, nominal='reuse saved per-seed validation-selected LR winner',
    test_access=False, validation_during_training=False, retries=0,
    torch_threads=1, gurobi_threads=1, max_joint_workers=16)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite numeric evidence')
    return value


def seconds(value):
    require(finite(value) >= 0, 'Negative duration')
    return value


def remaining_mp(old_count):
    require(type(old_count) is int and 0 <= old_count <= 40, 'Invalid original MP count')
    return 40 - old_count


def key(row):
    return row['R'], row['Ntotal'], row['seed'], row['radius']


def job_id(r, n, seed, radius):
    return f'R{r}_N{n}_s{seed}_rho{radius:g}'


def grid(r, sizes, radii=RADII):
    return {(r, n, seed, radius) for n in sizes for seed in range(10) for radius in radii}


def exact_grid(rows, expected):
    actual = [key(row) for row in rows]
    require(len(actual) == len(expected) and set(actual) == expected,
            'Incomplete, extra or duplicate configuration grid')


def check_inventory(manifest):
    jobs, cached = manifest['jobs'], manifest['cached_black']
    expected = grid(3, (200, 800)) | grid(10, (50, 200, 800), RADII + (2.,))
    exact_grid(jobs, expected)
    exact_grid(cached, grid(3, (50,)))
    require(len(jobs) == 480 and len(cached) == 90, 'Require exact 570 cohort')
    require(sum(j['source_summary_path'] is not None for j in jobs) == 450,
            'Require 450 original green trajectories')
    require(all((j['source_summary_path'] is None) == (j['R'] == 10 and j['radius'] == 2.)
                for j in jobs), 'Only thirty R10 radius-two jobs may start fresh')
    require(len({j['id'] for j in jobs + cached}) == 570, 'Duplicate job IDs')
    require(Counter(j['terminal_status'] for j in cached) ==
            Counter(separation_satisfied=89, iteration_cap=1), 'Cached black status drift')
    require([(j['seed'], j['radius']) for j in cached if j['terminal_status'] == 'iteration_cap']
            == [(9, .7)], 'Unexpected cached cap identity')
    for r, n in GROUPS:
        rows = [j for j in jobs if (j['R'], j['Ntotal']) == (r, n) and j['radius'] in RADII]
        require(sum(j['expected']['stop_every_center_pass'] for j in rows) == EXPECTED_PASS[r, n],
                f'Green-last closure inventory drift R{r} N{n}')


def merge_hashes(target, incoming):
    for path, sha in incoming.items():
        require(Path(path).is_absolute() and isinstance(sha, str) and len(sha) == 64,
                'Malformed absolute-path hash binding')
        require(path not in target or target[path] == sha, f'Conflicting original hash: {path}')
        target[path] = sha


def verify_hashes(mapping):
    for path, sha in mapping.items():
        require(digest(path) == sha, f'Frozen input changed: {path}')


class Capture:
    """Hash each original once at inventory construction; never modify originals."""
    def __init__(self):
        self.hashes = {}

    def add(self, path, expected=None):
        path = str(Path(path).resolve())
        if path not in self.hashes:
            self.hashes[path] = digest(path)
        sha = self.hashes[path]
        require(expected is None or sha == expected, f'Original binding mismatch: {path}')
        return sha

    def read(self, path, expected=None):
        self.add(path, expected)
        return read(path)


def local_paths():
    # Explicit nonrecursive pattern also captures a separately supplied launcher.
    paths = sorted(set(ROOT.glob('*.py')) | {ROOT / 'README.md'})
    require({p.name for p in paths} >= {'campaign.py', 'test_campaign.py', 'continuation.py', 'README.md'},
            'Campaign, tests, adapter and README must exist before capture')
    return paths


def verify_sources(manifest):
    require({str(p) for p in local_paths()} == set(manifest['local_source_sha256']),
            'Local source inventory changed; preflight again')
    verify_hashes(manifest['source_sha256'])


def candidate_binding(candidate, ntrain):
    require(candidate['assessed'] is True, 'Unassessed original candidate')
    samples = candidate['samples']
    require([s['sample_id'] for s in samples] == list(range(ntrain)), 'Incomplete center coverage')
    require(len(candidate['pool_costs']) == ntrain, 'Incomplete pool costs')
    residuals = []
    for index, sample in enumerate(samples):
        require(sample['controller_status'] == sample['status'] == 'completed', 'Failed source AP')
        require(sample['candidate_id'] == candidate['candidate_id'] and
                sample['candidate_sha256'] == candidate['state_sha256'] and
                sample['checkpoint_sha256'] == candidate['checkpoint_sha256'] and
                sample['pool_sha256'] == candidate['pool_sha256'], 'AP/checkpoint/pool identity mismatch')
        bound, pool = finite(sample['bound']), finite(sample['pool_cost'])
        require(pool == candidate['pool_costs'][index], 'Per-center pool mismatch')
        residual = max(0., bound - pool) / max(1., abs(bound), abs(pool))
        require(math.isclose(residual, finite(sample['residual']), rel_tol=1e-12, abs_tol=1e-12),
                'Stored separation residual mismatch')
        residuals.append(residual)
    ap = sum(finite(s['incumbent']['cost']) for s in samples) / ntrain
    require(math.isclose(ap, finite(candidate['ap_incumbent_mean']), rel_tol=1e-12, abs_tol=1e-12),
            'Stored AP mean mismatch')
    closed = all(v <= .01 for v in residuals)
    require((candidate['separation_status'] == 'separation_satisfied') == closed,
            'Stored separation status mismatch')
    return closed


def bind_trajectory(capture, compact, winner, r, n, green):
    """Retain full original summaries; no sample duplication in the job manifest."""
    path = Path(compact['summary_path']).resolve()
    record = capture.read(path, compact['summary_sha256'])
    require(record['status'] == 'completed' and record['complete'] is True, 'Incomplete original CCG')
    require((record['Ntotal'], record['seed'], record['radius']) ==
            (n, compact['seed'], compact['radius']), 'Original job identity mismatch')
    require(not green or record['R'] == r, 'Original retailer dimension mismatch')
    require(record['nominal_checkpoint_sha256'] == winner['checkpoint_sha256'] and
            record['nominal_state_sha256'] == winner['state_sha256'] and
            record['training_data_sha256'] == winner['training_data_sha256'] and
            record['data_sha256'] == winner['data_sha256'] and record['lr_net'] == winner['lr_net'],
            'Source trajectory is not bound to saved nominal winner')
    require(Path(record['summary_path']).resolve() == path, 'Original summary path mismatch')
    remaining_mp(record['mp_count'])
    candidates = record['iterations']
    require(len(candidates) == record['mp_count'] + 1, 'Original full-prefix/MP mismatch')
    files = {str(path): capture.add(path, compact['summary_sha256'])}

    def add(p, expected=None):
        files[str(Path(p).resolve())] = capture.add(p, expected)

    for name, sha in (('provenance.json', record['provenance_sha256']),
                      ('training_data.json', winner['training_data_sha256']),
                      ('nominal.pth', winner['checkpoint_sha256'])):
        add(path.parent / 'inputs' / name, sha)
    passes = []
    for i, candidate in enumerate(candidates):
        require(candidate['candidate_id'] == i, 'Noncontiguous original full prefix')
        folder = path.parent / f'candidate_{i:02d}'
        require(Path(candidate['checkpoint']).resolve() == folder / 'candidate.pth' and
                Path(candidate['ap_reference']).resolve() == folder / 'assessment.json',
                'Candidate path outside its original trajectory')
        passes.append(candidate_binding(candidate, 4 * n // 5))
        add(candidate['checkpoint'], candidate['checkpoint_sha256'])
        add(folder / 'pool.json', candidate['pool_file_sha256'])
        assessment = capture.read(folder / 'assessment.json')
        for field in ('candidate_id', 'checkpoint_sha256', 'state_sha256', 'pool_sha256',
                      'pool_file_sha256', 'samples', 'ap_incumbent_mean', 'restricted_mean'):
            require(assessment[field] == candidate[field], f'Assessment binding mismatch: {field}')
        # Includes all sample JSONs and mp/{pool,controller_training,training,...}.
        # Training reads the full summary rather than duplicating samples here.
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                require(not p.is_symlink(), f'Symlink inside original candidate: {p}')
                add(p)
        if green:
            ap, pool = candidate['ap_incumbent_mean'], finite(candidate['restricted_mean'])
            is_stop = abs(ap - pool) / max(abs(pool), 1e-9) < .01
            require(is_stop == (i == len(candidates) - 1), 'Not original first green STOP prefix')
    if green:
        require(record['terminal_status'] == 'green_gap_satisfied' and
                record['stopping_candidate_id'] == len(candidates) - 1, 'Wrong green stopping policy')
    add(record['selected_checkpoint'], record['selected_checkpoint_sha256'])
    stop = candidates[-1]
    expected = dict(old_mp_count=record['mp_count'], remaining_mp=remaining_mp(record['mp_count']),
        stop_candidate_id=stop['candidate_id'], stop_checkpoint=stop['checkpoint'],
        stop_checkpoint_sha256=stop['checkpoint_sha256'], stop_state_sha256=stop['state_sha256'],
        stop_pool_path=str(Path(stop['ap_reference']).parent / 'pool.json'),
        stop_pool_file_sha256=stop['pool_file_sha256'], stop_pool_sha256=stop['pool_sha256'],
        stop_every_center_pass=passes[-1], previous_ccg_wall_time_s=seconds(record['wall_time_s']),
        previous_started_at_utc=record['started_at_utc'], previous_ended_at_utc=record['ended_at_utc'])
    return record, files, expected


def load_reconstructor(capture, sources):
    path = BLACK / 'stopping_only_min_ap/reconstruct.py'
    sha = 'ed673cc82d6983bac07ca9d3f0ebd4f3776b21919eaf67dccd186d7ff19bfc42'
    capture.add(path, sha)
    merge_hashes(sources, {str(path): sha})
    spec = importlib.util.spec_from_file_location('_black_training_only_reconstruct', path)
    require(spec is not None and spec.loader is not None, 'Cannot load pure reconstruction module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reconstruct


def build_manifest():
    capture = Capture()
    sources, jobs, cached, nominal_rows, campaign_rows = {}, [], [], [], []
    sourceconfig = {}
    for r, run in [(3, CAMPAIGNS[3]), (10, CAMPAIGNS[10]), ('black', BLACK_RUN)]:
        provenance_path = run / 'provenance.json'
        provenance = capture.read(provenance_path, PROVENANCE_LOCKS[r])
        original_sources = provenance['source_sha256']
        if r != 'black':
            require(len(original_sources) == {3: 35, 10: 54}[r], 'Original provenance coverage changed')
            sourceconfig[str(r)] = provenance
        merge_hashes(sources, original_sources)
        for p, sha in original_sources.items():
            capture.add(p, sha)
        summary_path = run / 'summary.json'
        summary = capture.read(summary_path,
            '0e1760786ec5f33391137c66558c6b91d0e7b63cd24c8cc19c611c1f4d28d1ce' if r == 'black' else None)
        require(summary['status'] == 'completed' and summary['inputs_unchanged'] is True,
                'Original campaign incomplete/changed')
        require(summary['provenance_sha256'] == PROVENANCE_LOCKS[r], 'Original provenance not bound')
        phases = summary if r == 'black' else summary['phases']
        winners = {}
        for row in phases['nominal' if r == 'black' else 'lr_validation']['jobs']:
            require(row['status'] == 'completed' and row['complete'], 'Incomplete nominal selection')
            winner = dict(row['winner'])
            if r == 'black':
                winner['Ntotal'] = 50
            n, seed = winner['Ntotal'], winner['seed']
            require((n, seed) not in winners, 'Duplicate nominal winner')
            timing = winner['training_timing']
            require('epoch_history' not in timing, 'Unexpected bulky nominal timing schema')
            require(timing['status'] == 'completed', 'Uncompleted saved nominal training')
            seconds(timing['wall_time_s'])
            winners[n, seed] = winner
            nominal_rows.append(dict(R=3 if r == 'black' else r, Ntotal=n, seed=seed,
                radius=0., method='cached_nominal_training', status='cached_only',
                historical_status='completed', training_timing=timing,
                historical_wall_time_s=timing['wall_time_s'], current_training_time_s=0.))
        sizes = (50,) if r == 'black' else ((200, 800) if r == 3 else (50, 200, 800))
        require(set(winners) == {(n, s) for n in sizes for s in range(10)}, 'Nominal winner grid mismatch')
        raw_jobs = phases['ccg']['jobs']
        exact_grid([dict(j, R=3 if r == 'black' else r, Ntotal=50 if r == 'black' else j['Ntotal'])
                    for j in raw_jobs], grid(3 if r == 'black' else r, sizes))
        campaign_rows.append(dict(R=r, summary_path=str(summary_path),
            summary_sha256=capture.add(summary_path), provenance_path=str(provenance_path),
            provenance_sha256=PROVENANCE_LOCKS[r], historical_wall_time_s=summary['wall_time_s'],
            historical_phase_wall_times={name: phase.get('wall_time_s')
                for name, phase in phases.items() if isinstance(phase, dict) and 'wall_time_s' in phase},
            environment=provenance['environment']))
        reconstruct = load_reconstructor(capture, sources) if r == 'black' else None
        choices = {}
        if r == 'black':
            # This artifact was written BEFORE validation/test; never open the OOS summary.
            choice_path = BLACK / 'stopping_only_min_ap/results/training_only_choices.json'
            choice_data = capture.read(choice_path)
            require(choice_data['validation_used_for_candidate_selection'] is False and
                    choice_data['test_used_for_candidate_selection'] is False, 'Unsafe black choices')
            black_choices = [c for c in choice_data['choices'] if c['color'] == 'black']
            require(len(black_choices) == 90, 'Missing cached black choices')
            choices = {(c['seed'], c['radius']): c for c in black_choices}
            require(len(choices) == 90, 'Duplicate black choices')
        for compact in sorted(raw_jobs, key=lambda j: (j.get('Ntotal', 50), j['seed'], j['radius'])):
            n = 50 if r == 'black' else compact['Ntotal']
            winner = winners[n, compact['seed']]
            record, files, expected = bind_trajectory(capture, compact, winner,
                                                      3 if r == 'black' else r, n, r != 'black')
            for p, sha in ((summary_path, capture.add(summary_path)),
                           (provenance_path, PROVENANCE_LOCKS[r]),
                           (winner['checkpoint'], winner['checkpoint_sha256']),
                           (winner['training_data_path'], winner['training_data_sha256'])):
                files[str(Path(p).resolve())] = capture.add(p, sha)
            if r == 'black':
                require(reconstruct is not None, 'Missing black reconstructor')
                selected = reconstruct(record, 'black')
                original = choices[compact['seed'], compact['radius']]
                for field in ('checkpoint', 'checkpoint_sha256', 'state_sha256', 'selected_iteration',
                              'stop_candidate', 'mp_count', 'terminal_status', 'source_summary_path'):
                    require(selected[field] == original[field], f'Cached black identity mismatch: {field}')
                files[str(choice_path)] = capture.add(choice_path)
                cached.append(dict(id=job_id(3, 50, compact['seed'], compact['radius']), R=3, Ntotal=50,
                    seed=compact['seed'], radius=compact['radius'], status='cached_only',
                    terminal_status=selected['terminal_status'], checkpoint=selected['checkpoint'],
                    checkpoint_sha256=selected['checkpoint_sha256'], state_sha256=selected['state_sha256'],
                    selected_iteration=selected['selected_iteration'], mp_count=selected['mp_count'],
                    source_summary_path=compact['summary_path'], source_summary_sha256=compact['summary_sha256'],
                    previous_ccg_wall_time_s=expected['previous_ccg_wall_time_s'], current_ccg_wall_time_s=None,
                    input_sha256=files, retrained=False, cached_nominal=selected['cached_nominal']))
            else:
                jobs.append(dict(id=job_id(r, n, compact['seed'], compact['radius']), R=r, Ntotal=n,
                    seed=compact['seed'], radius=compact['radius'], source_summary_path=compact['summary_path'],
                    source_summary_sha256=compact['summary_sha256'], nominal_record=winner,
                    input_sha256=files, expected=expected))
        if r == 10:
            for (n, seed), winner in sorted(winners.items()):
                files = {str(p): capture.add(p, sha) for p, sha in (
                    (summary_path, capture.add(summary_path)), (provenance_path, PROVENANCE_LOCKS[r]),
                    (winner['checkpoint'], winner['checkpoint_sha256']),
                    (winner['training_data_path'], winner['training_data_sha256']))}
                jobs.append(dict(id=job_id(10, n, seed, 2.), R=10, Ntotal=n, seed=seed, radius=2.,
                    source_summary_path=None, source_summary_sha256=None, nominal_record=winner,
                    input_sha256=files, expected=dict(old_mp_count=0, remaining_mp=40,
                        stop_every_center_pass=False, previous_ccg_wall_time_s=0.)))
    local = {str(p): capture.add(p) for p in local_paths()}
    merge_hashes(sources, local)
    manifest = dict(schema_version=1, protocol=PROTOCOL, workers={'ccg': 1},
        sourceconfig=sourceconfig,
        jobs=jobs, cached_black=cached, cached_nominal=nominal_rows, original_campaigns=campaign_rows,
        source_sha256=sources, local_source_sha256=local, input_sha256=capture.hashes,
        source_capture_pattern='nonrecursive *.py plus README.md, including parent launcher if present',
        original_inputs_copied=False, created_at_utc=utc(),
        counts=dict(total=570, scheduled=480, existing_green=450, fresh_radius2=30, cached_black=90))
    check_inventory(manifest)
    return manifest


def seal_manifest(output, manifest):
    path = output / 'manifest.json'
    write(path, manifest)
    sha = digest(path)
    (output / 'manifest.sha256').write_text(sha + '\n')
    path.chmod(0o444)
    (output / 'manifest.sha256').chmod(0o444)
    return path, sha


def verified_manifest(path, expected):
    require(digest(path) == expected, 'Manifest seal mismatch')
    manifest = read(path)
    require(manifest['protocol'] == PROTOCOL, 'Protocol mismatch')
    check_inventory(manifest)
    verify_sources(manifest)
    return manifest


def schedule(jobs):
    # Radius-major, seed-round-robin and interleaved R3/R10 size groups.
    rank = {group: i for i, group in enumerate(GROUPS)}
    return sorted(jobs, key=lambda j: (j['radius'], j['seed'], rank[j['R'], j['Ntotal']]))


def resource_snapshot(manifest):
    """Reuse only audited, hash-bound pure definitions; no numerical imports."""
    paths = [(R3 / 'preflight.py', ('environment',)),
             (R10 / 'r10_resources.py', ('_cpus', 'resources'))]
    namespace = dict(Path=Path, os=os, sys=sys, platform=platform, utc=utc,
        THREAD_ENV=THREAD_ENV,
        peak_rss_bytes=lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    for path, names in paths:
        require(digest(path) == manifest['source_sha256'][str(path)], 'Resource source changed')
        tree = ast.parse(path.read_text())
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        require({n.name for n in definitions} == set(names), 'Resource definition coverage mismatch')
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['resources']()


def choose_workers(snapshot, peaks):
    require(peaks and all(type(v) in (float, int) and v > 0 for v in peaks), 'Missing preflight RSS')
    effective = len(snapshot['affinity'])
    if snapshot['cpu_quota_cores'] is not None:
        effective = min(effective, math.floor(snapshot['cpu_quota_cores']))
    cpu_reserve = max(4, math.ceil(.1 * effective))
    load = math.ceil(max(snapshot['load_average']))
    cpu_slots = effective - cpu_reserve - load
    available = snapshot['memory_available_bytes']
    memory_reserve = max(8 * 1024**3, math.ceil(.1 * available))
    per_worker = max(4 * 1024**3, math.ceil(1.5 * max(peaks)))
    workers = min(16, cpu_slots, (available - memory_reserve) // per_worker)
    require(workers >= 1, 'Insufficient resources; no work launched')
    return dict(ccg=int(workers), reserve_cpu_slots=cpu_reserve, background_load_slots=load,
        memory_reserve_bytes=memory_reserve, per_worker_memory_bytes=per_worker,
        observed_peak_rss_bytes=max(peaks), scope='one joint pool across R3 and R10',
        caveat='Observed RSS is not a bound on future AP branch-and-bound memory')


def nested_success(value):
    if isinstance(value, dict):
        if 'status' in value:
            require(value['status'] == 'completed', f'Incomplete nested preflight evidence: {value["status"]}')
        if 'returncode' in value:
            require(value['returncode'] == 0, 'Nested preflight subprocess failed')
        for child in value.values():
            nested_success(child)
    elif isinstance(value, list):
        for child in value:
            nested_success(child)


def validate_smoke(result):
    require(result['status'] == 'completed', 'Adapter smoke did not complete')
    require(result['preflight_only'] is True and result['production_reuse_forbidden'] is True,
        'Smoke is not marked disposable')
    require(digest(result['summary_path']) == result['summary_sha256'], 'Raw adapter smoke summary changed')
    full = read(result['summary_path'])
    require(full['status'] == 'completed' and full['mode'] == 'preflight' and full['complete'],
        'Incomplete raw smoke summary')
    nested_success(full)
    candidate = full['disposable']
    ap, mp = candidate['single_center_ap'], candidate['mp']
    require(ap['solver_status'] == 2 and ap['solcount'] >= 1 and ap['sample_id'] == 0,
        'One-center AP was not solved optimally')
    require(mp['max_epochs'] == 2 and mp['epochs_evaluated'] == 2 and
        mp['controller_status'] == 'completed', 'Wrong disposable MP workload')
    for phase in (full['input_verification'], full['native_last_pool_replay'],
          candidate['ap'], candidate['mp_call'], candidate['incoming_pool_evaluation'],
          candidate['returned_pool_evaluation']):
          require(phase['status'] == 'completed', 'Incomplete nested smoke phase')
          seconds(phase['wall_time_s'])
    require(result['input_verification'] == full['input_verification'] and
        result['source_prefix'] == full['source_prefix'], 'Compact/raw smoke evidence mismatch')
    require(result['peak_rss_bytes'] > 0, 'Missing smoke RSS')


def classify(result):
    if result.get('status') == 'failed':
        return 'failed'
    if result.get('status') == 'nonconverged' and result.get('terminal_status') in ('iteration_cap', 'stagnation'):
        return 'nonconverged'
    if (result.get('terminal_status') == 'converged' and result.get('status') == 'completed'
            and result.get('converged') is True and result.get('complete') is True):
        return 'cached_only' if result.get('cached_only') is True else 'completed'
    raise ValueError('Unknown adapter completion status; no success substitution')


def worker_entry(args):
    os.environ.update(THREAD_ENV)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    manifest = verified_manifest(args.manifest, args.sha)
    matches = [j for j in manifest['jobs'] if j['id'] == args.job]
    require(len(matches) == 1, 'Unknown job')
    job = matches[0]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    envelope = dict(id=job['id'], R=job['R'], Ntotal=job['Ntotal'], seed=job['seed'], radius=job['radius'],
        status='failed', started_at_utc=utc(), manifest_sha256=args.sha, workers=manifest['workers'],
        numerical_threads=1, timing_clock='time.perf_counter', timing_unit='seconds',
        adapter_result=None, error=None)
    try:
        verify_hashes(job['input_sha256'])
        import continuation
        callback = continuation.preflight_job if args.mode == '_smoke' else continuation.run_job
        # Adapter owns this nonexistent subdirectory and private weight/pool copies.
        result = callback(job, str(Path(args.manifest).resolve()), str(out / 'adapter'))
        require(isinstance(result, dict), 'Adapter must return its JSON-safe result dict')
        envelope['adapter_result'] = result
        if args.mode == '_smoke':
            validate_smoke(result)
            envelope['status'] = 'completed'
        else:
            envelope['status'] = classify(result)
            seconds(result['current_timings']['wall_time_s'])
            require(type(result['mp_count']) is int and
                    job['expected']['old_mp_count'] <= result['mp_count'] <= 40, 'Total MP cap violated')
    except Exception:
        envelope.update(status='failed', error=traceback.format_exc())
    finally:
        try:
            require(digest(args.manifest) == args.sha, 'Manifest changed during worker')
            verify_sources(manifest)
            verify_hashes(job['input_sha256'])
            envelope['inputs_unchanged'] = True
        except Exception:
            envelope.update(status='failed', inputs_unchanged=False, integrity_error=traceback.format_exc())
        envelope.update(ended_at_utc=utc(), current_invocation_wall_time_s=time.perf_counter() - start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        write(out / 'result.json', envelope)
    return 1 if envelope['status'] == 'failed' else 0


def run_process(command, folder):
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    row = dict(command=command, cwd=str(ROOT), started_at_utc=utc(), returncode=None,
        status='running', stdout_path=str(folder / 'stdout.log'), stderr_path=str(folder / 'stderr.log'))
    try:
        with (folder / 'stdout.log').open('x') as stdout, (folder / 'stderr.log').open('x') as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                env={**os.environ, **THREAD_ENV, 'PYTHONDONTWRITEBYTECODE': '1'}, check=False)
        row.update(returncode=completed.returncode, status='completed' if completed.returncode == 0 else 'failed')
    except Exception:
        row.update(status='failed', error=traceback.format_exc())
    finally:
        row.update(ended_at_utc=utc(), wall_time_s=time.perf_counter() - start)
        for name in ('stdout', 'stderr'):
            path = folder / (name + '.log')
            if path.exists():
                row[name + '_sha256'] = digest(path)
        write(folder / 'process.json', row)
    return row


def launch_job(job, manifest_path, sha, output, smoke=False):
    folder = output / job['id']
    command = [sys.executable, '-B', '-u', str(ROOT / 'campaign.py'), '_smoke' if smoke else '_job',
               str(folder / 'worker'), '--manifest', str(manifest_path), '--sha', sha, '--job', job['id']]
    process = run_process(command, folder)
    row = {k: job[k] for k in ('id', 'R', 'Ntotal', 'seed', 'radius')}
    row.update(status='failed', process=process, current_invocation_wall_time_s=process['wall_time_s'],
               previous_ccg_wall_time_s=job['expected']['previous_ccg_wall_time_s'])
    result_path = folder / 'worker/result.json'
    try:
        result = read(result_path)
        require(result['id'] == job['id'] and result['manifest_sha256'] == sha, 'Worker identity mismatch')
        row.update(worker_result_path=str(result_path), worker_result_sha256=digest(result_path), worker=result)
        if not smoke and result['adapter_result'] is not None:
            adapter = result['adapter_result']
            current = seconds(adapter['current_timings']['wall_time_s'])
            row.update(current_ccg_wall_time_s=current, mp_count=adapter['mp_count'],
                terminal_status=adapter['terminal_status'],
                full_model_active_phase_time_s=row['previous_ccg_wall_time_s'] + current,
                active_phase_scope='persisted original CCG wall plus current CCG; not fresh elapsed wall')
        require(process['returncode'] == 0 and result['inputs_unchanged'], 'Worker failed/integrity failure')
        row['status'] = result['status']
    except Exception:
        row.update(status='failed', error=traceback.format_exc())
    return row


def aggregates(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row['R'], row['Ntotal'], row.get('radius'), row['status']].append(row)
    result = []
    for (r, n, radius, status), group in sorted(groups.items(), key=lambda item: str(item[0])):
        values = [seconds(row[field]) for row in group if row.get(field) is not None]
        result.append(dict(R=r, Ntotal=n, radius=radius, status=status, field=field,
            n=len(group), measured_n=len(values), unavailable_n=len(group) - len(values),
            mean_s=statistics.mean(values) if values else None,
            median_s=statistics.median(values) if values else None,
            min_s=min(values) if values else None, max_s=max(values) if values else None))
    return result


def run_all(jobs, workers, launch, publish):
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(launch, job): job for job in schedule(jobs)}
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception:
                row = {k: job[k] for k in ('id', 'R', 'Ntotal', 'seed', 'radius')}
                row.update(status='failed', current_invocation_wall_time_s=None,
                           error=traceback.format_exc(), timing_unavailable='escaped launcher; see process artifact')
            rows.append(row)
            publish(rows)
    return rows


def report_runtime(output, manifest, rows, elapsed):
    fields = ('current_invocation_wall_time_s', 'current_ccg_wall_time_s',
              'previous_ccg_wall_time_s', 'full_model_active_phase_time_s')
    tables = {field: aggregates(rows, field) for field in fields}
    tables['cached_black'] = aggregates([
        dict(row, status='cached_' + row['terminal_status'])
        for row in manifest['cached_black']], 'previous_ccg_wall_time_s')
    tables['cached_nominal_training'] = aggregates(manifest['cached_nominal'], 'historical_wall_time_s')
    write(output / 'runtime_aggregates.json', dict(tables=tables, sweep_elapsed_s=elapsed,
        numerical_threads=1, workers=manifest['workers'],
        evaluation_and_cv='not run; historical campaign phase timings in manifest.original_campaigns',
        SRO_FA='not part of this campaign; unavailable, rerun required for measurement'))
    lines = ['# Runtime accounting', '', f'Current sweep: {elapsed:.3f} s / {elapsed/60:.3f} min / {elapsed/3600:.3f} h.',
        'Concurrent model durations are never summed as sweep elapsed.', '',
        '| Phase | R | N | Radius | Status | n/measured | Mean s | Median s | Min s | Max s |',
        '|---|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    for name, table in tables.items():
        for row in table:
            values = ['unavailable' if row[k] is None else f'{row[k]:.3f}'
                      for k in ('mean_s', 'median_s', 'min_s', 'max_s')]
            lines.append(f'| {name} | {row["R"]} | {row["Ntotal"]} | {row["radius"]} | {row["status"]} | '
                         f'{row["n"]}/{row["measured_n"]} | ' + ' | '.join(values) + ' |')
    lines += ['', '| Additional phase | Runtime |', '|---|---|',
        '| SRO-FA partition / solve / evaluation / total | Not run; unavailable |',
        '| New nominal training / validation / CV / final test | Not run |',
        '| Historical LR search / validation | Separate original campaign phase timings in manifest |',
        '', 'Unmeasured historical fields require a rerun; cached loading is not training time.']
    (output / 'RUNTIMES.md').write_text('\n'.join(lines) + '\n')


def validate_preflight(path, sha):
    require(digest(path) == sha, 'Preflight summary SHA mismatch')
    preflight = read(path)
    require(preflight['mode'] == 'preflight' and preflight['status'] == 'completed', 'Preflight not completed')
    manifest = verified_manifest(preflight['manifest_path'], preflight['manifest_sha256'])
    verify_hashes(manifest['input_sha256'])
    tests = preflight['tests']
    nested_success(tests)
    require(tests['status'] == 'completed' and tests['returncode'] == 0, 'Unit tests not passed')
    for stream in ('stdout', 'stderr'):
        require(digest(tests[stream + '_path']) == tests[stream + '_sha256'], 'Test log changed')
    rows = preflight['smokes']
    require(len(rows) == 6 and {(r['R'], r['Ntotal'], r['radius']) for r in rows} == set(SMOKES),
            'Incomplete smoke grid')
    for row in rows:
        require(row['seed'] == 0 and row['status'] == 'completed', 'Smoke identity/status mismatch')
        nested_success(row['process'])
        require(digest(row['worker_result_path']) == row['worker_result_sha256'], 'Smoke result changed')
        worker = read(row['worker_result_path'])
        require(worker == row['worker'] and worker['status'] == 'completed' and worker['inputs_unchanged']
                and worker['manifest_sha256'] == preflight['manifest_sha256'], 'Smoke worker binding mismatch')
        validate_smoke(worker['adapter_result'])
        for stream in ('stdout', 'stderr'):
            require(digest(row['process'][stream + '_path']) == row['process'][stream + '_sha256'],
                    'Smoke log changed')
    return preflight, manifest


def parent(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    summary = dict(experiment=PROTOCOL['experiment'], mode=args.mode, status='running',
        started_at_utc=utc(), command=sys.orig_argv, cwd=str(Path.cwd()),
        timing_clock='time.perf_counter', timing_unit='seconds', phases={}, jobs=[], smokes=[],
        tests={'status': 'not_run'}, counts=dict(total=570, scheduled=480, cached_black=90),
        numerical_threads=1, error=None)
    manifest = None
    try:
        freeze_start = time.perf_counter()
        if args.mode == 'production':
            preflight, manifest = validate_preflight(Path(args.preflight).resolve(), args.sha)
            # Preserve the entire preflight manifest verbatim as provenance as well
            # as the complete inventory in the new runtime manifest.
            original_bytes = Path(preflight['manifest_path']).read_bytes()
            (output / 'preflight_manifest.json').write_bytes(original_bytes)
            (output / 'preflight_manifest.json').chmod(0o444)
            require(digest(output / 'preflight_manifest.json') == preflight['manifest_sha256'], 'Copy changed')
            summary['preflight'] = dict(path=str(Path(args.preflight).resolve()), sha256=args.sha)
            environment = resource_snapshot(manifest)
            peaks = [r['worker']['peak_rss_bytes'] for r in preflight['smokes']]
            manifest['workers'] = choose_workers(environment, peaks)
            manifest['production_environment'] = environment
            manifest['preflight_manifest_sha256'] = preflight['manifest_sha256']
        else:
            manifest = build_manifest()
        manifest_path, sha = seal_manifest(output, manifest)
        summary.update(manifest_path=str(manifest_path), manifest_sha256=sha, workers=manifest['workers'],
            cached_black=manifest['cached_black'], environment=resource_snapshot(manifest))
        summary['phases']['freeze_or_verify_s'] = time.perf_counter() - freeze_start
        write(output / 'summary.json', summary)
        if args.mode == 'preflight':
            test_modules = [p.stem for p in sorted(ROOT.glob('test_*.py'))]
            require('test_campaign' in test_modules, 'Campaign tests missing')
            summary['tests'] = run_process([sys.executable, '-B', '-u', '-m', 'unittest', '-v', *test_modules],
                                           output / 'unit_tests')
            write(output / 'summary.json', summary)
            require(summary['tests']['status'] == 'completed', 'Unit tests failed; no real smoke launched')
            for r, n, radius in SMOKES:
                job = next(j for j in manifest['jobs'] if key(j) == (r, n, 0, radius))
                summary['smokes'].append(launch_job(job, manifest_path, sha, output / 'smokes', smoke=True))
                write(output / 'summary.json', summary)
            require(all(row['status'] == 'completed' for row in summary['smokes']), 'Real preflight failed')
        elif args.mode == 'production':
            pool_start = time.perf_counter()

            def publish(rows):
                summary['jobs'] = list(rows)
                summary['completed_job_count'] = len(rows)
                summary['status_counts'] = dict(Counter(row['status'] for row in rows))
                write(output / 'summary.json', summary)

            rows = run_all(manifest['jobs'], manifest['workers']['ccg'],
                lambda job: launch_job(job, manifest_path, sha, output / 'jobs'), publish)
            summary['phases']['joint_pool_elapsed_s'] = time.perf_counter() - pool_start
            exact_grid(rows, {key(j) for j in manifest['jobs']})
            summary['cohort_coverage'] = dict(total=570, executed=480, cached=90)
            summary['result_groups'] = {status: [r['id'] for r in rows if r['status'] == status]
                                        for status in ('completed', 'nonconverged', 'failed', 'cached_only')}
        verify_sources(manifest)
        verify_hashes(manifest['input_sha256'])
        require(digest(manifest_path) == sha, 'Manifest changed after execution')
        summary['inputs_unchanged'] = True
        statuses = {r['status'] for r in summary['jobs']}
        summary['status'] = 'failed' if 'failed' in statuses else ('nonconverged' if 'nonconverged' in statuses else 'completed')
    except Exception:
        summary.update(status='failed', error=traceback.format_exc())
    finally:
        elapsed = time.perf_counter() - start
        summary.update(ended_at_utc=utc(), wall_time_s=elapsed,
            wall_time_human=f'{elapsed:.3f} s / {elapsed/60:.3f} min / {elapsed/3600:.3f} h')
        write(output / 'summary.json', summary)
        if manifest is not None:
            try:
                report_runtime(output, manifest, summary['jobs'], elapsed)
            except Exception:
                summary.update(status='failed', runtime_report_error=traceback.format_exc())
                write(output / 'summary.json', summary)
        (output / 'summary.sha256').write_text(digest(output / 'summary.json') + '\n')
    return 1 if summary['status'] == 'failed' else (2 if summary['status'] == 'nonconverged' else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('freeze', 'preflight', 'production', '_job', '_smoke'):
        command = sub.add_parser(mode)
        command.add_argument('output')
        if mode == 'production':
            command.add_argument('--preflight', required=True)
            command.add_argument('--sha', required=True)
        if mode in ('_job', '_smoke'):
            command.add_argument('--manifest', required=True)
            command.add_argument('--sha', required=True)
            command.add_argument('--job', required=True)
    args = parser.parse_args()
    return worker_entry(args) if args.mode in ('_job', '_smoke') else parent(args)


if __name__ == '__main__':
    sys.exit(main())