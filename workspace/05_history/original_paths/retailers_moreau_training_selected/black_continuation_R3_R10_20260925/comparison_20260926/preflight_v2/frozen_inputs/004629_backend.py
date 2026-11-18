"""BLACK fixed-policy backend; no orchestration, rendering, or work on import.

Parent contract
---------------
source_paths() -> explicit local/evaluator source paths (no numerical imports).
collect_policies() -> JSON-safe {jobs, source_sha256, input_sha256,
    base_comparisons, selection_evidence, ...}; reads TRAINING evidence and
    original comparison manifests (which can contain historical OOS values).
    No test-array bytes are read and no OOS value enters checkpoint selection.
materialize(collection, folder) -> {original: {copy, sha256}}; exclusive copies
    of collection inputs, including original base manifests. Does not seal.

Parent owns the final manifest and its same-stem .sha256 seal. Required fields:
  status='frozen', policies_frozen_before_test=True, jobs (all 570),
  source_sha256, input_sha256, base_comparisons (unchanged from collection),
  files={absolute_original: {copy: absolute_copy, sha256: digest}},
  workers=actual integer concurrency in [1,32], threads=1,
  membership_tolerance=1e-9, physical_tolerance=1e-4,
  tests={'3': original_R3_test_metadata, '10': original_R10_test_metadata}.
The tests entries are the original base_comparisons[R]['test'] metadata;
parent copies/binds those arrays in files AFTER freezing ALL policies. It must
also copy/bind every added parent source in source_sha256. No regeneration.
training_smoke needs the same sealed full-cohort manifest but not tests/files
for test arrays. It evaluates exactly centers[0:2], never held-out data.

initialize_worker(R, manifest_path) is an importable per-process wrapper.
worker(job, manifest_path, output_folder) initializes itself if necessary,
evaluates 10,000 OOS pairs plus the selected policy's OWN finite pool, writes
output_folder/job['id']/result.json and raw QP journals, and returns its record
with result_path/summary_path and result_sha256/summary_sha256. Parent must
reject status != 'completed'. One R per fresh process; no bare module aliases.

Approved comparison-only cap rule: earliest exact minimum mean AP incumbent
over ALL retained + newly assessed candidates, including terminal candidate 40.
Nonconvergence remains explicit. No production artifact is modified. R3/N50
identities must match the existing training-only choices, not an OOS selector.
Historical nominal/prefix/continuation times are separate from this invocation.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from types import ModuleType


ROOT = Path(__file__).resolve().parent
CONTINUATION = ROOT.parent
BASE = CONTINUATION.parent
BASE_R3 = BASE / 'green_R3_N200_N800_20260923/comparison_20260924'
BASE_R10 = BASE / 'green_R10_N50_N200_N800_20260924/comparison_20260925'
BASE_MANIFESTS = {
    3: BASE_R3 / 'results/2026-09-24T091945.582854+0000/frozen/manifest.json',
    10: BASE_R10 / 'production_v1/frozen/manifest.json',
}
PRODUCTION = CONTINUATION / 'production_v1'
CHOICES = BASE / 'N50_R3_train40_val10/stopping_only_min_ap/results/training_only_choices.json'
COMPARE = BASE.parent / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison/compare.py'
LOCKS = {
    BASE_MANIFESTS[3]: '482f729123d52ad4c547817f309372c8d3ad00cc18bd63f4d7171e7925ffebcd',
    BASE_MANIFESTS[10]: '9f201b3ce4b2a0b88f2fb1ecb003d85a948ffeae3f6d795b0c3b8e6662ee53ec',
    PRODUCTION / 'manifest.json': '13e55cbad7f2bb9bd68a055ea29750e82560e7aff615c84cd8515956d4bfc0ca',
    PRODUCTION / 'summary.json': '211c48a02bcd08ff03a08202e7ecec72004c3635a9b63e9ca1125db14ad28c7b',
}
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
SELECTION = 'earliest minimum mean AP incumbent over complete retained-plus-new prefix including terminal40'
EXPERIMENT = 'black_fixed_policy_comparison_20260926'
_CONTEXT = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite numeric evidence')
    return value


def merge(target, incoming):
    for path, digest in incoming.items():
        require(Path(path).is_absolute() and isinstance(digest, str) and len(digest) == 64,
                f'Invalid absolute hash binding: {path}')
        require(path not in target or target[path] == digest, f'Conflicting hash: {path}')
        target[path] = digest


def source_paths():
    return sorted({ROOT / 'backend.py', ROOT / 'test_backend.py', COMPARE,
                   BASE_R3 / 'evaluation.py', BASE_R10 / 'evaluation.py',
                   BASE_R10 / 'reference.py', BASE_R10 / 'plot_comparison.py'})


class Capture:
    """Training-only inputs; shared hashes, per-policy explicit dependency set."""
    def __init__(self, expected=None):
        self.expected = dict(expected or {})
        self.hashes = {}
        self.used = set()

    def add(self, path, expected=None):
        path = str(Path(path).resolve())
        digest = sha(path)
        for bound in (expected, self.expected.get(path), self.hashes.get(path)):
            require(bound is None or bound == digest, f'Input hash mismatch: {path}')
        self.hashes[path] = digest
        self.used.add(path)
        return digest

    def read(self, path, expected=None):
        digest = self.add(path, expected)
        payload = Path(path).read_bytes()
        require(hashlib.sha256(payload).hexdigest() == digest, f'Input changed while reading: {path}')
        return json.loads(payload)


def cohort(jobs):
    expected = {(r, n, s, rho) for r in (3, 10) for n in (50, 200, 800)
                for s in range(10) for rho in RADII + ((2.,) if r == 10 else ())}
    actual = [(j['R'], j['Ntotal'], j['seed'], j['radius']) for j in jobs]
    require(len(jobs) == len({j['id'] for j in jobs}) == 570 and set(actual) == expected,
            'Require exact 570-policy cohort; no missing, duplicate, or filtered seeds')


def choose(prefix):
    require(bool(prefix), 'Empty assessed prefix')
    for i, candidate in enumerate(prefix):
        require(candidate['candidate_id'] == i and candidate['assessed'] is True,
                'Selection requires a complete contiguous assessed prefix')
        finite(candidate['ap_incumbent_mean'])
    return min(prefix, key=lambda candidate: candidate['ap_incumbent_mean'])


def joined_prefix(original, record):
    """Original assessments stay immutable; boundary replaces only outgoing MP."""
    prefix = list(original['iterations']) if original is not None else []
    boundary = record.get('extension_boundary')
    if prefix:
        require(original['mp_count'] == original['stopping_candidate_id'] == len(prefix) - 1,
            'Original stopping prefix/count mismatch')
        require(boundary is not None, 'Missing continuation boundary')
        require(all(boundary.get(k) == v for k, v in prefix[-1].items()),
                'Boundary changed original assessment/checkpoint/own pool')
        outgoing = {'mp', 'mp_call', 'added_sample_ids', 'incoming_pool_evaluation',
                'returned_pool_evaluation'}
        require(not outgoing.intersection(prefix[-1]) and
            set(boundary) - set(prefix[-1]) == (outgoing if record['new_iterations'] else set()),
            'Unexpected continuation boundary fields')
    else:
        require(boundary is None, 'Unexpected boundary without original prefix')
    prefix += record['new_iterations']
    choose(prefix)
    require(len(prefix) == record['mp_count'] + 1 and
            prefix[-1]['candidate_id'] == record['stopping_candidate_id'] and
            record['new_mp_count'] == record['mp_count'] - (len(original['iterations']) - 1
                                                          if original else 0),
            'Continuation MP count/prefix mismatch')
    chain = list(prefix)
    if original:
        chain[len(original['iterations']) - 1] = boundary
    return prefix, chain


def pool_digest(pools):
    import numpy as np
    digest = hashlib.sha256()
    for i, pool in enumerate(pools):
        digest.update(np.asarray([i, len(pool)], dtype='<i8').tobytes())
        digest.update(np.asarray(pool, dtype='<f8').tobytes(order='C'))
    return digest.hexdigest()


def validate_pools(pools, centers, radius):
    import numpy as np
    require(len(pools) == len(centers) > 0, 'Pool/center coverage mismatch')
    arrays = []
    for pool, center in zip(pools, centers):
        a = np.asarray(pool, dtype=np.float64)
        require(a.ndim == 2 and a.shape[1] == len(center) and len(a) > 0 and
                np.isfinite(a).all() and np.array_equal(a[0], center), 'Malformed own pool')
        residual = np.maximum(np.maximum(np.maximum(0, center - radius) - a,
                                         a - center - radius), 0).max()
        require(residual <= 1e-9, 'Pool violates original membership gate')
        arrays.append(a)
    return arrays


def assessment_equal(candidate, disk):
    a, b = dict(candidate), dict(disk)
    # Original _flush refreshes this field after the assessment write. The
    # continuation uses save(), not _flush: its new candidates omit it on both
    # sides. Do not normalize missing fields or arbitrary timing differences.
    require(set(a) == set(b), 'Assessment schema differs from full summary')
    if 'mp_total_time_s' in a:
        expected = finite(a['mp_call']['wall_time_s']) if 'mp_call' in a else 0.
        require(expected >= 0 and finite(a['mp_total_time_s']) == expected and
                finite(b['mp_total_time_s']) in (0., expected), 'Unexplained MP timing difference')
        a['mp_total_time_s'] = b['mp_total_time_s'] = expected
    require(a == b, 'Assessment differs from full summary')


def candidate_rows(candidate, job, centers, ids, pools):
    """No AP rerun: verify accepted raw rows, identities, bounds, and mean."""
    import numpy as np
    rows = candidate['samples']
    require(job['R'] in (3, 10) and centers.shape == (job['Ntrain'], 2 * job['R']) and
            len(pools) == len(ids) == job['Ntrain'], 'AP training dimensions mismatch')
    require(candidate['assessed'] is True and not candidate['failed_sample_ids'] and
            len(rows) == len(ids) == len(candidate['pool_costs']) and
            [r['sample_id'] for r in rows] == list(range(len(ids))), 'Incomplete AP coverage')
    legacy = job['R'] == 3 and job['Ntotal'] == 50
    # Original GREEN candidates do not persist R. Bind it through the locked
    # job/summary/protocol, AP metadata and stage dimensions; checkpoint tensor
    # dimensions are independently enforced by the original worker loader.
    for key in ('seed', 'radius') + (() if legacy else ('Ntotal', 'Ntrain')):
        require(candidate[key] == job[key], f'Candidate identity mismatch: {key}')
    if 'R' in candidate:
        require(candidate['R'] == job['R'], 'Candidate identity mismatch: R')
    if not legacy:
        require(candidate['ap']['R'] == job['R'], 'AP metadata retailer dimension mismatch')
    closed, points = True, []
    for i, row in enumerate(rows):
        require(all(len(row[key]) == job['R'] for key in ('center1', 'center2')) and
                all(len(row['incumbent'][key]) == job['R'] for key in ('worst1', 'worst2')),
                'AP stage dimensions mismatch')
        require(row['status'] == row['controller_status'] == 'completed' and
                row['solver_status'] == 2 and row['solcount'] >= 1, 'AP not accepted OPTIMAL')
        require(row['original_sample_id'] == ids[i] and
                row['candidate_id'] == candidate['candidate_id'] and
                row['candidate_sha256'] == candidate['state_sha256'] and
                row['checkpoint_sha256'] == candidate['checkpoint_sha256'] and
                row['pool_sha256'] == candidate['pool_sha256'] and
                row['seed'] == job['seed'] and row['radius'] == job['radius'] and
                np.array_equal(row['center1'] + row['center2'], centers[i]), 'AP identity mismatch')
        if not legacy:
            require(row['Ntrain'] == job['Ntrain'] and row['Ntotal'] == job['Ntotal'], 'AP size mismatch')
        bound, pool, inc = finite(row['bound']), finite(row['pool_cost']), finite(row['incumbent']['cost'])
        require(pool == candidate['pool_costs'][i], 'AP own-pool cost mismatch')
        residual = max(0., bound - pool) / max(1., abs(bound), abs(pool))
        require(abs(residual - finite(row['residual'])) <= 1e-12, 'Saved bound residual mismatch')
        closed = closed and residual <= .01
        point = np.asarray(row['incumbent']['worst1'] + row['incumbent']['worst2'], dtype=np.float64)
        require(point.shape == centers[i].shape and np.isfinite(point).all(), 'Malformed raw AP point')
        violation = float(np.maximum(np.maximum(np.maximum(0, centers[i] - job['radius']) - point,
                                    point - (centers[i] + job['radius'])), 0).max())
        require(violation <= 1e-9 and violation == row['membership_violation'], 'AP membership mismatch')
        duplicate = bool(np.any(np.all(pools[i] == point, axis=1)))
        violating = inc - pool > 1e-4 * max(1., abs(inc), abs(pool))
        require(duplicate == row['membership']['exact_duplicate'] and
                violating == row['incumbent_violating'], 'AP insertion gate mismatch')
        points.append(point)
    for name, values in (('ap_incumbent_mean', [r['incumbent']['cost'] for r in rows]),
                         ('ap_bound_mean', [r['bound'] for r in rows]),
                         ('restricted_mean', candidate['pool_costs'])):
        require(math.isclose(sum(map(finite, values)) / len(ids), finite(candidate[name]),
                             rel_tol=1e-12, abs_tol=1e-12), f'Saved mean mismatch: {name}')
    require(candidate['separation_status'] == ('separation_satisfied' if closed else 'violating'),
            'Saved separation flag mismatch')
    return closed, points


def transition(pools, points, rows):
    import numpy as np
    result = [p.copy() for p in pools]
    added = []
    for i, (point, row) in enumerate(zip(points, rows)):
        if row['incumbent_violating'] and not row['membership']['exact_duplicate']:
            result[i] = np.vstack((result[i], point))
            added.append(i)
    return result, added


def validate_prefix(cap, prefix, chain, job, training, continuation_path=None, original_count=0):
    """Verify all candidate pools and AP/MP links, including capped terminal40."""
    import numpy as np
    centers = np.asarray(training['centers'], dtype=np.float64)
    ids, n, r = training['original_sample_ids'], job['Ntotal'], job['R']
    require(centers.shape == (4*n//5, 2*r) and np.isfinite(centers).all() and
            (centers >= 0).all() and len(ids) == len(centers) and len(set(ids)) == len(ids) and
            all(type(i) is int and 0 <= i < n for i in ids), 'Invalid training-only centers/IDs')
    require(len(chain) == len(prefix), 'Truncated outgoing MP chain')
    selected = choose(prefix)
    previous = None
    selected_pool = None
    evidence, timings = [], []
    for i, (a, outgoing) in enumerate(zip(prefix, chain)):
        cap.add(a['checkpoint'], a['checkpoint_sha256'])
        assessment_equal(a, cap.read(a['ap_reference']))
        path = Path(a['ap_reference']).parent / 'pool.json'
        saved = cap.read(path, a['pool_file_sha256'])
        require(saved['original_sample_ids'] == ids, 'Own-pool original sample IDs changed')
        pools = validate_pools(saved['pools'], centers, job['radius'])
        require(pool_digest(pools) == saved['pool_sha256'] == a['pool_sha256'], 'Own-pool digest mismatch')
        closed, points = candidate_rows(a, job, centers, ids, pools)
        if i == 0:
            require(all(np.array_equal(p, c[None, :]) for p, c in zip(pools, centers)),
                    'Initial pool is not center-only')
        else:
            require(previous['returned_state_sha256'] == a['state_sha256'] and
                    previous['selected_checkpoint_sha256'] == a['checkpoint_sha256'] and
                    previous['pool_sha256'] == a['pool_sha256'] and
                    abs(previous['selected_training_loss'] - a['restricted_mean']) <= 1e-6,
                    'MP-to-candidate chain mismatch')
        if a is selected:
            selected_pool = path, saved
        evidence.append({k: a[k] for k in ('candidate_id', 'checkpoint', 'checkpoint_sha256',
            'state_sha256', 'ap_reference', 'pool_file_sha256', 'pool_sha256', 'ap_incumbent_mean',
            'restricted_mean', 'separation_status')})
        phase = dict(candidate_id=i, origin='original' if i < original_count else 'continuation',
                     ap=deepcopy(a['ap']), pool_evaluation=deepcopy(a['evaluation']))
        if i < len(prefix) - 1:
            mp = outgoing['mp']
            # Every newly executed MP, not only the boundary MP, lives here.
            if continuation_path is not None and i >= max(0, original_count - 1):
                folder = Path(continuation_path).parent / 'extension_boundary' / f'candidate_{i:02d}' / 'mp'
            else:
                folder = Path(a['ap_reference']).parent / 'mp'
            require(cap.read(folder / 'controller_training.json') == mp, 'MP controller artifact mismatch')
            native = cap.read(folder / 'training.json')
            require(all(k in mp and mp[k] == v for k, v in native.items()), 'Native MP artifact mismatch')
            require(mp['status'] == mp['controller_status'] == 'completed' and
                    mp['mp_iteration'] == mp['output_candidate_id'] == i + 1 and
                    mp['incoming_state_sha256'] == a['state_sha256'], 'MP status/state mismatch')
            require(mp['max_epochs'] == 3000 and mp['patience'] == 300 and
                    mp['lr_net'] == job['lr_net'] and
                    mp['torch_threads'] == mp['torch_interop_threads'] == mp['gurobi_threads'] == 1,
                    'Changed native MP configuration')
            require(all(mp[k] == job[k] for k in ('Ntotal', 'Ntrain', 'seed', 'radius')) and
                    (job['R'] == 3 and job['Ntotal'] == 50 or mp['R'] == job['R']),
                    'MP job identity mismatch')
            inserted, added = transition(pools, points, a['samples'])
            require(added and added == outgoing['added_sample_ids'] and
                    pool_digest(inserted) == mp['pool_sha256'], 'Exact AP insertion/MP chain mismatch')
            mp_pool = cap.read(folder / 'pool.json', mp['pool_file_sha256'])
            require(mp_pool['original_sample_ids'] == ids and mp_pool['pool_sha256'] == mp['pool_sha256'] and
                    pool_digest(mp_pool['pools']) == mp['pool_sha256'], 'MP pool artifact mismatch')
            cap.add(folder / 'selected.pth', mp['selected_checkpoint_sha256'])
            first, best, loss = (finite(mp[k]) for k in ('first_train', 'best_train', 'selected_training_loss'))
            require(best <= first and (best == first or best < first - 1e-4) and
                    abs(best - loss) <= 1e-6 and
                    abs(first - outgoing['incoming_pool_evaluation']['mean']) <= 1e-6 and
                    abs(loss - outgoing['returned_pool_evaluation']['mean']) <= 1e-6,
                    'MP selected training-loss contract mismatch')
            previous = mp
            phase.update(mp_origin='continuation' if continuation_path is not None and
                         i >= max(0, original_count - 1) else 'original',
                         mp=deepcopy(outgoing['mp_call']), epochs=mp['epochs_evaluated'],
                         incoming_pool_evaluation=deepcopy(outgoing['incoming_pool_evaluation']),
                         returned_pool_evaluation=deepcopy(outgoing['returned_pool_evaluation']))
        else:
            stopping_closed = closed
            require(not {'mp', 'mp_call'}.intersection(outgoing), 'Terminal candidate has an outgoing MP')
        for name in ('ap', 'pool_evaluation', 'mp', 'incoming_pool_evaluation', 'returned_pool_evaluation'):
            if name in phase:
                require(phase[name]['status'] == 'completed' and finite(phase[name]['wall_time_s']) >= 0,
                        f'Missing/failed historical timing: {name}')
        timings.append(phase)
    require(selected_pool is not None, 'Selected own pool missing')
    return selected, selected_pool, evidence, timings, stopping_closed


def _published_identity(selected, published, legacy=False):
    keys = {'candidate_id': 'selected_iteration' if legacy else 'selected_candidate_id',
            'checkpoint': 'checkpoint' if legacy else 'selected_checkpoint',
            'checkpoint_sha256': 'checkpoint_sha256' if legacy else 'selected_checkpoint_sha256',
            'state_sha256': 'state_sha256' if legacy else 'selected_state_sha256'}
    require(all(selected[a] == published[b] for a, b in keys.items()), 'Published policy identity mismatch')
    if not legacy:
        require(selected['ap_reference'] == published['selected_ap_reference'], 'Published AP identity mismatch')


def collect_policies():
    """No numerical evaluator/model imports; no test-array reads or OOS selection."""
    started, start = utc(), time.perf_counter()
    cap = Capture({str(p): d for p, d in LOCKS.items()})
    production = cap.read(PRODUCTION / 'summary.json')
    manifest = cap.read(PRODUCTION / 'manifest.json')
    require(production['inputs_unchanged'] and production['status'] == 'nonconverged' and
            production['manifest_sha256'] == LOCKS[PRODUCTION / 'manifest.json'], 'Production integrity/status mismatch')
    require(Counter(j['status'] for j in production['jobs']) ==
            Counter(cached_only=209, completed=261, nonconverged=10), 'Production cohort status drift')
    merge(cap.expected, manifest['input_sha256'])
    sources = dict(manifest['source_sha256'])
    bases = {}
    for r, path in BASE_MANIFESTS.items():
        base = cap.read(path)
        require(base['status'] == 'frozen' and base['n_test'] == 10000 and base['test_seed'] == 9999,
                'Original comparison protocol mismatch')
        # Preserve original copy metadata, but do not read unrelated caches/test arrays.
        bases[str(r)] = dict(manifest_path=str(path), manifest_sha256=LOCKS[path],
                            versions=base['versions'], test=base['test'], files=base['files'])
        for source, info in base['files'].items():
            if Path(source).suffix == '.py':
                merge(sources, {source: info['sha256']})
                require(sha(info['copy']) == info['sha256'], f'Original frozen source copy changed: {source}')
    for source, digest in list(sources.items()):
        cap.add(source, digest)
    for path in source_paths():
        merge(sources, {str(path): cap.add(path, sources.get(str(path)))})
    choice_data = cap.read(CHOICES)
    require(choice_data['validation_used_for_candidate_selection'] is False and
            choice_data['test_used_for_candidate_selection'] is False, 'Unsafe cached choices')
    choices = {(c['seed'], c['radius']): c for c in choice_data['choices'] if c['color'] == 'black'}
    require(len(choices) == 90, 'Incomplete cached black choices')
    originals = {j['id']: j for j in manifest['jobs']}
    require(len(originals) == 480 and len(manifest['cached_black']) == 90, 'Incomplete production inventory')
    require(production['cached_black'] == manifest['cached_black'], 'Cached cohort changed in production summary')
    jobs = []
    common_inputs = {str(PRODUCTION / 'summary.json'), str(PRODUCTION / 'manifest.json'), str(CHOICES),
                     *(str(p) for p in BASE_MANIFESTS.values())}
    entries = [(row, False) for row in production['jobs']] + [(row, True) for row in manifest['cached_black']]
    for entry_number, (row, legacy) in enumerate(sorted(entries, key=lambda pair: pair[0]['id']), 1):
        print(f'Collecting {entry_number}/570: {row["id"]}', flush=True)
        cap.used = set(common_inputs)
        job = {k: row[k] for k in ('id', 'R', 'Ntotal', 'seed', 'radius')}
        job['Ntrain'] = 4 * job['Ntotal'] // 5
        if legacy:
            record_path = row['source_summary_path']
            record = cap.read(record_path, row['source_summary_sha256'])
            require(record['status'] == 'completed', 'Incomplete cached raw black record')
            prefix = chain = record['iterations']
            require(len(prefix) == record['mp_count'] + 1, 'Cached black full prefix mismatch')
            original, original_count, continuation_path = record, len(prefix), None
            training_path = Path(record_path).parent / 'inputs/training_data.json'
            training_sha = record['training_data_sha256']
            nominal = record['cached_nominal']
        else:
            original_job = originals[row['id']]
            require(all(original_job[k] == job[k] for k in ('R', 'Ntotal', 'seed', 'radius')), 'Production job identity mismatch')
            require(manifest['sourceconfig'][str(job['R'])]['protocol']['R'] == job['R'],
                    'Bound original protocol retailer dimension mismatch')
            envelope = cap.read(row['worker_result_path'], row['worker_result_sha256'])
            require(envelope == row['worker'] and envelope['inputs_unchanged'] and
                    envelope['manifest_sha256'] == LOCKS[PRODUCTION / 'manifest.json'], 'Worker envelope mismatch')
            compact = envelope['adapter_result']
            record_path = compact['summary_path']
            record = cap.read(record_path, compact['summary_sha256'])
            require(all(record[k] == v for k, v in compact.items() if k != 'summary_sha256'),
                    'Compact/full continuation mismatch')
            require(record['mode'] == 'job' and record['heldout_access'] is False, 'Invalid continuation record')
            source = record['source_prefix']
            require(source['summary_path'] == original_job['source_summary_path'] and
                    source['summary_sha256'] == original_job['source_summary_sha256'], 'Original-prefix reference mismatch')
            original = cap.read(source['summary_path'], source['summary_sha256']) if source['summary_path'] else None
            require(original is not None or job['R'] == 10 and job['radius'] == 2., 'Unexpected missing original prefix')
            require(source['immutable'] is True and
                    source['candidates'] == (len(original['iterations']) if original else 0),
                    'Original-prefix coverage mismatch')
            prefix, chain = joined_prefix(original, record)
            original_count = len(original['iterations']) if original else 0
            continuation_path = record_path
            winner = original_job['nominal_record']
            training_path, training_sha = winner['training_data_path'], winner['training_data_sha256']
            nominal = record['historical_nominal']
        require(all(record[k] == job[k] for k in ('Ntotal', 'seed', 'radius')), 'Raw record identity mismatch')
        if not legacy:
            require(record['R'] == job['R'] and record['Ntrain'] == job['Ntrain'] and
                record['lr_net'] == winner['lr_net'], 'Continuation dimension/nominal winner mismatch')
        job['lr_net'] = record['lr_net']
        training = cap.read(training_path, training_sha)
        require(set(training) == {'centers', 'original_sample_ids', 'data_sha256'}, 'Unexpected training-only schema')
        require(nominal['training_timing']['status'] == 'completed' and
            finite(nominal['training_timing']['wall_time_s']) >= 0, 'Missing nominal training runtime')
        if not legacy:
            require(training['data_sha256'] == winner['data_sha256'], 'Saved nominal training-data mismatch')
        if original is not None:
            require(training['data_sha256'] == original['data_sha256'], 'Training raw-data binding mismatch')
            if not legacy:
                require(original['R'] == job['R'] and original['status'] == 'completed' and
                        original['Ntotal'] == job['Ntotal'] and original['Ntrain'] == job['Ntrain'] and
                        original['seed'] == job['seed'] and original['radius'] == job['radius'] and
                        original['lr_net'] == job['lr_net'] and
                        original['training_data_sha256'] == training_sha and
                        original['terminal_status'] == 'green_gap_satisfied' and
                        original['stopping_candidate_id'] == len(original['iterations']) - 1,
                        'Invalid retained original GREEN prefix')
        selected, (pool_path, pool), evidence, timing_rows, closed = validate_prefix(
            cap, prefix, chain, job, training, continuation_path, original_count)
        terminal = record['terminal_status']
        require(terminal in ('converged', 'separation_satisfied', 'iteration_cap'), 'Unapproved terminal status')
        converged = terminal != 'iteration_cap'
        require(closed == converged and (converged or prefix[-1]['candidate_id'] == 40), 'Stop closure/cap mismatch')
        if legacy:
            choice = choices[job['seed'], job['radius']]
            _published_identity(selected, row, legacy=True)
            _published_identity(selected, choice, legacy=True)
            require(choice['source_summary_path'] == record_path and choice['mp_count'] == record['mp_count'] and
                    choice['stop_candidate'] == prefix[-1]['candidate_id'] and choice['terminal_status'] == terminal,
                    'Cached training-only choice prefix mismatch')
        elif converged:
            require(record['converged'] is True and record['status'] == 'completed', 'Invalid converged record')
            _published_identity(selected, record)
        else:
            require(record['converged'] is False and record['status'] == 'nonconverged' and
                    record['selected_candidate_id'] is None, 'Production cap unexpectedly published a selection')
        if original is not None:
            require(prefix[0]['checkpoint_sha256'] == original['nominal_checkpoint_sha256'] and
                    prefix[0]['state_sha256'] == original['nominal_state_sha256'], 'Nominal candidate identity mismatch')
        else:
            require(prefix[0]['checkpoint_sha256'] == winner['checkpoint_sha256'] and
                    prefix[0]['state_sha256'] == winner['state_sha256'], 'Fresh rho2 nominal identity mismatch')
        job.update(method='black', training_backend='Moreau', Nval=job['Ntotal']//5,
            network_seed=0, split_seed=100000 + job['seed'], checkpoint=selected['checkpoint'],
            checkpoint_sha256=selected['checkpoint_sha256'], state_sha256=selected['state_sha256'],
            selected_candidate_id=selected['candidate_id'], stopping_candidate_id=prefix[-1]['candidate_id'],
            selected_separation_status=selected['separation_status'], stopping_separation_status=prefix[-1]['separation_status'],
            historically_converged=converged, converged=converged, terminal_status=terminal,
            historical_status='retained_converged' if converged else 'retained_capped_nonconverged',
            capped_nonconverged=not converged, retained=True, selection_rule=SELECTION,
            comparison_derived_cap_selection=not legacy and not converged,
            mp_count=record['mp_count'], new_mp_count=0 if legacy else record['new_mp_count'],
            retained_candidate_count=original_count, assessed_candidate_count=len(prefix),
            ap_reference=selected['ap_reference'], pool_path=str(pool_path),
            pool_file_sha256=selected['pool_file_sha256'], pool_sha256=selected['pool_sha256'],
            sizes=[len(p) for p in pool['pools']], centers=training['centers'],
            original_sample_ids=training['original_sample_ids'], training_data_path=str(training_path),
            training_data_sha256=training_sha, data_sha256=training['data_sha256'],
            source_summary_path=record_path, source_summary_sha256=cap.hashes[str(record_path)],
            selected_ap_incumbent_mean=selected['ap_incumbent_mean'],
            saved_native_finite_pool_mean=selected['restricted_mean'], selection_evidence=evidence,
            input_files=sorted(cap.used), input_sha256={p: cap.hashes[p] for p in sorted(cap.used)},
            training_runtime_reference=dict(status='persisted_not_current', nominal=nominal,
                original_prefix=None if original is None else {k: original.get(k) for k in
                    ('summary_path', 'wall_time_s', 'ap_total_time_s', 'mp_total_time_s', 'evaluation_time_s',
                     'started_at_utc', 'ended_at_utc', 'ccg_workers', 'environment')},
                continuation=None if legacy else {k: record.get(k) for k in
                    ('current_timings', 'started_at_utc', 'ended_at_utc', 'ccg_workers', 'cached_only', 'environment')},
                iterations=timing_rows, current_training_time_s=0.,
                note='Historical active phases are not current evaluation or fresh end-to-end sweep time; '
                     'missing historical fields are unavailable and require rerun.'))
        jobs.append(job)
    print('All 570 training-only choices validated', flush=True)
    cohort(jobs)
    require(sum(j['comparison_derived_cap_selection'] for j in jobs) == 10 and
            sum(j['capped_nonconverged'] for j in jobs) == 11, 'Cap retention inventory mismatch')
    return dict(experiment=EXPERIMENT, status='policies_collected', jobs=jobs,
        source_sha256=sources, input_sha256=cap.hashes, base_comparisons=bases,
        selection_evidence=sorted(common_inputs), selection_rule=SELECTION,
        policies_frozen_before_test=True, test_bytes_accessed=False, test_evaluated=False,
        new_training=False, radius_selection=False, lr_selection=False,
        membership_tolerance=1e-9, physical_tolerance=1e-4, threads=1,
        collection_timing=dict(start_utc=started, end_utc=utc(), wall_time_s=time.perf_counter()-start,
                               clock='time.perf_counter', status='completed'))


def materialize(collection, folder):
    """Copy every collected dependency, exclusively; parent still owns sealing."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    files = {}
    bindings = dict(collection['input_sha256'])
    merge(bindings, collection['source_sha256'])
    for i, (source, digest) in enumerate(sorted(bindings.items())):
        require(sha(source) == digest, f'Source changed before copying: {source}')
        destination = folder / f'{i:06d}_{Path(source).name}'
        with Path(source).open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        require(sha(source) == sha(destination) == digest, f'Source changed during copy: {source}')
        destination.chmod(0o444)
        files[source] = dict(copy=str(destination), sha256=digest)
    return files


def _bound(manifest, source, expected=None):
    source = str(Path(source).resolve())
    info = manifest['files'][source]
    require(Path(info['copy']).is_absolute() and Path(info['copy']).resolve() != Path(source), 'Require private copy')
    require(expected is None or info['sha256'] == expected, f'Binding mismatch: {source}')
    require(sha(source) == sha(info['copy']) == info['sha256'], f'Original/copy changed: {source}')
    return Path(info['copy'])


def _sealed(path):
    path = Path(path).resolve()
    digest = sha(path)
    require(path.with_suffix('.sha256').read_text().strip() == digest, 'Manifest seal mismatch')
    manifest = read(path)
    require(sha(path) == digest, 'Manifest changed while reading')
    require(manifest['status'] == 'frozen' and manifest['policies_frozen_before_test'] is True,
            'All policies must be frozen before evaluation')
    require(manifest['threads'] == 1 and manifest['physical_tolerance'] == 1e-4 and
            manifest['membership_tolerance'] == 1e-9 and type(manifest['workers']) is int and
            1 <= manifest['workers'] <= 32, 'Changed numerical/concurrency contract')
    cohort(manifest['jobs'])
    return manifest, digest


def _verify_sources(manifest):
    require({str(p) for p in source_paths()} <= set(manifest['source_sha256']), 'Unbound backend/evaluator source')
    for source, digest in manifest['source_sha256'].items():
        _bound(manifest, source, digest)


def _private_module(path, name, plot=None):
    """Full original namespace; only the bare plot import is privately rebound."""
    module = ModuleType(name)
    module.__file__ = str(path)
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    count = 0
    for i, node in enumerate(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == 'plot_comparison':
            require(plot is not None and len(node.names) == 1 and
                    node.names[0].name == 'validate_cache_protocol', 'Unexpected evaluator plot import')
            alias = node.names[0]
            module.__dict__['_private_plot'] = plot
            tree.body[i] = ast.copy_location(ast.Assign(
                targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                value=ast.Attribute(value=ast.Name(id='_private_plot', ctx=ast.Load()),
                                    attr=alias.name, ctx=ast.Load())), node)
            count += 1
    require(count == (1 if plot is not None else 0), 'Private import replacement coverage mismatch')
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module


def initialize_worker(R, manifest_path):
    """Top-level spawn wrapper; never replace sys.modules or add campaign paths."""
    global _CONTEXT
    require(type(R) is int and R in (3, 10), 'Require R3/R10')
    manifest, digest = _sealed(manifest_path)
    _verify_sources(manifest)
    if _CONTEXT is not None:
        require(_CONTEXT[0] == R and _CONTEXT[1] == digest, 'Fresh process required for another R/manifest')
        e = _CONTEXT[2]
    else:
        os.environ.update(THREAD_ENV)
        sys.dont_write_bytecode = True
        base = BASE_R3 if R == 3 else BASE_R10
        plot = _private_module(base / 'plot_comparison.py', '_black_private_plot_R10') if R == 10 else None
        e = _private_module(base / 'evaluation.py', f'_black_private_evaluation_R{R}', plot)
        e.initialize_worker()
        _CONTEXT = R, digest, e
    require(e.c.torch.get_num_threads() == e.c.torch.get_num_interop_threads() == 1,
            'Numerical worker thread contract changed')
    base = manifest['base_comparisons'][str(R)]
    require(base['manifest_path'] == str(BASE_MANIFESTS[R]) and base['manifest_sha256'] == LOCKS[BASE_MANIFESTS[R]],
            'Original version-lock manifest changed')
    original = read(_bound(manifest, base['manifest_path'], base['manifest_sha256']))
    require(e.versions() == base['versions'] == original['versions'], 'Original evaluator versions changed')
    require(e.c.PHYSICAL_TOL == 1e-4, 'Physical projection tolerance changed')
    return e


def _own_inputs(job, manifest, e):
    require([j for j in manifest['jobs'] if j['id'] == job['id']] == [job], 'Policy differs from sealed manifest')
    for source in job['input_files']:
        expected = job['input_sha256'][source]
        require(manifest['input_sha256'][source] == expected, 'Policy/manifest input hash mismatch')
        _bound(manifest, source, expected)
    checkpoint = _bound(manifest, job['checkpoint'], job['checkpoint_sha256'])
    model = e.c.load_model(checkpoint, 'new')
    require(e.c.state_hash(model) == job['state_sha256'], 'Selected learned state mismatch')
    training = read(_bound(manifest, job['training_data_path'], job['training_data_sha256']))
    require(training['centers'] == job['centers'] and training['original_sample_ids'] == job['original_sample_ids'] and
            training['data_sha256'] == job['data_sha256'], 'Training-center mapping changed')
    pool = read(_bound(manifest, job['pool_path'], job['pool_file_sha256']))
    require(pool['original_sample_ids'] == job['original_sample_ids'], 'Selected own-pool IDs mismatch')
    centers = e.np.asarray(job['centers'], dtype=e.np.float64)
    require(centers.shape == (job['Ntrain'], 2*job['R']), 'Training-center dimensions changed')
    pools = validate_pools(pool['pools'], centers, job['radius'])
    require(pool_digest(pools) == pool['pool_sha256'] == job['pool_sha256'] and
            [len(p) for p in pools] == job['sizes'], 'Selected own-pool sizes/digest mismatch')
    return model, centers, pools


def _test_arrays(manifest, r, e):
    test = manifest['tests'][str(r)]
    base = manifest['base_comparisons'][str(r)]
    require(base['manifest_path'] == str(BASE_MANIFESTS[r]) and base['manifest_sha256'] == LOCKS[BASE_MANIFESTS[r]],
            'Original comparison manifest binding changed')
    original = read(_bound(manifest, base['manifest_path'], base['manifest_sha256']))
    require(test == base['test'] == original['test'] and test['n_test'] == 10000 and test['seed'] == 9999,
            'Not the original matched seed9999 test')
    arrays = []
    for key in ('xi1', 'xi2'):
        info = test['arrays'][key]
        file = _bound(manifest, info['file'], info['file_sha256'])
        a = e.np.load(file, mmap_mode='r', allow_pickle=False)
        require(list(a.shape) == info['shape'] == [10000, r] and
                str(a.dtype) == info['dtype'] == 'float64' and e.np.isfinite(a).all() and (a >= 0).all(),
                'Invalid matched test-array shape/type/value')
        require(hashlib.sha256(a.tobytes(order='C')).hexdigest() == info['array_sha256'], 'Test array digest mismatch')
        arrays.append(a)
    return arrays


def _execute(job, manifest_path, output_folder, smoke):
    start = time.perf_counter()
    require(Path(job['id']).name == job['id'], 'Invalid policy ID')
    folder = Path(output_folder).resolve() / job['id']
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'result.json'
    record = dict(job, experiment=EXPERIMENT, status='running', start_utc=utc(),
        phases={}, status_counts={}, solves_attempted=0, solves_accepted=0, iterations=0., barrier_iterations=0,
        mean_oos=None, finite_pool_mean=None, cached=False, solver_threads=1, torch_threads=1,
        torch_interop_threads=1, timing_clock='time.perf_counter', timing_unit='seconds',
        preflight_only=smoke, test_evaluated=False, training_probe_mean=None,
        environment=dict(host=platform.node(), platform=platform.platform(), python=sys.version),
        result_path=str(path), summary_path=str(path), journals={})
    manifest = digest = model = e = None
    try:
        manifest, digest = _sealed(manifest_path)
        record.update(manifest_path=str(Path(manifest_path).resolve()), manifest_sha256=digest,
                      worker_count=manifest['workers'])
        e = initialize_worker(job['R'], manifest_path)
        with e.c.phase(record['phases'], 'input_loading'):
            model, centers, pools = _own_inputs(job, manifest, e)
        if smoke:
            record['training_probe_original_sample_ids'] = job['original_sample_ids'][:2]
            e._evaluate(model, centers[:2, :job['R']], centers[:2, job['R']:], 'training_probe', record, folder)
            record['training_probe_mean'], record['mean_oos'] = record['mean_oos'], None
            expected = 2
        else:
            with e.c.phase(record['phases'], 'test_loading'):
                arrays = _test_arrays(manifest, job['R'], e)
            record['test'] = manifest['tests'][str(job['R'])]
            e._evaluate(model, *arrays, 'oos', record, folder)
            points = e.np.concatenate(pools)
            e._evaluate(model, points[:, :job['R']], points[:, job['R']:], 'is', record, folder, job['sizes'])
            expected = 10000 + sum(job['sizes'])
            record['test_evaluated'] = True
        require(record['solves_attempted'] == record['solves_accepted'] == expected and
                record['status_counts'] == {'2': expected}, 'Incomplete strict QP evaluation')
        record['status'] = 'completed'
    except Exception:
        record.update(status='failed', error=traceback.format_exc())
    finally:
        try:
            if digest is not None:
                require(sha(manifest_path) == digest and
                        Path(manifest_path).with_suffix('.sha256').read_text().strip() == digest,
                        'Manifest changed during evaluation')
                _verify_sources(manifest)
                for source in job['input_files']:
                    _bound(manifest, source, job['input_sha256'][source])
                if model is not None:
                    require(e.c.state_hash(model) == job['state_sha256'], 'Evaluation mutated policy')
                if not smoke and 'test' in record:
                    for info in record['test']['arrays'].values():
                        _bound(manifest, info['file'], info['file_sha256'])
                record['inputs_unchanged'] = True
        except Exception:
            record.update(status='failed', inputs_unchanged=False, integrity_error=traceback.format_exc())
        if record['status'] != 'completed':
            record.update(mean_oos=None, finite_pool_mean=None, training_probe_mean=None)
        for journal in sorted(folder.glob('*_solves.jsonl')):
            record['journals'][journal.stem] = dict(path=str(journal), sha256=sha(journal))
        record.update(end_utc=utc(), wall_time_s=time.perf_counter()-start,
                      timing_scope='current evaluation invocation; historical training excluded; final JSON write excluded')
        record['wall_time_human'] = f"{record['wall_time_s']:.3f} s / {record['wall_time_s']/60:.3f} min"
        with path.open('x') as stream:
            json.dump(record, stream, allow_nan=False, indent=2)
            stream.write('\n')
    record['result_sha256'] = record['summary_sha256'] = sha(path)
    return record


def worker(job, manifest_path, output_folder):
    return _execute(job, manifest_path, output_folder, False)


def training_smoke(job, manifest_path, output_folder):
    return _execute(job, manifest_path, output_folder, True)