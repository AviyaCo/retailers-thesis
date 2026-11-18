"""Training-only stopping-checkpoint selection from the sealed 570-job audit.

collect(output) returns the JSON stored in selection_manifest.json (same-stem
.sha256 seal). It never evaluates a policy or imports an evaluator. All archive
reads use resolve_archive_path; historical paths in evidence remain untouched.
The old audit validated every AP/MP transition and state identity. We rehash its
bindings, compare the complete candidate evidence, and audit each chosen terminal
assessment/native pre-insertion pool, rather than rerunning optimization.

The old input map ALSO includes test arrays. Their hashes are checked only AFTER
the selection manifest is sealed, without loading arrays. Successful return also
requires selection_verification.json, a sealed current-invocation timing receipt.
A failed invocation raises and writes selection_failure.json; a selection seal
alone is not evidence that collection completed. Parent code must reverify inputs
and bind its own launch sources before test loading. Historical phase sums are
not fresh training time, nor end-to-end elapsed time across separate invocations.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
OLD_MANIFEST = 'comparison/black_green_evaluation/production_v2_20260928/manifest.json'
OLD_MANIFEST_SHA256 = '10d1cb5f2ce08700b0261a8674e82f6699952faad560503ad2dfaf65c38c063d'
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
RETURN_RULE = 'stopping_checkpoint'
OUTGOING_TIMINGS = ('mp', 'mp_origin', 'epochs', 'incoming_pool_evaluation',
                    'returned_pool_evaluation')
PHASES = ('ap', 'pool_evaluation', 'mp', 'incoming_pool_evaluation',
          'returned_pool_evaluation')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    # Importing helpers must not create bytecode inside the immutable archive.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


_paths = _load(ROOT / 'archive_paths.py', '_stopping_archive_paths')


class _Bindings:
    """Resolve locally, reject conflicting bindings, hash each distinct file once."""

    def __init__(self, root=ROOT):
        self.root = root
        self.paths = {}
        self.expected = {}
        self.verified = {}
        self.forbidden = set()

    def path(self, value):
        key = str(value)
        if key not in self.paths:
            self.paths[key] = _paths.resolve_archive_path(value, root=self.root)
        return self.paths[key]

    def bind(self, value, digest):
        path = str(self.path(value))
        _require(isinstance(digest, str) and len(digest) == 64 and
                 all(c in '0123456789abcdef' for c in digest), f'Invalid hash: {value}')
        _require(path not in self.expected or self.expected[path] == digest,
                 f'Conflicting binding: {path}')
        self.expected[path] = digest
        return path

    def add(self, value, expected=None):
        path = str(self.path(value))
        _require(path not in self.forbidden, f'Test bytes forbidden before selection seal: {path}')
        if expected is not None:
            self.bind(path, expected)
        if path not in self.verified:
            with self.path(path).open('rb') as stream:
                self.verified[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
        digest = self.verified[path]
        _require(path not in self.expected or digest == self.expected[path], f'Hash mismatch: {path}')
        self.bind(path, digest)
        return digest

    def read(self, value, expected=None):
        digest = self.add(value, expected)
        payload = self.path(value).read_bytes()
        _require(hashlib.sha256(payload).hexdigest() == digest, f'Changed while reading: {value}')
        return json.loads(payload)

    def sealed(self, value, expected=None):
        path = self.path(value)
        seal = self.path(path.with_suffix('.sha256'))
        self.add(seal)
        digest = seal.read_text().strip()
        _require(expected is None or digest == expected, 'Unexpected sealed manifest identity')
        return self.read(path, digest)


def _finite(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite candidate/timing evidence')
    return value


def _green_gap(candidate):
    pool = _finite(candidate['restricted_mean'])
    return abs(_finite(candidate['ap_incumbent_mean']) - pool) / max(abs(pool), 1e-9)


def _stopping(prefix, method, *, boundary=0, stored_stop=None, terminal_status=None):
    """Return the last assessment at the actual stop, never an AP minimum."""
    _require(bool(prefix), 'Empty assessed prefix')
    for i, candidate in enumerate(prefix):
        _require(type(candidate['candidate_id']) is int and candidate['candidate_id'] == i and
                 candidate['assessed'] is True, 'Noncontiguous or unassessed prefix')
        _finite(candidate['ap_incumbent_mean'])
    if method == 'green':
        passes = [i for i, c in enumerate(prefix) if _green_gap(c) < .01]
        _require(bool(passes), 'No GREEN stopping assessment')
        stop = passes[0]
        if stored_stop is not None:
            _require(stored_stop == stop == len(prefix) - 1 and
                     terminal_status == 'green_gap_satisfied', 'Saved GREEN stop differs from first pass')
    else:
        _require(method == 'black' and 0 <= boundary < len(prefix), 'Invalid BLACK boundary')
        stop = len(prefix) - 1
        _require(stored_stop == stop, 'Saved BLACK stopping ID differs from full prefix')
        _require(all(c['separation_status'] in ('violating', 'separation_satisfied') for c in prefix),
                 'Unknown separation status')
        _require(not any(c['separation_status'] == 'separation_satisfied'
                         for c in prefix[boundary:stop]), 'BLACK continued after its stopping condition')
        closed = prefix[-1]['separation_status'] == 'separation_satisfied'
        _require((closed and terminal_status in ('converged', 'separation_satisfied')) or
                 (not closed and terminal_status == 'iteration_cap' and stop == 40),
                 'BLACK saved convergence/cap disagrees with terminal assessment')
    return prefix[stop]


def _prefixes(old, cap, backend):
    record = cap.read(old['source_summary_path'], old['source_summary_sha256'])
    legacy = old['R'] == 3 and old['Ntotal'] == 50
    for key in ('Ntotal', 'Ntrain', 'seed', 'radius'):
        _require(record[key] == old[key], f'Source identity mismatch: {key}')
    if legacy:
        original = None
        prefix = record['iterations']
        boundary = 0
        _require(record['status'] == 'completed' and
                 ('stopping_candidate_id' not in record or
                  record['stopping_candidate_id'] == old['stopping_candidate_id']),
                 'Legacy saved status/stop mismatch')
    else:
        _require(record['R'] == old['R'] and record['mode'] == 'job' and
                 record['heldout_access'] is False and
                 record['status'] == ('completed' if old['converged'] else 'nonconverged'),
                 'Invalid raw continuation')
        source = record['source_prefix']
        original = cap.read(source['summary_path'], source['summary_sha256']) if source['summary_path'] else None
        _require(source['immutable'] is True and
                 source['candidates'] == (len(original['iterations']) if original else 0),
                 'Original GREEN prefix coverage mismatch')
        _require(original is not None or (old['R'] == 10 and old['radius'] == 2.),
                 'Missing retained GREEN prefix')
        prefix, _ = backend.joined_prefix(original, record)
        boundary = len(original['iterations']) - 1 if original else 0
        _require(record['new_mp_count'] == old['new_mp_count'] and
                 record['stopping_candidate_id'] == old['stopping_candidate_id'],
                 'Continuation count/stop drift')
    _require(len(prefix) == old['assessed_candidate_count'] == record['mp_count'] + 1 and
             record['mp_count'] == old['mp_count'] == old['stopping_candidate_id'] and
             record['terminal_status'] == old['terminal_status'] and
             record['converged'] is old['converged'] and
             old['retained_candidate_count'] == (len(prefix) if legacy else boundary + 1 if original else 0),
             'Old audit/source terminal status or counts disagree')
    _require(len(old['selection_evidence']) == len(prefix), 'Incomplete old candidate evidence')
    for candidate, evidence in zip(prefix, old['selection_evidence']):
        _require(all(candidate.get(k) == v for k, v in evidence.items()), 'Candidate differs from sealed audit')
    black = _stopping(prefix, 'black', boundary=boundary,
                      stored_stop=old['stopping_candidate_id'], terminal_status=old['terminal_status'])
    _require((black['separation_status'] == 'separation_satisfied') is old['converged'],
             'Old convergence flag differs from stopping assessment')
    green = None
    if legacy:
        green = _stopping(prefix, 'green')
    elif original:
        _require(original['status'] == 'completed' and original['converged'] is True and
                 all(original[k] == old[k] for k in ('R', 'Ntotal', 'Ntrain', 'seed', 'radius')) and
                 original['mp_count'] == original['stopping_candidate_id'] == boundary,
                 'Invalid original GREEN identity/status')
        green = _stopping(original['iterations'], 'green', stored_stop=boundary,
                          terminal_status=original['terminal_status'])
        for c in original['iterations']:
            _require(math.isclose(c['green_gap'], _green_gap(c), rel_tol=1e-12, abs_tol=1e-12) and
                     c['green_pass'] is (_green_gap(c) < .01), 'Saved GREEN gap/pass differs')
    return record, original, prefix, green, black


def _historical(old, stop, green_stop, method, record, original):
    reference = old['training_runtime_reference']
    saved = reference['iterations']
    _require(len(saved) == old['assessed_candidate_count'] and
             [r['candidate_id'] for r in saved] == list(range(len(saved))), 'Incomplete persisted timings')
    rows = []
    for i, row in enumerate(saved[:stop + 1]):
        row = deepcopy(row)
        if i == stop:
            for name in OUTGOING_TIMINGS:
                row.pop(name, None)
        for name in PHASES:
            if name in row:
                phase = row[name]
                _require(phase['status'] == 'completed' and _finite(phase['wall_time_s']) >= 0,
                         'Incomplete historical phase timing')
                phase.pop('per_ball_costs', None)
        rows.append(row)
    green_rows, black_rows = [], []
    for row in rows:
        i = row['candidate_id']
        if green_stop is not None and i <= green_stop:
            before = deepcopy(row)
            if i == green_stop:
                for name in OUTGOING_TIMINGS:
                    before.pop(name, None)
            green_rows.append(before)
            if i == green_stop and 'mp' in row:
                black_rows.append({k: deepcopy(v) for k, v in row.items()
                                   if k in ('candidate_id', *OUTGOING_TIMINGS)})
        else:
            black_rows.append(deepcopy(row))
    def segment(items):
        return dict(iterations=items, active_phase_time_s=sum(
            row[name]['wall_time_s'] for row in items for name in PHASES if name in row),
            timing_scope='sum of persisted sequential phases; NOT invocation/end-to-end wall time')
    nominal = deepcopy(reference['nominal'])
    _require(nominal['training_timing']['status'] == 'completed' and
             _finite(nominal['training_timing']['wall_time_s']) >= 0, 'Missing nominal timing')
    # Full legacy BLACK, fresh rho2 BLACK and original nonlegacy GREEN have
    # invocation totals. Truncated legacy GREEN/multi-invocation BLACK do not.
    full = original if method == 'green' and original else (
        record if method == 'black' and original is None else None)
    if full is not None:
        _require(_finite(full['wall_time_s']) >= 0, 'Invalid saved invocation duration')
    return dict(status='persisted_not_current', nominal=nominal, iterations=rows,
                green_through_stop=segment(green_rows), black_after_green_stop=segment(black_rows),
                total_wall_time_s=full['wall_time_s'] if full else None,
                original_prefix=deepcopy(reference['original_prefix']) if method == 'black' or original else None,
                continuation=deepcopy(reference['continuation']) if method == 'black' else None,
                timing_unit='seconds', current_training_performed=False,
                note='Nominal is separate. Missing end-to-end totals are unavailable; rerun required. '
                     'Legacy GREEN excludes the outgoing MP at its first passing AP assessment. '
                     'BLACK starts at the retained GREEN stop, not at its returned/min-AP checkpoint.')


def _terminal_job(old, candidate, method, record, original, green, cap, backend):
    import numpy as np

    training = cap.read(old['training_data_path'], old['training_data_sha256'])
    centers = np.asarray(training['centers'], dtype=np.float64)
    ids = training['original_sample_ids']
    _require(centers.shape == (old['Ntrain'], 2 * old['R']) and old['Ntrain'] == 4 * old['Ntotal'] // 5 and
             np.isfinite(centers).all() and (centers >= 0).all() and
             np.array_equal(centers, old['centers']) and ids == old['original_sample_ids'] and
             len(ids) == len(set(ids)) == old['Ntrain'] and
             all(type(i) is int and 0 <= i < old['Ntotal'] for i in ids) and
             training['data_sha256'] == old['data_sha256'], 'Training center/ID/hash mismatch')
    if original:
        _require(original['training_data_sha256'] == old['training_data_sha256'] and
                 original['data_sha256'] == training['data_sha256'], 'GREEN training data differs')
    checkpoint = cap.path(candidate['checkpoint'])
    cap.add(checkpoint, candidate['checkpoint_sha256'])
    ap = cap.path(candidate['ap_reference'])
    _require(str(ap) in cap.expected, 'AP assessment was not bound by old audit')
    backend.assessment_equal(candidate, cap.read(ap, cap.expected[str(ap)]))
    pool_path = cap.path(ap.parent / 'pool.json')
    saved = cap.read(pool_path, candidate['pool_file_sha256'])
    _require(saved['original_sample_ids'] == ids, 'Native pool IDs differ')
    pools = backend.validate_pools(saved['pools'], centers, old['radius'])
    _require(backend.pool_digest(pools) == saved['pool_sha256'] == candidate['pool_sha256'],
             'Native pre-insertion pool digest differs')
    closed, _ = backend.candidate_rows(candidate, old, centers, ids, pools)
    if method == 'black':
        _require(closed is old['converged'], 'Terminal row audit disagrees with BLACK convergence')
    else:
        _require(_green_gap(candidate) < .01, 'Terminal row audit fails GREEN gap')
    stop = candidate['candidate_id']
    inputs = {str(cap.path(p)): digest for p, digest in old['input_sha256'].items()}
    for path, digest in inputs.items():
        _require(cap.expected[path] == cap.verified[path] == digest, 'Job input not freshly verified')
    for path in (checkpoint, ap, pool_path, cap.path(old['training_data_path']), cap.path(old['source_summary_path'])):
        _require(str(path) in inputs, f'Terminal input missing from old job binding: {path}')
    source_path = old['source_summary_path'] if original is None or method == 'black' else record['source_prefix']['summary_path']
    source_path = str(cap.path(source_path))
    job = {k: old[k] for k in ('R', 'Ntotal', 'Ntrain', 'seed', 'radius')}
    job.update(id=f'{method}_{old["id"]}', method=method, return_rule=RETURN_RULE,
        checkpoint=str(checkpoint), checkpoint_sha256=candidate['checkpoint_sha256'],
        state_sha256=candidate['state_sha256'], pool_path=str(pool_path),
        pool_file_sha256=candidate['pool_file_sha256'], pool_sha256=candidate['pool_sha256'],
        sizes=[len(p) for p in pools], training_data_path=str(cap.path(old['training_data_path'])),
        training_data_sha256=old['training_data_sha256'], data_sha256=old['data_sha256'],
        training_centers_sha256=hashlib.sha256(np.asarray(centers, dtype='<f8').tobytes(order='C')).hexdigest(),
        training_centers_hash_encoding='little-endian float64, C order; shape=(Ntrain,2*R)',
        original_sample_ids=ids, selected_candidate_id=stop, stopping_candidate_id=stop,
        mp_count=stop, converged=True if method == 'green' else old['converged'],
        terminal_status='green_gap_satisfied' if method == 'green' else old['terminal_status'],
        global_robust_optimality_certified=False,
        selected_separation_status=candidate['separation_status'],
        ap_reference=str(ap), ap_reference_sha256=cap.expected[str(ap)],
        selection_record_path=source_path, selection_record_sha256=cap.expected[source_path],
        source_summary_path=str(cap.path(old['source_summary_path'])),
        source_summary_sha256=old['source_summary_sha256'],
        selected_ap_incumbent_mean=candidate['ap_incumbent_mean'],
        saved_native_finite_pool_mean=candidate['restricted_mean'],
        input_sha256=inputs,
        historical_training=_historical(old, stop, green['candidate_id'] if green else None,
                                        method, record, original))
    _require(job['selected_candidate_id'] == job['stopping_candidate_id'], 'Selected/stop mismatch')
    return job


def _cohort(jobs, *, selected):
    expected = {(method, r, n, seed, rho)
                for method in (('green', 'black') if selected else ('black',))
                for r in (3, 10) for n in (50, 200, 800) for seed in range(10)
                for rho in RADII + ((2.,) if method == 'black' and r == 10 else ())}
    actual = [(j['method'], j['R'], j['Ntotal'], j['seed'], j['radius']) for j in jobs]
    _require(len(actual) == len(set(actual)) == len(expected) and set(actual) == expected and
             len({j['id'] for j in jobs}) == len(actual), 'Incomplete/duplicate fixed-radius cohort')
    black = [j for j in jobs if j['method'] == 'black']
    _require(sum(j['converged'] is True for j in black) == 559 and
             sum(j['converged'] is False and j['terminal_status'] == 'iteration_cap' and
                 j['mp_count'] == j['stopping_candidate_id'] == 40 for j in black) == 11,
             'BLACK must retain 559 converged and 11 capped policies')
    if selected:
        _require(all(j['selected_candidate_id'] == j['stopping_candidate_id'] for j in jobs) and
                 sum(j['method'] == 'green' and j['converged'] is True and
                     j['terminal_status'] == 'green_gap_satisfied' for j in jobs) == 540,
                 'GREEN stop/pass inventory mismatch')


def _seal(path, data):
    payload = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    digest = hashlib.sha256(payload).hexdigest()
    with path.open('xb') as stream:
        stream.write(payload)
    path.chmod(0o444)
    with path.with_suffix('.sha256').open('x') as stream:
        stream.write(digest + '\n')
    path.with_suffix('.sha256').chmod(0o444)
    return digest


def collect(output: Path) -> dict:
    """Collect/seal all 1,110 jobs; no optimization, policy evaluation, or test selection.

    ROOT is the shared retailers_experiment root. output may already be a directory
    but none of this invocation's three artifact stems may exist. On success the
    returned object exactly equals selection_manifest.json; the verification receipt
    accounts for hashing test bytes AFTER that immutable selection was persisted.
    """
    output = Path(output).resolve()
    _require(not output.is_relative_to(ROOT / 'archive'), 'Cannot write into frozen archive')
    output.mkdir(parents=True, exist_ok=True)
    for stem in ('selection_manifest', 'selection_verification', 'selection_failure'):
        for suffix in ('.json', '.sha256'):
            _require(not (output / (stem + suffix)).exists(), 'Selection output must be exclusive')
    started, start = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    cap = _Bindings()
    jobs = []
    try:
        old = cap.sealed(OLD_MANIFEST, OLD_MANIFEST_SHA256)
        _require(old['status'] == 'frozen' and old['policies_frozen_before_test'] is True and
                 old['n_test'] == 10000 and old['test_seed'] == 9999, 'Unsealed/changed old protocol')
        _cohort(old['jobs'], selected=False)
        tests = deepcopy(old['tests'])
        test_paths = set()
        for test in tests.values():
            _require(test['n_test'] == 10000 and test['seed'] == 9999, 'Changed test protocol')
            for array in test['arrays'].values():
                array['file'] = str(cap.path(array['file']))
                test_paths.add(cap.bind(array['file'], array['file_sha256']))
        # Protect both original array names and materialized copies from any read.
        cap.forbidden.update(test_paths)
        for source, info in old['files'].items():
            if str(cap.path(source)) in test_paths:
                cap.forbidden.add(str(cap.path(info['copy'])))
        sources = {}
        for field in ('input_sha256', 'source_sha256'):
            for path, digest in old[field].items():
                local = cap.bind(path, digest)
                if field == 'source_sha256':
                    sources[local] = digest
        for old_job in old['jobs']:
            for path, digest in old_job['input_sha256'].items():
                local = str(cap.path(path))
                _require(local in cap.expected and cap.expected[local] == digest,
                         'Job dependency not bound globally')
        for path, digest in list(cap.expected.items()):
            if path not in cap.forbidden:
                cap.add(path, digest)
        summary = None
        summary_binding = None
        try:
            summary_path = cap.path(cap.path(OLD_MANIFEST).parent / 'summary.json')
        except FileNotFoundError:
            summary_path = None
        if summary_path is not None:
            summary = cap.sealed(summary_path)
            _require(summary['status'] == 'completed' and summary['job_count'] == 570 and
                     cap.path(summary['manifest']) == cap.path(OLD_MANIFEST) and
                     summary['manifest_sha256'] == OLD_MANIFEST_SHA256 and
                     summary['phases']['final_verification']['status'] == 'completed',
                     'Old successful summary does not bind the sealed audit')
            summary_binding = dict(path=str(summary_path), sha256=cap.expected[str(summary_path)])
        backend_path = cap.path('comparison/black_green_evaluation/backend.py')
        _require(str(backend_path) in sources, 'Reused backend is not in old source bindings')
        backend = _load(backend_path, '_stopping_archived_backend')
        for path in (Path(__file__), Path(__file__).with_name('test_selection.py'),
                     ROOT / 'archive_paths.py', ROOT / 'archive_paths.json'):
            sources[str(cap.path(path))] = cap.add(path)
        for old_job in old['jobs']:
            record, original, prefix, green, black = _prefixes(old_job, cap, backend)
            if green is not None:
                jobs.append(_terminal_job(old_job, green, 'green', record, original, green, cap, backend))
            jobs.append(_terminal_job(old_job, black, 'black', record, original, green, cap, backend))
        _cohort(jobs, selected=True)
        base_plots = {}
        for r in (3, 10):
            path = cap.path(f'comparison/black_green_figures/plot_data_R{r}.json')
            digest = cap.add(path, summary['artifacts'][path.name] if summary else None)
            base_plots[str(r)] = dict(path=str(path), sha256=digest)
        elapsed = time.perf_counter() - start
        result = dict(experiment='stopping_checkpoint_20260928', status='frozen',
            return_rule=RETURN_RULE, selection_rule=RETURN_RULE, jobs=jobs,
            source_old_manifest=dict(path=str(cap.path(OLD_MANIFEST)), sha256=OLD_MANIFEST_SHA256),
            source_old_summary=summary_binding, tests=tests, base_plot_data=base_plots,
            input_sha256=dict(cap.expected), source_sha256=sources,
            policies_frozen_before_test=True, test_bytes_accessed=False, test_evaluated=False,
            new_training=False, global_robust_optimality_certified=False,
            pending_test_hash_verification=sorted(test_paths),
            verification_receipt=str(output / 'selection_verification.json'),
            collection_timing=dict(start_utc=started, end_utc=datetime.now(timezone.utc).isoformat(),
                wall_time_s=elapsed, clock='time.perf_counter', timing_unit='seconds',
                status='selection_completed', timing_scope='current collection through selection; excludes seal and deferred test hashing'),
            counts=dict(green=540, black=570, black_converged=559, black_capped=11))
        manifest_sha = _seal(output / 'selection_manifest.json', result)
        cap.forbidden.clear()
        for path, digest in cap.expected.items():
            cap.add(path, digest)
        elapsed = time.perf_counter() - start
        _seal(output / 'selection_verification.json', dict(status='completed',
            selection_manifest_sha256=manifest_sha, all_old_bindings_verified=True,
            verified_file_count=len(cap.verified), test_array_hashes_checked_after_selection_seal=True,
            test_arrays_loaded=False, start_utc=started, end_utc=datetime.now(timezone.utc).isoformat(),
            wall_time_s=elapsed, wall_time_human=f'{elapsed:.3f} s / {elapsed / 60:.3f} min',
            timing_clock='time.perf_counter', timing_unit='seconds', workers=1,
            timing_scope='current collection including manifest seal/deferred hashes; receipt serialization excluded'))
        return result
    except BaseException as exc:
        _seal(output / 'selection_failure.json', dict(status='failed',
            error=f'{type(exc).__name__}: {exc}', jobs_collected=len(jobs),
            start_utc=started, end_utc=datetime.now(timezone.utc).isoformat(),
            wall_time_s=time.perf_counter() - start, timing_clock='time.perf_counter',
            timing_unit='seconds', workers=1, historical_training_time_included=False))
        raise