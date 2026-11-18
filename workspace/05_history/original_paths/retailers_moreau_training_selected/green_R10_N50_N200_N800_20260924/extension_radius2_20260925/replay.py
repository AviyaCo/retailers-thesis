"""Fixed rho=2 replay/append only. No training, selection, backend import or CLI.

Controller contract (all paths absolute; SHA256 values are lowercase hex):

* Seal manifest.json and manifest.sha256 AFTER all 60 policies are frozen and
  BEFORE calling evaluate_green. Manifest fields: status='frozen', R=10,
  radius=2., n_test=10000, test_seed=9999, new_selection=False,
  policies_frozen_before_test_eval=True, workers (1..32), threads=1,
  versions=comparison.evaluation.versions(), membership_tolerance=1e-9,
  physical_tolerance=1e-4, base_test (EXACT original manifest['test']),
  base_manifest/base_summary/base_plot each {file, sha256}, files mapping
  original paths to {copy, sha256}, source_sha256 mapping source paths to hashes,
  and jobs: exactly 30 green plus 30 sro_fa, N=50/200/800, seeds=0..9.
* Every job has id, method, Ntotal, Ntrain, seed, radius, input_files.
  Green jobs use the old frozen evaluation spec, PLUS training_data_path and
  split_path; source_summary_path points to the full old CCG summary (not the
  compact training-job return). Include EVERY prefix checkpoint, pool and AP
    assessment, summary, split and training data in files AND input_files.
    Keep original prefix pool paths available and unchanged too: the reused
    selected_green_pool validator checks their byte hashes as well as bound copies.
  SRO jobs have policy_path and policy_sha256 (frozen policy NPZ); replay never
  imports its backend or evaluates it. Parent verifies/trains its policy.
* source_sha256 must bind this file, comparison evaluation/reference/plot,
  and the two upstream definition sources in REQUIRED_SOURCES. Their frozen
  copies and currently executed source bytes must match before execution.
* evaluate_green(job, manifest_path, output) creates output/job['id'], writes
  result.json and both raw QP journals, and returns the FULL row, including
  result_path, test, manifest_sha256 and all plot _evidence fields. Failed
  evaluations return status='failed' with traceback and preserved journals;
  the parent MUST reject them. Optional spawn initializer:
  initialize_worker(manifest_path). No 810-job worker/load-inputs is invoked.
* merge_and_render(base_plot_path, green_rows, sro_rows, output, bindings)
  requires bindings={manifest: {file, sha256}, results: {job_id: {file, sha256}}}.
  Rows must equal these persisted JSON artifacts and extend their frozen job
  specs unchanged. SRO rows additionally require status='completed', cached=False,
  oos (finite), test=base_test, manifest_sha256, wall_time_s (evaluation only),
  start_utc/end_utc, phases, iterations, worker_count, solver_threads, environment,
  timing_clock='time.perf_counter', timing_unit='seconds',
  training_runtime_reference (separate persisted training timing/status).
  Green rows carry these fields plus strict solve counts and per-center maxima.
* output must be a NEW parent staging directory. Only after all 60 results
  pass does this write plot_data.json and comparison_R10_radius2.png/.pdf.
  It returns plot data; the parent handles overall/train runtime reporting and
  final atomic publication. No existing file or frozen artifact is overwritten.

Historical per-seed evidence/timing arrays remain verbatim source-invocation
records (including their old cached=False flags). replay_accounting, NOT those
archived flags/times, describes this invocation: cached 810 neural, fresh 30
green, new 30 SRO. Validation winners and common oracle are never recomputed.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback
import types

import numpy as np

HERE = Path(__file__).resolve().parent
COMPARISON = HERE.parent / 'comparison_20260925'
HOME = Path('/data/home/aviyac')
REQUIRED_SOURCES = (
    Path(__file__).resolve(), COMPARISON / 'evaluation.py',
    COMPARISON / 'reference.py', COMPARISON / 'plot_comparison.py',
    HOME / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison/compare.py',
    HOME / 'retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/evaluation.py',
)
SIZES = (50, 200, 800)
SEEDS = tuple(range(10))
_EVALUATORS = {}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    # Only our own new result may be updated; exclusive job/staging mkdir gates it.
    temp = Path(path).with_suffix('.json.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _bound(binding):
    path = Path(binding['file'])
    _require(path.is_absolute() and _digest(binding['sha256'])
             and _sha(path) == binding['sha256'], f'Artifact hash mismatch: {path}')
    return path


def _copy(manifest, source):
    source = str(source)
    _require(Path(source).is_absolute() and source in manifest['files'], f'Unbound input: {source}')
    entry = manifest['files'][source]
    return _bound(dict(file=entry['copy'], sha256=entry['sha256']))


def _sealed(binding):
    path = _bound(binding)
    _require(path.with_suffix('.sha256').read_text().strip() == binding['sha256'],
             'Manifest sidecar mismatch')
    return _json(path)


def _identity(job):
    _require(job['method'] in ('green', 'sro_fa') and type(job['Ntotal']) is int
             and job['Ntotal'] in SIZES and type(job['seed']) is int
             and job['seed'] in SEEDS and type(job['radius']) in (int, float)
             and job['radius'] == 2., 'Invalid radius-2 identity')
    _require(job['Ntrain'] == (4*job['Ntotal']//5 if job['method'] == 'green' else job['Ntotal']),
             'Training budget mismatch')
    _require(isinstance(job['id'], str) and job['id'] not in ('', '.', '..')
             and Path(job['id']).name == job['id'], 'Invalid job ID')
    return job['method'], job['Ntotal'], job['seed']


def _manifest(binding):
    m = _sealed(binding)
    _require(m['status'] == 'frozen' and m['R'] == 10 and m['radius'] == 2.
             and m['new_selection'] is False and m['policies_frozen_before_test_eval'] is True,
             'All policies must be frozen before any new test evaluation; no selection')
    _require(m['n_test'] == 10000 and m['test_seed'] == 9999 and m['threads'] == 1
             and type(m['workers']) is int and 1 <= m['workers'] <= 32
             and m['membership_tolerance'] == 1e-9 and m['physical_tolerance'] == 1e-4,
             'Changed evaluation protocol')
    expected = {(method, n, seed) for method in ('green', 'sro_fa') for n in SIZES for seed in SEEDS}
    _require(len(m['jobs']) == 60 and {_identity(j) for j in m['jobs']} == expected
             and len({j['id'] for j in m['jobs']}) == 60, 'Require all 60 frozen policies')
    # Small selected-policy barrier, not a replay or a rescan of 7047 base inputs.
    for job in m['jobs']:
        field = 'checkpoint' if job['method'] == 'green' else 'policy_path'
        digest = job['checkpoint_sha256'] if job['method'] == 'green' else job['policy_sha256']
        _require(job[field] in job['input_files'] and _sha(_copy(m, job[field])) == digest,
                 'Selected policy missing from complete freeze')
    base = _sealed(m['base_manifest'])
    _require(base['status'] == 'frozen' and base['R'] == 10 and base['job_count'] == 810
             and m['base_test'] == base['test'] and base['n_test'] == 10000
             and base['test_seed'] == 9999, 'Changed frozen base test')
    _require(m['base_test']['n_test'] == 10000 and m['base_test']['seed'] == 9999,
             'Invalid base test protocol')
    for key in ('xi1', 'xi2'):
        a = m['base_test']['arrays'][key]
        _require(a['shape'] == [10000, 10] and a['dtype'] == 'float64'
                 and _digest(a['array_sha256']) and _digest(a['file_sha256']), 'Invalid R10 test identity')
    return m


def _sources(m):
    base = _sealed(m['base_manifest'])
    for path in REQUIRED_SOURCES:
        source = str(path)
        _require(source in m['source_sha256'], f'Missing evaluator source binding: {source}')
        if path != Path(__file__).resolve():
            _require(source in base['files']
                     and base['files'][source]['sha256'] == m['source_sha256'][source],
                     f'Evaluator differs from frozen base contract: {source}')
    for source, digest in m['source_sha256'].items():
        _require(_digest(digest) and _sha(source) == _sha(_copy(m, source)) == digest,
                 f'Executed source changed: {source}')


def _private_module(path, text=None, injected=None):
    """No sys.path changes, bare-module aliases, or green_common imports."""
    path = Path(path)
    text = path.read_text() if text is None else text
    name = '_radius2_' + path.stem + '_' + hashlib.sha256(text.encode()).hexdigest()[:16]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(injected or {})
    exec(compile(text, str(path), 'exec'), module.__dict__)
    return module


def _plot_module():
    return _private_module(COMPARISON / 'plot_comparison.py')


def _evaluator(m):
    _sources(m)
    key = tuple(sorted(m['source_sha256'].items()))
    if key not in _EVALUATORS:
        path = COMPARISON / 'evaluation.py'
        text = path.read_text()
        original = 'from plot_comparison import validate_cache_protocol'
        _require(text.count(original) == 1, 'Evaluator import contract changed')
        text = text.replace(original, 'validate_cache_protocol = _private_plot.validate_cache_protocol')
        _EVALUATORS[key] = _private_module(path, text, {'_private_plot': _plot_module()})
    return _EVALUATORS[key]


def initialize_worker(manifest_path):
    path = Path(manifest_path).resolve()
    m = _manifest(dict(file=str(path), sha256=path.with_suffix('.sha256').read_text().strip()))
    e = _evaluator(m)
    if e.c.torch.get_num_threads() != 1 or e.c.torch.get_num_interop_threads() != 1:
        e.initialize_worker()


def _load_green_inputs(job, manifest_path):
    path = Path(manifest_path).resolve()
    digest = path.with_suffix('.sha256').read_text().strip()
    m = _manifest(dict(file=str(path), sha256=digest))
    _require(_identity(job)[0] == 'green', 'Only green policies can be replayed here')
    _require([j for j in m['jobs'] if j['id'] == job['id']] == [job], 'Job differs from frozen manifest')
    e = _evaluator(m)
    _require(m['versions'] == e.versions(), 'Evaluation environment changed')
    if e.c.torch.get_num_threads() != 1 or e.c.torch.get_num_interop_threads() != 1:
        e.initialize_worker()
    for source in job['input_files']:
        _copy(m, source)

    def read(source):
        _require(str(source) in job['input_files'], f'Prefix input absent from job: {source}')
        return _json(_copy(m, source))

    d, training, split = (read(job[k]) for k in ('source_summary_path', 'training_data_path', 'split_path'))
    _require(d['complete'] is True and d['converged'] is True and d['status'] == 'completed',
             'Incomplete green CCG summary')
    for field in ('Ntotal', 'Ntrain', 'Nval', 'seed', 'radius', 'selected_candidate_id',
                  'stopping_candidate_id', 'selected_green_gap', 'stopping_green_gap', 'selection_rule'):
        _require(d[field] == job[field], f'Summary/job mismatch: {field}')
    n, ids, arrays = job['Ntotal'], training['original_sample_ids'], split['arrays']
    _require(job['Nval'] == n//5 and len(ids) == job['Ntrain'] and len(set(ids)) == len(ids)
             and all(type(i) is int and 0 <= i < n for i in ids), 'Invalid training IDs')
    raw = np.concatenate([np.asarray(arrays[k], dtype=np.float64) for k in ('xi1', 'xi2')], axis=1)
    raw_hash = hashlib.sha256(b''.join(np.asarray(arrays[k], dtype='<f8').tobytes()
                                      for k in ('xi1', 'xi2'))).hexdigest()
    _require(raw.shape == (n, 20) and np.isfinite(raw).all()
             and raw_hash == training['data_sha256'] == split['raw_data_sha256'] == d['data_sha256']
             and ids == arrays['train_indices'] and set(ids).isdisjoint(arrays['val_indices'])
             and sorted(ids + arrays['val_indices']) == list(range(n))
             and np.array_equal(raw[ids], training['centers'])
             and training['centers'] == job['centers'] and ids == job['original_sample_ids'],
             'Training data/split/center binding mismatch')
    for candidate in d['iterations']:
        checkpoint = candidate['checkpoint']
        _require(checkpoint in job['input_files']
                 and _sha(_copy(m, checkpoint)) == candidate['checkpoint_sha256'], 'Prefix checkpoint changed')
    # Reuse the complete incoming-prefix and AP-assessment checks, including the
    # selected OWN pool. Its own gap need not be below the stopping threshold.
    candidate, pool_path, pools = e.selected_green_pool(d, training, read)
    _require(str(pool_path) == job['pool_path'] and _sha(_copy(m, pool_path)) == job['pool_file_sha256']
             and candidate['pool_file_sha256'] == job['pool_file_sha256']
             and candidate['pool_sha256'] == job['pool_sha256'] == e.pool_digest(pools)
             and [len(p) for p in pools] == job['sizes']
             and candidate['restricted_mean'] == job['saved_native_finite_pool_mean'], 'Selected own pool changed')
    _require(d['selected_checkpoint'] == candidate['checkpoint'] == job['checkpoint']
             and d['selected_checkpoint_sha256'] == job['checkpoint_sha256']
             and d['selected_state_sha256'] == job['state_sha256'], 'Selected state binding changed')
    model = e.c.load_model(_copy(m, job['checkpoint']), 'new')
    _require(e.c.state_hash(model) == job['state_sha256'], 'Selected learned state changed')
    test_arrays = []
    for name in ('xi1', 'xi2'):
        info = m['base_test']['arrays'][name]
        file = _bound(dict(file=info['file'], sha256=info['file_sha256']))
        a = np.load(file, mmap_mode='r', allow_pickle=False)
        _require(a.shape == (10000, 10) and str(a.dtype) == 'float64'
                 and np.isfinite(a).all() and (a >= 0).all()
                 and hashlib.sha256(a.tobytes(order='C')).hexdigest() == info['array_sha256'],
                 'Changed frozen test array')
        test_arrays.append(a)
    return e, m, digest, model, test_arrays, pools


def evaluate_green(job, manifest_path, output):
    start = time.perf_counter()
    _require(_identity(job)[0] == 'green', 'Only green replay is implemented')
    folder = Path(output).resolve() / job['id']
    folder.mkdir(parents=True, exist_ok=False)
    record = deepcopy(job)
    # Runtime values below describe this invocation, not copied training.
    record.update(experiment='R10_fixed_radius2_extension', status='running', cached=False,
        start_utc=datetime.now(timezone.utc).isoformat(), phases={}, status_counts={},
        solves_attempted=0, solves_accepted=0, iterations=0., barrier_iterations=0,
        mean_oos=None, finite_pool_mean=None, solver_threads=1, pid=os.getpid(),
        environment=dict(host=platform.node(), platform=platform.platform()),
        timing_clock='time.perf_counter', timing_unit='seconds', result_path=str(folder / 'result.json'))
    _write(folder / 'result.json', record)
    try:
        loading = time.perf_counter()
        record['phases']['input_loading'] = dict(status='running', start_utc=record['start_utc'])
        try:
            e, m, digest, model, arrays, pools = _load_green_inputs(job, manifest_path)
        except BaseException:
            record['phases']['input_loading']['status'] = 'failed'
            raise
        else:
            record['phases']['input_loading']['status'] = 'completed'
        finally:
            record['phases']['input_loading'].update(wall_time_s=time.perf_counter()-loading,
                end_utc=datetime.now(timezone.utc).isoformat())
        record.update(manifest_sha256=digest, test=m['base_test'], worker_count=m['workers'],
            torch_threads=e.c.torch.get_num_threads(), torch_interop_threads=e.c.torch.get_num_interop_threads())
        e._evaluate(model, *arrays, 'oos', record, folder)
        points = np.concatenate(pools).reshape(-1, 20)
        e._evaluate(model, points[:, :10], points[:, 10:], 'is', record, folder, job['sizes'])
        _require(e.c.state_hash(model) == job['state_sha256'], 'Evaluation changed learned state')
        _require(_sha(manifest_path) == digest, 'Manifest changed during evaluation')
        _sources(m)
        for source in job['input_files']:
            _copy(m, source)
        for info in m['base_test']['arrays'].values():
            _bound(dict(file=info['file'], sha256=info['file_sha256']))
        count = 10000 + sum(job['sizes'])
        _require(record['solves_attempted'] == record['solves_accepted'] == count
                 and record['status_counts'] == {'2': count}, 'Incomplete strict replay')
        record['status'] = 'completed'
        record['qp_journals'] = {name: dict(file=str(folder / f'{name}_solves.jsonl'),
            sha256=_sha(folder / f'{name}_solves.jsonl')) for name in ('oos', 'is')}
    except BaseException:
        record.update(status='failed', error=traceback.format_exc(), mean_oos=None, finite_pool_mean=None)
    finally:
        elapsed = time.perf_counter() - start
        record.update(end_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=elapsed,
                      wall_time_human=f'{elapsed:.3f} s ({elapsed/60:.3f} min)')
        _write(folder / 'result.json', record)
    return record


def _base(m, base_plot_path, p):
    _require(Path(base_plot_path).resolve() == Path(m['base_plot']['file']).resolve(), 'Wrong base plot')
    data = _json(_bound(m['base_plot']))
    summary = _json(_bound(m['base_summary']))
    manifest = _sealed(m['base_manifest'])
    _require(summary['status'] == 'completed' and summary['manifest_sha256'] == m['base_manifest']['sha256']
             and summary['manifest'] == m['base_manifest']['file']
             and summary['plot_data'] == m['base_plot']['file']
             and data['provenance']['manifest_sha256'] == m['base_manifest']['sha256']
             and data['test'] == m['base_test'], 'Base completion/binding mismatch')
    for source in manifest['files']:
        _copy(manifest, source)
    for info in m['base_test']['arrays'].values():
        _bound(dict(file=info['file'], sha256=info['file_sha256']))
    hashes = summary['policy_result_sha256']
    _require(len(hashes) == 810, 'Require 810 hash-bound historical neural results')
    rows = [_json(_bound(dict(file=path, sha256=digest))) for path, digest in hashes.items()]
    _require(all(row['manifest_sha256'] == m['base_manifest']['sha256'] for row in rows),
             'Historical result manifest binding mismatch')
    oracle = data['panels'][0]['horizontal'][0]['provenance']
    binding = summary['oracle_binding']
    raw = _json(_bound(binding))
    _bound(binding['solution'])
    oracle_binding_path = Path(m['base_summary']['file']).parent / 'oracle_binding.json'
    _require(_sealed(dict(file=str(oracle_binding_path), sha256=summary['oracle_binding_sha256'])) == binding,
             'Oracle binding changed')
    _require(oracle['result_binding'] == binding and raw['solution'] == binding['solution']
             and raw['manifest_sha256'] == m['base_manifest']['sha256']
             and oracle['historical_source_sha256'] == raw['source_sha256']
             and all(oracle[k] == v for k, v in raw.items() if k != 'source_sha256'),
             'Oracle raw result/solution binding mismatch')
    # Pure reconstruction of saved tables, NOT numerical replay or selection.
    _require(data == p.build_plot_data(manifest, rows, oracle), 'Base plot differs from frozen result tables')
    p._validate_plot_data(data)
    return data


def _timing(row):
    for field in ('wall_time_s', 'iterations'):
        value = row[field]
        _require(type(value) in (float, int) and np.isfinite(value) and value >= 0, f'Invalid {field}')
    _require(row['timing_clock'] == 'time.perf_counter' and row['timing_unit'] == 'seconds'
             and isinstance(row['environment'], dict) and bool(row['environment'])
             and isinstance(row['training_runtime_reference'], dict)
             and type(row['worker_count']) is int and 1 <= row['worker_count'] <= 32
             and type(row['solver_threads']) is int and row['solver_threads'] >= 1,
             'Missing runtime/concurrency provenance')
    start, end = (datetime.fromisoformat(row[k]) for k in ('start_utc', 'end_utc'))
    _require(start.tzinfo is not None and end.tzinfo is not None and end >= start, 'Invalid evaluation timestamps')
    _require(isinstance(row['phases'], dict) and bool(row['phases']), 'Missing evaluation phases')
    for phase in row['phases'].values():
        _require(phase['status'] == 'completed' and type(phase['wall_time_s']) in (float, int)
                 and np.isfinite(phase['wall_time_s']) and phase['wall_time_s'] >= 0,
                 'Incomplete/nonfinite evaluation phase')


def _journals(row):
    for name, count in (('oos', 10000), ('is', sum(row['sizes']))):
        seen = 0
        center_ids = np.repeat(np.arange(row['Ntrain']), row['sizes']) if name == 'is' else None
        with _bound(row['qp_journals'][name]).open() as stream:
            for i, line in enumerate(stream):
                d = json.loads(line)
                _require(d['row'] == i and d['phase'] == name and d['status'] == 2
                         and d['accepted'] is True, 'Incomplete/non-OPTIMAL QP journal')
                duration = d['controller_wall_time_s']
                _require(type(duration) in (int, float) and np.isfinite(duration) and duration >= 0,
                         'Invalid QP timing')
                if center_ids is not None:
                    _require(i < len(center_ids) and d['center_id'] == int(center_ids[i])
                             and d['original_sample_id'] == row['original_sample_ids'][center_ids[i]],
                             'QP journal center identity changed')
                seen += 1
        _require(seen == count, 'QP journal count mismatch')


def _new_rows(rows, method, m, digest, bindings, p):
    expected = {(method, n, s) for n in SIZES for s in SEEDS}
    _require(len(rows) == 30 and {_identity(r) for r in rows} == expected, f'Require 30 {method} rows')
    jobs = {j['id']: j for j in m['jobs']}
    indexed = {}
    for row in rows:
        key = _identity(row)
        spec = jobs[row['id']]
        _require(all(k in row and row[k] == v for k, v in spec.items()), 'Result differs from frozen policy')
        _require(_json(_bound(bindings['results'][row['id']])) == row, 'Result differs from bound JSON')
        _require(row['status'] == 'completed' and row['cached'] is False and row['test'] == m['base_test']
                 and row['manifest_sha256'] == digest, 'Failed/cached/mismatched new evaluation')
        for source in spec['input_files']:
            _copy(m, source)
        _timing(row)
        _require(row['worker_count'] == m['workers'], 'Actual worker count differs from manifest')
        if method == 'green':
            for field in ('checkpoint_sha256', 'state_sha256', 'pool_sha256', 'pool_file_sha256'):
                _require(_digest(row[field]), f'Invalid green {field}')
            _require(_sha(_copy(m, row['pool_path'])) == row['pool_file_sha256'], 'Selected pool file changed')
            sizes = row['sizes']
            _require(len(sizes) == row['Ntrain'] and all(type(v) is int and v > 0 for v in sizes),
                     'Invalid own-pool sizes')
            maxima = p._values(row['per_center_maxima'], (row['Ntrain'],), 'radius2 pool maxima')
            p._values([row['mean_oos'], row['finite_pool_mean']], (2,), 'radius2 green costs')
            count = 10000 + sum(sizes)
            _require(row['finite_pool_mean'] == float(maxima.mean()) and row['solver_threads'] == 1
                     and row['solves_attempted'] == row['solves_accepted'] == count
                     and row['status_counts'] == {'2': count}, 'Incomplete strict green evaluation')
            _journals(row)
        else:
            _require(type(row['oos']) in (float, int) and np.isfinite(row['oos']), 'Nonfinite SRO OOS')
            _require(_digest(row['policy_sha256'])
                     and _sha(_copy(m, row['policy_path'])) == row['policy_sha256'], 'SRO policy changed')
        indexed[key] = row
    return indexed


def _renderer(p):
    """Private source-checked copy; preserve every original numeric validation."""
    path = COMPARISON / 'plot_comparison.py'
    text = path.read_text()
    for before, after in (
        ('(a <= 1.5).all()', '(a <= 2.).all()'),
        ('ax.set_xlim(0, 1.5)', 'ax.set_xlim(0, 2.)'),
        ("'comparison_R10.png'", "'comparison_R10_radius2.png'"),
        ("'comparison_R10.pdf'", "'comparison_R10_radius2.pdf'"),
    ):
        _require(text.count(before) == 1, f'Plot render contract changed: {before}')
        text = text.replace(before, after)
    private = _private_module(path, text)
    private.GREEN_RADII = p.GREEN_RADII + [2.]
    private.TICKS = p.TICKS + [2.]
    _require(private.SCS_RADII == p.SCS_RADII and private.SCS_COUNTS == {50: 18, 200: 18, 800: 18},
             'SCS render contract changed')
    return private


def _append(base, green, sro, p, bindings):
    data = deepcopy(base)
    accounting = dict(cached_neural_rows=810, fresh_green_rows=30, new_sro_rows=30,
        fresh_scs_rows=0, new_selection=False, cached_common_oracle=True,
        source_caption=base['caption'], source_footnote=base['footnote'],
        bindings=deepcopy(bindings), panels={},
        timing_note='Original provenance, cached flags and timing arrays describe their source invocation. '
                    'Only current_evaluation_time_s below describes current calls; null means not measured here. '
                    'Training timings remain separate in each row; parent reports sweep elapsed time.')
    for panel in data['panels']:
        n = panel['Ntotal']
        runtime = accounting['panels'][str(n)] = {}
        for method, indexed in (('green', green), ('sro_fa', sro)):
            curve = panel[method]
            cohort = [indexed[method, n, seed] for seed in SEEDS]
            curve['radius'].append(2.)
            for field, value in (('oos', 'mean_oos' if method == 'green' else 'oos'),
                                 *((('finite_pool', 'finite_pool_mean'),) if method == 'green' else ())):
                stats = p._aggregate([[row[value] for row in cohort]])
                for k in ('per_seed', 'mean', 'p20', 'p80'):
                    curve[field][k].extend(stats[k])
            curve['provenance']['per_seed'].append(
                [dict(p._evidence(r), qp_journals=deepcopy(r['qp_journals'])) for r in cohort]
                if method == 'green' else deepcopy(cohort))
            times = [r['wall_time_s'] for r in cohort]
            if method == 'green':
                curve['provenance']['current_evaluation_time_s'].append(times.copy())
            runtime[method] = dict(cached_in_current_invocation=[True]*(len(curve['radius'])-1)+[False],
                current_evaluation_time_s=[[None]*10 for _ in curve['radius'][:-1]]+[times],
                source_evaluation_time_s=deepcopy(next(x for x in base['panels'] if x['Ntotal'] == n)
                    [method]['provenance']['current_evaluation_time_s']),
                completed_radius2_evaluation_s=dict(n=10, mean=float(np.mean(times)), median=float(np.median(times)),
                    min=min(times), max=max(times)))
        runtime['scs'] = dict(cached_in_current_invocation=True, current_evaluation_time_s=None,
            source_evaluation_time_s=deepcopy(panel['scs']['provenance']['current_evaluation_time_s']))
        runtime['common_oracle'] = dict(cached_in_current_invocation=True, current_evaluation_time_s=None,
            source_evaluation_time_s=panel['horizontal'][0]['provenance']['wall_time_s'])
    data['replay_accounting'] = accounting
    data['caption'] = ('Radius-2 extension: 810 neural results loaded from hash-verified completed history; '
        '30 fresh strict green OOS/own-pool IS replays and 30 new SRO-FA OOS evaluations. '
        'SCS stops at radius 1.5. All earlier plot values, saved validation selections and the common '
        'test-fitted oracle are unchanged. No LR/radius selection, interpolation or new oracle solve. '
        'Historical test seed 9999 is reused, not a new holdout. '
        'Archived per-seed flags/timings retain source-invocation meaning; see replay_accounting for current calls. '
        + base['is_definition'] + ' ' + base['band_definition'])
    data['footnote'] = ('Training: green 40/160/640; SCS 50/200/800. Bands: seed 20–80%, not CI. '
        'Historical test seed: 9999.\n'
        'SCS stops at 1.5; earlier curves cached unchanged. New radius 2: 30 green + 30 SRO-FA; no selection.\n'
        'Gray: cached common test-fitted clairvoyant LP; not deployable OOS or a population bound.')
    return data


def merge_and_render(base_plot_path, green_rows, sro_rows, output, bindings):
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    m = _manifest(bindings['manifest'])
    _sources(m)
    p = _plot_module()
    base = _base(m, base_plot_path, p)
    _require(set(bindings['results']) == {j['id'] for j in m['jobs']}, 'Require 60 result hash bindings')
    green = _new_rows(green_rows, 'green', m, bindings['manifest']['sha256'], bindings, p)
    sro = _new_rows(sro_rows, 'sro_fa', m, bindings['manifest']['sha256'], bindings, p)
    data = _append(base, green, sro, p, bindings)
    renderer = _renderer(p)
    renderer._validate_plot_data(data)
    # No output directory is created until the complete 60-row cohort passes.
    output = Path(output).resolve()
    _require(not output.is_relative_to(COMPARISON.resolve()), 'Cannot stage inside the original comparison')
    output.mkdir(parents=True, exist_ok=False)
    renderer.render(data, output)
    # Recheck immutable inputs/results after rendering, before the parent may publish.
    _bound(bindings['manifest'])
    _sources(m)
    for key in ('base_manifest', 'base_summary', 'base_plot'):
        _bound(m[key])
    for source in m['files']:
        _copy(m, source)
    summary = _json(_bound(m['base_summary']))
    for path, digest in summary['policy_result_sha256'].items():
        _bound(dict(file=path, sha256=digest))
    _bound(summary['oracle_binding'])
    _bound(summary['oracle_binding']['solution'])
    for info in m['base_test']['arrays'].values():
        _bound(dict(file=info['file'], sha256=info['file_sha256']))
    for binding in bindings['results'].values():
        _bound(binding)
    data['replay_accounting']['merge_and_render'] = dict(status='completed', start_utc=started,
        end_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter()-start,
        timing_clock='time.perf_counter', timing_unit='seconds', scope='verification and rendering only')
    for name in ('comparison_R10_radius2.png', 'comparison_R10_radius2.pdf'):
        _require((output / name).stat().st_size > 0, 'Missing staged figure')
    _write(output / 'plot_data.json', data)
    return data