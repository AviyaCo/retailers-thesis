"""Post-barrier fixed-policy replay and green-only descriptive reporting.

No data files are opened on import. Parent owns the training barrier and timing
table. Required compact['evaluation_gate'] is {'path': ..., 'sha256': ...}:
the sealed JSON has status='sealed', allow_test_regeneration=True,
source_bindings=source_bindings(), and configurations=[all 810 native compact
results, each additionally carrying kind]. Seal ONLY after all have converged.
This is an explicit authorization to regenerate the historical, already-seen
10000/9999 scenarios, NOT a fresh confirmatory test. No training is imported.

publish(out, rows, run_summary) requires that same evaluation_gate in run_summary.
Rows are evaluate_configuration return values (persisted result + hash). Parent
must supply/publish nominal CV, full-N initialization/training, AP/MP and sweep
timings; this adapter aggregates measured evaluation wall times only.
"""
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/data/home/aviyac/retailers_experiment')
ORIGINAL = PROJECT / 'workspace/05_history/original_paths'
CAMPAIGN = ORIGINAL / 'retailers_moreau_training_selected'
COMPARE = ORIGINAL / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison/compare.py'
POOL = CAMPAIGN / 'green_R3_N200_N800_20260923/comparison_20260924/evaluation.py'
CCG = COMPARE.parent.parent / 'ccg.py'
GREEN_CCG = CAMPAIGN / 'green_R3_N200_N800_20260923/green_ccg.py'
SHIPPING = CAMPAIGN / 'shipping_comparison_20260928'
R10_REFERENCE = CAMPAIGN / 'green_R10_N50_N200_N800_20260924/comparison_20260925/reference.py'
KINDS = ('R3', 'R10', 'shipping_R3')
SIZES = (50, 200, 800)
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
SEEDS = tuple(range(10))
NOTICE = ('Historical 10000-scenario test, seed 9999, already seen. Full-N refit after '
          'train-only five-fold nominal LR CV. Fixed-grid displayed-mean minimum is '
          'descriptive hindsight, not practical radius selection or a fresh confirmatory test.')
LOCKS = {
    CCG: 'b22c150eb163ee04d55077a033bd12018c0c503ec321af71c793f33a933ea5d9',
    GREEN_CCG: '73b389ca177e694e5d0e7ebe1727e59850752f649a63e8c58cda35cc29ba7afc',
    COMPARE: '8285365c2913687a6c8769fee720b56237c785cb9f6d37d81b3a82f8c9d1b366',
    POOL: '794d7d6f816f0afcc4373c5acaad0f949e94b7b28ab269892a77dcfc04ac1a22',
    SHIPPING / 'common.py': 'f0486a320e7afb1601ee4072cd7ee4e42307f668fae7cca7d29dd4246470c735',
    SHIPPING / 'evaluation.py': 'b41632d9e5749d4590a255b7dcd8707b5204d89af2a6469097d4539a3fed76ac',
    R10_REFERENCE: '3762749519a43e9a6d42e0bb569f04253d8ea8b8da3bcd64da9f3d7dd592105f',
    PROJECT / 'src/evaluate.py': '4a7e1f3754346e8f6d4e54315ac384e0e05a4510880a08c1496bae7da0864f25',
    PROJECT / 'src/plot.py': '2e875428749b3c4029998232690a3f1be15ae8feb64cdc59b7b1d8bc74c64124',
    PROJECT / 'src/artifacts.py': 'f28cbf0d5717293b54ae9fafea05d78b31780074856e5435268b18cdaec08d45',
    PROJECT / 'src/__init__.py': 'b9fa827bafb8fee123b520072d95cfca83e58e936cd5768e10650607e0e7628a',
}
IDENTITY = ('Ntotal', 'seed', 'radius', 'summary_path', 'summary_sha256',
            'selected_checkpoint', 'selected_checkpoint_sha256',
            'selected_state_sha256', 'selected_candidate_id', 'selected_ap_reference')


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_bindings():
    """Verify frozen transitive local sources; hash this adapter; no data access."""
    for path, expected in LOCKS.items():
        _check(_sha(path) == expected, f'Changed original source: {path}')
    return {**{str(p): h for p, h in LOCKS.items()}, str(ROOT / 'reporting.py'): _sha(ROOT / 'reporting.py')}


def _definitions(path, names, namespace, changes=()):
    """Reuse decorated definitions, excluding imports and experiment initializers."""
    _check(_sha(path) == LOCKS[path], f'Changed definition source: {path}')
    text = path.read_text()
    nodes = [n for n in ast.parse(text).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    _check(len(nodes) == len(names), 'Missing/duplicate original definition')
    lines = text.splitlines(keepends=True)
    text = '\n\n'.join(''.join(lines[min([n.lineno] + [d.lineno for d in n.decorator_list])-1:n.end_lineno])
                       for n in nodes)
    for old, new in changes:
        _check(text.count(old) == 1, f'Generator dimension fragment changed: {old}')
        text = text.replace(old, new)
    exec(compile(text, str(path), 'exec'), namespace)


def _shared(name):
    # Never use parent worker's generic `common`, `src`, or `backend` modules.
    package = '_green_fullN_reporting_shared'
    for suffix in ('', '.artifacts', '.' + name):
        key = package + suffix
        if key not in sys.modules:
            path = PROJECT / 'src' / (suffix[1:] + '.py' if suffix else '__init__.py')
            spec = importlib.util.spec_from_file_location(key, path,
                submodule_search_locations=[str(path.parent)] if not suffix else None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[key] = module
            spec.loader.exec_module(module)
    return sys.modules[package + '.' + name]


def _grid(rows):
    expected = {(k, n, s, r) for k in KINDS for n in SIZES for s in SEEDS for r in RADII}
    keys = [(r['kind'], r['Ntotal'], r['seed'], r['radius']) for r in rows]
    _check(len(keys) == 810 and len(set(keys)) == 810 and set(keys) == expected,
           'Require complete 810 grid: three kinds, three N, nine radii, ten seeds')
    _check(all(r['status'] == 'completed' and r['converged'] is True and r['complete'] is True
               for r in rows), 'Failed/incomplete/nonconverged training grid')


def _gate(info):
    _check(_sha(info['path']) == info['sha256'], 'Changed training barrier seal')
    gate = _read(info['path'])
    _check(gate['status'] == 'sealed' and gate['allow_test_regeneration'] is True,
           'Explicit post-barrier test regeneration authorization required')
    _grid(gate['configurations'])
    _check(gate['source_bindings'] == source_bindings(), 'Unbound reporting sources')
    return gate


def _identity(kind, compact, gate):
    _check(kind in KINDS, 'Unknown experiment kind')
    match = [r for r in gate['configurations'] if (r['kind'], r['Ntotal'], r['seed'], r['radius'])
             == (kind, compact['Ntotal'], compact['seed'], compact['radius'])]
    _check(len(match) == 1 and all(match[0][k] == compact[k] for k in IDENTITY),
           'Policy is not the sealed training selection')
    _check(_sha(compact['summary_path']) == compact['summary_sha256'], 'Changed training summary')
    summary = _read(compact['summary_path'])
    _check(all(summary[k] == compact[k] for k in IDENTITY if k != 'summary_sha256'),
           'Compact/summary selection mismatch')
    _check(summary['status'] == 'completed' and summary['converged'] is True and summary['complete'] is True,
           'Unfinished training selection')
    _check(summary['Ntrain'] == summary['Ntotal'] and summary['Nval'] == 0, 'Not a full-N refit')
    _check(_sha(compact['selected_checkpoint']) == compact['selected_checkpoint_sha256'], 'Changed checkpoint')
    return summary


def _runtime(kind):
    import numpy as np
    import torch
    import gurobipy as gp
    from gurobipy import GRB
    from scipy.stats import truncnorm

    _check(torch.get_default_dtype() == torch.float32 and torch.get_num_threads() == 1
           and torch.get_num_interop_threads() == 1, 'Parent must configure float32 and single-thread workers')
    namespace = dict(np=np, torch=torch, nn=torch.nn, gp=gp, GRB=GRB,
                     time=time, hashlib=hashlib, json=json, Path=Path,
                     datetime=datetime, timezone=timezone, contextmanager=contextmanager,
                     truncnorm=truncnorm, PHYSICAL_TOL=1e-6, check=_check)
    changes = (('size=(n, 3)', 'size=(n, 10)'),) if kind == 'R10' else ()
    _definitions(COMPARE, ('gen', 'utc', 'phase'), namespace, changes)
    _definitions(POOL, ('pool_digest', 'finite_pool_cost'), namespace)
    namespace.update(MEMBERSHIP_TOLERANCE=1e-9,
                     PROTOCOL=dict(green_gap_floor=1e-9, green_gap_tolerance=.01))
    _definitions(CCG, ('check_membership', 'raw_data_hash'), namespace,
                 (('points.shape[1] != 6', 'points.shape[1] != 20'),
                  ('paired6 vectors', 'paired20 vectors')) if kind == 'R10' else ())
    _definitions(GREEN_CCG, ('_finite', 'terminal_decision', 'choose_candidate'), namespace)
    shared = _shared('evaluate')
    # Private copy only: solver equations/settings unchanged, acceptance <=1e-6.
    shared.PHYSICAL_TOL = 1e-6
    if kind == 'shipping_R3':
        _definitions(COMPARE, ('Model', 'strict_project', 'scenario_costs'), namespace)
        base = SimpleNamespace(**namespace)
        shipping = dict(namespace, c=base)
        _definitions(SHIPPING / 'common.py', ('scenario_costs', 'load_model'), shipping)
        common = SimpleNamespace(**shipping, sha=_sha)
        evaluator = dict(json=json, np=np, check=_check, c=common,
                         NEURAL_FEE='0.1 * count(raw float64 QP q2 > 0), no clipping or tolerance')
        _definitions(SHIPPING / 'evaluation.py', ('evaluate',), evaluator)
        return shared, namespace, common.load_model, evaluator['evaluate']
    return shared, namespace, lambda p, _: shared.load_model(p, 10 if kind == 'R10' else 3), None


def _own_pool(summary, ns, R):
    np = ns['np']
    selected = summary['selected_candidate_id']
    candidates = summary['iterations']
    stopping = summary['stopping_candidate_id']
    _check(summary['selection_rule'] == 'stopping_checkpoint', 'Require stopping_checkpoint selection')
    _check(type(selected) is int and type(stopping) is int
           and 0 <= selected == stopping < len(candidates),
           'Invalid selected candidate index')
    ns['choose_candidate'](candidates, stopping)  # Frozen contiguous/finite prefix checks, not selection.
    for index, row in enumerate(candidates[:stopping + 1]):
        _check(type(row['candidate_id']) is int and row['candidate_id'] == index
               and row['assessed'] is True, 'Require contiguous fully assessed prefix')
        decision = ns['terminal_decision'](row)
        _check(row['green_pass'] is decision['green_pass']
               and row['green_gap'] == decision['green_gap']
               and decision['green_pass'] == (index == stopping),
               'STOP must be the first strict green-gap pass')
    candidate = candidates[selected]
    _check(candidate['candidate_id'] == selected and candidate['assessed'] is True,
           'Incomplete selected candidate')
    for k in ('state_sha256', 'checkpoint_sha256', 'ap_reference'):
        _check(candidate[k] == summary['selected_' + k], f'Selected candidate mismatch: {k}')
    path = Path(candidate['checkpoint']).parent / 'pool.json'
    _check(_sha(path) == candidate['pool_file_sha256'], 'Changed selected candidate pool')
    data = _read(path)
    pools = [np.asarray(p, dtype=np.float64) for p in data['pools']]
    _check(len(pools) == summary['Ntotal'] and sorted(data['original_sample_ids']) == list(range(summary['Ntotal'])),
           'Selected pool does not cover full N')
    _check(all(p.ndim == 2 and p.shape[1] == 2 * R and len(p) > 0
             and np.isfinite(p).all() for p in pools), 'Invalid selected pool arrays')
    _check(ns['pool_digest'](pools) == data['pool_sha256'] == candidate['pool_sha256'], 'Changed pool digest')
    training_path = Path(summary['summary_path']).parent / 'inputs/training_data.json'
    _check(_sha(training_path) == summary['training_data_sha256'], 'Changed training centers')
    training = _read(training_path)
    centers = np.asarray(training['centers'], dtype=np.float64)
    ids = training['original_sample_ids']
    _check(centers.shape == (summary['Ntotal'], 2 * R) and np.isfinite(centers).all()
           and (centers >= 0).all() and all(type(i) is int for i in ids)
           and sorted(ids) == list(range(summary['Ntotal']))
           and data['original_sample_ids'] == ids, 'Invalid/misaligned training centers')
    canonical = centers[np.argsort(ids)]
    _check(ns['raw_data_hash'](dict(xi1=canonical[:, :R], xi2=canonical[:, R:]))
           == training['data_sha256'] == summary['data_sha256'], 'Changed raw training data hash')
    radius = ns['_finite'](summary['radius'], 'radius')
    _check(radius >= 0, 'Negative radius')
    for pool, center in zip(pools, centers):
        ns['check_membership'](pool, center, radius)  # Raw coordinates, native 1e-9; never clip.
    return pools, dict(path=str(path), sha256=candidate['pool_file_sha256'],
                      pool_sha256=candidate['pool_sha256'], training_data_path=str(training_path),
                      training_data_sha256=summary['training_data_sha256'])


def evaluate_configuration(kind, compact, out):
    """Return full result plus result_path/result_sha256; failures persist and raise.

    Caller must pass an absent output directory. No own-pool skipping/fallback:
    a missing or invalid selected-candidate pool fails the configuration.
    """
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    gate = _gate(compact['evaluation_gate'])  # Before numerical imports or test generation.
    summary = _identity(kind, compact, gate)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(kind=kind, **{k: compact[k] for k in IDENTITY}, status='running',
                  complete=False, converged=True, started_at_utc=started, phases={},
                  evaluation_gate=compact['evaluation_gate'], source_bindings=gate['source_bindings'],
                  test_notice=NOTICE, physical_tolerance=1e-6, solver_threads=1,
                  worker_count=gate.get('evaluation_workers'), environment=dict(
                      platform=platform.platform(), hostname=platform.node()), mean_oos=None,
                  finite_pool_mean=None, artifacts={}, timing_clock='time.perf_counter', timing_unit='seconds')
    try:
        shared, ns, load, shipping_evaluate = _runtime(kind)
        np = ns['np']
        R = 10 if kind == 'R10' else 3
        phase = ns['phase']
        with phase(record['phases'], 'input_loading'):
            model = load(compact['selected_checkpoint'], 'green')
            _check(shared.state_hash(model) == compact['selected_state_sha256'], 'Changed learned state')
            pools, record['own_pool_binding'] = _own_pool(summary, ns, R)
        with phase(record['phases'], 'historical_test_regeneration'):
            rng = np.random.get_state()
            try:
                x1, x2 = ns['gen'](10000, 9999)
            finally:
                np.random.set_state(rng)
            test = dict(n_test=10000, seed=9999, already_seen=True, arrays={})
            for name, array in (('xi1', x1), ('xi2', x2)):
                _check(array.shape == (10000, R) and array.dtype == np.float64
                       and np.isfinite(array).all() and (array >= 0).all(), 'Invalid original test generator output')
                path = folder / f'{name}.npy'
                np.save(path, array, allow_pickle=False)
                test['arrays'][name] = dict(file=str(path), file_sha256=_sha(path),
                    array_sha256=hashlib.sha256(array.tobytes()).hexdigest(), shape=list(array.shape), dtype=str(array.dtype))
            record['test'] = test
            seal = dict(checkpoint=compact['selected_checkpoint'], checkpoint_sha256=compact['selected_checkpoint_sha256'],
                        state_sha256=compact['selected_state_sha256'], test=test, evaluation_gate=compact['evaluation_gate'])
            _write(folder / 'input_seal.json', seal)
            input_seal = dict(path=str(folder / 'input_seal.json'), sha256=_sha(folder / 'input_seal.json'))
        points = np.concatenate(pools)
        sizes = [len(p) for p in pools]
        for role, a, b, grouping in (('oos', x1, x2, None), ('own_pool', points[:, :R], points[:, R:], sizes)):
            if shipping_evaluate is not None:
                raw = dict(phases={}, status_counts={}, solves_attempted=0, solves_accepted=0,
                           saved_native_finite_pool_mean=summary['selected_restricted_mean'])
                record[role] = raw
                shipping_evaluate(model, a, b, role, raw, folder, grouping)
                _check(raw['solves_attempted'] == raw['solves_accepted'] == len(a)
                       and raw['status_counts'] == {'2': len(a)}, 'Incomplete shipping replay')
                mean = raw['mean_oos' if grouping is None else 'finite_pool_mean']
            else:
                raw = dict(role=role, input_seal=input_seal, kind=kind, Ntotal=compact['Ntotal'],
                           seed=compact['seed'], radius=compact['radius'], worker_count=record['worker_count'])
                record[role] = raw
                shared.evaluate_policy(model, a, b, folder / role, raw, grouping)
                _check(raw['status'] == 'completed' and raw['solves_attempted'] == raw['solves_accepted'] == len(a)
                       and raw['status_counts'] == {'2': len(a)}, 'Incomplete fixed-policy replay')
                mean = raw['mean']
            record['mean_oos' if grouping is None else 'finite_pool_mean'] = mean
        _check(shared.state_hash(model) == compact['selected_state_sha256'], 'Replay changed policy')
        _identity(kind, compact, _gate(compact['evaluation_gate']))
        binding = record['own_pool_binding']
        _check(_sha(binding['path']) == binding['sha256'], 'Pool changed during replay')
        _check(_sha(binding['training_data_path']) == binding['training_data_sha256'],
               'Training centers changed during replay')
        record['artifacts'] = {str(p): _sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        record.update(status='completed', complete=True)
    except BaseException as error:
        record.update(status='failed', complete=False, mean_oos=None, finite_pool_mean=None,
                      error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        record.update(ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter()-start)
        record['wall_time_human'] = f"{record['wall_time_s']:.3f} s / {record['wall_time_s']/60:.3f} min"
        _write(folder / 'result.json', record)
    return dict(record, result_path=str(folder / 'result.json'), result_sha256=_sha(folder / 'result.json'))


def _plot_data(kind, rows):
    import numpy as np
    from textwrap import fill
    R = 10 if kind == 'R10' else 3
    panels, hindsight = [], []
    lookup = {(r['Ntotal'], r['radius'], r['seed']): r for r in rows if r['kind'] == kind}
    test = next(iter(lookup.values()))['test']
    _check(all(r['test']['arrays'][name]['array_sha256'] == test['arrays'][name]['array_sha256']
               for r in lookup.values() for name in ('xi1', 'xi2')), 'Unpaired test arrays within kind')
    for n in SIZES:
        curve = dict(radius=list(RADII), seeds=list(SEEDS), label=f'green {kind}')
        for field, dest in (('mean_oos', 'oos'), ('finite_pool_mean', 'finite_pool')):
            values = np.asarray([[lookup[n, rho, seed][field] for seed in SEEDS] for rho in RADII], dtype=float)
            _check(values.shape == (9, 10) and np.isfinite(values).all(), 'Invalid/incomplete displayed grid')
            curve[dest] = dict(per_seed=values.tolist(), mean=values.mean(axis=-1).tolist(),
                               p20=np.percentile(values, 20, axis=-1).tolist(), p80=np.percentile(values, 80, axis=-1).tolist())
        means = curve['oos']['mean']
        best = min(means)
        hindsight.append(dict(kind=kind, Ntotal=n, mean_oos=best,
            radii=[r for r, value in zip(RADII, means) if value == best],
            label='Descriptive hindsight minimum of displayed ten-seed mean; fixed grid'))
        panels.append(dict(Ntotal=n, label=f'{kind} · N={n}', green=curve))
    return dict(status='completed', seeds=list(SEEDS), panels=panels, test=test,
                footnote=fill(f'{kind}: {NOTICE} Own-pool IS is a finite-pool objective, not a robust certificate.', width=155)), hindsight, R


def publish(out, evaluation_rows, run_summary):
    """Validate all 810 immutable results, render three original green-only plots.

    Return {'artifacts': [paths], 'report': dict, 'report_path': str,
    'report_sha256': str}. No per-seed radius oracle or radius selection.
    Parent's run_summary is preserved verbatim, including its full timing table.
    """
    start = time.perf_counter()
    gate = _gate(run_summary['evaluation_gate'])
    _grid(evaluation_rows)
    rows = []
    for row in evaluation_rows:
        _check(_sha(row['result_path']) == row['result_sha256'], 'Changed evaluation result')
        saved = _read(row['result_path'])
        _check(saved == {k: v for k, v in row.items() if k not in ('result_path', 'result_sha256')},
               'Evaluation row differs from persisted result')
        _check(saved['evaluation_gate'] == run_summary['evaluation_gate']
               and saved['source_bindings'] == gate['source_bindings'], 'Mixed evaluation provenance')
        _identity(saved['kind'], saved, gate)
        _check(bool(saved['artifacts']), 'Missing raw evaluation evidence')
        for path, sha in saved['artifacts'].items():
            _check(_sha(path) == sha, f'Changed raw evaluation artifact: {path}')
        _check(saved['test']['n_test'] == 10000 and saved['test']['seed'] == 9999
               and saved['test']['already_seen'] is True, 'Unexpected historical test protocol')
        rows.append(saved)
    # Validate complete data before making any plots; never omit failures.
    supplied = [_plot_data(kind, rows) for kind in KINDS]
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=False)
    renderer = _shared('plot')
    artifacts, hindsight, timings = [], [], []
    stats = _shared('artifacts').stats
    for kind, (data, minima, R) in zip(KINDS, supplied):
        artifacts.extend(str(p) for p in renderer.render(data, folder / kind, R))
        hindsight.extend(minima)
        for n in SIZES:
            for rho in RADII:
                values = [r['wall_time_s'] for r in rows if (r['kind'], r['Ntotal'], r['radius']) == (kind, n, rho)]
                _check(all(v >= 0 for v in values), 'Invalid evaluation timings')
                timing = stats(values)
                timings.append(dict(kind=kind, Ntotal=n, radius=rho, method='strict_fixed_policy_evaluation',
                    unit='seconds', **timing, mean_human=f"{timing['mean']:.3f} s / {timing['mean']/60:.3f} min"))
    report = dict(status='completed', notice=NOTICE, completed_count=len(rows),
        failed_or_incomplete=[], hindsight_mean_curve_minima=hindsight, evaluation_runtime_summary=timings,
        training_and_sweep_runtime_owner='parent run_summary; no concurrent times summed as elapsed',
        run_summary=run_summary, source_bindings=source_bindings(),
        evaluation_results=[dict(path=r['result_path'], sha256=r['result_sha256']) for r in evaluation_rows],
        publication_wall_time_s=time.perf_counter()-start)
    path = folder / 'report.json'
    _write(path, report)
    artifacts.append(str(path))
    return dict(artifacts=artifacts, report=report, report_path=str(path), report_sha256=_sha(path))