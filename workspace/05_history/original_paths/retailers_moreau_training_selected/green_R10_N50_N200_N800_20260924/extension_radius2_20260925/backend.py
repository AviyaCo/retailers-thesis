"""Radius-2 backend only; importing this module does not load numerical modules.

Parent integration (spawn, not fork):
* At preflight, capture source_hashes() plus every parent-local source. Supply
  source_sha256, local_source_paths (absolute, explicit, includes these two files),
  protocol=binding_protocol(), workers={ccg: int, sro: int, evaluation: int}.
  An integer workers is also accepted and denotes the same count for all phases.
  Seal that manifest externally. Hash capture here is NOT preflight certification.
* configure_worker(manifest_path) is a module-level spawn initializer. Use fresh
  dedicated workers: conflicting unqualified R10 imports are rejected, not replaced.
* green_job(n, seed, winner, N_specific_output_root, provenance_path) returns the
  original compact CCG schema. winner must equal the saved LR-validation winner.
  Full AP/MP/epoch timings remain in its summary; nominal timing is cached only.
* sro_train(n, seed, exclusive_folder, manifest_path) returns a JSON-safe training
  record, also written to folder/summary.json. solution_path/solution_sha256 bind
  a non-pickle NPZ with the entire policy, partitions, ragged partition map and
  original training rows. SRO uses ALL n rows, unlike green's 4*n//5 rows.
* After the parent's training barrier, sro_test(policy, (xi1, xi2), record) accepts
  that training record as policy and raw frozen test arrays. record must supply a
  new summary_path under this extension. It is updated, persisted, and the exact
  historical mean cost is returned. Test generation/selection is never done here.

Training-only wall time, later evaluation wall time and cached nominal timing
must stay separate. No combined end-to-end time including a barrier is invented;
the parent may report a labelled phase sum and its own sweep invocation elapsed.
Native Gurobi Runtime/IterCount/BarIterCount are retained, including on failure;
Gurobi does not expose a separate barrier wall time (explicitly unavailable).
No source/docs outside these two new files are modified by this implementation.
"""
from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import types

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
HOME = Path('/data/home/aviyac')
SOURCE = CAMPAIGN / 'results/20260924T131056.490113+0000'
ORIGINAL_PROVENANCE = SOURCE / 'provenance.json'
ORIGINAL_SUMMARY = SOURCE / 'summary.json'
R3_GREEN = CAMPAIGN.parent / 'green_R3_N200_N800_20260923/green_ccg.py'
SRO_SOURCE = HOME / 'retailers_problem/R10/sro_fa/sro_fa_sweep.py'
EXPERIMENT = 'retailer_green_R10_radius2_extension_20260925'
SIZES = (50, 200, 800)
SEEDS = tuple(range(10))
RADIUS = 2.0
LOCAL_NAMES = ('backend.py', 'test_backend.py')
COMPARISON_NAMES = ('evaluation.py', 'reference.py', 'test_evaluation.py',
    'plot_comparison.py', 'test_plot.py', 'run_comparison.py', 'test_runner.py',
    'preflight.py', 'launch.py', 'launch.sh', 'README.md', 'oracle.py', 'test_oracle.py')
TRAINING_NAMES = ('source_adapter.py', 'r10_core.py', 'r10_adversary.py', 'r10_pool.py',
    'r10_strict.py', 'r10_resources.py', 'green_common.py', 'green_training.py',
    'green_ccg.py', 'run_campaign.py', 'preflight.py', 'test_green.py',
    'test_campaign.py', 'test_r10.py', 'launch.sh', 'README.md')
THREAD_NAMES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
    'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'BLIS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS')
CACHED_ORIGIN = ('cached saved nominal LR-validation winner from '
    '20260924T131056.490113+0000; not freshly trained; no radius-1.5 warm start')
GREEN_CHANGES = (
    ('centers[sample_id, :3]', 'centers[sample_id, :10]'),
    ('centers[sample_id, 3:]', 'centers[sample_id, 10:]'),
    ('centers.shape == (n_train, 6)', 'centers.shape == (n_train, 20)'),
    ('paired6 training centers', 'paired20 training centers'),
    ('parent supplies fresh per-seed training-only LR-search winner', CACHED_ORIGIN),
)
SRO_NAMES = ('generate_data', 'build_partitions', 'solve_sro_fa', 'evaluate')
green = None
_common = None
_worker_manifest = None
_path_inserted = False


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def source_paths():
    """Read-only inventory; the parent freezes hashes BEFORE dispatching jobs."""
    original = _read(ORIGINAL_PROVENANCE)['source_sha256']
    check(len(original) == 54, 'Expected original 54 training bindings')
    check({str(CAMPAIGN / n) for n in TRAINING_NAMES} <= set(original),
          'Original training source set is incomplete')
    paths = {Path(p) for p in original}
    paths.update(CAMPAIGN / 'comparison_20260925' / n for n in COMPARISON_NAMES)
    paths.update(HERE / n for n in LOCAL_NAMES)
    paths.update((ORIGINAL_PROVENANCE, ORIGINAL_SUMMARY, SRO_SOURCE))
    return tuple(sorted(paths))


def source_hashes():
    """Capture only when explicitly called by the parent's initial preflight."""
    return {str(p): sha(p) for p in source_paths()}


def _extension_protocol(original):
    protocol = deepcopy(original)
    check(protocol['sizes'] == list(SIZES) and protocol['R'] == 10
          and protocol['data_seeds'] == list(SEEDS), 'Wrong original R10 protocol')
    check(protocol['radii'] == [0., .001, .003, .01, .03, .2, .4, .7, 1.5],
          'Changed original green grid')
    protocol['radii'].append(RADIUS)
    protocol['experiment'] = EXPERIMENT
    # Retain the historical selection description verbatim. This backend runs
    # one fixed radius and does not invoke the historical radius selector.
    return protocol


def binding_protocol():
    return _extension_protocol(_read(ORIGINAL_PROVENANCE)['protocol'])


def _workers(manifest, phase):
    workers = manifest['workers']
    count = workers[phase] if isinstance(workers, dict) else workers
    check(type(count) is int and count > 0, f'Invalid {phase} worker count')
    return count


def verify_sources(manifest):
    """Fail closed on historical, comparison, local and supplied parent bindings."""
    hashes = manifest['source_sha256']
    check(str(ORIGINAL_PROVENANCE) in hashes, 'Unbound original provenance')
    check(sha(ORIGINAL_PROVENANCE) == hashes[str(ORIGINAL_PROVENANCE)],
          'Original provenance changed')
    original = _read(ORIGINAL_PROVENANCE)
    check(len(original['source_sha256']) == 54, 'Expected original 54 source bindings')
    for path, expected in original['source_sha256'].items():
        check(hashes.get(path) == expected, f'Changed original source binding: {path}')
    required = {str(p) for p in source_paths()}
    local = manifest['local_source_paths']
    check(isinstance(local, list) and len(local) == len(set(local))
          and {str(HERE / n) for n in LOCAL_NAMES} <= set(local),
          'Missing/duplicate explicit extension source set')
    for path in local:
        check(Path(path).is_absolute() and Path(path).resolve().is_relative_to(HERE),
              f'Nonlocal extension source: {path}')
    check(required | set(local) <= set(hashes), 'Incomplete supplied source manifest')
    for path, expected in hashes.items():
        check(Path(path).is_absolute() and sha(path) == expected,
              f'Frozen source changed: {path}')
    check(manifest['protocol'] == _extension_protocol(original['protocol']),
          'Extension binding protocol mismatch')
    for phase in ('ccg', 'sro', 'evaluation'):
        _workers(manifest, phase)


def _manifest(path):
    path = Path(path).resolve()
    before = sha(path)
    manifest = _read(path)
    verify_sources(manifest)
    check(sha(path) == before, 'Manifest changed during verification')
    return manifest, before


def _definitions(text, names=None):
    nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and (names is None or n.name in names)]
    if names is not None:
        check(len(nodes) == len(names) and {n.name for n in nodes} == set(names),
              'Definition coverage mismatch')
    lines = text.splitlines(keepends=True)
    return '\n\n'.join(''.join(lines[min([n.lineno] +
        [d.lineno for d in n.decorator_list]) - 1:n.end_lineno]) for n in nodes)


def _green_definitions(text):
    text = _definitions(text)
    for before, after in GREEN_CHANGES:
        check(text.count(before) == 1, f'Green fragment count changed: {before!r}')
        text = text.replace(before, after)
    return text


def _private_green(original, common, text):
    """Re-exec ALL controller definitions: function globals must be PRIVATE."""
    module = types.ModuleType(__name__ + '.green')
    module.__dict__.update(vars(original))
    module.__name__ = __name__ + '.green'
    module.PROTOCOL = _extension_protocol(common.PROTOCOL)
    module.RADII = tuple(module.PROTOCOL['radii'])
    module.WORKER_COUNT = None  # Bound by configure_worker before any job.
    exec(compile(_green_definitions(text), str(R3_GREEN), 'exec'), module.__dict__)

    def metadata(n_total, seed=None, radius=None, method=None):
        row = common.metadata(n_total, seed, radius, method)
        row.update(experiment=EXPERIMENT, extension_radius=RADIUS,
                   worker_count=module.WORKER_COUNT,
                   nominal_origin_contract=CACHED_ORIGIN,
                   radius_selection='not performed; fixed radius extension only')
        return row

    module.metadata = metadata
    module.verify_sources = verify_sources
    return module


def configure_worker(manifest_path):
    """Spawn initializer; once only, bind sources BEFORE importing frozen R10."""
    global green, _common, _worker_manifest, _path_inserted
    manifest, digest = _manifest(manifest_path)
    identity = (str(Path(manifest_path).resolve()), digest)
    if _worker_manifest is not None:
        check(_worker_manifest == identity, 'Worker cannot switch manifests')
        return
    names = ('source_adapter', 'r10_core', 'r10_adversary', 'r10_pool',
             'green_common', 'green_training', 'green_ccg')
    for name in names:
        if name in sys.modules:
            path = getattr(sys.modules[name], '__file__', None)
            check(path is not None and Path(path).resolve() == CAMPAIGN / (name + '.py'),
                  f'Unqualified import collision: {name}; use a fresh spawn worker')
    os.environ.update({name: '1' for name in THREAD_NAMES})
    os.environ.setdefault('GRB_LICENSE_FILE', str(HOME / 'gurobi.lic'))
    sys.dont_write_bytecode = True
    if not _path_inserted:
        sys.path.insert(0, str(CAMPAIGN))
        _path_inserted = True
    original = importlib.import_module('green_ccg')
    common = importlib.import_module('green_common')
    for name in names:
        check(Path(sys.modules[name].__file__).resolve() == CAMPAIGN / (name + '.py'),
              f'Unexpected imported module: {name}')
    check(common.PROTOCOL == _read(ORIGINAL_PROVENANCE)['protocol'],
          'Imported original protocol differs from saved protocol')
    check(sha(R3_GREEN) == manifest['source_sha256'][str(R3_GREEN)],
          'Original R3 controller source changed')
    private = _private_green(original, common, R3_GREEN.read_text())
        private.WORKER_COUNT = _workers(manifest, 'ccg')
    common.configure_worker()
    check(common.torch.get_num_threads() == common.torch.get_num_interop_threads() == 1,
          'Require single-threaded Torch worker')
    verify_sources(manifest)
    green, _common, _worker_manifest = private, common, identity


def _identity(n, seed):
    check(type(n) is int and n in SIZES and type(seed) is int and seed in SEEDS,
          'Require N50/200/800 and seed0..9')


def _output(path):
    path = Path(path).resolve()
    check(path != HERE and path.is_relative_to(HERE), 'Outputs must stay in extension')
    return path


def _saved_winner(n, seed, winner):
    summary = _read(ORIGINAL_SUMMARY)
    check(summary['status'] == 'completed', 'Original campaign not completed')
    jobs = summary['phases']['lr_validation']['jobs']
    matches = [j['winner'] for j in jobs if j['Ntotal'] == n and j['seed'] == seed]
    check(len(matches) == 1 and winner == matches[0],
          'Require the SAME saved nominal LR winner, never radius-1.5 weights')
    for path_key, hash_key in (('checkpoint', 'checkpoint_sha256'),
                               ('training_data_path', 'training_data_sha256')):
        check(Path(winner[path_key]).resolve().is_relative_to(SOURCE)
              and sha(winner[path_key]) == winner[hash_key], 'Saved nominal input changed')


def green_job(n, seed, winner, output_root, provenance_path):
    _identity(n, seed)
    started = time.perf_counter()
    configure_worker(provenance_path)
    _saved_winner(n, seed, winner)
    result = green.run_configuration(n, seed, RADIUS, _output(output_root),
                                     deepcopy(winner), provenance_path)
    # The inherited wall_time_s remains the CCG invocation, excluding cached
    # nominal training. This distinct timer includes wrapper source verification.
    result.update(backend_invocation_time_s=time.perf_counter() - started,
                  worker_count=green.WORKER_COUNT,
                  nominal_origin_contract=CACHED_ORIGIN,
                  cached_nominal_training_timing=deepcopy(winner['training_timing']))
    record = _read(result['summary_path'])
    record.update({k: result[k] for k in ('backend_invocation_time_s',
        'nominal_origin_contract', 'cached_nominal_training_timing')})
    _write(result['summary_path'], record)
    result['summary_sha256'] = sha(result['summary_path'])
    return result


@contextmanager
def _phase(record, name):
    start = time.perf_counter()
    phase = record.setdefault('phases', {}).setdefault(name, {})
    phase.update(started_at_utc=_utc(), status='running')
    try:
        yield
    except BaseException as error:
        phase.update(status='failed', error=dict(type=type(error).__name__, message=str(error)))
        raise
    else:
        phase['status'] = 'completed'
    finally:
        elapsed = time.perf_counter() - start
        phase.update(wall_time_s=elapsed, ended_at_utc=_utc())
        record[name + '_time_s'] = elapsed


def _capture_solver(model, record, gp):
    """Unavailable solver attributes are labelled; never substitute a result."""
    for field, attribute, cast in (
        ('solver_status', 'Status', int), ('solver_runtime_s', 'Runtime', float),
        ('iterations', 'IterCount', float), ('barrier_iterations', 'BarIterCount', int)):
        try:
            record[field] = cast(getattr(model, attribute))
        except (AttributeError, gp.GurobiError) as error:
            record[field] = None
            record.setdefault('unavailable_solver_attributes', {})[field] = str(error)


def _instrument_solve(text):
    """AST-only lifecycle/timing edits; all LP equations/status gates unchanged."""
    tree = ast.parse(_definitions(text, ('solve_sro_fa',)))
    function = tree.body[0]
    body = function.body

    def assignment(node, name):
        return (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name)

    def index(name):
        indices = [i for i, node in enumerate(body) if assignment(node, name)]
        check(len(indices) == 1, f'Changed solve assignment: {name}')
        return indices[0]

    boundary = index('solve_start')
    extraction = index('q10_value')
    objective = index('objective')
    check(ast.unparse(body[boundary]) == 'solve_start = time.time()'
          and ast.unparse(body[boundary + 1]) == 'model.optimize()'
          and ast.unparse(body[boundary + 2]) == 'solve_time = time.time() - solve_start',
          'Changed original optimize/timing sequence')
    check(extraction == boundary + 4 and isinstance(body[boundary + 3], ast.If)
          and isinstance(body[-1], ast.Return), 'Changed original status/extraction sequence')

    class ManagedResources(ast.NodeTransformer):
        def visit_Assign(self, node):
            if assignment(node, 'env') or assignment(node, 'model'):
                expected = 'gp.Env(empty=True)' if assignment(node, 'env') else 'gp.Model(env=env)'
                check(ast.unparse(node.value) == expected, 'Changed Gurobi construction')
                node.value = ast.Call(func=ast.Attribute(value=ast.Name(id='_stack', ctx=ast.Load()),
                    attr='enter_context', ctx=ast.Load()), args=[node.value], keywords=[])
            return node

        def visit_Expr(self, node):
            if ast.unparse(node) in ('model.dispose()', 'env.dispose()'):
                return None  # ExitStack owns both, including failed construction/solve.
            return node

    manager = ManagedResources()
    construction = [manager.visit(deepcopy(n)) for n in body[:boundary]]
    gate = manager.visit(deepcopy(body[boundary + 3]))

    def phase(name, nodes):
        wrapper = ast.parse(f"with _phase(_solver_record, '{name}'):\n    pass").body[0]
        wrapper.body = nodes
        return wrapper

    optimization = ast.parse('try:\n    model.optimize()\nfinally:\n'
                             '    _capture_solver(model, _solver_record, gp)').body[0]
    instrumented = [phase('construction', construction),
        phase('optimization', [optimization]),
        ast.parse("solve_time = _solver_record['optimization_time_s']").body[0], gate,
        phase('extraction', deepcopy(body[extraction:objective + 1])), deepcopy(body[-1])]
    # Capture status even when constraint construction/extraction fails, while
    # ExitStack still owns a live model. No attribute access to solution X here.
    lifecycle = ast.parse("try:\n    pass\nfinally:\n"
        "    if 'model' in locals():\n"
        "        _capture_solver(model, _solver_record, gp)").body[0]
    lifecycle.body = instrumented
    function.body = [lifecycle]
    return ast.fix_missing_locations(tree)


def sro_namespace(manifest_path):
    """Expose original four SRO definitions without importing its main/sweep."""
    manifest, _ = _manifest(manifest_path)
    for name in THREAD_NAMES:
        os.environ[name] = '1'
    os.environ.setdefault('GRB_LICENSE_FILE', str(HOME / 'gurobi.lic'))
    import gurobipy as gp
    import numpy as np
    from scipy.stats import truncnorm

    data = SRO_SOURCE.read_bytes()
    check(hashlib.sha256(data).hexdigest() == manifest['source_sha256'][str(SRO_SOURCE)],
          'SRO source changed before definition extraction')
    text = data.decode()
    module = types.ModuleType(__name__ + '.sro')
    module.__dict__.update(gp=gp, GRB=gp.GRB, np=np, truncnorm=truncnorm, time=time)
    constants = [node for node in ast.parse(text).body if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) in ('R', '(C, H, B)')]
    check(len(constants) == 2, 'Missing SRO R/cost constants')
    for node in constants:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRO_SOURCE), 'exec'),
             module.__dict__)
    check((module.R, module.C, module.H, module.B) == (10, .25, .05, .5),
          'Changed SRO dimensions/cost constants')
    exec(compile(_definitions(text, SRO_NAMES), str(SRO_SOURCE), 'exec'), module.__dict__)
    module.source_sha256 = hashlib.sha256(data).hexdigest()
    module.definition_sha256 = {name: hashlib.sha256(
        _definitions(text, (name,)).encode()).hexdigest() for name in SRO_NAMES}
    module.instrumented_solve_ast = _instrument_solve(text)
    return module


def _finite_boxes(np, xi1, xi2, n):
    check(xi1.shape == xi2.shape == (n, 10)
          and xi1.dtype == xi2.dtype == np.dtype('float64')
          and np.isfinite(xi1).all() and np.isfinite(xi2).all()
          and (xi1 >= 0).all() and (xi2 >= 0).all(), 'Malformed raw R10 training arrays')
    maxima = [float(np.max(x + RADIUS)) for x in (xi1, xi2)]
    check(all(math.isfinite(x) and x < 1e9 for x in maxima),
          'Cannot prove inherited 1e9 infinite-cell replacement equivalent')
    return dict(inherited_infinite_cell_bound=1e9, xi1_box_upper_max=maxima[0],
        xi2_box_upper_max=maxima[1], equivalent=True,
        proof='all finite box upper bounds < 1e9; min(box_hi, inf) == min(box_hi, 1e9)')


def _array_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def _base_record(n, seed, manifest, manifest_path, manifest_sha, method):
    return dict(experiment=EXPERIMENT, method=method, Ntotal=n, Ntrain=n, Nval=0,
        R=10, f=0, seed=seed, radius=RADIUS, worker_count=_workers(manifest, 'sro'),
        gurobi_threads=1, timing_clock='time.perf_counter', timing_unit='seconds',
        manifest_path=str(Path(manifest_path).resolve()), manifest_sha256=manifest_sha,
        source_sha256=deepcopy(manifest['source_sha256']), status='running', complete=False,
        solver_status=None, solver_runtime_s=None, iterations=None, barrier_iterations=None,
        barrier_time_s=None, barrier_time_status='unavailable: no separate Gurobi barrier wall timer',
        construction_time_s=None, optimization_time_s=None, extraction_time_s=None,
        partition_time_s=None, evaluation_time_s=None, total_end_to_end_time_s=None,
        total_end_to_end_time_status='separate training/evaluation invocations; parent reports labelled phase sum',
        environment=dict(platform=platform.platform(), hostname=platform.node(),
                         python=sys.version, executable=sys.executable))


def sro_train(n, seed, folder, manifest_path):
    """Exact original all-N SRO-FA LP; failures persist evidence and are re-raised."""
    _identity(n, seed)
    start, started = time.perf_counter(), _utc()
    manifest, manifest_sha = _manifest(manifest_path)
    folder = _output(folder)
    folder.mkdir(parents=True, exist_ok=False)
    record = _base_record(n, seed, manifest, manifest_path, manifest_sha, 'SRO-FA')
    record.update(started_at_utc=started, summary_path=str(folder / 'summary.json'),
        solution_path=None, solution_sha256=None, data_access='training only; no held-out reads',
        timing_scope='training invocation including source checks, generation, partitions, LP and persistence; final summary write excluded')
    try:
        original = sro_namespace(manifest_path)
        np = original.np
        record.update(sro_source_sha256=original.source_sha256,
                      sro_definition_sha256=original.definition_sha256)
        with _phase(record, 'data_generation'):
            rng = np.random.get_state()
            try:
                xi1, xi2 = original.generate_data(n, seed)
            finally:
                np.random.set_state(rng)
            record['infinite_cell_bound_proof'] = _finite_boxes(np, xi1, xi2, n)
            record['training_array_sha256'] = dict(xi1=_array_sha(xi1), xi2=_array_sha(xi2))
        with _phase(record, 'partition'):
            partitions, partition_map = original.build_partitions(xi1, RADIUS)
            record['K'] = len(partitions)
        namespace = dict(vars(original))
        with ExitStack() as stack:
            namespace.update(_stack=stack, _solver_record=record, _phase=_phase,
                             _capture_solver=_capture_solver)
            exec(compile(original.instrumented_solve_ast, str(SRO_SOURCE), 'exec'), namespace)
            q10, q1r, q2, objective, _ = namespace['solve_sro_fa'](
                xi1, xi2, RADIUS, partitions, partition_map)
        check(np.isfinite(q10) and np.isfinite(q1r).all()
              and np.isfinite(q2).all() and np.isfinite(objective), 'Nonfinite SRO solution')
        check(q1r.shape == (10,) and q2.shape == (10, len(partitions)), 'Malformed SRO solution')
        with _phase(record, 'policy_persistence'):
            solution = folder / 'policy.npz'
            offsets = np.concatenate(([0], np.cumsum([len(row) for row in partition_map])))
            indices = np.asarray([p for row in partition_map for p in row], dtype=np.int64)
            with solution.open('xb') as stream:
                np.savez(stream, Q10=np.asarray(q10), Q1r=q1r, Q2=q2,
                    part_lo=np.asarray([p[0] for p in partitions]),
                    part_hi=np.asarray([p[1] for p in partitions]),
                    partition_map_offsets=offsets, partition_map_indices=indices,
                    original_sample_ids=np.arange(n), xi1_train=xi1, xi2_train=xi2)
            record.update(solution_path=str(solution), solution_sha256=sha(solution),
                          objective=float(objective), Q10=float(q10), Q1r=q1r.tolist())
        verify_sources(manifest)
        check(sha(manifest_path) == manifest_sha, 'Manifest changed during SRO training')
        record.update(status='completed', complete=True)
    except BaseException as error:
        record.update(status='failed', complete=False,
                      error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        elapsed = time.perf_counter() - start
        record.update(ended_at_utc=_utc(), wall_time_s=elapsed, training_only_time_s=elapsed,
                      wall_time_human=f'{elapsed:.3f} s / {elapsed / 60:.3f} min')
        _write(record['summary_path'], record)
    return record


def sro_test(policy, testarrays, record):
    """Exact frozen evaluate; only xi1 selects a cell. No test-array generation."""
    destination = _output(record['summary_path'])
    check(not destination.exists(), 'Evaluation record must be new, never overwrite training')
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    record.update(started_at_utc=_utc(), status='running', complete=False,
                  method='SRO-FA evaluation', timing_clock='time.perf_counter', timing_unit='seconds')
    try:
        check(policy['status'] == 'completed' and policy['complete'], 'Require completed SRO training')
        _identity(policy['Ntotal'], policy['seed'])
        check(policy['radius'] == RADIUS and policy['R'] == 10, 'Wrong SRO policy identity')
        manifest, digest = _manifest(policy['manifest_path'])
        check(digest == policy['manifest_sha256'], 'Policy manifest changed')
        check(_read(policy['summary_path']) == policy, 'Supplied training record differs from persisted record')
        original = sro_namespace(policy['manifest_path'])
        np = original.np
        check(sha(policy['solution_path']) == policy['solution_sha256'], 'SRO policy changed')
        xi1, xi2 = testarrays
        check(isinstance(xi1, np.ndarray) and isinstance(xi2, np.ndarray)
              and xi1.shape == xi2.shape == (10000, 10)
              and xi1.dtype == xi2.dtype == np.dtype('float64')
              and np.isfinite(xi1).all() and np.isfinite(xi2).all()
              and (xi1 >= 0).all() and (xi2 >= 0).all(), 'Require frozen raw float64 R10 test arrays')
        record.update(experiment=EXPERIMENT, Ntotal=policy['Ntotal'], Ntrain=policy['Ntrain'],
            seed=policy['seed'], radius=RADIUS, R=10, Ntest=len(xi1),
            worker_count=_workers(manifest, 'evaluation'), gurobi_threads=1,
            solution_path=policy['solution_path'], solution_sha256=policy['solution_sha256'],
            training_summary_path=policy['summary_path'], training_summary_sha256=sha(policy['summary_path']),
            test_array_sha256=dict(xi1=_array_sha(xi1), xi2=_array_sha(xi2)),
            test_protocol='parent supplies frozen paired test seed9999; no generation or selection here',
            source_sha256=manifest['source_sha256'],
            training_only_time_s=policy['training_only_time_s'], training_timing_scope='previous invocation',
            total_end_to_end_time_s=None,
            total_end_to_end_time_status='unmeasured across separate invocations; barrier excluded from phase sum')
        with np.load(policy['solution_path'], allow_pickle=False) as saved:
            partitions = list(zip(saved['part_lo'], saved['part_hi']))
            with _phase(record, 'evaluation'):
                value = original.evaluate(xi1, xi2, float(saved['Q10']),
                                          saved['Q1r'], saved['Q2'], partitions)
        check(math.isfinite(value), 'Nonfinite SRO evaluation')
        check(record['test_array_sha256'] == dict(xi1=_array_sha(xi1), xi2=_array_sha(xi2)),
              'Evaluation mutated test arrays')
        check(sha(policy['solution_path']) == policy['solution_sha256'], 'Evaluation policy changed')
        verify_sources(manifest)
        check(sha(policy['manifest_path']) == digest, 'Manifest changed during evaluation')
        record.update(oos=value, status='completed', complete=True)
        return value
    except BaseException as error:
        record.update(status='failed', complete=False,
                      error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        elapsed = time.perf_counter() - start
        record.update(ended_at_utc=_utc(), wall_time_s=elapsed,
                      evaluation_invocation_time_s=elapsed,
                      wall_time_human=f'{elapsed:.3f} s / {elapsed / 60:.3f} min')
        _write(destination, record)