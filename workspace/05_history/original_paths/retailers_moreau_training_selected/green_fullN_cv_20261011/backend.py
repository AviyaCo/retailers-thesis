"""Fresh full-N bindings of frozen GREEN implementations; one kind per process.

Only sample guards, raw-data return, metadata and the approved STOP/advisory
bindings differ. No numerical solver, trainer or CCG loop is implemented here.
Run through the consolidation namespace wrapper, never by relocating sources.
"""
import ast
from copy import deepcopy
import hashlib
import importlib
from pathlib import Path
import sys
from types import FunctionType, SimpleNamespace

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
SIZES = (50, 200, 800)
TRAIN_SIZES = (40, 160, 640, 50, 200, 800)
_backend = None
_requested_kind = None
_configured = False


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _definitions(path, names, namespace, bindings, changes=()):
    """Same definition-only, exact-fragment adaptation as the frozen adapters."""
    path = Path(path)
    if bindings.get(str(path)) != _digest(path):
        raise RuntimeError(f'Unbound/changed source: {path}')
    text = path.read_text()
    nodes = [n for n in ast.parse(text).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError(f'Missing definitions: {path}')
    lines = text.splitlines(keepends=True)
    text = '\n\n'.join(''.join(lines[min([n.lineno] +
        [d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
    for old, new in changes:
        if text.count(old) != 1:
            raise RuntimeError(f'Ambiguous adaptation in {path}: {old!r}')
        text = text.replace(old, new)
    exec(compile(text, str(path), 'exec'), namespace)


def _controller_namespace(module):
    """Rebind existing functions, not the original module or its globals."""
    scope = dict(vars(module))
    for name, value in vars(module).items():
        if isinstance(value, FunctionType) and value.__globals__ is vars(module):
            clone = FunctionType(value.__code__, scope, value.__name__,
                                 value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            scope[name] = clone
    return scope


def load_backend(kind):
    global _backend, _requested_kind
    if kind not in ('R3', 'R10', 'shipping_R3'):
        raise ValueError(f'Unknown backend kind: {kind}')
    if _requested_kind is not None and _requested_kind != kind:
        raise RuntimeError('Use a fresh process for each backend kind')
    if _backend is not None:
        return _backend
    _requested_kind = kind
    sys.dont_write_bytecode = True
    green = PARENT / 'green_R3_N200_N800_20260923'
    r10 = PARENT / 'green_R10_N50_N200_N800_20260924'
    nominal = PARENT / 'shipping_nominal_R3_N50_N200_N800_20260924'
    live = PARENT / 'shipping_live_green_20260926'
    advisory_root = PARENT / 'shipping_advisory_continue_20260928'
    directories = [green, PARENT / 'N50_R3_train40_val10']
    directories += [r10] if kind == 'R10' else [nominal, live, advisory_root] if kind == 'shipping_R3' else []
    bindings = {str(p): _digest(p) for d in directories for p in d.iterdir()
                if p.suffix in ('.py', '.sh') or p.name == 'README.md'}
    bindings[str(ROOT / 'backend.py')] = _digest(ROOT / 'backend.py')
    primary = nominal if kind == 'shipping_R3' else r10 if kind == 'R10' else green
    for name in ('source_adapter', 'green_common', 'green_training', 'green_ccg',
                 'shipping_common', 'shipping_training', 'shipping_live_common', 'advisory'):
        if name in sys.modules:
            raise RuntimeError(f'Numerical module already loaded; require fresh process: {name}')
    sys.path.insert(0, str(primary))
    if kind == 'shipping_R3':
        common = importlib.import_module('shipping_common')
        training = importlib.import_module('shipping_training')
        models = importlib.import_module('shipping_model')
        sys.path.insert(0, str(live))
        controller = importlib.import_module('shipping_ccg')
        live_common = importlib.import_module('shipping_live_common')
        sys.path.insert(0, str(advisory_root))
        advisory = importlib.import_module('advisory')
        continuation_common = importlib.import_module('continue_common')
        original_verify = live_common.verify_sources
        inherited = deepcopy(live_common.PROTOCOL)
        core = SimpleNamespace(**vars(common.core))
        core.make_model, core.model_from_state = models.make_model, models.model_from_state
        core.digest = _digest
        freezes = [live_common.FROZEN, continuation_common.FROZEN]
    else:
        common = importlib.import_module('green_common')
        training = importlib.import_module('green_training')
        controller = importlib.import_module('green_ccg')
        core = common.core
        original_verify = common.verify_sources
        inherited = deepcopy(common.PROTOCOL)
        freezes = [common.FROZEN]
    if Path(common.__file__).resolve().parent != primary:
        raise RuntimeError('Numerical import collision')
    for freeze in freezes:
        for p, h in freeze.items():
            if p in bindings and bindings[p] != h:
                raise RuntimeError(f'Conflicting source freeze: {p}')
            bindings[p] = h
    # Include actually imported transitive Python sources, including native wrappers.
    for module in tuple(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and p.endswith('.py') and str(p).startswith('/data/home/aviyac/'):
            path = Path(p).resolve()
            if path == ROOT / 'test_backend.py' or path.is_relative_to(ROOT):
                continue
            bindings.setdefault(str(path), _digest(path))
    for p, h in bindings.items():
        if _digest(p) != h:
            raise RuntimeError(f'Frozen source changed during import: {p}')
    np, torch = common.np, common.torch
    scope = _controller_namespace(controller)
    protocol = inherited
    for obsolete in ('not_run', 'initialization_barrier', 'nominal_arms',
                     'full_CCG_launched', 'radius_selection_run', 'continuation_scope'):
        protocol.pop(obsolete, None)
    protocol.update(experiment=ROOT.name + '_' + kind, sizes=list(SIZES),
        train_fraction=1., validation_fraction=0., split_seed=None,
        data_seeds=list(range(10)), network_seed=0, refit=True, resume=False,
        cv_folds=5, cv_scope='nominal LR only; fold-local SAA and fresh model/trainer',
        nominal_lr_selection='minimum five-fold mean native hard cost; smaller LR exact tie',
        nominal_origin='fresh full-N SAA and full-N refit after five-fold nominal LR CV',
        ccg_selection='stopping_checkpoint',
        validation='five-fold nominal LR CV only; no held-out rows inside full-N MP/CCG',
        radius_selection='none; run all nine original GREEN radii independently',
        test='not accessed', solver_fallback=False)
    advisory_flags = dict(activation_link_rejection_enabled=False,
        ap_positive_epsilon_rejection_enabled=False, exact_support_rejection_enabled=False,
        exact_duplicate_separation_rejection_enabled=False,
        float32_collision_rejection_enabled=True)
    if kind == 'shipping_R3':
        protocol.update(advisory_flags,
            shipping_advisory_scope='all fresh configurations, all sizes/seeds/radii',
            acceptance_interpretation='explicitly approximate; advisory activation mismatches are not certificates',
            hard_fee='0.1 per raw native returned q2 strictly > 0; no clipping or cutoff',
            training_loss='smooth native shipping cost, gamma=5; unchanged',
            pool_cost='native strict-positive hard cost; unchanged', original_box_certificate=False)

    def metadata(n, seed=None, radius=None, method=None):
        if type(n) is not int or n not in SIZES:
            raise ValueError('Unapproved total sample size')
        row = dict(experiment=protocol['experiment'], method=method, Ntotal=n,
            Ntrain=n, Nval=0, seed=seed, radius=radius, R=10 if kind == 'R10' else 3,
            f=.1 if kind == 'shipping_R3' else 0, network_seed=0, split_seed=None,
            torch_threads=1, gurobi_threads=1, timing_unit='seconds',
            timing_clock='time.perf_counter')
        if kind == 'shipping_R3':
            row.update(advisory_flags, gamma=5., original_box_certificate=False,
                native_physical_tolerance=1e-6, negative_bound_pool_rejection_enabled=False,
                replay_rejection_enabled=False)
        return row

    def sources():
        return dict(bindings)

    def verify_sources(manifest):
        mapping = manifest['source_sha256']
        for p, h in bindings.items():
            if mapping.get(p) != h or _digest(p) != h:
                raise RuntimeError(f'Missing/changed source binding: {p}')
        original_verify(manifest)
        for p, h in manifest.get('input_sha256', {}).items():
            if _digest(p) != h:
                raise RuntimeError(f'Changed input binding: {p}')

    def configure_worker():
        global _configured
        if not _configured:
            common.core.configure_worker() if kind == 'shipping_R3' else common.configure_worker()
            _configured = True

    train_scope = dict(vars(training))
    _definitions(green / 'green_training.py', ('train_model',), train_scope, bindings,
        [('n_train not in (160, 640)', 'n_train not in (40, 160, 640, 50, 200, 800)'),
         ('Require Ntrain=160 or 640', 'Require approved CV-fold or full-N train size')])
    saa_scope = dict(vars(common))
    if kind == 'shipping_R3':
        _definitions(nominal / 'shipping_common.py', ('solve_saa',), saa_scope, bindings,
            [('n in (40,160,640)', 'n in (40,160,640,50,200,800)')])
    else:
        _definitions(green / 'green_common.py', ('solve_saa',), saa_scope, bindings,
            [('n in (160, 640)', 'n in (40, 160, 640, 50, 200, 800)')])
        if kind == 'R10':
            text_scope = dict(vars(common))
            adapter = importlib.import_module('source_adapter')
            adapter.load_into(text_scope, green / 'green_common.py',
                [('n in (160, 640)', 'n in (40, 160, 640, 50, 200, 800)'),
                 ('(n, 3)', '(n, 10)'), ('paired R3', 'paired R10'),
                 ('model.addVars(3,', 'model.addVars(10,'),
                 ('model.addVars(n, 3,', 'model.addVars(n, 10,'),
                 ('for r in range(3)', 'for r in range(10)')], names=('solve_saa',))
            saa_scope = text_scope

    # Return the canonical raw rows BEFORE the old split/permutation; no regenerated subset.
    gen_scope = dict(vars(common), metadata=metadata)
    source_text = (green / 'green_common.py').read_text()
    node = next(n for n in ast.parse(source_text).body
                if isinstance(n, ast.FunctionDef) and n.name == 'split_data')
    original = ast.get_source_segment(source_text, node)
    tail = original[original.index('    indices ='):]
    gen_changes = [(tail, '    return dict(xi1=xi1, xi2=xi2)')]
    _definitions(green / 'green_common.py', ('split_data',), gen_scope, bindings, gen_changes)
    if kind == 'R10':
        importlib.import_module('source_adapter').load_into(gen_scope, green / 'green_common.py',
            [(tail, '    return dict(xi1=xi1, xi2=xi2)'),
             ('(n_total, 3)', '(n_total, 10)'), ('paired R3', 'paired R10')], names=('split_data',))

    eval_scope = dict(vars(common), core=core, legacy_ccg=controller.legacy_ccg)
    _definitions(PARENT / 'N50_R3_train40_val10/common.py', ('evaluate',),
        eval_scope, bindings, [('model.per_sample_cost(x1,x2)', 'model.hard_shipping_cost(x1,x2)')]
        if kind == 'shipping_R3' else ())

    def choose_candidate(candidates, stopping_id):
        # Retain every original contiguous-prefix/finite-AP check without minAP publication.
        controller.choose_candidate(candidates, stopping_id)
        for index, candidate in enumerate(candidates[:stopping_id + 1]):
            decision = scope['terminal_decision'](candidate)
            if candidate.get('green_pass') is not decision['green_pass']:
                raise ValueError('Assessed prefix has inconsistent green decision')
            if candidate.get('green_gap') != decision['green_gap']:
                raise ValueError('Assessed prefix has inconsistent green gap')
            if decision['green_pass'] != (index == stopping_id):
                raise ValueError('STOP must be the first passing assessed candidate')
        return candidates[stopping_id]

    scope.update(PROTOCOL=protocol, metadata=metadata, core=core,
        verify_sources=verify_sources, train_model=train_scope['train_model'],
        choose_candidate=choose_candidate)
    if kind == 'shipping_R3':
        scope.update(adversary=advisory, legacy_ccg=advisory, validate_ap=advisory.validate_ap)
    # Correct the inherited narrative only; the original loop remains bytecode-identical.
    run = scope['run_configuration']
    old_contract = ('cached shipping smooth fits; parent hard-validation LR selection'
                    if kind == 'shipping_R3' else
                    'parent supplies fresh per-seed training-only LR-search winner')
    if old_contract not in run.__code__.co_consts:
        raise RuntimeError('Changed controller nominal-origin contract')
    run.__code__ = run.__code__.replace(co_consts=tuple(
        protocol['nominal_origin'] if c == old_contract else c for c in run.__code__.co_consts))
    _backend = SimpleNamespace(kind=kind, np=np, torch=torch, core=core,
        make_model=core.make_model, model_from_state=core.model_from_state, weights=core.weights,
        state_hash=scope['state_hash'], generate=gen_scope['split_data'],
        solve_saa=saa_scope['solve_saa'], train_model=train_scope['train_model'],
        evaluate=eval_scope['evaluate'], metadata=metadata, configure_worker=configure_worker,
        protocol=protocol, sources=sources, verify_sources=verify_sources,
        run_configuration=run, choose_candidate=choose_candidate,
        _original=SimpleNamespace(common=common, training=training, controller=controller),
        _controller=scope)
    return _backend