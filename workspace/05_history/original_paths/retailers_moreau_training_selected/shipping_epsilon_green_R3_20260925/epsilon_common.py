"""Private imports of frozen shipping primitives; never run a historical entrypoint."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
HOME = Path('/data/home/aviyac')
NOMINAL = ROOT.parent / 'shipping_nominal_R3_N50_N200_N800_20260924'
PRODUCTION = NOMINAL / 'production_v1'
OLD = HOME / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10'
DIAGNOSTIC = ROOT.parent / 'shipping_exact_activation_diagnostic_20260924/diagnostic.py'
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREAD_ENV)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


PARENT = read(PRODUCTION / 'provenance.json')
FROZEN = PARENT['source_sha256']
for _path, _sha in FROZEN.items():
    check(digest(_path) == _sha, f'Changed nominal dependency: {_path}')
sys.path.insert(0, str(NOMINAL))
import shipping_core as core
import shipping_model
import shipping_training
import shipping_resources
for _module in (core, shipping_model, shipping_training, shipping_resources):
    check(Path(_module.__file__).parent == NOMINAL, 'Shipping import collision')
np, torch = core.np, core.torch

EPSILON = 1e-6
PROTOCOL = dict(experiment=ROOT.name, R=3, f=.1, gamma=5., epsilon=EPSILON,
    sizes=[50,200,800], data_seeds=list(range(10)), network_seed=0,
    split_seed='100000 + seed', train_fraction=.8, lr_grid=[.005,.01,.05],
    radii=[0.,.001,.003,.01,.03,.2,.4,.7,1.5], max_mp=40, max_epochs=3000,
    patience=300, nominal='reuse completed shipping smooth training; no refit',
    mp='unchanged smooth shipping objective; training-only checkpoint/scheduler',
    hard='native returned shipment > 0; no clipping or fee cutoff',
    ap='epsilon*z <= q <= Q*z; folded-float64 real-coefficient KKT MIP',
    replay='componentwise binary vs exact folded-policy support and native support',
    epsilon_numerical_tolerance=1e-9, ap_mip_gap=1e-4, ap_cost_replay_tolerance=1e-4,
    gap='abs(mean restricted hard AP - mean native hard original pool)/max(abs(pool),1e-9)',
    green_gap_tolerance=.01, initial_candidate_eligible=True,
    returned_candidate='earliest minimum AP incumbent mean through first passing candidate',
    original_box_certificate=False, hard_cost_improvement_guaranteed=False,
    failed_case_substitution=False, test_evaluated=False, fallback=False,
    workers='resource selected; single thread each')


def definitions(namespace, path, names, changes=()):
    """Reuse specific existing definitions only, with exact-count source edits."""
    path = Path(path)
    if str(path) in FROZEN:
        check(digest(path) == FROZEN[str(path)], 'Changed definition source')
    lines = path.read_text().splitlines(keepends=True)
    nodes = [n for n in ast.parse(''.join(lines)).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    check({n.name for n in nodes} == set(names), 'Missing reused definition')
    text = '\n\n'.join(''.join(lines[min([n.lineno] +
        [d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
    for old, new in changes:
        check(text.count(old) == 1, f'Ambiguous source adaptation: {old}')
        text = text.replace(old, new)
    exec(compile(text, str(path), 'exec'), namespace)


def sources():
    return {**FROZEN, str(PRODUCTION/'provenance.json'):digest(PRODUCTION/'provenance.json'),
        str(DIAGNOSTIC):digest(DIAGNOSTIC),
        **{str(p):digest(p) for p in ROOT.iterdir()
           if p.suffix in ('.py','.sh') or p.name=='README.md'}}


def verify(mapping):
    for path, sha in mapping.items():
        check(digest(path)==sha, f'Changed frozen input: {path}')
    for path, sha in FROZEN.items():
        check(mapping.get(path)==sha, f'Missing nominal source: {path}')
    for p in ROOT.iterdir():
        if p.suffix in ('.py','.sh') or p.name=='README.md':
            check(str(p) in mapping, f'Unbound new source: {p}')


def metadata(n, seed, radius, method):
    return dict(experiment=ROOT.name,method=method,Ntotal=n,Ntrain=4*n//5,Nval=n//5,
        R=3,seed=seed,network_seed=0,split_seed=100000+seed,radius=radius,
        epsilon=EPSILON,torch_threads=1,gurobi_threads=1,workers=1,
        timing_clock='time.perf_counter',timing_unit='seconds')