"""Common strict replay primitives; frozen R3 implementation is never edited."""
import ast
import importlib.util
import sys
from pathlib import Path

HOME = Path('/data/home/aviyac')
ROOT = Path(__file__).resolve().parent
REFERENCE = HOME / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison/compare.py'
POOL_REFERENCE = HOME / 'retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/evaluation.py'
RESOURCE_REFERENCE = POOL_REFERENCE.parent / 'run_comparison.py'
spec = importlib.util.spec_from_file_location('_shipping_comparison_reference', REFERENCE)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
np, torch = c.np, c.torch
read, write, sha, utc, phase = c.read_json, c.write_json, c.sha, c.utc, c.phase
RADII, SIZES, SEEDS = list(c.RADII), (50, 200, 800), list(range(10))
PHYSICAL_TOL = 1e-6


def check(ok, message):
    if not ok:
        raise ValueError(message)


def definition(path, name, namespace):
    """Reuse a single definition, without module-level experiment execution."""
    text = Path(path).read_text()
    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    check(len(nodes) == 1, f'Missing/ambiguous definition: {name}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


strict_project = definition(REFERENCE, 'strict_project', {**vars(c), 'PHYSICAL_TOL': PHYSICAL_TOL})
pool_digest = definition(POOL_REFERENCE, 'pool_digest', {**vars(c), 'check': check})
finite_pool_cost = definition(POOL_REFERENCE, 'finite_pool_cost', {'np': np, 'check': check})


def scenario_costs(q10, q1r, q2, x1, x2):
    """Common R3 cost arithmetic plus unmodified strict-positive shipping fees."""
    return c.scenario_costs(q10, q1r, q2, x1, x2) + .1 * (q2 > 0).sum(axis=1)


def load_model(path, method):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    check(method in ('green', 'legacy'), 'Unknown model method')
    state = payload if method == 'green' else payload['model_state_dict']
    model = c.Model()
    expected = model.state_dict()
    extra = set(state) - set(expected)
    check(extra <= ({'proj_layer.q', 'proj_layer.A'} if method == 'legacy' else set()),
          f'Unknown nonlearned checkpoint fields: {extra}')
    check(not (set(expected) - set(state)), 'Missing learned tensor')
    for key, ref in expected.items():
        value = state[key]
        check(value.shape == ref.shape and value.dtype == torch.float32 and torch.isfinite(value).all(),
              f'Invalid tensor: {key}')
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    return model.eval()


def initialize_worker():
    c.initialize_worker()


def versions():
    return dict(python=sys.version, numpy=np.__version__, scipy=c.scipy.__version__,
                torch=torch.__version__, gurobi=list(c.gp.gurobi.version()))


def bound_file(manifest, source):
    info = manifest['files'][str(Path(source).resolve())]
    check(sha(source) == sha(info['copy']) == info['sha256'], f'Changed frozen file: {source}')
    return Path(info['copy'])


def verify(manifest):
    for source in manifest['files']:
        bound_file(manifest, source)
    check(manifest['versions'] == versions(), 'Runtime version change')


def stats(values):
    a = np.asarray(values, dtype=float)
    check(a.ndim == 1 and len(a) > 0 and np.isfinite(a).all() and (a >= 0).all(), 'Invalid timing values')
    return dict(count=len(a), mean=float(a.mean()), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()))