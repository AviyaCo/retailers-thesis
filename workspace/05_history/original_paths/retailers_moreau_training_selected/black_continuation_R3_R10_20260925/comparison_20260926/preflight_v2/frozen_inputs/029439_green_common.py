"""Isolated R3/f=0 primitives; never call historical split/SAA/training controllers."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_COMMON = ROOT.parent / 'N50_R3_train40_val10/common.py'
_COMMON_SHA256 = '51638dddb42dd84b86a8afc99f0e09b479c9f048a3ed5ce768248af6dca5ec81'
with _COMMON.open('rb') as _stream:
    if hashlib.file_digest(_stream, 'sha256').hexdigest() != _COMMON_SHA256:
        raise RuntimeError(f'Historical common source changed: {_COMMON}')
_spec = importlib.util.spec_from_file_location('_green_R3_20260923_frozen_common', _COMMON)
if _spec is None or _spec.loader is None:
    raise ImportError(f'Cannot load {_COMMON}')
_historical = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_historical)
core, legacy_ccg, adversary = _historical.core, _historical.legacy_ccg, _historical.adversary
np, torch = _historical.np, _historical.torch
THREAD_ENV, check = _historical.THREAD_ENV, _historical.check
evaluate, raw_hash = _historical.evaluate, _historical.raw_hash
FROZEN = {**_historical.FROZEN, str(_COMMON): _COMMON_SHA256}
# The historical loader uses plain numerical-module imports. Reject a preloaded
# module from another campaign rather than silently computing with that module.
for _module, _name in ((core, 'core'), (legacy_ccg, 'ccg'), (adversary, 'adversary')):
    check(Path(_module.__file__).resolve() == _historical.OLD / f'{_name}.py',
          f'Historical numerical import collision: {_name}')

from scipy.stats import truncnorm

SIZES = (200, 800)
SEEDS = tuple(range(10))
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
LR_GRID = (.005, .01, .05)
MAX_MP = 40
PROTOCOL = dict(experiment='retailer_live_green_R3_N200_N800_20260923',
    sizes=list(SIZES), R=3, f=0, train_fraction=.8, validation_fraction=.2,
    data_seeds=list(SEEDS), network_seed=0, split_seed='100000 + data seed',
    lr_grid=list(LR_GRID), radii=list(RADII), hidden_width=16,
    raw_dtype='float64', tensor_dtype='float32', batch='full training set',
    generator='legacy scipy truncnorm; global numpy seed saved/restored',
    max_epochs=3000, patience=300, improvement_threshold=1e-4,
    scheduler=dict(metric='training_loss', factor=.7, patience=150, threshold=1e-3),
    checkpoint='thresholded training-best evaluated PREUPDATE state including incoming state',
    optimizer='fresh Adam per nominal LR and MP', lr_inventory=.01, gradient_clip_norm=1.,
    nominal_lr_selection='after all three training runs; minimum validation mean, smaller LR tie',
    ccg_stop='first abs(APmean-poolmean)/max(abs(poolmean),1e-9) < .01 including candidate0',
    ccg_selection='earliest minimum mean AP incumbent over complete executed prefix including stop',
    green_gap_tolerance=.01, green_gap_floor=1e-9,
    validation='no validation calls inside training or CCG; only after complete CCG sweep',
    radius_selection='all nine converged configurations required; minimum validation mean, smaller radius tie',
    max_mp=MAX_MP, separation_tolerance=.01, membership_tolerance=1e-9,
    ap_replay_tolerance=1e-4, ap_mip_gap=1e-4, capacity_buffer=0.,
    nominal_workers='manifest.workers', ccg_workers='manifest.workers',
    torch_threads=1, gurobi_threads=1, test='not accessed', refit=False,
    resume=False, solver_fallback=False, global_robust_optimality_certified=False)


def configure_worker():
    """Call once in each new worker process, before any parallel Torch work."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def verify_sources(manifest):
    """Require the complete historical freeze and new computational source hashes."""
    sources = manifest['source_sha256']
    for path, expected in FROZEN.items():
        check(sources.get(path) == expected, f'Missing/changed historical freeze: {path}')
    for name in ('green_common.py', 'green_training.py', 'green_ccg.py'):
        check(str(ROOT / name) in sources, f'Missing computational source freeze: {name}')
    _historical.verify_sources(manifest)


def metadata(n_total, seed=None, radius=None, method=None):
    check(type(n_total) is int and n_total in SIZES, 'Unapproved total sample size')
    return dict(experiment=PROTOCOL['experiment'], method=method, seed=seed, radius=radius,
        Ntotal=n_total, Ntrain=4 * n_total // 5, Nval=n_total // 5, R=3, f=0,
        network_seed=0, split_seed=None if seed is None else 100000 + seed,
        torch_threads=1, gurobi_threads=1, timing_unit='seconds', timing_clock='time.perf_counter')


def split_data(n_total, seed):
    """Paired 80/20 split; exact canonical generator without mutating model cfg."""
    info = metadata(n_total, seed)
    check(type(seed) is int and seed in SEEDS, 'Unapproved data seed')
    rng = np.random.get_state()
    try:
        np.random.seed(seed)
        xi1 = truncnorm.rvs((0. - 6.) / 2.5, np.inf, loc=6., scale=2.5,
                            size=(n_total, 3))
        mu2 = 2 * (6. - xi1) ** 2
        xi2 = truncnorm.rvs((0. - mu2) / 2.5, np.inf, loc=mu2, scale=2.5)
    finally:
        np.random.set_state(rng)
    check(xi1.shape == xi2.shape == (n_total, 3), 'Invalid paired R3 shapes')
    indices = np.random.default_rng(100000 + seed).permutation(n_total)
    train_indices, val_indices = indices[:info['Ntrain']], indices[info['Ntrain']:]
    return dict(xi1=xi1, xi2=xi2, train_indices=train_indices, val_indices=val_indices,
        train_x1=torch.tensor(xi1[train_indices], dtype=torch.float32),
        train_x2=torch.tensor(xi2[train_indices], dtype=torch.float32),
        val_x1=torch.tensor(xi1[val_indices], dtype=torch.float32),
        val_x2=torch.tensor(xi2[val_indices], dtype=torch.float32))


def solve_saa(train1, train2, record):
    """Unchanged canonical LP equations, with n=len(train1), strictly OPTIMAL."""
    gp, GRB = core.gp, core.GRB
    n = len(train1)
    record.update(method='SAA initialization', n_train=n, threads=1,
                  solver_status=None, solver_runtime_s=None, iterations=None,
                  q10=None, q1r=None, objective=None)
    with core.timed(record):
        x1, x2 = np.asarray(train1, dtype=np.float64), np.asarray(train2, dtype=np.float64)
        check(n in (160, 640) and x1.shape == x2.shape == (n, 3),
              'SAA requires approved train80 paired R3 arrays')
        if not np.isfinite(x1).all() or not np.isfinite(x2).all():
            raise FloatingPointError('Nonfinite SAA input')
        cfg = core.baseline.cfg
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            with gp.Model(env=env) as model:
                model.Params.Threads = 1
                q10 = model.addVar(lb=0., name='Q10')
                q1r = model.addVars(3, lb=0., name='Q1r')
                q2 = model.addVars(n, 3, lb=0., name='Q2')
                v = model.addVars(n, 3, lb=-GRB.INFINITY, name='v')
                for i in range(n):
                    model.addConstr(gp.quicksum(q2[i, r] for r in range(3)) <= q10)
                    for r in range(3):
                        model.addConstr(v[i, r] >= cfg.b * (x1[i, r] + x2[i, r]
                            - q2[i, r] - q1r[r]) - cfg.h * q2[i, r])
                        model.addConstr(v[i, r] >= cfg.h * (q1r[r] - x1[i, r] - x2[i, r]))
                        model.addConstr(v[i, r] >= cfg.b * (x1[i, r] - q1r[r]) - cfg.h * x2[i, r])
                objective = (cfg.c * (q10 + gp.quicksum(q1r[r] for r in range(3)))
                             + cfg.h * q10
                             + gp.quicksum(v[i, r] for i in range(n) for r in range(3)) / n)
                model.setObjective(objective, GRB.MINIMIZE)
                try:
                    model.optimize()
                finally:
                    record.update(solver_status=int(model.Status),
                                  solver_runtime_s=float(model.Runtime),
                                  iterations=float(model.IterCount))
                    if model.SolCount:
                        record.update(q10=float(q10.X), q1r=[float(q1r[r].X) for r in range(3)],
                                      objective=float(model.ObjVal))
                if model.Status != GRB.OPTIMAL:
                    raise RuntimeError(f'SAA did not reach GRB.OPTIMAL: {model.Status}')
                if not np.isfinite([record['q10'], *record['q1r'], record['objective']]).all():
                    raise FloatingPointError('Nonfinite SAA solution')
    return record['q10'], np.asarray(record['q1r'], dtype=np.float64)