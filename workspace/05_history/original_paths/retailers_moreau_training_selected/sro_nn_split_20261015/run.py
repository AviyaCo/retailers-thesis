"""SRO-FA (and SRO-FA+LDR for R3) on the NN's exact train/validation split — 15 October 2026.

Purpose: compare test cost at the validation-selected radius with the green NN under
the same data protocol. For data seed s and Ntotal in {50, 200, 800}, the NN split is
reproduced exactly: the canonical paired truncated-normal draw, then
default_rng(100000+s).permutation; the first 80% (40/160/640) train, the last 20%
(10/40/160) validate. (These held-out rows also chose the NN's learning rate.)

Methods
  FA      unchanged SRO-FA LP/MIP (piecewise-constant recourse on a data-driven
          rectangular partition), imported from the historical
          retailers_problem/select_sro_fa_rho_by_validation.py.
  FA+LDR  same partition, but within partition k the shipment is affine in xi1:
          q2(xi1) = a_k + G_k xi1 (full R x R matrix). Feasibility (q2 >= 0,
          sum q2 <= Q10) holds robustly on every training box cell. The per-
          retailer worst case is bounded term-by-term, so the LP minimises an upper
          bound on the true worst case (exact for G = 0, i.e. FA). At test/validation
          points outside all boxes, q2 is clipped at 0 and scaled down if
          sum > Q10; the fraction of such points is recorded. R3 only for now.

Protocol: every (method, case, N, seed, radius) is solved on training rows only and
scored on its validation rows. Per (method, case, N, seed) the radius with minimum
validation mean is selected (since 2026-11-18: the largest radius within 1e-6 of the minimum validation mean,
src/config.py; originally exact ties -> smaller radius). All selections are
sealed (SHA-256) before the seed-9999 10,000-row test set is generated; only selected
policies are then test-evaluated. Shipping fees: 0.1 per raw q2 > 0 (NN convention);
the historical q2 > 1e-10 cost is also recorded. Radius grids: SRO's historical
per-N grids (R3 23/17/17, R10 17, shipping 34 incl. 0). No fallback: any non-OPTIMAL
solve (shipping keeps the historical 1% MIP gap and 3 h time limit) is a failure and
blocks that seed's selection.

Usage (inside tmux):  python -B -u run.py --output runs/production_v1 --workers 28
Re-running the same command resumes: completed job files are reused, never overwritten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
import traceback

os.environ.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
os.environ.setdefault('GRB_LICENSE_FILE', '/data/home/aviyac/gurobi.lic')

import gurobipy as gp  # noqa: E402
from gurobipy import GRB  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
HISTORY = Path('/data/home/aviyac/retailers_experiment/workspace/05_history/original_paths')
OLD_PATH = HISTORY / 'retailers_problem/select_sro_fa_rho_by_validation.py'
_spec = importlib.util.spec_from_file_location('historical_sro_fa', OLD_PATH)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)  # module has a __main__ guard; import runs no experiment

SIZES = (50, 200, 800)
SEEDS = tuple(range(10))
TRAIN_FRACTION = .8
SPLIT_SEED_BASE = 100000
TEST_SIZE, TEST_SEED = 10000, 9999
# Expected SHA-256 prefixes of the green test arrays (xi1), so all methods share one test set.
TEST_XI1_SHA = {3: '4c210efecc', 10: '6ae4fbaad3'}
CASES = {
    'R3': dict(R=3, shipping=False, grid=HISTORY / 'retailers_problem/R3/sro_fa_rho_validation_results.json'),
    'R10': dict(R=10, shipping=False, grid=HISTORY / 'retailers_problem/R10/sro_fa_rho_validation_results.json'),
    'shipping_R3': dict(R=3, shipping=True, grid=None),  # 34-radius full sweep grid, see radius_grid()
}
METHODS = {'FA': ('R3', 'R10', 'shipping_R3'), 'FA+LDR': ('R3',)}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def radius_grid(case, n):
    """SRO's own historical grid for this case and sample size."""
    if case == 'shipping_R3':
        sweep = json.loads((HISTORY / f'retailers_with_shipping/N{n}/sro_fa/sro_fa_shipping_results_N{n}_R3.json').read_text())
        return sorted(float(k) for k in sweep if k not in ('meta', 'runtime'))
    return [float(r) for r in json.loads(CASES[case]['grid'].read_text())['by_N'][str(n)]['candidate_rhos']]


def nn_split(n, seed, R):
    """Exactly the NN's paired draw and 80/20 split (green_common.split_data)."""
    xi1, xi2 = old.generate_data(n, seed, R)
    order = np.random.default_rng(SPLIT_SEED_BASE + seed).permutation(n)
    n_train = int(round(TRAIN_FRACTION * n))
    train, val = order[:n_train], order[n_train:]
    return xi1[train], xi2[train], xi1[val], xi2[val], dict(
        n_train=n_train, n_val=n - n_train,
        paired_sha256=sha256(np.asarray(xi1, '<f8').tobytes() + np.asarray(xi2, '<f8').tobytes()))


def solve_sro_fa_ldr(xi1_train, xi2_train, delta, part_lo, part_hi, partition_map):
    """SRO with finite adaptability + linear decision rules inside each partition (no shipping).

    Mirrors old.solve_sro_fa; only the recourse q2 changes from a constant per partition to
    a_k + G_k xi1. Box robust counterparts use max_{x in [lo,hi]} c*x = max(c*lo, c*hi).
    """
    n_train, R = xi1_train.shape
    K = len(part_lo)
    B, H, C = old.B, old.H, old.C
    xi1_lo, xi1_hi = np.maximum(0., xi1_train - delta), xi1_train + delta
    xi2_lo, xi2_hi = np.maximum(0., xi2_train - delta), xi2_train + delta
    finite_hi = np.where(np.isinf(part_hi), 1e9, part_hi)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model(env=env)
    m.setParam('Threads', 1)
    q10 = m.addVar(lb=0., name='Q10')
    q1r = m.addVars(R, lb=0., name='Q1r')
    a = m.addVars(R, K, lb=-GRB.INFINITY, name='a')
    G = m.addVars(R, R, K, lb=-GRB.INFINITY, name='G')
    robust_cost = m.addVars(n_train, lb=-GRB.INFINITY, name='obj')

    def box_max(coefficient, lo, hi):
        """Epigraph variable t >= coefficient*x for x in {lo, hi}, i.e. t >= max over [lo,hi]."""
        t = m.addVar(lb=-GRB.INFINITY)
        m.addConstr(t >= coefficient * lo)
        m.addConstr(t >= coefficient * hi)
        return t

    for j in range(n_train):
        for k in partition_map[j]:
            lo = np.maximum(xi1_lo[j], part_lo[k])
            hi = np.minimum(xi1_hi[j], finite_hi[k])
            # Robust feasibility on the cell: q2_r(xi1) >= 0 and sum_r q2_r(xi1) <= Q10.
            for r in range(R):
                m.addConstr(a[r, k] - gp.quicksum(box_max(-G[r, s, k], lo[s], hi[s]) for s in range(R)) >= 0)
            m.addConstr(gp.quicksum(a[r, k] for r in range(R))
                        + gp.quicksum(box_max(gp.quicksum(G[r, s, k] for r in range(R)), lo[s], hi[s])
                                      for s in range(R)) <= q10)
            costs = []
            for r in range(R):
                u = m.addVar(lb=-GRB.INFINITY)
                # Backorder piece: b*(xi2_r + xi1_r - q2_r - Q1r) - h*q2_r, worst case per retailer.
                m.addConstr(u >= B * (xi2_hi[j, r] - q1r[r]) - (B + H) * a[r, k]
                            + gp.quicksum(box_max((B if s == r else 0.) - (B + H) * G[r, s, k], lo[s], hi[s])
                                          for s in range(R)))
                # The other two pieces do not involve q2 and are identical to old.solve_sro_fa.
                m.addConstr(u >= H * (q1r[r] - xi2_lo[j, r] - lo[r]))
                m.addConstr(u >= B * (hi[r] - q1r[r]) - H * xi2_lo[j, r])
                costs.append(u)
            m.addConstr(robust_cost[j] >= gp.quicksum(costs))
    m.setObjective(C * (q10 + q1r.sum()) + H * q10 + robust_cost.sum() / n_train, GRB.MINIMIZE)
    started = time.perf_counter()
    m.optimize()
    solve_time = time.perf_counter() - started
    status = int(m.Status)
    if status != GRB.OPTIMAL or m.SolCount <= 0:
        m.dispose(); env.dispose()
        raise RuntimeError(f'SRO-FA+LDR solve failed: status={status}, sol_count={m.SolCount}')
    policy = dict(q10=float(q10.X), q1r=np.array([q1r[r].X for r in range(R)]),
                  a=np.array([[a[r, k].X for k in range(K)] for r in range(R)]),
                  G=np.array([[[G[r, s, k].X for k in range(K)] for s in range(R)] for r in range(R)]))
    record = dict(status=status, status_name='OPTIMAL', objective=float(m.ObjVal),
                  gurobi_runtime_s=float(m.Runtime), measured_solve_time_s=solve_time,
                  variables=int(m.NumVars), constraints=int(m.NumConstrs), threads=1)
    m.dispose(); env.dispose()
    return policy, record


def shipments(method, policy, xi1, part_lo, part_hi):
    """Shipment q2 at each scenario; LDR outputs are clipped at 0 and scaled to sum <= Q10."""
    membership = np.all((xi1[:, None, :] >= part_lo[None]) & (xi1[:, None, :] <= part_hi[None]), axis=2)
    if not membership.any(axis=1).all():
        raise RuntimeError('Evaluation point outside every partition')
    k = np.argmax(membership, axis=1)  # first matching partition, as in old.evaluate_policy
    if method == 'FA':
        return policy['q2'][:, k].T, dict()
    raw = policy['a'][:, k].T + np.einsum('rsn,ns->nr', policy['G'][:, :, k], xi1)
    q2 = np.maximum(raw, 0.)
    total = q2.sum(axis=1)
    over = total > policy['q10']
    q2[over] *= (policy['q10'] / total[over])[:, None]
    # Diagnostics count only material corrections; sub-1e-6 ones are Gurobi feasibility tolerance.
    return q2, dict(fraction_clipped_negative=float((raw < -1e-6).any(axis=1).mean()),
                    fraction_scaled_capacity=float((total > policy['q10'] + 1e-6).mean()))


def cost(policy, q2, xi1, xi2, shipping):
    """Mean cost; same formula as old.evaluate_policy. Fee 0.1 per raw q2 > 0 (and > 1e-10)."""
    B, H, C = old.B, old.H, old.C
    first = C * (policy['q10'] + policy['q1r'].sum()) + H * policy['q10']
    recourse = np.maximum(B * (xi1 + xi2 - policy['q1r'] - q2) - H * q2,
                          np.maximum(H * (policy['q1r'] - xi1 - xi2), B * (xi1 - policy['q1r']) - H * xi2))
    base = first + recourse.sum(axis=1)
    if not shipping:
        return dict(mean=float(base.mean()))
    return dict(mean=float((base + old.SHIPPING_CHARGE * (q2 > 0).sum(axis=1)).mean()),
                mean_fee_threshold_1e10=float((base + old.SHIPPING_CHARGE * (q2 > old.SHIPPING_THRESHOLD).sum(axis=1)).mean()))


def job_id(method, case, n, seed, rho):
    return f"{method.replace('+', '_')}_{case}_N{n}_s{seed}_r{rho:g}"


def train_job(spec, output):
    """Solve on training rows, score on validation rows, save the policy. No test access."""
    method, case, n, seed, rho = spec['method'], spec['case'], spec['n'], spec['seed'], spec['rho']
    R, shipping = CASES[case]['R'], CASES[case]['shipping']
    record = dict(spec, started_at_utc=utc(), status='running')
    started = time.perf_counter()
    try:
        x1, x2, v1, v2, record['split'] = nn_split(n, seed, R)
        t = time.perf_counter()
        part_lo, part_hi, partition_map = old.build_partitions(x1, rho)
        record['partition_time_s'], record['partition_count'] = time.perf_counter() - t, len(part_lo)
        if method == 'FA':
            q10, q1r, q2, record['solve'] = old.solve_sro_fa(x1, x2, rho, part_lo, part_hi, partition_map, shipping)
            policy = dict(q10=q10, q1r=q1r, q2=q2)
        else:
            policy, record['solve'] = solve_sro_fa_ldr(x1, x2, rho, part_lo, part_hi, partition_map)
        t = time.perf_counter()
        q2v, diagnostics = shipments(method, policy, v1, part_lo, part_hi)
        record['validation'] = dict(cost(policy, q2v, v1, v2, shipping), **diagnostics,
                                    wall_time_s=time.perf_counter() - t)
        if method == 'FA' and not shipping:  # the new evaluator must reproduce the historical one exactly
            reference = old.evaluate_policy(v1, v2, q10, q1r, q2, part_lo, part_hi, False)
            if abs(reference - record['validation']['mean']) > 1e-9:
                raise RuntimeError('Evaluator mismatch with historical SRO-FA evaluator')
        policy_path = output / 'policies' / f"{spec['id']}.npz"
        np.savez_compressed(policy_path, part_lo=part_lo, part_hi=part_hi, **policy)
        record.update(policy_path=str(policy_path), status='completed')
    except Exception as error:
        record.update(status='failed', error=dict(type=type(error).__name__, message=str(error),
                                                  traceback=traceback.format_exc()))
    record.update(ended_at_utc=utc(), wall_time_s=time.perf_counter() - started)
    write_json(output / 'jobs' / f"{spec['id']}.json", record)
    return record


def plan(methods):
    specs = []
    for method in methods:
        for case in METHODS[method]:
            for n in SIZES:
                for seed in SEEDS:
                    for rho in radius_grid(case, n):
                        specs.append(dict(id=job_id(method, case, n, seed, rho), method=method,
                                          case=case, n=n, seed=seed, rho=rho))
    # Expected-longest first (shipping MIPs, then larger N) to avoid a long tail.
    return sorted(specs, key=lambda s: (-CASES[s['case']]['shipping'], -s['n'], -CASES[s['case']]['R']))


def _rule():
    """Shared radius-selection rule (retailers_experiment/src/config.py), 2026-11-18."""
    import importlib
    import sys
    shared = '/data/home/aviyac/retailers_experiment'
    if shared not in sys.path:
        sys.path.insert(0, shared)
    return importlib.import_module('src.config')


def select(records):
    """Per (method, case, N, seed): radius by the shared rule (largest radius within 1e-6 of the minimum)."""
    groups = {}
    for r in records:
        groups.setdefault((r['method'], r['case'], r['n'], r['seed']), []).append(r)
    selections, blocked = [], []
    for key, rows in sorted(groups.items()):
        if any(r['status'] != 'completed' for r in rows):
            blocked.append(dict(zip(('method', 'case', 'n', 'seed'), key),
                                failed=[r['id'] for r in rows if r['status'] != 'completed']))
            continue
        chosen = _rule().select_radius({r['rho']: r['validation']['mean'] for r in rows})
        best = next(r for r in rows if r['rho'] == chosen)
        selections.append(dict(zip(('method', 'case', 'n', 'seed'), key), rho=best['rho'], id=best['id'],
                               validation_mean=best['validation']['mean'], policy_path=best['policy_path'],
                               policy_sha256=sha256(Path(best['policy_path']).read_bytes()),
                               candidates={str(r['rho']): r['validation']['mean'] for r in rows}))
    return selections, blocked


def test_selected(selections):
    """Runs only after the selection seal exists; one test draw per R, checked against green's hash."""
    tests = {}
    for R in sorted({CASES[s['case']]['R'] for s in selections}):
        x1, x2 = old.generate_data(TEST_SIZE, TEST_SEED, R)
        if not sha256(np.asarray(x1, '<f8').tobytes()).startswith(TEST_XI1_SHA[R]):
            raise RuntimeError(f'Test set differs from the green test set for R={R}')
        tests[R] = (x1, x2)
    rows = []
    for s in selections:
        started = time.perf_counter()
        if sha256(Path(s['policy_path']).read_bytes()) != s['policy_sha256']:
            raise RuntimeError(f"Selected policy changed after sealing: {s['id']}")
        with np.load(s['policy_path']) as saved:
            policy = {k: saved[k] for k in saved.files}
        policy['q10'] = float(policy['q10'])
        x1, x2 = tests[CASES[s['case']]['R']]
        q2, diagnostics = shipments(s['method'], policy, x1, policy['part_lo'], policy['part_hi'])
        row = {k: v for k, v in s.items() if k != 'candidates'}
        row['test'] = dict(cost(policy, q2, x1, x2, CASES[s['case']]['shipping']), **diagnostics,
                           wall_time_s=time.perf_counter() - started)
        rows.append(row)
    return rows


def summarize(tests, records, output, elapsed, workers):
    lines = ['# SRO-FA / SRO-FA+LDR on the NN split — validation-selected test cost', '',
             "Mean (std) over seeds of test cost at each seed's validation-selected radius. Shipping: raw q2 > 0 fee.", '',
             '| Method | Case | N | Seeds | Mean test | Std | Selected radii (seeds 0-9) |', '|---|---|---:|---:|---:|---:|---|']
    groups = {}
    for t in tests:
        groups.setdefault((t['method'], t['case'], t['n']), []).append(t)
    for (method, case, n), rows in sorted(groups.items()):
        values = np.array([r['test']['mean'] for r in rows])
        radii = ', '.join(f"{r['rho']:g}" for r in sorted(rows, key=lambda r: r['seed']))
        lines.append(f'| {method} | {case} | {n} | {len(rows)} | {values.mean():.4f} | {values.std(ddof=1):.4f} | {radii} |')
    lines += ['', '## Runtime per training job (partition + solve + validation), seconds', '',
              '| Method | Case | N | Jobs | Mean | Median | Min | Max |', '|---|---|---:|---:|---:|---:|---:|---:|']
    timing = {}
    for r in records:
        if r['status'] == 'completed':
            timing.setdefault((r['method'], r['case'], r['n']), []).append(r['wall_time_s'])
    for key, values in sorted(timing.items()):
        lines.append(f'| {key[0]} | {key[1]} | {key[2]} | {len(values)} | {np.mean(values):.2f} | '
                     f'{np.median(values):.2f} | {min(values):.2f} | {max(values):.2f} |')
    failed = [r['id'] for r in records if r['status'] != 'completed']
    lines += ['', f'Failed jobs: {len(failed)}' + (f" ({', '.join(failed)})" if failed else ''),
              f'Overall elapsed for this invocation: {elapsed:.1f} s ({elapsed / 3600:.2f} h) with {workers} '
              'single-thread workers; concurrent job times are not summed as elapsed.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--methods', nargs='+', choices=tuple(METHODS), default=list(METHODS))
    args = parser.parse_args()
    output = (HERE / args.output).resolve()
    if (output / 'selection_seal.json').exists():
        raise RuntimeError('Selection already sealed; refusing to retrain')
    for sub in ('jobs', 'policies'):
        (output / sub).mkdir(parents=True, exist_ok=True)
    invocation = dict(started_at_utc=utc(), workers=args.workers, methods=args.methods, host=platform.node(),
                      gurobi=list(gp.gurobi.version()), historical_module=str(OLD_PATH),
                      historical_module_sha256=sha256(OLD_PATH.read_bytes()),
                      this_file_sha256=sha256(Path(__file__).read_bytes()))
    started = time.perf_counter()
    specs = plan(args.methods)
    pending = [s for s in specs if not (output / 'jobs' / f"{s['id']}.json").exists()]
    print(utc(), f'{len(specs)} jobs, {len(specs) - len(pending)} already done, {len(pending)} to run, '
          f'{args.workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(train_job, s, output) for s in pending]):
            r = future.result()
            print(utc(), r['id'], r['status'], f"{r['wall_time_s']:.1f}s", flush=True)
    records = [json.loads((output / 'jobs' / f"{s['id']}.json").read_text()) for s in specs]
    selections, blocked = select(records)
    write_json(output / 'selection_seal.json', dict(
        sealed_at_utc=utc(), selections=selections, blocked=blocked,
        failed_jobs=[r['id'] for r in records if r['status'] != 'completed'],
        rule=_rule().RADIUS_SELECTION_RULE + '; test not yet generated'))
    seal_sha = sha256((output / 'selection_seal.json').read_bytes())
    print(utc(), f'sealed {len(selections)} selections ({len(blocked)} blocked), sha256 {seal_sha}', flush=True)
    tests = test_selected(selections)
    elapsed = time.perf_counter() - started
    write_json(output / 'test_results.json', dict(selection_seal_sha256=seal_sha, results=tests,
                                                  invocation=dict(invocation, ended_at_utc=utc(), elapsed_s=elapsed)))
    summarize(tests, records, output, elapsed, args.workers)
    print(utc(), 'done; see RESULTS.md', flush=True)
    return 0 if not blocked else 1


if __name__ == '__main__':
    raise SystemExit(main())
