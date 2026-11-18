"""Full-N SRO-FA+LDR radius sweep for the comparison plot (R3) — 15 October 2026.

Companion to run.py (same folder), reusing its LDR solver and evaluator unchanged.
Like the historical full-N SRO-FA curve, each policy is fitted on ALL N = 50/200/800
rows of data seed s (the canonical paired draw) for every radius of SRO's historical R3
grid, and scored on the shared seed-9999 10,000-row test set. This is a descriptive
test curve over radius (the test set is already seen; no radius is selected here).
Pre-launch check: the same code reproduces the historical full-N SRO-FA test costs exactly.

Usage (inside tmux):  python -B -u run_full_n.py --output runs/full_n_v1 --workers 24
Re-running resumes; finished job files are never overwritten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np

import run as S

CASE, R = 'R3', 3


def full_n_job(spec, output, test):
    n, seed, rho = spec['n'], spec['seed'], spec['rho']
    record = dict(spec, method='FA+LDR', case=CASE, protocol='full_N', started_at_utc=S.utc(), status='running')
    started = time.perf_counter()
    try:
        x1, x2 = S.old.generate_data(n, seed, R)
        t = time.perf_counter()
        part_lo, part_hi, partition_map = S.old.build_partitions(x1, rho)
        record['partition_time_s'], record['partition_count'] = time.perf_counter() - t, len(part_lo)
        policy, record['solve'] = S.solve_sro_fa_ldr(x1, x2, rho, part_lo, part_hi, partition_map)
        t = time.perf_counter()
        q2, diagnostics = S.shipments('FA+LDR', policy, test[0], part_lo, part_hi)
        record['test'] = dict(S.cost(policy, q2, *test, False), **diagnostics, wall_time_s=time.perf_counter() - t)
        record['status'] = 'completed'
    except Exception as error:
        record.update(status='failed', error=dict(type=type(error).__name__, message=str(error),
                                                  traceback=traceback.format_exc()))
    record.update(ended_at_utc=S.utc(), wall_time_s=time.perf_counter() - started)
    S.write_json(output / 'jobs' / f"{spec['id']}.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    output = (S.HERE / args.output).resolve()
    (output / 'jobs').mkdir(parents=True, exist_ok=True)
    x1, x2 = S.old.generate_data(S.TEST_SIZE, S.TEST_SEED, R)
    if not S.sha256(np.asarray(x1, '<f8').tobytes()).startswith(S.TEST_XI1_SHA[R]):
        raise RuntimeError('Test set differs from the green test set')
    specs = sorted((dict(id=f'fullN_FA_LDR_{CASE}_N{n}_s{seed}_r{rho:g}', n=n, seed=seed, rho=rho)
                    for n in S.SIZES for seed in S.SEEDS for rho in S.radius_grid(CASE, n)),
                   key=lambda s: (-s['n'], -s['rho']))  # largest N and radius are slowest
    pending = [s for s in specs if not (output / 'jobs' / f"{s['id']}.json").exists()]
    started = time.perf_counter()
    print(S.utc(), f'{len(specs)} jobs, {len(pending)} to run, {args.workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(full_n_job, s, output, (x1, x2)) for s in pending]):
            r = future.result()
            print(S.utc(), r['id'], r['status'], f"{r['wall_time_s']:.1f}s", flush=True)
    records = [json.loads((output / 'jobs' / f"{s['id']}.json").read_text()) for s in specs]
    failed = [r['id'] for r in records if r['status'] != 'completed']
    S.write_json(output / 'summary.json', dict(
        jobs=len(records), failed=failed, workers=args.workers, host=platform.node(),
        elapsed_s=time.perf_counter() - started, ended_at_utc=S.utc(),
        this_file_sha256=S.sha256(Path(__file__).read_bytes()), run_py_sha256=S.sha256((S.HERE / 'run.py').read_bytes())))
    print(S.utc(), f'done; {len(failed)} failed', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
