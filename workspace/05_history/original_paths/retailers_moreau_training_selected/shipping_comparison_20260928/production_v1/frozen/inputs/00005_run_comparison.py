"""Freeze -> tests -> training-only probes -> sealed test -> replay -> publish."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import common as c
import freeze as f
import evaluation as e
import plot_comparison as p

WORKERS = 16
_resource_ns = dict(Path=Path, os=os, sys=sys, platform=platform, resource=resource, math=math,
    GIB=1024**3, THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'),
    e=SimpleNamespace(c=c.c, check=c.check))
for _name in ('rss_bytes', 'resources', 'choose_workers'):
    c.definition(c.RESOURCE_REFERENCE, _name, _resource_ns)


def resource_gate(peak):
    snapshot = _resource_ns['resources']()
    policy = _resource_ns['choose_workers'](snapshot, peak)
    c.check(policy['workers'] >= WORKERS, 'Insufficient headroom for 16 workers; no degraded launch')
    policy['workers'] = WORKERS
    return dict(environment=snapshot, policy=policy)


def probe(spec, output):
    """Two original training centers per family/size; no test rows or selection."""
    start = time.perf_counter()
    folder = Path(output) / spec['id']
    folder.mkdir(parents=True, exist_ok=False)
    record = dict(id=spec['id'], Ntotal=spec['Ntotal'], seed=spec['seed'], radius=spec['radius'],
        method=spec['method'], start_utc=c.utc(), status='running', phases={}, status_counts={},
        solves_attempted=0, solves_accepted=0, saved_native_finite_pool_mean=None)
    try:
        model = c.load_model(spec['checkpoint'], spec['method'])
        c.check(c.sha(spec['checkpoint']) == spec['checkpoint_sha256']
                and c.c.state_hash(model) == spec['state_sha256'], 'Probe identity changed')
        centers = c.np.asarray(spec['centers'][:2], dtype=float)
        e.evaluate(model, centers[:, :3], centers[:, 3:], 'training_only', record, folder)
        record['status'] = 'completed'
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record.update(end_utc=c.utc(), wall_time_s=time.perf_counter()-start,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        c.write(folder / 'result.json', record)
    return record


def verify_one(spec, compact, manifest_path):
    seal = c.sha(manifest_path)
    c.check(seal == Path(manifest_path).with_suffix('.sha256').read_text().strip(), 'Changed verification manifest')
    manifest = c.read(manifest_path)
    matches = [j for j in manifest['jobs'] if j['id'] == spec['id']]
    c.check(matches == [spec], 'Verification job differs from sealed manifest')
    manifest['seal'] = seal
    return e.verify_result(spec, compact, manifest)


def report(manifest, rows, summary, output):
    groups = []
    for method in ('green', 'legacy'):
        for n in c.SIZES:
            for radius in sorted({r['radius'] for r in rows if r['method'] == method and r['Ntotal'] == n}):
                cohort = [r for r in rows if (r['method'], r['Ntotal'], r['radius']) == (method, n, radius)]
                c.check(len(cohort) == 10 and all(r['status'] == 'completed' for r in cohort), 'Incomplete runtime cohort')
                groups.append(dict(method=method, Ntotal=n, radius=radius,
                    evaluation_wall_s=c.stats([r['wall_time_s'] for r in cohort]),
                    oos_s=c.stats([r['phases']['oos']['wall_time_s'] for r in cohort]),
                    is_s=c.stats([r['phases']['is']['wall_time_s'] for r in cohort])))
    c.write(output / 'runtime_report.json', dict(current_evaluation_groups=groups,
        sweep_phases=summary['phases'], workers=WORKERS, environment=summary['resource_gate'],
        historical=[dict(id=r['id'], method=r['method'], Ntotal=r['Ntotal'], seed=r['seed'], radius=r['radius'],
            timing=r['training_runtime_reference']) for r in rows],
        note='Concurrent job walls are not summed as elapsed. Cached model timings are excluded from current replay.'))
    lines = ['# Shipping comparison — completed fixed-policy replay', '',
        '750 policies: 270 green (265 original + five advisory continuations), 480 historical Clarabel.',
        'Same already-inspected 10,000 scenarios, seed 9999; descriptive, not a fresh confirmatory test.',
        'Both neural families: fresh Gurobi projection, raw q>0 fee, no clipping or cutoff. '
        'Physical tolerance 1e-6; all projection statuses must be OPTIMAL.',
        'IS uses each returned checkpoint own saved finite pool; not AP mean or full robust supremum.',
        'Green membership remains strict. Historical coordinates are unchanged with explicit residual disclosure.',
        'No black shipping method was trained. No test-driven choice, radius validation, refit, CV, or new benchmark fitting.',
        'Historical kNN and SRO-FA retain thresholds 1e-6 and 1e-10 respectively; '
        'their cached evaluations are descriptive, not certified common-fee comparisons.',
        'The gray conditional finite 1M-scenario MIP estimate is not an exact population lower bound.', '',
        f'Host: {summary["resource_gate"]["environment"]["host"]}; {WORKERS} workers, one solver/Torch thread each.',
        'Bands: pointwise 20–80% across ten training seeds, not confidence intervals.', '',
        '## Current evaluation runtimes (seconds)', '',
        '| Method | N | Radius | Runs | Mean | Median | Min | Max | Mean OOS | Mean IS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        t = g['evaluation_wall_s']
        lines.append(f'| {g["method"]} | {g["Ntotal"]} | {g["radius"]:g} | {t["count"]} | '
            f'{t["mean"]:.3f} | {t["median"]:.3f} | {t["min"]:.3f} | {t["max"]:.3f} | '
            f'{g["oos_s"]["mean"]:.3f} | {g["is_s"]["mean"]:.3f} |')
    lines += ['', '## Model-runtime coverage', '',
        '| Category | Current work / historical scope |', '|---|---|',
        '| Nominal neural training | Not rerun; persisted per-seed nominal records in runtime report |',
        '| Robust neural/CCG | Not rerun; persisted AP/MP/current/cached timing distinctions retained |',
        '| SRO-FA partition/optimization/evaluation/total | Not rerun; original per-seed time_partition/time_lp/time_eval/time_total retained in plot-data provenance |',
        '| Hyperparameter selection/CV/radius validation | Not run |',
        '| Common neural OOS and own-pool evaluation | Fresh, separately timed above |',
        'Missing historical fields remain unavailable; a rerun is required to measure them. '
        'Cache-loading time is never substituted for training time.', '', '## Phase runtimes', '']
    for name, phase in summary['phases'].items():
        if 'wall_time_s' in phase:
            lines.append(f'- {name}: {c.c.human(phase["wall_time_s"])} ({phase["status"]}).')
    lines += ['', 'Whole-child runtime and UTC boundaries are recorded in the adjacent invocation JSON. '
              'Phases are nested; concurrent job times are not elapsed campaign time.', '',
              'Raw per-row projection journals and cost/target/shipment arrays were verified before publication. '
              'All original and frozen source/input hashes were checked again; no failed models are silently omitted.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    summary = dict(status='running', start_utc=c.utc(), phases={}, results=[], workers=WORKERS,
        training_run=False, selection_run=False, test_generated=False)
    def persist():
        summary['wall_time_s'] = time.perf_counter()-start
        c.write(output / 'summary.json', summary)
    try:
        c.initialize_worker()
        with c.phase(summary['phases'], 'freeze'):
            manifest = f.freeze(output)
        with c.phase(summary['phases'], 'tests'):
            with (output / 'tests.log').open('x') as log:
                test = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
                    'test_freeze', 'test_plot', 'test_evaluation', 'test_runner'], cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT)
            c.check(test.returncode == 0, 'Regression tests failed')
            f.verify(manifest)
        with c.phase(summary['phases'], 'training_only_preflight'):
            chosen = [next(j for j in manifest['jobs'] if (j['method'],j['Ntotal'],j['seed'],j['radius']) == (m,n,0,0.))
                      for n in c.SIZES for m in ('green','legacy')]
            # Genuine spawn regression before any held-out rows are generated.
            with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
                    initializer=c.initialize_worker) as pool:
                probes = [pool.submit(probe, j, str(output/'preflight')).result() for j in chosen]
            summary['probes'] = probes
            f.verify(manifest)
        summary['resource_gate'] = resource_gate(max(r['peak_rss_bytes'] for r in probes))
        # Verify cached-reference schemas using only their frozen artifacts. No neural test scores exist yet.
        for n in c.SIZES:
            p._baselines(manifest, n)
        with c.phase(summary['phases'], 'test_generation'):
            x1, x2 = c.c.gen(10000, 9999)
            info = dict(n_test=10000, seed=9999, arrays={})
            for name, a in (('xi1',x1),('xi2',x2)):
                path = output / f'{name}.npy'
                c.np.save(path, a); path.chmod(0o444)
                info['arrays'][name] = dict(file=str(path),file_sha256=c.sha(path),
                    array_sha256=hashlib.sha256(a.tobytes()).hexdigest(), shape=list(a.shape), dtype=str(a.dtype))
            manifest.update(test=info, test_generated=True, workers=WORKERS)
            manifest_path = output / 'manifest.json'
            c.write(manifest_path, manifest); manifest_path.chmod(0o444)
            seal = c.sha(manifest_path)
            manifest_path.with_suffix('.sha256').write_text(seal+'\n')
            manifest_path.with_suffix('.sha256').chmod(0o444)
            summary.update(manifest_path=str(manifest_path),manifest_sha256=seal,test_generated=True)
        with c.phase(summary['phases'], 'evaluation'):
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                    initializer=e.initialize, initargs=(str(manifest_path),)) as pool:
                futures = {pool.submit(e.worker,j['id'],str(output/'evaluations')):j['id'] for j in manifest['jobs']}
                for future in as_completed(futures):
                    row = future.result(); summary['results'].append(row); persist()
                    print(f'{len(summary["results"])}/750 {row["id"]} {row["status"]}',flush=True)
        c.check(len(summary['results']) == 750 and all(r['status']=='completed' for r in summary['results']),
                'Incomplete or failed grid; no partial plot')
        with c.phase(summary['phases'], 'journal_verification'):
            indexed = {r['id']:r for r in summary['results']}
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context('spawn'),
                    initializer=c.initialize_worker) as pool:
                futures = [pool.submit(verify_one,j,indexed[j['id']],str(manifest_path)) for j in manifest['jobs']]
                rows = [future.result() for future in futures]
        with c.phase(summary['phases'], 'final_integrity'):
            f.verify(manifest)
            c.check(c.sha(manifest_path)==seal,'Changed manifest')
            for a in manifest['test']['arrays'].values():
                c.check(c.sha(a['file'])==a['file_sha256'],'Changed test')
        with c.phase(summary['phases'], 'render'):
            data = p.build_plot_data(manifest,rows)
            published=output/'published'
            published.mkdir()
            p.render(data,published)
            c.write(published/'plot_data.json',data)
        report(manifest,rows,summary,published)
        summary.update(status='completed',solves_accepted=sum(r['solves_accepted'] for r in rows),
            published={str(x):c.sha(x) for x in published.iterdir() if x.is_file()})
    except BaseException:
        summary.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        summary['end_utc']=c.utc();persist()


if __name__ == '__main__':
    main()