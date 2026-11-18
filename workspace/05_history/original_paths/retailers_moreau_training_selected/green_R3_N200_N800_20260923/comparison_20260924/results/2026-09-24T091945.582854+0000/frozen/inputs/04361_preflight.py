"""Full disposable freeze and training-only numerical checks; NEVER generate test."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import run_comparison as r

e = r.e


def candidate_disk_bytes():
    """Conservative stat-only inventory: no test generation or held-out evaluation."""
    paths = set(r.local_sources()) | {e.COMPARE}
    for root in (e.SOURCE, e.N50, e.COMPARE.parent,
                 *(e.c.OLD/f'R3/N{n}' for n in (50, 200, 800))):
        paths.update(p.resolve() for p in root.rglob('*') if p.is_file())
    # Cached proof sources can reside outside the primary experiment directories.
    paths.update(Path(p) for p in e.read_json(e.N50/'summary.json')['input_sha256'])
    paths.update(Path(p) for p in e.read_json(e.SOURCE/'provenance.json')['source_sha256'])
    return sum(p.stat().st_size for p in paths)


def training_checks(manifest, report):
    """Two real training centers per fixed policy; no held-out rows or new tolerances."""
    expected = {(m,n,rho) for m,ns in [('green',(200,800)),('scs',(50,200,800))]
                for n in ns for rho in (0., .2)}
    chosen = [s for s in manifest['jobs'] if s['seed'] == 0 and (s['method'],s['Ntotal'],s['radius']) in expected]
    e.check(len(chosen) == len(expected) == 10, 'Missing training-only smoke cohort')
    report['checks'] = []
    for spec in chosen:
        row = dict(id=spec['id'], method=spec['method'], Ntotal=spec['Ntotal'], Ntrain=spec['Ntrain'],
            radius=spec['radius'], seed=0, status='running', phases={}, checks=[], worker_count=1, solver_threads=1)
        report['checks'].append(row)
        with e.c.phase(row['phases'], 'training_only_replay'):
            checkpoint = e._bound_file(manifest, spec['checkpoint'])
            model = e.c.load_model(checkpoint, {'green':'new','scs':'old'}[spec['method']])
            e.check(e.c.state_hash(model) == spec['state_sha256'], 'Smoke checkpoint identity mismatch')
            points = e.np.asarray(spec['centers'], dtype=e.np.float64)[:2]
            row['original_sample_ids'] = spec['original_sample_ids'][:2]
            z, q10, q1r = model.targets(points[:, :3])
            projected = []
            with e.c.gp.Env(empty=True) as env:
                env.setParam('OutputFlag', 0)
                env.setParam('Threads', 1)
                env.start()
                for target in z:
                    diagnostic = dict(start_utc=e.c.utc())
                    row['checks'].append(diagnostic)
                    try:
                        q2 = e.c.strict_project(target, q10, env, diagnostic)
                        e.check(diagnostic['status'] == e.c.GRB.OPTIMAL and diagnostic['accepted']
                                and e.np.isfinite(q2).all(), 'Smoke strict projection acceptance failed')
                        projected.append(q2)
                    finally:
                        diagnostic['end_utc'] = e.c.utc()
            costs = e.c.scenario_costs(q10, q1r, e.np.asarray(projected), points[:,:3], points[:,3:])
            e.check(costs.shape == (2,) and e.np.isfinite(costs).all(), 'Nonfinite smoke replay')
            row.update(costs=costs.tolist(), training_subset_mean=float(costs.mean()),
                saved_native_full_pool_mean=spec['saved_native_finite_pool_mean'],
                note='Subset training-center smoke only; not comparable to full-pool native mean, no difference tolerance asserted.')
            e.check(e.c.state_hash(model) == spec['state_sha256'], 'Smoke replay modified state')
            row['peak_rss_bytes'] = r.rss_bytes()
        row['status'] = 'completed'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New exclusive disposable preflight directory')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(status='running', start_utc=e.c.utc(), output=str(output), phases={}, test_generated=False,
        job_count=0, timing_clock='time.perf_counter', timing_unit='seconds', versions=e.versions())
    try:
        with e.c.phase(report['phases'], 'syntax'):
            sources = r.local_sources()
            for path in sources:
                if path.suffix == '.py':
                    tree = ast.parse(path.read_text(), filename=str(path))
                    compile(tree, str(path), 'exec')
            subprocess.run(['bash','-n',str(r.ROOT/'launch.sh')], check=True)
        with e.c.phase(report['phases'], 'unit_tests') as phase:
            command = [sys.executable, '-B', '-m', 'unittest', '-v', 'test_evaluation', 'test_plot', 'test_runner']
            phase['command'] = command
            with (output/'unit_tests.log').open('x') as log:
                result = subprocess.run(command, cwd=r.ROOT, stdout=log, stderr=subprocess.STDOUT)
            phase['returncode'] = result.returncode
            e.check(result.returncode == 0, 'Unit tests failed; inspect unit_tests.log')
        with e.c.phase(report['phases'], 'resources'):
            report['resources'] = r.resources()
            report['disk'] = r.disk_gate(output, 2*candidate_disk_bytes()+2*r.GIB, 'disposable full-freeze copies and manifests')
        original_gen = e.c.gen

        def training_only_gen(n, seed):
            e.check(n in (50,200,800) and seed in range(10), 'Preflight forbids test generation')
            return original_gen(n, seed)

        with patch.object(e.c, 'gen', side_effect=training_only_gen):
            with e.c.phase(report['phases'], 'full_freeze') as phase:
                manifest = r.freeze_all(output)
                manifest['workers'] = 1
                report['job_count'] = len(manifest['jobs'])
                report['source_sha256'] = r.hashset(manifest)
                report['scs_membership'] = r.membership_report(manifest)
                report['frozen_input_bytes'] = sum(Path(info['copy']).stat().st_size for info in manifest['files'].values())
                path = output/'frozen/manifest.json'
                report['manifest_sha256'] = r.seal(path, manifest)
                report['manifest'] = str(path)
                report['manifest_bytes'] = path.stat().st_size
                report['disk_note'] = 'Retain this disposable prefix until production validation finishes: its copies and sealed manifest are required. No automatic deletion.'
                report['production_disk_estimate_bytes'] = (2*report['frozen_input_bytes'] + 2*r.GIB
                    + 4096*(780*10000 + sum(sum(j['sizes']) for j in manifest['jobs'])))
                phase['peak_rss_bytes'] = r.rss_bytes()
            e.initialize_worker()
            with e.c.phase(report['phases'], 'numerical_training_only'):
                training_checks(manifest, report)
            with e.c.phase(report['phases'], 'final_verification'):
                r.verify_files(manifest)
                e.check(manifest['runner_sources'] == {str(p): e.c.sha(p) for p in r.local_sources()},
                        'Sources/docs changed during preflight')
                e.check(e.c.sha(path) == report['manifest_sha256'], 'Preflight manifest changed')
                report['resource_policy_example'] = r.choose_workers(r.resources(), r.rss_bytes())
        report['status'] = 'completed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report.update(end_utc=e.c.utc(), wall_time_s=time.perf_counter()-start, peak_rss_bytes=r.rss_bytes(),
                      peak_rss_scope='Linux process lifetime high-water including full freeze, not phase delta')
        report['wall_time_human'] = e.c.human(report['wall_time_s'])
        r.seal(output/'summary.json', report)
        print(f"Preflight {report['status']}: {output/'summary.json'}", flush=True)
    return output/'summary.json'


if __name__ == '__main__':
    main()