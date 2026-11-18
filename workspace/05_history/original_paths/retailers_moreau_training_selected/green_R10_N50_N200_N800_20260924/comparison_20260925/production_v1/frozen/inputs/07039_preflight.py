"""Full 810-policy freeze, 24 training QPs and ONE two-row training oracle LP."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import run_comparison as r

e = r.e
e.c.load_definitions(globals(), r.R3_PREFLIGHT, ('training_checks',),
    (("[('green',(200,800)),('scs',(50,200,800))]", "[('green',(50,200,800)),('scs',(50,200,800))]", 1),
     ('len(expected) == 10', 'len(expected) == 12', 1),
     ('points[:, :3]', 'points[:, :10]', 1),
     ('points[:,:3], points[:,3:]', 'points[:,:10], points[:,10:]', 1)))


def oracle_training_check(manifest, report, output):
    chosen = [j for j in manifest['jobs'] if (j['method'], j['Ntotal'], j['radius'], j['seed'])
              == ('green', 50, 0., 0)]
    e.check(len(chosen) == 1, 'Missing oracle training cohort')
    spec = chosen[0]
    points = e.np.asarray(spec['centers'], dtype=e.np.float64)[:2]
    e.check(points.shape == (2, 20), 'Oracle smoke requires two saved 20D training centers')
    row = dict(experiment=e.EXPERIMENT, scope='training_only', id=spec['id'],
               Ntotal=spec['Ntotal'], Ntrain=spec['Ntrain'], seed=spec['seed'],
               original_sample_ids=spec['original_sample_ids'][:2],
               training_points_sha256=hashlib.sha256(points.tobytes(order='C')).hexdigest(),
               manifest_sha256=report['manifest_sha256'])
    report['oracle_training_only'] = row  # Retain mutated failed diagnostics too.
    r.oracle.solve(points[:, :10], points[:, 10:], row, Path(output) / 'oracle_training_solution.npz')
    r.require_training_oracle(report, manifest)


def candidate_disk_bytes():
    paths = set(r.local_sources())
    for root in (e.SOURCE, e.c.OLD / 'R10'):
        paths.update(p.resolve() for p in root.rglob('*') if p.is_file())
    paths.update(Path(p) for p in e.read_json(e.SOURCE / 'provenance.json')['source_sha256'])
    return sum(p.stat().st_size for p in paths)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(experiment=e.EXPERIMENT, status='running', start_utc=e.c.utc(), output=str(output),
        phases={}, test_generated=False, job_count=0, timing_clock='time.perf_counter',
        timing_unit='seconds', versions=e.versions(), workers=1, threads=1,
        solve_counts_scope='strict_training_QP_only; oracle LP recorded separately')
    original_gen = e.c.gen

    def training_only_gen(n, seed):
        e.check(n in (50, 200, 800) and type(seed) is int and seed in range(10),
                'Preflight forbids test generation')
        return original_gen(n, seed)

    try:
        with patch.object(e.c, 'gen', side_effect=training_only_gen):
            with e.c.phase(report['phases'], 'syntax') as phase:
                sources = r.local_sources()
                phase['python_files'] = [str(p) for p in sources if p.suffix == '.py']
                for path in sources:
                    if path.suffix == '.py':
                        compile(ast.parse(path.read_text(), filename=str(path)), str(path), 'exec')
                result = subprocess.run(['bash', '-n', str(r.ROOT / 'launch.sh')])
                phase['bash_returncode'] = result.returncode
                e.check(result.returncode == 0, 'Shell syntax failed')
            with e.c.phase(report['phases'], 'unit_tests') as phase:
                command = [sys.executable, '-B', '-m', 'unittest', '-v',
                           'test_evaluation', 'test_plot', 'test_runner', 'test_oracle']
                phase['command'] = command
                with (output / 'unit_tests.log').open('x') as log:
                    result = subprocess.run(command, cwd=r.ROOT, stdout=log, stderr=subprocess.STDOUT)
                phase.update(returncode=result.returncode, log_sha256=e.c.sha(output / 'unit_tests.log'))
                e.check(result.returncode == 0, 'Unit tests failed')
            with e.c.phase(report['phases'], 'resources'):
                report['resources'] = r.resources()
                report['resource_policy_before'] = r.choose_workers(report['resources'], r.rss_bytes())
                report['disk'] = r.disk_gate(output, 2*candidate_disk_bytes()+2*r.GIB, 'full preflight freeze')
            with e.c.phase(report['phases'], 'full_freeze'):
                manifest = r.freeze_all(output)
                manifest['workers'] = 1
                report.update(job_count=len(manifest['jobs']), source_sha256=r.hashset(manifest),
                              scs_membership=r.membership_report(manifest))
                path = output / 'frozen/manifest.json'
                report.update(manifest_sha256=r.seal(path, manifest), manifest=str(path))
            e.initialize_worker()
            with e.c.phase(report['phases'], 'numerical_training_only'):
                training_checks(manifest, report)
            with e.c.phase(report['phases'], 'oracle_training_only'):
                oracle_training_check(manifest, report, output)
            with e.c.phase(report['phases'], 'final_verification'):
                r.verify_files(manifest)
                r.require_training_oracle(report, manifest)
                e.check(manifest['runner_sources'] == {str(p): e.c.sha(p) for p in r.local_sources()}, 'Sources changed')
                e.check(e.c.sha(path) == report['manifest_sha256'], 'Preflight manifest changed')
                report['resource_policy_after'] = r.choose_workers(r.resources(), r.rss_bytes())
            report['status'] = 'completed'
        diagnostics = [d for row in report['checks'] for d in row['checks']]
        report.update(solves_attempted=len(diagnostics), solves_accepted=sum(d['accepted'] is True for d in diagnostics),
                      status_counts={str(status): sum(d['status'] == status for d in diagnostics)
                                     for status in {d['status'] for d in diagnostics}},
                      end_utc=e.c.utc(), wall_time_s=time.perf_counter()-start, peak_rss_bytes=r.rss_bytes())
        r.validate_preflight_evidence(report, manifest)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        diagnostics = [d for row in report.get('checks', []) for d in row['checks']]
        report.update(solves_attempted=len(diagnostics),
                      solves_accepted=sum(d.get('accepted') is True for d in diagnostics),
                      status_counts={str(status): sum(d.get('status') == status for d in diagnostics)
                                     for status in {d.get('status') for d in diagnostics}})
        for row in report.get('checks', []):
            if row['status'] != 'completed':
                row['status'] = 'failed'
        report.update(end_utc=e.c.utc(), wall_time_s=time.perf_counter()-start, peak_rss_bytes=r.rss_bytes())
        report['wall_time_human'] = e.c.human(report['wall_time_s'])
        report['peak_rss_scope'] = 'Linux lifetime high-water including freeze, not a phase delta'
        digest = r.seal(output / 'summary.json', report)
        print(f"Preflight {report['status']}: {output / 'summary.json'} SHA256={digest}", flush=True)
    return output / 'summary.json'


if __name__ == '__main__':
    main()