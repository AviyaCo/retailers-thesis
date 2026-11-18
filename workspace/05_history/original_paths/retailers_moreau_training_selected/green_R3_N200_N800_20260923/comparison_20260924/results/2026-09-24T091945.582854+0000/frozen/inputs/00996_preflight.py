"""No-solver preflight: real schemas, hashes, coverage, cache origin and tests."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent


def main():
    start = time.perf_counter()
    result = dict(status='running', timing_clock='time.perf_counter', timing_unit='seconds',
                  new_solves=0, source_sha256={}, phases={})
    output = HERE / 'preflight.json'
    if output.exists():
        raise FileExistsError(output)
    try:
        t = time.perf_counter()
        for p in HERE.glob('*.py'):
            ast.parse(p.read_text(), filename=str(p))
        proc = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v', 'test_update'],
                              cwd=HERE, capture_output=True, text=True)
        result['tests'] = dict(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                               wall_time_s=time.perf_counter()-t)
        print(proc.stderr, flush=True)
        if proc.returncode:
            raise RuntimeError('Unit tests failed')
        import run_update as u
        from reconstruct import reconstruct, identity
        result['started_at_utc'] = u.c.utc()

        def capture(path, expected=None):
            path = Path(path).resolve()
            digest = u.c.sha(path)
            u.require(expected is None or expected == digest, f'Historical hash mismatch: {path}')
            result['source_sha256'][str(path)] = digest
            return digest

        def read(path, expected=None):
            capture(path, expected)
            return u.c.read_json(path)

        with u.c.phase(result['phases'], 'real_trajectory_reconstruction'):
            for path in [*HERE.glob('*.py'), HERE/'PROTOCOL.md', HERE/'launch.sh']:
                capture(path)
            manifest_path = u.ROOT/'matched_oos_update/frozen/manifest.json'
            m = read(manifest_path, manifest_path.with_suffix('.sha256').read_text().strip())
            for path, info in m['files'].items():
                capture(path, info['sha256'])
            train = read(u.TRAIN/'summary.json')
            choices = []
            for job in train['ccg']['jobs']:
                d = read(job['summary_path'], job['summary_sha256'])
                for color in ('green','black'):
                    row = reconstruct(d, color)
                    capture(row['checkpoint'], row['checkpoint_sha256'])
                    choices.append(row)
            u.require(len(choices) == 180, 'Incomplete choices')
            result['choices'] = [{k:r[k] for k in ('color','seed','radius','stop_candidate',
                'selected_iteration','checkpoint_sha256','state_sha256')} for r in choices]
            result['choices_before_cache_lookup'] = True
        with u.c.phase(result['phases'], 'cache_identity_and_origin'):
            vp = read(u.ROOT/'supervisor_comparison/evidence_complete/protocol.json')
            for path, digest in vp['split_hashes'].items():
                capture(path, digest)
            val = train['validation']['evaluations'] + read(
                u.ROOT/'supervisor_comparison/evidence_complete/green_validation_progress.json')
            val_cache = {}
            for row in val:
                u.require(row['status']=='completed' and row['sample_count']==10
                          and row['projection']['statuses']=={'1':10}, 'Cached native status failed')
                u.cache_insert(val_cache, row, 'mean')
            prior = read(u.ROOT/'matched_oos_update/summary.json')
            u.require(prior['manifest_sha256']==capture(manifest_path), 'Prior manifest link')
            cp = u.ROOT/'short_rule_completion/evaluation_frozen/manifest.json'
            cm = read(cp)
            # The completion evaluator froze this job list against its source manifest.
            origin = read(u.ROOT/'short_rule_completion/frozen/manifest.json', cm['source_manifest_sha256'])
            for path, info in origin['files'].items():
                capture(path, info['sha256'])
            complete = [read(u.ROOT/'short_rule_completion'/j['id']/'result.json') for j in cm['jobs']]
            jobs = {j['id']:j for j in m['jobs']+cm['jobs']}
            test_cache = {}
            for rows, expected in ((prior['jobs'],capture(manifest_path)),(complete,capture(cp))):
                for row in rows:
                    u.check_test(row, prior['test_arrays'])
                    u.require(row['manifest_sha256']==expected and identity(row)==identity(jobs[row['id']]),
                              'Evaluation provenance mismatch')
                    u.cache_insert(test_cache, row, 'mean_oos')
            expected_missing = {('black',1,.7,11),('black',7,.7,12),('black',9,.7,39)}
            for name, cache in (('validation',val_cache),('test',test_cache)):
                missing = {(r['color'],r['seed'],r['radius'],r['selected_iteration'])
                           for r in choices if identity(r) not in cache}
                u.require(missing==expected_missing, f'Unexpected missing {name} inventory')
                result[f'missing_{name}'] = sorted(missing)
            for info in prior['test_arrays'].values():
                capture(u.OLD/'matched_oos_comparison/results'/info['file'],info['file_sha256'])
            capture(u.ROOT/'short_rule_completion/plot_data.json')
        result.update(status='completed', ended_at_utc=u.c.utc())
    except BaseException:
        result.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        result['wall_time_s'] = time.perf_counter()-start
        with output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(f'Preflight {result["status"]}: {result["wall_time_s"]:.3f}s', flush=True)


if __name__ == '__main__':
    main()