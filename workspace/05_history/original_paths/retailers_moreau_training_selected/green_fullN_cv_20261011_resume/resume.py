"""Resume production_v1 of green_fullN_cv_20261011 with more concurrent workers.

Approved by the user on 14 October 2026: the original orchestrator (hard cap of
16 workers, no resume) is stopped; its in-flight worker processes keep running
and their results are adopted; completed jobs are reused unchanged; only
never-started CCG configurations are launched, through the frozen runner.invoke.
The remaining stages (training seal, test evaluation, publication, runtime
report) are the unchanged sequence from runner.run. This file lives outside the
experiment folder because workers hash every .py file there.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading
import time

EXP = Path('/data/home/aviyac/retailers_moreau_training_selected/green_fullN_cv_20261011')
sys.path.insert(0, str(EXP))
os.chdir(EXP)
import runner as R  # noqa: E402


def live_workers(jobs):
    """Job ids of every live `runner.py --worker <jobs/ID.json>` process."""
    found = set()
    prefix = str(jobs) + '/'
    for entry in os.listdir('/proc'):
        if not entry.isdigit():
            continue
        try:
            args = [a.decode(errors='replace') for a in Path(f'/proc/{entry}/cmdline').read_bytes().split(b'\0') if a]
        except OSError:
            continue
        if '--worker' in args[:-1]:
            spec = args[args.index('--worker') + 1]
            if spec.startswith(prefix) and spec.endswith('.json'):
                found.add(Path(spec).name[:-len('.json')])
    return found


def orchestrator_alive(output):
    marker = str(output)
    for entry in os.listdir('/proc'):
        if entry.isdigit():
            try:
                args = Path(f'/proc/{entry}/cmdline').read_bytes().decode(errors='replace').split('\0')
            except OSError:
                continue
            if any(a.endswith('runner.py') for a in args) and '--output' in args and '--worker' not in args:
                return True
    return False


def ccg_specs(output, summary):
    manifests = {k: str(output / k / 'manifest.json') for k in R.KINDS}
    nominals = summary['phases']['nominal']['results']
    if len(nominals) != 90 or any(r['status'] != 'completed' for r in nominals):
        raise RuntimeError('Nominal stage is not complete; refusing to resume CCG')
    lookup = {(r['kind'], r['Ntotal'], r['seed']): r['nominal'] for r in nominals}
    return [dict(id=f'ccg_{k}_N{n}_s{s}_r{rho:g}', action='ccg', kind=k, n=n, seed=s, radius=rho,
                 nominal=lookup[k, n, s], manifest=manifests[k], folder=str(output / k / f'N{n}/ccg'))
            for k in R.KINDS for n in R.SIZES for s in R.SEEDS for rho in R.RADII]


def classify(name, jobs, live):
    spec, result, process = (jobs / f'{name}{s}' for s in ('.json', '.result.json', '.process.json'))
    if name in live:
        return 'running_orphan'
    if result.is_file() and process.is_file():
        return 'done' if R.read(result)['status'] == 'completed' else 'failed'
    if spec.is_file() or process.is_file() or result.is_file():
        return 'inconsistent'
    return 'pending'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True,
                        help='Cap on total live workers, including adopted in-flight jobs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = (EXP / args.output).resolve()
    jobs = output / 'jobs'
    if not args.dry_run and orchestrator_alive(output):
        raise RuntimeError('Original orchestrator still running; stop it first')
    summary = R.read(output / 'summary.json')
    specs = ccg_specs(output, summary)
    live = live_workers(jobs)
    states = {s['id']: classify(s['id'], jobs, live) for s in specs}
    counts = {k: sum(v == k for v in states.values()) for k in sorted(set(states.values()))}
    print(R.utc(), 'classification', json.dumps(counts), flush=True)
    pending_by = {}
    for s in specs:
        if states[s['id']] == 'pending':
            key = f"{s['kind']}_N{s['n']}"
            pending_by[key] = pending_by.get(key, 0) + 1
    print(R.utc(), 'pending by case', json.dumps(pending_by), flush=True)
    bad = sorted(i for i, v in states.items() if v in ('failed', 'inconsistent'))
    if bad:
        raise RuntimeError(f'Failed/inconsistent jobs need manual review; no automatic rerun: {bad}')
    if args.dry_run:
        return 0
    if not 1 <= args.workers <= len(os.sched_getaffinity(0)):
        raise ValueError('Workers must be within process CPU affinity')

    workers = args.workers
    resume = dict(status='running', started_at_utc=R.utc(), workers_cap=workers,
        original_workers=summary['workers'], classification_at_start=counts,
        adopted_orphans=sorted(i for i, v in states.items() if v == 'running_orphan'),
        note=('Original orchestrator stopped by user request on 14 Oct 2026 to raise concurrency. '
              'Completed jobs reused unchanged; in-flight orphan workers adopted from their own '
              'result/process records (no parent invocation receipt); never-started jobs launched '
              'via frozen runner.invoke. Job-level ccg_workers metadata still records the frozen '
              'manifest value 16; actual concurrency was 16 before and up to workers_cap after resume.'))
    summary.setdefault('resumes', []).append(resume)
    original = {k: v for k, v in summary['phases']['ccg'].items() if k != 'results'}
    original['results_count_recorded'] = len(summary['phases']['ccg']['results'])
    summary['ccg_original_orchestrator'] = original
    R.write(output / 'summary.json', summary)

    # Longest configurations first (N800, N200, N50) to shorten the tail; results are order-independent.
    pending = sorted((s for s in specs if states[s['id']] == 'pending'), key=lambda s: -s['n'])
    adopted = set(resume['adopted_orphans'])
    active, lock = set(), threading.Lock()

    def run(spec):
        try:
            return R.invoke(spec, output)
        finally:
            with lock:
                active.discard(spec['id'])

    with R.timed(resume.setdefault('ccg', {})):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            while pending:
                with lock:
                    orphans = live_workers(jobs) - active
                    free = workers - len(active) - len(orphans)
                    launch = pending[:max(0, free)]
                    del pending[:len(launch)]
                    active.update(s['id'] for s in launch)
                futures += [executor.submit(run, s) for s in launch]
                if pending:
                    time.sleep(10)
            for future in futures:
                future.result()
        while live_workers(jobs) & adopted:
            time.sleep(30)
        for name in sorted(adopted):
            print(R.utc(), name, 'adopted', R.read(jobs / f'{name}.result.json')['status'], flush=True)

    record = summary['phases']['ccg']
    trained = []
    for spec in specs:
        path = jobs / f"{spec['id']}.result.json"
        trained.append(R.read(path) if path.is_file() else dict(status='failed', id=spec['id']))
    record.update(planned=810, results=trained)
    failures = [s['id'] for s, r in zip(specs, trained) if r['status'] != 'completed']
    record['status'] = 'failed' if failures else 'completed'
    R.write(output / 'summary.json', summary)

    try:
        if failures:
            raise RuntimeError(f'ccg: incomplete/failed grid; next stage not permitted: {failures}')
        # Unchanged remainder of runner.run after the CCG stage.
        from reporting import _grid, source_bindings, publish
        _grid(trained)
        for r in trained:
            if R.sha(r['summary_path']) != r['summary_sha256'] or R.sha(r['selected_checkpoint']) != r['selected_checkpoint_sha256']:
                raise ValueError('Training seal checkpoint/summary changed')
            saved = R.read(r['summary_path'])
            if saved['selected_candidate_id'] != saved['stopping_candidate_id']:
                raise ValueError('Not a STOP selection')
        gate_path = output / 'training_seal.json'
        R.write(gate_path, dict(status='sealed', sealed_at_utc=R.utc(), allow_test_regeneration=True,
            source_bindings=source_bindings(), evaluation_workers=workers, configurations=trained))
        gate = dict(path=str(gate_path), sha256=R.sha(gate_path))
        summary.update(evaluation_gate=gate, test_accessed=True)
        R.write(output / 'summary.json', summary)
        evaluations = R.stage('evaluation', [dict(id=f"eval_{r['kind']}_N{r['Ntotal']}_s{r['seed']}_r{r['radius']:g}",
            action='evaluate', kind=r['kind'], compact=dict(r, evaluation_gate=gate),
            folder=str(output / r['kind'] / f"N{r['Ntotal']}/evaluation/seed_{r['seed']}/radius_{r['radius']:g}"))
            for r in trained], output, summary, workers)
        publication = summary['phases']['publication'] = {}
        with R.timed(publication):
            result = publish(output / 'published', evaluations, dict(evaluation_gate=gate,
                final_runtime_report_path=str(output / 'runtime_report.json'),
                runtime_note='Final elapsed time is written after publication; see the final runtime report.',
                workers=f'16 (original) then up to {workers} after resume', environment=summary['environment']))
            publication['artifacts'] = result['artifacts']
        summary['status'] = resume['status'] = 'completed'
    except BaseException as error:
        summary['status'] = resume['status'] = 'failed'
        resume['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        resume['ended_at_utc'] = R.utc()
        summary['ended_at_utc'] = resume['ended_at_utc']
        start = datetime.fromisoformat(summary['started_at_utc'])
        summary['wall_time_s'] = (datetime.now(timezone.utc) - start).total_seconds()
        summary['wall_time_note'] = 'UTC elapsed across original orchestrator and resume (not perf_counter).'
        summary['workers'] = f"16 until resume at {resume['started_at_utc']}, then up to {workers}"
        summary['runtime_report'] = R.runtime_report(output, summary)
        R.write(output / 'summary.json', summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
