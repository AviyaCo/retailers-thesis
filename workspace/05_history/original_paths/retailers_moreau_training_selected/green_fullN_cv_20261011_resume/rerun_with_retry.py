"""Rerun one CCG configuration with a logged Gurobi-INTERRUPTED retry for the shipping AP.

User-approved on 14 October 2026 after shipping_R3 N200 seed 9 radius 1.5 failed
twice, identically: for candidate 4 / sample 89, Gurobi 13.0.1 found incumbents
(best 12.10772, gap 0.83%), then stopped with status 11 INTERRUPTED (internal error
10005, no time limit, callback or signal) and discarded all solutions. The same
model reaches OPTIMAL (12.10772) with Heuristics=0 or Presolve=0.

Rule: an AP solve is retried ONCE, with only Heuristics=0 added, and ONLY if the
original call raises after Gurobi returned status 11. The retry must still pass every
unchanged check (OPTIMAL, replay, membership, ordering). The first attempt is kept
verbatim in the row under `gurobi_interrupted_retry`. Every other solve is unchanged.

Usage (host, inside tmux; the namespace wrapper keeps old experiments read-only):
    python -B -u rerun_with_retry.py --job ccg_shipping_R3_N200_s9_r1.5 --spec SPEC_JSON
The child is started via `python -c` and this module is imported only AFTER the frozen
backend computed its source bindings, so frozen-source verification is unchanged; this
file's hash is recorded separately in jobs/<id>.retry_provenance.json.
"""
import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / 'green_fullN_cv_20261011'
VIRTUAL_EXP = Path('/data/home/aviyac/retailers_moreau_training_selected/green_fullN_cv_20261011')
VIRTUAL_SELF = VIRTUAL_EXP.parent / HERE.name / Path(__file__).name
WRAPPER = Path('/data/home/aviyac/retailers_experiment/.consolidation_20261005/run_legacy.py')
RETRY_PARAMETERS = {'Heuristics': 0}
GRB_INTERRUPTED = 11

# Child bootstrap: bind the frozen backend first, then import and install this module's patch.
BOOTSTRAP = (
    "import sys, importlib.util; sys.path.insert(0, {exp!r}); import runner as R; "
    "R.backend_for('shipping_R3'); "
    "s = importlib.util.spec_from_file_location('rerun_with_retry', {me!r}); "
    "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); m.install(); "
    "sys.argv = ['runner.py', '--worker', {spec!r}]; sys.exit(R.main())")


def install():
    """Wrap advisory.solve_ap (looked up at call time by the frozen CCG loop)."""
    advisory = sys.modules['advisory']
    original = advisory.solve_ap

    def solve_ap(model, center1, center2, radius, record):
        identity = dict(record)
        try:
            return original(model, center1, center2, radius, record)
        except Exception as error:
            if record.get('solver_status') != GRB_INTERRUPTED:
                raise
            first = {k: v for k, v in record.items() if k not in identity}
            first['error'] = dict(type=type(error).__name__, message=str(error))
        record.clear()
        record.update(identity)
        saved = dict(advisory.PARAMETERS)
        advisory.PARAMETERS.update(RETRY_PARAMETERS)
        try:
            result = original(model, center1, center2, radius, record)
        finally:
            advisory.PARAMETERS.clear()
            advisory.PARAMETERS.update(saved)
            record['gurobi_interrupted_retry'] = dict(
                reason='Gurobi status 11 INTERRUPTED (internal) on first attempt; user-approved single retry',
                retry_parameters=RETRY_PARAMETERS, first_attempt=first,
                retry_module=str(VIRTUAL_SELF), retry_module_sha256=_sha(Path(__file__)))
        return result

    advisory.solve_ap = solve_ap


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent(job, spec_source):
    """Same bookkeeping as runner.invoke, but the child installs the retry patch."""
    sys.path.insert(0, str(EXP))
    import runner as R
    jobs = EXP / 'runs/production_v1/jobs'
    spec = R.read(spec_source)
    if spec['id'] != job:
        raise ValueError('Spec does not match the requested job')
    for suffix in ('.json', '.log', '.result.json', '.process.json', '.invocation.json'):
        if (jobs / f'{job}{suffix}').exists():
            raise FileExistsError(f'Archive the previous attempt first: {job}{suffix}')
    spec_path = jobs / f'{job}.json'
    R.write(spec_path, spec)
    virtual_spec = str(VIRTUAL_EXP / spec_path.relative_to(EXP))
    child = [sys.executable, '-B', '-u', '-c',
             BOOTSTRAP.format(exp=str(VIRTUAL_EXP), me=str(VIRTUAL_SELF), spec=virtual_spec)]
    command = [sys.executable, '-B', str(WRAPPER), '--cwd', str(EXP), '--write', str(EXP / 'runs'), '--', *child]
    R.write(jobs / f'{job}.retry_provenance.json', dict(
        description=__doc__, retry_parameters=RETRY_PARAMETERS, trigger='solver_status == 11 only',
        retry_module=str(VIRTUAL_SELF), retry_module_sha256=_sha(Path(__file__)), command=command))
    receipt = dict(id=job, action=spec['action'], command=command, exact_command=shlex.join(command),
                   cwd=str(EXP), log=str(jobs / f'{job}.log'))
    with R.timed(receipt), (jobs / f'{job}.log').open('x') as log:
        receipt['returncode'] = subprocess.run(command, cwd=EXP, stdout=log, stderr=subprocess.STDOUT).returncode
    R.write(jobs / f'{job}.invocation.json', receipt)
    result = R.read(jobs / f'{job}.result.json')
    print(R.utc(), job, result['status'], flush=True)
    return 0 if result['status'] == 'completed' and receipt['returncode'] == 0 else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(parent(args.job, args.spec))
