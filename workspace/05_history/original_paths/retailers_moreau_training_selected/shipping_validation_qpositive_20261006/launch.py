"""Persistent tmux supervisor outside the historical-path mount namespace."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/data/home/aviyac/ap_env/bin/python'
WRAPPER = '/data/home/aviyac/retailers_experiment/.consolidation_20261005/run_legacy.py'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--child', action='store_true')
    args = p.parse_args()
    if args.output in ('', '.', '..') or Path(args.output).name != args.output:
        p.error('--output must be a single fresh directory name within this experiment')
    output = ROOT / args.output
    command = [PYTHON, '-B', WRAPPER, '--cwd', str(ROOT), '--write', str(output),
               '--', PYTHON, '-B', '-u', str(ROOT / 'run_validation.py'), '--output', str(output)]
    log_path = ROOT / f'{args.output}.log'
    if not args.child:
        output.mkdir(exist_ok=False)
        record = dict(session=args.session, cwd=str(ROOT), command=command, log=str(log_path))
        with (ROOT / f'{args.output}.launch.json').open('x') as stream:
            stream.write(json.dumps(record, indent=2)+'\n')
        child = [PYTHON, '-B', str(ROOT / 'launch.py'), '--output', args.output,
                 '--session', args.session, '--child']
        subprocess.run(['tmux', 'new-session', '-d', '-s', args.session, '-c', str(ROOT),
                        shlex.join(child)], check=True)
        print(json.dumps(record, indent=2))
        return 0
    start = time.perf_counter()
    record = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
                  session=args.session, command=command, cwd=str(ROOT), log=str(log_path),
                  timing_scope='Whole child including compatibility wrapper, imports and all phases')
    env = dict(os.environ)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS'):
        env[name] = '1'
    try:
        with log_path.open('x', buffering=1) as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, status='completed' if result.returncode == 0 else 'failed')
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record.update(ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter()-start)
        with (ROOT / f'{args.output}.invocation.json').open('x') as stream:
            stream.write(json.dumps(record, indent=2)+'\n')
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())