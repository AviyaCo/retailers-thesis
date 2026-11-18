"""Host-side tmux launcher for resume.py (same namespace wrapper as the original launch.py)."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / 'green_fullN_cv_20261011'
VIRTUAL = Path('/data/home/aviyac/retailers_moreau_training_selected') / HERE.name / 'resume.py'
WRAPPER = Path('/data/home/aviyac/retailers_experiment/.consolidation_20261005/run_legacy.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command = [sys.executable, '-B', str(WRAPPER), '--cwd', str(EXP), '--write', str(EXP / 'runs'), '--',
               sys.executable, '-B', '-u', str(VIRTUAL), '--output', 'runs/production_v1',
               '--workers', str(args.workers)] + (['--dry-run'] if args.dry_run else [])
    receipt = HERE / f'{args.name}.invocation.json'
    log = HERE / f'{args.name}.log'
    record = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(), command=command,
                  exact_command=shlex.join(command), cwd=str(EXP), log=str(log), tmux=os.environ.get('TMUX'),
                  workers=args.workers, child_exit_code=None)
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2)
    start = time.perf_counter()
    try:
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=EXP, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record.update(status='completed' if result.returncode == 0 else 'failed', child_exit_code=result.returncode)
    finally:
        record.update(ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter() - start)
        receipt.write_text(json.dumps(record, indent=2) + '\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
