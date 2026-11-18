"""Host-side tmux supervisor; immutable originals, one explicitly writable output root."""
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
WRAPPER = Path('/data/home/aviyac/retailers_experiment/.consolidation_20261005/run_legacy.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    if not args.name.replace('_', '').isalnum() or not 1 <= args.workers <= 16:
        parser.error('Require simple fresh output name and 1..16 workers')
    runs = ROOT / 'runs'
    runs.mkdir(exist_ok=True)
    output = runs / args.name
    receipt = ROOT / (args.name + '.invocation.json')
    log = ROOT / (args.name + '.log')
    if output.exists() or receipt.exists() or log.exists():
        raise FileExistsError('New output, log and receipt required; no resume or overwrite')
    command = [sys.executable, '-B', str(WRAPPER), '--cwd', str(ROOT), '--write', str(runs), '--',
               sys.executable, '-B', '-u', 'runner.py', '--output', 'runs/' + args.name,
               '--workers', str(args.workers)]
    record = dict(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(),
        command=command, exact_command=shlex.join(command), cwd=str(ROOT), log=str(log),
        tmux=os.environ.get('TMUX'), workers=args.workers, child_exit_code=None,
        timing_scope='whole namespace invocation including startup; per-model times are in child artifacts')
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2)
    start = time.perf_counter()
    try:
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        record.update(status='completed' if result.returncode == 0 else 'failed', child_exit_code=result.returncode)
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record.update(ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter()-start)
        receipt.write_text(json.dumps(record, indent=2) + '\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())