"""Named tmux launcher; exclusive output, persistent log and whole-child timing."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, allow_nan=False, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    Path(path).chmod(0o444)


def command_for(args):
    if args.mode == 'preflight':
        if args.preflight is not None or args.preflight_sha256 is not None:
            raise ValueError('Preflight mode does not consume a prior preflight')
        return [sys.executable, '-B', '-u', str(ROOT / 'preflight.py'), '--output', str(args.output.resolve())]
    if args.preflight is None or not re.fullmatch('[0-9a-f]{64}', args.preflight_sha256 or ''):
        raise ValueError('Production requires --preflight and exact --preflight-sha256')
    return [sys.executable, '-B', '-u', str(ROOT / 'run_comparison.py'), '--output', str(args.output.resolve()),
            '--preflight', str(args.preflight.resolve()), '--preflight-sha256', args.preflight_sha256]


def monitor(invocation, digest):
    """Private tmux child: verify launch binding, then time the entire subprocess."""
    if sha(invocation) != digest:
        raise ValueError('Invocation hash changed')
    data = json.loads(Path(invocation).read_text())
    start = time.perf_counter()
    record = dict(experiment='R10_fixed_policy_comparison_20260925', mode=data['mode'],
        session=data['session'], command=data['command'], exact_command=data['exact_command'],
        completion_channel=data['completion_channel'],
        cwd=data['cwd'], output=data['output'], log=data['log'], start_utc=utc(), status='running',
        returncode=None, invocation_sha256=digest, pid=os.getpid(),
        timing_clock='time.perf_counter', timing_unit='seconds',
        timing_scope='Monitor including binding checks; child_wall_time_s isolates the entire child, not model training.',
        child_start_utc=None, child_end_utc=None, child_wall_time_s=None,
        new_training=False, new_selection=False)
    try:
        if not os.environ.get('TMUX'):
            raise RuntimeError('Monitor must run inside the named tmux session')
        actual = subprocess.run(['tmux', 'display-message', '-p', '#S'], capture_output=True, text=True, check=True)
        if actual.stdout.strip() != data['session']:
            raise RuntimeError('Wrong tmux session')
        if Path(data['output']).exists():
            raise FileExistsError('Output already exists; no overwrite or retry')
        for source, expected in data['source_sha256'].items():
            if sha(source) != expected:
                raise RuntimeError(f'Source changed since launch: {source}')
        env = dict(os.environ, COMPARISON_TMUX_SESSION=data['session'], COMPARISON_LOG=data['log'],
                   COMPARISON_EXIT=data['exit'], PYTHONDONTWRITEBYTECODE='1')
        with Path(data['log']).open('x', buffering=1) as log:
            child_start = time.perf_counter()
            record['child_start_utc'] = utc()
            try:
                result = subprocess.run(data['command'], cwd=data['cwd'], env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            finally:
                record.update(child_end_utc=utc(), child_wall_time_s=time.perf_counter()-child_start)
        record.update(returncode=result.returncode, status='completed' if result.returncode == 0 else 'failed')
    except BaseException:
        record.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        record.update(end_utc=utc(), wall_time_s=time.perf_counter()-start)
        record['wall_time_human'] = f"{record['wall_time_s']:.3f} s ({record['wall_time_s']/60:.3f} min)"
        write_once(data['runtime'], record)
        with Path(data['exit']).open('x') as file:
            file.write(str(record['returncode'] if record['returncode'] is not None else 1) + '\n')
            file.flush()
            os.fsync(file.fileno())
        # Completion only: persisted status/runtime must precede the wake-up.
        subprocess.run(['tmux', 'wait-for', '-S', data['completion_channel']], check=True)
    return record['returncode']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'production'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--preflight-sha256')
    args = parser.parse_args(argv)
    if not re.fullmatch('[A-Za-z0-9_-]+', args.session):
        raise ValueError('Session must contain only letters, digits, underscores and hyphens')
    if shutil.which('tmux') is None:
        raise RuntimeError('tmux unavailable; no automatic nohup fallback')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    command = command_for(args)
    existing = subprocess.run(['tmux', 'has-session', '-t', '=' + args.session], capture_output=True, text=True)
    if existing.returncode == 0:
        raise FileExistsError(f'tmux session already exists: {args.session}')
    import run_comparison as r
    if args.mode == 'production':
        # Fail before starting tmux if the explicit checksum does not match.
        if sha(args.preflight) != args.preflight_sha256:
            raise ValueError('Exact preflight SHA256 mismatch')
    metadata = output.with_name(output.name + '.launch')
    metadata.mkdir(parents=True, exist_ok=False)
    invocation = metadata / 'invocation.json'
    data = dict(mode=args.mode, session=args.session, cwd=str(ROOT), output=str(output),
        completion_channel=args.session + '_done',
        wait_command=shlex.join(['tmux', 'wait-for', args.session + '_done']),
        command=command, exact_command=shlex.join(command), requested_at_utc=utc(),
        log=str(metadata / 'child.log'), exit=str(metadata / 'child.exit'),
        runtime=str(metadata / 'child_runtime.json'),
        source_sha256={str(p): sha(p) for p in r.local_sources()},
        attach_command=shlex.join(['tmux', 'attach-session', '-t', args.session]),
        status_command=shlex.join(['tmux', 'has-session', '-t', '=' + args.session]),
        check_exit_command=shlex.join(['cat', str(metadata / 'child.exit')]))
    write_once(invocation, data)
    digest = sha(invocation)
    monitor_command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--monitor', str(invocation), digest]
    tmux_command = ['tmux', 'new-session', '-d', '-s', args.session, '-c', str(ROOT), shlex.join(monitor_command)]
    write_once(metadata / 'tmux_command.json', dict(command=tmux_command, exact_command=shlex.join(tmux_command),
                                                  invocation_sha256=digest))
    result = subprocess.run(tmux_command, capture_output=True, text=True)
    write_once(metadata / 'tmux_start.json', dict(returncode=result.returncode, stdout=result.stdout,
                                                stderr=result.stderr, at_utc=utc()))
    if result.returncode:
        raise RuntimeError(f'tmux start failed; no retry: {result.stderr}')
    print(json.dumps({k: data[k] for k in ('session', 'exact_command', 'cwd', 'log', 'exit', 'runtime',
                                          'completion_channel', 'wait_command',
                                          'attach_command', 'status_command', 'check_exit_command')}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--monitor':
        raise SystemExit(monitor(Path(sys.argv[2]), sys.argv[3]))
    raise SystemExit(main())