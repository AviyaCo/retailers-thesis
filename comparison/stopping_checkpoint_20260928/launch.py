"""Named tmux launch; preflight never starts production. --wait uses a tmux event."""
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

HERE = Path(__file__).resolve().parent
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def command_for(mode, output, preflight=None, sha=None):
    if mode not in ('preflight', 'production'):
        raise ValueError('Invalid mode')
    command = [sys.executable, '-B', '-u', str(HERE / 'run.py'), mode, str(Path(output).resolve())]
    if mode == 'production':
        if preflight is None or not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None:
            raise ValueError('Production requires explicit --preflight and --sha')
        command += ['--preflight', str(Path(preflight).resolve()), '--sha', sha]
    elif preflight is not None or sha is not None:
        raise ValueError('Preflight cannot accept production bindings')
    return command


def child(launch):
    spec = json.loads((launch / 'invocation.json').read_text())
    start, code = time.perf_counter(), None
    row = dict(start_utc=utc(), status='running', command=spec['command'], cwd=str(HERE),
        tmux_session=spec['tmux_session'], timing_clock='time.perf_counter', timing_unit='seconds')
    write(launch / 'child_runtime.json', row)
    try:
        with (launch / 'child.log').open('x') as log:
            code = subprocess.call(spec['command'], cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, **THREAD_ENV, 'PYTHONDONTWRITEBYTECODE': '1',
                     'GRB_LICENSE_FILE': os.environ.get('GRB_LICENSE_FILE', '/data/home/aviyac/gurobi.lic')})
    except BaseException:
        row['error'] = traceback.format_exc()
    finally:
        elapsed = time.perf_counter()-start
        row.update(status='completed' if code == 0 else 'failed', exit_code=code, end_utc=utc(),
            wall_time_s=elapsed,
            wall_time_human=f'{elapsed:.6f} s / {elapsed/60:.6f} min / {elapsed/3600:.6f} h',
            scope='whole run.py child including imports and final serialization/publication; excludes this receipt write')
        summary = Path(spec['output']) / ('summary.json' if spec['mode'] == 'preflight' else 'pipeline_summary.json')
        if summary.exists():
            with summary.open('rb') as stream:
                row['parent_summary'] = dict(path=str(summary), sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
        write(launch / 'child_runtime.json', row)
        with (launch / 'child_runtime.json').open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        with (launch / 'child_runtime.sha256').open('x') as stream:
            stream.write(digest + '\n')
        with (launch / 'child.exit').open('x') as stream:
            stream.write(str(code if code is not None else 1) + '\n')
        subprocess.run(['tmux', 'wait-for', '-S', spec['completion_channel']], check=True)
    return code if code is not None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'production', '_child'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--session')
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--sha')
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    launch = output.with_name(output.name + '.launch')
    if args.mode == '_child':
        return child(launch)
    command = command_for(args.mode, output, args.preflight, args.sha)
    if not args.session or re.fullmatch('[A-Za-z0-9_-]+', args.session) is None:
        raise ValueError('Explicit alphanumeric/underscore/hyphen tmux session required')
    if output.is_relative_to(HERE.parents[1] / 'archive'):
        raise ValueError('Cannot write into archive')
    if output.exists() or launch.exists():
        raise FileExistsError('Output or launch directory exists; no overwrite or resume')
    if shutil.which('tmux') is None:
        raise RuntimeError('tmux unavailable; nothing launched, no automatic nohup fallback')
    if subprocess.run(['tmux', 'has-session', '-t', args.session], capture_output=True).returncode == 0:
        raise ValueError('tmux session already exists')
    if args.mode == 'production':
        with args.preflight.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != args.sha or args.preflight.with_suffix('.sha256').read_text().strip() != digest:
            raise ValueError('Preflight hash/seal mismatch; nothing launched')
    launch.mkdir(parents=True, exist_ok=False)
    spec = dict(mode=args.mode, command=command, exact_command=shlex.join(command), cwd=str(HERE),
        tmux_session=args.session, completion_channel=args.session + '_stop_done',
        output=str(output), log=str(launch / 'child.log'), created_at_utc=utc(),
        attach_command=shlex.join(['tmux', 'attach', '-t', args.session]),
        status_command=shlex.join(['cat', str(launch / 'child_runtime.json')]),
        log_command=shlex.join(['tail', '-n', '50', str(launch / 'child.log')]))
    write(launch / 'invocation.json', spec)
    wrapper = [sys.executable, '-B', '-u', str(HERE / 'launch.py'), '_child', str(output)]
    subprocess.run(['tmux', 'new-session', '-d', '-s', args.session, '-c', str(HERE), shlex.join(wrapper)], check=True)
    print(json.dumps(spec, indent=2), flush=True)
    if args.wait:
        subprocess.run(['tmux', 'wait-for', spec['completion_channel']], check=True)
        row = json.loads((launch / 'child_runtime.json').read_text())
        print(json.dumps(row, indent=2), flush=True)
        return row['exit_code'] if row['exit_code'] is not None else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())