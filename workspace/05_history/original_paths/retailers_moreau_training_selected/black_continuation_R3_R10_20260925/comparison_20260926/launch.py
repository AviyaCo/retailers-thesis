"""Named tmux wrapper for this directory's run.py; never launches production implicitly."""
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
PYTHON = '/data/home/aviyac/ap_env/bin/python'


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def command_for(mode, output, preflight=None, sha=None):
    if mode not in ('preflight', 'production'):
        raise ValueError('Invalid launch mode')
    command = [PYTHON, '-B', '-u', str(ROOT / 'run.py'), mode, str(Path(output).resolve())]
    if mode == 'production':
        if preflight is None or not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None:
            raise ValueError('Production requires an explicit preflight summary and SHA256')
        command += ['--preflight', str(Path(preflight).resolve()), '--sha', sha]
    elif preflight is not None or sha is not None:
        raise ValueError('Preflight does not accept production bindings')
    return command


def child(launch):
    spec = json.loads((launch / 'invocation.json').read_text())
    start, code = time.perf_counter(), None
    row = dict(start_utc=utc(), status='running', command=spec['command'], cwd=str(ROOT),
               tmux_session=spec['tmux_session'], timing_clock='time.perf_counter', timing_unit='seconds')
    write(launch / 'child_runtime.json', row)
    try:
        with (launch / 'child.log').open('x') as log:
            code = subprocess.call(spec['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                     'GRB_LICENSE_FILE': '/data/home/aviyac/gurobi.lic'})
    except BaseException:
        row['error'] = traceback.format_exc()
    finally:
        elapsed = time.perf_counter()-start
        row.update(status='completed' if code == 0 else 'failed', exit_code=code,
                   end_utc=utc(), wall_time_s=elapsed,
                   wall_time_human=f'{elapsed:.6f} s / {elapsed/60:.6f} min / {elapsed/3600:.6f} h',
                   scope='whole run.py child, including imports, serialization and publication')
        write(launch / 'child_runtime.json', row)
        with (launch / 'child.exit').open('x') as stream:
            stream.write(str(code if code is not None else 1)+'\n')
        subprocess.run(['tmux', 'wait-for', '-S', spec['completion_channel']], check=True)
    return code if code is not None else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('preflight', 'production', '_child'))
    p.add_argument('output', type=Path)
    p.add_argument('--session')
    p.add_argument('--preflight', type=Path)
    p.add_argument('--sha')
    p.add_argument('--wait', action='store_true')
    a = p.parse_args(argv)
    output = a.output.resolve()
    launch = output.with_name(output.name+'.launch')
    if a.mode == '_child':
        return child(launch)
    command = command_for(a.mode, output, a.preflight, a.sha)
    if not a.session or re.fullmatch('[A-Za-z0-9_-]+', a.session) is None:
        raise ValueError('Require an explicit alphanumeric/underscore/hyphen tmux session name')
    if output.exists() or launch.exists():
        raise ValueError('Output or launch directory already exists; no overwrite/resume')
    if shutil.which('tmux') is None:
        raise RuntimeError('tmux unavailable; nothing launched (no automatic nohup fallback)')
    if subprocess.run(['tmux', 'has-session', '-t', a.session], capture_output=True).returncode == 0:
        raise ValueError('tmux session already exists')
    if a.mode == 'production':
        with a.preflight.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != a.sha:
            raise ValueError('Explicit preflight SHA256 mismatch; nothing launched')
    launch.mkdir(parents=True, exist_ok=False)
    spec = dict(command=command, exact_command=shlex.join(command), cwd=str(ROOT),
        tmux_session=a.session, completion_channel=a.session+'_comparison_done',
        output=str(output), log=str(launch / 'child.log'), created_at_utc=utc(),
        attach_command=shlex.join(['tmux', 'attach', '-t', a.session]),
        status_command=shlex.join(['cat', str(launch / 'child_runtime.json')]),
        log_command=shlex.join(['tail', '-n', '50', str(launch / 'child.log')]))
    write(launch / 'invocation.json', spec)
    wrapper = [PYTHON, '-B', '-u', str(ROOT / 'launch.py'), '_child', str(output)]
    subprocess.run(['tmux', 'new-session', '-d', '-s', a.session, '-c', str(ROOT), shlex.join(wrapper)], check=True)
    print(json.dumps(spec, indent=2), flush=True)
    if a.wait:
        subprocess.run(['tmux', 'wait-for', spec['completion_channel']], check=True)
        row = json.loads((launch / 'child_runtime.json').read_text())
        print(json.dumps(row, indent=2), flush=True)
        return row['exit_code'] if row['exit_code'] is not None else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())