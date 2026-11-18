"""Named tmux launcher and whole-child timing; never retries a failed run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/data/home/aviyac/ap_env/bin/python'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('preflight', 'production', '_child'))
    p.add_argument('output')
    p.add_argument('--session')
    p.add_argument('--preflight')
    p.add_argument('--sha')
    p.add_argument('--wait', action='store_true')
    a = p.parse_args()
    output = Path(a.output).resolve()
    launch = output.with_name(output.name + '.launch')
    if a.mode == '_child':
        spec = json.loads((launch / 'invocation.json').read_text())
        started, code = time.perf_counter(), None
        row = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), command=spec['command'],
                   cwd=str(ROOT), status='running', timing_clock='time.perf_counter', timing_unit='seconds')
        write(launch / 'child_runtime.json', row)
        try:
            with (launch / 'child.log').open('x') as log:
                code = subprocess.call(spec['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                         'GRB_LICENSE_FILE': '/data/home/aviyac/gurobi.lic'})
        finally:
            row.update(exit_code=code, status='completed' if code == 0 else 'nonconverged' if code == 2 else 'failed',
                ended_at_utc=datetime.now(timezone.utc).isoformat(), wall_time_s=time.perf_counter()-started)
            write(launch / 'child_runtime.json', row)
            (launch / 'child.exit').write_text(str(code) + '\n')
            subprocess.run(['tmux', 'wait-for', '-S', spec['completion_channel']], check=True)
        return code if code is not None else 1
    if shutil.which('tmux') is None:
        raise RuntimeError('tmux unavailable; no experiment launched')
    if not a.session or output.exists():
        raise ValueError('Require a named session and new output directory')
    if subprocess.run(['tmux', 'has-session', '-t', a.session], capture_output=True).returncode == 0:
        raise ValueError('tmux session already exists')
    command = [PYTHON, '-B', '-u', str(ROOT / 'campaign.py'), a.mode, str(output)]
    if a.mode == 'production':
        if not a.preflight or not a.sha:
            raise ValueError('Production requires exact preflight path and SHA')
        command += ['--preflight', str(Path(a.preflight).resolve()), '--sha', a.sha]
    launch.mkdir(parents=True, exist_ok=False)
    spec = dict(command=command, cwd=str(ROOT), tmux_session=a.session,
                completion_channel=a.session + '_done', log=str(launch / 'child.log'),
                output=str(output), created_at_utc=datetime.now(timezone.utc).isoformat(),
                attach_command=f'tmux attach -t {a.session}',
                status_command=f'cat {shlex.quote(str(launch / "child_runtime.json"))}')
    write(launch / 'invocation.json', spec)
    wrapper = [PYTHON, '-B', '-u', str(Path(__file__).resolve()), '_child', str(output)]
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