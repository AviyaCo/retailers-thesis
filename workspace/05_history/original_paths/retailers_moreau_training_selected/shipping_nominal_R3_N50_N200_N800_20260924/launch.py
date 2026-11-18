"""Explicit tmux launcher; production requires exact validated preflight hash."""
from source_adapter import ROOT, check, digest, sources, verify
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


def exclusive_json(path, data):
    with path.open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('preflight','production'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--session',required=True)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--preflight-sha256')
    p.add_argument('--inside',action='store_true')
    args = p.parse_args()
    output = args.output.resolve()
    check(not output.exists(),'Output exists; no resume/overwrite')
    check(shutil.which('tmux') is not None,'tmux required on this host')
    if args.mode=='production':
        check(args.preflight is not None and args.preflight_sha256 is not None,'Exact preflight required')
        from validate_preflight import validate
        validate(args.preflight,args.preflight_sha256)
    else:
        check(args.preflight is None and args.preflight_sha256 is None,'Preflight does not consume research artifacts')
    command = [sys.executable,'-B','-u',str(ROOT/('preflight.py' if args.mode=='preflight' else 'run_nominal.py')),
               '--output',str(output)]
    if args.mode=='production':
        command += ['--preflight',str(args.preflight.resolve()),'--preflight-sha256',args.preflight_sha256]
    log = Path(str(output)+'.log')
    launch_path = Path(str(output)+'.launch.json')
    if not args.inside:
        frozen = sources()
        verify(frozen)
        check(output.parent.is_dir(),'Output parent must exist')
        argv = [v for v in sys.argv[1:]]
        inside = ['bash',str(ROOT/'launch.sh'),*argv]
        exclusive_json(launch_path,dict(session=args.session,mode=args.mode,command=shlex.join(command),
            tmux_command=shlex.join(inside),cwd=str(ROOT),output=str(output),log=str(log),
            source_sha256=frozen,attach=f'tmux attach -t {args.session}',
            status=f'tmux has-session -t {args.session}',completion_channel=args.session+'_done'))
        subprocess.run(['tmux','new-session','-d','-s',args.session,'-c',str(ROOT),shlex.join(inside)],check=True)
        print(launch_path,flush=True)
        return
    launch = json.loads(launch_path.read_text())
    start = time.perf_counter()
    record = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),status='running',
                  command=shlex.join(command),cwd=str(ROOT),session=args.session,mode=args.mode)
    code = 1
    try:
        verify(launch['source_sha256'])
        check(sources()==launch['source_sha256'],'Launcher source changed')
        with log.open('x') as stream:
            code = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
        record['status'] = 'completed' if code==0 else 'failed'
    finally:
        record.update(returncode=code,ended_at_utc=datetime.now(timezone.utc).isoformat(),
                      wall_time_s=time.perf_counter()-start,timing_scope='whole child invocation including imports/teardown')
        exclusive_json(Path(str(output)+'.invocation.json'),record)
        with Path(str(output)+'.exit').open('x') as stream:
            stream.write(str(code)+'\n')
        subprocess.run(['tmux','wait-for','-S',args.session+'_done'],check=True)
    raise SystemExit(code)


if __name__=='__main__':
    main()