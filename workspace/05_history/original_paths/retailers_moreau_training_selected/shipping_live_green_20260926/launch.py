"""Detached named-tmux supervisor; durable logs and whole-child timing."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','production'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--session',required=True)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--preflight-sha256')
    parser.add_argument('--supervise',action='store_true')
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    command=[sys.executable,'-B','-u',str(ROOT/('preflight.py' if args.mode=='preflight' else 'run_campaign.py')),
        '--output',str(output)]
    if args.mode=='production' and args.preflight is not None and args.preflight_sha256 is not None:
        command+=['--preflight',str(args.preflight.resolve()),'--preflight-sha256',args.preflight_sha256]
    channel=args.session+'_done'
    if args.supervise:
        start=time.perf_counter()
        invocation=dict(command=command,cwd=str(ROOT),session=args.session,mode=args.mode,
            started_at_utc=datetime.now(timezone.utc).isoformat(),timing_scope='whole child including imports/teardown')
        try:
            with output.with_suffix('.log').open('x') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            invocation.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed')
        except BaseException as error:
            invocation.update(status='failed',error=repr(error))
        finally:
            invocation.update(wall_time_s=time.perf_counter()-start,ended_at_utc=datetime.now(timezone.utc).isoformat())
            try:
                save(output.with_suffix('.invocation.json'),invocation)
            finally:
                subprocess.run(['tmux','wait-for','-S',channel],check=True)
        return
    if output.exists() or any(output.with_suffix(s).exists() for s in ('.log','.launch.json','.invocation.json')):
        raise FileExistsError('Use fresh output; preserve previous evidence')
    if args.mode=='production':
        if args.preflight is None or args.preflight_sha256 is None:
            raise ValueError('Require preflight binding')
        from shipping_live_common import validate_integration
        validate_integration(args.preflight,args.preflight_sha256)
    supervisor=[sys.executable,'-B',str(Path(__file__).resolve()),args.mode,'--output',str(output),
        '--session',args.session,'--supervise']
    if args.mode=='production':
        supervisor+=['--preflight',str(args.preflight.resolve()),'--preflight-sha256',args.preflight_sha256]
    save(output.with_suffix('.launch.json'),dict(command=command,supervisor=supervisor,cwd=str(ROOT),
        session=args.session,log=str(output.with_suffix('.log')),
        attach=f'tmux attach -t {args.session}',status=f'tmux has-session -t {args.session}'))
    subprocess.run(['tmux','new-session','-d','-s',args.session,'-c',str(ROOT),shlex.join(supervisor)],check=True)
    if args.wait:
        subprocess.run(['tmux','wait-for',channel],check=True)
        result=json.loads(output.with_suffix('.invocation.json').read_text());print(json.dumps(result,indent=2))
        raise SystemExit(result.get('returncode',1))
    print(json.dumps(dict(session=args.session,command=command,log=str(output.with_suffix('.log'))),indent=2))


if __name__=='__main__':
    main()