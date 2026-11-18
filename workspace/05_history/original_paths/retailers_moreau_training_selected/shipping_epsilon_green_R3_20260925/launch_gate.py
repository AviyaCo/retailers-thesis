"""Named tmux launch and whole-child monotonic accounting for the AP gate."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def save(path,data):
    with path.open('x') as stream:
        json.dump(data,stream,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--session',required=True)
    parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    command=[sys.executable,'-B','-u',str(ROOT/'ap_gate.py'),'--output',str(output)]
    channel=args.session+'_done'
    if args.supervise:
        start=time.perf_counter()
        record=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),command=command,
            cwd=str(ROOT),session=args.session,workers=1,threads=1,
            timing_scope='whole AP gate child including imports and teardown; not a CCG run')
        try:
            with output.with_suffix('.log').open('x') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed')
        except BaseException as error:
            record.update(status='failed',error=repr(error))
            raise
        finally:
            record.update(wall_time_s=time.perf_counter()-start,
                          ended_at_utc=datetime.now(timezone.utc).isoformat())
            save(output.with_suffix('.invocation.json'),record)
            subprocess.run(['tmux','wait-for','-S',channel],check=True)
        return
    if output.exists() or any(output.with_suffix(s).exists() for s in ('.log','.launch.json','.invocation.json')):
        raise FileExistsError('Preserve previous gate attempts; use a fresh output name')
    output.parent.mkdir(parents=True,exist_ok=True)
    supervisor=[sys.executable,'-B',str(Path(__file__).resolve()),'--supervise',
                '--output',str(output),'--session',args.session]
    save(output.with_suffix('.launch.json'),dict(command=command,supervisor=supervisor,
        cwd=str(ROOT),session=args.session,log=str(output.with_suffix('.log')),
        attach=f'tmux attach -t {args.session}',status=f'tmux has-session -t {args.session}'))
    subprocess.run(['tmux','new-session','-d','-s',args.session,'-c',str(ROOT),shlex.join(supervisor)],check=True)
    subprocess.run(['tmux','wait-for',channel],check=True)
    record=json.loads(output.with_suffix('.invocation.json').read_text())
    print(json.dumps(record,indent=2))
    raise SystemExit(record.get('returncode',1))


if __name__=='__main__':
    main()