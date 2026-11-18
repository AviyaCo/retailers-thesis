"""Named tmux supervisor; five-case continuation log and whole-child timing."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--session',required=True);p.add_argument('--supervise',action='store_true')
    a=p.parse_args();output=a.output.resolve();channel=a.session+'_done'
    command=[sys.executable,'-B','-u',str(ROOT/'run_continue.py'),'--output',str(output)]
    record=dict(command=command,cwd=str(ROOT),session=a.session,log=str(output.with_suffix('.log')))
    if a.supervise:
        start=time.perf_counter();record['started_at_utc']=datetime.now(timezone.utc).isoformat()
        try:
            with output.with_suffix('.log').open('x') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            record.update(returncode=result.returncode,status='completed' if result.returncode==0 else 'failed')
        except BaseException as error:record.update(status='failed',error=repr(error))
        finally:
            record.update(wall_time_s=time.perf_counter()-start,ended_at_utc=datetime.now(timezone.utc).isoformat())
            try:
                with output.with_suffix('.invocation.json').open('x') as stream:json.dump(record,stream,indent=2)
            finally:subprocess.run(['tmux','wait-for','-S',channel],check=True)
        return
    if output.exists() or any(output.with_suffix(s).exists() for s in ('.log','.invocation.json','.launch.json')):
        raise FileExistsError('Require fresh output; preserve prior evidence')
    with output.with_suffix('.launch.json').open('x') as stream:json.dump(record,stream,indent=2)
    supervisor=[sys.executable,'-B',str(ROOT/'launch.py'),'--output',str(output),'--session',a.session,'--supervise']
    subprocess.run(['tmux','new-session','-d','-s',a.session,'-c',str(ROOT),shlex.join(supervisor)],check=True)
    subprocess.run(['tmux','wait-for',channel],check=True)
    final=json.loads(output.with_suffix('.invocation.json').read_text());print(json.dumps(final,indent=2))
    raise SystemExit(final.get('returncode',1))

if __name__=='__main__':main()