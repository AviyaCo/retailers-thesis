"""Disposable, sequential seed0 training-only integration; never a research fit."""
from source_adapter import ROOT, sources, verify, digest, check, THREAD_ENV
import shipping_core as core
from shipping_common import metadata, PROTOCOL
from shipping_resources import resources, choose_workers
from run_nominal import worker, read
import argparse
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys
import time
import traceback


def checked_sources(record, label):
    verify(record['source_sha256'])
    check(sources()==record['source_sha256'], 'Sources changed during preflight')
    record['source_checks'].append(dict(label=label,status='completed',
        source_sha256=record['source_sha256'],at_utc=core.utc_now()))


def child(n, output):
    output.mkdir(parents=True,exist_ok=False)
    record = metadata(n,0,'disposable two-epoch preflight')
    record.update(epochs=2,preflight_only=True,production_reuse_forbidden=True,
        heldout_evaluated=False,workers=1,phases={},phase_sha256={},source_checks=[],source_sha256=sources())
    try:
        with core.timed(record):
            core.configure_worker()
            checked_sources(record,'before')
            manifest = output/'provenance.json'
            core.write_json(manifest,dict(source_sha256=record['source_sha256'],workers=dict(initialization=1,nominal=1)))
            manifest.chmod(0o444)
            setup = record['phases']['initialization'] = worker('initialization',dict(n=n,seed=0),
                output/'initialization',manifest)
            record['phase_sha256']['initialization'] = digest(output/'initialization/result.json')
            check(setup['status']=='completed', f'SAA/model initialization failed: {setup.get("error")}')
            nominal = record['phases']['nominal'] = worker('nominal',dict(n=n,seed=0,lr=.01,setup=setup),
                output/'nominal',manifest,epochs=2)
            record['phase_sha256']['nominal'] = digest(output/'nominal/result.json')
            check(nominal['status']=='completed', f'Disposable training failed: {nominal.get("error")}')
            checked_sources(record,'after')
    finally:
        record['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        core.write_json(output/'summary.json',record)


def invoke(command, log, record):
    record['command'] = shlex.join(command)
    with core.timed(record), log.open('x') as stream:
        result = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                env={**os.environ,**THREAD_ENV,'PYTHONHASHSEED':'0'})
        record['returncode'] = result.returncode
        log.with_suffix('.exit').write_text(str(result.returncode)+'\n')
        check(result.returncode==0, f'Child failed: {log}')


def parent(output):
    output.mkdir(parents=True,exist_ok=False)
    record = dict(preflight_only=True,production_reuse_forbidden=True,heldout_evaluated=False,
        workers=1,children=[],tests={},source_checks=[],source_sha256=sources(),not_run=PROTOCOL['not_run'],
        environment=resources(),command=shlex.join(sys.orig_argv),cwd=str(Path.cwd()))
    try:
        with core.timed(record):
            checked_sources(record,'before')
            # Require one worker's guarded resource headroom before any MIP.
            record['preflight_resource_guard'] = choose_workers(record['environment'],0)
            invoke([sys.executable,'-B','test_nominal.py','--output',str(output/'tests.json')],
                   output/'tests.log',record['tests'])
            fixture = read(output/'tests.json')
            record['tests'].update({k:fixture[k] for k in ('tests_run','failures','errors')},
                fixture_run=fixture,sha256=digest(output/'tests.json'))
            for n in (50,200,800):
                item = dict(Ntotal=n,path=str(output/f'N{n}/summary.json'))
                record['children'].append(item)
                invoke([sys.executable,'-B','-u','preflight.py','--size',str(n),'--output',str(output/f'N{n}')],
                       output/f'N{n}.log',item)
                data = read(item['path'])
                check(data['status']=='completed', 'Failed child summary')
                item.update(sha256=digest(item['path']),peak_rss_bytes=data['peak_rss_bytes'])
                core.write_json(output/'summary.json',record)
            checked_sources(record,'after')
    except BaseException:
        record['traceback'] = traceback.format_exc()
        raise
    finally:
        core.write_json(output/'summary.json',record)
    from validate_preflight import validate
    validate(output/'summary.json',digest(output/'summary.json'))
    print('PREFLIGHT_COMPLETED',digest(output/'summary.json'),record['wall_time_s'],flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,choices=(50,200,800))
    args = p.parse_args()
    if args.size is None:
        parent(args.output.resolve())
    else:
        child(args.size,args.output.resolve())