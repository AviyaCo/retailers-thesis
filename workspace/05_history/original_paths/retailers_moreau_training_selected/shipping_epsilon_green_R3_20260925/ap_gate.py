"""Training-center-only launch gate on fixed saved policies; failures stay visible."""
import argparse
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
from epsilon_common import (ROOT, PRODUCTION, PROTOCOL, core, torch, np, read,
    check, digest, sources, verify, metadata, shipping_model, shipping_resources)
from epsilon_adversary import solve_ap


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    record=dict(protocol=PROTOCOL,status='running',started_at_utc=core.utc_now(),
        method='fixed saved-policy epsilon AP launch gate',seed=0,lr_net=.01,
        sample_ids=[0],radii=[0.,.2],cases=[],workers=1,threads=1,
        test_evaluated=False,validation_evaluated=False,training_performed=False,
        nominal_cached_not_retrained=True,full_CCG_launched=False,
        environment=shipping_resources.resources())
    frozen=sources()
    protected={}
    try:
        core.configure_worker()
        for n in (50,200,800):
            setup=PRODUCTION/f'N{n}/initialization/seed0/result.json'
            fit=PRODUCTION/f'N{n}/nominal/seed0/lr0.01/result.json'
            for path in (setup,fit):
                protected[str(path)]=digest(path)
            s,f=read(setup),read(fit)
            check(s['status']==f['status']=='completed' and s['complete'] and f['complete'],
                  'Require completed nominal/source data')
            for path,sha in ((s['training_data_path'],s['training_data_sha256']),
                             (f['checkpoint'],f['checkpoint_sha256'])):
                check(digest(path)==sha,'Changed saved nominal artifact')
                protected[path]=sha
        core.write_json(output/'provenance.json',dict(source_sha256=frozen,
            input_sha256=protected,protocol=PROTOCOL))
        with core.timed(record.setdefault('unit_tests',{})):
            with (output/'unit.log').open('x') as log:
                tests=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_epsilon'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record['unit_tests']['returncode']=tests.returncode
            check(tests.returncode==0,'Unit contract tests failed')
        for n in (50,200,800):
            setup=read(PRODUCTION/f'N{n}/initialization/seed0/result.json')
            fit=read(PRODUCTION/f'N{n}/nominal/seed0/lr0.01/result.json')
            data=read(setup['training_data_path'])
            model=shipping_model.model_from_state(torch.load(fit['checkpoint'],
                map_location='cpu',weights_only=True))
            x1,x2=np.asarray(data['x1'][0]),np.asarray(data['x2'][0])
            for radius in (0.,.2):
                row=metadata(n,0,radius,'saved-policy epsilon AP acceptance')
                row.update(sample_id=0,original_sample_id=data['original_sample_ids'][0],
                    lr_net=.01,checkpoint=fit['checkpoint'],checkpoint_sha256=fit['checkpoint_sha256'],
                    training_data_path=setup['training_data_path'],
                    cached_nominal_training_time_s=fit['training']['training_time_s'])
                try:
                    solve_ap(model,x1,x2,radius,row)
                    row['accepted']=True
                except Exception as error:
                    row.update(accepted=False,error=dict(type=type(error).__name__,
                        message=str(error),traceback=traceback.format_exc()))
                record['cases'].append(row)
                core.write_json(output/f'N{n}_radius{radius:g}.json',row)
                core.write_json(output/'summary.json',record)
                print('AP_GATE',n,radius,row['status'],row.get('error',{}).get('message',''),flush=True)
        check(len(record['cases'])==6,'Incomplete predeclared AP case grid')
        check(all(c['accepted'] for c in record['cases']),
              'Epsilon AP gate failed; no full CCG or validation is authorized by this gate')
        record['status']='completed'
    except BaseException as error:
        record.update(status='failed',error=dict(type=type(error).__name__,message=str(error),
                                               traceback=traceback.format_exc()))
    finally:
        try:
            verify(frozen)
            for p,h in protected.items():
                check(digest(p)==h,f'Changed saved input: {p}')
            record['inputs_unchanged']=True
        except Exception:
            record.update(status='failed',inputs_unchanged=False,verification_traceback=traceback.format_exc())
        record.update(ended_at_utc=core.utc_now(),wall_time_s=time.perf_counter()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            timing_scope='current AP gate including tests; excludes imports/final serialization; not training time')
        core.write_json(output/'summary.json',record)
    return 0 if record['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())