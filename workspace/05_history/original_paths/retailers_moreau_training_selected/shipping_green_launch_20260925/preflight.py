"""Full-center fixed-policy readiness; no training or held-out access."""
import argparse
from collections import Counter
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from campaign_common import (ROOT,PRODUCTION,PROTOCOL,REPLAY,core,np,torch,adversary,
    shipping_model,shipping_resources,metadata,read,check,digest,sources,verify)
from shipping_pool import hard_objective,validate_ap,state_hash,pool_hash


def error_info(error):
    return dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    summary=dict(protocol=PROTOCOL,status='running',started_at_utc=core.utc_now(),
        cases=[],seed=0,lr_net=.01,workers=1,threads=1,training_performed=False,
        validation_evaluated=False,test_evaluated=False,full_CCG_launched=False,
        environment=shipping_resources.resources())
    frozen=sources()
    protected={}
    try:
        core.configure_worker()
        for n in (50,200,800):
            for path in (PRODUCTION/f'N{n}/initialization/seed0/result.json',
                         PRODUCTION/f'N{n}/nominal/seed0/lr0.01/result.json'):
                protected[str(path)]=digest(path)
        core.write_json(output/'provenance.json',dict(source_sha256=frozen,
            input_sha256=protected,protocol=PROTOCOL))
        with core.timed(summary.setdefault('unit_tests',{})):
            with (output/'unit.log').open('x') as log:
                for cwd,module in ((REPLAY,'test_epsilon'),(ROOT,'test_pool')):
                    result=subprocess.run([sys.executable,'-B','-m','unittest','-v',module],
                        cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
                    check(result.returncode==0,f'Regression tests failed: {module}')
        for n in (50,200,800):
            setup=read(PRODUCTION/f'N{n}/initialization/seed0/result.json')
            fit=read(PRODUCTION/f'N{n}/nominal/seed0/lr0.01/result.json')
            check(setup['status']==fit['status']=='completed' and setup['complete'] and fit['complete'],
                  'Incomplete saved nominal/data')
            check(digest(fit['checkpoint'])==fit['checkpoint_sha256'] and
                  digest(setup['training_data_path'])==setup['training_data_sha256'],'Changed saved inputs')
            data=read(setup['training_data_path'])
            centers=np.concatenate((data['x1'],data['x2']),axis=1).astype(np.float64)
            check(centers.shape==(4*n//5,6),'Invalid training centers')
            for radius in (0.,.2):
                folder=output/f'N{n}_radius{radius:g}'
                folder.mkdir()
                case=metadata(n,0,radius,'full-center shipping controller readiness')
                case.update(samples=[],failed_sample_ids=[],lr_net=.01,
                    checkpoint=fit['checkpoint'],checkpoint_sha256=fit['checkpoint_sha256'],
                    cached_nominal_training_time_s=fit['training']['training_time_s'])
                summary['cases'].append(case)
                try:
                    with core.timed(case):
                        model=shipping_model.model_from_state(torch.load(fit['checkpoint'],
                            map_location='cpu',weights_only=True))
                        frozen_state=state_hash(model)
                        pools=[center[None,:].copy() for center in centers]
                        case['pool_sha256']=pool_hash(pools)
                        evaluation=case['hard_pool_evaluation']={}
                        with core.timed(evaluation),torch.no_grad():
                            model.eval()
                            maxima=hard_objective(pools)(model).cpu().numpy()
                            evaluation.update(per_ball_costs=maxima.tolist(),mean=float(maxima.mean()),
                                projection=model.proj_layer.drain_summary())
                        for i,center in enumerate(centers):
                            row=metadata(n,0,radius,'epsilon_AP_and_controller_validation')
                            row.update(sample_id=i,original_sample_id=data['original_sample_ids'][i],
                                checkpoint_sha256=fit['checkpoint_sha256'],pool_cost=float(maxima[i]),
                                controller_started_at_utc=core.utc_now(),controller_status='running')
                            t=time.perf_counter()
                            try:
                                result=adversary.solve_ap(model,center[:3],center[3:],radius,row)
                                validate_ap(result,row,i,center,pools[i],radius,float(maxima[i]))
                                check(state_hash(model)==frozen_state,'AP changed fixed policy')
                                row['controller_status']='completed'
                            except Exception as error:
                                case['failed_sample_ids'].append(i)
                                row.update(controller_status='failed',controller_error=error_info(error))
                            finally:
                                row.update(controller_wall_time_s=time.perf_counter()-t,
                                    controller_ended_at_utc=core.utc_now())
                                core.write_json(folder/f'sample_{i:03d}.json',row)
                                case['samples'].append(row)
                        check(len(case['samples'])==len(centers),'Incomplete AP coverage')
                        check(pool_hash(pools)==case['pool_sha256'],'Changed initial pool')
                        case.update(replay_exceptions=sum(r.get('consistency_exception',False) for r in case['samples']),
                            failure_types=dict(Counter(r['controller_error']['message'] for r in case['samples']
                                if r['controller_status']=='failed')))
                        check(not case['failed_sample_ids'],'Controller readiness has failed samples')
                        a=float(np.mean([r['incumbent']['cost'] for r in case['samples']]))
                        p=evaluation['mean']
                        case.update(ap_incumbent_mean=a,green_gap=abs(a-p)/max(abs(p),1e-9),accepted=True)
                except Exception as error:
                    case.update(status='failed',accepted=False,error=error_info(error))
                case['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                core.write_json(folder/'summary.json',case)
                core.write_json(output/'summary.json',summary)
                print('READINESS',n,radius,case['status'],'failures',len(case['failed_sample_ids']),flush=True)
        check(len(summary['cases'])==6 and all(c.get('accepted') for c in summary['cases']),
              'Full-center readiness failed; full CCG launch blocked')
        summary['status']='completed'
    except BaseException as error:
        summary.update(status='failed',error=error_info(error))
    finally:
        try:
            verify(frozen)
            for p,h in protected.items():
                check(digest(p)==h,f'Changed protected input: {p}')
        except Exception as error:
            summary.update(status='failed',inputs_unchanged=False,verification_error=error_info(error))
        else:
            summary['inputs_unchanged']=True
        summary.update(ended_at_utc=core.utc_now(),wall_time_s=time.perf_counter()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            timing_scope='current readiness body; excludes imports/teardown, no training')
        core.write_json(output/'summary.json',summary)
    return 0 if summary['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())