"""Disposable native full-cycle integration; fixed seed0/LR.01, train only."""
import argparse
from pathlib import Path
import resource
import subprocess
import sys
import traceback
from shipping_live_common import *
from run_campaign import run_stage


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output.resolve();output.mkdir(parents=True,exist_ok=False)
    record=dict(disposable=True,test_evaluated=False,validation_evaluated=False,
        method='fixed native shipping controller integration',cases=[],workers=1,
        phases={},protocol={**PROTOCOL,'full_CCG_launched':False},full_CCG_launched=False)
    try:
        with core.timed(record):
            core.configure_worker()
            frozen=sources()
            manifest=output/'provenance.json'
            core.write_json(manifest,dict(source_sha256=frozen,protocol=PROTOCOL,workers={'ccg':1}))
            manifest.chmod(0o444)
            record.update(provenance_path=str(manifest),provenance_sha256=digest(manifest))
            with core.timed(record.setdefault('tests',{})):
                with (output/'tests.log').open('x') as log:
                    for folder,name in ((GUARD,'test_guards'),(ROOT,'test_live'),(ROOT,'test_campaign')):
                        result=subprocess.run([sys.executable,'-B','-m','unittest','-v',name],cwd=folder,
                            stdout=log,stderr=subprocess.STDOUT)
                        check(result.returncode==0,f'Tests failed: {name}')
            setup=read(PRODUCTION/'N50/initialization/seed0/result.json')
            fit=read(PRODUCTION/'N50/nominal/seed0/lr0.01/result.json')
            check(digest(setup['training_data_path'])==setup['training_data_sha256'],'Changed training data')
            check(digest(fit['checkpoint'])==fit['checkpoint_sha256'],'Changed checkpoint')
            data=read(setup['training_data_path']);train=output/'training_data.json'
            core.write_json(train,dict(centers=np.concatenate((data['x1'],data['x2']),axis=1),
                original_sample_ids=data['original_sample_ids'],data_sha256=data['paired_data_sha256']))
            model=shipping_model.model_from_state(torch.load(fit['checkpoint'],map_location='cpu',weights_only=True))
            nominal=dict(Ntotal=50,seed=0,lr_net=.01,train_mean=fit['training']['selected_training_loss'],
                data_sha256=data['paired_data_sha256'],training_data_path=str(train),training_data_sha256=digest(train),
                checkpoint=fit['checkpoint'],checkpoint_sha256=fit['checkpoint_sha256'],
                state_sha256=legacy_ccg.state_hash(model),training_timing={k:fit['training'][k] for k in (
                    'status','wall_time_s','epochs_evaluated','started_at_utc','ended_at_utc')})
            tasks=[(dict(n_total=50,seed=0,radius=radius,output_root=str(output/'native'),
                nominal_record=nominal),output/f'workers/radius{radius:g}') for radius in (0.,.2)]
            record['cases']=sorted(run_stage('ccg',tasks,1,output,manifest,record),key=lambda r:r['radius'])
            check(all(r['status']=='completed' and r['converged'] for r in record['cases']),
                'Native integration failed/nonconverged')
            check(record['cases'][0]['mp_count']==0 and record['cases'][1]['mp_count']>=1,
                'Integration must exercise candidate0 stop and a real MP cycle')
            verify_sources(read(manifest))
    except Exception as error:
        record.update(error=dict(message=str(error),traceback=traceback.format_exc()))
    finally:
        record['parent_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        record['peak_rss_bytes']=max([record['parent_peak_rss_bytes']]+[
            r['peak_rss_bytes'] for r in record.get('phases',{}).get('ccg',{}).get('jobs',[])
            if 'peak_rss_bytes' in r])
        record['peak_rss_scope']='Maximum parent or worker lifetime peak; not concurrent total; native cases N50 only'
        core.write_json(output/'summary.json',record)
    print(record['status'],record.get('wall_time_s'),flush=True)
    return 0 if record['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())