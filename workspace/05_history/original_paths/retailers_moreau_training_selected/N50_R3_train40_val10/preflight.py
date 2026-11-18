"""Tests plus a tiny real train-only native/AP integration check, not a tuned pilot."""
from common import ROOT,core,legacy_ccg,adversary,np,torch,configure_worker,metadata,check
from training import train_model
from datetime import datetime,timezone
import json
import subprocess
import sys
import time
import traceback


def main():
    folder=ROOT/'preflight'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder.mkdir(parents=True,exist_ok=False)
    record=dict(experiment='training_selected_retailer_preflight',status='running',
        started_at_utc=core.utc_now(),seed=0,radius=.2,sample_size=40,
        test_access=False,validation_evaluation=False,threads=1,
        scope='unit tests plus two nominal training epochs and one sample AP; not model performance')
    start=time.perf_counter()
    try:
        configure_worker()
        test=record['tests']={}
        with core.timed(test):
            result=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_training','test_ccg_training','test_campaign'],
                cwd=ROOT,capture_output=True,text=True)
            (folder/'tests.log').write_text(result.stdout+result.stderr)
            test['exit_code']=result.returncode
            check(result.returncode==0,'Unit tests failed; inspect preserved test log')
        data=core.split_data(0)
        ids=data['train_indices']
        saa=record['saa']=metadata(0,0.,'preflight_SAA')
        q10,q1r=core.solve_saa(data['xi1'][ids],data['xi2'][ids],saa)
        model=core.make_model(q10,q1r)
        training=record['training']=metadata(0,0.,'preflight_two_epochs')
        model=train_model(model,lambda m:m.per_sample_cost(data['train_x1'],data['train_x2']).mean(),
            .005,folder/'training',training,epochs=2,patience=300)
        with torch.no_grad():
            measured=float(model.per_sample_cost(data['train_x1'],data['train_x2']).mean())
        check(abs(measured-training['selected_training_loss'])<=1e-6,'Native selected replay differs')
        check(training['validation_time_s']==0.,'Validation called during training')
        center=np.concatenate([data['xi1'][ids[0]],data['xi2'][ids[0]]])
        pool=center[None,:]
        with torch.no_grad():
            value=float(model.per_sample_cost(torch.tensor(pool[:,:3],dtype=torch.float32),
                torch.tensor(pool[:,3:],dtype=torch.float32))[0])
        ap=record['ap']=dict(seed=0,radius=.2,sample_id=0,candidate_id=0)
        answer=adversary.solve_ap(model,center[:3],center[3:],.2,ap)
        legacy_ccg.validate_ap(answer,ap,0,center,pool,.2,value)
        record.update(status='completed',source_sha256={str(p):core.digest(p) for p in ROOT.glob('*.py')})
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise
    finally:
        record.update(ended_at_utc=core.utc_now(),wall_time_s=time.perf_counter()-start)
        core.write_json(folder/'summary.json',record)
        print(f"Preflight {record['status']}: {folder}; {record['wall_time_s']:.6f}s",flush=True)


if __name__=='__main__':
    main()