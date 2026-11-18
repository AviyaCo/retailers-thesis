"""Capture unchanged executions and separate target error from projection error."""
import argparse
from collections import Counter
from fractions import Fraction
import importlib.metadata
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from common import (ROOT,READINESS,PRODUCTION,PROTOCOL,core,np,torch,shipping_model,
    resources,project,binary,read,check,digest,sources,verify,metadata,state_hash)


def error_info(error):
    return dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())


def observe_call(layer,call,record):
    """Passive hooks: no replacement tensors, solves, casts or clamps in hooks."""
    inputs=[]
    outputs=[]
    def before(module,args):
        inputs.append((args[0].detach().clone(),args[1].detach().clone()))
    def after(module,args,result):
        outputs.append(result[0].detach().clone())
    pre=layer.register_forward_pre_hook(before)
    post=layer.register_forward_hook(after)
    result=None
    try:
        with core.timed(record),torch.no_grad():
            result=call()
            check(len(inputs)==len(outputs)==1,'Expected exactly one projection call')
    except Exception as error:
        record['exception']=error_info(error)
    finally:
        pre.remove();post.remove()
        record['telemetry']=layer.drain_summary()
        record['q_available']=bool(outputs)
        if inputs:
            z,cap=inputs[0]
            record.update(z=z.cpu().tolist(),capacity=cap.cpu().tolist(),
                z_dtype=str(z.dtype),capacity_dtype=str(cap.dtype),batch_size=len(z),
                float32_to_float64_value_preserved=bool(torch.equal(z,z.double().to(z.dtype))))
        record['q']=outputs[0].cpu().tolist() if outputs else None
    return result,inputs[0] if inputs else None


def reference_rows(capture,record):
    """No numerical reference replaces production values or acceptance checks."""
    with core.timed(record):
        rows=[]
        for i,(z,cap) in enumerate(zip(capture['z'],capture['capacity'])):
            exact=project(tuple(binary(v) for v in z),binary(cap[0]))
            entry=dict(exact_q=[str(v) for v in exact],exact_q_float=list(map(float,exact)),
                       exact_support=[v>0 for v in exact])
            if capture['q'] is not None:
                q=capture['q'][i]
                entry.update(native_support=[v>0 for v in q],
                    max_projection_error=float(max(abs(binary(v)-e) for v,e in zip(q,exact))),
                    support_matches=all((v>0)==(e>0) for v,e in zip(q,exact)),
                    physical_residual=float(max(Fraction(0),-min(map(binary,q)),sum(map(binary,q))-binary(cap[0]))),
                    raw_fee=.1*sum(v>0 for v in q),reference_fee=.1*sum(v>0 for v in exact))
            rows.append(entry)
        record['rows']=rows
    return rows


def native(model,x1,x2,record):
    result,_=observe_call(model.proj_layer,lambda:model.costs_and_shipments(x1,x2),record)
    if result is not None:
        smooth,hard,q=result
        check(record['q']==q.detach().cpu().tolist(),'Hook output differs from returned q')
        record.update(smooth_cost=smooth.cpu().tolist(),hard_cost=hard.cpu().tolist())
    if 'z' in record:
        reference_rows(record,record.setdefault('reference',{}))


def compare(row,saved,batch,i,center,radius):
    if row['native']['status']!='completed':
        return dict(available=False)
    a=row['native'];ref=a['reference']['rows'][0]
    old=saved['activation_replay']
    exact_ap=[Fraction(v) for v in old['exact_q']]
    exact_native=[Fraction(v) for v in ref['exact_q']]
    point=np.concatenate((saved['incumbent']['worst1'],saved['incumbent']['worst2']))
    same=np.array_equal(point,center)
    comparison=dict(available=True,saved_failure=saved.get('controller_error',{}).get('message'),
        saved_q_reproduced=a['q'][0]==old['native_q'],
        saved_cost_reproduced=a['hard_cost'][0]==old['native_hard_cost'],
        max_projection_error=ref['max_projection_error'],physical_residual=ref['physical_residual'],
        native_vs_own_exact_support_matches=ref['support_matches'],
        native_exact_vs_folded_exact_support_matches=ref['exact_support']==[q>0 for q in exact_ap],
        max_native_vs_folded_exact_q_difference=float(max(abs(x-y) for x,y in zip(exact_native,exact_ap))),
        max_native_vs_folded_target_difference=float(max(abs(binary(x)-Fraction(y))
            for x,y in zip(a['z'][0],old['exact_target']))),
        active_components=[dict(component=j,AP_q=saved['incumbent']['q2'][j],
            native_q=a['q'][0][j],exact_native_q=str(exact_native[j]),
            exact_native_below_epsilon_minus_tol=exact_native[j]<binary(1e-6-1e-9))
            for j,v in enumerate(saved['incumbent']['zship']) if v>.5],
        center_equals_AP_point=bool(same))
    if radius==0:
        check(same,'Radius-zero point differs from center')
        if batch['status']=='completed':
            br=batch['reference']['rows'][i]
            comparison.update(batch_and_singleton_target_identical=batch['z'][i]==a['z'][0],
                max_batch_singleton_target_difference=float(np.max(np.abs(np.array(batch['z'][i])-a['z'][0]))),
                batch_singleton_q_identical=batch['q'][i]==a['q'][0],
                batch_singleton_exact_projection_identical=br['exact_q']==ref['exact_q'],
                batch_singleton_exact_support_identical=br['exact_support']==ref['exact_support'],
                batch_singleton_native_support_identical=br['native_support']==ref['native_support'],
                batch_minus_singleton_hard_cost=batch['hard_cost'][i]-a['hard_cost'][0],
                batch_minus_singleton_fee=br['raw_fee']-ref['raw_fee'],
                batch_capacity_equals_singleton=batch['capacity'][i]==a['capacity'][0])
    return comparison


def fixed_controls(batch,n,radius,repeat):
    record=metadata(n,radius,'fixed_target_projection_singletons',repeat)
    record['rows']=[]
    layer=core.NativeProjection()
    with core.timed(record):
        for i,(z,cap) in enumerate(zip(batch['z'],batch['capacity'])):
            item=dict(sample_id=i)
            record['rows'].append(item)
            zt=torch.tensor([z],dtype=torch.float32)
            ct=torch.tensor([cap],dtype=torch.float32)
            check(zt.tolist()==[z] and ct.tolist()==[cap],'Fixed-target input changed')
            observe_call(layer,lambda:layer(zt,ct,core._SOURCE_SOLVER_ARGS),item)
            if item['q_available']:
                reference_rows(item,item.setdefault('reference',{}))
                item.update(q_equals_batch=item['q'][0]==batch['q'][i],
                    max_q_difference=float(np.max(np.abs(np.array(item['q'][0])-batch['q'][i]))),
                    support_equals_batch=[v>0 for v in item['q'][0]]==[v>0 for v in batch['q'][i]])
        check(all(r['status']=='completed' for r in record['rows']),'Fixed-target solve failed; no replacement')
    return record


def run_case(n,radius,repeat,saved_case,output):
    case=metadata(n,radius,'native_projection_diagnostic',repeat)
    case.update(rows=[],full_CCG_launched=False)
    folder=output/f'N{n}_radius{radius:g}_repeat{repeat}'
    folder.mkdir()
    try:
        with core.timed(case):
            with core.timed(case.setdefault('initialization',{})):
                fit=read(PRODUCTION/f'N{n}/nominal/seed0/lr0.01/result.json')
                setup=read(PRODUCTION/f'N{n}/initialization/seed0/result.json')
                check(digest(fit['checkpoint'])==fit['checkpoint_sha256'],'Checkpoint changed')
                check(digest(setup['training_data_path'])==setup['training_data_sha256'],'Training data changed')
                data=read(setup['training_data_path'])
                centers=np.concatenate((data['x1'],data['x2']),axis=1).astype(np.float64)
                check(centers.shape==(4*n//5,6),'Unexpected training data size')
                model=shipping_model.model_from_state(torch.load(fit['checkpoint'],map_location='cpu',weights_only=True))
                model.eval()
                frozen_state=state_hash(model)
                case.update(checkpoint_sha256=fit['checkpoint_sha256'],state_sha256=frozen_state,
                    cached_nominal_training_time_s=fit['training']['training_time_s'],
                    settings=model.proj_layer.settings,P_values=model.proj_layer.p.tolist(),
                    A_values=model.proj_layer.a.tolist())
            batch=case['batch']={}
            native(model,torch.tensor(centers[:,:3],dtype=torch.float32),
                   torch.tensor(centers[:,3:],dtype=torch.float32),batch)
            if batch['status']=='completed':
                batch['saved_pool_costs_reproduced']=batch['hard_cost']==saved_case['hard_pool_evaluation']['per_ball_costs']
            with core.timed(case.setdefault('singleton_sequence',{})):
                for i,saved in enumerate(saved_case['samples']):
                    check(saved['sample_id']==i and saved['original_sample_id']==data['original_sample_ids'][i],
                          'Saved row identity mismatch')
                    row=metadata(n,radius,'saved_AP_witness_native_replay',repeat)
                    row.update(sample_id=i,original_sample_id=data['original_sample_ids'][i])
                    with core.timed(row):
                        inc=saved['incumbent']
                        native(model,torch.tensor([inc['worst1']],dtype=torch.float32),
                            torch.tensor([inc['worst2']],dtype=torch.float32),row.setdefault('native',{}))
                        row['comparison']=compare(row,saved,batch,i,centers[i],radius)
                    core.write_json(folder/f'sample_{i:03d}.json',row)
                    case['rows'].append(row)
            # The historical sequence above is complete before controls run.
            if batch['status']=='completed':
                case['fixed_controls']=fixed_controls(batch,n,radius,repeat)
            check(state_hash(model)==frozen_state,'Diagnostic changed learned state')
            case['state_unchanged']=True
            check(batch['status']=='completed' and all(r['native']['status']=='completed' for r in case['rows']),
                  'Native execution failed; no fallback or omitted row')
            case['observations']=summarize(case)
    except Exception as error:
        case.update(status='failed',error=error_info(error))
    case['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    core.write_json(folder/'summary.json',case)
    return case


def summarize(case):
    comparisons=[r['comparison'] for r in case['rows']]
    batch=case['batch']
    controls=case['fixed_controls']['rows']
    return dict(samples=len(comparisons),
        saved_pool_costs_reproduced=batch['saved_pool_costs_reproduced'],
        saved_q_reproduced=sum(c['saved_q_reproduced'] for c in comparisons),
        saved_cost_reproduced=sum(c['saved_cost_reproduced'] for c in comparisons),
        projection_support_mismatches=sum(not c['native_vs_own_exact_support_matches'] for c in comparisons),
        max_projection_error=max(c['max_projection_error'] for c in comparisons),
        physical_residual_over_1e9=sum(c['physical_residual']>1e-9 for c in comparisons),
        target_support_mismatches=sum(not c['native_exact_vs_folded_exact_support_matches'] for c in comparisons),
        batch_singleton_target_differences=sum(not c.get('batch_and_singleton_target_identical',True) for c in comparisons)
            if case['radius']==0 else None,
        batch_singleton_native_support_differences=sum(not c.get('batch_singleton_native_support_identical',True)
            for c in comparisons) if case['radius']==0 else None,
        batch_singleton_exact_support_differences=sum(not c.get('batch_singleton_exact_support_identical',True)
            for c in comparisons) if case['radius']==0 else None,
        fixed_target_q_differences=sum(not r['q_equals_batch'] for r in controls),
        fixed_target_support_differences=sum(not r['support_equals_batch'] for r in controls))


def repetitions_match(first,second):
    fields=('z','capacity','q','hard_cost','smooth_cost')
    return dict(batch_identical=all(first['batch'][k]==second['batch'][k] for k in fields),
        singleton_identical_count=sum(all(a['native'][k]==b['native'][k] for k in fields)
            for a,b in zip(first['rows'],second['rows'])),
        fixed_target_identical_count=sum(a['q']==b['q'] for a,b in
            zip(first['fixed_controls']['rows'],second['fixed_controls']['rows'])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    output=parser.parse_args().output.resolve();output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    result=dict(protocol=PROTOCOL,status='running',started_at_utc=core.utc_now(),cases=[],
        environment=resources(),timing_clock='time.perf_counter',timing_unit='seconds')
    frozen=sources()
    try:
        core.configure_worker()
        result['versions']={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','gurobipy','moreau-cpu')}
        core.write_json(output/'provenance.json',dict(source_and_input_sha256=frozen,protocol=PROTOCOL))
        with core.timed(result.setdefault('tests',{})):
            with (output/'unit.log').open('x') as log:
                tests=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_diagnose'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            check(tests.returncode==0,'Diagnostic tests failed')
        saved=read(READINESS/'preflight_v1/summary.json')
        check([(c['Ntotal'],c['radius']) for c in saved['cases']]==[(n,r) for n in (50,200,800) for r in (0.,.2)],
              'Unexpected saved configuration grid')
        for saved_case in saved['cases']:
            n,radius=saved_case['Ntotal'],saved_case['radius']
            for repeat in (0,1):
                case=run_case(n,radius,repeat,saved_case,output)
                result['cases'].append(case)
                print('DIAGNOSTIC',n,radius,repeat,case['status'],case.get('observations',case.get('error')),flush=True)
                core.write_json(output/'summary.json',result)
        check(all(c['status']=='completed' for c in result['cases']),'Diagnostic execution failures retained')
        result['repeatability']=[dict(Ntotal=a['Ntotal'],radius=a['radius'],**repetitions_match(a,b))
            for a,b in zip(result['cases'][::2],result['cases'][1::2])]
        result['runtime_groups']=[dict(Ntotal=n,radius=r,status=status,
            wall_time_s=core.stats([c['wall_time_s'] for c in result['cases']
                if (c['Ntotal'],c['radius'],c['status'])==(n,r,status)]))
            for n in (50,200,800) for r in (0.,.2)
            for status in sorted({c['status'] for c in result['cases'] if (c['Ntotal'],c['radius'])==(n,r)})]
        result.update(status='completed',full_CCG_launched=False,acceptance_or_cost_changed=False)
    except BaseException as error:
        result.update(status='failed',error=error_info(error))
    finally:
        try:
            verify(frozen);result['inputs_unchanged']=True
        except Exception as error:
            result.update(status='failed',inputs_unchanged=False,verification_error=error_info(error))
        result.update(ended_at_utc=core.utc_now(),wall_time_s=time.perf_counter()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            timing_scope='current diagnostic body; cached training excluded; imports/teardown measured by supervisor')
        core.write_json(output/'summary.json',result)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())