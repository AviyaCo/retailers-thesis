"""Fresh training-selected retailer rerun; validation only after fixed checkpoints."""
from common import (ROOT,OLD,FROZEN,PROTOCOL,THREAD_ENV,core,legacy_ccg,np,torch,
                    configure_worker,check,verify_sources,metadata,raw_hash,evaluate)
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import shlex
import sys
import time
import traceback

from training import train_model
from ccg_training import run_configuration


def error_info(error):
    return dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())


def failed_future(error,seed,radius,submitted):
    return dict(seed=seed,radius=radius,status='failed',complete=False,error=error_info(error),
        wall_time_s=None,worker_timing='unavailable; exception escaped worker; rerun needed to measure',
        parent_observed_seconds=time.perf_counter()-submitted,
        parent_timing_scope='submission to collection, including queueing; NOT worker runtime')


def publish_radius_selections(evaluations):
    check(len(evaluations)==90 and {(r['seed'],r['radius']) for r in evaluations}==
          {(s,r) for s in core.SEEDS for r in core.RADII} and
          all(r['status']=='completed' for r in evaluations),'Require all90 successful fixed-policy evaluations')
    selections=[]
    for seed in core.SEEDS:
        chosen=min((r for r in evaluations if r['seed']==seed),key=lambda r:(r['mean'],r['radius']))
        selections.append(dict(seed=seed,radius=chosen['radius'],validation_mean=chosen['mean'],
            checkpoint_sha256=chosen['checkpoint_sha256'],state_sha256=chosen['state_sha256'],
            terminal_status=chosen['terminal_status']))
    return selections


def freeze(output):
    source=dict(FROZEN)
    for path in ROOT.glob('*'):
        if path.is_file() and path.suffix in ('.py','.sh','.md'):
            source[str(path)]=core.digest(path)
    original_summary=OLD/'ccg_results/summary.json'
    source[str(original_summary)]=core.digest(original_summary)
    source[str(OLD/'ccg_results/provenance.json')]=core.digest(OLD/'ccg_results/provenance.json')
    for seed in core.SEEDS:
        path=OLD/'lr_search_results_v2'/f'seed_{seed}'/'split.json'
        source[str(path)]=core.digest(path)
    manifest=dict(protocol=PROTOCOL,source_sha256=source,created_at_utc=core.utc_now(),
        versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','gurobipy','moreau-cpu')},
        environment=dict(host=platform.node(),platform=platform.platform(),python=sys.version,
            executable=sys.executable,logical_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0))))
    verify_sources(manifest)
    core.write_json(output/'provenance.json',manifest)
    (output/'provenance.json').chmod(0o444)
    return manifest


def nominal_seed(seed,output,provenance_path):
    """Train all three fresh LRs before evaluating any of their fixed checkpoints."""
    folder=Path(output)/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=False)
    record=metadata(seed,0.,'nominal_LR_search')
    record.update(configurations=[],complete=False,winner=None,workers=10,validation_used_during_training=False)
    try:
        with core.timed(record):
            manifest=json.loads(Path(provenance_path).read_text())
            verify_sources(manifest)
            data=core.split_data(seed)
            oldsplit=json.loads((OLD/'lr_search_results_v2'/f'seed_{seed}'/'split.json').read_text())
            check(raw_hash(data)==oldsplit['raw_data_sha256'],'Old/new sample identity mismatch')
            for key,value in data.items():
                check(np.array_equal(np.asarray(value),np.asarray(oldsplit['arrays'][key])),f'Split differs: {key}')
            del oldsplit
            data_hash=raw_hash(data)
            ids=data['train_indices']
            centers=np.concatenate([data['xi1'][ids],data['xi2'][ids]],axis=1)
            train_path=folder/'training_data.json'
            core.write_json(train_path,dict(centers=centers,original_sample_ids=ids,data_sha256=data_hash))
            train_path.chmod(0o444)
            record.update(data_sha256=data_hash,training_data_path=str(train_path),
                          training_data_sha256=core.digest(train_path))
            core.write_json(folder/'split.json',dict(arrays=data,raw_data_sha256=data_hash))
            saa=record['saa_initialization']=metadata(seed,0.,'SAA_initialization')
            q10,q1r=core.solve_saa(centers[:,:3],centers[:,3:],saa)
            initial=core.make_model(q10,q1r)
            initial_path=folder/'initial.pth'
            torch.save(core.weights(initial),initial_path)
            record['initial_sha256']=core.digest(initial_path)
            del initial
            for lr in core.LR_GRID:
                lr_folder=folder/f'lr_{lr:g}'
                item=metadata(seed,0.,'nominal_moreau')
                item.update(lr_net=lr,initial_sha256=record['initial_sha256'])
                record['configurations'].append(item)
                with core.timed(item):
                    model=core.model_from_state(torch.load(initial_path,map_location='cpu',weights_only=True))
                    training=item['training']=metadata(seed,0.,'nominal_training')
                    model=train_model(model,lambda m:m.per_sample_cost(data['train_x1'],data['train_x2']).mean(),
                        lr,lr_folder,training,epochs=3000,patience=300)
                    checkpoint=lr_folder/'selected.pth'
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    check(all(torch.equal(v,saved[k]) for k,v in core.weights(model).items()),'Nominal returned checkpoint differs')
                    item.update(checkpoint=str(checkpoint),checkpoint_sha256=core.digest(checkpoint),
                                state_sha256=legacy_ccg.state_hash(model))
                    train_eval=item['train_evaluation']=metadata(seed,0.,'nominal_train_evaluation')
                    train_mean=evaluate(model,data['train_x1'],data['train_x2'],train_eval)
                    check(abs(train_mean-training['selected_training_loss'])<=1e-6,'Nominal train checkpoint replay mismatch')
                    check(training['selected_training_loss']<=training['first_train'],'Nominal selected training loss worsened')
                    item['train_mean']=train_mean
                    # Full raw epoch records remain in training.json/epochs.jsonl.
                    item['training']={k:v for k,v in training.items() if k!='epoch_history'}
                    del model
                core.write_json(lr_folder/'result.json',item)
                core.write_json(folder/'summary.json',record)
            fixed=[dict(lr_net=r['lr_net'],checkpoint=r['checkpoint'],
                        checkpoint_sha256=r['checkpoint_sha256'],state_sha256=r['state_sha256']) for r in record['configurations']]
            core.write_json(folder/'pre_validation_checkpoints.json',dict(checkpoints=fixed,
                fixed_at_utc=core.utc_now(),rule=PROTOCOL['checkpoint']))
            record['pre_validation_sha256']=core.digest(folder/'pre_validation_checkpoints.json')
            validation=record['validation_phase']={}
            with core.timed(validation):
                for item in record['configurations']:
                    check(core.digest(item['checkpoint'])==item['checkpoint_sha256'],'Nominal changed before validation')
                    model=core.model_from_state(torch.load(item['checkpoint'],map_location='cpu',weights_only=True))
                    check(legacy_ccg.state_hash(model)==item['state_sha256'],'Nominal validation state mismatch')
                    evaluation=item['validation_evaluation']=metadata(seed,0.,'nominal_validation_after_training')
                    item['validation_mean']=evaluate(model,data['val_x1'],data['val_x2'],evaluation)
                    core.write_json(Path(item['checkpoint']).parent/'result.json',item)
            selection=record['lr_selection']={}
            with core.timed(selection):
                winner=min(record['configurations'],key=lambda r:(r['validation_mean'],r['lr_net']))
                record['winner']=dict(seed=seed,lr_net=winner['lr_net'],checkpoint=winner['checkpoint'],
                    checkpoint_sha256=winner['checkpoint_sha256'],state_sha256=winner['state_sha256'],
                    train_mean=winner['train_mean'],validation_mean=winner['validation_mean'],
                    training_timing=winner['training'],data_sha256=data_hash,
                    training_data_path=str(train_path),training_data_sha256=record['training_data_sha256'])
            verify_sources(manifest)
            record['complete']=True
    except Exception as error:
        record.update(status='failed',complete=False,error=error_info(error),winner=None)
    finally:
        core.write_json(folder/'summary.json',record)
    return dict(seed=seed,status=record['status'],complete=record['complete'],winner=record['winner'],
        summary_path=str(folder/'summary.json'),summary_sha256=core.digest(folder/'summary.json'),
        wall_time_s=record['wall_time_s'])


def require_complete(rows,expected):
    check(len(rows)==expected and all(r['status']=='completed' and r['complete'] for r in rows),
          'Incomplete phase: preserve failures; do not proceed or select from partial grid')


def final_validation(output,sweep):
    """Freeze ALL90 terminal policies before any post-CCG validation evaluation."""
    jobs=sweep['ccg']['jobs']
    require_complete(jobs,90)
    check({(r['seed'],r['radius']) for r in jobs}=={(s,r) for s in core.SEEDS for r in core.RADII},'Grid coverage mismatch')
    fixed=[]
    for job in jobs:
        check(core.digest(job['summary_path'])==job['summary_sha256'],'Job summary changed')
        check(core.digest(job['selected_checkpoint'])==job['selected_checkpoint_sha256'],'Terminal checkpoint changed')
        fixed.append({k:job[k] for k in ('seed','radius','selected_candidate_id','selected_checkpoint',
            'selected_checkpoint_sha256','selected_state_sha256','selected_separation_status','terminal_status')})
    path=output/'pre_validation_terminal_policies.json'
    core.write_json(path,dict(fixed_at_utc=core.utc_now(),policies=fixed,rule='terminal; no validation-based candidate choice'))
    path.chmod(0o444)
    result=sweep['validation']=dict(pre_validation_sha256=core.digest(path),evaluations=[],selections=[])
    with core.timed(result):
        for seed in core.SEEDS:
            data=core.split_data(seed)
            nominal=next(x['winner'] for x in sweep['nominal']['jobs'] if x['seed']==seed)
            check(raw_hash(data)==nominal['data_sha256'],'Validation regenerated split mismatch')
            for job in sorted((x for x in jobs if x['seed']==seed),key=lambda x:x['radius']):
                record=metadata(seed,job['radius'],'terminal_validation')
                record.update(checkpoint_sha256=job['selected_checkpoint_sha256'],state_sha256=job['selected_state_sha256'],
                    terminal_status=job['terminal_status'],candidate_id=job['selected_candidate_id'])
                result['evaluations'].append(record)
                try:
                    with core.timed(record.setdefault('load_and_evaluation',{})):
                        check(core.digest(job['selected_checkpoint'])==job['selected_checkpoint_sha256'],'Terminal bytes changed before load')
                        model=core.model_from_state(torch.load(job['selected_checkpoint'],map_location='cpu',weights_only=True))
                        check(legacy_ccg.state_hash(model)==job['selected_state_sha256'],'Terminal evaluation state differs')
                        evaluate(model,data['val_x1'],data['val_x2'],record)
                except Exception as error:
                    record.update(status='failed',error=error_info(error))
                    raise
                finally:
                    core.write_json(output/'validation_progress.json',result)
        for job in jobs:
            check(core.digest(job['selected_checkpoint'])==job['selected_checkpoint_sha256'],'Checkpoint changed in final validation')
        check(core.digest(path)==result['pre_validation_sha256'],'Frozen terminal selection changed')
        result['selections']=publish_radius_selections(result['evaluations'])


def summarize(output,sweep):
    nominal=[json.loads(Path(j['summary_path']).read_text()) for j in sweep['nominal']['jobs']]
    jobs=[json.loads(Path(j['summary_path']).read_text()) for j in sweep['ccg']['jobs']]
    rows=[]
    mp_rows=[]
    for j in jobs:
        for c in j['iterations']:
            if 'mp' in c:
                t=c['mp']
                mp_rows.append(dict(seed=j['seed'],radius=j['radius'],mp_iteration=t['mp_iteration'],
                    incoming=t['first_train'],selected=t['selected_training_loss'],
                    returned=t['returned_pool_mean'],delta=t['same_pool_change'],
                    unchanged_weights=t['unchanged_weights'],epochs=t['epochs_evaluated'],
                    started_at_utc=t['started_at_utc'],ended_at_utc=t['ended_at_utc'],
                    wall_time_s=t['wall_time_s'],status=t['status']))
    for radius in core.RADII:
        group=[j for j in jobs if j['radius']==radius]
        evaluations=[r for r in sweep['validation']['evaluations'] if r['radius']==radius]
        rows.append(dict(radius=radius,Ntrain=40,count=len(group),
            validation_mean=float(np.mean([r['mean'] for r in evaluations])),
            initial_ap_mean=float(np.mean([j['iterations'][0]['ap_incumbent_mean'] for j in group])),
            terminal_ap_mean=float(np.mean([j['iterations'][-1]['ap_incumbent_mean'] for j in group])),
            terminal_ap_upper_mean=float(np.mean([j['iterations'][-1]['ap_bound_mean'] for j in group])),
            mean_paired_ap_change=float(np.mean([j['iterations'][-1]['ap_incumbent_mean']-j['iterations'][0]['ap_incumbent_mean'] for j in group])),
            mp_count=sum(j['mp_count'] for j in group),
            unchanged_nominal_count=sum(j['nominal_state_sha256']==j['selected_state_sha256'] for j in group),
            separated_count=sum(j['terminal_status']=='separation_satisfied' for j in group),
            stagnation_count=sum(j['terminal_status']=='stagnation' for j in group),
            iteration_cap_count=sum(j['terminal_status']=='iteration_cap' for j in group),
            runtime_s={k:core.stats(j[k] for j in group) for k in ('wall_time_s','ap_total_time_s','mp_total_time_s','evaluation_time_s')},
            validation_time_s=core.stats(r['wall_time_s'] for r in evaluations)))
    metrics=dict(per_radius=rows,mp=mp_rows,
        mp_selected_improved=sum(r['selected']<r['incoming'] for r in mp_rows),
        mp_selected_worsened=sum(r['selected']>r['incoming'] for r in mp_rows),
        mp_unchanged=sum(r['unchanged_weights'] for r in mp_rows),
        nominal_validation_mean=float(np.mean([j['winner']['validation_mean'] for j in nominal])),
        radius_selected_validation_mean=float(np.mean([r['validation_mean'] for r in sweep['validation']['selections']])),
        nominal_training_by_lr={str(lr):core.stats(c['training']['wall_time_s'] for j in nominal for c in j['configurations'] if c['lr_net']==lr) for lr in core.LR_GRID},
        nominal_winner_training=core.stats(j['winner']['training_timing']['wall_time_s'] for j in nominal),
        not_applicable=dict(SRO_FA='not run',CV='fixed40/10 split, not CV',test='not evaluated'),
        scope='Validation reused for LR and radius selection; not independent generalization or a U-shape test')
    core.write_json(output/'aggregates.json',metrics)
    return metrics


def main():
    output=ROOT/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    sweep=dict(experiment=PROTOCOL['experiment'],status='running',protocol=PROTOCOL,
        started_at_utc=core.utc_now(),started_perf_counter=started,
        command=shlex.join(sys.orig_argv),cwd=str(Path.cwd()),host=platform.node(),
        timing_clock='time.perf_counter',timing_unit='seconds',thread_environment=THREAD_ENV,
        nominal=dict(jobs=[]),ccg=dict(jobs=[]),validation=None,errors=[],
        final_selection_published=False,test_evaluated=False,output_directory=str(output))
    exitcode=1
    try:
        configure_worker()
        manifest=freeze(output)
        sweep['provenance_sha256']=core.digest(output/'provenance.json')
        context=multiprocessing.get_context('spawn')
        core.write_json(output/'summary.json',sweep)
        print(f'OUTPUT={output}',flush=True)
        with core.timed(sweep['nominal']):
            with ProcessPoolExecutor(max_workers=10,mp_context=context,initializer=configure_worker) as executor:
                futures={executor.submit(nominal_seed,s,output/'nominal',output/'provenance.json'):(s,time.perf_counter()) for s in core.SEEDS}
                for future in as_completed(futures):
                    try:
                        item=future.result()
                        sweep['nominal']['jobs'].append(item)
                        print(f"nominal seed={item['seed']} {item['status']}",flush=True)
                    except Exception as error:
                        seed,submitted=futures[future]
                        sweep['nominal']['jobs'].append(failed_future(error,seed,0.,submitted))
                        sweep['errors'].append(dict(phase='nominal',seed=seed,**error_info(error)))
                    core.write_json(output/'summary.json',sweep)
            require_complete(sweep['nominal']['jobs'],10)
        verify_sources(manifest)
        with core.timed(sweep['ccg']):
            with ProcessPoolExecutor(max_workers=16,mp_context=context,initializer=configure_worker) as executor:
                futures={}
                for n in sweep['nominal']['jobs']:
                    check(core.digest(n['summary_path'])==n['summary_sha256'],'Nominal selection changed')
                    for radius in core.RADII:
                        future=executor.submit(run_configuration,n['seed'],radius,output/'ccg',n['winner'],output/'provenance.json')
                        futures[future]=(n['seed'],radius,time.perf_counter())
                for future in as_completed(futures):
                    try:
                        item=future.result()
                        sweep['ccg']['jobs'].append(item)
                        print(f"CCG seed={item['seed']} radius={item['radius']:g} {item['status']} {item['terminal_status']} MPs={item['mp_count']}",flush=True)
                    except Exception as error:
                        seed,radius,submitted=futures[future]
                        sweep['ccg']['jobs'].append(failed_future(error,seed,radius,submitted))
                        sweep['errors'].append(dict(phase='ccg',configuration=[seed,radius],**error_info(error)))
                    core.write_json(output/'summary.json',sweep)
            require_complete(sweep['ccg']['jobs'],90)
        verify_sources(manifest)
        final_validation(output,sweep)
        verify_sources(manifest)
        metrics=summarize(output,sweep)
        sweep.update(status='completed',final_selection_published=True,inputs_unchanged=True,
            aggregates_path=str(output/'aggregates.json'),selections=sweep['validation']['selections'])
        exitcode=0
        print(f"ALL COMPLETE nominal validation={metrics['nominal_validation_mean']:.9f}; selected validation={metrics['radius_selected_validation_mean']:.9f}; MPs improved={metrics['mp_selected_improved']} worsened={metrics['mp_selected_worsened']} unchanged={metrics['mp_unchanged']}",flush=True)
    except Exception as error:
        sweep.update(status='failed',final_selection_published=False,error=error_info(error))
        sweep['selections']=[]
        if sweep['validation'] is not None:
            sweep['validation']['selections']=[]
        print(traceback.format_exc(),flush=True)
    finally:
        elapsed=time.perf_counter()-started
        sweep.update(ended_at_utc=core.utc_now(),ended_perf_counter=time.perf_counter(),wall_time_s=elapsed,
            wall_time_human=f'{elapsed:.6f}s / {elapsed/60:.6f}min / {elapsed/3600:.6f}h')
        core.write_json(output/'summary.json',sweep)
        print(f"{sweep['status']}: {output} elapsed={elapsed:.3f}s",flush=True)
    return exitcode


if __name__=='__main__':
    sys.exit(main())