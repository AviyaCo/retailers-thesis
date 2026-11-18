"""Fresh R3/no-shipping live green, N200/N800; no test access or legacy training."""
from green_common import (ROOT, FROZEN, THREAD_ENV, PROTOCOL, SIZES, SEEDS, RADII,
    LR_GRID, core, legacy_ccg, np, torch, configure_worker, check, verify_sources,
    metadata, split_data, solve_saa, evaluate, raw_hash)
from green_training import train_model
from green_ccg import run_configuration
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import argparse
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shlex
import time
import traceback
import sys


def read(path):
    return json.loads(Path(path).read_text())


def error_info(error):
    return dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())


def require_grid(rows, expected, keys):
    actual = [tuple(r[k] for k in keys) for r in rows]
    check(len(actual) == len(expected) and set(actual) == expected,
          'Incomplete or duplicate configuration grid')
    check(all(r['status'] == 'completed' and r.get('complete', True) for r in rows),
          'Phase failed/nonconverged: no partial selection or substituted run')


def resources():
    mem = {line.split(':')[0]: int(line.split()[1])*1024
           for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:', 'MemTotal:'))}
    affinity = sorted(os.sched_getaffinity(0))
    quota = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    quota_cores = None
    if quota.exists() and int(quota.read_text()) > 0:
        quota_cores = int(quota.read_text()) / int(period.read_text())
    v2 = Path('/sys/fs/cgroup/cpu.max')
    if v2.exists():
        q, p = v2.read_text().split()
        if q != 'max':
            quota_cores = int(q)/int(p)
    memory_limits = []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, group = line.split(':', 2)
        if 'memory' in controllers.split(','):
            location = Path('/sys/fs/cgroup/memory') / group.lstrip('/')
            while location.is_relative_to('/sys/fs/cgroup/memory'):
                limit, usage = location/'memory.limit_in_bytes', location/'memory.usage_in_bytes'
                if limit.exists() and usage.exists():
                    memory_limits.append(max(0,int(limit.read_text())-int(usage.read_text())))
                if location == Path('/sys/fs/cgroup/memory'):
                    break
                location=location.parent
    limit, usage = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and limit.read_text().strip() != 'max':
        memory_limits.append(max(0,int(limit.read_text())-int(usage.read_text())))
    available=min([mem['MemAvailable'], *memory_limits])
    return dict(host=platform.node(), platform=platform.platform(), python=sys.version,
        executable=sys.executable, logical_cpus=os.cpu_count(), affinity=affinity,
        load_average=list(os.getloadavg()), cpu_quota_cores=quota_cores,
        memory_available_bytes=available, host_memory=mem,
        cgroup_memory_headrooms_bytes=memory_limits,
        cgroup=Path('/proc/self/cgroup').read_text(), captured_at_utc=core.utc_now())


def choose_workers(snapshot, preflight):
    effective=len(snapshot['affinity'])
    if snapshot['cpu_quota_cores'] is not None:
        effective=min(effective,math.floor(snapshot['cpu_quota_cores']))
    cpu_slots=effective-math.ceil(max(snapshot['load_average']))-4
    # Resource safety, not an optimization budget: future AP trees can exceed
    # the preflight peak. Reserve at least 4 GiB/worker and 16 GiB for the host.
    peak=max(r['peak_rss_bytes'] for r in preflight['sizes'])
    per_worker=max(4*1024**3,2*peak)
    mem_slots=(snapshot['memory_available_bytes']-16*1024**3)//per_worker
    workers=min(cpu_slots,mem_slots,180)
    check(workers>=1,'Insufficient resources; do not launch')
    return dict(ccg=int(workers), nominal=int(min(workers,60)), initialization=int(min(workers,20)),
        validation=int(min(workers,20)), reserve_cpu_slots=4,
        reserved_background_cpu=math.ceil(max(snapshot['load_average'])),
        memory_budget_bytes_per_worker=per_worker, host_memory_reserve_bytes=16*1024**3,
        observed_preflight_peak_rss_bytes=peak,
        limitation='Sampled resource headroom, not a bound on future MIP tree memory; no solver/time fallback')


def setup_job(n, seed, folder, manifest_path, record):
    folder=Path(folder)
    record.update(metadata(n,seed,0.,'training_only_initialization'))
    with core.timed(record):
        manifest=read(manifest_path)
        verify_sources(manifest)
        data=split_data(n,seed)
        data_hash=raw_hash(data)
        ids=data['train_indices']
        centers=np.concatenate([data['xi1'][ids],data['xi2'][ids]],axis=1)
        train_path=folder/'training_data.json'
        core.write_json(train_path,dict(centers=centers,original_sample_ids=ids,data_sha256=data_hash))
        split_path=folder/'split.json'
        core.write_json(split_path,dict(arrays=data,raw_data_sha256=data_hash))
        saa=record['saa_initialization']=metadata(n,seed,0.,'SAA_initialization')
        q10,q1r=solve_saa(centers[:,:3],centers[:,3:],saa)
        model=core.make_model(q10,q1r)
        initial_path=folder/'initial.pth'
        torch.save(core.weights(model),initial_path)
        record.update(initial_path=str(initial_path),initial_sha256=core.digest(initial_path),
            initial_state_sha256=legacy_ccg.state_hash(model),data_sha256=data_hash,
            training_data_path=str(train_path),training_data_sha256=core.digest(train_path),
            split_path=str(split_path),split_sha256=core.digest(split_path),complete=True)
        for path in (train_path,split_path,initial_path):
            path.chmod(0o444)
        verify_sources(manifest)
    return record


def nominal_job(n, seed, lr, setup, folder, manifest_path, record):
    record.update(metadata(n,seed,0.,'nominal_moreau'))
    record['lr_net']=lr
    with core.timed(record):
        manifest=read(manifest_path)
        verify_sources(manifest)
        for key, hashkey in (('initial_path','initial_sha256'),('training_data_path','training_data_sha256')):
            check(core.digest(setup[key])==setup[hashkey],f'Nominal input changed: {key}')
        # Only the training subset is loaded. All LR arms finish before any validation.
        centers=np.asarray(read(setup['training_data_path'])['centers'],dtype=np.float64)
        check(centers.shape==(record['Ntrain'],6),'Training shape mismatch')
        x1=torch.tensor(centers[:,:3],dtype=torch.float32)
        x2=torch.tensor(centers[:,3:],dtype=torch.float32)
        model=core.model_from_state(torch.load(setup['initial_path'],map_location='cpu',weights_only=True))
        check(legacy_ccg.state_hash(model)==setup['initial_state_sha256'],'Initial policy mismatch')
        training=record['training']=metadata(n,seed,0.,'nominal_training')
        model=train_model(model,lambda current:current.per_sample_cost(x1,x2).mean(),lr,
                          Path(folder)/'training',training,epochs=3000,patience=300)
        checkpoint=Path(folder)/'training/selected.pth'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        check(all(torch.equal(v,saved[k]) for k,v in core.weights(model).items()),'Returned weights mismatch')
        evaluation=record['train_evaluation']=metadata(n,seed,0.,'nominal_train_replay')
        train_mean=evaluate(model,x1,x2,evaluation)
        check(abs(train_mean-training['selected_training_loss'])<=1e-6,'Nominal train replay mismatch')
        check(train_mean<=training['first_train']+1e-6,'Nominal selected training objective worsened')
        record['training']={k:v for k,v in training.items() if k!='epoch_history'}
        record.update(checkpoint=str(checkpoint),checkpoint_sha256=core.digest(checkpoint),
            state_sha256=legacy_ccg.state_hash(model),train_mean=train_mean,
            initial_sha256=setup['initial_sha256'],complete=True)
        checkpoint.chmod(0o444)
        verify_sources(manifest)
    return record


def select_lr_job(n,seed,setup,arms,folder,manifest_path,record):
    record.update(metadata(n,seed,0.,'fixed_nominal_validation_and_LR_selection'))
    with core.timed(record):
        verify_sources(read(manifest_path))
        require_grid(arms,{(n,seed,lr) for lr in LR_GRID},('Ntotal','seed','lr_net'))
        check(core.digest(setup['split_path'])==setup['split_sha256'],'Split changed')
        arrays=read(setup['split_path'])['arrays']
        x1=torch.tensor(arrays['val_x1'],dtype=torch.float32)
        x2=torch.tensor(arrays['val_x2'],dtype=torch.float32)
        record['evaluations']=[]
        for arm in sorted(arms,key=lambda r:r['lr_net']):
            check(core.digest(arm['checkpoint'])==arm['checkpoint_sha256'],'Nominal changed before validation')
            model=core.model_from_state(torch.load(arm['checkpoint'],map_location='cpu',weights_only=True))
            check(legacy_ccg.state_hash(model)==arm['state_sha256'],'Nominal state mismatch')
            evaluation=metadata(n,seed,0.,'nominal_validation')
            evaluation.update(lr_net=arm['lr_net'],checkpoint_sha256=arm['checkpoint_sha256'])
            record['evaluations'].append(evaluation)
            evaluate(model,x1,x2,evaluation)
        selection=record['selection']={}
        with core.timed(selection):
            chosen=min(record['evaluations'],key=lambda r:(r['mean'],r['lr_net']))
            arm=next(a for a in arms if a['lr_net']==chosen['lr_net'])
            record['winner']=dict(Ntotal=n,seed=seed,lr_net=arm['lr_net'],checkpoint=arm['checkpoint'],
                checkpoint_sha256=arm['checkpoint_sha256'],state_sha256=arm['state_sha256'],
                train_mean=arm['train_mean'],validation_mean=chosen['mean'],training_timing=arm['training'],
                data_sha256=setup['data_sha256'],training_data_path=setup['training_data_path'],
                training_data_sha256=setup['training_data_sha256'])
        record['complete']=True
    return record


def validation_job(n,seed,setup,jobs,folder,manifest_path,record):
    record.update(metadata(n,seed,None,'fixed_green_validation_and_radius_selection'))
    with core.timed(record):
        verify_sources(read(manifest_path))
        require_grid(jobs,{(n,seed,r) for r in RADII},('Ntotal','seed','radius'))
        check(core.digest(setup['split_path'])==setup['split_sha256'],'Final validation split changed')
        arrays=read(setup['split_path'])['arrays']
        x1=torch.tensor(arrays['val_x1'],dtype=torch.float32)
        x2=torch.tensor(arrays['val_x2'],dtype=torch.float32)
        record['evaluations']=[]
        for job in sorted(jobs,key=lambda r:r['radius']):
            check(core.digest(job['summary_path'])==job['summary_sha256'],'CCG summary changed')
            check(core.digest(job['selected_checkpoint'])==job['selected_checkpoint_sha256'],'Selected policy changed')
            model=core.model_from_state(torch.load(job['selected_checkpoint'],map_location='cpu',weights_only=True))
            check(legacy_ccg.state_hash(model)==job['selected_state_sha256'],'Selected state mismatch')
            evaluation=metadata(n,seed,job['radius'],'green_validation')
            evaluation.update(candidate_id=job['selected_candidate_id'],
                stopping_candidate_id=job['stopping_candidate_id'],terminal_status=job['terminal_status'],
                checkpoint=job['selected_checkpoint'],checkpoint_sha256=job['selected_checkpoint_sha256'],
                state_sha256=job['selected_state_sha256'])
            record['evaluations'].append(evaluation)
            evaluate(model,x1,x2,evaluation)
        with core.timed(record.setdefault('selection',{})):
            record['winner']=min(record['evaluations'],key=lambda r:(r['mean'],r['radius']))
        record['complete']=True
    return record


def worker(kind,params,folder,manifest_path):
    """One task's stdout/stderr and escaped failures are persisted, never hidden."""
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    t=time.perf_counter()
    record=metadata(params['n'],params['seed'],params.get('radius',0.),kind)
    record.update(started_at_utc=core.utc_now(),status='running',complete=False)
    try:
        with (folder/'worker.log').open('x',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
            print('START',core.utc_now(),kind,params['n'],params['seed'],params.get('lr'),params.get('radius'),flush=True)
            if kind=='ccg':
                record=run_configuration(params['n'],params['seed'],params['radius'],
                    Path(params['output_root']),params['winner'],manifest_path)
            else:
                functions={'initialization':setup_job,'nominal':nominal_job,'lr_validation':select_lr_job,
                           'radius_validation':validation_job}
                record=functions[kind](**params,folder=folder,manifest_path=manifest_path,record=record)
            print('END',core.utc_now(),record['status'],flush=True)
    except BaseException as error:
        record.update(status='failed',complete=False,error=error_info(error))
        if 'lr' in params:
            record['lr_net']=params['lr']
        if not isinstance(error,Exception):
            raise
    finally:
        record.update(worker_wall_time_s=time.perf_counter()-t,worker_ended_at_utc=core.utc_now(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_rss_scope='worker process lifetime; may include earlier tasks',pid=os.getpid(),
            configured_workers=read(manifest_path)['workers'])
        if 'wall_time_s' not in record:
            record.update(wall_time_s=record['worker_wall_time_s'],
                          timing_scope='outer task timer after function escaped; inner phase records may be partial')
        core.write_json(folder/'result.json',record)
    return record


def run_stage(name,tasks,workers,output,manifest_path,sweep):
    phase=sweep['phases'][name]=dict(jobs=[],max_workers=workers,submitted_count=len(tasks))
    with core.timed(phase):
        context=multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers,mp_context=context,initializer=configure_worker) as executor:
            futures={executor.submit(worker,kind,params,folder,str(manifest_path)):(params,time.perf_counter())
                     for kind,params,folder in tasks}
            for future in as_completed(futures):
                params,submitted=futures[future]
                try:
                    item=future.result()
                except Exception as error:
                    item=dict(Ntotal=params['n'],seed=params['seed'],radius=params.get('radius',0.),
                        lr_net=params.get('lr'),status='failed',complete=False,error=error_info(error),
                        wall_time_s=None,worker_timing='unavailable: escaped worker failure',
                        parent_observed_queue_inclusive_s=time.perf_counter()-submitted)
                phase['jobs'].append(item)
                print(f'{name} {len(phase["jobs"])}/{len(tasks)} N{item["Ntotal"]} seed{item["seed"]} '
                      f'rho={item.get("radius")} LR={item.get("lr_net")} {item["status"]}',flush=True)
                core.write_json(output/'summary.json',sweep)
    return phase['jobs']


def aggregate(sweep):
    rows=[]
    for phase_name,phase in sweep['phases'].items():
        for n in SIZES:
            jobs=[j for j in phase.get('jobs',[]) if j['Ntotal']==n]
            for status in sorted({j['status'] for j in jobs}):
                cohort=[j for j in jobs if j['status']==status]
                groups=sorted({j['radius'] for j in cohort}) if phase_name=='ccg' else (
                       sorted({j.get('lr_net') for j in cohort},key=str) if phase_name=='nominal' else [None])
                for group in groups:
                    selected=[j for j in cohort if (phase_name=='ccg' and j['radius']==group)
                        or (phase_name=='nominal' and j.get('lr_net')==group)
                        or phase_name not in ('ccg','nominal')]
                    values=[j['wall_time_s'] for j in selected if j.get('wall_time_s') is not None]
                    rows.append(dict(phase=phase_name,Ntotal=n,Ntrain=4*n//5,status=status,group=group,
                        jobs=len(selected),unmeasured=len(selected)-len(values),wall_time_s=core.stats(values)))
    return dict(unit='seconds',groups=rows,
        overall_sweep_wall_s=sweep['wall_time_s'],scope='Per-model wall times, not a sum interpreted as parallel elapsed',
        not_run=dict(SRO_FA='not run',CV='single paired split, not CV',test='not accessed'),
        raw_nominal_AP_MP_times='Nested nominal training records and each CCG summary retain all phases/iterations')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    output=ROOT/'results'/core.utc_now().replace(':','').replace('-','')
    output.mkdir(parents=True,exist_ok=False)
    sweep=dict(experiment=PROTOCOL['experiment'],status='running',started_at_utc=core.utc_now(),
        protocol=PROTOCOL,command=shlex.join(sys.orig_argv),cwd=str(Path.cwd()),output_directory=str(output),
        timing_unit='seconds',timing_clock='time.perf_counter',phases={},selections=[],
        final_selection_published=False,test_evaluated=False,thread_environment=THREAD_ENV)
    try:
        configure_worker()
        preflight = read(args.preflight)
        check(preflight['status'] == 'completed', 'Require completed real preflight')
        check({r['Ntotal'] for r in preflight['sizes']} == set(SIZES), 'Preflight omitted a size')
        required_sources = ('green_common.py', 'green_training.py', 'green_ccg.py', 'test_green.py',
                            'run_campaign.py', 'test_campaign.py', 'preflight.py', 'launch.sh', 'README.md')
        check(set(preflight['sources_sha256']) == {str(ROOT / name) for name in required_sources},
              'Preflight source coverage omitted controller/launcher/tests')
        for path, h in preflight['sources_sha256'].items():
            check(core.digest(path) == h, 'Computational source changed since preflight')
        snapshot = resources()
        workers=choose_workers(snapshot,preflight)
        sources=dict(FROZEN)
        for path in ROOT.iterdir():
            if path.is_file() and path.suffix in ('.py','.sh','.md'):
                sources[str(path)]=core.digest(path)
        sources[str(args.preflight.resolve())]=core.digest(args.preflight)
        manifest=dict(protocol=PROTOCOL,workers=workers,source_sha256=sources,
            environment=snapshot,created_at_utc=core.utc_now(),preflight_path=str(args.preflight.resolve()),
            versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scipy','gurobipy','moreau-cpu')})
        verify_sources(manifest)
        manifest_path=output/'provenance.json'
        core.write_json(manifest_path,manifest)
        manifest_path.chmod(0o444)
        sweep.update(workers=workers,environment=snapshot,provenance_sha256=core.digest(manifest_path))
        core.write_json(ROOT/'launch.json',dict(output_directory=str(output),command=sweep['command'],
            workers=workers,started_at_utc=sweep['started_at_utc'],summary_path=str(output/'summary.json')))
        core.write_json(output/'summary.json',sweep)
        print(f'OUTPUT={output}\nWORKERS={workers}',flush=True)
        setup=run_stage('initialization',[
            ('initialization',dict(n=n,seed=s),output/f'N{n}/initialization/seed_{s}')
            for s in SEEDS for n in SIZES],workers['initialization'],output,manifest_path,sweep)
        require_grid(setup,{(n,s) for n in SIZES for s in SEEDS},('Ntotal','seed'))
        setup_map={(j['Ntotal'],j['seed']):j for j in setup}
        nominal=run_stage('nominal',[
            ('nominal',dict(n=n,seed=s,lr=lr,setup=setup_map[n,s]),output/f'N{n}/nominal/seed_{s}/lr_{lr:g}')
            for lr in LR_GRID for s in SEEDS for n in SIZES],workers['nominal'],output,manifest_path,sweep)
        require_grid(nominal,{(n,s,lr) for n in SIZES for s in SEEDS for lr in LR_GRID},('Ntotal','seed','lr_net'))
        fixed=output/'nominal_checkpoints_before_validation.json'
        core.write_json(fixed,dict(frozen_at_utc=core.utc_now(),checkpoints=[{k:r[k] for k in
            ('Ntotal','seed','lr_net','checkpoint','checkpoint_sha256','state_sha256')} for r in nominal]))
        frozen_hashes={str(fixed):core.digest(fixed)}
        lrjobs=run_stage('lr_validation',[
            ('lr_validation',dict(n=n,seed=s,setup=setup_map[n,s],arms=[r for r in nominal if r['Ntotal']==n and r['seed']==s]),
                output/f'N{n}/lr_selection/seed_{s}') for s in SEEDS for n in SIZES],
            workers['validation'],output,manifest_path,sweep)
        require_grid(lrjobs,{(n,s) for n in SIZES for s in SEEDS},('Ntotal','seed'))
        winners={(j['Ntotal'],j['seed']):j['winner'] for j in lrjobs}
        jobs=run_stage('ccg',[
            ('ccg',dict(n=n,seed=s,radius=r,winner=winners[n,s],output_root=str(output/f'N{n}/ccg')),
                output/f'N{n}/job_logs/seed_{s}/radius_{r:g}')
            for r in RADII for s in SEEDS for n in SIZES],workers['ccg'],output,manifest_path,sweep)
        require_grid(jobs,{(n,s,r) for n in SIZES for s in SEEDS for r in RADII},('Ntotal','seed','radius'))
        fixed=output/'green_checkpoints_before_validation.json'
        core.write_json(fixed,dict(frozen_at_utc=core.utc_now(),policies=[{k:r[k] for k in
            ('Ntotal','seed','radius','selected_candidate_id','stopping_candidate_id','selected_checkpoint',
             'selected_checkpoint_sha256','selected_state_sha256')} for r in jobs]))
        frozen_hashes[str(fixed)]=core.digest(fixed)
        valid=run_stage('radius_validation',[
            ('radius_validation',dict(n=n,seed=s,setup=setup_map[n,s],jobs=[r for r in jobs if r['Ntotal']==n and r['seed']==s]),
                output/f'N{n}/radius_selection/seed_{s}') for s in SEEDS for n in SIZES],
            workers['validation'],output,manifest_path,sweep)
        require_grid(valid,{(n,s) for n in SIZES for s in SEEDS},('Ntotal','seed'))
        verify_sources(manifest)
        check(core.digest(manifest_path)==sweep['provenance_sha256'],'Manifest changed')
        for path,h in frozen_hashes.items():
            check(core.digest(path)==h,'Selection freeze changed')
        for r in jobs:
            check(core.digest(r['selected_checkpoint'])==r['selected_checkpoint_sha256'],'Final policy changed')
        sweep.update(status='completed',inputs_unchanged=True,selections=[r['winner'] for r in valid],
                     final_selection_published=True)
    except BaseException as error:
        sweep.update(status='failed',error=error_info(error),selections=[],final_selection_published=False)
        print(traceback.format_exc(),flush=True)
        if not isinstance(error,Exception):
            raise
    finally:
        sweep.update(ended_at_utc=core.utc_now(),wall_time_s=time.perf_counter()-started)
        sweep['wall_time_human']=f'{sweep["wall_time_s"]:.3f} s / {sweep["wall_time_s"]/3600:.3f} h'
        core.write_json(output/'summary.json',sweep)
        core.write_json(output/'runtime_aggregates.json',aggregate(sweep))
        print(sweep['status'],sweep['wall_time_human'],flush=True)
    return 0 if sweep['status']=='completed' else 1


if __name__=='__main__':
    sys.exit(main())