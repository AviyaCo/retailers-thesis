"""30 fresh training-only initializations, barrier, 90 nominal arms, STOP."""
from source_adapter import ROOT, sources, verify, digest, check, THREAD_ENV
import shipping_core as core
from shipping_common import SIZES, SEEDS, LR_GRID, PROTOCOL, metadata, split_data, solve_saa
from shipping_model import make_model, model_from_state
from shipping_training import train_model
from shipping_resources import resources, choose_workers
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shlex
import sys
import time
import traceback

np, torch = core.np, core.torch


def read(path):
    return json.loads(Path(path).read_text())


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype='<f8').tobytes())
    return h.hexdigest()


def setup_job(n, seed, folder, record):
    with core.timed(record):
        data_record = record['data_generation'] = metadata(n,seed,'paired generation')
        with core.timed(data_record):
            data = split_data(n,seed)
            ids = data['train_indices']
            x1, x2 = data['xi1'][ids], data['xi2'][ids]
            train = folder/'training_data.json'
            split = folder/'split.json'
            paired_hash = array_hash(data['xi1'],data['xi2'])
            core.write_json(train, dict(Ntotal=n,seed=seed,split_seed=100000+seed,
                x1=x1,x2=x2,original_sample_ids=ids,paired_data_sha256=paired_hash,
                training_array_sha256=array_hash(x1,x2)))
            core.write_json(split,dict(Ntotal=n,seed=seed,split_seed=100000+seed,
                train_ids=ids,holdout_ids=data['val_indices'],paired_data_sha256=paired_hash,
                train_array_sha256=array_hash(x1,x2),
                holdout_array_sha256=array_hash(data['xi1'][data['val_indices']],data['xi2'][data['val_indices']]),
                heldout_evaluated=False))
            train.chmod(0o444)
            split.chmod(0o444)
            record.update(training_data_path=str(train),training_data_sha256=digest(train),
                split_path=str(split),split_sha256=digest(split),paired_data_sha256=paired_hash)
        saa = record['saa_initialization'] = metadata(n,seed,'shipping SAA initialization')
        q10, q1 = solve_saa(x1,x2,saa)
        initial = record['model_initialization'] = metadata(n,seed,'SAA-to-native weights')
        with core.timed(initial):
            model = make_model(q10,q1)
            path = folder/'initial.pth'
            with path.open('xb') as stream:
                torch.save(core.weights(model),stream)
            path.chmod(0o444)
            record.update(initial_path=str(path),initial_sha256=digest(path),complete=True)
    return record


def nominal_job(n, seed, lr, setup, folder, record, epochs=3000):
    check(lr in LR_GRID and epochs in (2,3000), 'Unapproved nominal configuration')
    record.update(lr_net=lr,disposable_preflight=epochs==2)
    with core.timed(record):
        inputs = record['input_loading'] = metadata(n,seed,'immutable training inputs')
        with core.timed(inputs):
            # Deliberately no access to split/holdout artifacts in this worker.
            for key in ('initial','training_data'):
                check(digest(setup[key+'_path']) == setup[key+'_sha256'], 'Changed nominal input')
            data = read(setup['training_data_path'])
            x1, x2 = np.asarray(data['x1']), np.asarray(data['x2'])
            check(data['Ntotal']==n and data['seed']==seed and data['split_seed']==100000+seed,
                  'Mismatched training identities')
            check(x1.shape == x2.shape == (4*n//5,3) and
                  len(set(data['original_sample_ids'])) == len(x1), 'Invalid training pairs')
            check(array_hash(x1,x2)==data['training_array_sha256'], 'Training arrays changed')
            x1, x2 = torch.tensor(x1,dtype=torch.float32), torch.tensor(x2,dtype=torch.float32)
            state = torch.load(setup['initial_path'],map_location='cpu',weights_only=True)
            model = model_from_state(state)
            check(all(torch.equal(v,state[k]) for k,v in core.weights(model).items()), 'Changed initial state')
        training = record['training'] = metadata(n,seed,'smooth nominal training')
        train_model(model,lambda m:m.per_sample_cost(x1,x2).mean(),lr,folder/'training',
                    training,epochs=epochs,patience=300)
        selected = folder/'training/selected.pth'
        saved = torch.load(selected,map_location='cpu',weights_only=True)
        check(all(torch.equal(v,saved[k]) for k,v in core.weights(model).items()), 'Restoration mismatch')
        selected.chmod(0o444)
        record.update(checkpoint=str(selected),checkpoint_sha256=digest(selected),
            initial_sha256=setup['initial_sha256'],complete=True)
        if epochs == 2:
            replay = record['training_only_replay'] = metadata(n,seed,'disposable smooth/hard same-q replay')
            with core.timed(replay):
                smooth, hard, q = model.costs_and_shipments(x1,x2)
                check(smooth.shape==hard.shape==(len(x1),) and q.shape==(len(x1),3), 'Replay dimensions')
                check(bool(torch.isfinite(smooth).all() and torch.isfinite(hard).all()), 'Nonfinite replay')
                mean = float(smooth.mean().detach())
                check(abs(mean-training['selected_training_loss'])<=1e-6, 'Smooth checkpoint replay mismatch')
                check(mean<=training['first_train']+1e-6, 'Smooth preupdate best worsened')
                model.zero_grad(set_to_none=True)
                smooth.mean().backward()
                check(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                          for p in model.parameters()), 'Nonfinite/missing replay gradient')
                replay.update(smooth_mean=mean,hard_mean=float(hard.mean().detach()),
                    same_returned_shipments=True,hard_improvement_guaranteed=False,
                    real_arithmetic_certificate=False,finite_gradients=True,
                    projection=model.proj_layer.drain_summary(),sample_count=len(x1))
        verify(read(record['manifest_path'])['source_sha256'])
    return record


def worker(kind, params, folder, manifest_path, epochs=3000):
    folder = Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    record = metadata(params['n'],params['seed'],kind)
    manifest = read(manifest_path)
    record.update(manifest_path=str(manifest_path),configured_workers=manifest['workers'],
                  complete=False,lr_net=params.get('lr'),pid=os.getpid())
    start = time.perf_counter()
    record.update(started_at_utc=core.utc_now(),status='running')
    try:
        with (folder/'worker.log').open('x',buffering=1) as log, redirect_stdout(log),redirect_stderr(log):
            verify(manifest['source_sha256'])
            check(torch.get_num_threads()==torch.get_num_interop_threads()==1, 'Worker threading changed')
            if kind == 'initialization':
                setup_job(**params,folder=folder,record=record)
            elif kind == 'nominal':
                nominal_job(**params,folder=folder,record=record,epochs=epochs)
            else:
                raise ValueError('Only initialization/nominal tasks are authorized')
            verify(manifest['source_sha256'])
    except BaseException as error:
        record.update(status='failed',complete=False,error=dict(type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc()))
        if not isinstance(error,Exception):
            raise
    finally:
        record.update(worker_wall_time_s=time.perf_counter()-start,worker_ended_at_utc=core.utc_now(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_rss_scope='process lifetime',actual_torch_threads=torch.get_num_threads(),
            actual_interop_threads=torch.get_num_interop_threads())
        if 'wall_time_s' not in record:
            record.update(wall_time_s=record['worker_wall_time_s'],ended_at_utc=record['worker_ended_at_utc'])
        core.write_json(folder/'result.json',record)
    return record


def require_grid(rows, expected, keys):
    actual = [tuple(r[k] for k in keys) for r in rows]
    check(len(actual)==len(expected) and set(actual)==expected, 'Incomplete/duplicate grid')
    check(all(r['status']=='completed' and r['complete'] for r in rows), 'Failed phase: stop')


def run_stage(name, tasks, workers, output, manifest, sweep):
    phase = sweep['phases'][name] = dict(jobs=[],max_workers=workers,submitted_count=len(tasks))
    with core.timed(phase):
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                                 initializer=core.configure_worker) as pool:
            futures = {pool.submit(worker,name,p,str(f),str(manifest)):p for p,f in tasks}
            failed = False
            for future in as_completed(futures):
                if future.cancelled():
                    phase.setdefault('cancelled',[]).append({k:v for k,v in futures[future].items() if k!='setup'})
                    continue
                try:
                    row = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
                phase['jobs'].append(row)
                if row['status']!='completed':
                    failed = True
                    for pending in futures:
                        pending.cancel()
                print(name,row['Ntotal'],row['seed'],row.get('lr_net'),row['status'],flush=True)
                core.write_json(output/'summary.json',sweep)
            check(not failed, 'Worker failure; pending jobs cancelled, no next phase')
    return phase['jobs']


def aggregate(sweep):
    groups = []
    incomplete = []
    for name, phase in sweep['phases'].items():
        for n in SIZES:
            for lr in LR_GRID if name=='nominal' else (None,):
                rows = [r for r in phase['jobs'] if r['Ntotal']==n and r.get('lr_net')==lr]
                good = [r for r in rows if r['status']=='completed' and r['complete']]
                groups.append(dict(method=name,Ntotal=n,lr_net=lr,
                    seeds=[r['seed'] for r in good],wall_time_s=core.stats([r['wall_time_s'] for r in good]),
                    training_time_s=core.stats([r['training']['training_time_s'] for r in good])
                        if name=='nominal' else None))
                incomplete.extend(r for r in rows if r not in good)
    return dict(groups=groups,incomplete=incomplete,not_run=PROTOCOL['not_run'],
        overall_sweep_wall_time_s=sweep['wall_time_s'],workers=sweep.get('workers'),
        timing_scope='current invocation; per-model concurrent times are not summed as sweep elapsed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--preflight-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    from validate_preflight import validate
    preflight = validate(args.preflight,args.preflight_sha256)
    snapshot = resources()
    workers = choose_workers(snapshot,max(c['peak_rss_bytes'] for c in preflight['children']))
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    sweep = dict(protocol=PROTOCOL,phases={},workers=workers,environment=snapshot,
                 not_run=PROTOCOL['not_run'],command=shlex.join(sys.orig_argv),cwd=str(Path.cwd()))
    try:
        with core.timed(sweep):
            frozen = sources()
            verify(frozen)
            manifest = output/'provenance.json'
            core.write_json(manifest,dict(source_sha256=frozen,workers=workers,protocol=PROTOCOL,
                preflight_path=str(args.preflight.resolve()),preflight_sha256=args.preflight_sha256,
                environment=snapshot,thread_environment=THREAD_ENV,
                versions={k:importlib.metadata.version(k) for k in ('torch','numpy','scipy','gurobipy','moreau-cpu')}))
            manifest.chmod(0o444)
            setup = run_stage('initialization',[(dict(n=n,seed=s),output/f'N{n}/initialization/seed{s}')
                for s in SEEDS for n in SIZES],workers['initialization'],output,manifest,sweep)
            require_grid(setup,{(n,s) for n in SIZES for s in SEEDS},('Ntotal','seed'))
            by_id = {(r['Ntotal'],r['seed']):r for r in setup}
            nominal = run_stage('nominal',[(dict(n=n,seed=s,lr=lr,setup=by_id[n,s]),
                output/f'N{n}/nominal/seed{s}/lr{lr:g}') for lr in LR_GRID for s in SEEDS for n in SIZES],
                workers['nominal'],output,manifest,sweep)
            require_grid(nominal,{(n,s,lr) for n in SIZES for s in SEEDS for lr in LR_GRID},
                         ('Ntotal','seed','lr_net'))
            verify(frozen)
            sweep['terminal_status'] = 'NOMINAL_GRID_COMPLETE_STOP_NO_SELECTION'
    except BaseException:
        sweep['traceback'] = traceback.format_exc()
        raise
    finally:
        core.write_json(output/'summary.json',sweep)
        core.write_json(output/'runtime_aggregates.json',aggregate(sweep))
        print(sweep['status'],f'{sweep["wall_time_s"]:.3f}s / {sweep["wall_time_s"]/3600:.3f}h',flush=True)


if __name__ == '__main__':
    main()