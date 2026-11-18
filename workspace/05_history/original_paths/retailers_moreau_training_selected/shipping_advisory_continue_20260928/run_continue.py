"""Exactly five hash-bound AP-boundary continuations; no held-out evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
import importlib.metadata
import multiprocessing
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from continue_common import *
import continuation


def worker(task,output,manifest):
    n,seed,radius=task['key']
    folder=Path(output)/f'N{n}/worker_seed{seed}_radius{radius:g}'
    folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    record=metadata(n,seed,radius,'advisory_boundary_continuation')
    record.update(status='running',started_at_utc=core.utc_now())
    try:
        with (folder/'worker.log').open('x',buffering=1) as log,redirect_stdout(log),redirect_stderr(log):
            result=continuation.run_configuration(n,seed,radius,Path(output)/f'N{n}/ccg',task['nominal'],manifest)
            record.update(result)
    except Exception as error:
        record.update(status='failed',complete=False,converged=False,error=dict(
            type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
    finally:
        record.update(worker_wall_time_s=time.perf_counter()-started,worker_ended_at_utc=core.utc_now(),
            worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        core.write_json(folder/'result.json',record)
    return record


def _probe_body(task,evidence):
    """Exactly the already-diagnosed AP of each failed config, unchanged solve."""
    old=read(task['nominal']['resume_summary_path'])
    candidate=old['iterations'][-1]
    sample=candidate['failed_sample_ids'][0]
    row=candidate['samples'][sample]
    n,seed,radius=task['key']
    evidence.update(sample_id=sample,candidate_id=candidate['candidate_id'])
    with core.timed(evidence):
        model=continuation._verify_checkpoint(candidate['checkpoint'],candidate['checkpoint_sha256'],candidate['state_sha256'])
        model.eval()
        ap=evidence['ap']={}
        result=continuation.adversary.solve_ap(model,np.array(row['center1']),np.array(row['center2']),radius,ap)
        evidence['raw_historical_incumbent_reproduced']=ap['incumbent']==row['incumbent']
        pools=[np.asarray(p,dtype=np.float64) for p in read(Path(candidate['checkpoint']).parent/'pool.json')['pools']]
        ap['sample_id']=sample
        continuation.validate_ap(result,ap,sample,np.array(row['center1']+row['center2']),
            pools[sample],radius,row['pool_cost'] if 'pool_cost' in row else candidate['pool_costs'][sample])
        check(evidence['raw_historical_incumbent_reproduced'],'Native probe differs; inspect before continuation')


def run_native_probe(task,output,summary):
    n,seed,radius=task['key']
    evidence=metadata(n,seed,radius,'native_advisory_AP_preflight')
    summary['native_probes'].append(evidence)
    try:
        _probe_body(task,evidence)
    except BaseException as error:
        evidence.update(status='failed',error=dict(type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc()))
        raise
    finally:
        evidence['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        core.write_json(output/f'probe_N{n}_seed{seed}_radius{radius:g}.json',evidence)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output.resolve();output.mkdir(parents=True,exist_ok=False)
    summary=dict(experiment=ROOT.name,protocol=PROTOCOL,results=[],native_probes=[],test_evaluated=False,
        validation_evaluated=False,training_restarted_from_nominal=False,full_grid_rerun=False,
        concurrency=5,timing_clock='time.perf_counter',started_at_utc=core.utc_now(),status='running')
    start=time.perf_counter()
    def persist():
        summary['wall_time_s']=time.perf_counter()-start
        core.write_json(output/'summary.json',summary)
    try:
        core.configure_worker()
        tasks=[];bindings={}
        def bind(p,expected=None):
            p=Path(p).resolve();h=digest(p)
            check(expected is None or expected==h,f'Changed artifact: {p}')
            bindings[str(p)]=h
            return h
        campaign_path=PARENT/'production_v1/summary.json'
        bind(campaign_path);bind(SOURCE_MANIFEST)
        campaign=read(campaign_path)
        failed={ (r['Ntotal'],r['seed'],r['radius']):r for r in campaign['phases']['ccg']['jobs'] if r['status']=='failed'}
        check(set(failed)==set(continuation.EXPECTED_BOUNDARIES),'Unexpected failed configuration set')
        with core.timed(summary.setdefault('input_binding',{})):
            for key,old in sorted(failed.items()):
                bind(old['summary_path'],old['summary_sha256'])
                n,seed,radius=key
                nominal_path=PARENT/f'production_v1/N{n}/nominal_selection/seed{seed}/result.json'
                bind(nominal_path)
                nominal=dict(read(nominal_path)['nominal_record'])
                nominal.update(resume_summary_path=old['summary_path'],resume_summary_sha256=old['summary_sha256'])
                bind(nominal['checkpoint'],nominal['checkpoint_sha256'])
                bind(nominal['training_data_path'],nominal['training_data_sha256'])
                # Bind the complete consumed historical record, not just the restart checkpoint.
                for p in Path(old['summary_path']).parent.rglob('*'):
                    if p.is_file():bind(p)
                tasks.append(dict(key=key,nominal=nominal))
        manifest=output/'provenance.json'
        resource_snapshot=parent.shipping_resources.resources()
        measured_peak=read(PARENT/'preflight_v1/summary.json')['peak_rss_bytes']
        bind(PARENT/'preflight_v1/summary.json')
        resource_policy=parent.shipping_resources.choose_workers(resource_snapshot,measured_peak)
        check(resource_policy['nominal']>=5,'Insufficient headroom for five fixed workers; no degraded launch')
        summary.update(environment=resource_snapshot,resource_policy=resource_policy,
            versions={name:importlib.metadata.version(name) for name in ('torch','numpy','gurobipy','moreau-cpu')})
        provenance=dict(source_sha256=sources(),input_sha256=bindings,protocol=PROTOCOL,workers={'ccg':5})
        verify_sources(provenance);core.write_json(manifest,provenance);manifest.chmod(0o444)
        summary.update(provenance_path=str(manifest),provenance_sha256=digest(manifest))
        with core.timed(summary.setdefault('tests',{})):
            with (output/'tests.log').open('x') as log:
                test=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_resume','test_advisory'],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                check(test.returncode==0,'Continuation regression tests failed')
        verify_sources(provenance)
        with core.timed(summary.setdefault('native_preflight',{})):
            for task in tasks:
                try:
                    run_native_probe(task,output,summary)
                finally:persist()
        verify_sources(provenance)
        with core.timed(summary.setdefault('continuation_stage',{})):
            with ProcessPoolExecutor(max_workers=5,mp_context=multiprocessing.get_context('spawn'),
                initializer=core.configure_worker) as pool:
                futures={pool.submit(worker,t,str(output),str(manifest)):t for t in tasks}
                for future in as_completed(futures):
                    row=future.result();summary['results'].append(row);persist()
                    print(row['Ntotal'],row['seed'],row['radius'],row['status'],row.get('terminal_status'),flush=True)
        verify_sources(provenance)
        check(len(summary['results'])==5,'Incomplete continuation results')
        summary.update(status='completed',all_five_converged=all(r.get('converged',False) for r in summary['results']),
            radius_selection_run=False,radius_selection_note='No automatic selection on a mixed or incomplete grid')
        summary['runtime_groups']=[dict(Ntotal=r['Ntotal'],seed=r['seed'],radius=r['radius'],status=r['status'],
            current_wall_time_s=r.get('wall_time_s'),current_wall_time_human=f"{r.get('wall_time_s',0.):.3f} s",
            completed_only_stats=core.stats([r['wall_time_s']]) if r.get('converged') else None,
            prior_failed_wall_time_s=r.get('cached_prior_job_timing',{}).get('wall_time_s')) for r in summary['results']]
    except BaseException as error:
        summary.update(status='failed',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise
    finally:
        summary['ended_at_utc']=core.utc_now();persist()


if __name__=='__main__':main()