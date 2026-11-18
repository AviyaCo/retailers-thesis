"""Three fixed AP diagnostic replays; frozen numerical settings, no campaign repair."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import math
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent/'shipping_live_green_20260926'
sys.path.insert(0,str(CAMPAIGN))
import shipping_live_common as c

CASES=((200,5,1.5,1,6),(50,7,1.5,9,33),(800,7,.2,1,271))


def error_record(error):
    return dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())


def inputs(spec):
    n,seed,radius,candidate,sample=spec
    folder=CAMPAIGN/f'production_v1/N{n}/ccg/seed_{seed}/radius_{radius:g}/candidate_{candidate:02d}'
    assessment=c.read(folder/'assessment.json')
    saved=c.read(folder/f'sample_{sample:02d}.json')
    c.check(c.digest(assessment['checkpoint'])==assessment['checkpoint_sha256'],'Changed checkpoint')
    c.check(c.digest(folder/'pool.json')==assessment['pool_file_sha256'],'Changed pool file')
    c.check(saved['candidate_sha256']==assessment['state_sha256'],'Wrong candidate')
    pools=[c.np.asarray(p,dtype=c.np.float64) for p in c.read(folder/'pool.json')['pools']]
    c.check(c.legacy_ccg.pool_hash(pools)==assessment['pool_sha256']==saved['pool_sha256'],'Changed pool')
    return folder,assessment,saved,pools


@contextmanager
def observe_projection(model,record):
    """Read-only forward hooks; never replace outputs or change tensor values."""
    calls=record.setdefault('native_calls',[])
    def before(layer,args):
        calls.append(dict(target=args[0].detach().cpu().tolist(),
            capacity=args[1].detach().cpu().tolist(),target_dtype=str(args[0].dtype)))
    def after(layer,args,result):
        calls[-1]['q']=result[0].detach().cpu().tolist()
    pre=model.proj_layer.register_forward_pre_hook(before)
    post=model.proj_layer.register_forward_hook(after)
    try:
        yield
    finally:
        pre.remove();post.remove()


def capture_solver(solver,folder,record):
    """Read original-model rows and returned values, before AP post-validation."""
    with c.core.timed(record.setdefault('capture',{})):
        for name in ('Status','SolCount','ObjVal','ObjBound','MIPGap','Runtime',
                     'IterCount','NodeCount','Fingerprint','NumVars','NumConstrs','NumGenConstrs',
                     'MaxVio','ConstrVio','ConstrVioIndex','BoundVio','BoundVioIndex','IntVio','IntVioIndex'):
            try:
                record[name]=solver.getAttr(name)
            except (c.adversary.gp.GurobiError,AttributeError) as error:
                record.setdefault('unavailable_attributes',{})[name]=str(error)
        record['effective_parameters']={name:solver.getParamInfo(name)[2] for name in
            (*c.adversary.PARAMETERS,'Seed','Presolve','ScaleFlag','IntegralityFocus','LogFile','LogToConsole')}
        if solver.SolCount:
            variables=solver.getVars()
            record['raw_variables']={v.VarName:v.X for v in variables}
            rows=[]
            for constraint in solver.getConstrs():
                expression=solver.getRow(constraint)
                terms=[dict(name=expression.getVar(i).VarName,coefficient=expression.getCoeff(i),
                    value=expression.getVar(i).X) for i in range(expression.size())]
                lhs=math.fsum(t['coefficient']*t['value'] for t in terms)
                rhs,sense=constraint.RHS,constraint.Sense
                violation=max(0.,lhs-rhs) if sense=='<' else max(0.,rhs-lhs) if sense=='>' else abs(lhs-rhs)
                rows.append(dict(name=constraint.ConstrName,sense=sense,rhs=rhs,lhs=lhs,
                    violation=violation,solver_slack=constraint.Slack,terms=terms))
            record['shipping_rows']=[r for r in rows if r['name'].startswith('shipping_')]
            record['violated_linear_rows']=[r for r in rows if r['violation']>solver.Params.FeasibilityTol]
            record['maximum_recomputed_linear_violation']=max(r['violation'] for r in rows)
            solver.write(str(folder/'solution.sol'))
        c.core.write_json(folder/'solver_capture.json',record)


@contextmanager
def solver_logging(folder,record):
    """Only logging changes: baseline comparison detects any changed result."""
    original=c.adversary.gp
    class CapturingModel(original.Model):
        def optimize(self,*args,**kwargs):
            self.update()
            self.write(str(folder/'original.lp'))
            self.write(str(folder/'original.mps'))
            self.setParam('LogToConsole',0)
            self.setParam('LogFile',str(folder/'solver.log'))
            self.setParam('OutputFlag',1)
            started=time.perf_counter()
            record['started_at_utc']=c.core.utc_now()
            try:
                return super().optimize(*args,**kwargs)
            finally:
                record.update(optimize_wall_time_s=time.perf_counter()-started,
                    ended_at_utc=c.core.utc_now())
                capture_solver(self,folder,record)
    class Proxy:
        Model=CapturingModel
        def __getattr__(self,name):
            return getattr(original,name)
    c.adversary.gp=Proxy()
    try:
        yield
    finally:
        c.adversary.gp=original


def solve_once(spec,mode,output):
    n,seed,radius,candidate,sample=spec
    folder,assessment,saved,pools=inputs(spec)
    destination=output/f'N{n}_seed{seed}_radius{radius:g}_candidate{candidate}_sample{sample}'/mode
    destination.mkdir(parents=True,exist_ok=False)
    record=dict(c.metadata(n,seed,radius,'fixed AP replay'),candidate_id=candidate,sample_id=sample,
        mode=mode,workers=1,source_sample=str(folder/f'sample_{sample:02d}.json'),
        saved_controller_status=saved['controller_status'],checkpoint=assessment['checkpoint'],
        checkpoint_sha256=assessment['checkpoint_sha256'],full_CCG_launched=False,
        training_run=False,test_evaluated=False,validation_evaluated=False)
    try:
        with c.core.timed(record):
            model=c.shipping_model.model_from_state(c.torch.load(assessment['checkpoint'],
                map_location='cpu',weights_only=True))
            model.eval()
            state=c.legacy_ccg.state_hash(model)
            c.check(state==assessment['state_sha256'],'Changed learned state')
            # Compare exactly the saved pool for the duplicate-point case only.
            if n==800:
                pool_record=record['pool_evaluation']={}
                points=c.np.concatenate(pools)
                with c.core.timed(pool_record),c.torch.no_grad(),observe_projection(model,pool_record):
                    smooth,hard,q=model.costs_and_shipments(
                        c.torch.tensor(points[:,:3],dtype=c.torch.float32),
                        c.torch.tensor(points[:,3:],dtype=c.torch.float32))
                    sizes=[len(p) for p in pools]
                    maxima=c.torch.stack([v.max() for v in c.torch.split(hard,sizes)])
                    start=sum(sizes[:sample]);stop=start+sizes[sample]
                    witness=c.np.array(saved['incumbent']['worst1']+saved['incumbent']['worst2'])
                    indices=c.np.flatnonzero(c.np.all(pools[sample]==witness,axis=1))
                    pool_record.update(batch_size=len(points),pool_cost=float(maxima[sample]),
                        all_pool_costs_reproduced=maxima.tolist()==assessment['pool_costs'],
                        witness_indices=indices.tolist(),ball_hard_costs=hard[start:stop].tolist(),
                        witness_batch_indices=(indices+start).tolist(),
                        telemetry=model.proj_layer.drain_summary())
            ap=record['ap']={}
            solver=record['solver_capture']={}
            # AP failures are diagnostic outcomes, never replaced or relabelled successful.
            try:
                if mode=='logged':
                    with solver_logging(destination,solver),observe_projection(model,record):
                        result=c.adversary.solve_ap(model,c.np.array(saved['center1']),
                            c.np.array(saved['center2']),radius,ap)
                else:
                    result=c.adversary.solve_ap(model,c.np.array(saved['center1']),
                        c.np.array(saved['center2']),radius,ap)
                record['ap_returned']=True
            except Exception as error:
                record.update(ap_returned=False,ap_error=error_record(error))
            if mode=='logged':
                c.check(solver.get('capture',{}).get('status')=='completed'
                    and len(solver.get('shipping_rows',[]))==6,'Missing/incomplete solver capture')
                c.check(all((destination/name).is_file() for name in
                    ('original.lp','original.mps','solution.sol','solver.log','solver_capture.json')),
                    'Missing solver diagnostic artifact')
            record['comparison']={key:ap.get(key)==saved.get(key) for key in
                ('solver_status','bound','incumbent','status')}
            for key in ('solver_q','binary_values','link_residual','native_q','native_hard_cost'):
                record['comparison'][key]=ap.get('activation_replay',{}).get(key)==saved.get('activation_replay',{}).get(key)
            if record['ap_returned'] and n==800:
                controller=record['controller_recheck']=dict(ap,sample_id=sample)
                try:
                    c.legacy_ccg.validate_ap(result,controller,sample,
                        c.np.array(saved['center1']+saved['center2']),pools[sample],radius,
                        record['pool_evaluation']['pool_cost'])
                    record['controller_recheck_status']='completed'
                except Exception as error:
                    record.update(controller_recheck_status='failed',controller_recheck_error=error_record(error))
            if mode=='logged' and n==800 and record.get('native_calls'):
                same_point=all(ap.get('incumbent',{}).get(k)==saved['incumbent'][k]
                    for k in ('worst1','worst2'))
                record['same_point_identity_verified']=same_point
                c.check(same_point and record['pool_evaluation']['witness_batch_indices'],
                    'Rerun AP point differs or is absent from pool; cannot attribute same-point discrepancy')
                pool_call=record['pool_evaluation']['native_calls'][0]
                singleton=record['native_calls'][0]
                record['same_point_comparison']=[dict(pool_index=int(i),
                    batch_target=pool_call['target'][i],singleton_target=singleton['target'][0],
                    batch_q=pool_call['q'][i],singleton_q=singleton['q'][0],
                    batch_capacity=pool_call['capacity'][i],singleton_capacity=singleton['capacity'][0])
                    for i in record['pool_evaluation']['witness_batch_indices']]
            c.check(c.legacy_ccg.state_hash(model)==state,'Diagnostic mutated learned policy')
    except Exception as error:
        record['diagnostic_error']=error_record(error)
    finally:
        record['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        c.core.write_json(destination/'result.json',record)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output.resolve();output.mkdir(parents=True,exist_ok=False)
    summary=dict(experiment=ROOT.name,cases=[],workers=1,
        protocol='Three fixed saved APs; baseline unchanged, then identical solve with logging/capture only',
        no_reoptimization_or_fallback=True,numerical_settings=dict(c.adversary.PARAMETERS),
        omitted_history='Fresh model per call; previous MP/other AP calls not replayed; compare saved outputs explicitly',
        timing_clock='time.perf_counter',environment=dict(host=platform.node(),platform=platform.platform(),
        python=sys.version,versions={k:importlib.metadata.version(k) for k in ('torch','numpy','gurobipy','moreau-cpu')}))
    try:
        with c.core.timed(summary):
            c.core.configure_worker()
            production_manifest=CAMPAIGN/'production_v1/provenance.json'
            c.verify_sources(c.read(production_manifest))
            bound={str(production_manifest):c.digest(production_manifest)}
            for spec in CASES:
                folder,assessment,saved,pools=inputs(spec)
                for p in (folder/'assessment.json',folder/'pool.json',Path(assessment['checkpoint']),
                          folder/f'sample_{spec[-1]:02d}.json'):
                    bound[str(p)]=c.digest(p)
            sources={str(p):c.digest(p) for p in ROOT.iterdir() if p.suffix=='.py' or p.name=='README.md'}
            c.core.write_json(output/'provenance.json',dict(source_sha256=sources,input_sha256=bound,
                frozen_production_source_sha256=c.read(production_manifest)['source_sha256']))
            for spec in CASES:
                for mode in ('baseline','logged'):
                    row=solve_once(spec,mode,output)
                    summary['cases'].append(row)
                    c.core.write_json(output/'summary.json',summary)
                    print(spec,mode,'diagnostic',row['status'],'AP',row.get('ap',{}).get('status'),
                          row.get('ap_error',{}).get('message'),flush=True)
            for p,h in {**bound,**sources}.items():
                c.check(c.digest(p)==h,f'Changed diagnostic input/source: {p}')
            c.verify_sources(c.read(production_manifest))
            c.check(all(r['status']=='completed' for r in summary['cases']),'Diagnostic execution failure; inspect raw artifacts')
    except BaseException as error:
        summary['error']=error_record(error)
        raise
    finally:
        c.core.write_json(output/'summary.json',summary)


if __name__=='__main__':
    main()