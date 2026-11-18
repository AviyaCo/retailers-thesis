"""Frozen shipping primitives and the explicitly approved full-campaign protocol."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
GUARD=ROOT.parent/'shipping_approximate_guard_20260926'
READY=GUARD/'preflight_v1/summary.json'
READY_SHA='12443bf75e70adf33619cd2854909503f04c2a4e7b46634109511c4ac8ccd4f6'
MANIFEST=GUARD/'preflight_v1/provenance.json'
MANIFEST_SHA='e93531068565867146e8008c1a4ffcee53aaa9aa41bc043468fdbcd6794855f4'
GREEN=ROOT.parent/'green_R3_N200_N800_20260923'
NOMINAL=ROOT.parent/'shipping_nominal_R3_N50_N200_N800_20260924'
PRODUCTION=NOMINAL/'production_v1'


def check(condition,message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


check(digest(MANIFEST)==MANIFEST_SHA and digest(READY)==READY_SHA,'Changed approved readiness')
_ready=read(READY)
check(_ready['status']=='completed' and len(_ready['cases'])==6
      and all(c['accepted'] for c in _ready['cases']), 'Approved readiness did not pass')
_manifest=read(MANIFEST)
FROZEN={**_manifest['source_sha256'],**_manifest['input_sha256'],
    str(MANIFEST):MANIFEST_SHA,str(READY):READY_SHA}
for _p in (GREEN/'green_ccg.py',GREEN/'test_green.py',NOMINAL/'run_nominal.py'):
    FROZEN.setdefault(str(_p),digest(_p))
for _p,_h in FROZEN.items():
    check(digest(_p)==_h,f'Changed frozen dependency: {_p}')
sys.path.insert(0,str(GUARD))
import guard_common as approved
import guard_adversary as adversary
import guard_pool as legacy_ccg
import shipping_core
import shipping_model
import shipping_training
import shipping_common as nominal_common
import shipping_resources
check(Path(adversary.__file__).resolve()==GUARD/'guard_adversary.py','Wrong shipping AP')
check(Path(shipping_model.__file__).resolve()==NOMINAL/'shipping_model.py','Wrong shipping model')
core=SimpleNamespace(**{k:getattr(shipping_core,k) for k in (
    'timed','utc_now','write_json','weights','_json_value','stats','NativeProjection','configure_worker')},
    model_from_state=shipping_model.model_from_state,digest=digest)
np,torch=shipping_core.np,shipping_core.torch
THREAD_ENV=approved.parent.prior.previous.THREAD_ENV
sys.path.insert(0,str(ROOT))
SIZES=(50,200,800)
SEEDS=tuple(range(10))
LR_GRID=(.005,.01,.05)
RADII=(0.,.001,.003,.01,.03,.2,.4,.7,1.5)
MAX_MP=40
PROTOCOL={**approved.PROTOCOL,'experiment':ROOT.name,
    'sizes':list(SIZES),'data_seeds':list(SEEDS),'lr_grid':list(LR_GRID),'radii':list(RADII),
    'max_mp':MAX_MP,'max_epochs':3000,'patience':300,'improvement_threshold':1e-4,
    'green_gap_floor':1e-9,'green_gap_tolerance':.01,'separation_tolerance':.01,
    'membership_tolerance':1e-9,'ap_replay_tolerance':1e-4,
    'nominal_lr_selection':'minimum held-out native hard cost; smaller LR exact tie',
    'radius_selection':'minimum held-out native hard cost; smaller radius exact tie; all270 converged required',
    'ccg_selection':'earliest minimum AP incumbent over complete prefix through first strict green pass',
    'validation':'only before CCG for LR and after complete converged CCG grid for radius; never inside MP/CCG',
    'training_loss':'smooth native shipping cost; training-only checkpoints and scheduler',
    'pool_cost':'native strict-positive hard cost for AP comparisons and stopping',
    'nominal_origin':'reuse all90 completed fits; no refit',
    'full_CCG_launched':True,'test_evaluated':False,'refit':False,'resume':False,
    'solver_fallback':False,'global_robust_optimality_certified':False}


def definitions(namespace,path,names,changes=()):
    path=Path(path)
    check(str(path) in FROZEN and digest(path)==FROZEN[str(path)],'Unbound definition source')
    lines=path.read_text().splitlines(keepends=True)
    nodes=[n for n in ast.parse(''.join(lines)).body
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    check({n.name for n in nodes}==set(names),'Missing reused definition')
    text='\n\n'.join(''.join(lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
    for old,new in changes:
        check(text.count(old)==1,f'Ambiguous source adaptation: {old}')
        text=text.replace(old,new)
    exec(compile(text,str(path),'exec'),namespace)


def sources():
    return {**FROZEN,**{str(p):digest(p) for p in ROOT.iterdir()
        if p.suffix in ('.py','.sh') or p.name=='README.md'}}


def verify_sources(manifest):
    mapping=manifest['source_sha256']
    for p,h in FROZEN.items():
        check(mapping.get(p)==h,f'Missing frozen input: {p}')
    for p in ROOT.iterdir():
        if p.suffix in ('.py','.sh') or p.name=='README.md':
            check(str(p) in mapping,f'Unbound local source: {p}')
    for p,h in mapping.items():
        check(digest(p)==h,f'Changed source/input: {p}')
    for p,h in manifest.get('input_sha256',{}).items():
        check(digest(p)==h,f'Changed nominal input: {p}')


def validate_integration(path,expected):
    path=Path(path).resolve()
    check(digest(path)==expected,'Changed integration summary')
    summary=read(path)
    check(summary['status']=='completed' and summary['tests']['status']=='completed'
        and summary['disposable'] is True,'Integration did not complete')
    cases=summary['cases']
    check(len(cases)==2 and {(r['Ntotal'],r['seed'],r['radius']) for r in cases}=={(50,0,0.),(50,0,.2)}
        and all(r['status']=='completed' and r['converged'] for r in cases),'Wrong integration grid')
    check(next(r for r in cases if r['radius']==0.)['mp_count']==0
        and next(r for r in cases if r['radius']==.2)['mp_count']>=1,'Missing native MP coverage')
    for row in cases:
        check(digest(row['summary_path'])==row['summary_sha256'],'Changed integration case')
        check(digest(row['selected_checkpoint'])==row['selected_checkpoint_sha256'],
            'Changed integration selected checkpoint')
    manifest_path=path.parent/'provenance.json'
    check(str(manifest_path)==summary['provenance_path']
        and digest(manifest_path)==summary['provenance_sha256'],'Changed integration manifest')
    manifest=read(manifest_path)
    check(manifest['protocol']==PROTOCOL,'Integration protocol differs')
    verify_sources(manifest)
    return summary


def metadata(n_total,seed=None,radius=None,method=None):
    check(n_total in SIZES,'Unapproved total size')
    return dict(experiment=ROOT.name,method=method,Ntotal=n_total,Ntrain=4*n_total//5,
        Nval=n_total//5,seed=seed,radius=radius,R=3,f=.1,gamma=5.,network_seed=0,
        split_seed=None if seed is None else 100000+seed,torch_threads=1,gurobi_threads=1,
        timing_clock='time.perf_counter',timing_unit='seconds',
        original_box_certificate=False,native_physical_tolerance=1e-6,
        negative_bound_pool_rejection_enabled=False,replay_rejection_enabled=False)