"""Frozen readiness inputs; no production source modifications."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
READINESS=ROOT.parent/'shipping_green_launch_20260925'
MANIFEST=READINESS/'preflight_v1/provenance.json'
SHA='7c25460d3b5dba2c164072d7a20d9ab46d002f43d3badb25f90b45a393aa0a6a'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


check(digest(MANIFEST)==SHA,'Changed readiness manifest')
_m=read(MANIFEST)
FROZEN={**_m['source_sha256'],**_m['input_sha256'],str(MANIFEST):SHA}
for _p,_h in FROZEN.items():
    check(digest(_p)==_h,f'Changed frozen input: {_p}')
sys.path.insert(0,str(READINESS))
import campaign_common as parent
from shipping_pool import state_hash
check(Path(parent.__file__).resolve()==READINESS/'campaign_common.py','Wrong parent module')
core,np,torch=parent.core,parent.np,parent.torch
shipping_model,resources=parent.shipping_model,parent.shipping_resources.resources
PRODUCTION=parent.PRODUCTION
project,binary=parent.adversary.project,parent.adversary.binary
PROTOCOL=dict(experiment=ROOT.name,seed=0,lr_net=.01,sizes=[50,200,800],radii=[0.,.2],
    repetitions=2,workers=1,threads=1,network_dtype='float32',projection_dtype='float64',
    hard_fee='unchanged 0.1 per q>0',epsilon=1e-6,replay_zero_tolerance=1e-9,
    reference='exact rational projection of represented native target/capacity; diagnostic only',
    sequence='fresh model; original-center batch then all AP singletons in saved order; controls afterwards',
    controls='fresh projection instance; captured batch target rows individually, no network recomputation',
    training=False,new_AP_solves=False,validation=False,test=False,full_CCG=False)


def sources():
    result={**FROZEN,**{str(p):digest(p) for p in ROOT.iterdir()
        if p.suffix in ('.py','.sh') or p.name=='README.md'}}
    # Generated readiness outputs were not covered by the pre-run manifest.
    for p in (READINESS/'preflight_v1').rglob('*.json'):
        result[str(p)]=digest(p)
    return result


def verify(mapping):
    for p,h in mapping.items():
        check(digest(p)==h,f'Input changed: {p}')


def metadata(n,radius,method,repeat):
    return {**parent.metadata(n,0,radius,method),'experiment':ROOT.name,
        'lr_net':.01,'repeat':repeat,'workers':1,'threads':1,
        'timing_clock':'time.perf_counter','timing_unit':'seconds'}