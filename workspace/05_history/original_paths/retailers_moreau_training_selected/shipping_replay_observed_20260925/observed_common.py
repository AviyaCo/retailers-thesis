"""Isolated user-authorized replay observation, preserving frozen experiments."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'shipping_green_launch_20260925'
MANIFEST=PREVIOUS/'preflight_v1/provenance.json'
MANIFEST_SHA='7c25460d3b5dba2c164072d7a20d9ab46d002f43d3badb25f90b45a393aa0a6a'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


check(digest(MANIFEST)==MANIFEST_SHA,'Changed readiness manifest')
_manifest=read(MANIFEST)
FROZEN={**_manifest['source_sha256'],**_manifest['input_sha256'],str(MANIFEST):MANIFEST_SHA}
for _path,_sha in FROZEN.items():
    check(digest(_path)==_sha,f'Frozen input changed: {_path}')
sys.path.insert(0,str(PREVIOUS))
import campaign_common as previous
check(Path(previous.__file__).resolve()==PREVIOUS/'campaign_common.py','Wrong previous module')
core,np,torch=previous.core,previous.np,previous.torch
shipping_model,shipping_resources=previous.shipping_model,previous.shipping_resources
PRODUCTION,OLD,REPLAY=previous.PRODUCTION,previous.OLD,previous.REPLAY
THREAD_ENV=previous.THREAD_ENV
PROTOCOL={k:v for k,v in previous.PROTOCOL.items() if not k.startswith('replay')}
PROTOCOL.update(experiment=ROOT.name,replay='observed only; no native support/cost/epsilon rejection',
    replay_rejection_enabled=False,full_CCG_launched=False,
    ap_internal_validation='unchanged binary/epsilon links and rational folded-policy support',
    native_physical_tolerance=1e-9,
    signed_gap_checks='unchanged; native replay observations never waive signed ordering',
    ap_cost_replay_tolerance=1e-4,ap_cost_replay_tolerance_role='diagnostic only; not an acceptance threshold')


def definitions(namespace,path,names,changes=()):
    path=Path(path)
    check(str(path) in FROZEN and digest(path)==FROZEN[str(path)],'Unbound source definition')
    previous.previous._previous.definitions(namespace,path,names,changes)


def sources():
    return {**FROZEN,**{str(p):digest(p) for p in ROOT.iterdir()
        if p.suffix in ('.py','.sh') or p.name=='README.md'}}


def verify(mapping):
    for p,h in FROZEN.items():
        check(mapping.get(p)==h,f'Missing frozen input: {p}')
    for p,h in mapping.items():
        check(digest(p)==h,f'Changed input: {p}')


def metadata(n,seed,radius,method):
    return {**{k:v for k,v in previous.metadata(n,seed,radius,method).items()
        if not k.startswith('replay')},'experiment':ROOT.name,
        'replay_rejection_enabled':False}