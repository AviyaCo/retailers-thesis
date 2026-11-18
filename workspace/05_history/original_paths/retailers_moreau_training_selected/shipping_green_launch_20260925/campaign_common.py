"""Hash-bound imports for the approved shipping launch integration."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
REPLAY=ROOT.parent/'shipping_replay_tolerance_20260925'
MANIFEST=REPLAY/'ap_gate_v1/provenance.json'
MANIFEST_SHA='18a21bf16d3e9e81093087caef54e548d824ec6bf9b2bfb9a7c12ad686d29042'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


check(digest(MANIFEST)==MANIFEST_SHA,'Changed completed replay gate manifest')
_manifest=read(MANIFEST)
FROZEN={**_manifest['source_sha256'],**_manifest['input_sha256'],str(MANIFEST):MANIFEST_SHA}
for _path,_sha in FROZEN.items():
    check(digest(_path)==_sha,f'Frozen input changed: {_path}')
sys.path.insert(0,str(REPLAY))
import epsilon_common as previous
import epsilon_adversary as adversary
check(Path(previous.__file__).resolve()==REPLAY/'epsilon_common.py','Wrong replay common import')
check(Path(adversary.__file__).resolve()==REPLAY/'epsilon_adversary.py','Wrong adversary import')
core,np,torch=previous.core,previous.np,previous.torch
shipping_model,shipping_resources=previous.shipping_model,previous.shipping_resources
PRODUCTION,OLD=previous.PRODUCTION,previous.OLD
THREAD_ENV=previous.THREAD_ENV
PROTOCOL={**previous.PROTOCOL,'experiment':ROOT.name,
    'launch_authorized':True,'full_CCG_launched':False,
    'readiness_scope':'all training centers; seed0 LR0.01; N50/200/800; radii0/0.2',
    'pool_cost':'unchanged native strict-positive hard fee',
    'MP_cost':'unchanged smooth fee',
    'signed_gap_checks':'unchanged; replay exceptions never waive negative bound/pool checks'}


def definitions(namespace,path,names,changes=()):
    check(str(path) in FROZEN and digest(path)==FROZEN[str(path)],'Unbound source definition')
    previous.definitions(namespace,path,names,changes)


def sources():
    return {**FROZEN,**{str(p):digest(p) for p in ROOT.iterdir()
        if p.suffix in ('.py','.sh') or p.name=='README.md'}}


def verify(mapping):
    for p,h in FROZEN.items():
        check(mapping.get(p)==h,f'Missing frozen input: {p}')
    for p,h in mapping.items():
        check(digest(p)==h,f'Changed input: {p}')


def metadata(n,seed,radius,method):
    return {**previous.metadata(n,seed,radius,method),'experiment':ROOT.name}