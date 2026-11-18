"""Hash-bound revision of the explicitly approved approximate acceptance rules."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'shipping_replay_observed_complete_20260925'
MANIFEST=PARENT/'preflight_v1/provenance.json'
MANIFEST_SHA='edbd4e52f9bbc7cea37488ad0cfdaed06750c12e9442cca691798ff123557839'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


check(digest(MANIFEST)==MANIFEST_SHA,'Changed parent readiness manifest')
_manifest=read(MANIFEST)
FROZEN={**_manifest['source_sha256'],**_manifest['input_sha256'],str(MANIFEST):MANIFEST_SHA}
for _path in (PARENT/'preflight_v1').rglob('*.json'):
    FROZEN.setdefault(str(_path),digest(_path))
for _p,_h in FROZEN.items():
    check(digest(_p)==_h,f'Changed frozen input: {_p}')
sys.path.insert(0,str(PARENT))
import common as parent
import adversary as parent_adversary
check(Path(parent.__file__).resolve()==PARENT/'common.py','Wrong parent common import')
check(Path(parent_adversary.__file__).resolve()==PARENT/'adversary.py','Wrong parent adversary import')
core,np,torch=parent.core,parent.np,parent.torch
shipping_model,shipping_resources=parent.shipping_model,parent.shipping_resources
PRODUCTION,OLD,REPLAY=parent.PRODUCTION,parent.OLD,parent.REPLAY
PHYSICAL_TOLERANCE=1e-6
PROTOCOL={**parent.PROTOCOL,'experiment':ROOT.name,
    'native_physical_tolerance':PHYSICAL_TOLERANCE,
    'signed_gap_checks':'bound/incumbent remains enforced; negative bound/native-pool is diagnostic only',
    'negative_bound_pool_rejection_enabled':False,
    'acceptance_interpretation':'approximate numerical readiness, not an exact native hard-cost certificate',
    'full_CCG_launched':False}


def definitions(namespace,path,names,changes=()):
    path=Path(path)
    check(str(path) in FROZEN and digest(path)==FROZEN[str(path)],'Unbound source definition')
    parent.prior.previous.previous._previous.definitions(namespace,path,names,changes)


def sources():
    return {**FROZEN,**{str(p):digest(p) for p in ROOT.iterdir()
        if p.suffix in ('.py','.sh') or p.name=='README.md'}}


def verify(mapping):
    for p,h in FROZEN.items():
        check(mapping.get(p)==h,f'Missing frozen input: {p}')
    for p,h in mapping.items():
        check(digest(p)==h,f'Changed input: {p}')


def metadata(n,seed,radius,method):
    return {**parent.metadata(n,seed,radius,method),'experiment':ROOT.name,
        'native_physical_tolerance':PHYSICAL_TOLERANCE,
        'negative_bound_pool_rejection_enabled':False,
        'original_box_certificate':False}