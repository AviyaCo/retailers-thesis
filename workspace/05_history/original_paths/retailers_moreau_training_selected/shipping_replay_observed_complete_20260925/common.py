"""Complete advisory logging, including witnesses rejected for physical failure."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parent
OBSERVED=ROOT.parent/'shipping_replay_observed_20260925'
MANIFEST=OBSERVED/'preflight_v1/provenance.json'
MANIFEST_SHA='1cc1d1f63b9b9c673f5a203196a04e73a5f695c330cb7dbdb79cc1f7f757694c'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


check(digest(MANIFEST)==MANIFEST_SHA,'Changed first advisory run manifest')
_manifest=read(MANIFEST)
FIXTURE=ROOT.parent/'shipping_replay_tolerance_20260925/ap_gate_v1/N800_radius0.2.json'
FROZEN={**_manifest['source_sha256'],**_manifest['input_sha256'],str(MANIFEST):MANIFEST_SHA,
    str(FIXTURE):'7a24c85e76e54476b3c1330af2e922492f4fcc4ffb37c3c3bc300abc6559fe7e'}
# Bind comparison artifacts explicitly as well as the imported sources.
for _folder in (OBSERVED/'preflight_v1',ROOT.parent/'shipping_green_launch_20260925/preflight_v1'):
    FROZEN.update({str(p):digest(p) for p in _folder.rglob('*.json')})
for _p,_h in FROZEN.items():
    check(digest(_p)==_h,f'Changed frozen input: {_p}')
sys.path.insert(0,str(OBSERVED))
import observed_common as prior
import observed_adversary as prior_adversary
import observed_pool as prior_pool
check(Path(prior.__file__).resolve()==OBSERVED/'observed_common.py','Wrong advisory dependency')
core,np,torch=prior.core,prior.np,prior.torch
shipping_model,shipping_resources=prior.shipping_model,prior.shipping_resources
PRODUCTION,OLD,REPLAY,PREVIOUS=prior.PRODUCTION,prior.OLD,prior.REPLAY,prior.PREVIOUS
PROTOCOL={**prior.PROTOCOL,'experiment':ROOT.name,
    'cost_observation_coverage':'all finite native witnesses, including physical failures'}


def definitions(namespace,path,names,changes=()):
    check(str(path) in FROZEN and digest(path)==FROZEN[str(path)],'Unbound source definition')
    prior.previous.previous._previous.definitions(namespace,path,names,changes)


def sources():
    return {**FROZEN,**{str(p):digest(p) for p in ROOT.iterdir()
        if p.suffix in ('.py','.sh') or p.name=='README.md'}}


def verify(mapping):
    for p,h in FROZEN.items():
        check(mapping.get(p)==h,f'Missing frozen input: {p}')
    for p,h in mapping.items():
        check(digest(p)==h,f'Changed input: {p}')


def metadata(n,seed,radius,method):
    return {**prior.metadata(n,seed,radius,method),'experiment':ROOT.name}