"""Isolated replay-only revision; previous failed runs remain hash-bound."""
import importlib.util
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'shipping_epsilon_green_R3_20260925'
PARENT_MANIFEST = PREVIOUS / 'ap_gate_v2/provenance.json'
PARENT_SHA = '1fbd48c6b1e3e198c8c8b395c3d8083d9198e6f4fb7f0b5a96c6b4c9ab299670'
SAVED_FAILURE = PREVIOUS / 'ap_gate_v2/N800_radius0.2.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


check(digest(PARENT_MANIFEST)==PARENT_SHA,'Changed previous gate manifest')
_manifest=read(PARENT_MANIFEST)
FROZEN={**_manifest['source_sha256'],**_manifest['input_sha256'],
        str(PARENT_MANIFEST):PARENT_SHA,
        str(SAVED_FAILURE):'24efe811880c9ec91d4c19ee8c91c0bdf111782d104f23f0069731d47e15123f'}
for _path,_sha in FROZEN.items():
    check(digest(_path)==_sha,f'Changed previous evidence: {_path}')
_spec=importlib.util.spec_from_file_location('_frozen_epsilon_common',PREVIOUS/'epsilon_common.py')
check(_spec is not None and _spec.loader is not None,'Cannot load prior common definitions')
_previous=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_previous)
core,np,torch=_previous.core,_previous.np,_previous.torch
shipping_model,shipping_resources=_previous.shipping_model,_previous.shipping_resources
PRODUCTION,OLD,DIAGNOSTIC=_previous.PRODUCTION,_previous.OLD,_previous.DIAGNOSTIC
THREAD_ENV,EPSILON=_previous.THREAD_ENV,_previous.EPSILON
REPLAY_ZERO_TOLERANCE=1e-9
PROTOCOL={**_previous.PROTOCOL,'experiment':ROOT.name,
    'replay':'strict rational support; logged native zero-residue exception with fee-explained cost check',
    'replay_zero_tolerance':REPLAY_ZERO_TOLERANCE,
    'replay_exception_changes_cost_or_shipments':False,
    'replay_exception':'AP q=0 and rational q=0, native 0<q<=1e-9; unexplained cost residual must pass',
    'full_CCG_launched':False}


def definitions(namespace,path,names,changes=()):
    check(str(path) in FROZEN and digest(path)==FROZEN[str(path)],'Unbound definition input')
    _previous.definitions(namespace,path,names,changes)


def sources():
    return {**FROZEN,**{str(p):digest(p) for p in ROOT.iterdir()
        if p.suffix in ('.py','.sh') or p.name=='README.md'}}


def verify(mapping):
    for path,sha in FROZEN.items():
        check(mapping.get(path)==sha,f'Missing frozen evidence: {path}')
    for path,sha in mapping.items():
        check(digest(path)==sha,f'Changed input: {path}')
    for p in ROOT.iterdir():
        if p.suffix in ('.py','.sh') or p.name=='README.md':
            check(str(p) in mapping,f'Unbound local source: {p}')


def metadata(n,seed,radius,method):
    return {**_previous.metadata(n,seed,radius,method),'experiment':ROOT.name,
            'replay_zero_tolerance':REPLAY_ZERO_TOLERANCE}