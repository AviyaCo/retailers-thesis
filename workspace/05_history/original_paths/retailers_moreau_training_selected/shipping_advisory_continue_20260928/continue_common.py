"""Isolated user-authorized advisory checks and five fixed boundary resumes."""
import ast
import copy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'shipping_live_green_20260926'
sys.path.insert(0,str(PARENT))
import shipping_live_common as parent
sys.path.insert(0,str(ROOT))
core,np,torch=parent.core,parent.np,parent.torch
check,digest,read=parent.check,parent.digest,parent.read
GREEN,MAX_MP=parent.GREEN,40
shipping_training,legacy_ccg=parent.shipping_training,parent.legacy_ccg
SOURCE_MANIFEST=PARENT/'production_v1/provenance.json'
FROZEN=dict(read(SOURCE_MANIFEST)['source_sha256'])
parent.verify_sources(read(SOURCE_MANIFEST))
PROTOCOL={**copy.deepcopy(parent.PROTOCOL),'experiment':ROOT.name,
    'activation_link_rejection_enabled':False,'ap_positive_epsilon_rejection_enabled':False,
    'exact_support_rejection_enabled':False,'exact_duplicate_separation_rejection_enabled':False,
    'float32_collision_rejection_enabled':True,'resume':True,
    'continuation_scope':'five failed AP boundaries, cached valid prefixes; original forty-MP budget',
    'acceptance_interpretation':'explicitly approximate; invalid activation links may be retained, never certified',
    'full_CCG_launched':False,'radius_selection_run':False}


def metadata(n_total,seed=None,radius=None,method=None):
    return {**parent.metadata(n_total,seed,radius,method),'experiment':ROOT.name,
        'activation_link_rejection_enabled':False,'exact_duplicate_separation_rejection_enabled':False}


def definitions(namespace,path,names,changes=()):
    path=Path(path)
    check(str(path) in FROZEN and digest(path)==FROZEN[str(path)],f'Unbound definition: {path}')
    lines=path.read_text().splitlines(keepends=True)
    nodes=[n for n in ast.parse(''.join(lines)).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    check({n.name for n in nodes}==set(names),'Missing definition')
    text='\n\n'.join(''.join(lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
    for old,new in changes:
        check(text.count(old)==1,f'Ambiguous adaptation: {old}')
        text=text.replace(old,new)
    exec(compile(text,str(path),'exec'),namespace)


def sources():
    return {**FROZEN,**{str(p):digest(p) for p in ROOT.iterdir() if p.suffix=='.py' or p.name=='README.md'}}


def verify_sources(manifest):
    mapping=manifest['source_sha256']
    for p,h in FROZEN.items():check(mapping.get(p)==h,f'Missing historical binding: {p}')
    for p in ROOT.iterdir():
        if p.suffix=='.py' or p.name=='README.md':check(str(p) in mapping,f'Missing local source: {p}')
    for p,h in {**mapping,**manifest['input_sha256']}.items():check(digest(p)==h,f'Changed source/input: {p}')