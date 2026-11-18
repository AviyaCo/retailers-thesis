"""Hash-bound definition-only imports. No historical entrypoint is executed."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
HOME = Path('/data/home/aviyac')
GREEN = ROOT.parent / 'green_R3_N200_N800_20260923'
OLD = HOME / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10'
CANONICAL = HOME / 'retailers_problem/R3/N50/saainit_net16_v2/run_all_net16_v2.py'
RESOURCE = ROOT.parent / 'green_R10_N50_N200_N800_20260924/r10_resources.py'
PROVENANCE = OLD / 'ccg_results/provenance.json'
THREAD_ENV = {k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREAD_ENV)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('GRB_LICENSE_FILE', str(HOME / 'gurobi.lic'))
sys.dont_write_bytecode = True


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


LOCKS = {
    PROVENANCE: 'a09a43cfa7a54ce295240c3c20114b69819d78dfb91e34c42e8adb78948fe0f5',
    GREEN / 'green_common.py': '0eb398fcad73fe348ac0a48612ca96f12dc6244b79f54916ecb2aa7a61951746',
    GREEN / 'green_training.py': 'e68dc3d94e23a5ea0ce59c576019a9f747a008a385f9f3a60c146eb102c8ab25',
    GREEN / 'run_campaign.py': 'd46731fd99b69978622a074ccc4d9956d0412cfd5b31b2c0a4680dd5e4a07468',
    GREEN / 'preflight.py': '7ad50db9c17024d292c7bc7a4270932d699f71e7a4ec552800367bd96761b318',
    RESOURCE: 'b55ee483f869306b5b15bf3c884568a2553fbb4d363e80e183e55f3f0c692d27',
    HOME / 'retailers_with_shipping/N50/run_cv_nominal.py':
        '2be3d35b513f487688dc468ca112351f449d2578a26bdd2224c9a193cdc91d0e',
}
if digest(PROVENANCE) != LOCKS[PROVENANCE]:
    raise RuntimeError('Changed historical provenance')
FROZEN = {**json.loads(PROVENANCE.read_text())['source_sha256'],
          **{str(p): h for p, h in LOCKS.items()}}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(sources):
    for p, h in FROZEN.items():
        check(sources.get(p) == h, f'Missing historical binding: {p}')
    for p, h in sources.items():
        check(digest(p) == h, f'Changed source: {p}')
    for p in ROOT.iterdir():
        if p.suffix in ('.py', '.sh') or p.name == 'README.md':
            check(str(p) in sources, f'Unbound local source: {p}')


for _p, _h in FROZEN.items():
    check(digest(_p) == _h, f'Changed historical source: {_p}')


def load_into(namespace, path, names, changes=()):
    """Whitelist definitions, retaining decorators; exact single-fragment edits."""
    path = Path(path)
    check(str(path) in FROZEN and digest(path) == FROZEN[str(path)], 'Unbound adapter input')
    text = path.read_text()
    nodes = [n for n in ast.parse(text).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    check({n.name for n in nodes} == set(names), 'Missing adapter definition')
    lines = text.splitlines(keepends=True)
    text = '\n\n'.join(''.join(lines[min([n.lineno] +
        [d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
    for old, new in changes:
        check(text.count(old) == 1, f'Ambiguous adapter fragment: {old}')
        text = text.replace(old, new)
    exec(compile(text, str(path), 'exec'), namespace)


def sources():
    return {**FROZEN, **{str(p): digest(p) for p in ROOT.iterdir()
        if p.suffix in ('.py', '.sh') or p.name == 'README.md'}}