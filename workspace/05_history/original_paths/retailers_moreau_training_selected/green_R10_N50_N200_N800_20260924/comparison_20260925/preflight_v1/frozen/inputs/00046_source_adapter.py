"""Fail-closed, explicit source adapters; never mutate/import legacy controllers.

Every edit below is an exact, hash-locked source fragment, not a numeric/global
substitution. Selected historical definitions execute in NEW module namespaces.
The original file and the adapter jointly determine the executable program.
"""
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
R10_REFERENCE = HOME / 'retailers_problem/R10/N50/saainit_net16_v2/run_all_net16_v2_N50_R10.py'
COMPARE = OLD / 'matched_oos_comparison/compare.py'
COMMON = ROOT.parent / 'N50_R3_train40_val10/common.py'
PROVENANCE = OLD / 'ccg_results/provenance.json'
SOURCE_NAMES = ('source_adapter.py', 'r10_core.py', 'r10_adversary.py', 'r10_pool.py',
    'r10_strict.py', 'r10_resources.py', 'green_common.py', 'green_training.py',
    'green_ccg.py', 'run_campaign.py', 'preflight.py', 'test_green.py',
    'test_campaign.py', 'test_r10.py', 'launch.sh', 'README.md')
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREAD_ENV)
os.environ.setdefault('GRB_LICENSE_FILE', str(HOME / 'gurobi.lic'))
sys.dont_write_bytecode = True

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

LOCKS = {
    PROVENANCE: 'a09a43cfa7a54ce295240c3c20114b69819d78dfb91e34c42e8adb78948fe0f5',
    COMMON: '51638dddb42dd84b86a8afc99f0e09b479c9f048a3ed5ce768248af6dca5ec81',
    COMPARE: '8285365c2913687a6c8769fee720b56237c785cb9f6d37d81b3a82f8c9d1b366',
    R10_REFERENCE: '0dcfc47d0e6e1133a5b083ddfd959f17f87a5c847b3adae47633cff5d5df8506',
}
LOCKS.update({GREEN / name: sha for name, sha in {
    'green_common.py': '0eb398fcad73fe348ac0a48612ca96f12dc6244b79f54916ecb2aa7a61951746',
    'green_training.py': 'e68dc3d94e23a5ea0ce59c576019a9f747a008a385f9f3a60c146eb102c8ab25',
    'green_ccg.py': '73b389ca177e694e5d0e7ebe1727e59850752f649a63e8c58cda35cc29ba7afc',
    'run_campaign.py': 'd46731fd99b69978622a074ccc4d9956d0412cfd5b31b2c0a4680dd5e4a07468',
    'preflight.py': '7ad50db9c17024d292c7bc7a4270932d699f71e7a4ec552800367bd96761b318',
    'test_green.py': 'bc8542101df331abc041fbc31741ac50767aa0c7cc4f379c12af1cea783693d6',
    'test_campaign.py': '81d730beb058acdff5e046ba4c35f855314ea8f15e685f28a4e8354ea1d9c830',
    'launch.sh': '4e4162c0159faaaa87a96eda9611dbefa3284867bb3b3d8b48a0b8186933b124',
    'README.md': 'a069e6aa9c2fea9cf96d5287a433a0b780064b18a88f85026f0c3367a78999d8',
}.items()})
if digest(PROVENANCE) != LOCKS[PROVENANCE]:
    raise RuntimeError('Historical provenance changed')
FROZEN = {**json.loads(PROVENANCE.read_text())['source_sha256'],
          **{str(p): h for p, h in LOCKS.items()}}
for _path, _sha in FROZEN.items():
    if digest(_path) != _sha:
        raise RuntimeError(f'Frozen source changed: {_path}')

ADAPTATIONS = []

def source(path):
    path = Path(path)
    if str(path) not in FROZEN or digest(path) != FROZEN[str(path)]:
        raise RuntimeError(f'Unbound/changed source: {path}')
    return path.read_text()

def rewrite(text, changes, path):
    """Each fragment must occur; multiplicity is recorded against locked bytes."""
    for old, new in changes:
        count = text.count(old)
        if not count:
            raise RuntimeError(f'Adapter fragment absent in {path}: {old!r}')
        text = text.replace(old, new)
        ADAPTATIONS.append(dict(source=str(path), before=old, after=new, occurrences=count))
    return text

def load_into(namespace, path, changes=(), names=None):
    text = source(path)
    if names is not None:
        # No top-level executable code, imports, historical trainers or launchers.
        nodes = [n for n in ast.parse(text).body
                 if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise RuntimeError(f'Definition coverage mismatch: {path}')
        # Function/Class source segments begin AFTER top-level decorators.
        # Preserve those lines, notably @contextmanager on timed/_phase.
        lines = text.splitlines(keepends=True)
        text = '\n\n'.join(''.join(lines[min([n.lineno] +
            [d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
    text = rewrite(text, changes, path)
    tree = ast.parse(text, filename=str(path))
    # Entrypoints are called only by the new local wrappers, never on import.
    tree.body = [n for n in tree.body if not (isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
        and n.test.left.id == '__name__')]
    exec(compile(tree, str(path), 'exec'), namespace)

def forbidden(*args, **kwargs):
    raise RuntimeError('Historical split/SAA/training/controller is unavailable in isolated R10')