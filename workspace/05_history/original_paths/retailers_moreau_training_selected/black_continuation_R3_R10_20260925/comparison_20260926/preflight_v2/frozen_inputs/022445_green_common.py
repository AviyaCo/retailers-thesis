"""Frozen green R3 protocol, explicitly adapted only to R10 and N50/200/800."""
from source_adapter import (ROOT, GREEN, COMMON, FROZEN, SOURCE_NAMES, THREAD_ENV,
                            load_into, source, rewrite)
import ast
import hashlib
from pathlib import Path
from scipy.stats import truncnorm
import r10_core as core
import r10_pool as legacy_ccg
import r10_adversary as adversary

np, torch = core.np, core.torch
load_into(globals(), COMMON, names=('check', 'raw_hash', 'evaluate'))
SIZES = (50, 200, 800)
SEEDS = tuple(range(10))
RADII = core.RADII
LR_GRID = core.LR_GRID
MAX_MP = 40
_tree = ast.parse(source(GREEN / 'green_common.py'))
_protocol = next(n for n in _tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'PROTOCOL' for t in n.targets))
_text = ast.get_source_segment(source(GREEN / 'green_common.py'), _protocol)
exec(rewrite(_text, [
    ('retailer_live_green_R3_N200_N800_20260923', 'retailer_live_green_R10_N50_N200_N800_20260924'),
    ('R=3,', 'R=10,'),
], GREEN / 'green_common.py'), globals())
load_into(globals(), GREEN / 'green_common.py', [
    ('Nval=n_total // 5, R=3, f=0', 'Nval=n_total // 5, R=10, f=0'),
    ('(n_total, 3)', '(n_total, 10)'),
    ('n in (160, 640)', 'n in (40, 160, 640)'),
    ('(n, 3)', '(n, 10)'),
    ('paired R3', 'paired R10'),
    ('model.addVars(3,', 'model.addVars(10,'),
    ('model.addVars(n, 3,', 'model.addVars(n, 10,'),
    ('for r in range(3)', 'for r in range(10)'),
], names=('configure_worker', 'metadata', 'split_data', 'solve_saa'))

def verify_sources(manifest):
    sources = manifest['source_sha256']
    for path, expected in FROZEN.items():
        check(sources.get(path) == expected, f'Missing/changed historical freeze: {path}')
    for name in SOURCE_NAMES:
        check(str(ROOT / name) in sources, f'Missing computational source freeze: {name}')
    for path, expected in sources.items():
        check(core.digest(path) == expected, f'Frozen source changed: {path}')