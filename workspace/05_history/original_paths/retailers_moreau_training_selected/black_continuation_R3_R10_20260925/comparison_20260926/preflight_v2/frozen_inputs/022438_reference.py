"""Hash-locked definition reuse: no historical module execution or training imports.

Model, projection and cost fragments use the same R10 adaptations as r10_strict.
Only gen changes its retailer dimension; its legacy RNG and precision are intact.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('GRB_LICENSE_FILE', '/data/home/aviyac/gurobi.lic')

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy
from scipy.stats import truncnorm
import torch
from torch import nn

HOME = Path('/data/home/aviyac')
OLD = HOME / 'retailers_problem'
COMPARE = HOME / 'retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10/matched_oos_comparison/compare.py'
R3_EVALUATION = HOME / 'retailers_moreau_training_selected/green_R3_N200_N800_20260923/comparison_20260924/evaluation.py'
LOCKS = {
    COMPARE: '8285365c2913687a6c8769fee720b56237c785cb9f6d37d81b3a82f8c9d1b366',
    R3_EVALUATION: '794d7d6f816f0afcc4373c5acaad0f949e94b7b28ab269892a77dcfc04ac1a22',
}
PHYSICAL_TOL = 1e-4
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
ADAPTATIONS = []


def load_definitions(namespace, path, names, changes=()):
    """Execute only named definitions; require exact source and fragment counts."""
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != LOCKS[path]:
        raise RuntimeError(f'Bound evaluator source changed: {path}')
    text = data.decode()
    nodes = [n for n in ast.parse(text).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if len(nodes) != len(names) or {n.name for n in nodes} != set(names):
        raise RuntimeError(f'Definition coverage mismatch: {path}')
    lines = text.splitlines(keepends=True)
    text = '\n\n'.join(''.join(lines[min([n.lineno] +
        [d.lineno for d in n.decorator_list])-1:n.end_lineno]) for n in nodes)
    for before, after, count in changes:
        if text.count(before) != count:
            raise RuntimeError(f'Exact fragment count mismatch: {path}: {before!r}')
        text = text.replace(before, after)
        ADAPTATIONS.append(dict(source=str(path), before=before, after=after, occurrences=count))
    exec(compile(text, str(path), 'exec'), namespace)


load_definitions(globals(), COMPARE,
    ('utc', 'sha', 'read_json', 'write_json', 'phase', 'initialize_worker',
     'Model', 'load_model', 'state_hash', 'gen', 'scenario_costs', 'strict_project', 'human'),
    (('nn.Linear(7, 16)', 'nn.Linear(21, 16)', 1),
     ('nn.Linear(16, 3)', 'nn.Linear(16, 10)', 1),
     ('torch.zeros(3)', 'torch.zeros(10)', 1),
     ('(5, 5)', '(12, 12)', 1), ('(17, 5)', '(45, 12)', 1),
     ('size=(n, 3)', 'size=(n, 10)', 1),
     ('np.shape(z) != (3,)', 'np.shape(z) != (10,)', 1),
     ('m.addVars(3,', 'm.addVars(10,', 1),
     ('for r in range(3)', 'for r in range(10)', 3)))


def old_choice(rho, seed, directories, nominal):
    """R10 has no special-radius fallback; final_mp_iter alone selects the MP."""
    if rho == 0:
        return nominal / f'seed_{seed}_nominal.pth', 0, None
    if rho not in directories:
        raise ValueError(f'Missing SCS radius directory: {rho}')
    folder = directories[rho]
    selector = folder / f'seed_{seed}_summary.json'
    summary = read_json(selector)
    iteration = summary['final_mp_iter']
    if type(iteration) is not int or iteration < 0:
        raise ValueError('Invalid final_mp_iter')
    if summary['seed'] != seed or summary['rho'] != rho or summary['completed'] is not True:
        raise ValueError('SCS summary identity/completion mismatch')
    # completed is a computational status, not convergence certification.
    if type(summary['converged']) is not bool:
        raise ValueError('Missing/invalid historical convergence status')
    checkpoint = (nominal / f'seed_{seed}_nominal.pth' if iteration == 0 else
                  folder / f'seed_{seed}_mp_iter{iteration}.pth')
    return checkpoint, iteration, selector