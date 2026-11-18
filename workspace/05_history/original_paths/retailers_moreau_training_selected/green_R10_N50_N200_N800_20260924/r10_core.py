"""Private R10 model/projection; same frozen numerical implementation/settings."""
from source_adapter import OLD, CANONICAL, HOME, load_into, forbidden
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from types import SimpleNamespace

_DEPS = HOME / 'hydro_rcr_rescaled/.moreau_diagnostic_deps'
sys.path.insert(0, str(_DEPS))
import gurobipy as gp
from gurobipy import GRB
import moreau
from moreau.torch import Solver
import numpy as np
from scipy import sparse
from scipy.stats import truncnorm
import torch
from torch import nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

if moreau.__version__ != '0.4.0' or not Path(moreau.__file__).resolve().is_relative_to(_DEPS):
    raise ImportError('Require frozen isolated Moreau 0.4.0')

# Whitelisted definitions in a NEW dictionary, not the old loader's globals.
_scope = dict(torch=torch, nn=nn, np=np, cp=cp, CvxpyLayer=CvxpyLayer, truncnorm=truncnorm)
_names = ('Cfg', 'softplus_inverse', 'generate_data', 'Model', 'make_model_saa_init')
load_into(_scope, CANONICAL, [('R             = 3', 'R             = 10')], _names)
_scope['cfg'] = _scope['Cfg']()
_scope['cfg'].LR_NET = .05
baseline = SimpleNamespace(**{n: _scope[n] for n in _names}, cfg=_scope['cfg'])
# Provenance compatibility only; the original loader is NOT imported/executed.
_BASELINE_PATH = HOME / 'retailers_moreau_N50_R3_net16_v2/baseline.py'
_loader = SimpleNamespace(SOURCE=CANONICAL)
_SOURCE_SOLVER_ARGS = {'solve_method': 'SCS', 'eps': 1e-4, 'max_iters': 200000}
LR_GRID = (.005, .01, .05)
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
SEEDS = tuple(range(10))

load_into(globals(), OLD / 'core.py', [
    ("sparse.eye(3, format='csr')", "sparse.eye(10, format='csr')"),
    ('np.vstack([-np.eye(3), np.ones((1, 3))])', 'np.vstack([-np.eye(10), np.ones((1, 10))])'),
    ('dict(n=3, m=4,', 'dict(n=10, m=11,'),
    ('moreau.Cones(num_nonneg_cones=4)', 'moreau.Cones(num_nonneg_cones=11)'),
    ('z.shape[1] != 3', 'z.shape[1] != 10'),
    ('Expected z=(batch,3)', 'Expected z=(batch,10)'),
], names=('utc_now', 'digest', 'weights', '_json_value', 'write_json', 'timed',
          'stats', 'NativeProjection', '_check_weights', '_load_learned',
          'make_model', 'model_from_state', '_phase'))
split_data = solve_saa = train_model = forbidden