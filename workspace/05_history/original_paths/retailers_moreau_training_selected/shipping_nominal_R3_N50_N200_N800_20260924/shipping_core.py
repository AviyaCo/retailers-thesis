"""Native green R3 numerical primitives in an importable, isolated module."""
from source_adapter import *
from collections import Counter, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import math
import random
import statistics
import tempfile
import time

sys.path.insert(0, str(HOME / 'hydro_rcr_rescaled/.moreau_diagnostic_deps'))
import gurobipy as gp
from gurobipy import GRB
import moreau
from moreau.torch import Solver
import numpy as np
from scipy import sparse
from scipy.stats import truncnorm
import torch
from torch import nn

check(moreau.__version__ == '0.4.0', 'Require frozen Moreau 0.4.0')
_SOURCE_SOLVER_ARGS = {'solve_method': 'SCS', 'eps': 1e-4, 'max_iters': 200000}
load_into(globals(), OLD / 'core.py', ('utc_now', 'weights', '_json_value',
    'write_json', 'timed', 'stats', 'NativeProjection', '_check_weights',
    '_load_learned', '_phase'))
# Replace only the unused historical CVXPY construction. Native projection has
# zero physical capacity buffer; the baseline network/forward remains intact.
load_into(globals(), CANONICAL, ('Cfg', 'softplus_inverse', 'Model'), changes=((
    '''        x       = cp.Variable(cfg.R)
        z_param = cp.Parameter(cfg.R)
        Q10_p   = cp.Parameter(1)
        cons    = [cp.sum(x) <= Q10_p + 1e-6, x >= 0] # add small tolerance to the sum constraint to avoid numerical issues with SCS
        self.proj_layer = CvxpyLayer(
            cp.Problem(cp.Minimize(cp.sum_squares(x - z_param)), cons),
            parameters=[z_param, Q10_p], variables=[x]
        )''', '        self.proj_layer = NativeProjection()'),))
cfg = Cfg()
cfg.R, cfg.HIDDEN_WIDTH, cfg.NET_SEED = 3, 16, 0
check((cfg.c, cfg.h, cfg.b, cfg.SCS_EPS, cfg.SCS_MAX_ITERS) ==
      (.25, .05, .5, 1e-4, 200000), 'Canonical constants changed')


def configure_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)