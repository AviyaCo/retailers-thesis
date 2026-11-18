"""Whitelisted strict QP definitions only; no OOS generator/evaluator imported."""
from source_adapter import COMPARE, load_into
import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
from torch import nn

PHYSICAL_TOL = 1e-4
load_into(globals(), COMPARE, [
    ('nn.Linear(7, 16)', 'nn.Linear(21, 16)'),
    ('nn.Linear(16, 3)', 'nn.Linear(16, 10)'),
    ('torch.zeros(3)', 'torch.zeros(10)'),
    ('np.shape(z) != (3,)', 'np.shape(z) != (10,)'),
    ('m.addVars(3,', 'm.addVars(10,'),
    ('for r in range(3)', 'for r in range(10)'),
], names=('Model', 'strict_project', 'scenario_costs'))

def load_model(path, method):
    if method != 'new':
        raise ValueError('Only this campaign learned-only checkpoints are accepted')
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = Model()
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError('Unexpected learned schema')
    for k, value in state.items():
        if value.dtype != torch.float32 or value.shape != expected[k].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid learned tensor: {k}')
    model.load_state_dict(state, strict=True)
    return model.eval()