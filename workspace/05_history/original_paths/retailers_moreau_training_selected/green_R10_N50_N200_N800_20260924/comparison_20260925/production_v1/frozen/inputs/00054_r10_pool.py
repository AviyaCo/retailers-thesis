"""Only frozen pool/hash/AP-gate definitions; no historical controller loaded."""
from source_adapter import OLD, load_into, forbidden
import hashlib
import numpy as np
import torch
import r10_core as core
import r10_adversary as adversary

SEPARATION_TOLERANCE = .01
MEMBERSHIP_TOLERANCE = 1e-9
load_into(globals(), OLD / 'ccg.py', [
    ('points.shape[1] != 6', 'points.shape[1] != 20'),
    ('paired6 vectors', 'paired20 vectors'),
    ('points[:, :3]', 'points[:, :10]'),
    ('points[:, 3:]', 'points[:, 10:]'),
    ('np.shape(x1) != (3,) or np.shape(x2) != (3,)',
     'np.shape(x1) != (10,) or np.shape(x2) != (10,)'),
], names=('state_hash', 'pool_hash', 'check_membership', 'point_membership',
          'restricted_objective', '_finite', 'validate_ap'))
load_nominal = assess_candidate = run_configuration = forbidden