"""Frozen green trainer: only the explicit approved train40 guard is extended."""
from collections import Counter
import json
from pathlib import Path
import random
import time
import shipping_core as core
from shipping_core import np, torch
from source_adapter import GREEN, load_into

load_into(globals(), GREEN / 'green_training.py', ('train_model',), changes=(
    ('if n_train not in (160, 640):', 'if n_train not in (40, 160, 640):'),
    ("raise ValueError('Require Ntrain=160 or 640 in caller record')",
     "raise ValueError('Require Ntrain=40, 160 or 640 in caller record')")))