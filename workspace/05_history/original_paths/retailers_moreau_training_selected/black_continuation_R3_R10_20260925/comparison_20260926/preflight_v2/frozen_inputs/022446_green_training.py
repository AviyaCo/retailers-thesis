"""Exact frozen full-batch preupdate trainer, adding only approved train40."""
from source_adapter import GREEN, load_into
load_into(globals(), GREEN / 'green_training.py', [
    ('n_train not in (160, 640)', 'n_train not in (40, 160, 640)'),
    ('Require Ntrain=160 or 640', 'Require Ntrain=40, 160 or 640'),
])