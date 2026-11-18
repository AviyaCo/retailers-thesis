"""Shared R3/R10 protocol and paired training/validation data, never test data.

Sources: retailers_moreau_training_selected/green_R3_N200_N800_20260923/
green_common.py; green_R10_N50_N200_N800_20260924/{green_common,source_adapter}.py;
N50_R3_train40_val10/common.py (raw_hash); black_continuation_R3_R10_20260925/
continuation.py (black stopping/selection). The generator is the canonical
retailers_problem/R3/N50/saainit_net16_v2/run_all_net16_v2.py generate_data.
No historical source is imported or executed.
"""
from dataclasses import dataclass
import hashlib
import math
from numbers import Real
import os
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from scipy.stats import truncnorm
import torch


MOREAU_DEPS = Path(os.environ.get('RETAILERS_MOREAU_DEPS',
    str(Path(__file__).resolve().parents[1] /
        'archive/hydro_rcr_rescaled/.moreau_diagnostic_deps'))).resolve()
cfg = SimpleNamespace(c=.25, h=.05, b=.5)
SIZES = (50, 200, 800)
SEEDS = tuple(range(10))
LR_GRID = (.005, .01, .05)
RADII = (0., .001, .003, .01, .03, .2, .4, .7, 1.5)
MAX_MP = 40
MAX_EPOCHS = 3000
PATIENCE = 300


@dataclass(frozen=True)
class Config:
    R: int
    Ntotal: int
    seed: int
    radius: float = 0.
    method: str = 'green'

    LR_GRID: ClassVar[tuple] = LR_GRID
    MAX_MP: ClassVar[int] = MAX_MP
    MAX_EPOCHS: ClassVar[int] = MAX_EPOCHS
    PATIENCE: ClassVar[int] = PATIENCE

    def __post_init__(self):
        if type(self.R) is not int or self.R not in (3, 10):
            raise ValueError('R must be 3 or 10')
        if type(self.Ntotal) is not int or self.Ntotal not in SIZES:
            raise ValueError('Ntotal must be 50, 200, or 800')
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise ValueError('seed must be an integer in 0..9')
        if (isinstance(self.radius, bool) or not isinstance(self.radius, Real)
                or not math.isfinite(self.radius)
                or self.radius not in RADII + ((2.,) if self.R == 10 else ())):
            raise ValueError('Unapproved radius for R')
        if self.method not in ('black', 'green'):
            raise ValueError('method must be black or green')

    @property
    def Ntrain(self):
        return 4 * self.Ntotal // 5

    @property
    def Nval(self):
        return self.Ntotal // 5

    def protocol(self):
        """Fresh mapping: common arithmetic, explicit method-dependent CCG rule.

        Black describes its stopping/return rule, not an implicit continuation
        or permission to resume. Artifact origin/resume metadata belongs to the
        caller; nothing here loads a cached policy or accesses held-out test data.
        """
        green = self.method == 'green'
        return dict(experiment='retailers_experiment', method=self.method,
            R=self.R, f=0, Ntotal=self.Ntotal, Ntrain=self.Ntrain, Nval=self.Nval,
            seed=self.seed, radius=self.radius, sizes=list(SIZES),
            train_fraction=.8, validation_fraction=.2, data_seeds=list(SEEDS),
            network_seed=0, split_seed=100000 + self.seed,
            lr_grid=list(LR_GRID), radii=list(RADII + ((2.,) if self.R == 10 else ())),
            hidden_width=16, c=cfg.c, h=cfg.h, b=cfg.b,
            raw_dtype='float64', tensor_dtype='float32', batch='full training set',
            generator='legacy scipy truncnorm; global numpy seed saved/restored',
            max_epochs=MAX_EPOCHS, patience=PATIENCE, improvement_threshold=1e-4,
            scheduler=dict(metric='training_loss', factor=.7, patience=150, threshold=1e-3),
            checkpoint='thresholded training-best evaluated PREUPDATE state including incoming state',
            optimizer='fresh Adam per nominal LR and MP', lr_inventory=.01,
            gradient_clip_norm=1.,
            nominal_lr_selection='after all three training runs; minimum validation mean, smaller LR tie',
            ccg_stop=('first abs(APmean-poolmean)/max(abs(poolmean),1e-9) < .01 including candidate0'
                      if green else 'all max(0,bound-pool)/max(1,abs(bound),abs(pool)) <= .01'),
            ccg_selection='stopping checkpoint after inclusive assessed-prefix validation; no earlier AP minimum',
            green_gap_tolerance=.01, green_gap_floor=1e-9,
            validation='no validation calls inside training or CCG; only after complete CCG sweep',
            radius_selection='all approved converged configurations required; minimum validation mean, smaller radius tie',
            max_mp=MAX_MP, separation_tolerance=.01, membership_tolerance=1e-9,
            insertion_tolerance=1e-4, pool_replay_tolerance=1e-6,
            ap_replay_tolerance=1e-4, ap_mip_gap=1e-4, capacity_buffer=0.,
            physical_projection_tolerance=1e-6,
            nominal_workers='manifest.workers', ccg_workers='manifest.workers',
            torch_threads=1, gurobi_threads=1, timing_unit='seconds',
            timing_clock='time.perf_counter', test='not accessed', refit=False,
            resume=False, solver_fallback=False, global_robust_optimality_certified=False)


def split_data(config):
    """Exact paired float64 generator/split, preserving NumPy's global RNG.

    xi2 intentionally has NO size argument, matching SciPy's original broadcast
    call. data_sha256 is exactly common.raw_hash: xi1 then xi2, little-endian f8,
    C-order bytes without shape/name headers. Both methods use identical data.
    """
    if not isinstance(config, Config):
        raise TypeError('split_data requires Config')
    rng = np.random.get_state()
    try:
        np.random.seed(config.seed)
        xi1 = truncnorm.rvs((0. - 6.) / 2.5, np.inf, loc=6., scale=2.5,
                            size=(config.Ntotal, config.R))
        mu2 = 2 * (6. - xi1) ** 2
        xi2 = truncnorm.rvs((0. - mu2) / 2.5, np.inf, loc=mu2, scale=2.5)
    finally:
        np.random.set_state(rng)
    if xi1.shape != xi2.shape or xi1.shape != (config.Ntotal, config.R):
        raise ValueError('Invalid paired sample shapes')
    indices = np.random.default_rng(100000 + config.seed).permutation(config.Ntotal)
    train_indices, val_indices = indices[:config.Ntrain], indices[config.Ntrain:]
    hasher = hashlib.sha256()
    for value in (xi1, xi2):
        hasher.update(np.asarray(value, dtype='<f8').tobytes())
    return dict(xi1=xi1, xi2=xi2, train_indices=train_indices, val_indices=val_indices,
        train_x1=torch.tensor(xi1[train_indices], dtype=torch.float32),
        train_x2=torch.tensor(xi2[train_indices], dtype=torch.float32),
        val_x1=torch.tensor(xi1[val_indices], dtype=torch.float32),
        val_x2=torch.tensor(xi2[val_indices], dtype=torch.float32),
        R=config.R, Ntotal=config.Ntotal, seed=config.seed,
        split_seed=100000 + config.seed, data_sha256=hasher.hexdigest())