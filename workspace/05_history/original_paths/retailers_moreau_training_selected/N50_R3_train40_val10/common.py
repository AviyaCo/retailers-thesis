"""Reuse frozen tested numerical primitives; new protocol never calls old trainers."""
import os
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREAD_ENV)

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OLD = Path('/data/home/aviyac/retailers_moreau/retailers_base/N50_R3/net16_v2_train40_val10')
OLD_PROVENANCE = OLD/'ccg_results/provenance.json'
with OLD_PROVENANCE.open() as stream:
    FROZEN = json.load(stream)['source_sha256']
for path, expected in FROZEN.items():
    with Path(path).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest() != expected:
            raise RuntimeError(f'Historical numerical source changed: {path}')
sys.path.insert(0,str(OLD))
import core
import ccg as legacy_ccg
import adversary
import numpy as np
import torch

PROTOCOL = dict(experiment='retailer_moreau_training_selected_20260916',
    Ntotal=50,Ntrain=40,Nval=10,data_seeds=list(core.SEEDS),network_seed=0,
    split_seed='100000 + data seed',lr_grid=list(core.LR_GRID),radii=list(core.RADII),
    max_epochs=3000,patience=300,improvement_threshold=1e-4,
    scheduler=dict(metric='training_loss',factor=.7,patience=150,threshold=1e-3),
    checkpoint='thresholded training-best evaluated PREUPDATE state including incoming state',
    optimizer='fresh Adam per nominal LR and MP',lr_inventory=.01,gradient_clip_norm=1.,
    nominal_lr_selection='after all three training runs; minimum validation mean, smaller LR tie',
    ccg_selection='terminal fully AP-assessed policy; no intermediate validation selection',
    validation='no validation calls inside training or CCG; terminal policies evaluated after CCG sweep',
    radius_selection='all nine configurations required; minimum terminal validation mean, smaller radius tie',
    max_mp=40,separation_tolerance=.01,membership_tolerance=1e-9,
    ap_replay_tolerance=1e-4,ap_mip_gap=1e-4,capacity_buffer=0.,
    nominal_workers=10,ccg_workers=16,torch_threads=1,gurobi_threads=1,
    test='not accessed',refit=False,resume=False,solver_fallback=False,
    global_robust_optimality_certified=False)


def configure_worker():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_sources(manifest):
    for path, expected in manifest['source_sha256'].items():
        check(core.digest(path)==expected, f'Frozen source changed: {path}')


def raw_hash(data):
    digest=hashlib.sha256()
    for key in ('xi1','xi2'):
        digest.update(np.asarray(data[key],dtype='<f8').tobytes())
    return digest.hexdigest()


def metadata(seed=None,radius=None,method=None):
    return dict(experiment=PROTOCOL['experiment'],method=method,seed=seed,radius=radius,
        Ntotal=50,Ntrain=40,Nval=10,network_seed=0,
        split_seed=None if seed is None else 100000+seed,
        torch_threads=1,gurobi_threads=1,timing_unit='seconds',timing_clock='time.perf_counter')


def evaluate(model,x1,x2,record):
    """Explicit evaluation; training APIs have no validation arguments."""
    state=legacy_ccg.state_hash(model)
    with core.timed(record),torch.no_grad():
        model.eval()
        values=model.per_sample_cost(x1,x2)
        check(values.shape==(len(x1),) and bool(torch.isfinite(values).all()),'Invalid evaluation costs')
        record.update(mean=float(values.mean()),per_sample_costs=values.tolist(),sample_count=len(x1))
        check(legacy_ccg.state_hash(model)==state,'Evaluation changed policy')
        record.update(state_sha256=state,projection=model.proj_layer.drain_summary())
    return record['mean']