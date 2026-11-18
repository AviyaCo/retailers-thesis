"""Artifact primitives transcribed from retailers_moreau/retailers_base/
N50_R3/net16_v2_train40_val10/core.py (no historical module imports).

Nonfinite values are diagnostic strings, never replacement numerical values.
Timing contexts retain caller metadata and surface failures without recovery.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

import numpy as np
import torch


# Computational entry points only: editing tests or documentation does not
# invalidate a run, while every current shared runtime module is bound.
EXECUTION_SOURCE_FILES = (
    'run.py', 'archive_paths.py', 'src/__init__.py', 'src/ap.py', 'src/artifacts.py',
    'src/ccg.py', 'src/config.py', 'src/evaluate.py', 'src/plot.py',
    'src/policy.py', 'src/select_lr.py', 'src/training.py',
)


def execution_identity():
    """JSON-safe code/environment identity for nominal and native CCG artifacts.

    No solver calls, historical imports, timestamps, or test-file hashes. Call
    after configuring worker threads. Missing distribution metadata is an error,
    not an unknown-version fallback. Moreau is the module actually used by policy.
    """
    from . import policy
    from .config import MOREAU_DEPS

    root = Path(__file__).resolve().parent.parent
    packages = {name: metadata.version(name) for name in (
        'numpy', 'scipy', 'torch', 'gurobipy', 'cvxpy', 'cvxpylayers',
        'diffcp', 'scs', 'clarabel', 'pydantic', 'moreau', 'moreau-cpu',
    )}
    moreau_path = Path(policy.moreau.__file__).resolve()
    distribution = metadata.distribution('moreau')
    if (not moreau_path.is_relative_to(MOREAU_DEPS)
            or moreau_path != Path(distribution.locate_file('moreau/__init__.py')).resolve()
            or packages['moreau'] != policy.moreau.__version__):
        raise ValueError('Moreau execution module/distribution identity mismatch')
    return dict(format='retailers-execution-identity-v1',
        sources={name: digest(root / name) for name in EXECUTION_SOURCE_FILES},
        packages=packages,
        python=dict(version=platform.python_version(), implementation=platform.python_implementation()),
        platform=platform.platform(), machine=platform.machine(),
        moreau=dict(module_path=str(moreau_path), module_sha256=digest(moreau_path)),
        runtime=dict(torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
            torch_default_dtype=str(torch.get_default_dtype()),
            thread_environment={key: os.environ.get(key) for key in (
                'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
            gurobi_threads=1))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return _json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_json(path, obj):
    """Strict JSON, fsynced and atomically replaced in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(_json_value(obj), stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextmanager
def timed(record):
    start = time.perf_counter()
    record.update(started_at_utc=utc_now(), status='running',
                  timing_unit='seconds', timing_clock='time.perf_counter')
    try:
        yield record
    except BaseException as error:
        record.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    else:
        record['status'] = 'completed'
    finally:
        record.update(ended_at_utc=utc_now(), wall_time_s=time.perf_counter() - start)


def stats(values):
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise FloatingPointError('Nonfinite input to timing statistics')
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,
                min=min(values) if values else None, max=max(values) if values else None)


@contextmanager
def _phase(record, epoch_record, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        epoch_record[name] = epoch_record.get(name, 0.) + elapsed
        record[name] = record.get(name, 0.) + elapsed