"""Disposable real R3/f=0 preflight; never launch or reuse production training.

Parent CLI: python -B -u preflight.py OUTPUTDIR
Child CLI:  python -B -u preflight.py --size 200 --outputdir OUTPUTDIR
Both require a nonexistent output directory. The parent runs test_green once
in a subprocess, then independent, sequential N200 and N800 child processes.
It does NOT choose production concurrency. summary.json contains status,
started_at_utc, ended_at_utc, wall_time_s, sources_sha256 (three green modules
and test_green), tests, phases, environment snapshots, and sizes. Each size
contains Ntotal, peak_rss_bytes, timings, solver diagnostics and all phases.

Two epochs are an explicitly disposable integration/resource exercise, NOT a
nominal fit, LR search, production MP budget or performance estimate. The live
radius-zero controller receives this labeled checkpoint only inside this
preflight directory. Constructed maximum-pool points are NOT AP solutions or
research data. No validation values or test data are evaluated. Unit-test
fixtures run in a separate process, so their global mocks cannot leak here.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
SOURCE_NAMES = ('green_common.py', 'green_training.py', 'green_ccg.py', 'test_green.py',
                'run_campaign.py', 'test_campaign.py', 'preflight.py', 'launch.sh', 'README.md')
THREAD_ENV = {name: '1' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS',
    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
os.environ.update(THREAD_ENV)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
_json_value = lambda value: value


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sources():
    return {str(ROOT / name): digest(ROOT / name) for name in SOURCE_NAMES}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(_json_value(value), stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def peak_rss_bytes():
    # Linux ru_maxrss is KiB and includes this process's import/startup peak.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def environment():
    """Raw resource evidence, not a worker-budget recommendation or guarantee."""
    meminfo = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            meminfo[key + '_bytes'] = int(value.split()[0]) * 1024
    cpuinfo = Path('/proc/cpuinfo').read_text().splitlines()
    cgroup_text = Path('/proc/self/cgroup').read_text()
    mountinfo = Path('/proc/self/mountinfo').read_text()
    groups = [line.split(':', 2) for line in cgroup_text.splitlines()]
    directories = {Path('/sys/fs/cgroup')}
    # Include mounted hierarchy roots AND this process's ancestors: quotas and
    # memory ceilings may be imposed above the leaf. Preserve values verbatim.
    for line in mountinfo.splitlines():
        left, right = line.split(' - ', 1)
        fields, filesystem = left.split(), right.split()
        if filesystem[0] not in ('cgroup', 'cgroup2'):
            continue
        mount_root, mountpoint = Path(fields[3]), Path(fields[4])
        directories.add(mountpoint)
        for _, controllers, group in groups:
            if filesystem[0] == 'cgroup' and not set(controllers.split(',')) & set(filesystem[2].split(',')):
                continue
            if filesystem[0] == 'cgroup2' and controllers:
                continue
            group_path = Path(os.path.normpath(group))
            if not group_path.is_relative_to(mount_root):
                continue
            leaf = mountpoint / group_path.relative_to(mount_root)
            while leaf.is_relative_to(mountpoint):
                directories.add(leaf)
                if leaf == mountpoint:
                    break
                leaf = leaf.parent
    names = ('cpu.max', 'cpu.stat', 'cpuset.cpus.effective', 'memory.max',
        'memory.current', 'memory.peak', 'memory.events', 'pids.max', 'pids.current',
        'cpu.cfs_quota_us', 'cpu.cfs_period_us', 'memory.limit_in_bytes',
        'memory.usage_in_bytes', 'cpuset.cpus')
    cgroups = {}
    for directory in sorted(directories):
        for name in names:
            path = directory / name
            if path.is_file():
                cgroups[str(path)] = path.read_text().strip()
    return dict(at_utc=utc(), pid=os.getpid(), executable=sys.executable,
        python=sys.version, platform=platform.platform(), hostname=platform.node(),
        cpu_count=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
        loadavg=list(os.getloadavg()), memory=meminfo,
        cpu_model=next((line.split(':', 1)[1].strip() for line in cpuinfo
                        if line.startswith('model name')), None),
        cgroup_membership=cgroup_text, cgroup_mountinfo=mountinfo,
        cgroup_files=cgroups, thread_environment={k: os.environ.get(k) for k in THREAD_ENV},
        peak_rss_bytes=peak_rss_bytes(), worker_selection='deferred to parent campaign',
        memory_scope='observed preflight high-water RSS; NOT a future AP MIP memory bound')


class Evidence:
    def __init__(self, folder, size):
        self.folder = folder
        self.expected = None
        self.manifest = None
        self.verify_manifest = None
        self.record = dict(experiment='disposable_R3_f0_green_preflight',
            status='running', started_at_utc=utc(), Ntotal=size, seed=0,
            network_seed=0, split_seed=100000, lr_net=.01,
            torch_threads=1, gurobi_threads=1, workers=1,
            preflight_only=True, production_reuse_forbidden=True,
            heldout_value_evaluation=False, test_data_access=False,
            timing_clock='time.perf_counter', timing_unit='seconds',
            timing_scope='current invocation; no historical training time substitution',
            phases={}, source_checks=[], sizes=[], tests={'status': 'not_run'})

    def save(self):
        write_json(self.folder / 'summary.json', self.record)

    def verify(self, label):
        current = sources()
        row = dict(label=label, at_utc=utc(), sources_sha256=current,
                   status='completed' if current == self.expected else 'failed')
        self.record['source_checks'].append(row)
        require(current == self.expected, 'Green computational/test source changed during preflight')
        if self.manifest is not None:
            self.verify_manifest(self.manifest)

    @contextmanager
    def phase(self, name, **metadata):
        row = self.record['phases'][name] = dict(status='running', started_at_utc=utc(), **metadata)
        start = time.perf_counter()
        self.save()
        try:
            self.verify('before_' + name)
            yield row
            self.verify('after_' + name)
        except BaseException as error:
            row.update(status='failed', error=dict(type=type(error).__name__, message=str(error)),
                       traceback=traceback.format_exc())
            raise
        else:
            row['status'] = 'completed'
        finally:
            row.update(ended_at_utc=utc(), wall_time_s=time.perf_counter() - start,
                       peak_rss_bytes=peak_rss_bytes())
            self.save()


def subprocess_phase(evidence, name, command):
    """Persist separate raw streams even when the child fails or is killed."""
    with evidence.phase(name, command=command, cwd=str(ROOT), subprocess_workers=1) as row:
        stdout = evidence.folder / (name + '.stdout.log')
        stderr = evidence.folder / (name + '.stderr.log')
        row.update(stdout_path=str(stdout), stderr_path=str(stderr), returncode=None)
        with stdout.open('x') as out, stderr.open('x') as err:
            result = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err,
                                    env={**os.environ, **THREAD_ENV}, check=False)
        row['returncode'] = result.returncode
        row.update(stdout_sha256=digest(stdout), stderr_sha256=digest(stderr))
        require(result.returncode == 0, f'{name} exited {result.returncode}; see raw logs')
    return row


def parent(evidence):
    record = evidence.record
    with evidence.phase('environment_before') as row:
        row['snapshot'] = record['environment_before'] = environment()
    with evidence.phase('ast') as row:
        row['parsed'] = []
        for path in [ROOT / name for name in SOURCE_NAMES]:
            if path.suffix == '.py':
                ast.parse(path.read_text(), filename=str(path))
                row['parsed'].append(str(path))
    snapshot = evidence.folder / 'sources.json'
    write_json(snapshot, evidence.expected)
    try:
        subprocess_phase(evidence, 'tests',
            [sys.executable, '-B', '-u', '-m', 'unittest', '-v', 'test_green', 'test_campaign'])
    finally:
        record['tests'] = record['phases'].get('tests', {'status': 'not_run'})
        evidence.save()
    for size in (200, 800):
        folder = evidence.folder / f'N{size}'
        name = f'size_{size}'
        try:
            subprocess_phase(evidence, name, [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                '--size', str(size), '--outputdir', str(folder), '--expected-sources', str(snapshot)])
        finally:
            path = folder / 'summary.json'
            if path.exists():
                child = json.loads(path.read_text())
                child.update(summary_path=str(path), summary_sha256=digest(path),
                             process=record['phases'][name])
            else:
                child = dict(Ntotal=size, status='failed', peak_rss_bytes=None,
                    wall_time_s=None, error='Child summary unavailable; inspect raw process logs',
                    process=record['phases'][name])
            record['sizes'].append(child)
            evidence.save()
        require(child['status'] == 'completed', f'N{size} did not complete')
        require(child['sources_sha256'] == evidence.expected, 'Child source snapshot mismatch')
    with evidence.phase('environment_after') as row:
        row['snapshot'] = record['environment_after'] = environment()
    record['resource_decision'] = dict(workers=None,
        reason='Campaign parent selects budget using CPU/load/cgroup memory and per-size peak RSS',
        preflight_size_concurrency=1, future_mip_memory_bounded=False)


def child(evidence, size):
    global _json_value
    record, folder = evidence.record, evidence.folder
    record.update(Ntrain=4 * size // 5, Nval=size // 5,
                  tests=dict(status='not_run_in_child', scope='parent subprocess only'))
    with evidence.phase('startup') as startup:
        # Historical common inserts its OLD root; use unique green names, then
        # restore this directory's precedence. Never import plain common/train.
        sys.path.insert(0, str(ROOT))
        import green_common as common
        sys.path.insert(0, str(ROOT))
        import green_ccg as ccg
        import green_training as training
        for module in (common, ccg, training):
            require(Path(module.__file__).resolve().parent == ROOT, 'Green import collision')
        core, np, torch = common.core, common.np, common.torch
        _json_value = core._json_value
        common.configure_worker()  # Exactly once in each fresh numerical child.
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 1,
                'Expected single-threaded Torch worker')
        require(core.baseline.cfg.R == 3, 'Expected canonical R3 model')
        compare_path = Path(core.__file__).resolve().parent / 'matched_oos_comparison/compare.py'
        complete_sources = {**common.FROZEN, **evidence.expected,
                            str(compare_path): digest(compare_path),
                            str(Path(__file__).resolve()): digest(Path(__file__).resolve())}
        manifest = dict(source_sha256=complete_sources, protocol=common.PROTOCOL, workers=1,
            preflight_only=True, production_reuse_forbidden=True,
            nominal_origin='disposable preflight two epochs at LR .01; NOT nominal LR-search winner')
        common.verify_sources(manifest)
        evidence.manifest, evidence.verify_manifest = manifest, common.verify_sources
        manifest_path = folder / 'preflight_manifest.json'
        core.write_json(manifest_path, manifest)
        record.update(manifest_path=str(manifest_path), manifest_sha256=digest(manifest_path))
        startup['versions'] = dict(numpy=np.__version__, torch=torch.__version__,
                                   moreau=core.moreau.__version__, gurobi=list(core.gp.gurobi.version()))
        startup['environment'] = environment()
        record['environment_before'] = startup['environment']
    with evidence.phase('ast') as row:
        row['parsed'] = []
        for path in complete_sources:
            if Path(path).suffix == '.py':
                ast.parse(Path(path).read_text(), filename=path)
                row['parsed'].append(path)
    with evidence.phase('split_seed0_train_only') as row:
        data = common.split_data(size, 0)
        ids, val_ids = data['train_indices'], data['val_indices']
        ntrain = 4 * size // 5
        require(len(ids) == ntrain and len(val_ids) == size // 5
                and not set(ids) & set(val_ids)
                and set(ids) | set(val_ids) == set(range(size)), 'Invalid disjoint split')
        for name in ('xi1', 'xi2'):
            require(data[name].shape == (size, 3) and data[name].dtype == np.float64,
                    'Invalid raw shape/dtype')
        for prefix, count in (('train', ntrain), ('val', size // 5)):
            for index in (1, 2):
                tensor = data[f'{prefix}_x{index}']
                require(tensor.shape == (count, 3) and tensor.dtype == torch.float32,
                        'Invalid tensor shape/dtype')
        # Slice train values only; do not hash/inspect held-out values.
        centers = np.concatenate((data['xi1'][ids], data['xi2'][ids]), axis=1)
        train1, train2 = data['train_x1'], data['train_x2']
        original_ids = ids.tolist()
        del tensor, data, ids, val_ids
        require(np.isfinite(centers).all() and (centers >= 0).all(), 'Invalid training centers')
        train_hash = hashlib.sha256(centers.astype('<f8').tobytes()
                                    + np.asarray(original_ids, dtype='<i8').tobytes()).hexdigest()
        row.update(Ntrain=ntrain, Nval=size // 5, original_sample_ids=original_ids,
                   data_sha256=train_hash, hash_scope='train centers and IDs ONLY; no held-out values')
        train_path = folder / 'training_data.json'
        core.write_json(train_path, dict(centers=centers, original_sample_ids=original_ids,
                                        data_sha256=train_hash))
    with evidence.phase('saa_train_only') as row:
        solver_record = row['solver'] = common.metadata(size, 0, 0., 'preflight_train_only_dynamic_LP')
        q10, q1r = common.solve_saa(centers[:, :3], centers[:, 3:], solver_record)
    with evidence.phase('native_model_initialization') as row:
        model = core.make_model(q10, q1r)
        require(isinstance(model.proj_layer, core.NativeProjection), 'Expected native projection')
        row.update(state_sha256=ccg.state_hash(model), projection_settings=model.proj_layer.settings)
    with evidence.phase('disposable_nominal_two_epochs', radius=0., epochs=2, lr_net=.01) as row:
        nominal_training = row['training'] = common.metadata(size, 0, 0., 'preflight_nominal_2epochs_NOT_fit')
        nominal_training.update(preflight_only=True, production_reuse_forbidden=True)
        model = training.train_model(model, lambda m: m.per_sample_cost(train1, train2).mean(),
            .01, folder / 'disposable_nominal', nominal_training, epochs=2, patience=300)
        require(nominal_training['optimizer_steps'] == nominal_training['epochs_evaluated'] == 2,
                'Disposable nominal must perform exactly two real Adam epochs')
        require(nominal_training['validation_time_s'] == 0., 'Unexpected held-out evaluation')
        row['finite_gradient_norms'] = [entry['gradient_norm_preclip']
                                      for entry in nominal_training['epoch_history']]
        require(np.isfinite(row['finite_gradient_norms']).all(), 'Nonfinite nominal gradients')
    pools = [center[None, :].copy() for center in centers]
    with evidence.phase('restored_nominal_replay') as row:
        values = ccg._pool_mean(model, pools, row.setdefault('evaluation', {}))
        train_mean = float(values.mean())
        row['absolute_discrepancy'] = abs(train_mean - nominal_training['selected_training_loss'])
        require(row['absolute_discrepancy'] <= 1e-6, 'Restored nominal replay mismatch')
        nominal_checkpoint = folder / 'disposable_nominal/selected.pth'
        saved = torch.load(nominal_checkpoint, map_location='cpu', weights_only=True)
        require(set(saved) == set(core.weights(model))
                and all(torch.equal(saved[k], v) for k, v in core.weights(model).items()),
                'Selected checkpoint differs from restored nominal')
        nominal_state_hash = ccg.state_hash(model)
    cache_id = id(model.proj_layer._solvers[ntrain])
    record['batch_lifecycle'] = [dict(stage='nominal', batch=ntrain,
                                    cached_sizes=sorted(model.proj_layer._solvers))]
    for radius in (0., .2):
        with evidence.phase(f'real_AP_center0_rho{radius:g}', radius=radius, sample_id=0) as row:
            ap = row['ap'] = common.metadata(size, 0, radius, 'preflight_real_AP')
            ap.update(sample_id=0, original_sample_id=original_ids[0], candidate_id=0)
            result = common.adversary.solve_ap(model, centers[0, :3], centers[0, 3:], radius, ap)
            point = ccg.validate_ap(result, ap, 0, centers[0], pools[0], radius, float(values[0]))
            native_pool = [np.vstack((centers[0], point))]
            with torch.no_grad():
                pool_cost = float(ccg.restricted_objective(native_pool)(model)[0])
            row.update(native_enlarged_pool_cost=pool_cost, point=point.tolist())
            require(np.isfinite(pool_cost) and pool_cost >= float(values[0]) - 1e-6,
                    'Native AP pool replay lost original center')
            require(ccg.state_hash(model) == nominal_state_hash, 'AP changed nominal weights')
    record['batch_lifecycle'].append(dict(stage='AP_replay', batch=1,
                                        cached_sizes=sorted(model.proj_layer._solvers)))
    with evidence.phase('strict_projection_training_row0', sample_id=0) as row:
        # Safe module import: compare's launch code is behind __main__. Never
        # call gen/evaluate_job/freeze or import the canonical launch script.
        spec = importlib.util.spec_from_file_location('_green_preflight_strict_compare', compare_path)
        require(spec is not None and spec.loader is not None, 'Cannot load strict projection definitions')
        compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compare)
        reference = compare.load_model(nominal_checkpoint, 'new')
        z, inventory, store_inventory = reference.targets(centers[:1, :3])
        captured = {}
        def capture_target(module, inputs, output):
            captured['target'] = output.detach().cpu().numpy().copy()
        def capture_projection(module, inputs, output):
            captured['projection'] = output[0].detach().cpu().numpy().copy()
        handles = [model.fc_layers.register_forward_hook(capture_target),
                   model.proj_layer.register_forward_hook(capture_projection)]
        try:
            with torch.no_grad():
                native_cost = float(model.per_sample_cost(train1[:1], train2[:1])[0])
        finally:
            for handle in handles:
                handle.remove()
        with core.gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('Threads', 1)
            env.start()
            with core.gp.Model(env=env) as settings:
                row['gurobi_settings'] = {key: settings.getParamInfo(key)[2] for key in
                    ('FeasibilityTol', 'OptimalityTol', 'BarConvTol', 'Method', 'Threads')}
            projection = compare.strict_project(z[0], inventory, env, row.setdefault('solver', {}))
        strict_cost = float(compare.scenario_costs(inventory, store_inventory, projection[None],
                                                  centers[:1, :3], centers[:1, 3:])[0])
        comparisons = dict(target=(z, captured['target']),
                           projection=(projection, captured['projection'][0]),
                           cost=(np.asarray(strict_cost), np.asarray(native_cost)))
        row.update(tolerance=common.adversary.REPLAY_TOLERANCE,
                   tolerance_rule='abs max discrepancy / max(1, max abs reference, max abs native)',
                   precision='float32 native network/input vs float64 strict QP/raw training cost',
                   physical_tolerance=compare.PHYSICAL_TOL, comparisons={})
        for name, (expected, actual) in comparisons.items():
            discrepancy = float(np.max(np.abs(expected - actual)))
            scaled = discrepancy / max(1., float(np.max(np.abs(expected))), float(np.max(np.abs(actual))))
            row['comparisons'][name] = dict(reference=np.asarray(expected).tolist(),
                native=np.asarray(actual).tolist(), absolute_discrepancy=discrepancy,
                floor_scaled_discrepancy=scaled)
            require(np.isfinite(scaled) and scaled <= common.adversary.REPLAY_TOLERANCE,
                    f'Strict/native {name} discrepancy exceeds unchanged replay tolerance')
        require(ccg.state_hash(model) == nominal_state_hash, 'Projection comparison changed policy')
    with evidence.phase('maximum_pool_construction', radius=1.5) as row:
        # Deliberately constructed, not optimized adversarial points. Offsets
        # are nonnegative and lie within each paired6 radius-1.5 box.
        offsets = np.linspace(0., 1.5, common.MAX_MP + 1, dtype=np.float64)
        stress_pools = [center[None, :] + offsets[:, None] for center in centers]
        require(common.MAX_MP + 1 == 41, 'Expected maximum 41 points per center')
        for center, pool in zip(centers, stress_pools):
            ccg.check_membership(pool, center, 1.5)
            require(np.unique(pool, axis=0).shape == (41, 6) and (pool >= 0).all(),
                    'Stress points must be distinct and nonnegative')
        stress_hash = ccg.pool_hash(stress_pools)
        row.update(fixture='CONSTRUCTED forward/backward resource fixture; NOT AP points/research data',
                   construction='center + linspace(0, 1.5, 41) in all six coordinates',
                   points_per_center=41, total_points=ntrain * 41, pool_sha256=stress_hash)
        stress_objective = ccg.restricted_objective(stress_pools)
    with evidence.phase('maximum_pool_two_MP_epochs', radius=1.5, epochs=2, lr_net=.01) as row:
        stress_training = row['training'] = common.metadata(size, 0, 1.5, 'preflight_constructed_pool_2_MP_epochs')
        stress_training.update(preflight_only=True, production_reuse_forbidden=True,
                               pool_sha256=stress_hash, total_points=ntrain * 41)
        model = training.train_model(model, lambda m: stress_objective(m).mean(), .01,
            folder / 'disposable_stress_MP', stress_training, epochs=2, patience=300)
        require(stress_training['optimizer_steps'] == stress_training['epochs_evaluated'] == 2,
                'Maximum-pool fixture must perform two actual Adam MP epochs')
        norms = [entry['gradient_norm_preclip'] for entry in stress_training['epoch_history']]
        row['finite_gradient_norms'] = norms
        require(np.isfinite(norms).all(), 'Nonfinite stress gradients')
    record['batch_lifecycle'].append(dict(stage='maximum_pool', batch=ntrain * 41,
                                        cached_sizes=sorted(model.proj_layer._solvers)))
    with evidence.phase('maximum_pool_selected_replay', radius=1.5) as row:
        with torch.no_grad():
            replay = float(stress_objective(model).mean())
        row.update(mean=replay, absolute_discrepancy=abs(replay - stress_training['selected_training_loss']),
                   pool_sha256=ccg.pool_hash(stress_pools), selected_state_sha256=ccg.state_hash(model))
        require(np.isfinite(replay) and row['absolute_discrepancy'] <= 1e-6,
                'Maximum-pool restored checkpoint replay mismatch')
        require(row['pool_sha256'] == stress_hash, 'Stress pool changed')
        saved = torch.load(folder / 'disposable_stress_MP/selected.pth', map_location='cpu', weights_only=True)
        require(set(saved) == set(core.weights(model))
                and all(torch.equal(saved[k], v) for k, v in core.weights(model).items()),
                'Stress checkpoint differs from restored selected policy')
    with evidence.phase('return_to_Ntrain_forward_backward') as row:
        frozen = ccg.state_hash(model)
        model.zero_grad(set_to_none=True)
        loss = model.per_sample_cost(train1, train2).mean()
        require(bool(torch.isfinite(loss)), 'Nonfinite return-to-Ntrain loss')
        loss.backward()
        gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
        require(all(gradient is not None and bool(torch.isfinite(gradient).all())
                    for gradient in gradients.values()), 'Missing/nonfinite lifecycle gradient')
        require(id(model.proj_layer._solvers[ntrain]) == cache_id, 'Ntrain solver cache was replaced')
        require({ntrain, 1, ntrain * 41} <= set(model.proj_layer._solvers), 'Missing batch-specific caches')
        require(ccg.state_hash(model) == frozen, 'Gradient-only lifecycle check changed weights')
        row.update(loss=float(loss.detach()), finite_gradient_parameters=list(gradients),
                   cached_sizes=sorted(model.proj_layer._solvers),
                   reused_original_Ntrain_solver=True, projection=model.proj_layer.drain_summary())
    record['batch_lifecycle'].append(dict(stage='return_Ntrain', batch=ntrain,
                                        cached_sizes=sorted(model.proj_layer._solvers)))
    with evidence.phase('live_green_zero_radius_all_centers', radius=0., sample_count=ntrain) as row:
        nominal = dict(Ntotal=size, seed=0, lr_net=.01, checkpoint=str(nominal_checkpoint),
            checkpoint_sha256=digest(nominal_checkpoint), state_sha256=nominal_state_hash,
            train_mean=train_mean, data_sha256=train_hash, training_data_path=str(train_path),
            training_data_sha256=digest(train_path), training_timing=nominal_training,
            preflight_only=True, production_reuse_forbidden=True,
            origin='disposable preflight nominal two epochs; NOT production nominal fit')
        core.write_json(folder / 'disposable_nominal_record.json', nominal)
        result = ccg.run_configuration(size, 0, 0., folder / 'disposable_live_green', nominal, manifest_path)
        row['result'] = result
        require(result['status'] == 'completed' and result['converged'],
                f'Live zero-radius controller failed: {result}')
        require(result['mp_count'] == 0 and result['ap_sweeps_completed'] == 1
                and result['stopping_candidate_id'] == result['selected_candidate_id'] == 0,
                'Expected real initial-pass zero-MP candidate0 publication')
        summary = json.loads(Path(result['summary_path']).read_text())
        require(len(summary['iterations']) == 1, 'Unexpected live controller iterations')
        candidate = summary['iterations'][0]
        require(len(candidate['samples']) == ntrain
                and [sample['sample_id'] for sample in candidate['samples']] == list(range(ntrain))
                and all(sample['controller_status'] == 'completed' and sample['solver_status'] == 2
                        for sample in candidate['samples']), 'Incomplete real AP sweep')
        require('mp' not in candidate and 'added_sample_ids' not in candidate,
                'Initial green pass performed pool insertion/MP')
        persisted = json.loads(Path(candidate['ap_reference']).read_text())
        require(persisted['assessed'] and persisted['candidate_id'] == 0 and persisted['green_pass'],
                'Candidate0 assessment was not persisted')
        require(result['selected_state_sha256'] == nominal_state_hash
                and digest(result['selected_checkpoint']) == digest(nominal_checkpoint),
                'Published zero-MP candidate differs from disposable nominal')
        row.update(real_AP_count=ntrain, candidate0_persisted=True,
                   prior_disposable_nominal_time_s=nominal_training['wall_time_s'],
                   prior_nominal_time_excluded_from_CCG_total=True)
    with evidence.phase('environment_after') as row:
        row['snapshot'] = record['environment_after'] = environment()
    record.update(maximum_pool_points=ntrain * 41, resource_workers_recommendation=None,
                  future_AP_MIP_memory_bounded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outputdir', nargs='?', type=Path)
    parser.add_argument('--size', type=int, choices=(200, 800))
    parser.add_argument('--outputdir', dest='child_outputdir', type=Path)
    parser.add_argument('--expected-sources', type=Path)
    args = parser.parse_args()
    if args.size is None:
        if args.outputdir is None or args.child_outputdir is not None or args.expected_sources is not None:
            parser.error('Parent requires exactly one positional OUTPUTDIR')
        folder = args.outputdir.resolve()
    else:
        if args.child_outputdir is None or args.outputdir is not None:
            parser.error('Child requires --size {200,800} --outputdir OUTPUTDIR')
        folder = args.child_outputdir.resolve()
    # Never overwrite/resume an existing preflight, even after a prior failure.
    folder.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    evidence = Evidence(folder, args.size)
    record = evidence.record
    evidence.save()
    try:
        require(sys.platform.startswith('linux'), 'Resource accounting requires Linux ru_maxrss units')
        evidence.expected = sources()
        record['sources_sha256'] = evidence.expected
        if args.expected_sources is not None:
            require(evidence.expected == json.loads(args.expected_sources.read_text()),
                    'Child startup sources differ from parent snapshot')
        evidence.verify('initial')
        if args.size is None:
            parent(evidence)
        else:
            child(evidence, args.size)
        evidence.verify('final')
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=dict(type=type(error).__name__, message=str(error)),
                      traceback=traceback.format_exc())
        print(record['traceback'], file=sys.stderr, flush=True)
    finally:
        try:
            record['sources_sha256_final'] = sources()
            if record['sources_sha256_final'] != evidence.expected:
                record.update(status='failed', source_integrity_error='Final green source hashes changed')
        except BaseException:
            record.update(status='failed', source_integrity_error=traceback.format_exc())
        record.update(ended_at_utc=utc(), wall_time_s=time.perf_counter() - started,
                      peak_rss_bytes=peak_rss_bytes())
        record['wall_time_human'] = f"{record['wall_time_s']:.3f} s / {record['wall_time_s'] / 60:.3f} min"
        evidence.save()
        print(f"Preflight {record['status']}: {folder / 'summary.json'}; {record['wall_time_human']}", flush=True)
    return 0 if record['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())