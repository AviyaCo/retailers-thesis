"""Four-arm, saved-weight projection attribution; explicit main only, no training.

Launch/logging belongs to an external tmux parent. Set every THREAD_KEYS entry
to 1 before Python starts. Output is exclusive; no resume or overwrite mode.
Tests use small synthetic fixtures, never main or the saved training inputs.
"""
from collections import Counter
from contextlib import contextmanager
import copy
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback
from unittest.mock import patch
import warnings

from audit_gradients import (CHECKPOINTS, IMPLEMENTATION, INPUT, THREAD_KEYS,
                             array_hash, digest, json_numbers, load_parameters,
                             parameter_state, state_hash, utc)

import numpy as np
from scipy import sparse
import scs
import torch

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "ablation_results"
ARMS = ("A", "B", "C", "D")
REPEATS = 5
BUFFER = 1e-6
UPSTREAM_SEED = 20260909
ARGS = {"solve_method": "SCS", "eps": 1e-4, "max_iters": 200000,
        "n_jobs_forward": 1, "n_jobs_backward": 1}
SCS_SETTINGS = {"eps_abs": 1e-4, "eps_rel": 1e-4,
                "max_iters": 200000, "verbose": False}
# Exactly the installed diffcp retry statuses, not a new tolerance policy.
INACCURATE = {"Solved/Inaccurate", "solved (inaccurate - reached max_iters)",
              "solved (inaccurate - reached time_limit_secs)"}


@contextmanager
def timed(record):
    """No serialization inside measured phases, including nested phases."""
    start = time.perf_counter()
    record.update(started_at_utc=utc(), status="running")
    try:
        yield record
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        seconds = time.perf_counter() - start
        record.update(ended_at_utc=utc(), runtime_seconds=seconds,
                      runtime_human=f"{seconds:.6f} s ({seconds / 60:.6f} min)")


def persist(report, output):
    temporary = output / "ablation.tmp"
    with temporary.open("w") as stream:
        json.dump(json_numbers(report), stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "ablation.json")


def packed_data(data):
    """Lossless small canonical matrices, including sparse ordering and zeros."""
    return {k: ({"format": v.format, "shape": list(v.shape),
                 "data": v.data.tolist(), "indices": v.indices.tolist(),
                 "indptr": v.indptr.tolist()} if sparse.issparse(v)
                else None if v is None else np.asarray(v).tolist())
            for k, v in data.items()}


def finite(a):
    return bool(np.isfinite(np.asarray(a)).all())


def scs_outcome(result):
    return {"info": dict(result["info"]),
            "vectors": {k: None if result.get(k) is None else result[k].tolist()
                        for k in ("x", "y", "s")},
            "all_finite": all(result.get(k) is not None and finite(result[k])
                              for k in ("x", "y", "s"))}


def recovery_mapping(layer):
    """Use the installed compiler's actual mapping; fail on unknown layouts."""
    variables = layer.ctx.var_recover
    if len(variables) != 1:
        raise RuntimeError("expected one compiler output variable")
    var = variables[0]
    if (var.source != "primal" or var.unpack_fn != "reshape"
            or tuple(var.shape) != (3,) or not isinstance(var.primal, slice)):
        raise RuntimeError("unsupported compiler recovery; do not substitute guessed indices")
    return {"source": var.source, "unpack_fn": var.unpack_fn,
            "shape": list(var.shape),
            "slice": [var.primal.start, var.primal.stop, var.primal.step],
            "provenance": "CvxpyLayer.ctx.var_recover (CVXPY compiler inverse mapping)"}


def recover_primal(solution, mapping):
    return np.asarray(solution)[slice(*mapping["slice"])].reshape(mapping["shape"], order="F")


@contextmanager
def observe_a(record, captured=None):
    """Observe actual scs.solve calls, preserving diffcp results/retries verbatim.

    Serial diffcp solve_internal calls establish sample IDs independently of
    retry count. SCS constructor and solve are timed without replacing either.
    Capturing data adds overhead to the first forward only, reported separately.
    """
    import diffcp.cone_program as cone_program
    import cvxpylayers.interfaces.diffcp_if as interface
    layer_module = importlib.import_module("cvxpylayers.torch.cvxpylayer")
    record.update(scs_calls=[], extraction_canonicalization_events=[], warnings=[])
    original_solve, original_class = scs.solve, scs.SCS
    original_internal = cone_program.solve_internal
    current = {"sample": -1, "attempt": 0, "call": None}

    def sample_call(*args, **kwargs):
        current["sample"] += 1
        current["attempt"] = 0
        return original_internal(*args, **kwargs)

    class ObservedSCS:
        def __init__(self, *args, **kwargs):
            call = current["call"]
            with timed(call.setdefault("setup", {})):
                self.solver = original_class(*args, **kwargs)

        def solve(self, *args, **kwargs):
            call = current["call"]
            call["solve_arguments"] = kwargs.copy()
            with timed(call.setdefault("solve", {})):
                return self.solver.solve(*args, **kwargs)

    def observed_solve(data, cone, **settings):
        attempt = current["attempt"]
        current["attempt"] += 1
        call = {"sample": current["sample"], "attempt": attempt,
                "settings": settings.copy(), "cone": copy.deepcopy(cone),
                "solution_warm_start_supplied": any(k in data for k in ("x", "y", "s"))}
        record["scs_calls"].append(call)
        current["call"] = call
        if call["solution_warm_start_supplied"]:
            raise RuntimeError("unexpected A solution warm start")
        if captured is not None and attempt == 0:
            with timed(call.setdefault("canonical_capture", {})):
                captured.append({"data": copy.deepcopy(data), "cone": copy.deepcopy(cone),
                                 "settings": settings.copy(), "sample": current["sample"]})
        with timed(call):
            result = original_solve(data, cone, **settings)
        call.update(scs_outcome(result))
        return result

    def observe_function(original, label):
        def wrapped(*args, **kwargs):
            event = {"phase": label}
            record["extraction_canonicalization_events"].append(event)
            with timed(event):
                return original(*args, **kwargs)
        return wrapped

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            with (patch.object(cone_program, "solve_internal", sample_call),
                  patch.object(scs, "solve", observed_solve),
                  patch.object(scs, "SCS", ObservedSCS),
                  patch.object(interface, "_build_diffcp_matrices", observe_function(
                      interface._build_diffcp_matrices, "build_diffcp_matrices")),
                  patch.object(layer_module, "_flatten_and_batch_params", observe_function(
                      layer_module._flatten_and_batch_params, "flatten_parameters")),
                  patch.object(layer_module._ScipySparseMatmul, "forward", staticmethod(observe_function(
                      layer_module._ScipySparseMatmul.forward, "parameter_map_matmul"))),
                  patch.object(layer_module, "_recover_results", observe_function(
                      layer_module._recover_results, "recover_variables"))):
                yield
        finally:
            record["warning_counts"] = dict(Counter(
                w.category.__name__ + ": " + str(w.message) for w in seen))
            record["warnings"] = [{"category": w.category.__name__, "message": str(w.message),
                                    "filename": w.filename, "lineno": w.lineno} for w in seen]


def direct_data(z, capacity):
    """Original sum_squares objective, dropping only z'z, not its scale."""
    p = 2 * sparse.eye(3, format="csc")
    a = sparse.csc_matrix(np.vstack((-np.eye(3), np.ones((1, 3)))))
    return [{"data": {"P": p.copy(), "A": a.copy(), "c": -2 * target.copy(),
                       "b": np.r_[np.zeros(3), cap.item() + BUFFER]},
             "cone": {"l": 4}, "settings": SCS_SETTINGS.copy(), "sample": i}
            for i, (target, cap) in enumerate(zip(z, capacity))]


class NativeSCS:
    """One cached base solver per sample; no primal/dual solution warm starts.

    B mirrors the existing diffcp inaccurate-status retry. Its retry solver is
    constructed lazily once per affected sample and its setup is timed. C has
    no retries. Every returned status, including inaccurate/nonfinite, is kept.
    """
    def __init__(self, problems, record, mapping=None):
        self.problems, self.mapping = problems, mapping
        self.solvers, self.retry_solvers = [], {}
        record["samples"] = []
        with timed(record):
            for problem in problems:
                entry = {"sample": problem["sample"], "settings": problem["settings"]}
                record["samples"].append(entry)
                with timed(entry):
                    self.solvers.append(scs.SCS(problem["data"], problem["cone"], **problem["settings"]))

    def solve(self, record):
        outputs = []
        record["scs_calls"] = []
        for i, (solver, problem) in enumerate(zip(self.solvers, self.problems)):
            def solve_once(selected, settings, attempt):
                call = {"sample": i, "attempt": attempt, "settings": settings.copy(),
                        "warm_start": False, "setup_cached": True}
                record["scs_calls"].append(call)
                with timed(call):
                    result = selected.solve(warm_start=False)
                call.update(scs_outcome(result))
                return result

            result = solve_once(solver, problem["settings"], 0)
            if self.mapping is not None and result["info"]["status"] in INACCURATE:
                settings = dict(problem["settings"], acceleration_lookback=0)
                if i not in self.retry_solvers:
                    setup = {"sample": i, "settings": settings}
                    record.setdefault("legacy_retry_setup", []).append(setup)
                    with timed(setup):
                        self.retry_solvers[i] = scs.SCS(problem["data"], problem["cone"], **settings)
                result = solve_once(self.retry_solvers[i], settings, 1)
            q = result["x"]
            if q is None:
                raise RuntimeError(f"SCS returned no primal vector: {result['info']}")
            outputs.append(recover_primal(q, self.mapping) if self.mapping else q.copy())
        return np.stack(outputs)


class NativeMoreau:
    """Standalone P=2I Solver; never modify the production P=I projection."""
    def __init__(self, record):
        from native_projection import moreau, Solver
        self.events = []
        p = 2 * sparse.eye(3, format="csr")
        a = sparse.csr_matrix(np.vstack((-np.eye(3), np.ones((1, 3)))))
        self.p, self.a = (torch.tensor(m.data, dtype=torch.float64) for m in (p, a))
        settings = moreau.Settings(solver="active_set", device="cpu", enable_grad=True,
                                  auto_tune=False,
                                  active_set_settings=moreau.ActiveSetSettings(diff_method="exact"))
        with timed(record):
            self.solver = Solver(n=3, m=4,
                P_row_offsets=torch.tensor(p.indptr, dtype=torch.int64),
                P_col_indices=torch.tensor(p.indices, dtype=torch.int64),
                A_row_offsets=torch.tensor(a.indptr, dtype=torch.int64),
                A_col_indices=torch.tensor(a.indices, dtype=torch.int64),
                cones=moreau.Cones(num_nonneg_cones=4), settings=settings)
        record.update(settings=settings.model_dump(mode="json"),
                      wrapper_compiled_settings_repr=repr(self.solver._impl._settings),
                      compiled_settings_note="Wrapper IPM defaults are not the active-set settings; active-set uses _original_settings above. repr retains inf explicitly.",
                      lazy_native_setup="Native batch construction and matrix setup occur in first forward, measured there")
        original_solve = self.solver._impl.solve
        original_create = self.solver._impl._create_batch_solver
        owner = self

        class ObservedBatch:
            def __init__(self, native):
                self.native = native

            def __getattr__(self, name):
                return getattr(self.native, name)

            def setup(self, *args, **kwargs):
                event = {"phase": "native_matrix_setup"}
                owner.events.append(event)
                with timed(event):
                    return self.native.setup(*args, **kwargs)

        def create(*args, **kwargs):
            event = {"phase": "native_batch_constructor"}
            self.events.append(event)
            with timed(event):
                original_create(*args, **kwargs)
            self.solver._impl._batch_solver = ObservedBatch(self.solver._impl._batch_solver)

        def checked(*args, **kwargs):
            event = {"phase": "native_impl_solve"}
            self.events.append(event)
            with timed(event):
                result = original_solve(*args, **kwargs)
                event.update(status_codes=result["status"].reshape(-1).tolist(),
                             iterations=result["iterations"].reshape(-1).tolist(),
                             solver_reported_seconds=float(result["solve_time"]),
                             all_finite=all(bool(torch.isfinite(result[k]).all())
                                            for k in ("x", "z", "s", "obj_val")))
                # Do not report wrapper placeholder zero residuals as measured.
                if not all(s == 1 for s in event["status_codes"]) or not event["all_finite"]:
                    raise RuntimeError(f"native status/nonfinite failure: {event}")
                return result

        self.solver._impl._create_batch_solver = create
        self.solver._impl.solve = checked

    def solve(self, z, capacity, record):
        self.events = []
        record["native_events"] = self.events
        b = torch.cat((torch.zeros_like(z), capacity + BUFFER), dim=1)
        return self.solver.solve(self.p, self.a, -2 * z, b, warm_start=None).x


def prepare_inputs(model, x1):
    """Match canonical float32 neural operation order, then promote QP inputs."""
    with torch.no_grad():
        q10 = torch.nn.functional.softplus(model.Q10)
        q1r = torch.nn.functional.softplus(model.Q1r)
        net_in = torch.cat([x1, q1r.unsqueeze(0).expand(len(x1), -1),
                            q10.expand(len(x1), 1)], dim=1)
        z = model.fc_layers(net_in)
        if z.dtype != torch.float32 or q10.dtype != torch.float32:
            raise RuntimeError("neural preparation must retain original float32 arithmetic")
        return z.double().detach(), q10.expand(len(x1), 1).double().detach()


def diagnostics(q, cap, oracle):
    return {"q": q.tolist(), "all_finite": finite(q), "min_q": float(q.min()),
            "capacity_violation": (q.sum(1) - cap.reshape(-1) - BUFFER).tolist(),
            "max_capacity_violation": float(np.maximum(q.sum(1) - cap.reshape(-1) - BUFFER, 0).max()),
            "max_error_gurobi": float(np.abs(q - oracle).max())}


def run_pass(arm, repeat, backend, z, cap, upstream, oracle, record, captured=None):
    record.update(arm=arm, repeat=repeat, first_pass=repeat == 0, forward={},
                  backward_requested=arm in ("A", "D"))
    # Detached leaves, including one capacity leaf per sample, in BOTH arms.
    target = z.detach().clone().requires_grad_(arm in ("A", "D"))
    capacity = cap.detach().clone().requires_grad_(arm in ("A", "D"))
    with timed(record):
        if arm == "A":
            with observe_a(record, captured), timed(record["forward"]):
                output, = backend(target, capacity, solver_args=ARGS, warm_start=False)
        elif arm == "D":
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                try:
                    with timed(record["forward"]):
                        output = backend.solve(target, capacity, record)
                finally:
                    record["warning_counts"] = dict(Counter(
                        w.category.__name__ + ": " + str(w.message) for w in seen))
        else:
            with timed(record["forward"]):
                output = backend.solve(record)
        q = output.detach().numpy() if isinstance(output, torch.Tensor) else output
        record.update(diagnostics(q, cap.numpy(), oracle))
        if arm in ("A", "D"):
            record["backward"] = {}
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                try:
                    with timed(record["backward"]):
                        output.backward(upstream)
                finally:
                    record["backward_warning_counts"] = dict(Counter(
                        w.category.__name__ + ": " + str(w.message) for w in seen))
                    record["gradients"] = {name: None if t.grad is None else t.grad.tolist()
                                           for name, t in (("target", target), ("capacity", capacity))}
                    record["gradients_finite"] = all(t.grad is not None and bool(torch.isfinite(t.grad).all())
                                                      for t in (target, capacity))
            if not record["gradients_finite"]:
                raise RuntimeError("absent or nonfinite projection gradients")
        # Inaccurate/nonfinite SCS outputs are preserved, never clamped.
        if arm in ("A", "B", "C"):
            record["solver_outcome"] = ("solved" if record["all_finite"] and all(
                c["info"]["status_val"] == 1 and c["all_finite"] for c in record["scs_calls"])
                else "nonstandard_status_or_nonfinite_retained")
            record["solver_status_counts"] = dict(Counter(str(c["info"]["status_val"])
                                                         for c in record["scs_calls"]))
            record["solver_iterations"] = [c["info"]["iter"] for c in record["scs_calls"]]
        else:
            record["solver_outcome"] = "solved"
    return q


def summary(passes):
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for phase in ("forward", "backward"):
            eligible = [r for r in passes if r["arm"] == arm and r["repeat"] > 0
                        and r.get("status") == "completed" and r.get("solver_outcome") == "solved"
                        and phase in r]
            values = [r[phase]["runtime_seconds"] for r in eligible]
            result[arm][phase] = ({"count": len(values), "mean": statistics.mean(values),
                "median": statistics.median(values), "minimum": min(values), "maximum": max(values),
                "unit": "seconds", "mean_human": f"{statistics.mean(values):.6f} s"} if values else None)
        result[arm]["first_pass"] = [r for r in passes if r["arm"] == arm and r["repeat"] == 0]
        result[arm]["failed_or_incomplete_repeats"] = [r["repeat"] for r in passes
            if r["arm"] == arm and r.get("status") != "completed"]
        result[arm]["nonstandard_solver_repeats"] = [r["repeat"] for r in passes
            if r["arm"] == arm and r.get("status") == "completed" and r.get("solver_outcome") != "solved"]
    return result


def main():
    OUTPUT.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = {"experiment": "retailer_four_arm_projection_attribution", "started_at_utc": utc(),
        "status": "running", "host": platform.node(), "platform": platform.platform(),
        "cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": sys.version, "command": [sys.executable, *sys.argv], "cwd": os.getcwd(),
        "tmux_parent": os.environ.get("TMUX"), "logging": "external parent owns stdout/stderr log",
        "sample_size": 50, "radius": 0, "data_seed": 0, "network_seed": 0,
        "upstream_seed": UPSTREAM_SEED, "subsequent_repeats": REPEATS,
        "workers": 1, "thread_environment": {k: os.environ.get(k) for k in THREAD_KEYS},
        "protocol": {
            "inputs": "original all-50 seed0 training pairs only; saved learned parameters only; float32 neural arithmetic then identical detached float64 QP leaves",
            "feasible_set": "q>=0, sum(q)<=capacity+1e-6; buffer is the old trial feasible-set choice, NOT a solver tolerance",
            "A": "original CVXPYLayers/diffcp sum_squares SOC; eps=1e-4, max_iters=200000; unchanged inaccurate acceptance/retry; n_jobs_forward/backward=1 controlled deviation from legacy default pools",
            "B": "actual first A scs.solve canonical data/cone/settings, actual compiler variable recovery; serial cached native SCS; same diffcp inaccurate retry policy, cached retry solver lazily constructed",
            "C": "serial native SCS direct QP P=2I,c=-2z,A=[-I;ones],b=[0;cap+buffer],l=4; no retries",
            "D": "standalone native Moreau exact active_set P=2I,c=-2z, same direct constraints; native batching, torch/Rayon=1; strict status/finite checks",
            "scaling": "sum_squares has NO intrinsic 1/2; C/D drop constant z'z only. Historical Moreau P=I is NOT the same numerical scaling",
            "warm_start": "none supplied: A scs.solve uses warm_start=True on a NEW solver with no x/y/s; B/C solve(warm_start=False); D warm_start=None. Setup caches retained, including possible SCS adaptive scaling state",
            "cold": "first pass per checkpoint/arm, NOT process/machine cold; A captures canonical data on this pass; B/C constructor setup separate; D lazy native setup in first forward",
            "attribution": "A forward includes canonicalization, new SCS setup per call/retry and diffcp derivative preparation. A-B is NOT pure framework overhead. C-D retains serial Python versus native batching and distinct solver tolerances/algorithms",
            "backward": "one backward per A/D base forward, including first; same seed20260909 unnormalized Gaussian upstream (stored); target and capacity leaf gradients only; no B/C derivatives, optimizer or neural training",
            "timing": "perf_counter wall seconds in current invocation; nested timings overlap; per-pass atomic persistence outside setup/forward/backward/pass timers; checkpoint and overall elapsed include persistence; first separate, summaries use subsequent five only, excluding nonstandard solver outcomes",
            "scope": "no validation/test, CV, SAA, training, CCG, SRO-FA, alternative tolerance or new arm; Gurobi training-input projection reference only",
            "native_residuals": "Moreau CPU wrapper emits placeholder zero residuals; not reported as measurements",
            "metadata": "host/versions/seeds/threads inherited by every checkpoint/arm/phase; SCS info setup_time/solve_time in milliseconds, perf_counter durations seconds",
        }, "setup": {}, "candidates": [], "integrity": {}}
    paths, before, states, arrays, tensors, prepared, models = {}, {}, {}, [], [], {}, {}
    error = None
    try:
        with timed(report["setup"]):
            if any(os.environ.get(k) != "1" for k in THREAD_KEYS):
                raise RuntimeError("Set every THREAD_KEYS variable to 1 before starting Python")
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            sys.path.insert(0, str(IMPLEMENTATION))
            import run_trial
            from baseline import SOURCE, load_baseline
            import native_projection
            if Path(run_trial.__file__).resolve() != IMPLEMENTATION / "run_trial.py":
                raise RuntimeError("unexpected trial module")
            report.update(torch_threads=torch.get_num_threads(),
                          torch_interop_threads=torch.get_num_interop_threads(),
                          versions={p: importlib.metadata.version(p) for p in (
                              "numpy", "scipy", "torch", "cvxpy", "cvxpylayers", "diffcp",
                              "scs", "moreau", "moreau-cpu", "gurobipy")})
            paths = {k: INPUT / f"{k}.pth" for k in CHECKPOINTS}
            paths.update(trial=INPUT / "trial.json", canonical=SOURCE,
                         harness=Path(__file__).resolve(), tests=HERE / "test_ablation.py",
                         audit=HERE / "audit_gradients.py",
                         **{k: IMPLEMENTATION / f"{k}.py" for k in ("baseline", "run_trial", "native_projection")})
            for name in ("cvxpylayers.torch.cvxpylayer", "cvxpylayers.utils.parse_args",
                         "cvxpylayers.interfaces.diffcp_if", "diffcp.cone_program", "scs",
                         "moreau.torch", "moreau.torch._cpu_impl", "moreau_cpu._cpu"):
                paths[name] = Path(inspect.getfile(importlib.import_module(name)))
            before = {k: digest(p) for k, p in paths.items()}
            report["integrity"].update(paths={k: str(v) for k, v in paths.items()}, files_before=before)
            trial = json.loads(paths["trial"].read_text())
            for key, expected in (("status", "completed"), ("sample_size", 50),
                                  ("data_seed", 0), ("network_seed", 0), ("radius", 0)):
                if trial.get(key) != expected:
                    raise RuntimeError(f"unexpected original trial {key}")
            for name, field in (("canonical", "source_sha256"), ("initial_state", "initial_state_sha256"),
                                ("scs_selected", "scs_selected_sha256"), ("moreau_selected", "moreau_selected_sha256")):
                expected = trial.get(field)
                report["integrity"].setdefault("trial_hash_checks", {})[name] = {
                    "expected": expected, "actual": before[name],
                    "status": "unavailable_in_trial" if expected is None else "checked"}
                if expected is not None and expected != before[name]:
                    raise RuntimeError(f"trial hash mismatch: {name}")
            baseline = load_baseline()
            if (baseline.cfg.N_SAMPLES, baseline.cfg.R, baseline.cfg.HIDDEN_WIDTH,
                baseline.cfg.SCS_EPS, baseline.cfg.SCS_MAX_ITERS) != (50, 3, 16, 1e-4, 200000):
                raise RuntimeError("canonical config changed")
            arrays = list(baseline.generate_data(0))
            raw_hash = array_hash(*arrays)
            if raw_hash != trial["training_data_sha256"]:
                raise RuntimeError("original all-50 training data hash mismatch")
            tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays]
            report["integrity"].update(data_before=raw_hash,
                execution_data_before=array_hash(*(t.numpy() for t in tensors)))
            states = {k: torch.load(paths[k], map_location="cpu", weights_only=True) for k in CHECKPOINTS}
            report["integrity"]["states_before"] = {k: state_hash(s) for k, s in states.items()}
            upstream = torch.randn((50, 3), generator=torch.Generator().manual_seed(UPSTREAM_SEED),
                                   dtype=torch.float64)
            report["upstream"] = upstream.tolist()
            report["upstream_sha256"] = array_hash(upstream.numpy())
        for checkpoint in CHECKPOINTS:
            candidate = {"checkpoint": checkpoint, "setup": {}, "input_preparation": {},
                         "reference": {}, "passes": [], "canonical_extraction": {}}
            report["candidates"].append(candidate)
            try:
                with timed(candidate):
                    with timed(candidate["setup"].setdefault("A", {})):
                        model = baseline.Model()
                        load_parameters(model, states[checkpoint], torch)
                        model.eval()
                        models[checkpoint] = model
                    with timed(candidate["input_preparation"]):
                        z, cap = prepare_inputs(model, tensors[0])
                    prepared[checkpoint] = (z, cap)
                    candidate.update(qp_inputs={"target": z.tolist(), "capacity": cap.tolist(),
                        "sha256_before": array_hash(z.numpy(), cap.numpy())},
                        learned_parameters_sha256=state_hash(parameter_state(model)))
                    oracle = run_trial.reference_projection(model, tensors[0], candidate["reference"]).numpy()
                    candidate["reference"]["q"] = oracle.tolist()
                    captured = []
                    backends = {"A": model.proj_layer}
                    a_outputs = {}
                    # A first pass supplies B's real input; no extra capture/warm-up solve.
                    for repeat in range(REPEATS + 1):
                        order = ARMS[repeat % 4:] + ARMS[:repeat % 4]
                        for arm in order:
                            if repeat == 0 and arm == "B":
                                with timed(candidate["canonical_extraction"]):
                                    mapping = recovery_mapping(model.proj_layer)
                                    if [p["sample"] for p in captured] != list(range(50)):
                                        raise RuntimeError("expected first base scs.solve data for each of 50 samples")
                                    if any(p["settings"] != SCS_SETTINGS for p in captured):
                                        raise RuntimeError("unexpected captured SCS settings")
                                    candidate["canonical"] = [{**p, "data": packed_data(p["data"])} for p in captured]
                                    candidate["recovery_mapping"] = mapping
                                backends["B"] = NativeSCS(captured, candidate["setup"].setdefault("B", {}), mapping)
                            elif repeat == 0 and arm == "C":
                                with timed(candidate.setdefault("direct_formulation", {})):
                                    problems = direct_data(z.numpy(), cap.numpy())
                                candidate["direct_data"] = [{**p, "data": packed_data(p["data"])} for p in problems]
                                backends["C"] = NativeSCS(problems, candidate["setup"].setdefault("C", {}))
                            elif repeat == 0 and arm == "D":
                                backends["D"] = NativeMoreau(candidate["setup"].setdefault("D", {}))
                            record = {"order": list(order), "order_position": order.index(arm)}
                            candidate["passes"].append(record)
                            try:
                                q = run_pass(arm, repeat, backends[arm], z, cap, upstream, oracle,
                                             record, captured if repeat == 0 and arm == "A" else None)
                                if arm == "A":
                                    a_outputs[repeat] = q.copy()
                                if arm == "B" and repeat == 0:
                                    record["max_error_A_first"] = float(np.abs(q - a_outputs[0]).max())
                                    # Same algorithm/data on fresh instances: a recovery/capture regression, not a solver tolerance change.
                                    np.testing.assert_allclose(q, a_outputs[0], rtol=0, atol=1e-10)
                            except BaseException:
                                record["status"] = "failed"
                                raise
                            finally:
                                persist(report, OUTPUT)
                        a_q = a_outputs[repeat]
                        for record in candidate["passes"]:
                            if record["repeat"] == repeat:
                                record["max_error_A_same_repeat"] = float(np.abs(np.asarray(record["q"]) - a_q).max())
                    candidate["summary"] = summary(candidate["passes"])
            finally:
                candidate["summary"] = summary(candidate["passes"])
                persist(report, OUTPUT)
    except BaseException as caught:
        error = caught
        report.update(error=f"{type(caught).__name__}: {caught}", traceback=traceback.format_exc())
    finally:
        try:
            with timed(report.setdefault("final_integrity", {})):
                after = {k: digest(paths[k]) for k in before}
                checks = {"files": after == before}
                report["integrity"]["files_after"] = after
                if arrays:
                    checks["data"] = array_hash(*arrays) == report["integrity"]["data_before"]
                    checks["execution_data"] = array_hash(*(t.numpy() for t in tensors)) == report["integrity"]["execution_data_before"]
                if states:
                    checks["states"] = {k: state_hash(s) for k, s in states.items()} == report["integrity"]["states_before"]
                for candidate in report["candidates"]:
                    checkpoint = candidate["checkpoint"]
                    if checkpoint in prepared:
                        checks[checkpoint + "_qp_inputs"] = array_hash(*(t.numpy() for t in prepared[checkpoint])) == candidate["qp_inputs"]["sha256_before"]
                    if checkpoint in models:
                        checks[checkpoint + "_model"] = state_hash(parameter_state(models[checkpoint])) == state_hash(states[checkpoint])
                report["integrity"]["unchanged"] = checks
                if not all(checks.values()):
                    raise RuntimeError("source/input/model changed during ablation")
        except BaseException as caught:
            report["integrity_error"] = f"{type(caught).__name__}: {caught}"
            if error is None:
                error = caught
        report["runtime_table"] = [{"method": method, "status": "not_run", "runtime_seconds": None}
            for method in ("nominal_neural_training", "robust_neural_ccg", "sro_fa", "cv_validation_test")]
        for candidate in report["candidates"]:
            for phase in ("input_preparation", "reference"):
                report["runtime_table"].append({"checkpoint": candidate["checkpoint"], "method": phase,
                                                **candidate[phase]})
            for arm, entry in candidate["setup"].items():
                report["runtime_table"].append({"checkpoint": candidate["checkpoint"], "arm": arm,
                                                "method": "setup", **entry})
            for entry in candidate["passes"]:
                for phase in ("forward", "backward"):
                    if phase in entry:
                        report["runtime_table"].append({"checkpoint": candidate["checkpoint"], "arm": entry["arm"],
                            "repeat": entry["repeat"], "method": phase, **entry[phase]})
        elapsed = time.perf_counter() - start
        report.update(status="completed" if error is None else "failed", ended_at_utc=utc(),
                      total_benchmark_elapsed_seconds=elapsed,
                      runtime_human=f"{elapsed:.6f} s ({elapsed / 60:.6f} min)")
        persist(report, OUTPUT)
        print(f"ablation {report['status']}: {elapsed:.6f}s; {OUTPUT / 'ablation.json'}", flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()