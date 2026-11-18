"""Matched saved-weight, training-only gradient diagnostics; never train.

Importing this file performs no dependency imports, data access, or solves.
Run explicitly with BLAS/OpenMP/NumExpr/Rayon thread variables set to 1.
The output directory must not exist. Nonfinite JSON numbers are encoded as
explicit strings (NaN, +Infinity, -Infinity), never substituted in arithmetic.
"""

from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from unittest.mock import patch
import warnings


HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE.parents[3] / "retailers_moreau_N50_R3_net16_v2"
INPUT = HERE / "results"
OUTPUT = HERE / "gradient_audit_results_v2"
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")
BACKENDS = ("scs", "moreau")
CHECKPOINTS = ("initial_state", "scs_selected", "moreau_selected")
BOUNDARY_TOLS = (0.0, 1e-8, 1e-6, 1e-4)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_numbers(value):
    """Serialization only: leave all computational values untouched."""
    if isinstance(value, dict):
        return {str(k): json_numbers(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_numbers(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    return value


def persist(report):
    temporary = OUTPUT / "audit.tmp"
    with temporary.open("w") as stream:
        json.dump(json_numbers(report), stream, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(OUTPUT / "audit.json")


@contextmanager
def phase(record, report):
    start = time.perf_counter()
    record.update(status="running", started_at_utc=utc())
    try:
        yield record
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        seconds = time.perf_counter() - start
        record.update(runtime_seconds=seconds, runtime_human=f"{seconds:.3f} s ({seconds / 60:.3f} min)",
                      ended_at_utc=utc())
        persist(report)


def array_hash(*arrays):
    return hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def parameter_state(model):
    return {k: p.detach().clone() for k, p in model.named_parameters()}


def load_parameters(model, state, torch):
    """Load learned parameters only; backend buffers are not checkpoints."""
    parameters = dict(model.named_parameters())
    if parameters.keys() != state.keys():
        raise RuntimeError("learned parameter keys do not match checkpoint")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = state[name]
            if parameter.shape != value.shape or parameter.dtype != value.dtype:
                raise RuntimeError(f"parameter shape/dtype mismatch: {name}")
            parameter.copy_(value)


def finite_summary(a, np):
    a = np.asarray(a)
    return {"all_finite": bool(np.isfinite(a).all()), "nan_count": int(np.isnan(a).sum()),
            "positive_infinity_count": int(np.isposinf(a).sum()),
            "negative_infinity_count": int(np.isneginf(a).sum())}


def margins(a, np):
    a = np.asarray(a)
    return {"raw": a.tolist(), "minimum": float(a.min()), "maximum": float(a.max()),
            "minimum_absolute": float(np.abs(a).min()),
            "boundary_counts": {str(t): int((np.abs(a) <= t).sum()) for t in BOUNDARY_TOLS},
            "nonfinite": finite_summary(a, np)}


def vector_metrics(a, b, np):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb, nd = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(a - b)
    denominator = max(na, nb)
    return {"scs_norm": float(na), "moreau_norm": float(nb),
            "difference_norm": float(nd), "max_absolute_difference": float(np.abs(a - b).max()),
            "symmetric_relative_difference": float(nd / denominator) if denominator != 0 else None,
            "cosine": float(np.dot(a, b) / (na * nb)) if na != 0 and nb != 0 else None,
            "undefined_zero_norm": bool(na == 0 or nb == 0),
            "scs_nonfinite": finite_summary(a, np), "moreau_nonfinite": finite_summary(b, np)}


def signatures(model, x1, x2, captured, cfg, torch, np):
    """Mirror canonical cost operation ordering; no diagnostic casts in it."""
    with torch.no_grad():
        q10 = torch.nn.functional.softplus(model.Q10)
        q1r = torch.nn.functional.softplus(model.Q1r)
        q2 = captured["q2"]
        t1 = cfg.b * (x2 + x1 - q2 - q1r) - cfg.h * q2
        t2 = cfg.h * (q1r - x1 - x2)
        t3 = cfg.b * (x1 - q1r) - cfg.h * x2
        branches = torch.stack((t1, t2, t3), dim=-1)
        winning = branches == branches.max(dim=-1, keepdim=True).values
        gap = branches.sort(dim=-1, descending=True).values
        # NativeProjection adds its buffer after promotion to double; SCS
        # canonicalization also applies the buffer to the numeric parameter.
        cap = captured["capacity"].double() + 1e-6
        slack = cap.squeeze(-1) - q2.double().sum(-1)
        arrays = {"relu_" + k: v.numpy() for k, v in captured["relu"].items()}
        arrays.update(cost_top_gap=(gap[..., 0] - gap[..., 1]).numpy(),
                      cost_inner_gap=(t1 - t2).numpy(),
                      cost_outer_gap=(torch.maximum(t1, t2) - t3).numpy(),
                      projected_q2=q2.numpy(), capacity_slack=slack.numpy())
        masks = {"relu_" + k: (v > 0).numpy().tolist() for k, v in captured["relu"].items()}
        masks.update(cost_winners=winning.numpy().tolist(),
                     cost_inner_sign=torch.sign(t1 - t2).numpy().tolist(),
                     cost_outer_sign=torch.sign(torch.maximum(t1, t2) - t3).numpy().tolist(),
                     projected_positive=(q2 > 0).numpy().tolist(),
                     capacity_feasible=(slack >= 0).numpy().tolist())
        for tol in BOUNDARY_TOLS:
            masks[f"projected_zero_at_{tol}"] = (q2.abs() <= tol).numpy().tolist()
            masks[f"capacity_binding_at_{tol}"] = (slack.abs() <= tol).numpy().tolist()
        return {"z": captured["z"].numpy().tolist(), "capacity": captured["capacity"].numpy().tolist(),
                "buffered_capacity": cap.numpy().tolist(), "q2": q2.numpy().tolist(),
                "q10": q10.numpy().tolist(), "q1r": q1r.numpy().tolist(),
                "projection_dtype": str(q2.dtype), "masks": masks,
                "margins": {k: margins(a, np) for k, a in arrays.items()},
                "cost_branches": branches.numpy().tolist()}


def mask_changes(base, other, np):
    return {k: int(np.count_nonzero(np.asarray(v) != np.asarray(other["masks"][k])))
            for k, v in base["masks"].items()}


@contextmanager
def solver_observation(model, backend, record, scs, np):
    """Observe without changing solver settings, results, or legacy retries."""
    record.update(scs_calls=[], native_calls=[], warnings=[])
    original_scs = scs.solve

    def observe_scs(*args, **kwargs):
        call = {"started_at_utc": utc(), "status": "running",
                "settings": {k: v for k, v in kwargs.items() if k not in ("x", "y", "s")}}
        record["scs_calls"].append(call)
        start = time.perf_counter()
        try:
            result = original_scs(*args, **kwargs)
            call.update(status="completed", info={k: result["info"].get(k) for k in
                        ("status", "status_val", "iter", "res_pri", "res_dual", "gap", "solve_time", "setup_time")})
            call["nonfinite"] = {k: finite_summary(result[k], np) for k in ("x", "y", "s") if result.get(k) is not None}
            call["nonfinite_raw"] = {k: np.asarray(result[k]).tolist() for k, summary in call["nonfinite"].items()
                                     if not summary["all_finite"]}
            return result
        except BaseException as error:
            call.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            call.update(runtime_seconds=time.perf_counter() - start, ended_at_utc=utc())

    native = model.proj_layer if backend == "moreau" else None
    if native is not None:
        native.calls.clear()
        original_native = native.solver._impl.solve

        def observe_native(*args, **kwargs):
            call = {"started_at_utc": utc(), "status": "running"}
            record["native_calls"].append(call)
            start = time.perf_counter()
            try:
                result = original_native(*args, **kwargs)  # strict checks retained
                call["status"] = "completed"
                return result
            except BaseException as error:
                call.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                call.update(runtime_seconds=time.perf_counter() - start, ended_at_utc=utc())
                call["solver_records_available"] = list(native.calls)

        native.solver._impl.solve = observe_native
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            try:
                with patch.object(scs, "solve", new=observe_scs):
                    yield
            finally:
                record["warnings"] = [{"category": w.category.__name__, "message": str(w.message),
                                        "filename": w.filename, "lineno": w.lineno} for w in seen]
                record["warning_counts"] = dict(Counter(w["category"] + ": " + w["message"] for w in record["warnings"]))
                record["legacy_diffcp_warning_count"] = sum("diffcp" in w["filename"] for w in record["warnings"])
    finally:
        if native is not None:
            native.solver._impl.solve = original_native
            record["native_status_iterations"] = list(native.calls)


def evaluate(model, x1, x2, backend, record, report, cfg, torch, np, scs, backward):
    captured = {"relu": {}}
    handles = []

    def projection_hook(module, args, kwargs, output):
        captured.update(z=args[0].detach().clone(), capacity=args[1].detach().clone(),
                        q2=output[0].detach().clone())

    handles.append(model.proj_layer.register_forward_hook(projection_hook, with_kwargs=True))
    for i, layer in enumerate(model.fc_layers):
        if isinstance(layer, torch.nn.Linear) and i + 1 < len(model.fc_layers) and isinstance(model.fc_layers[i + 1], torch.nn.ReLU):
            def linear_hook(module, args, output, name=str(i)):
                captured["relu"][name] = output.detach().clone()
            handles.append(layer.register_forward_hook(linear_hook))
    record.update(backend=backend, backward_requested=backward, forward={}, diagnostics={})
    try:
        with phase(record, report), solver_observation(model, backend, record, scs, np):
            model.zero_grad(set_to_none=True)
            with phase(record["forward"], report), torch.set_grad_enabled(backward):
                costs = model.per_sample_cost(x1, x2)
                loss = costs.mean()
                record.update(loss=float(loss.detach()), loss_dtype=str(loss.dtype),
                              per_sample_cost=costs.detach().numpy().tolist(),
                              loss_ulp=float(np.abs(np.spacing(loss.detach().numpy()))),
                              cost_nonfinite=finite_summary(costs.detach().numpy(), np))
            # Persist the full forward evidence even if backward subsequently fails.
            with phase(record["diagnostics"], report):
                record["signature"] = signatures(model, x1, x2, captured, cfg, torch, np)
            if backward:
                record["backward"] = {}
                try:
                    with phase(record["backward"], report):
                        loss.backward()
                finally:
                    # Preserve partial/absent/nonfinite gradients even if backward failed.
                    record["gradients"] = {k: None if p.grad is None else p.grad.detach().numpy().tolist()
                                           for k, p in model.named_parameters()}
                    record["gradient_norms"] = {k: None if p.grad is None else float(p.grad.detach().double().norm())
                                                for k, p in model.named_parameters()}
                    record["gradient_nonfinite"] = {k: None if p.grad is None else finite_summary(p.grad.detach().numpy(), np)
                                                    for k, p in model.named_parameters()}
    finally:
        for handle in handles:
            handle.remove()


def compare_gradients(left, right, groups, np):
    result = {"loss_scs_minus_moreau": left["loss"] - right["loss"], "parameters": {}, "groups": {}}
    for target, definitions in (("parameters", {k: [k] for k in left["gradients"]}), ("groups", groups)):
        for label, names in definitions.items():
            if any(arm["gradients"][k] is None for arm in (left, right) for k in names):
                result[target][label] = {"status": "absent_gradient", "metrics": None}
            else:
                arrays = [np.concatenate([np.asarray(arm["gradients"][k]).ravel() for k in names]) for arm in (left, right)]
                result[target][label] = vector_metrics(*arrays, np)
    result["signature_changes"] = mask_changes(left["signature"], right["signature"], np)
    result["projection_max_absolute_difference"] = float(np.abs(np.asarray(left["signature"]["q2"]) - np.asarray(right["signature"]["q2"])).max())
    return result


def fd_block(model, state, direction, name, h, x1, x2, backend, base, record,
             report, cfg, torch, np, scs):
    record.update(direction=name, h=h, backend=backend, plus={}, minus={})
    with phase(record, report):
        dot = None if any(g is None for g in base["gradients"].values()) else sum(
            float(np.dot(np.asarray(base["gradients"][k], dtype=np.float64).ravel(), d.double().numpy().ravel()))
            for k, d in direction.items())
        record["autograd_dot_direction"] = dot
        try:
            for side, sign in (("plus", 1), ("minus", -1)):
                with torch.no_grad():
                    for k, p in model.named_parameters():
                        p.copy_(state[k] + (sign * h) * direction[k])
                record[side]["realized_perturbation"] = {}
                for k, p in model.named_parameters():
                    delta = p.detach().double() - state[k].double()
                    requested = sign * h * direction[k].double()
                    record[side]["realized_perturbation"][k] = {
                        "norm": float(delta.norm()), "max_error_from_requested": float((delta - requested).abs().max()),
                        "unchanged_nonzero_direction_coordinates": int(((delta == 0) & (requested != 0)).sum())}
                evaluate(model, x1, x2, backend, record[side], report, cfg, torch, np, scs, False)
            fp, fm, f0 = record["plus"]["loss"], record["minus"]["loss"], base["loss"]
            central, forward, backward = (fp - fm) / (2 * h), (fp - f0) / h, (f0 - fm) / h
            changes = {side: mask_changes(base["signature"], record[side]["signature"], np) for side in ("plus", "minus")}
            changes["minus_to_plus"] = mask_changes(record["minus"]["signature"], record["plus"]["signature"], np)
            record.update(central_slope=central, forward_slope=forward, backward_slope=backward,
                          central_minus_autograd=None if dot is None else central - dot,
                          forward_minus_autograd=None if dot is None else forward - dot,
                          backward_minus_autograd=None if dot is None else backward - dot,
                          one_sided_gap=forward - backward, loss_changes={"plus": fp - f0, "minus": fm - f0},
                          mask_changes=changes,
                          relu_or_cost_switch_observed=any(n for side in changes.values() for k, n in side.items() if k.startswith(("relu_", "cost_"))),
                          projection_mask_switch_observed=any(n for side in changes.values() for k, n in side.items() if k.startswith(("projected_", "capacity_"))))
            ulp = max(base["loss_ulp"], record["plus"]["loss_ulp"], record["minus"]["loss_ulp"])
            record["numerical_resolution"] = {"max_loss_ulp": ulp, "ulp_over_h": ulp / h,
                "plus_change_in_ulps": abs(fp - f0) / ulp, "minus_change_in_ulps": abs(fm - f0) / ulp,
                "note": "ULP is rounding evidence only, not a bound on solver error; small-h instability without observed branch changes suggests numerical noise, not proof. Endpoint masks cannot exclude an interior crossing."}
        finally:
            load_parameters(model, state, torch)
            record["base_weights_restored"] = state_hash(parameter_state(model)) == state_hash(state)
            if not record["base_weights_restored"]:
                raise RuntimeError("finite-difference base state restoration failed")


def main():
    OUTPUT.mkdir(exist_ok=False)
    start = time.perf_counter()
    report = {"experiment": "retailers_matched_saved_checkpoint_full_model_gradient_audit", "status": "running",
        "started_at_utc": utc(), "method": "diagnostics_only", "data_seed": 0, "direction_seed": 20260909,
        "radius": 0, "sample_size": 50, "optimizer": None, "iterations": 0, "epochs": 0,
        "parallel_model_workers": 1, "reference_gurobi_threads": 1,
        "command": [sys.executable, *sys.argv], "cwd": os.getcwd(), "host": platform.node(),
        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "python": sys.version,
        "thread_environment": {k: os.environ.get(k) for k in THREAD_KEYS},
        "protocol": {"data": "All 50 training samples, seed0; no validation/test access or selection",
            "weights": "Exactly each saved checkpoint, no initialization training or optimizer; raw Q parameters differentiated through softplus",
            "float32": "Original neural/data float32 execution; backend output dtypes preserved, Moreau internally float64",
            "float64": "Supplemental neural diagnostic only: promote saved weights AND original float32-rounded inputs to float64; not a modification of original runs",
            "capacity_buffer": 1e-6, "scs_settings": {"solve_method": "SCS", "eps": 1e-4, "max_iters": 200000},
            "legacy_scs": "Unmodified diffcp behavior, including inaccurate-result acceptance and internal SCS retries; all scs.solve calls observed and warnings counted. Default diffcp batch pools retained.",
            "native": "Existing strict NativeProjection exact active-set derivative; no fallback/retry/settings change",
            "boundary_tolerances": BOUNDARY_TOLS, "boundary_interpretation": "Tolerances label diagnostics only. Binding projection constraints alone are not evidence of nondifferentiability. Tiny projected sign changes can be solver noise.",
            "fd_interpretation": "No automatic pass/fail. Inspect one-sided disagreement, multi-h stability, ReLU/cost branch changes, projection mask changes at multiple tolerances, margins, ULPs, and solver residuals together. Smooth endpoint signatures do not prove smoothness throughout the segment.",
            "finite_encoding": "Nonfinite values remain in computations; JSON encodes NaN/+Infinity/-Infinity as strings. Absent or undefined zero-norm metrics are null, not zero.",
            "scope": "174 full-model forward passes, 12 backward passes, 6 references of 50 QPs; no repeat noise estimates or training",
            "timing": "Current invocation only, perf_counter seconds; sequential candidates; nested phase times overlap and must not be summed",
            "training_ccg_sro_fa_cv_test_runtimes": "not applicable: none run; no historical training runtime inferred"},
        "setup": {}, "input_integrity": {}, "candidates": [], "final_integrity": {}}
    persist(report)
    paths, before, original_states, data_arrays, data_tensors = {}, {}, {}, [], []
    error = None
    try:
        with phase(report["setup"], report):
            if any(os.environ.get(k) != "1" for k in THREAD_KEYS):
                raise RuntimeError("Launch with every listed BLAS/OpenMP/NumExpr/Rayon thread environment variable set to 1")
            # Delayed imports: neither importing this audit nor reusing run_trial
            # executes the guarded trial main or the canonical launch pipeline.
            sys.path.insert(0, str(IMPLEMENTATION))
            import numpy as np
            import torch
            import scs
            import run_trial
            from baseline import SOURCE, load_baseline
            from native_projection import NativeProjection

            if Path(run_trial.__file__).resolve() != IMPLEMENTATION / "run_trial.py":
                raise RuntimeError("unexpected run_trial module resolution")
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            report.update(torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                          versions={p: importlib.metadata.version(p) for p in
                                    ("numpy", "scipy", "torch", "cvxpy", "cvxpylayers", "diffcp", "scs", "moreau", "moreau-cpu", "gurobipy")})
            paths = {name: INPUT / f"{name}.pth" for name in CHECKPOINTS}
            paths.update(trial=INPUT / "trial.json", source=SOURCE, audit=Path(__file__).resolve(),
                         **{name: IMPLEMENTATION / f"{name}.py" for name in ("baseline", "native_projection", "run_trial")})
            report["input_integrity"].update(paths={k: str(v) for k, v in paths.items()}, file_sha256_before=before)
            for name, path in paths.items():
                before[name] = digest(path)
            trial = json.loads(paths["trial"].read_text())
            report["trial_metadata"] = {k: trial.get(k) for k in ("status", "data_seed", "network_seed", "radius", "sample_size", "versions", "protocol")}
            matches = report["input_integrity"]["trial_hash_checks"] = {}
            for name, key in (("source", "source_sha256"), ("initial_state", "initial_state_sha256"),
                              ("scs_selected", "scs_selected_sha256"), ("moreau_selected", "moreau_selected_sha256")):
                expected = trial.get(key)
                matches[name] = {"metadata_key": key, "expected": expected, "actual": before[name],
                                 "matches": None if expected is None else before[name] == expected,
                                 "status": "unavailable_in_trial" if expected is None else "checked"}
                if expected is not None and before[name] != expected:
                    raise RuntimeError(f"trial hash mismatch: {name}")
            for key, expected in (("status", "completed"), ("data_seed", 0), ("radius", 0), ("sample_size", 50)):
                if trial.get(key) != expected:
                    raise RuntimeError(f"unexpected trial {key}: {trial.get(key)}")
            baseline = load_baseline()
            if (baseline.cfg.N_SAMPLES, baseline.cfg.R, baseline.cfg.HIDDEN_WIDTH,
                baseline.cfg.SCS_EPS, baseline.cfg.SCS_MAX_ITERS) != (50, 3, 16, 1e-4, 200000):
                raise RuntimeError("canonical configuration changed")
            data_arrays = list(baseline.generate_data(0))
            raw_hash = array_hash(*data_arrays)
            expected = trial.get("training_data_sha256")
            matches["training_data"] = {"expected": expected, "actual": raw_hash,
                "matches": None if expected is None else raw_hash == expected,
                "status": "unavailable_in_trial" if expected is None else "checked"}
            if expected is not None and raw_hash != expected:
                raise RuntimeError("training-data hash mismatch")
            data_tensors = [torch.tensor(a, dtype=torch.float32) for a in data_arrays]
            report["input_integrity"].update(training_data_sha256_before=raw_hash,
                execution_data_sha256_before=array_hash(*(t.numpy() for t in data_tensors)))
            original_states = {k: torch.load(paths[k], map_location="cpu", weights_only=True) for k in CHECKPOINTS}
            template = baseline.Model()
            names = list(dict(template.named_parameters()))
            if any(set(s) != set(names) for s in original_states.values()):
                raise RuntimeError("checkpoint keys differ from canonical parameters")
            if any(v.dtype != torch.float32 for s in original_states.values() for v in s.values()):
                raise RuntimeError("expected original float32 checkpoint tensors")
            for state in original_states.values():
                load_parameters(template, state, torch)
            report["input_integrity"]["state_tensor_sha256_before"] = {k: state_hash(s) for k, s in original_states.items()}
            report["input_integrity"]["state_nonfinite"] = {k: {n: finite_summary(t.numpy(), np) for n, t in s.items()} for k, s in original_states.items()}
            groups = {"network": [k for k in names if k.startswith("fc_layers.")],
                      "inventory": ["Q10", "Q1r"], "Q10": ["Q10"], "Q1r": ["Q1r"], "all": names}
            report["parameter_shapes"] = {k: list(v.shape) for k, v in original_states["initial_state"].items()}
            report["groups"] = groups
            generator = torch.Generator(device="cpu").manual_seed(20260909)
            directions = {}
            for label in ("network", "Q10", "Q1r"):
                direction = {k: torch.zeros_like(v, dtype=torch.float64) for k, v in original_states["initial_state"].items()}
                for k in groups[label]:
                    direction[k] = torch.randn(direction[k].shape, generator=generator, dtype=torch.float64)
                norm = torch.cat([v.flatten() for v in direction.values()]).norm()
                directions[label] = {k: v / norm for k, v in direction.items()}
            report["directions"] = {label: {"float64": {k: v.tolist() for k, v in d.items()},
                "float32": {k: v.float().tolist() for k, v in d.items()},
                "norm_float64": float(torch.cat([v.flatten() for v in d.values()]).norm()),
                "norm_float32_as_float64": float(torch.cat([v.float().double().flatten() for v in d.values()]).norm())}
                for label, d in directions.items()}
            report["direction_definition"] = "Three seed20260909 Gaussian directions normalized jointly over the named raw-parameter group in float64, zero elsewhere; same directions at every checkpoint. Float32 uses their rounded representation; actual norms and perturbation rounding recorded."

        for checkpoint in CHECKPOINTS:
            for dtype in (torch.float32, torch.float64):
                dtype_name = str(dtype)
                candidate = {"checkpoint": checkpoint, "dtype": dtype_name, "setup": {}, "backends": {}, "reference": {}, "finite_differences": []}
                report["candidates"].append(candidate)
                print(f"audit {checkpoint} {dtype_name}", flush=True)
                with phase(candidate, report):
                    with phase(candidate["setup"], report):
                        state = {k: v.clone().to(dtype) for k, v in original_states[checkpoint].items()}
                        x1, x2 = [t.to(dtype) for t in data_tensors]
                        models = {}
                        for backend in BACKENDS:
                            setup = candidate["setup"][backend] = {}
                            with phase(setup, report):
                                model = baseline.Model().to(dtype)
                                if backend == "moreau":
                                    model.proj_layer = NativeProjection()
                                    candidate["native_settings"] = model.proj_layer.settings
                                    if model.proj_layer.settings != trial["setup"]["moreau"]["native_settings"]:
                                        raise RuntimeError("native settings differ from original trial")
                                load_parameters(model, state, torch)
                                model.eval()
                                models[backend] = model
                            candidate["backend_buffer_names"] = {b: list(dict(m.named_buffers())) for b, m in models.items()}
                            candidate["matched_state_sha256"] = {b: state_hash(parameter_state(m)) for b, m in models.items()}
                        if len(set(candidate["matched_state_sha256"].values())) != 1:
                            raise RuntimeError("backends did not receive identical weights")
                    with phase(candidate["reference"], report):
                        # Reuse the existing strict Gurobi reference, Threads=1.
                        oracle = run_trial.reference_projection(models["scs"], x1, candidate["reference"])
                        candidate["reference"]["q2"] = oracle.numpy().tolist()
                    for backend, model in models.items():
                        record = candidate["backends"][backend] = {}
                        evaluate(model, x1, x2, backend, record, report, baseline.cfg, torch, np, scs, True)
                        delta = np.asarray(record["signature"]["q2"]) - oracle.numpy()
                        record["projection_vs_gurobi"] = {"max_absolute_error": float(np.abs(delta).max()),
                            "frobenius_error": float(np.linalg.norm(delta)), "raw_difference": delta.tolist(),
                            "nonfinite": finite_summary(delta, np)}
                        if state_hash(parameter_state(model)) != state_hash(state):
                            raise RuntimeError("gradient evaluation modified weights")
                        persist(report)
                    candidate["comparison"] = {}
                    with phase(candidate["comparison"], report):
                        candidate["comparison"].update(compare_gradients(candidate["backends"]["scs"], candidate["backends"]["moreau"], groups, np))
                    hs = (.01, .001, .0001) if dtype == torch.float32 else (.001, .0001, .00001)
                    for backend in (("moreau",) if dtype == torch.float32 else BACKENDS):
                        for label, d in directions.items():
                            direction = {k: v.to(dtype) for k, v in d.items()}
                            series = []
                            for h in hs:
                                record = {}
                                candidate["finite_differences"].append(record)
                                fd_block(models[backend], state, direction, label, h, x1, x2, backend,
                                         candidate["backends"][backend], record, report, baseline.cfg, torch, np, scs)
                                if series:
                                    record["central_change_from_previous_h"] = record["central_slope"] - series[-1]["central_slope"]
                                series.append(record)
                                persist(report)
    except BaseException as caught:
        error = caught
        report.update(error=f"{type(caught).__name__}: {caught}", traceback=traceback.format_exc())
    finally:
        try:
            with phase(report["final_integrity"], report):
                final = report["final_integrity"]
                final["file_sha256_after"] = {k: digest(paths[k]) for k in before}
                final["files_unchanged"] = final["file_sha256_after"] == before
                checks = [final["files_unchanged"]]
                if "training_data_sha256_before" in report["input_integrity"]:
                    final["training_data_sha256_after"] = array_hash(*data_arrays)
                    final["execution_data_sha256_after"] = array_hash(*(t.numpy() for t in data_tensors))
                    final["data_unchanged"] = (final["training_data_sha256_after"] == report["input_integrity"]["training_data_sha256_before"] and
                                               final["execution_data_sha256_after"] == report["input_integrity"]["execution_data_sha256_before"])
                    checks.append(final["data_unchanged"])
                if "state_tensor_sha256_before" in report["input_integrity"]:
                    final["state_tensor_sha256_after"] = {k: state_hash(s) for k, s in original_states.items()}
                    final["states_unchanged"] = final["state_tensor_sha256_after"] == report["input_integrity"]["state_tensor_sha256_before"]
                    checks.append(final["states_unchanged"])
                if not all(checks):
                    raise RuntimeError("input changed during audit")
        except BaseException as integrity_error:
            report["integrity_error"] = f"{type(integrity_error).__name__}: {integrity_error}"
            if error is None:
                error = integrity_error
        elapsed = time.perf_counter() - start
        # One audit per configuration: these are measured evaluation timings,
        # not cached training timings and not a cross-seed performance claim.
        report["runtime_table"] = [
            {"method": "nominal_neural_training", "status": "not_run", "runtime_seconds": None},
            {"method": "robust_neural_ccg", "status": "not_run", "runtime_seconds": None},
            {"method": "sro_fa", "status": "not_run", "runtime_seconds": None},
            {"method": "cv_validation_test", "status": "not_run", "runtime_seconds": None},
        ]
        for candidate in report["candidates"]:
            report["runtime_table"].append({"method": "gradient_and_fd_diagnostics",
                "checkpoint": candidate["checkpoint"], "dtype": candidate["dtype"],
                "sample_size": 50, "radius": 0, "seed": 0, "iterations": 0,
                "workers": 1, "status": candidate.get("status"),
                "runtime_seconds": candidate.get("runtime_seconds"),
                "runtime_human": candidate.get("runtime_human")})
        report.update(status="completed" if error is None else "failed", ended_at_utc=utc(),
                      runtime_seconds=elapsed, runtime_human=f"{elapsed:.3f} s ({elapsed / 60:.3f} min)")
        persist(report)
        print(f"audit {report['status']}: {elapsed:.3f}s; {OUTPUT / 'audit.json'}", flush=True)
    if error is not None:
        raise error


if __name__ == "__main__":
    main()