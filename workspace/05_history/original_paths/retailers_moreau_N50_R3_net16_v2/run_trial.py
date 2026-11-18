"""Seed0 nominal SCS/Moreau trial; no held-out data or old checkpoint loading."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from unittest.mock import patch

import gurobipy as gp
import numpy as np
import scs
import torch
from baseline import SOURCE, load_baseline
from native_projection import NativeProjection, derivative_fixture

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()
            if not k.startswith("proj_layer.")}


def save(report):
    temp = OUT / "trial.tmp"
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(OUT / "trial.json")


@contextmanager
def timed(record):
    start = time.perf_counter()
    record.update(status="running", started_at_utc=datetime.now(timezone.utc).isoformat())
    try:
        yield
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record.update(runtime_seconds=time.perf_counter() - start,
                      ended_at_utc=datetime.now(timezone.utc).isoformat())


def stats(values):
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values))


def reference_projection(model, x1, record):
    with timed(record), torch.no_grad(), gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        q10 = torch.nn.functional.softplus(model.Q10)
        q1r = torch.nn.functional.softplus(model.Q1r)
        z = model.fc_layers(torch.cat([x1, q1r.expand(len(x1), -1), q10.expand(len(x1), 1)], dim=1))
        with gp.Model(env=env) as qp:
            qp.Params.Threads = 1
            qp.Params.FeasibilityTol = 1e-9
            qp.Params.OptimalityTol = 1e-9
            qp.Params.BarConvTol = 1e-12
            q = qp.addMVar(3, lb=0)
            qp.addConstr(q.sum() <= float(q10.item()) + 1e-6)
            solutions, statuses, iterations = [], [], []
            for target in z.numpy():
                qp.setObjective(.5 * (q @ q) - target @ q)
                qp.optimize()
                statuses.append(qp.Status)
                iterations.append(float(qp.IterCount + qp.BarIterCount))
                if qp.Status != gp.GRB.OPTIMAL:
                    raise RuntimeError(f"reference QP status {qp.Status}")
                solutions.append(q.X.copy())
            record.update(status_codes=statuses, iterations=iterations)
        return torch.tensor(np.asarray(solutions), dtype=torch.float64)


def reference_cost(model, x1, x2, record, cfg):
    q2 = reference_projection(model, x1, record)
    with torch.no_grad():
        q10 = torch.nn.functional.softplus(model.Q10)
        q1r = torch.nn.functional.softplus(model.Q1r)
        t1 = cfg.b * (x2 + x1 - q2 - q1r) - cfg.h * q2
        t2 = cfg.h * (q1r - x1 - x2)
        t3 = cfg.b * (x1 - q1r) - cfg.h * x2
        value = (cfg.c * (q10 + q1r.sum()) + cfg.h * q10 + torch.maximum(torch.maximum(t1, t2), t3).sum(1)).mean()
        record["training_mean_cost"] = float(value)


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    baseline = load_baseline()
    historical_path = ROOT.parent / "retailers_problem/R3/N50/nominal_timing_net16_v2/seed_0_summary.json"
    historical = json.loads(historical_path.read_text())
    report = {"experiment": "retailers_N50_R3_net16_v2_nominal_matched", "data_seed": 0,
        "network_seed": 0, "radius": 0, "sample_size": 50, "hidden_widths": [16, 16],
        "epochs_cap": 3000, "early_stop_patience": 300, "lr_net": .05, "lr_inventory": .01,
        "source": str(SOURCE), "source_sha256": digest(SOURCE),
        "historical_reference": historical, "historical_sha256": digest(historical_path),
        "command": [sys.executable, *sys.argv], "cwd": os.getcwd(), "host": platform.node(),
        "cpu_count": os.cpu_count(), "parallel_model_workers": 1, "torch_threads": 1,
        "native_threads": "unmeasured; diffcp default pools retained",
        "versions": {p: importlib.metadata.version(p) for p in
            ("torch", "cvxpy", "cvxpylayers", "diffcp", "scs", "moreau", "moreau-cpu", "gurobipy")},
        "thread_environment": {k: os.environ.get(k) for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")},
        "protocol": {"selection": "corrected evaluated pre-update weights in BOTH arms",
            "initialization": "fresh network seed0 plus one shared seed0 SAA solve, no trained checkpoint",
            "data": "training only; no validation, test, LR search or radius selection",
            "execution_order": "SCS then Moreau training, same shared initial state",
            "projection_buffer": 1e-6, "native": "P=I q=-z, library exact active-set derivatives",
            "failure": "abort/preserve; no trial-level retries/fallback/tolerance relaxation; legacy diffcp SCS retry and inaccurate-result acceptance retained",
            "timing": "monotonic seconds; one seed, no causal isolated-solver speed claim",
            "checkpoint_scope": "other retailer script copies and existing artifacts unchanged"},
        "fixtures": {}, "saa": {}, "setup": {}, "initial_reference": {}, "fixed_weight_passes": [], "arms": {}}
    save(report)
    try:
        with timed(report):
            with timed(report["fixtures"]):
                report["fixtures"].update(derivative_fixture())
            x1_np, x2_np = baseline.generate_data(0)
            x1, x2 = torch.tensor(x1_np, dtype=torch.float32), torch.tensor(x2_np, dtype=torch.float32)
            report["training_data_sha256"] = hashlib.sha256(x1_np.tobytes() + x2_np.tobytes()).hexdigest()
            created = []
            original_model = gp.Model
            def track(*args, **kwargs):
                model = original_model(*args, **kwargs)
                created.append(model)
                return model
            with timed(report["saa"]):
                try:
                    with patch.object(gp, "Model", side_effect=track):
                        q10, q1r, objective = baseline.solve_saa(x1_np, x2_np)
                    record = created[0]
                    report["saa"].update(objective=float(objective), q10=float(q10), q1r=q1r.tolist(),
                        solver_status=record.Status, solver_seconds=record.Runtime,
                        iterations=float(record.IterCount + record.BarIterCount), threads=record.Params.Threads)
                    if record.Status != gp.GRB.OPTIMAL:
                        raise RuntimeError("SAA not optimal")
                finally:
                    for record in created:
                        record.dispose()
            models = {}
            for backend in ("scs", "moreau"):
                setup = report["setup"][backend] = {}
                with timed(setup):
                    model = baseline.make_model_saa_init(q10, q1r, net_seed=0)
                    if backend == "moreau":
                        model.proj_layer = NativeProjection()
                        setup["native_settings"] = model.proj_layer.settings
                    models[backend] = model
            initial = weights(models["scs"])
            if not all(torch.equal(v, weights(models["moreau"])[k]) for k, v in initial.items()):
                raise RuntimeError("initial state mismatch")
            torch.save(initial, OUT / "initial_state.pth")
            report["initial_state_sha256"] = digest(OUT / "initial_state.pth")
            oracle = reference_projection(models["scs"], x1, report["initial_reference"])
            for repeat in range(4):
                for backend in (("scs", "moreau") if repeat % 2 == 0 else ("moreau", "scs")):
                    model = models[backend]
                    record = {"backend": backend, "repeat": repeat, "forward": {}, "backward": {}}
                    report["fixed_weight_passes"].append(record)
                    model.zero_grad(set_to_none=True)
                    capture = []
                    hook = model.proj_layer.register_forward_hook(lambda module, args, output: capture.append(output[0].detach()))
                    try:
                        with timed(record):
                            with timed(record["forward"]):
                                loss = model.per_sample_cost(x1, x2).mean()
                            with timed(record["backward"]):
                                loss.backward()
                            if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                                raise RuntimeError("nonfinite or absent parameter gradient")
                            record.update(loss=float(loss.detach()), projection_max_error=float((capture[0]-oracle).abs().max()))
                            if backend == "moreau" and record["projection_max_error"] > 1e-5:
                                raise RuntimeError("native/reference initial projection mismatch")
                    finally:
                        hook.remove()
                        save(report)
            report["fixed_weight_summary"] = {b: {phase: stats([r[phase]["runtime_seconds"] for r in report["fixed_weight_passes"]
                if r["backend"] == b and r["repeat"] > 0]) for phase in ("forward", "backward")} for b in models}
            for backend, model in models.items():
                arm = report["arms"][backend] = {"training": {}, "evaluation_own": {}, "evaluation_reference": {}, "epochs": []}
                calls = []
                original_scs = scs.solve
                def observe_scs(*args, **kwargs):
                    result = original_scs(*args, **kwargs)
                    calls.append({k: result["info"].get(k) for k in ("status", "status_val", "iter")})
                    return result
                def check_grad(grad):
                    if not bool(torch.isfinite(grad).all()):
                        raise RuntimeError("nonfinite training gradient")
                    return grad
                handles = [p.register_hook(check_grad) for p in model.parameters()]
                def loss_fn(current):
                    start = time.perf_counter()
                    value = current.per_sample_cost(x1, x2).mean()
                    if not bool(torch.isfinite(value)):
                        raise RuntimeError("nonfinite training loss")
                    if backend == "moreau":
                        solver_calls = current.proj_layer.calls
                        counts = Counter(str(s) for c in solver_calls for s in c["status_codes"])
                        iterations = [i for c in solver_calls for i in c["iterations"]]
                    else:
                        counts = Counter(str(c["status_val"]) for c in calls)
                        iterations = [c["iter"] for c in calls]
                    arm["epochs"].append({"epoch": len(arm["epochs"])+1, "loss": float(value.detach()),
                        "forward_seconds": time.perf_counter()-start, "solver_status_counts": dict(counts),
                        "solver_iterations": stats(iterations) if iterations else None})
                    calls.clear()
                    if backend == "moreau":
                        current.proj_layer.calls.clear()
                    if len(arm["epochs"]) % 100 == 0:
                        save(report)
                    return value
                if backend == "moreau":
                    model.proj_layer.calls.clear()
                try:
                    with timed(arm["training"]), patch.object(scs, "solve", side_effect=observe_scs):
                        best, state, epochs, trainer_seconds = baseline.train(model, loss_fn, backend)
                    arm["training"].update(best_evaluated_loss=best, epochs_completed=epochs,
                        optimizer_updates_completed=epochs, trainer_seconds=trainer_seconds)
                    torch.save(weights(model), OUT / f"{backend}_selected.pth")
                    with timed(arm["evaluation_own"]), torch.no_grad():
                        arm["evaluation_own"]["training_mean_cost"] = float(model.per_sample_cost(x1, x2).mean())
                    reference_cost(model, x1, x2, arm["evaluation_reference"], baseline.cfg)
                except BaseException:
                    arm["failure_solver_calls"] = model.proj_layer.calls if backend == "moreau" else list(calls)
                    torch.save(weights(model), OUT / f"{backend}_failed_state.pth")
                    raise
                finally:
                    for handle in handles:
                        handle.remove()
                    save(report)
                print(json.dumps({"backend": backend, "training": arm["training"],
                    "own_training_cost": arm["evaluation_own"]["training_mean_cost"],
                    "reference_training_cost": arm["evaluation_reference"]["training_mean_cost"]}), flush=True)
    finally:
        report["source_unchanged_during_trial"] = digest(SOURCE) == report["source_sha256"]
        report["historical_unchanged"] = digest(historical_path) == report["historical_sha256"]
        save(report)
    print(json.dumps({"status": report["status"], "runtime_seconds": report["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()