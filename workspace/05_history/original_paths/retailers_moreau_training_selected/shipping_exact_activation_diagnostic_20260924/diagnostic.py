"""Synthetic counterexamples only; not an implementation or validation of an AP."""
import time

START = time.perf_counter()
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import traceback
from contextlib import contextmanager
from fractions import Fraction as F
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = {
    "experiment": ROOT.name, "method": "synthetic_R3_exact_activation_diagnostic",
    "R": 3, "sample_size": None, "prospective_N_only_not_run": [50, 200, 800],
    "seed": 0, "workers": 1, "threads": 1, "radius": None,
    "activation": "q > 0 exactly; no epsilon or denominator limit",
    "solver": {"Threads": 1, "FeasibilityTol": 1e-9, "IntFeasTol": 1e-9,
               "OptimalityTol": 1e-9, "NumericFocus": 3, "MIPGap": 1e-4},
}
REPORT = {"config": CONFIG, "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
          "status": "RUNNING", "phases": [], "cases": []}


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


@contextmanager
def phase(name, records):
    row = {"name": name, "start_utc": utc(), "status": "RUNNING",
           "workers": 1, "threads": 1, "seed": 0, "iterations": None}
    records.append(row)
    start = time.perf_counter()
    try:
        yield row
        row["status"] = "COMPLETED"
    except BaseException:
        row["status"] = "FAILED"
        row["traceback"] = traceback.format_exc()
        raise
    finally:
        row["wall_seconds"] = time.perf_counter() - start
        row["end_utc"] = utc()


def binary(value):
    """Exact rational value of an already-represented binary floating value."""
    return F.from_float(float(value))


def project(z, cap):
    """Standard sorting projection onto q>=0, sum(q)<=cap, in exact R3 arithmetic."""
    assert len(z) == 3 and cap >= 0
    assert all(isinstance(v, F) for v in z) and isinstance(cap, F)
    if cap == 0:
        return (F(0),) * 3
    positive = tuple(max(F(0), v) for v in z)
    if sum(positive) <= cap:
        return positive
    ordered = sorted(z, reverse=True)
    prefix = F(0)
    theta = None
    for k, value in enumerate(ordered, 1):
        prefix += value
        candidate = (prefix - cap) / k
        if value > candidate:
            theta = candidate
    assert theta is not None
    return tuple(max(F(0), value - theta) for value in z)


def support(q):
    return [i for i, value in enumerate(q) if value > 0]


def fee(q):
    return F(len(support(q)), 10)


def squared_distance(q, z):
    return sum((a - b) ** 2 for a, b in zip(q, z))


def feasible_perturbation(row):
    z, cap = (F(2), F(1), F(-1)), F(1)
    delta = binary(2.0 ** -30)
    exact = project(z, cap)
    perturbed = (1 - delta, delta, F(0))
    residual = max(F(0), -min(perturbed), sum(perturbed) - cap)
    assert exact == (F(1), F(0), F(0))
    assert residual == 0 and fee(perturbed) - fee(exact) == F(1, 10)
    assert squared_distance(perturbed, z) - squared_distance(exact, z) == 2 * delta ** 2
    row.update(z=z, capacity=cap, delta=delta, exact_projection=exact,
               feasible_perturbation=perturbed, primal_feasibility_residual=residual,
               exact_fee=fee(exact), perturbed_fee=fee(perturbed),
               squared_distance_gap=2 * delta ** 2,
               interpretation="Zero primal residual is not projection optimality or exact support.")


def tiny_positive(row):
    z = (binary(2.0 ** -21), F(0), F(-1))
    q = project(z, F(1))
    assert q == (z[0], F(0), F(0)) and fee(q) == F(1, 10)
    row.update(z=z, capacity=F(1), exact_projection=q, exact_support=support(q), fee=fee(q))


def zero_capacity(row):
    z = (F(2), F(1), F(-1))
    q = project(z, F(0))
    assert q == (F(0),) * 3 and fee(q) == 0
    row.update(z=z, capacity=F(0), exact_projection=q, exact_support=support(q), fee=fee(q))


def network_rounding(row):
    phases = row.setdefault("phases", [])
    with phase("fixture_construction", phases):
        net = torch.nn.Sequential(torch.nn.Linear(7, 16), torch.nn.ReLU(),
                                  torch.nn.Linear(16, 16), torch.nn.ReLU(),
                                  torch.nn.Linear(16, 3)).to(dtype=torch.float32)
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            net[0].weight[0, 0] = 1
            net[0].weight[0, 3] = 1
            net[0].bias[0] = -1
            net[2].weight[0, 0] = 1
            net[4].weight[0, 0] = 1
        net.eval()
        x = torch.tensor([[2.0 ** -25, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)
    with phase("native_float32_forward", phases):
        with torch.no_grad():
            observed = net(x).numpy()[0]
    with phase("exact_unfolded_forward_and_checks", phases):
        values = [binary(v) for v in x.numpy()[0]]
        # Ordinary dense affine/ReLU operations over stored binary coefficients;
        # no folding of the I or Q inputs and no alternative production policy.
        for layer in net:
            if isinstance(layer, torch.nn.Linear):
                weights = layer.weight.detach().numpy()
                biases = layer.bias.detach().numpy()
                values = [binary(b) + sum(binary(w) * v for w, v in zip(ws, values))
                          for ws, b in zip(weights, biases)]
            else:
                values = [max(F(0), v) for v in values]
        assert values == [binary(2.0 ** -25), F(0), F(0)]
        native = tuple(binary(v) for v in observed)
        exact_q, native_q = project(tuple(values), F(1)), project(native, F(1))
        reproduced = native != tuple(values)
        row.update(input_binary_exact=[binary(v) for v in x.numpy()[0]],
                   native_float32_target=observed.tolist(), native_target_exact=native,
                   exact_unfolded_target=values, exact_projection=exact_q,
                   native_target_exact_projection=native_q,
                   exact_fee=fee(exact_q), native_target_fee=fee(native_q),
                   counterexample_reproduced=reproduced,
                   outcome="REPRODUCED" if reproduced else "NOT_REPRODUCED_ON_THIS_BACKEND")


def projection_target_rounding(row):
    cap = binary(2.0 ** -26)
    z = (F(1), F(1) - binary(2.0 ** -25), F(-1))
    cast = tuple(binary(v) for v in np.array([float(v) for v in z], dtype=np.float32))
    exact, rounded = project(z, cap), project(cast, cap)
    assert exact == (cap, F(0), F(0))
    assert cast == (F(1), F(1), F(-1)) and rounded == (cap / 2, cap / 2, F(0))
    assert support(exact) == [0] and support(rounded) == [0, 1]
    row.update(capacity=cap, exact_target=z, float32_cast_target=cast,
               exact_projection=exact, cast_target_projection=rounded,
               exact_support=support(exact), cast_support=support(rounded),
               exact_fee=fee(exact), cast_fee=fee(rounded))


def gurobi_shipping_relaxation(row):
    phases = row.setdefault("phases", [])
    with phase("environment_and_KKT_model_construction", phases):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("SYNTHETIC_CLOSED_SHIPPING_RELAXATION_NOT_AP", env=env)
        for key, value in CONFIG["solver"].items():
            model.setParam(key, value)
        q = model.addVars(3, lb=0, ub=1, name="q")
        mu = model.addVars(3, lb=0, name="mu")
        lam = model.addVar(lb=0, name="lambda")
        slack = model.addVar(lb=0, name="slack")
        active = model.addVars(3, vtype=gp.GRB.BINARY, name="complementarity")
        cap_active = model.addVar(vtype=gp.GRB.BINARY, name="capacity_complementarity")
        ship = model.addVars(3, vtype=gp.GRB.BINARY, name="zship")
        model.addConstr(slack == 1 - gp.quicksum(q.values()))
        model.addGenConstrIndicator(cap_active, 0, lam == 0)
        model.addGenConstrIndicator(cap_active, 1, slack == 0)
        for i in range(3):
            model.addConstr(q[i] + 1 + lam - mu[i] == 0)
            model.addGenConstrIndicator(active[i], 0, q[i] == 0)
            model.addGenConstrIndicator(active[i], 1, mu[i] == 0)
            model.addConstr(q[i] <= ship[i])  # Q=1; deliberately no positive lower link.
        model.setObjective(gp.quicksum(ship.values()), gp.GRB.MAXIMIZE)
    try:
        with phase("solver_optimize", phases) as timing:
            model.optimize()
            timing["solver_status"] = model.Status
            timing["iterations"] = model.IterCount
            row["solver_status"] = model.Status
            assert model.Status == gp.GRB.OPTIMAL, f"Expected OPTIMAL, got {model.Status}"
        with phase("exact_witness_and_solver_checks", phases):
            actual = project((F(-1),) * 3, F(1))
            # Exact feasible relaxation witness q=0, mu=1, lambda=0, slack=1,
            # complementarity=0, capacity_complementarity=0, ship=1.
            assert actual == (F(0),) * 3 and fee(actual) == 0
            assert all(v + 1 - F(1) == 0 and v <= 1 for v in actual)
            assert 1 - sum(actual) == 1
            assert model.ObjVal == 3 and all(ship[i].X == 1 for i in range(3))
            assert all(q[i].X == 0 for i in range(3))
            row.update(label="CLOSED_SHIPPING_RELAXATION_NOT_AP_SOLUTION",
                       target=(F(-1),) * 3, capacity=F(1),
                       solver_q=[q[i].X for i in range(3)], solver_zship=[ship[i].X for i in range(3)],
                       solver_mu=[mu[i].X for i in range(3)], solver_lambda=lam.X,
                       solver_slack=slack.X, solver_objective=model.ObjVal,
                       solver_objective_bound=model.ObjBound, solver_mip_gap=model.MIPGap,
                       iterations=model.IterCount, nodes=model.NodeCount,
                       gurobi_reported_runtime_seconds=model.Runtime,
                       exact_witness={"q": actual, "mu": [F(1)] * 3, "lambda": F(0),
                                      "slack": F(1), "zship": [1] * 3,
                                      "complementarity": [0] * 3, "capacity_complementarity": 0},
                       actual_fee=fee(actual), relaxation_fee=F(3, 10), ghost_fee_gap=F(3, 10),
                       unit_fee_objective_gap=F(3))
    finally:
        model.dispose()
        env.dispose()


def unattained_supremum(row):
    def cost(x1, x2):
        inventory = (F(3), F(4), F(4))
        q = project((x1[0], F(0), F(0)), F(1))
        base = F(1, 4) * (1 + sum(inventory)) + F(1, 20)
        variable = sum(max(F(1, 2) * (a + b - i - c) - F(1, 20) * c,
                           F(1, 20) * (i - a - b),
                           F(1, 2) * (a - i) - F(1, 20) * b)
                       for a, b, i, c in zip(x1, x2, inventory, q))
        return base + variable + fee(q)

    chain = [F(0), F(1), F(1, 1024), binary(2.0 ** -21),
             binary(2.0 ** -30), binary(2.0 ** -60)]
    results = []
    for t in chain:
        formula = F(33, 10) - t / 20 + (F(1, 10) if t > 0 else F(0))
        corners = [cost((t, a, b), x2) for a, b in product([F(0), F(1)], repeat=2)
                   for x2 in product([F(2), F(3)], repeat=3)]
        assert max(corners) == formula and formula < F(17, 5)
        results.append({"t": t, "exact_cost": formula, "gap_to_supremum": F(17, 5) - formula})
    assert all(a["exact_cost"] < b["exact_cost"] for a, b in zip(results[1:], results[2:]))
    row.update(radius=F(1, 2), capacity=F(1), center_x1=[F(1, 2)] * 3,
               center_x2=[F(5, 2)] * 3, inventory=[F(3), F(4), F(4)],
               chain=results, corners_per_t=32, supremum=F(17, 5), maximum_attained=False,
               proof="For t=0, C=33/10<17/5. For every 0<t<=1, C=17/5-t/20<17/5. "
                     "As t decreases to zero through positives, C tends to 17/5. "
                     "Finite checks reproduce examples; this argument proves nonattainment.")


def encode(value):
    if isinstance(value, F):
        return {"numerator": value.numerator, "denominator": value.denominator,
                "exact": str(value)}
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    try:
        with phase("dependency_import_and_single_thread_setup", REPORT["phases"]):
            import numpy as np
            import torch
            import gurobipy as gp
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.manual_seed(0)
            torch.use_deterministic_algorithms(True)
            REPORT["environment"] = {
                "hostname": platform.node(), "platform": platform.platform(),
                "python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                "torch": torch.__version__, "gurobi": gp.gurobi.version(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "torch_intraop_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "torch_build": torch.__config__.show(),
                "thread_env": {k: os.environ.get(k) for k in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
        for case in (feasible_perturbation, tiny_positive, zero_capacity, network_rounding,
                     projection_target_rounding, gurobi_shipping_relaxation, unattained_supremum):
            with phase(case.__name__, REPORT["cases"]) as row:
                case(row)
            print(f"{case.__name__}: {row.get('outcome', row['status'])}; {row['wall_seconds']:.6f}s", flush=True)
        REPORT["status"] = "COMPLETED_SYNTHETIC_CHECKS_NOT_PRODUCTION_AP_VALIDATION"
    except BaseException:
        REPORT["status"] = "FAILED"
        REPORT["traceback"] = traceback.format_exc()
        raise
    finally:
        REPORT["end_utc"] = utc()
        REPORT["diagnostic_wall_seconds_before_final_serialization"] = time.perf_counter() - START
        REPORT["completed_cases"] = sum(r["status"] == "COMPLETED" for r in REPORT["cases"])
        REPORT["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (ROOT / "diagnostic.py", ROOT / "launch.py", ROOT / "README.md")}
        (ROOT / "summary.json").write_text(json.dumps(REPORT, default=encode, indent=2) + "\n")