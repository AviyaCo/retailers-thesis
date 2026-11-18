"""Compare width-8, width-13, and width-16 OOS curves with kNN."""

import json
import os
import platform
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import gurobipy as gp
import numpy as np
import torch
import torch.nn as nn
from gurobipy import GRB
from scipy.stats import truncnorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


BASE_DIR = Path("/data/home/aviyac/retailers_problem/R3/N50")
NET8_SWEEP_DIR = BASE_DIR / "rho_sweep_net8_v2"
NET8_RHO_04_DIR = BASE_DIR / "saainit_net8_v2"
NET13_SWEEP_DIR = BASE_DIR / "rho_sweep_net13_v2"
NET13_NOMINAL_DIR = BASE_DIR / "saainit_net13_v2"
NET16_EVAL_PATH = BASE_DIR / "rho_sweep_net16_v2" / "sweep_eval_net16_v2_10k.json"
NET11_EVAL_PATH = BASE_DIR / "width11_oos_10k.json"
NET11_IS_PATH = BASE_DIR / "width11_augmented_is.json"
LOWER_BOUND_PATH = BASE_DIR.parent / "saa_cond_lb_large.json"
OUTPUT_JSON = BASE_DIR / "net8_vs_net16_all_radii_10k.json"
OUTPUT_PNG = BASE_DIR / "net8_vs_net16_all_radii_10k.png"
OUTPUT_PDF = BASE_DIR / "net8_vs_net16_all_radii_10k.pdf"

R = 3
N_TEST = 10_000
TEST_SEED = 9999
VALIDATION_SIZE = 13
VALIDATION_SEED_BASE = 88880
SEEDS = list(range(10))
COST_C = 0.25
COST_H = 0.05
COST_B = 0.5
PROJECTION_METHOD = "gurobi_qp"
EVALUATOR_VERSION = "gurobi_qp_final_mp_is_v2"
KNN_SELECTION_VERSION = "per_seed_loo_grid_v3"

torch.set_num_threads(1)


def generate_data(n_samples, seed):
    np.random.seed(seed)
    lower_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(
        lower_xi1, np.inf, loc=6.0, scale=2.5, size=(n_samples, R)
    )
    mean_xi2 = 2 * (6.0 - xi1) ** 2
    lower_xi2 = (0.0 - mean_xi2) / 2.5
    xi2 = truncnorm.rvs(lower_xi2, np.inf, loc=mean_xi2, scale=2.5)
    return xi1, xi2


_gurobi_environment = None


def get_gurobi_environment():
    global _gurobi_environment
    if _gurobi_environment is None:
        _gurobi_environment = gp.Env(empty=True)
        _gurobi_environment.setParam("OutputFlag", 0)
        _gurobi_environment.start()
    return _gurobi_environment


def project_with_gurobi(raw_values, capacity):
    projected = []
    for raw_row in np.asarray(raw_values, dtype=float):
        model = gp.Model(env=get_gurobi_environment())
        model.setParam("OutputFlag", 0)
        q2 = model.addVars(R, lb=0.0)
        model.addConstr(gp.quicksum(q2[index] for index in range(R)) <= capacity)
        model.setObjective(
            gp.quicksum(
                (q2[index] - float(raw_row[index])) ** 2 for index in range(R)
            ),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.status != GRB.OPTIMAL:
            status = model.status
            model.dispose()
            raise RuntimeError(f"Gurobi projection failed with status {status}")
        projected.append([q2[index].X for index in range(R)])
        model.dispose()
    return np.asarray(projected)


class Model(nn.Module):
    def __init__(self, hidden_width):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(7, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, R),
        )
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))

    @torch.no_grad()
    def decisions(self, xi1):
        xi1_tensor = torch.as_tensor(xi1, dtype=torch.float32)
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).cpu().numpy()
        network_input = torch.cat(
            [
                xi1_tensor,
                torch.as_tensor(q1r, dtype=torch.float32)
                .unsqueeze(0)
                .expand(len(xi1_tensor), -1),
                torch.full((len(xi1_tensor), 1), q10),
            ],
            dim=1,
        )
        raw_q2 = self.fc_layers(network_input).cpu().numpy()
        q2 = project_with_gurobi(raw_q2, q10)
        return q2, q10, q1r

    @torch.no_grad()
    def mean_cost(self, xi1, xi2):
        q2, q10, q1r = self.decisions(xi1)
        return float(scenario_costs(q10, q1r, q2, xi1, xi2).mean())


def scenario_costs(q10, q1r, q2, xi1, xi2):
    stage_one = COST_C * (q10 + q1r.sum()) + COST_H * q10
    recourse = np.maximum(
        COST_B * (xi1 + xi2 - q1r - q2) - COST_H * q2,
        np.maximum(
            COST_H * (q1r - xi1 - xi2),
            COST_B * (xi1 - q1r) - COST_H * xi2,
        ),
    )
    return stage_one + recourse.sum(axis=1)


def exact_augmented_is(model, directory, seed, mp_iteration, xi1, xi2):
    adversarial_xi1 = {}
    adversarial_xi2 = {}
    for iteration in range(1, mp_iteration + 1):
        ap_path = directory / f"seed_{seed}_ap_iter{iteration}.json"
        with ap_path.open() as stream:
            scenarios = json.load(stream)
        ball_ids = [int(entry["ball_id"]) for entry in scenarios]
        if sorted(ball_ids) != list(range(len(xi1))):
            raise RuntimeError(f"Incomplete AP file: {ap_path}")
        for entry in scenarios:
            ball_id = int(entry["ball_id"])
            adversarial_xi1.setdefault(ball_id, []).append(entry["worst_xi1"])
            adversarial_xi2.setdefault(ball_id, []).append(entry["worst_xi2"])

    columns_xi1 = [xi1]
    columns_xi2 = [xi2]
    if adversarial_xi1:
        column_count = max(len(values) for values in adversarial_xi1.values())
        columns_xi1.extend(xi1.copy() for _ in range(column_count))
        columns_xi2.extend(xi2.copy() for _ in range(column_count))
        for ball_id, values in adversarial_xi1.items():
            for column, value in enumerate(values, start=1):
                columns_xi1[column][ball_id] = value
                columns_xi2[column][ball_id] = adversarial_xi2[ball_id][column - 1]

    per_column_costs = []
    for column_xi1, column_xi2 in zip(columns_xi1, columns_xi2):
        q2, q10, q1r = model.decisions(column_xi1)
        per_column_costs.append(
            scenario_costs(q10, q1r, q2, column_xi1, column_xi2)
        )
    return float(np.stack(per_column_costs, axis=1).max(axis=1).mean())


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    hidden_width = int(state_dict["fc_layers.0.weight"].shape[0])
    model = Model(hidden_width)
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_unexpected = {"proj_layer.q", "proj_layer.A"}
    if incompatible.missing_keys or not set(incompatible.unexpected_keys).issubset(
        allowed_unexpected
    ):
        raise RuntimeError(f"Incompatible checkpoint: {checkpoint_path}")
    model.eval()
    return model


def source_for_rho(rho):
    if abs(rho - 0.4) < 1e-12:
        return NET8_RHO_04_DIR
    return next(
        path
        for path in NET8_SWEEP_DIR.glob("rho_*")
        if abs(float(path.name[4:].replace("p", ".")) - rho) < 1e-12
    )


def incumbent_checkpoint(rho, seed):
    directory = source_for_rho(rho)
    summary_path = directory / f"seed_{seed}_summary.json"
    with summary_path.open() as stream:
        summary = json.load(stream)
    if summary.get("completed") is not True:
        raise RuntimeError(f"Incomplete summary: {summary_path}")
    if "final_mp_iter" in summary:
        mp_iteration = int(summary["final_mp_iter"])
        best_ap_iteration = int(summary["best_ap_iter"])
    else:
        best_ap = min(
            summary["ccg_trajectory"], key=lambda entry: entry["ap_mean"]
        )
        best_ap_iteration = int(best_ap["iter"])
        mp_iteration = best_ap_iteration - 1
    checkpoint_path = (
        Path(summary.get("nominal_checkpoint", directory / f"seed_{seed}_nominal.pth"))
        if mp_iteration <= 0
        else directory / f"seed_{seed}_mp_iter{mp_iteration}.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path, best_ap_iteration, mp_iteration


def net13_incumbent_checkpoint(rho, seed):
    directory = next(
        path
        for path in NET13_SWEEP_DIR.glob("rho_*")
        if abs(float(path.name[4:].replace("p", ".")) - rho) < 1e-12
    )
    summary_path = directory / f"seed_{seed}_summary.json"
    with summary_path.open() as stream:
        summary = json.load(stream)
    if summary.get("completed") is not True:
        raise RuntimeError(f"Incomplete summary: {summary_path}")
    best_ap_iteration = int(summary["best_ap_iter"])
    mp_iteration = int(summary["final_mp_iter"])
    checkpoint_path = (
        NET13_NOMINAL_DIR / f"seed_{seed}_nominal.pth"
        if mp_iteration <= 0
        else directory / f"seed_{seed}_mp_iter{mp_iteration}.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path, best_ap_iteration, mp_iteration


def curve_statistics(values):
    array = np.asarray(values, dtype=float)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "percentile_20": float(np.percentile(array, 20)),
        "percentile_80": float(np.percentile(array, 80)),
    }


def plot_results(result):
    rhos = np.array(result["rhos"], dtype=float)
    net8_is_means = np.array(
        [result["net8_is"][str(rho)]["mean"] for rho in rhos]
    )
    net16_is_means = np.array(
        [result["net16_is"][str(rho)]["mean"] for rho in rhos]
    )
    net13_is_means = np.array(
        [result["net13_is"][str(rho)]["mean"] for rho in rhos]
    )
    net8_means = np.array([result["net8"][str(rho)]["mean"] for rho in rhos])
    net8_low = np.array(
        [result["net8"][str(rho)]["percentile_20"] for rho in rhos]
    )
    net8_high = np.array(
        [result["net8"][str(rho)]["percentile_80"] for rho in rhos]
    )
    net16_means = np.array([result["net16"][str(rho)]["mean"] for rho in rhos])
    net16_low = np.array(
        [result["net16"][str(rho)]["percentile_20"] for rho in rhos]
    )
    net16_high = np.array(
        [result["net16"][str(rho)]["percentile_80"] for rho in rhos]
    )
    net13_means = np.array([result["net13"][str(rho)]["mean"] for rho in rhos])
    net13_low = np.array(
        [result["net13"][str(rho)]["percentile_20"] for rho in rhos]
    )
    net13_high = np.array(
        [result["net13"][str(rho)]["percentile_80"] for rho in rhos]
    )
    net11 = result.get("net11")
    net11_is = result.get("net11_is")

    fig, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.plot(rhos, net16_means, "o-", color="#2166ac", lw=2, ms=3.5, label="NN width 16 OOS")
    axis.fill_between(rhos, net16_low, net16_high, color="#2166ac", alpha=0.14)
    axis.plot(rhos, net16_is_means, "--", color="#2166ac", lw=1.7, label="NN width 16 IS")
    axis.plot(rhos, net8_means, "s-", color="#c43c39", lw=2, ms=3.5, label="NN width 8 OOS")
    axis.fill_between(rhos, net8_low, net8_high, color="#c43c39", alpha=0.14)
    axis.plot(rhos, net8_is_means, "--", color="#c43c39", lw=1.7, label="NN width 8 IS")
    axis.plot(rhos, net13_means, "^-", color="#7b3294", lw=2, ms=3.8, label="NN width 13 OOS")
    axis.fill_between(rhos, net13_low, net13_high, color="#7b3294", alpha=0.14)
    axis.plot(rhos, net13_is_means, "--", color="#7b3294", lw=1.7, label="NN width 13 IS")
    if net11 is not None:
        net11_means = np.array([net11[str(rho)]["mean"] for rho in rhos])
        net11_low = np.array(
            [net11[str(rho)]["percentile_20"] for rho in rhos]
        )
        net11_high = np.array(
            [net11[str(rho)]["percentile_80"] for rho in rhos]
        )
        axis.plot(
            rhos,
            net11_means,
            "D-",
            color="#1f6f5f",
            lw=2,
            ms=3.5,
            label="NN width 11 OOS (post-hoc)",
        )
        axis.fill_between(rhos, net11_low, net11_high, color="#1f6f5f", alpha=0.14)
        if net11_is is not None:
            net11_is_means = np.array(
                [net11_is[str(rho)]["mean"] for rho in rhos]
            )
            axis.plot(
                rhos,
                net11_is_means,
                "--",
                color="#1f6f5f",
                lw=1.7,
                label="NN width 11 IS",
            )
    axis.axhline(result["knn"]["mean"], color="#d69e00", ls="-.", lw=1.8, label="SAA-kNN OOS")
    axis.axhline(result["lower_bound"]["value"], color="#2f855a", ls="--", lw=1.8, label="Conditional clairvoyant LB")

    display_ticks = [
        rho
        for rho in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
        if rho in rhos
    ]
    axis.set_xscale("log")
    axis.set_xlim(rhos.min() / 1.15, rhos.max() * 1.15)
    axis.set_xticks(display_ticks)
    axis.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda value, _: f"{value:g}")
    )
    axis.tick_params(axis="x", labelsize=8.5, rotation=45)
    axis.set_xlabel("Training radius $\\rho$")
    axis.grid(axis="y", color="#cccccc", alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)

    axis.set_ylabel("Mean cost")
    axis.set_title("In-sample and out-of-sample performance")
    fig.suptitle(
        "Retailers R=3, N=50: widths 8, 11, 13, and 16 under robustness\n"
        "10,000 common test scenarios, seed 9999",
        fontsize=15,
    )
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.2)
    fig.subplots_adjust(bottom=0.27, top=0.81)
    fig.savefig(OUTPUT_PNG, dpi=250, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def atomic_json_dump(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as stream:
        json.dump(payload, stream, indent=2)
    temporary_path.replace(path)


def evaluate_and_merge_width11_is():
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with OUTPUT_JSON.open() as stream:
        comparison = json.load(stream)
    rhos = comparison.get("rhos", [])
    checkpoints = comparison.get("checkpoint_provenance", {}).get("net11", {})
    if len(rhos) != 23 or len(checkpoints) != len(rhos):
        raise RuntimeError("Incomplete width-11 checkpoint provenance")

    for rho in rhos:
        for seed in SEEDS:
            details = checkpoints.get(str(rho), {}).get(str(seed))
            if details is None:
                raise RuntimeError(
                    f"Missing width-11 provenance at rho={rho:g}, seed={seed}"
                )
            checkpoint_path = Path(details["checkpoint"])
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            mp_iteration = int(details["incumbent_mp_iteration"])
            for iteration in range(1, mp_iteration + 1):
                ap_path = checkpoint_path.parent / f"seed_{seed}_ap_iter{iteration}.json"
                if not ap_path.exists():
                    raise FileNotFoundError(ap_path)

    artifact = {
        "meta": {
            "experiment": "R3_N50_width11_complete_augmented_is",
            "status": "running",
            "hidden_widths": [11, 11],
            "n_train": 50,
            "training_seeds": SEEDS,
            "radii": rhos,
            "projection_method": PROJECTION_METHOD,
            "evaluator_version": EVALUATOR_VERSION,
            "in_sample_definition": "mean per-training-sample maximum over nominal and AP columns used by the selected MP checkpoint",
            "incumbent_rule": "minimum AP mean; AP iteration t evaluates MP iteration t-1",
            "interval": "20th-80th percentile across training seeds",
        },
        "net11_is": {},
        "checkpoint_provenance_source": str(OUTPUT_JSON),
        "runtime": {
            "started_at_utc": started_at,
            "status": "running",
            "per_radius": {},
            "parallel_workers": 1,
            "thread_limit": 1,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    atomic_json_dump(NET11_IS_PATH, artifact)

    try:
        for rho in rhos:
            radius_started = time.perf_counter()
            values = []
            radius_runtime = {"status": "running", "per_seed": {}}
            artifact["runtime"]["per_radius"][str(rho)] = radius_runtime
            for seed in SEEDS:
                seed_started = time.perf_counter()
                details = checkpoints[str(rho)][str(seed)]
                checkpoint_path = Path(details["checkpoint"])
                model = load_model(checkpoint_path)
                xi1_train, xi2_train = generate_data(50, seed)
                value = exact_augmented_is(
                    model,
                    checkpoint_path.parent,
                    seed,
                    int(details["incumbent_mp_iteration"]),
                    xi1_train,
                    xi2_train,
                )
                values.append(value)
                radius_runtime["per_seed"][str(seed)] = {
                    "wall_time_s": time.perf_counter() - seed_started,
                    "status": "completed",
                }
            artifact["net11_is"][str(rho)] = curve_statistics(values)
            radius_runtime["wall_time_s"] = time.perf_counter() - radius_started
            radius_runtime["status"] = "completed"
            atomic_json_dump(NET11_IS_PATH, artifact)
            print(
                f"rho={rho:g}: width11 augmented IS mean="
                f"{artifact['net11_is'][str(rho)]['mean']:.6f} "
                f"time={radius_runtime['wall_time_s']:.3f}s",
                flush=True,
            )

        artifact["meta"]["status"] = "completed"
        artifact["runtime"]["status"] = "completed"
    except Exception as error:
        artifact["meta"]["status"] = "failed"
        artifact["runtime"]["status"] = "failed"
        artifact["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        artifact["runtime"]["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        artifact["runtime"]["total_time_s"] = time.perf_counter() - started
        atomic_json_dump(NET11_IS_PATH, artifact)

    comparison["net11_is"] = artifact["net11_is"]
    for rho in rhos:
        comparison["radius_provenance"]["net11"][str(rho)][
            "in_sample_definition"
        ] = artifact["meta"]["in_sample_definition"]
    comparison["runtime"]["net11_is_source"] = str(NET11_IS_PATH)
    comparison["runtime"]["net11_is_evaluation_time_s"] = artifact["runtime"][
        "total_time_s"
    ]
    plot_started = time.perf_counter()
    plot_results(comparison)
    comparison["runtime"]["net11_is_merge_and_plot_time_s"] = (
        time.perf_counter() - plot_started
    )
    comparison["runtime"]["net11_is_merged_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    atomic_json_dump(OUTPUT_JSON, comparison)
    print(f"Saved {NET11_IS_PATH}")
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


def merge_width11_into_comparison():
    merge_started = time.perf_counter()
    with OUTPUT_JSON.open() as stream:
        result = json.load(stream)
    with NET11_EVAL_PATH.open() as stream:
        width11 = json.load(stream)

    meta = width11.get("meta", {})
    if (
        meta.get("status") != "completed"
        or meta.get("interim") is not False
        or meta.get("n_test") != N_TEST
        or meta.get("test_seed") != TEST_SEED
        or meta.get("training_seeds") != SEEDS
        or meta.get("projection_method") != PROJECTION_METHOD
        or width11.get("radii") != result.get("rhos")
    ):
        raise RuntimeError("Incomplete or incompatible width-11 OOS artifact")
    if any(
        len(width11.get("width11_oos", {}).get(str(rho), {}).get("values", []))
        != len(SEEDS)
        for rho in result["rhos"]
    ):
        raise RuntimeError("Incomplete width-11 radius data")

    result["net11"] = width11["width11_oos"]
    result["meta"]["experiment"] = (
        "R3_N50_net8_vs_net11_vs_net13_vs_net16_all_radii"
    )
    result["meta"]["width11_interpretation"] = (
        "post-hoc visualization only; seed-9999 results are ineligible for "
        "architecture or radius selection"
    )
    result["meta"]["optimizer_comparison"]["net11_lr_net"] = 0.05
    result["radius_provenance"]["net11"] = {
        str(rho): {
            "projection_method": PROJECTION_METHOD,
            "evaluator_version": meta["evaluator_version"],
            "n_seeds": len(SEEDS),
            "post_hoc_exploratory": True,
            "selection_eligible": False,
        }
        for rho in result["rhos"]
    }
    result["checkpoint_provenance"]["net11"] = width11[
        "checkpoint_provenance"
    ]
    result["runtime"]["net11_source"] = str(NET11_EVAL_PATH)
    result["runtime"]["net11_oos_evaluation_time_s"] = width11["runtime"][
        "total_time_s"
    ]
    result["runtime"]["width11_merge_preparation_time_s"] = (
        time.perf_counter() - merge_started
    )
    plot_results(result)
    result["runtime"]["width11_merge_and_plot_time_s"] = time.perf_counter() - merge_started
    result["runtime"]["width11_merged_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary_path = OUTPUT_JSON.with_suffix(".json.tmp")
    with temporary_path.open("w") as stream:
        json.dump(result, stream, indent=2)
    temporary_path.replace(OUTPUT_JSON)
    print(f"Merged {NET11_EVAL_PATH}")
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


def validation_selection(rhos, checkpoints):
    started = time.perf_counter()
    selected_rhos = []
    per_seed = {}
    for seed in SEEDS:
        xi1_validation, xi2_validation = generate_data(
            VALIDATION_SIZE, VALIDATION_SEED_BASE + seed
        )
        costs = {}
        for rho in rhos:
            model = load_model(checkpoints[str(rho)][str(seed)]["checkpoint"])
            costs[rho] = model.mean_cost(xi1_validation, xi2_validation)
        selected_rho = min(costs, key=lambda rho: (costs[rho], rho))
        selected_rhos.append(selected_rho)
        per_seed[str(seed)] = {
            "validation_seed": VALIDATION_SEED_BASE + seed,
            "selected_rho": selected_rho,
            "selected_validation_cost": costs[selected_rho],
            "validation_costs": {str(rho): costs[rho] for rho in rhos},
        }
    counts = Counter(selected_rhos)
    return {
        "selected_rhos": selected_rhos,
        "selected_rho_counts": {str(rho): counts[rho] for rho in sorted(counts)},
        "validation_evaluation_time_s": time.perf_counter() - started,
        "per_seed": per_seed,
    }


def attach_selected_oos(selection, values_by_rho):
    selected_oos = []
    for seed, rho in enumerate(selection["selected_rhos"]):
        value = float(values_by_rho[str(rho)][seed])
        selected_oos.append(value)
        selection["per_seed"][str(seed)]["selected_10k_oos"] = value
    selection["selected_10k_oos_mean"] = float(np.mean(selected_oos))
    selection["selected_10k_oos_std"] = float(np.std(selected_oos, ddof=1))


def collect_checkpoint_provenance(rhos, resolver):
    provenance = {}
    for rho in rhos:
        details = {}
        for seed in SEEDS:
            checkpoint_path, best_ap_iteration, mp_iteration = resolver(rho, seed)
            if not checkpoint_path.exists():
                raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")
            details[str(seed)] = {
                "checkpoint": str(checkpoint_path),
                "best_ap_iteration": best_ap_iteration,
                "incumbent_mp_iteration": mp_iteration,
            }
        provenance[str(rho)] = details
    return provenance


def net16_incumbent_checkpoint(rho, seed):
    if abs(rho - 0.4) < 1e-12:
        directory = BASE_DIR / "saainit_net16_v2"
    else:
        directory = next(
            path
            for path in (BASE_DIR / "rho_sweep_net16_v2").glob("rho_*")
            if abs(float(path.name[4:].replace("p", ".")) - rho) < 1e-12
        )
    summary_path = directory / f"seed_{seed}_summary.json"
    with summary_path.open() as stream:
        summary = json.load(stream)
    if abs(rho - 0.4) < 1e-12:
        trajectory = summary.get("ccg_trajectory", [])
        legacy_complete = (
            summary.get("converged") is True
            and summary.get("ccg_iters") == len(trajectory)
            and bool(trajectory)
            and trajectory[-1].get("converged") is True
            and all("iter" in item and "ap_mean" in item for item in trajectory)
        )
        if not legacy_complete:
            raise RuntimeError(f"Incomplete legacy summary: {summary_path}")
        best_ap = min(trajectory, key=lambda item: item["ap_mean"])
        best_ap_iteration = int(best_ap["iter"])
        mp_iteration = best_ap_iteration - 1
    else:
        if summary.get("completed") is not True:
            raise RuntimeError(f"Incomplete summary: {summary_path}")
        mp_iteration = int(summary.get("final_mp_iter", 0))
        best_ap_iteration = mp_iteration + 1
    checkpoint_path = (
        BASE_DIR / "saainit_net16_v2" / f"seed_{seed}_nominal.pth"
        if mp_iteration <= 0
        else directory / f"seed_{seed}_mp_iter{mp_iteration}.pth"
    )
    return checkpoint_path, best_ap_iteration, mp_iteration


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    with LOWER_BOUND_PATH.open() as stream:
        lower_bound = json.load(stream)

    rhos = sorted(
        set(
            [0.4]
            + [
                float(path.name[4:].replace("p", "."))
                for path in NET8_SWEEP_DIR.glob("rho_*")
                if path.is_dir()
            ]
        )
    )
    net8_checkpoints = collect_checkpoint_provenance(rhos, incumbent_checkpoint)
    net13_checkpoints = collect_checkpoint_provenance(
        rhos, net13_incumbent_checkpoint
    )
    net16_checkpoints = collect_checkpoint_provenance(
        rhos, net16_incumbent_checkpoint
    )
    validation_started = time.perf_counter()
    validation_results = {
        "net8": validation_selection(rhos, net8_checkpoints),
        "net13": validation_selection(rhos, net13_checkpoints),
        "net16": validation_selection(rhos, net16_checkpoints),
    }
    validation_time_s = time.perf_counter() - validation_started

    # The seed-9999 artifact and test draw are accessed only after selection.
    with NET16_EVAL_PATH.open() as stream:
        net16_evaluation = json.load(stream)
    if net16_evaluation.get("runtime", {}).get("status") != "completed":
        raise RuntimeError("Incomplete width-16 reference artifact")
    if (
        net16_evaluation["meta"]["N_test"] != N_TEST
        or net16_evaluation["meta"]["seed_test"] != TEST_SEED
        or net16_evaluation["meta"].get("projection_method") != PROJECTION_METHOD
    ):
        raise RuntimeError("Unexpected width-16 reference metadata")
    for rho in rhos:
        entry = net16_evaluation.get("sweep", {}).get(str(float(rho)))
        if entry is None or any(
            len(entry.get(field, [])) != len(SEEDS)
            for field in ("rob_is", "rob_oos")
        ) or (
            entry.get("projection_method") != PROJECTION_METHOD
            or entry.get("evaluator_version") != EVALUATOR_VERSION
        ):
            raise RuntimeError(f"Incomplete width-16 rho={rho:g}")
    reference_benchmarks = net16_evaluation.get("benchmarks", {})
    reference_knn = reference_benchmarks.get("saa_knn", {})
    if (
        reference_benchmarks.get("_knn_k_selection") != KNN_SELECTION_VERSION
        or len(reference_knn.get("selected_k_by_seed", [])) != len(SEEDS)
        or not all(
            len(reference_knn.get(field, [])) == len(SEEDS)
            and np.isfinite(reference_knn[field]).all()
            for field in ("is", "val", "oos")
        )
    ):
        raise RuntimeError("Incomplete or stale width-16 kNN benchmark")
    xi1_test, xi2_test = generate_data(N_TEST, TEST_SEED)
    net8 = {}
    net8_is = {}
    net8_values = {}
    net8_evaluation_started = time.perf_counter()
    for rho in rhos:
        values = []
        is_values = []
        for seed in SEEDS:
            checkpoint_details = net8_checkpoints[str(rho)][str(seed)]
            checkpoint_path = Path(checkpoint_details["checkpoint"])
            mp_iteration = int(checkpoint_details["incumbent_mp_iteration"])
            model = load_model(checkpoint_path)
            values.append(model.mean_cost(xi1_test, xi2_test))
            xi1_train, xi2_train = generate_data(50, seed)
            is_values.append(
                exact_augmented_is(
                    model,
                    checkpoint_path.parent,
                    seed,
                    mp_iteration,
                    xi1_train,
                    xi2_train,
                )
            )
            print(
                f"rho={rho:g} seed={seed}: NN8 IS={is_values[-1]:.6f} "
                f"OOS={values[-1]:.6f}",
                flush=True,
            )
        net8[str(rho)] = curve_statistics(values)
        net8_is[str(rho)] = curve_statistics(is_values)
        net8_values[str(rho)] = values
    net8_evaluation_time_s = time.perf_counter() - net8_evaluation_started

    net16_load_started = time.perf_counter()
    net16 = {}
    net16_is = {}
    net16_values = {}
    for rho in rhos:
        entry = net16_evaluation["sweep"][str(float(rho))]
        values = [float(value) for value in entry["rob_oos"]]
        if len(values) != len(SEEDS):
            raise RuntimeError(f"Incomplete width-16 rho={rho:g}")
        net16[str(rho)] = curve_statistics(values)
        is_values = [float(value) for value in entry["rob_is"]]
        if len(is_values) != len(SEEDS):
            raise RuntimeError(f"Incomplete width-16 IS rho={rho:g}")
        net16_is[str(rho)] = curve_statistics(is_values)
        net16_values[str(rho)] = values
    net16_artifact_load_time_s = time.perf_counter() - net16_load_started

    net13 = {}
    net13_is = {}
    net13_values = {}
    net13_evaluation_started = time.perf_counter()
    for rho in rhos:
        values = []
        is_values = []
        for seed in SEEDS:
            checkpoint_details = net13_checkpoints[str(rho)][str(seed)]
            checkpoint_path = Path(checkpoint_details["checkpoint"])
            mp_iteration = int(checkpoint_details["incumbent_mp_iteration"])
            model = load_model(checkpoint_path)
            values.append(model.mean_cost(xi1_test, xi2_test))
            xi1_train, xi2_train = generate_data(50, seed)
            is_values.append(
                exact_augmented_is(
                    model,
                    checkpoint_path.parent,
                    seed,
                    mp_iteration,
                    xi1_train,
                    xi2_train,
                )
            )
            print(
                f"rho={rho:g} seed={seed}: NN13 IS={is_values[-1]:.6f} "
                f"OOS={values[-1]:.6f}",
                flush=True,
            )
        net13[str(rho)] = curve_statistics(values)
        net13_is[str(rho)] = curve_statistics(is_values)
        net13_values[str(rho)] = values
    net13_evaluation_time_s = time.perf_counter() - net13_evaluation_started

    attach_selected_oos(validation_results["net8"], net8_values)
    attach_selected_oos(validation_results["net13"], net13_values)
    attach_selected_oos(validation_results["net16"], net16_values)

    knn = curve_statistics(reference_knn["oos"])
    result = {
        "meta": {
            "experiment": "R3_N50_net8_vs_net13_vs_net16_all_radii",
            "N_train": 50,
            "N_test": N_TEST,
            "test_seed": TEST_SEED,
            "training_seeds": SEEDS,
            "validation_size": VALIDATION_SIZE,
            "validation_seed_base": VALIDATION_SEED_BASE,
            "projection_method": PROJECTION_METHOD,
            "evaluator_version": EVALUATOR_VERSION,
            "test_information_boundary": "seed-9999 artifact and draw opened only after all width/radius validation selections",
            "in_sample_definition": "mean per-training-sample maximum over nominal and AP columns used by the selected MP checkpoint",
            "incumbent_rule": "minimum AP mean; AP iteration t evaluates MP iteration t-1",
            "interval": "20th-80th percentile across training seeds",
            "robustness_response": "seed-paired OOS change from rho=0.001",
            "optimizer_comparison": {
                "net8_lr_net": 0.005,
                "net13_lr_net": 0.01,
                "net16_lr_net": 0.05,
                "lr_q": 0.01,
                "scope": "architecture-specific learning rates selected by the nominal training protocol; not a controlled learning-rate ablation",
            },
        },
        "rhos": rhos,
        "net8": net8,
        "net8_is": net8_is,
        "net13": net13,
        "net13_is": net13_is,
        "net16": net16,
        "net16_is": net16_is,
        "knn": knn,
        "lower_bound": {
            "value": float(lower_bound["saa_cond_lb_large"]),
            "source": str(LOWER_BOUND_PATH),
        },
        "validation_selection": {
            "net8": validation_results["net8"],
            "net13": validation_results["net13"],
            "net16": validation_results["net16"],
        },
        "radius_provenance": {
            width: {
                str(rho): {
                    "projection_method": PROJECTION_METHOD,
                    "evaluator_version": EVALUATOR_VERSION,
                    "in_sample_definition": "nominal plus AP1 through selected final_mp_iter",
                    "n_seeds": len(SEEDS),
                }
                for rho in rhos
            }
            for width in ("net8", "net13", "net16")
        },
        "checkpoint_provenance": {
            "net8": net8_checkpoints,
            "net13": net13_checkpoints,
            "net16": net16_checkpoints,
        },
        "runtime": {
            "started_at_utc": started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_time_s": time.perf_counter() - started,
            "validation_selection_time_s": validation_time_s,
            "net8_oos_and_is_evaluation_time_s": net8_evaluation_time_s,
            "net13_oos_and_is_evaluation_time_s": net13_evaluation_time_s,
            "net16_source": "existing canonical 10k evaluation artifact",
            "net16_artifact_load_time_s": net16_artifact_load_time_s,
            "status": "completed",
        },
    }
    temporary_path = OUTPUT_JSON.with_suffix(".json.tmp")
    with temporary_path.open("w") as stream:
        json.dump(result, stream, indent=2)
    temporary_path.replace(OUTPUT_JSON)
    plot_results(result)
    print(f"Saved {OUTPUT_JSON}")
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--merge-width11"]:
        merge_width11_into_comparison()
    elif sys.argv[1:] == ["--add-width11-is"]:
        evaluate_and_merge_width11_is()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit(
            "Usage: evaluate_plot_net8_vs_net16.py "
            "[--merge-width11|--add-width11-is]"
        )