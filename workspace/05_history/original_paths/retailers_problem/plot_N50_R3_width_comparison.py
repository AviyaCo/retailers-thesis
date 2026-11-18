"""Evaluate NN width 8 and plot the R=3, N=50 base-case OOS comparison."""

import json
import os
import time
from datetime import datetime, timezone

import gurobipy as gp
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from gurobipy import GRB
from scipy.stats import truncnorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = "/data/home/aviyac/retailers_problem"
NET8_DIR = os.path.join(BASE, "R3/N50/saainit_net8_v2")
REFERENCE_PATH = os.path.join(
    BASE, "R3/N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json"
)
LOWER_BOUND_PATH = os.path.join(BASE, "R3/saa_cond_lb_large.json")
RESULT_PATH = os.path.join(BASE, "R3/N50/base_width_comparison_10k.json")
PNG_PATH = os.path.join(BASE, "R3/N50/base_width_comparison_10k.png")
PDF_PATH = os.path.join(BASE, "R3/N50/base_width_comparison_10k.pdf")

R = 3
N_TRAIN = 50
N_TEST = 10_000
TEST_SEED = 9999
RHO = 0.4
SEEDS = list(range(10))
COST_C = 0.25
COST_H = 0.05
COST_B = 0.5

os.environ["GRB_LICENSE_FILE"] = "/data/home/aviyac/gurobi.lic"
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


class Model(nn.Module):
    def __init__(self, hidden_width):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R * 2 + 1, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, R),
        )
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))


def project_rows(raw_q2, q10):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    projected = np.empty_like(raw_q2)
    for index, row in enumerate(raw_q2):
        model = gp.Model(env=env)
        model.setParam("OutputFlag", 0)
        q2 = model.addVars(R, lb=0.0)
        model.addConstr(gp.quicksum(q2[retailer] for retailer in range(R)) <= q10)
        model.setObjective(
            gp.quicksum(
                (q2[retailer] - float(row[retailer])) ** 2
                for retailer in range(R)
            ),
            GRB.MINIMIZE,
        )
        model.optimize()
        if model.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(
                f"Projection failed for test row {index}: Gurobi status {model.status}"
            )
        projected[index] = [q2[retailer].X for retailer in range(R)]
        model.dispose()
    env.dispose()
    return projected


def evaluate_checkpoint(checkpoint_path, xi1, xi2):
    model = Model(hidden_width=8)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_unexpected = {"proj_layer.q", "proj_layer.A"}
    if incompatible.missing_keys or set(incompatible.unexpected_keys) != allowed_unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint {checkpoint_path}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    with torch.no_grad():
        q10 = torch.nn.functional.softplus(model.Q10).item()
        q1r = torch.nn.functional.softplus(model.Q1r).numpy()
        xi1_tensor = torch.tensor(xi1, dtype=torch.float32)
        network_input = torch.cat(
            [
                xi1_tensor,
                torch.tensor(q1r).unsqueeze(0).expand(len(xi1), -1),
                torch.full((len(xi1), 1), q10),
            ],
            dim=1,
        )
        raw_q2 = model.fc_layers(network_input).numpy()

    q2 = project_rows(raw_q2, q10)
    stage_one = COST_C * (q10 + q1r.sum()) + COST_H * q10
    recourse = np.maximum(
        COST_B * (xi1 + xi2 - q1r - q2) - COST_H * q2,
        np.maximum(
            COST_H * (q1r - xi1 - xi2),
            COST_B * (xi1 - q1r) - COST_H * xi2,
        ),
    )
    return float((stage_one + recourse.sum(axis=1)).mean())


def incumbent_checkpoint(seed):
    summary_path = os.path.join(NET8_DIR, f"seed_{seed}_summary.json")
    with open(summary_path) as stream:
        summary = json.load(stream)
    if summary.get("completed") is not True:
        raise RuntimeError(f"Width-8 seed {seed} is incomplete")
    best_ap = min(summary["ccg_trajectory"], key=lambda entry: entry["ap_mean"])
    mp_iteration = int(best_ap["iter"]) - 1
    if mp_iteration == 0:
        checkpoint_path = os.path.join(NET8_DIR, f"seed_{seed}_nominal.pth")
    else:
        checkpoint_path = os.path.join(
            NET8_DIR, f"seed_{seed}_mp_iter{mp_iteration}.pth"
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    return checkpoint_path, int(best_ap["iter"]), mp_iteration


def percentile_summary(values):
    array = np.asarray(values, dtype=float)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std_across_training_seeds": float(array.std()),
        "percentile_20": float(np.percentile(array, 20)),
        "percentile_80": float(np.percentile(array, 80)),
    }


def make_plot(results):
    labels = ["SAA-kNN", "NN width 16", "NN width 8", "Conditional LB"]
    keys = ["knn", "nn_width_16", "nn_width_8"]
    colors = ["#d69e00", "#2166ac", "#c43c39", "#2f855a"]
    means = [results["methods"][key]["mean"] for key in keys]
    means.append(results["lower_bound"]["value"])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    positions = np.arange(len(labels))
    ax.bar(
        positions,
        means,
        width=0.68,
        color=colors,
        edgecolor="#303030",
        linewidth=0.8,
        zorder=2,
    )

    for position, key in enumerate(keys):
        percentile_20 = results["methods"][key]["percentile_20"]
        percentile_80 = results["methods"][key]["percentile_80"]
        ax.vlines(position, percentile_20, percentile_80, color="#202020", lw=1.4, zorder=4)
        ax.hlines(
            [percentile_20, percentile_80],
            position - 0.08,
            position + 0.08,
            color="#202020",
            lw=1.4,
            zorder=4,
        )

    offsets = np.linspace(-0.22, 0.22, len(SEEDS))
    for position, key in enumerate(keys):
        ax.scatter(
            position + offsets,
            results["methods"][key]["values"],
            s=24,
            color="white",
            edgecolor="#303030",
            linewidth=0.7,
            zorder=3,
        )

    label_heights = [
        max(results["methods"][key]["values"]) + 0.08 for key in keys
    ] + [means[-1] + 0.08]
    for position, (mean, label_height) in enumerate(zip(means, label_heights)):
        ax.text(
            position,
            label_height,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(
        "Three-stage inventory: R=3, N=50\n"
        "Out-of-sample cost on 10,000 common test scenarios",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_ylabel("Mean OOS cost")
    ax.set_xticks(positions, labels)
    ax.set_ylim(19.0, max(label_heights) + 0.45)
    ax.grid(axis="y", color="#d7d7d7", linewidth=0.8, alpha=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        0.015,
        "Bars: mean; whiskers: 20th-80th percentiles; dots: training seeds.\n"
        "NN policies use robust CCG at rho=0.4. Conditional LB uses the "
        "separate 5,000 x 200 benchmark.",
        fontsize=8.5,
        color="#444444",
        ha="center",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PNG_PATH, dpi=200, bbox_inches="tight")
    fig.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(fig)


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    evaluation_start = time.perf_counter()

    with open(REFERENCE_PATH) as stream:
        reference = json.load(stream)
    with open(LOWER_BOUND_PATH) as stream:
        lower_bound = json.load(stream)

    meta = reference["meta"]
    if meta["N_test"] != N_TEST or meta["seed_test"] != TEST_SEED:
        raise RuntimeError(f"Unexpected reference evaluation metadata: {meta}")

    xi1_test, xi2_test = generate_data(N_TEST, TEST_SEED)
    width8_oos = []
    width8_checkpoints = {}
    per_seed_times = {}
    for seed in SEEDS:
        checkpoint_path, best_ap_iteration, mp_iteration = incumbent_checkpoint(seed)
        seed_start = time.perf_counter()
        width8_oos.append(evaluate_checkpoint(checkpoint_path, xi1_test, xi2_test))
        per_seed_times[str(seed)] = time.perf_counter() - seed_start
        width8_checkpoints[str(seed)] = {
            "best_ap_iteration": best_ap_iteration,
            "incumbent_mp_iteration": mp_iteration,
            "checkpoint": os.path.relpath(checkpoint_path, BASE),
        }
        print(f"seed={seed}: NN8 OOS={width8_oos[-1]:.6f}", flush=True)

    evaluation_time = time.perf_counter() - evaluation_start
    methods = {
        "knn": percentile_summary(reference["benchmarks"]["saa_knn"]["oos"]),
        "nn_width_16": percentile_summary(reference["sweep"]["0.4"]["rob_oos"]),
        "nn_width_8": percentile_summary(width8_oos),
    }
    results = {
        "experiment": "retailer_R3_N50_nn_width_oos_comparison",
        "methodology": {
            "n_train": N_TRAIN,
            "n_test": N_TEST,
            "test_seed": TEST_SEED,
            "training_seeds": SEEDS,
            "rho": RHO,
            "nn8_incumbent_rule": "min AP mean; AP iteration t evaluates MP iteration t-1",
            "knn_selection": reference["benchmarks"].get("_knn_k_selection"),
            "knn_selected_k_by_seed": reference["benchmarks"]["saa_knn"].get(
                "selected_k_by_seed"
            ),
            "reference_artifact": os.path.relpath(REFERENCE_PATH, BASE),
        },
        "methods": methods,
        "lower_bound": {
            "value": float(lower_bound["saa_cond_lb_large"]),
            "n_xi1": int(lower_bound["N"]),
            "conditional_samples_per_xi1": int(lower_bound["M"]),
            "seed": int(lower_bound["seed"]),
            "source": os.path.relpath(LOWER_BOUND_PATH, BASE),
        },
        "nn8_checkpoints": width8_checkpoints,
        "runtime": {
            "started_at_utc": started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "nn8_evaluation_time_s": evaluation_time,
            "nn8_evaluation_time_by_seed_s": per_seed_times,
            "nn16_reference_evaluation_time_s": reference["sweep"]["0.4"].get(
                "evaluation_time_s"
            ),
            "status": "completed",
        },
    }

    with open(RESULT_PATH, "w") as stream:
        json.dump(results, stream, indent=2)
    make_plot(results)
    print(f"Saved {RESULT_PATH}")
    print(f"Saved {PNG_PATH}")
    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()