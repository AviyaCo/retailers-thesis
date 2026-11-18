"""Select one hidden-width learning rate by independent validation across seeds."""

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import truncnorm


BASE_DIR = Path("/data/home/aviyac/retailers_problem/R3/N50")
parser = argparse.ArgumentParser()
parser.add_argument("--hidden-width", type=int, default=13)
args = parser.parse_args()
if args.hidden_width <= 0:
    parser.error("--hidden-width must be positive")
HIDDEN_WIDTH = args.hidden_width
AUDIT_DIR = BASE_DIR / f"width{HIDDEN_WIDTH}_lr_audit"
NOMINAL_DIR = BASE_DIR / f"saainit_net{HIDDEN_WIDTH}_v2"
OUTPUT_PATH = AUDIT_DIR / "lr_selection.json"
LRS = [0.005, 0.01, 0.05]
SEEDS = list(range(10))
N_VALIDATION = 13
VALIDATION_SEED_BASE = 88880
R = 3
PROJECTION_METHOD = "gurobi_qp"

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
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(7, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.ReLU(),
            nn.Linear(HIDDEN_WIDTH, R),
        )
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))

    @torch.no_grad()
    def mean_cost(self, xi1, xi2):
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
        q2 = project_with_gurobi(self.fc_layers(network_input).cpu().numpy(), q10)
        stage_one = 0.25 * (q10 + q1r.sum()) + 0.05 * q10
        recourse = np.maximum(
            0.5 * (xi1 + xi2 - q1r - q2) - 0.05 * q2,
            np.maximum(
                0.05 * (q1r - xi1 - xi2),
                0.5 * (xi1 - q1r) - 0.05 * xi2,
            ),
        )
        return float((stage_one + recourse.sum(axis=1)).mean())


def candidate_path(lr, seed):
    tag = f"{lr:.3f}".replace(".", "p")
    return AUDIT_DIR / f"lr_{tag}" / f"seed_{seed}" / f"seed_{seed}_nominal.pth"


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    by_lr = {}
    checkpoints = {}
    for lr in LRS:
        validation_costs = []
        training_times = []
        epochs = []
        checkpoints[str(lr)] = {}
        for seed in SEEDS:
            path = candidate_path(lr, seed)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = Model()
            incompatible = model.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
            allowed_unexpected = {"proj_layer.q", "proj_layer.A"}
            if incompatible.missing_keys or not set(
                incompatible.unexpected_keys
            ).issubset(allowed_unexpected):
                raise RuntimeError(f"Incompatible checkpoint: {path}")
            model.eval()
            xi1_validation, xi2_validation = generate_data(
                N_VALIDATION, VALIDATION_SEED_BASE + seed
            )
            validation_cost = model.mean_cost(xi1_validation, xi2_validation)
            validation_costs.append(validation_cost)
            training_times.append(float(checkpoint["training_time_s"]))
            epochs.append(int(checkpoint["epochs_completed"]))
            checkpoints[str(lr)][str(seed)] = {
                "path": str(path),
                "validation_seed": VALIDATION_SEED_BASE + seed,
                "validation_cost": validation_cost,
                "training_loss": float(checkpoint["best_loss"]),
                "training_time_s": float(checkpoint["training_time_s"]),
                "epochs_completed": int(checkpoint["epochs_completed"]),
            }
        by_lr[str(lr)] = {
            "mean_validation_cost": float(np.mean(validation_costs)),
            "std_validation_cost": float(np.std(validation_costs, ddof=1)),
            "validation_costs": validation_costs,
            "mean_training_time_s": float(np.mean(training_times)),
            "median_training_time_s": float(np.median(training_times)),
            "min_training_time_s": float(np.min(training_times)),
            "max_training_time_s": float(np.max(training_times)),
            "epochs_completed": epochs,
        }

    selected_lr = min(
        LRS, key=lambda lr: (by_lr[str(lr)]["mean_validation_cost"], lr)
    )
    NOMINAL_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        source_path = candidate_path(selected_lr, seed)
        destination_path = NOMINAL_DIR / f"seed_{seed}_nominal.pth"
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        checkpoint["lr_selection_protocol"] = (
            "global mean independent-validation cost across seeds"
        )
        checkpoint["lr_selection_artifact"] = str(OUTPUT_PATH)
        checkpoint["validation_seed"] = VALIDATION_SEED_BASE + seed
        checkpoint["validation_cost"] = checkpoints[str(selected_lr)][str(seed)][
            "validation_cost"
        ]
        torch.save(checkpoint, destination_path)
        summary_source = source_path.parent / f"seed_{seed}_summary.json"
        summary_destination = NOMINAL_DIR / f"seed_{seed}_summary.json"
        with summary_source.open() as stream:
            summary = json.load(stream)
        summary["selected_lr_net"] = selected_lr
        summary["lr_selection_protocol"] = (
            "global mean independent-validation cost across seeds"
        )
        summary["lr_selection_artifact"] = str(OUTPUT_PATH)
        summary["validation_seed"] = VALIDATION_SEED_BASE + seed
        summary["validation_cost"] = checkpoints[str(selected_lr)][str(seed)][
            "validation_cost"
        ]
        with summary_destination.open("w") as stream:
            json.dump(summary, stream, indent=2)

    result = {
        "experiment": f"retailer_R3_N50_net{HIDDEN_WIDTH}_lr_audit",
        "status": "completed",
        "hidden_widths": [HIDDEN_WIDTH, HIDDEN_WIDTH],
        "total_parameters_including_Q": sum(
            parameter.numel() for parameter in Model().parameters()
        ),
        "candidate_lrs": LRS,
        "training_seeds": SEEDS,
        "validation_seeds": [VALIDATION_SEED_BASE + seed for seed in SEEDS],
        "validation_size_per_seed": N_VALIDATION,
        "selection_rule": "minimum mean independent-validation cost; smaller LR breaks exact ties",
        "test_data_used": False,
        "projection_method": PROJECTION_METHOD,
        "selected_lr_net": selected_lr,
        "by_lr": by_lr,
        "checkpoints": checkpoints,
        "candidate_model_time_s": float(
            sum(
                item["training_time_s"]
                for lr_checkpoints in checkpoints.values()
                for item in lr_checkpoints.values()
            )
        ),
        "selection_evaluation_time_s": time.perf_counter() - started,
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with temporary_path.open("w") as stream:
        json.dump(result, stream, indent=2)
    temporary_path.replace(OUTPUT_PATH)
    shutil.copy2(OUTPUT_PATH, NOMINAL_DIR / "lr_selection.json")
    print(f"Selected width-{HIDDEN_WIDTH} LR={selected_lr:g}")
    for lr in LRS:
        print(
            f"  LR={lr:g}: validation mean="
            f"{by_lr[str(lr)]['mean_validation_cost']:.6f}"
        )
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()