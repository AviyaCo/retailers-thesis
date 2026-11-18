"""
Select rho for the R=3 net16-v2 retailer experiments by independent validation.

This script does not retrain any model. For each N and training seed, it loads
the existing rho-specific checkpoints, evaluates them on a seed-specific
validation draw, selects the rho with the lowest validation cost for that seed,
and reports the already-saved 10k OOS cost at the selected rho.

Output: retailers_problem/R3/rho_validation_results_net16_v2.json
"""

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy.stats import truncnorm
import torch
import torch.nn as nn


BASE = "/data/home/aviyac/retailers_problem/R3"
OUT_JSON = os.path.join(BASE, "rho_validation_results_net16_v2.json")

SEEDS = list(range(10))
VALIDATION_FRACTION = 0.25
SEED_VAL_BASE = 88880
R = 3
c, h, b = 0.25, 0.05, 0.5

N50_FINAL_ITER_040 = {0: 3, 1: 5, 2: 4, 3: 5, 4: 7, 5: 4, 6: 7, 7: 5, 8: 5, 9: 5}

CONFIGS = {
    50: {
        "nominal_dir": os.path.join(BASE, "N50/saainit_net16_v2"),
        "sweep_dir": os.path.join(BASE, "N50/rho_sweep_net16_v2"),
        "eval_json": os.path.join(BASE, "N50/rho_sweep_net16_v2/sweep_eval_net16_v2_10k.json"),
        "legacy_saainit_rho": 0.4,
    },
    200: {
        "nominal_dir": os.path.join(BASE, "N200/saainit_net16_v2"),
        "sweep_dir": os.path.join(BASE, "N200/rho_sweep_net16_v2"),
        "eval_json": os.path.join(BASE, "N200/rho_sweep_net16_v2/sweep_eval_net16_v2_N200_10k.json"),
    },
    800: {
        "nominal_dir": os.path.join(BASE, "N800/saainit_net16_v2"),
        "sweep_dir": os.path.join(BASE, "N800/rho_sweep_net16_v2"),
        "eval_json": os.path.join(BASE, "N800/rho_sweep_net16_v2/sweep_eval_net16_v2_N800_10k.json"),
    },
}


def gen(n_samples, seed):
    np.random.seed(seed)
    a_xi1 = (0.0 - 6.0) / 2.5
    xi1 = truncnorm.rvs(a_xi1, np.inf, loc=6.0, scale=2.5, size=(n_samples, R))
    mu2 = 2 * (6.0 - xi1) ** 2
    a_xi2 = (0.0 - mu2) / 2.5
    xi2 = truncnorm.rvs(a_xi2, np.inf, loc=mu2, scale=2.5)
    return xi1, xi2


def validation_size(n_train):
    return int(np.ceil(VALIDATION_FRACTION * n_train))


def project_simplex_scaled(z, cap):
    scalar = z.ndim == 1
    if scalar:
        z = z[None, :]
    z_nn = np.maximum(z, 0.0)
    if np.all(z_nn.sum(axis=1) <= cap):
        return z_nn[0] if scalar else z_nn
    u = np.sort(z_nn, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1)
    rho = np.sum(u * np.arange(1, z.shape[1] + 1) > (cssv - cap), axis=1) - 1
    theta = (cssv[np.arange(len(z)), rho] - cap) / (rho + 1.0)
    q = np.maximum(z_nn - theta[:, None], 0.0)
    needs = z_nn.sum(axis=1) > cap
    q[~needs] = z_nn[~needs]
    return q[0] if scalar else q


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(R * 2 + 1, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, R),
        )
        self.Q10 = nn.Parameter(torch.zeros(1))
        self.Q1r = nn.Parameter(torch.zeros(R))

    @torch.no_grad()
    def get_Q2(self, xi1_t):
        n_samples = xi1_t.shape[0]
        q10 = torch.nn.functional.softplus(self.Q10).item()
        q1r = torch.nn.functional.softplus(self.Q1r).numpy()
        net_input = torch.cat([
            xi1_t,
            torch.tensor(q1r).unsqueeze(0).expand(n_samples, -1),
            torch.full((n_samples, 1), q10),
        ], dim=1)
        z = self.fc_layers(net_input).numpy()
        q2 = project_simplex_scaled(z, q10)
        return q2, q10, q1r


def load_model(path):
    model = Model()
    model.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"], strict=False)
    model.eval()
    return model


def mean_cost(q10, q1r, q2, xi1, xi2):
    first_stage = c * (q10 + q1r.sum()) + h * q10
    recourse = np.maximum(
        b * (xi1 + xi2 - q1r - q2) - h * q2,
        np.maximum(h * (q1r - xi1 - xi2), b * (xi1 - q1r) - h * xi2),
    )
    return float((first_stage + recourse.sum(axis=1)).mean())


def discover_rho_sources(config):
    sources = {}
    for entry in sorted(os.listdir(config["sweep_dir"])):
        full = os.path.join(config["sweep_dir"], entry)
        if entry.startswith("rho_") and os.path.isdir(full):
            rho = float(entry[4:].replace("p", "."))
            if 0.0 < rho <= 1.5:
                sources[rho] = ("sweep", full)
    legacy_rho = config.get("legacy_saainit_rho")
    if legacy_rho is not None and legacy_rho not in sources:
        sources[legacy_rho] = ("saainit", config["nominal_dir"])
    return dict(sorted(sources.items()))


def model_path_for_source(config, source, seed):
    kind, directory = source
    nominal_path = os.path.join(config["nominal_dir"], f"seed_{seed}_nominal.pth")
    if kind == "saainit":
        final_iter = N50_FINAL_ITER_040[seed]
        return os.path.join(directory, f"seed_{seed}_mp_iter{final_iter}.pth")

    summary_path = os.path.join(directory, f"seed_{seed}_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    final_iter = int(summary.get("final_mp_iter", 0))
    if final_iter == 0:
        return nominal_path
    return os.path.join(directory, f"seed_{seed}_mp_iter{final_iter}.pth")


def oos_for_seed(eval_data, rho, seed):
    entry = eval_data["sweep"].get(str(float(rho)))
    if entry is None or len(entry.get("rob_oos", [])) <= seed:
        return None
    return float(entry["rob_oos"][seed])


def evaluate_n(n_train, config):
    with open(config["eval_json"]) as f:
        eval_data = json.load(f)

    n_val = validation_size(n_train)
    rho_sources = discover_rho_sources(config)
    per_seed = {}
    selected_rhos = []
    selected_oos = []

    for seed in SEEDS:
        xi1_val, xi2_val = gen(n_val, SEED_VAL_BASE + seed)
        xi1_val_t = torch.tensor(xi1_val, dtype=torch.float32)
        val_costs = {}
        oos_costs = {}

        for rho, source in rho_sources.items():
            model_path = model_path_for_source(config, source, seed)
            if model_path is None or not os.path.exists(model_path):
                continue
            oos_cost = oos_for_seed(eval_data, rho, seed)
            if oos_cost is None:
                continue

            model = load_model(model_path)
            q2_val, q10, q1r = model.get_Q2(xi1_val_t)
            val_costs[rho] = mean_cost(q10, q1r, q2_val, xi1_val, xi2_val)
            oos_costs[rho] = oos_cost

        if not val_costs:
            raise RuntimeError(f"No validation candidates found for N={n_train}, seed={seed}")

        selected_rho = min(val_costs, key=lambda rho: (val_costs[rho], rho))
        selected_rhos.append(selected_rho)
        selected_oos.append(oos_costs[selected_rho])
        per_seed[str(seed)] = {
            "N_val": n_val,
            "validation_seed": SEED_VAL_BASE + seed,
            "selected_rho": selected_rho,
            "selected_validation_cost": val_costs[selected_rho],
            "selected_10k_oos": oos_costs[selected_rho],
            "validation_costs": {str(rho): val_costs[rho] for rho in sorted(val_costs)},
        }
        print(
            f"N={n_train:>3} seed={seed}: rho={selected_rho:g} "
            f"val={val_costs[selected_rho]:.4f} 10k OOS={oos_costs[selected_rho]:.4f}"
        )

    counts = Counter(selected_rhos)
    return {
        "selected_10k_oos_mean": float(np.mean(selected_oos)),
        "selected_10k_oos_std": float(np.std(selected_oos, ddof=1)) if len(selected_oos) > 1 else 0.0,
        "selected_rhos": selected_rhos,
        "selected_rho_counts": {str(rho): counts[rho] for rho in sorted(counts)},
        "per_seed": per_seed,
    }


def main():
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    validation_time_by_n = {}
    results = {
        "meta": {
            "protocol": "per_seed_independent_validation_rho_selection",
            "validation_fraction": VALIDATION_FRACTION,
            "validation_size_rule": "ceil(validation_fraction * N_train)",
            "N_val_by_N": {str(n_train): validation_size(n_train) for n_train in CONFIGS},
            "seed_val_base": SEED_VAL_BASE,
            "seeds": SEEDS,
            "test_set": "saved 10k OOS from seed 9999 eval JSONs; not used for rho selection",
            "rho_grid": "cached positive rho values with rho <= 1.5",
        },
        "by_N": {},
    }

    for n_train, config in CONFIGS.items():
        print(f"\nSelecting rho for N={n_train} using N_val={validation_size(n_train)} ...")
        n_started = time.perf_counter()
        results["by_N"][str(n_train)] = evaluate_n(n_train, config)
        validation_time_by_n[str(n_train)] = time.perf_counter() - n_started

    results["runtime"] = {
        "start_utc": started_utc,
        "end_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "validation_time_by_n_s": validation_time_by_n,
        "total_time_s": time.perf_counter() - started,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {OUT_JSON}")
    print("Summary:")
    for n_train in CONFIGS:
        entry = results["by_N"][str(n_train)]
        print(
            f"  N={n_train}: validation-selected Neural CCG 10k OOS="
            f"{entry['selected_10k_oos_mean']:.4f}, rho_counts={entry['selected_rho_counts']}"
        )


if __name__ == "__main__":
    main()