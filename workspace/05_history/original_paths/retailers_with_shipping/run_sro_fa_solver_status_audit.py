"""Rerun fixed-shipping SRO-FA for N=50 and N=200 with solver metadata."""

import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "sro_fa_solver_status_audit_20260803"
CAMPAIGN_JSON = OUTPUT_DIR / "campaign_runtime.json"
SUMMARY_JSON = OUTPUT_DIR / "audit_summary.json"
TRAINING_SIZES = (50, 200)


def atomic_json_dump(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as stream:
        json.dump(payload, stream, indent=2)
    temporary_path.replace(path)


def result_path(n_train):
    return OUTPUT_DIR / f"sro_fa_shipping_results_N{n_train}_R3_status_audit.json"


def runtime_path(n_train):
    return OUTPUT_DIR / f"sro_fa_shipping_results_N{n_train}_R3_status_audit_runtime.json"


def historical_path(n_train):
    return (
        BASE_DIR
        / f"N{n_train}"
        / "sro_fa"
        / f"sro_fa_shipping_results_N{n_train}_R3.json"
    )


def summarize_size(n_train):
    with result_path(n_train).open() as stream:
        results = json.load(stream)
    with historical_path(n_train).open() as stream:
        historical = json.load(stream)

    rows = []
    is_differences = []
    oos_differences = []
    for rho, by_seed in results.items():
        for seed, row in by_seed.items():
            rows.append(row)
            old = historical[rho][seed]
            is_differences.append(abs(float(row["is"]) - float(old["is"])))
            oos_differences.append(abs(float(row["oos"]) - float(old["oos"])))

    completed = [row for row in rows if row["sol_count"] > 0]
    incomplete = [row for row in rows if row["sol_count"] <= 0]
    wall_times = [float(row["time_total"]) for row in completed]
    solver_times = [float(row["gurobi_runtime_s"]) for row in completed]
    gaps = [float(row["mip_gap"]) for row in completed]
    return {
        "n_train": n_train,
        "expected_runs": 340,
        "completed_runs": len(completed),
        "incomplete_runs": len(incomplete),
        "status_counts": dict(Counter(row["solver_status_name"] for row in rows)),
        "target_gap_satisfied_count": sum(gap <= 0.01 + 1e-12 for gap in gaps),
        "time_limit_count": sum(
            row["solver_status_name"] == "TIME_LIMIT" for row in rows
        ),
        "maximum_mip_gap": max(gaps),
        "runtime_s": {
            "runs": len(wall_times),
            "mean": statistics.mean(wall_times),
            "median": statistics.median(wall_times),
            "minimum": min(wall_times),
            "maximum": max(wall_times),
            "sum_per_run_wall_times": sum(wall_times),
            "mean_solver": statistics.mean(solver_times),
            "maximum_solver": max(solver_times),
        },
        "historical_comparison": {
            "maximum_absolute_is_difference": max(is_differences),
            "maximum_absolute_oos_difference": max(oos_differences),
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    campaign_started = time.perf_counter()
    campaign = {
        "experiment": "fixed_shipping_sro_fa_solver_status_audit_N50_N200",
        "status": "running",
        "training_sizes": list(TRAINING_SIZES),
        "rhos_per_training_size": 34,
        "seeds_per_radius": 10,
        "expected_runs": 680,
        "execution": "sequential by training size; one Python sweep process at a time",
        "sweep_workers": 1,
        "gurobi_threads_per_run": 2,
        "concurrent_experiment_at_launch": "width11CapacitySweep",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "per_training_size": {},
    }
    atomic_json_dump(CAMPAIGN_JSON, campaign)

    try:
        for n_train in TRAINING_SIZES:
            script = (
                BASE_DIR / f"N{n_train}" / "sro_fa" / "sro_fa_shipping_sweep.py"
            )
            command = [sys.executable, str(script)]
            environment = os.environ.copy()
            environment.update(
                {
                    "SRO_FA_OUT_JSON": str(result_path(n_train)),
                    "SRO_FA_RUNTIME_JSON": str(runtime_path(n_train)),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            size_started = time.perf_counter()
            campaign["per_training_size"][str(n_train)] = {
                "status": "running",
                "command": command,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "result_path": str(result_path(n_train)),
                "runtime_path": str(runtime_path(n_train)),
            }
            atomic_json_dump(CAMPAIGN_JSON, campaign)
            completed = subprocess.run(command, env=environment, check=False)
            size_record = campaign["per_training_size"][str(n_train)]
            size_record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            size_record["elapsed_time_s"] = time.perf_counter() - size_started
            size_record["return_code"] = completed.returncode
            size_record["status"] = (
                "completed" if completed.returncode == 0 else "failed"
            )
            atomic_json_dump(CAMPAIGN_JSON, campaign)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"N={n_train} audit failed with return code {completed.returncode}"
                )

        summary = {
            "experiment": campaign["experiment"],
            "status": "completed",
            "method": "SRO-FA with fixed shipping cost f=0.1",
            "solver": "Gurobi",
            "target_mip_gap": 0.01,
            "time_limit_s": 3000,
            "by_training_size": {
                str(n_train): summarize_size(n_train)
                for n_train in TRAINING_SIZES
            },
        }
        atomic_json_dump(SUMMARY_JSON, summary)
        campaign["status"] = "completed"
    except Exception as error:
        campaign["status"] = "failed"
        campaign["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        campaign["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        campaign["overall_elapsed_time_s"] = time.perf_counter() - campaign_started
        atomic_json_dump(CAMPAIGN_JSON, campaign)


if __name__ == "__main__":
    main()