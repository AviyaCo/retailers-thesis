"""Rerun validation-selected R=3 net16-v2 CCG paths with structured timing."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = Path("/data/home/aviyac/ap_env/bin/python")
SELECTION_PATH = ROOT / "rho_validation_results_net16_v2.json"
WORKERS = {
    50: ROOT / "N50/rho_sweep_net16_v2/run_rho_seed_net16_v2.py",
    200: ROOT / "N200/rho_sweep_net16_v2/run_rho_seed_net16_v2_N200.py",
    800: ROOT / "N800/rho_sweep_net16_v2/run_rho_seed_net16_v2_N800.py",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as stream:
        json.dump(payload, stream, indent=2)
    temporary_path.replace(path)


def rho_tag(rho):
    value = f"{rho:.6f}".rstrip("0")
    if value.endswith("."):
        value += "000"
    elif "." in value and len(value.split(".")[1]) < 3:
        value += "0" * (3 - len(value.split(".")[1]))
    return value.replace(".", "p")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=30)
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "selected_timing_net16_v2")
    args = parser.parse_args()
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")

    output_root = args.output_root.resolve()
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with SELECTION_PATH.open() as stream:
        selection = json.load(stream)

    jobs = []
    for n_samples in (50, 200, 800):
        nominal_dir = ROOT / f"N{n_samples}/saainit_net16_v2"
        sweep_dir = output_root / f"N{n_samples}"
        per_seed = selection["by_N"][str(n_samples)]["per_seed"]
        for seed in range(10):
            rho = float(per_seed[str(seed)]["selected_rho"])
            nominal_checkpoint = nominal_dir / f"seed_{seed}_nominal.pth"
            if not nominal_checkpoint.is_file():
                raise FileNotFoundError(nominal_checkpoint)
            command = [
                str(PYTHON), "-u", str(WORKERS[n_samples]),
                "--rho", str(rho), "--seed", str(seed),
                "--nominal-dir", str(nominal_dir),
                "--sweep-dir", str(sweep_dir),
            ]
            jobs.append({
                "n_samples": n_samples,
                "seed": seed,
                "rho": rho,
                "command": command,
                "log_path": str(logs_dir / f"N{n_samples}_rho{rho_tag(rho)}_seed{seed}.log"),
                "summary_path": str(
                    sweep_dir / f"rho_{rho_tag(rho)}" / f"seed_{seed}_summary.json"),
                "status": "pending",
            })

    campaign_path = output_root / "campaign_runtime.json"
    campaign_started_at = utc_now()
    campaign_start = time.perf_counter()
    campaign = {
        "experiment": "retailer_R3_net16_v2_validation_selected_timing",
        "method": "SAA-initialized neural CCG",
        "selection_protocol": selection["meta"]["protocol"],
        "selection_artifact": str(SELECTION_PATH),
        "scope": "validation-selected radius for each of 10 seeds at N=50,200,800",
        "status": "running",
        "started_at_utc": campaign_started_at,
        "max_workers": min(args.max_workers, len(jobs)),
        "worker_thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    write_json(campaign_path, campaign)

    environment = os.environ.copy()
    environment.update(campaign["worker_thread_environment"])

    def run_job(job):
        started_at = utc_now()
        started = time.perf_counter()
        with open(job["log_path"], "w") as log_stream:
            completed = subprocess.run(
                job["command"], cwd=ROOT, env=environment,
                stdout=log_stream, stderr=subprocess.STDOUT, check=False)
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "started_at_utc": started_at,
            "ended_at_utc": utc_now(),
            "current_invocation_time_s": time.perf_counter() - started,
        }

    with ThreadPoolExecutor(max_workers=campaign["max_workers"]) as executor:
        future_jobs = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            job.update(future.result())
            campaign["current_invocation_elapsed_time_s"] = time.perf_counter() - campaign_start
            write_json(campaign_path, campaign)

    failed_jobs = [job for job in jobs if job["status"] != "completed"]
    campaign.update({
        "status": "completed" if not failed_jobs else "failed",
        "ended_at_utc": utc_now(),
        "overall_sweep_elapsed_time_s": time.perf_counter() - campaign_start,
        "n_completed": len(jobs) - len(failed_jobs),
        "n_failed": len(failed_jobs),
    })
    write_json(campaign_path, campaign)
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    sys.exit(main())