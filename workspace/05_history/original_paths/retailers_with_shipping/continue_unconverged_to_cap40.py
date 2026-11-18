"""Continue fixed-shipping CCG runs that stopped unconverged at cap 8."""

import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path("/data/home/aviyac")
PYTHON = WORKSPACE / "ap_env/bin/python"
LOG_DIR = WORKSPACE / "retailers_with_shipping/cap40_continuation_logs"
RUNTIME_PATH = WORKSPACE / "retailers_with_shipping/cap40_continuation_runtime.json"
CAP = 40
MAX_WORKERS = 5
JOBS = [
    {"n_train": 50, "rho": 0.7, "seed": 2},
    {"n_train": 200, "rho": 0.5, "seed": 0},
    {"n_train": 200, "rho": 1.0, "seed": 0},
    {"n_train": 200, "rho": 1.0, "seed": 9},
    {"n_train": 200, "rho": 2.0, "seed": 2},
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def rho_tag(rho):
    text = f"{rho:.6f}".rstrip("0")
    parts = text.split(".")
    if text.endswith("."):
        text += "000"
    elif len(parts) > 1 and len(parts[1]) < 3:
        text += "0" * (3 - len(parts[1]))
    return text.replace(".", "p")


def paths_for(job):
    root = WORKSPACE / f"retailers_with_shipping/N{job['n_train']}"
    script = root / "rho_sweep/run_rho_seed.py"
    rho_directory = root / "rho_sweep" / f"rho_{rho_tag(job['rho'])}"
    summary = rho_directory / f"seed_{job['seed']}_summary.json"
    log = LOG_DIR / (
        f"N{job['n_train']}_rho_{rho_tag(job['rho'])}_seed_{job['seed']}.log"
    )
    return script, summary, log


def preflight(job):
    script, summary_path, log_path = paths_for(job)
    if not script.exists():
        raise FileNotFoundError(script)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    with summary_path.open() as stream:
        summary = json.load(stream)
    trajectory = summary.get("ccg_trajectory", [])
    if not trajectory:
        raise RuntimeError(f"Empty trajectory: {summary_path}")
    last_iteration = max(int(entry["iter"]) for entry in trajectory)
    if summary.get("converged") is True:
        raise RuntimeError(f"Target is already converged: {summary_path}")
    if last_iteration >= CAP:
        raise RuntimeError(f"Target already reached cap {CAP}: {summary_path}")
    for iteration in range(1, last_iteration + 1):
        ap_path = summary_path.parent / (
            f"seed_{job['seed']}_ap_iter{iteration}.json"
        )
        if not ap_path.exists():
            raise FileNotFoundError(ap_path)
    mp_path = summary_path.parent / (
        f"seed_{job['seed']}_mp_iter{last_iteration}.pth"
    )
    if not mp_path.exists():
        raise FileNotFoundError(mp_path)
    command = [
        str(PYTHON),
        str(script),
        "--rho",
        str(job["rho"]),
        "--seed",
        str(job["seed"]),
        "--max_iter",
        str(CAP),
        "--resume",
    ]
    return {
        **job,
        "prior_last_ap_iteration": last_iteration,
        "summary": str(summary_path.relative_to(WORKSPACE)),
        "log": str(log_path.relative_to(WORKSPACE)),
        "command": command,
        "status": "pending",
    }


def write_runtime(record):
    temporary = RUNTIME_PATH.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    temporary.replace(RUNTIME_PATH)


def run_job(job):
    started_at = utc_now()
    started = time.perf_counter()
    log_path = WORKSPACE / job["log"]
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[variable] = "1"
    with log_path.open("w") as log_stream:
        log_stream.write(f"started_at_utc={started_at}\n")
        log_stream.write(f"command={' '.join(job['command'])}\n")
        log_stream.flush()
        completed = subprocess.run(
            job["command"],
            cwd=WORKSPACE,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    ended_at = utc_now()
    return {
        **job,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "current_invocation_time_s": time.perf_counter() - started,
    }


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [preflight(job) for job in JOBS]
    started_at = utc_now()
    started = time.perf_counter()
    record = {
        "experiment": "fixed_shipping_continue_unconverged_to_cap40",
        "method": "resume retained CCG trajectories from MP iteration 8",
        "status": "running",
        "stopping_rule": "absolute relative AP-MP difference below 1 percent",
        "max_ap_iterations": CAP,
        "max_parallel_workers": MAX_WORKERS,
        "thread_limit_per_worker": 1,
        "started_at_utc": started_at,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "jobs": jobs,
    }
    write_runtime(record)

    print(f"Launching {len(jobs)} continuations with {MAX_WORKERS} workers", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            record["jobs"] = sorted(
                results
                + [job for job in jobs if job["summary"] not in {r["summary"] for r in results}],
                key=lambda job: (job["n_train"], job["rho"], job["seed"]),
            )
            record["current_invocation_elapsed_time_s"] = time.perf_counter() - started
            write_runtime(record)
            print(
                f"N={result['n_train']} rho={result['rho']:g} seed={result['seed']} "
                f"status={result['status']} time={result['current_invocation_time_s']:.1f}s",
                flush=True,
            )

    failures = [result for result in results if result["returncode"] != 0]
    record.update(
        {
            "status": "failed" if failures else "completed",
            "ended_at_utc": utc_now(),
            "current_invocation_elapsed_time_s": time.perf_counter() - started,
            "jobs": sorted(
                results, key=lambda job: (job["n_train"], job["rho"], job["seed"])
            ),
            "failure_count": len(failures),
        }
    )
    write_runtime(record)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
