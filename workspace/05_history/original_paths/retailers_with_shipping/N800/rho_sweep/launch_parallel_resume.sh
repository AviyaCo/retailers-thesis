#!/bin/bash
# Resume the corrected N=800 minibatch sweep without duplicating completed or active jobs.

set -u

MAX_PARALLEL=${1:-24}
SCRIPT_DIR="/data/home/aviyac/retailers_with_shipping/N800/rho_sweep"
SWEEP_DIR="/data/home/aviyac/retailers_with_shipping/N800/rho_sweep_minibatch"
SCRIPT="${SCRIPT_DIR}/run_rho_seed.py"
LOG_DIR="${SWEEP_DIR}/logs"
RUNTIME_PATH="${SWEEP_DIR}/parallel_resume_runtime.json"
PYTHON="/data/home/aviyac/ap_env/bin/python3"

mkdir -p "${LOG_DIR}"

RHOS=(0.001 0.003 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.500 0.700 1.000 1.500)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
STARTED_AT_UTC=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
SECONDS=0

rho_tag() {
    "${PYTHON}" - "$1" <<'PY'
import sys

value = float(sys.argv[1])
text = f"{value:.6f}".rstrip("0")
parts = text.split(".")
if text.endswith("."):
    text += "000"
elif len(parts) > 1 and len(parts[1]) < 3:
    text += "0" * (3 - len(parts[1]))
print(text.replace(".", "p"))
PY
}

is_completed() {
    local summary="$1"
    [[ -f "${summary}" ]] || return 1
    "${PYTHON}" - "${summary}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as stream:
        data = json.load(stream)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("completed") is True else 1)
PY
}

is_active() {
    local lock_path="$1"
    [[ -e "${lock_path}" ]] || return 1
    ! flock -n "${lock_path}" -c true
}

write_runtime() {
    local status="$1"
    local ended_at="$2"
    "${PYTHON}" - "${RUNTIME_PATH}" "${STARTED_AT_UTC}" "${ended_at}" \
        "${SECONDS}" "${MAX_PARALLEL}" "${completed}" "${active}" \
        "${launched}" "${failures}" "${status}" <<'PY'
import json
import sys

(path, started, ended, elapsed, workers, completed, active,
 launched, failures, status) = sys.argv[1:]
record = {
    "experiment": "fixed_shipping_N800_neural_ccg_minibatch_mp_parallel_resume",
    "started_at_utc": started,
    "max_parallel_workers_this_launcher": int(workers),
    "jobs_skipped_completed_at_scan": int(completed),
    "jobs_skipped_active_at_scan": int(active),
    "jobs_launched_current_invocation": int(launched),
    "worker_failures_observed": int(failures),
    "current_invocation_elapsed_time_s": float(elapsed),
    "status": status,
}
if ended:
    record["ended_at_utc"] = ended
with open(path, "w") as stream:
    json.dump(record, stream, indent=2)
PY
}

echo "======================================================"
echo "Corrected shipping N=800 parallel resume"
echo "  Max additional workers: ${MAX_PARALLEL}"
echo "  Output: ${SWEEP_DIR}"
echo "======================================================"

completed=0
active=0
launched=0
failures=0
write_runtime "running" ""

for rho in "${RHOS[@]}"; do
    tag=$(rho_tag "${rho}")
    for seed in "${SEEDS[@]}"; do
        summary="${SWEEP_DIR}/rho_${tag}/seed_${seed}_summary.json"
        lock_path="${SWEEP_DIR}/rho_${tag}/seed_${seed}.lock"

        if is_completed "${summary}"; then
            completed=$((completed + 1))
            continue
        fi
        if is_active "${lock_path}"; then
            active=$((active + 1))
            continue
        fi

        while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
            wait -n || failures=$((failures + 1))
        done

        logfile="${LOG_DIR}/parallel_rho_${tag}_seed_${seed}_$(date --utc +%Y%m%dT%H%M%SZ).log"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
            nice -n 10 "${PYTHON}" "${SCRIPT}" --rho "${rho}" --seed "${seed}" \
            > "${logfile}" 2>&1 &
        launched=$((launched + 1))
        echo "  Launched rho=${rho} seed=${seed} (pid=$!, log=${logfile})"
    done
done

while (( $(jobs -r | wc -l) > 0 )); do
    wait -n || failures=$((failures + 1))
done

ENDED_AT_UTC=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
write_runtime "completed" "${ENDED_AT_UTC}"

echo "======================================================"
echo "Parallel resume completed."
echo "  Skipped completed: ${completed}"
echo "  Skipped active: ${active}"
echo "  Launched: ${launched}"
echo "  Worker failures observed: ${failures}"
echo "======================================================"