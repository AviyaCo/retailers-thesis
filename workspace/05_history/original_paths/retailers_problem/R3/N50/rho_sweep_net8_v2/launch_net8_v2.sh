#!/bin/bash
# Resume the R=3, N=50 width-8 sweep through rho=1.5.

set -u

MAX_PARALLEL=${1:-3}
ROOT="/data/home/aviyac/retailers_problem/R3/N50"
NOMINAL_DIR="${ROOT}/saainit_net8_v2"
SWEEP_DIR="${ROOT}/rho_sweep_net8_v2"
WORKER="${ROOT}/rho_sweep_net16_v2/run_rho_seed_net16_v2.py"
PYTHON="/data/home/aviyac/ap_env/bin/python3"
LOG_DIR="${SWEEP_DIR}/logs"
RUNTIME_PATH="${SWEEP_DIR}/sweep_runtime.json"

mkdir -p "${LOG_DIR}"

# rho=0.4 already completed for all ten seeds in saainit_net8_v2.
ALL_RHOS=(0.001 0.0015 0.002 0.003 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.450 0.500 0.550 0.600 0.650 0.700 0.750 0.800 1.000 1.500)
RUN_RHOS=(0.001 0.0015 0.002 0.003 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.450 0.500 0.550 0.600 0.650 0.700 0.750 0.800 1.000 1.500)
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
    "experiment": "retailer_R3_N50_net8_v2_rho_sweep",
    "n_samples": 50,
    "hidden_widths": [8, 8],
    "lr_net": 0.005,
    "radii": [0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05,
              0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
              0.7, 0.75, 0.8, 1.0, 1.5],
    "seeds": list(range(10)),
    "rho_0p4_source": "../saainit_net8_v2",
    "rho_0p4_status": "completed_before_sweep",
    "jobs_expected_in_sweep_directory": 220,
    "max_parallel_workers": int(workers),
    "started_at_utc": started,
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
echo "R=3, N=50 width-8 radius sweep"
echo "  Full grid: ${#ALL_RHOS[@]} radii through 1.5"
echo "  Existing: rho=0.4, 10/10 seeds"
echo "  Jobs in this launcher: $((${#RUN_RHOS[@]} * ${#SEEDS[@]}))"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "======================================================"

completed=0
active=0
launched=0
failures=0
write_runtime "running" ""

for rho in "${RUN_RHOS[@]}"; do
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

        logfile="${LOG_DIR}/rho_${tag}_seed_${seed}_$(date --utc +%Y%m%dT%H%M%SZ).log"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
            nice -n 10 "${PYTHON}" "${WORKER}" \
            --rho "${rho}" --seed "${seed}" \
            --nominal-dir "${NOMINAL_DIR}" --sweep-dir "${SWEEP_DIR}" \
            --hidden-width 8 --lr-net 0.005 \
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
echo "Width-8 sweep completed."
echo "  Skipped completed: ${completed}"
echo "  Skipped active: ${active}"
echo "  Launched: ${launched}"
echo "  Worker failures observed: ${failures}"
echo "======================================================"