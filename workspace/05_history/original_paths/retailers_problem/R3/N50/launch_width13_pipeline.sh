#!/bin/bash
# Configurable-width LR audit followed by the preregistered 23-radius sweep.

set -u

ROOT="/data/home/aviyac/retailers_problem/R3/N50"
PYTHON="/data/home/aviyac/ap_env/bin/python"
WIDTH=${WIDTH:-13}
TOTAL_PARAMETERS=$((WIDTH * WIDTH + 12 * WIDTH + 7))
NOMINAL_TRAINER="${ROOT}/saainit_net16_v2/run_all_net16_v2.py"
SELECTOR="${ROOT}/select_width13_lr.py"
WORKER="${ROOT}/rho_sweep_net16_v2/run_rho_seed_net16_v2.py"
AUDIT_DIR="${ROOT}/width${WIDTH}_lr_audit"
NOMINAL_DIR="${ROOT}/saainit_net${WIDTH}_v2"
SWEEP_DIR="${ROOT}/rho_sweep_net${WIDTH}_v2"
PIPELINE_LOG_DIR="${ROOT}/width${WIDTH}_pipeline_logs"
RUNTIME_PATH="${ROOT}/width${WIDTH}_pipeline_runtime.json"
AUDIT_PARALLEL=${AUDIT_PARALLEL:-15}
SWEEP_PARALLEL=${SWEEP_PARALLEL:-24}
LRS=(0.005 0.01 0.05)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
RHOS=(0.001 0.0015 0.002 0.003 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.450 0.500 0.550 0.600 0.650 0.700 0.750 0.800 1.000 1.500)
STARTED_AT_UTC=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
SECONDS=0

mkdir -p "${AUDIT_DIR}" "${NOMINAL_DIR}" "${SWEEP_DIR}/logs" "${PIPELINE_LOG_DIR}"

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

lr_tag() {
    "${PYTHON}" - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.3f}".replace(".", "p"))
PY
}

is_completed() {
    local summary="$1"
    [[ -f "${summary}" ]] || return 1
    "${PYTHON}" - "${summary}" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("completed") is True else 1)
PY
}

write_runtime() {
    local stage="$1"
    local status="$2"
    local selected_lr="${3:-}"
    local ended_at="${4:-}"
    "${PYTHON}" - "${RUNTIME_PATH}" "${STARTED_AT_UTC}" "${ended_at}" \
        "${SECONDS}" "${stage}" "${status}" "${selected_lr}" \
        "${AUDIT_PARALLEL}" "${SWEEP_PARALLEL}" "${WIDTH}" \
        "${TOTAL_PARAMETERS}" <<'PY'
import json, pathlib, sys
(path, started, ended, elapsed, stage, status, selected_lr,
 audit_workers, sweep_workers, width, total_parameters) = sys.argv[1:]
width = int(width)
record = {
    "experiment": f"retailer_R3_N50_net{width}_lr_audit_and_rho_sweep",
    "status": status,
    "stage": stage,
    "hidden_widths": [width, width],
    "total_parameters_including_Q": int(total_parameters),
    "candidate_lrs": [0.005, 0.01, 0.05],
    "radii": [0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05,
              0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
              0.7, 0.75, 0.8, 1.0, 1.5],
    "seeds": list(range(10)),
    "lr_selection": "minimum mean independent-validation cost across 10 seeds",
    "test_data_used_for_selection": False,
    "audit_parallel_workers": int(audit_workers),
    "sweep_parallel_workers": int(sweep_workers),
    "thread_limit_per_worker": 1,
    "started_at_utc": started,
    "current_invocation_elapsed_time_s": float(elapsed),
}
if selected_lr:
    record["selected_lr_net"] = float(selected_lr)
if ended:
    record["ended_at_utc"] = ended
pathlib.Path(path).write_text(json.dumps(record, indent=2))
PY
}

write_runtime "lr_audit" "running"
audit_failures=0
for lr in "${LRS[@]}"; do
    tag=$(lr_tag "${lr}")
    for seed in "${SEEDS[@]}"; do
        candidate_dir="${AUDIT_DIR}/lr_${tag}/seed_${seed}"
        summary="${candidate_dir}/seed_${seed}_summary.json"
        if is_completed "${summary}"; then
            continue
        fi
        while (( $(jobs -r | wc -l) >= AUDIT_PARALLEL )); do
            wait -n || audit_failures=$((audit_failures + 1))
        done
        mkdir -p "${candidate_dir}"
        log="${PIPELINE_LOG_DIR}/audit_lr_${tag}_seed_${seed}.log"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
            "${PYTHON}" "${NOMINAL_TRAINER}" --hidden-width "${WIDTH}" --nominal-only \
            --fixed-lr "${lr}" --seeds "${seed}" --base-dir "${candidate_dir}" \
            > "${log}" 2>&1 &
    done
done
while (( $(jobs -r | wc -l) > 0 )); do
    wait -n || audit_failures=$((audit_failures + 1))
done
if (( audit_failures > 0 )); then
    write_runtime "lr_audit" "failed"
    echo "LR audit failed in ${audit_failures} worker(s)." >&2
    exit 1
fi

"${PYTHON}" "${SELECTOR}" --hidden-width "${WIDTH}" \
    | tee "${PIPELINE_LOG_DIR}/lr_selection.log"
selected_lr=$("${PYTHON}" - "${AUDIT_DIR}/lr_selection.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_lr_net"])
PY
)
write_runtime "rho_sweep" "running" "${selected_lr}"

sweep_failures=0
for rho in "${RHOS[@]}"; do
    tag=$(rho_tag "${rho}")
    for seed in "${SEEDS[@]}"; do
        summary="${SWEEP_DIR}/rho_${tag}/seed_${seed}_summary.json"
        if is_completed "${summary}"; then
            continue
        fi
        while (( $(jobs -r | wc -l) >= SWEEP_PARALLEL )); do
            wait -n || sweep_failures=$((sweep_failures + 1))
        done
        log="${SWEEP_DIR}/logs/rho_${tag}_seed_${seed}.log"
        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
            NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
            nice -n 10 "${PYTHON}" "${WORKER}" --rho "${rho}" --seed "${seed}" \
            --nominal-dir "${NOMINAL_DIR}" --sweep-dir "${SWEEP_DIR}" \
            --hidden-width "${WIDTH}" --lr-net "${selected_lr}" > "${log}" 2>&1 &
    done
done
while (( $(jobs -r | wc -l) > 0 )); do
    wait -n || sweep_failures=$((sweep_failures + 1))
done
if (( sweep_failures > 0 )); then
    write_runtime "rho_sweep" "failed" "${selected_lr}"
    echo "Robust sweep failed in ${sweep_failures} worker(s)." >&2
    exit 1
fi

ENDED_AT_UTC=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
write_runtime "rho_sweep" "completed" "${selected_lr}" "${ENDED_AT_UTC}"
echo "Width-${WIDTH} LR audit and robust sweep completed."