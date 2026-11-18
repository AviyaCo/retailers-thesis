#!/bin/bash
# accelerate_nominal_and_sweep.sh -- Launch missing N=200 shipping nominal seeds now,
# then start rho sweep after all 10 nominal CV checkpoints exist.
# Usage: bash accelerate_nominal_and_sweep.sh [sweep_parallel]

set -euo pipefail

SWEEP_PARALLEL=${1:-10}
BASE="/data/home/aviyac/retailers_with_shipping/N200"
PYTHON="/data/home/aviyac/ap_env/bin/python3"
CV_SCRIPT="${BASE}/run_cv_nominal.py"
CV_LOG_DIR="${BASE}/cv_logs"
SWEEP_SCRIPT="${BASE}/rho_sweep/launch_sweep.sh"
SEEDS=(0 1 2 3 4 5 6 7 8 9)
START_NOW=(3 4 5 6 7 8 9)

mkdir -p "${CV_LOG_DIR}"
source /data/home/aviyac/ap_env/bin/activate

has_cv_checkpoint() {
    local ckpt="$1"
    [[ -f "${ckpt}" ]] || return 1
    "${PYTHON}" - "${ckpt}" <<'PY'
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location="cpu")
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if "cv_lr_used" in ck else 1)
PY
}

echo "======================================================"
echo "Shipping N=200 accelerated nominal launcher"
echo "  Launching missing seeds: ${START_NOW[*]}"
echo "  Sweep parallel after all nominals: ${SWEEP_PARALLEL}"
echo "======================================================"

launched=0
for seed in "${START_NOW[@]}"; do
    ckpt="${BASE}/seed_${seed}_nominal.pth"
    if has_cv_checkpoint "${ckpt}"; then
        echo "  skip seed=${seed}: completed checkpoint exists"
        continue
    fi
    if pgrep -f "${CV_SCRIPT} --seed ${seed}" >/dev/null; then
        echo "  skip seed=${seed}: already running"
        continue
    fi
    log="${CV_LOG_DIR}/seed_${seed}.log"
    "${PYTHON}" "${CV_SCRIPT}" --seed "${seed}" > "${log}" 2>&1 &
    launched=$((launched + 1))
    echo "  launched nominal seed=${seed}  pid=$!  log=${log}"
done

wait
echo "Launched missing nominal seeds finished. launched=${launched}"

while true; do
    missing=()
    for seed in "${SEEDS[@]}"; do
        ckpt="${BASE}/seed_${seed}_nominal.pth"
        if ! has_cv_checkpoint "${ckpt}"; then
            missing+=("${seed}")
        fi
    done
    if (( ${#missing[@]} == 0 )); then
        break
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') waiting for nominal checkpoints: ${missing[*]}"
    sleep 60
done

echo "All nominal CV checkpoints ready. Starting rho sweep..."
bash "${SWEEP_SCRIPT}" "${SWEEP_PARALLEL}"
