#!/bin/bash
# launch_pipeline.sh -- Run shipping N=200 nominal CV checkpoints, then rho sweep.
# Usage: bash launch_pipeline.sh [cv_parallel] [sweep_parallel]

set -euo pipefail

CV_PARALLEL=${1:-3}
SWEEP_PARALLEL=${2:-10}

BASE="/data/home/aviyac/retailers_with_shipping/N200"
PYTHON="/data/home/aviyac/ap_env/bin/python3"
CV_SCRIPT="${BASE}/run_cv_nominal.py"
CV_LOG_DIR="${BASE}/cv_logs"
SWEEP_SCRIPT="${BASE}/rho_sweep/launch_sweep.sh"

mkdir -p "${CV_LOG_DIR}"
source /data/home/aviyac/ap_env/bin/activate

echo "======================================================"
echo "Shipping N=200 pipeline"
echo "  CV parallel: ${CV_PARALLEL}"
echo "  Sweep parallel: ${SWEEP_PARALLEL}"
echo "======================================================"

launched=0
skipped=0

for seed in 0 1 2 3 4 5 6 7 8 9; do
    ckpt="${BASE}/seed_${seed}_nominal.pth"
    if [[ -f "${ckpt}" ]]; then
        has_cv=$(${PYTHON} - <<PY
import torch
try:
    ck = torch.load("${ckpt}", map_location="cpu")
    print("yes" if "cv_lr_used" in ck else "no")
except Exception:
    print("no")
PY
)
        if [[ "${has_cv}" == "yes" ]]; then
            skipped=$((skipped + 1))
            continue
        fi
    fi

    while (( $(jobs -r | wc -l) >= CV_PARALLEL )); do
        wait -n || true
    done

    log="${CV_LOG_DIR}/seed_${seed}.log"
    ${PYTHON} "${CV_SCRIPT}" --seed "${seed}" > "${log}" 2>&1 &
    launched=$((launched + 1))
    echo "  launched nominal seed=${seed}  pid=$!  log=${log}"
done

wait
echo "Nominal CV done. launched=${launched} skipped=${skipped}"

echo "Starting rho sweep..."
bash "${SWEEP_SCRIPT}" "${SWEEP_PARALLEL}"
