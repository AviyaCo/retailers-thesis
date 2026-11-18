#!/bin/bash
# relaunch_final.sh -- Relaunch non-converged shipping N800 seeds with more iterations
set -euo pipefail

BASE="/data/home/aviyac/retailers_with_shipping/N800/rho_sweep"
SCRIPT="${BASE}/run_rho_seed.py"
LOG_DIR="${BASE}/logs"
PYTHON="/data/home/aviyac/ap_env/bin/python3"

source /data/home/aviyac/ap_env/bin/activate

mkdir -p "${LOG_DIR}"

# Non-converged: (rho, seed) pairs
# rho_0p700 seeds 5,9 and rho_1p000 seed 8 hit default max_iter=8
# rho_0p500 seed 9, rho_1p000 seed 9, rho_1p500 seed 9 hit max_iter=25
JOBS=(
    "0.500 9"
    "0.700 5"
    "0.700 9"
    "1.000 8"
    "1.000 9"
    "1.500 9"
)

for job in "${JOBS[@]}"; do
    rho=$(echo $job | awk '{print $1}')
    seed=$(echo $job | awk '{print $2}')
    echo "Launching rho=${rho} seed=${seed} (max_iter=50, force)..."
    $PYTHON "${SCRIPT}" --rho ${rho} --seed ${seed} --max_iter 50 --force \
        > "${LOG_DIR}/relaunch_final_rho${rho}_seed${seed}.txt" 2>&1 &
    echo "  pid=$!"
done

wait
echo ""
echo "=== All done ==="
