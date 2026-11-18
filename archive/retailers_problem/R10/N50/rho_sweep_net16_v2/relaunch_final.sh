#!/bin/bash
# relaunch_final.sh -- Relaunch rho_0p750 seed 0 (non-converged) and rho_1p500 seed 5 (missing)
set -euo pipefail

BASE="/data/home/aviyac/retailers_problem/R10/N50/rho_sweep_net16_v2"
SCRIPT="${BASE}/run_rho_seed_net16_v2_N50_R10.py"
LOG_DIR="${BASE}/logs"
PYTHON="/data/home/aviyac/ap_env/bin/python3"

source /data/home/aviyac/ap_env/bin/activate

mkdir -p "${LOG_DIR}"

# Remove non-converged summary so the script doesn't skip it
SUMM_0p750="${BASE}/rho_0p750/seed_0_summary.json"
if [ -f "${SUMM_0p750}" ]; then
    echo "Removing non-converged summary: ${SUMM_0p750}"
    rm "${SUMM_0p750}"
fi

echo "Launching rho=0.750 seed=0 (max_iter=50)..."
$PYTHON "${SCRIPT}" --rho 0.750 --seed 0 --max_iter 50 \
    > "${LOG_DIR}/relaunch_final_rho0p750_seed0.txt" 2>&1 &
PID1=$!
echo "  pid=$PID1"

echo "Launching rho=1.500 seed=5 (max_iter=25)..."
$PYTHON "${SCRIPT}" --rho 1.500 --seed 5 --max_iter 25 \
    > "${LOG_DIR}/relaunch_final_rho1p500_seed5.txt" 2>&1 &
PID2=$!
echo "  pid=$PID2"

wait
echo ""
echo "=== All done ==="
