#!/bin/bash
# launch_timing_test.sh -- Run rho=0.001, 0.3, 1.5 x seeds 10/11/12 for clean timing data
# Seeds 10-12 are new (not in the original sweep) so they run from scratch with timing fixed.

set -euo pipefail

SWEEP_DIR="/data/home/aviyac/retailers_problem/R3/N50/rho_sweep_net16_v2"
SCRIPT="${SWEEP_DIR}/run_rho_seed_net16_v2.py"
LOG_DIR="${SWEEP_DIR}/logs/timing"
mkdir -p "${LOG_DIR}"

source /data/home/aviyac/ap_env/bin/activate

RHOS=(0.001 0.300 1.500)
SEEDS=(0 1 2)

echo "======================================================"
echo "Timing test: rho=[0.001, 0.3, 1.5]  seeds=[0, 1, 2]  (--force)"
echo "  Total: 9 jobs  (3 parallel)"
echo "======================================================"

pids=()
for rho in "${RHOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        log="${LOG_DIR}/rho${rho}_seed${seed}.txt"
        echo "Launching rho=${rho}  seed=${seed}"
        python3 "${SCRIPT}" --rho "${rho}" --seed "${seed}" --max_iter 15 --force \
            > "${log}" 2>&1 &
        pids+=($!)
    done
done

echo ""
echo "Waiting for all 9 jobs (PIDs: ${pids[*]})..."
wait
echo ""
echo "All done! Collect timings with:"
echo "  python3 ${SWEEP_DIR}/collect_timing.py"
