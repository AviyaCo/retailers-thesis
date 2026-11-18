#!/bin/bash
# launch_sweep.sh  --  Run all (rho, seed) CCG jobs for shipping problem N=200
# ============================================================================
# Usage:  bash launch_sweep.sh [max_parallel]
# Default max_parallel = 10

set -euo pipefail

MAX_PARALLEL=${1:-10}

SWEEP_DIR="/data/home/aviyac/retailers_with_shipping/N200/rho_sweep"
SCRIPT="${SWEEP_DIR}/run_rho_seed.py"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"

source /data/home/aviyac/ap_env/bin/activate

RHOS=(0.001 0.003 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.500 0.700 1.000 1.500 2.000)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

total=0; skipped=0; launched=0

echo "======================================================"
echo "Shipping N=200 robustness sweep launcher"
echo "  RHOs  : ${#RHOS[@]} values"
echo "  Seeds : ${#SEEDS[@]}"
echo "  Total : $((${#RHOS[@]} * ${#SEEDS[@]})) jobs"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "======================================================"

for rho in "${RHOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))

        # Compute rho tag (same logic as in run_rho_seed.py)
        tag=$(python3 -c "
v = float('$rho')
s = f'{v:.6f}'.rstrip('0')
parts = s.split('.')
if s.endswith('.'):
    s += '000'
elif len(parts) > 1 and len(parts[1]) < 3:
    s += '0' * (3 - len(parts[1]))
print(s.replace('.', 'p'))
")
        summ="${SWEEP_DIR}/rho_${tag}/seed_${seed}_summary.json"

        # Skip if already completed
        if [[ -f "$summ" ]]; then
            completed=$(python3 -c "import json; d=json.load(open('$summ')); print(d.get('completed', False))" 2>/dev/null || echo "False")
            if [[ "$completed" == "True" ]]; then
                skipped=$((skipped + 1))
                continue
            fi
        fi

        # Throttle: wait if too many jobs running
        while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
            sleep 5
        done

        logfile="${LOG_DIR}/rho_${tag}_seed_${seed}.log"
        python3 "${SCRIPT}" --rho "${rho}" --seed "${seed}" > "${logfile}" 2>&1 &
        launched=$((launched + 1))
        echo "  Launched rho=${rho} seed=${seed}  (log: ${logfile})"
    done
done

# Wait for all background jobs
wait

echo ""
echo "======================================================"
echo "All jobs done."
echo "  Total : ${total}"
echo "  Skipped (done): ${skipped}"
echo "  Launched: ${launched}"
echo "======================================================"
