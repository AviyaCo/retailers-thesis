#!/bin/bash
# launch_net16_v2.sh  --  Run all (rho, seed) CCG jobs for width-16 network (truncnorm)
# ======================================================================================
# Usage:  bash launch_net16_v2.sh [max_parallel]
# Default max_parallel = 20.

set -euo pipefail

MAX_PARALLEL=${1:-20}

SWEEP_DIR="/data/home/aviyac/retailers_problem/R3/N50/rho_sweep_net16_v2"
SCRIPT="${SWEEP_DIR}/run_rho_seed_net16_v2.py"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"

source /data/home/aviyac/ap_env/bin/activate

RHOS=(0.001 0.0015 0.002 0.003 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.450 0.500 0.550 0.600 0.650 0.700 0.750 0.800 1.000 1.500 2.000)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

total=0; skipped=0; launched=0

echo "======================================================"
echo "Width-16 v2 (truncnorm) robustness sweep launcher"
echo "  RHOs  : ${#RHOS[@]} values"
echo "  Seeds : ${#SEEDS[@]}"
echo "  Total : $((${#RHOS[@]} * ${#SEEDS[@]})) jobs"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "======================================================"

for rho in "${RHOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))

        tag=$(python3 -c "
v = float('$rho')
s = f'{v:.6f}'.rstrip('0')
parts = s.split('.')
if len(parts[1]) < 3:
    parts[1] = parts[1].ljust(3, '0')
s = '.'.join(parts)
print(s.replace('.', 'p'))
")
        summ="${SWEEP_DIR}/rho_${tag}/seed_${seed}_summary.json"

        if [ -f "$summ" ]; then
            done=$(python3 -c "
import json, sys
try:
    d = json.load(open('$summ'))
    sys.exit(0 if d.get('completed', False) else 1)
except:
    sys.exit(1)
" 2>/dev/null && echo "yes" || echo "no")
            if [ "$done" = "yes" ]; then
                skipped=$((skipped + 1))
                continue
            fi
        fi

        while true; do
            running=$(jobs -rp 2>/dev/null | wc -l)
            if [ "$running" -lt "$MAX_PARALLEL" ]; then
                break
            fi
            sleep 5
        done

        log="${LOG_DIR}/rho${rho}_seed${seed}.txt"
        python3 "${SCRIPT}" --rho "${rho}" --seed "${seed}" > "${log}" 2>&1 &
        launched=$((launched + 1))
        echo "Launched [${launched}/${total}]  rho=${rho}  seed=${seed}  (pid=$!)"
    done
done

echo ""
echo "======================================================="
echo "All jobs launched."
echo "  Total  : ${total}"
echo "  Skipped: ${skipped} (already completed)"
echo "  Launched: ${launched}"
echo "======================================================="
echo ""
echo "Waiting for all background jobs to finish..."
wait
echo ""
echo "All done!  Next: python3 ${SWEEP_DIR}/eval_sweep_net16_v2.py"
