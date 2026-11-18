#!/bin/bash
# launch_net16_v2_N50_R10.sh  --  Wait for saainit, then run R=10 N=50 rho sweep
# ===============================================================================
# Usage:  bash launch_net16_v2_N50_R10.sh [max_parallel]
# Default max_parallel = 20.

set -euo pipefail
MAX_PARALLEL=${1:-20}

SAAINIT_DIR="/data/home/aviyac/retailers_problem/R10/N50/saainit_net16_v2"
SWEEP_DIR="/data/home/aviyac/retailers_problem/R10/N50/rho_sweep_net16_v2"
SCRIPT="${SWEEP_DIR}/run_rho_seed_net16_v2_N50_R10.py"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"

source /data/home/aviyac/ap_env/bin/activate

RHOS=(0.001 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.450 0.500 0.600 0.750 0.800 1.000 1.500)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

echo "======================================================"
echo "net16_v2 N=50 R=10 (truncnorm) robustness sweep"
echo "  RHOs  : ${#RHOS[@]} values"
echo "  Seeds : ${#SEEDS[@]}"
echo "  Total : $((${#RHOS[@]} * ${#SEEDS[@]})) jobs"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "======================================================"

# ── Incremental launch: start jobs as soon as each nominal is ready ────────────
echo ""
echo "Incremental sweep launcher — launching jobs as nominals become available..."

total=$(( ${#RHOS[@]} * ${#SEEDS[@]} ))
skipped=0; launched=0
declare -A seed_launched

make_tag() {
    python3 -c "
v = float('$1')
s = f'{v:.6f}'.rstrip('0')
parts = s.split('.')
if len(parts[1]) < 3:
    parts[1] = parts[1].ljust(3, '0')
s = '.'.join(parts)
print(s.replace('.', 'p'))
"
}

is_done() {
    local summ="$1"
    [ -f "$summ" ] && python3 -c "
import json, sys
try:
    d = json.load(open('$summ'))
    sys.exit(0 if d.get('completed', False) else 1)
except:
    sys.exit(1)
" 2>/dev/null
}

while true; do
    all_seeds_launched=true
    for seed in "${SEEDS[@]}"; do
        if [ "${seed_launched[$seed]+_}" ]; then continue; fi
        nominal="${SAAINIT_DIR}/seed_${seed}_nominal.pth"
        if [ ! -f "$nominal" ]; then
            all_seeds_launched=false
            continue
        fi
        echo "  $(date '+%H:%M:%S')  Nominal for seed=${seed} ready — launching ${#RHOS[@]} rho jobs..."
        for rho in "${RHOS[@]}"; do
            tag=$(make_tag "$rho")
            summ="${SWEEP_DIR}/rho_${tag}/seed_${seed}_summary.json"
            if is_done "$summ"; then
                skipped=$((skipped + 1))
                continue
            fi
            while true; do
                running=$(jobs -rp 2>/dev/null | wc -l)
                [ "$running" -lt "$MAX_PARALLEL" ] && break
                sleep 5
            done
            log="${LOG_DIR}/rho${rho}_seed${seed}.txt"
            python3 "${SCRIPT}" --rho "${rho}" --seed "${seed}" > "${log}" 2>&1 &
            launched=$((launched + 1))
            echo "Launched [${launched}/${total}]  rho=${rho}  seed=${seed}  (pid=$!)"
        done
        seed_launched[$seed]=1
    done
    $all_seeds_launched && break
    sleep 30
done

wait
echo ""
echo "======================================================="
echo "All jobs done."
echo "  Total   : ${total}"
echo "  Launched: ${launched}"
echo "  Skipped : ${skipped}"
echo "======================================================="
