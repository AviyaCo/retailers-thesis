#!/bin/bash
# launch_net16_v2_N200.sh  --  Wait for saainit to finish, then run N=200 rho sweep
# ===================================================================================
# Usage:  bash launch_net16_v2_N200.sh [max_parallel]
# Default max_parallel = 20.
#
# Waits until all 10 seed_*_nominal.pth files exist in the N200 saainit dir,
# then launches the CCG sweep jobs in parallel.

set -euo pipefail

MAX_PARALLEL=${1:-20}

SAAINIT_DIR="/data/home/aviyac/retailers_problem/R3/N200/saainit_net16_v2"
SWEEP_DIR="/data/home/aviyac/retailers_problem/R3/N200/rho_sweep_net16_v2"
SCRIPT="${SWEEP_DIR}/run_rho_seed_net16_v2_N200.py"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"

source /data/home/aviyac/ap_env/bin/activate

RHOS=(0.001 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.450 0.500 0.600 0.750 0.800 1.000 1.500)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

echo "======================================================"
echo "Width-16 v2 N=200 (truncnorm) robustness sweep launcher"
echo "  RHOs  : ${#RHOS[@]} values"
echo "  Seeds : ${#SEEDS[@]}"
echo "  Total : $((${#RHOS[@]} * ${#SEEDS[@]})) jobs"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "======================================================"

# ── Incremental launch: start jobs as soon as each nominal is ready ────────────
echo ""
echo "Incremental sweep launcher — launching jobs as nominals become available..."

total=0; skipped=0; launched=0
declare -A seed_launched   # tracks which seeds have had their jobs submitted

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

total=$(( ${#RHOS[@]} * ${#SEEDS[@]} ))

while true; do
    all_seeds_launched=true
    for seed in "${SEEDS[@]}"; do
        if [ "${seed_launched[$seed]+_}" ]; then
            continue  # already submitted this seed's jobs
        fi
        nominal="${SAAINIT_DIR}/seed_${seed}_nominal.pth"
        if [ ! -f "$nominal" ]; then
            all_seeds_launched=false
            continue
        fi
        # nominal ready — launch all rho jobs for this seed
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

    if $all_seeds_launched; then
        break
    fi
    sleep 30
done

wait
echo ""
echo "======================================================="
echo "All jobs done."
echo "  Total  : ${total}"
echo "  Skipped: ${skipped} (already completed)"
echo "  Launched: ${launched}"
echo "======================================================="
