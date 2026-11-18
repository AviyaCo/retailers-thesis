#!/bin/bash
# boost_sweep_remaining_net16_v2_N800_R10.sh -- duplicate-safe R10 N=800 rho booster.

set -euo pipefail

MAX_TOTAL=${1:-10}

SAAINIT_DIR="/data/home/aviyac/retailers_problem/R10/N800/saainit_net16_v2"
SWEEP_DIR="/data/home/aviyac/retailers_problem/R10/N800/rho_sweep_net16_v2"
SCRIPT="${SWEEP_DIR}/run_rho_seed_net16_v2_N800_R10.py"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"

source /data/home/aviyac/ap_env/bin/activate

RHOS=(1.500 1.000 0.800 0.750 0.600 0.500 0.450 0.400 0.300 0.200 0.150 0.100 0.050 0.020 0.010 0.005 0.001)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

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

is_completed() {
    local summ="$1"
    [[ -f "$summ" ]] && python3 -c "
import json, sys
try:
    d = json.load(open('$summ'))
    sys.exit(0 if d.get('completed', False) else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

is_running() {
    local rho="$1"
    local seed="$2"
    pgrep -f "${SCRIPT} --rho ${rho} --seed ${seed}" >/dev/null 2>&1
}

active_count() {
    pgrep -fc "${SCRIPT} --rho" || true
}

completed_count() {
    find "${SWEEP_DIR}" -path '*/rho_*/*_summary.json' -print0 2>/dev/null |
        xargs -0 -r python3 -c '
import json, sys
n = 0
for path in sys.argv[1:]:
    try:
        n += bool(json.load(open(path)).get("completed", False))
    except Exception:
        pass
print(n)
'
}

total=$(( ${#RHOS[@]} * ${#SEEDS[@]} ))

echo "======================================================"
echo "R10 N=800 duplicate-safe rho booster"
echo "  Total jobs       : ${total}"
echo "  Max active total : ${MAX_TOTAL}"
echo "======================================================"

while true; do
    done_now=$(completed_count)
    running_now=$(active_count)
    echo "$(date '+%F %T') completed=${done_now}/${total} active=${running_now}/${MAX_TOTAL}"
    [[ "${done_now}" -ge "${total}" ]] && break

    launched_any=false
    for seed in "${SEEDS[@]}"; do
        nominal="${SAAINIT_DIR}/seed_${seed}_nominal.pth"
        [[ -f "${nominal}" ]] || continue
        for rho in "${RHOS[@]}"; do
            tag=$(make_tag "${rho}")
            summ="${SWEEP_DIR}/rho_${tag}/seed_${seed}_summary.json"
            is_completed "${summ}" && continue
            is_running "${rho}" "${seed}" && continue

            while (( $(active_count) >= MAX_TOTAL )); do
                sleep 10
            done
            is_completed "${summ}" && continue
            is_running "${rho}" "${seed}" && continue

            logfile="${LOG_DIR}/boost_rho_${tag}_seed_${seed}.log"
            python3 "${SCRIPT}" --rho "${rho}" --seed "${seed}" > "${logfile}" 2>&1 &
            echo "  launched rho=${rho} seed=${seed} pid=$! log=${logfile}"
            launched_any=true
        done
    done

    ${launched_any} || sleep 30
done

wait
echo "All R10 N=800 rho jobs completed."