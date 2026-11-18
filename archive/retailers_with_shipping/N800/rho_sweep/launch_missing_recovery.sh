#!/bin/bash
# Relaunch missing N=800 fixed-shipping rho-sweep workers without duplicating
# completed or currently active rho/seed jobs.

set -euo pipefail

MAX_PARALLEL=${1:-10}
SWEEP_DIR="/data/home/aviyac/retailers_with_shipping/N800/rho_sweep"
SCRIPT="${SWEEP_DIR}/run_rho_seed.py"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"

source /data/home/aviyac/ap_env/bin/activate

RHOS=(0.001 0.003 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.500 0.700 1.000 1.500 2.000)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

rho_tag() {
    python3 - "$1" <<'PY'
import sys
v = float(sys.argv[1])
s = f'{v:.6f}'.rstrip('0')
parts = s.split('.')
if s.endswith('.'):
    s += '000'
elif len(parts) > 1 and len(parts[1]) < 3:
    s += '0' * (3 - len(parts[1]))
print(s.replace('.', 'p'))
PY
}

is_completed() {
    local summary="$1"
    [[ -f "${summary}" ]] || return 1
    python3 - "${summary}" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get('completed') is True else 1)
PY
}

is_active() {
    local rho="$1"
    local seed="$2"
    pgrep -f "${SCRIPT} --rho ${rho} --seed ${seed}" >/dev/null 2>&1
}

echo "======================================================"
echo "Shipping N=800 recovery launcher"
echo "  Max parallel: ${MAX_PARALLEL}"
echo "======================================================"

total=0
completed=0
active=0
launched=0

for rho in "${RHOS[@]}"; do
    tag=$(rho_tag "${rho}")
    for seed in "${SEEDS[@]}"; do
        total=$((total + 1))
        summary="${SWEEP_DIR}/rho_${tag}/seed_${seed}_summary.json"
        if is_completed "${summary}"; then
            completed=$((completed + 1))
            continue
        fi
        if is_active "${rho}" "${seed}"; then
            active=$((active + 1))
            continue
        fi

        while (( $(jobs -r | wc -l) >= MAX_PARALLEL )); do
            wait -n || true
        done

        logfile="${LOG_DIR}/recovery_rho_${tag}_seed_${seed}_$(date +%Y%m%d_%H%M%S).log"
        python3 "${SCRIPT}" --rho "${rho}" --seed "${seed}" > "${logfile}" 2>&1 &
        launched=$((launched + 1))
        echo "  Launched rho=${rho} seed=${seed}  log=${logfile}"
    done
done

wait

echo ""
echo "======================================================"
echo "Recovery launcher done."
echo "  Total checked : ${total}"
echo "  Completed     : ${completed}"
echo "  Active skipped: ${active}"
echo "  Launched      : ${launched}"
echo "======================================================"