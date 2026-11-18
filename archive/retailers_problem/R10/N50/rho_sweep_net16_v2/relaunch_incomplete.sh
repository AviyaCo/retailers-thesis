#!/bin/bash
# relaunch_incomplete.sh  --  Relaunch missing + non-converged R10 N50 jobs
# Uses --max_iter 25 for all.
set -euo pipefail
MAX_PARALLEL=${1:-20}
MAX_ITER=25

BASE="/data/home/aviyac/retailers_problem/R10/N50/rho_sweep_net16_v2"
SCRIPT="${BASE}/run_rho_seed_net16_v2_N50_R10.py"
LOG_DIR="${BASE}/logs"
PYTHON="/data/home/aviyac/ap_env/bin/python3"

source /data/home/aviyac/ap_env/bin/activate

# ── Delete summaries for non-converged (completed but not converged) ──────────
echo "Removing non-converged summaries..."
$PYTHON -c "
import json, glob, os
base = '${BASE}'
removed = 0
for p in glob.glob(f'{base}/rho_*/seed_*_summary.json'):
    with open(p) as f: d = json.load(f)
    if d.get('completed', False) and not d.get('converged', False):
        os.remove(p)
        print(f'  removed {p}')
        removed += 1
print(f'Removed {removed} non-converged summaries.')
"

# ── Jobs to launch: all (rho,seed) without completed summary ─────────────────
RHOS=(0.001 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.450 0.500 0.600 0.750 0.800 1.000 1.500)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

is_done() {
    local summ="$1"
    [ -f "$summ" ] && $PYTHON -c "
import json, sys
try:
    d = json.load(open('$summ'))
    sys.exit(0 if d.get('completed', False) else 1)
except:
    sys.exit(1)
" 2>/dev/null
}

make_tag() {
    $PYTHON -c "
v = float('$1')
s = f'{v:.6f}'.rstrip('0')
parts = s.split('.')
if len(parts[1]) < 3:
    parts[1] = parts[1].ljust(3, '0')
s = '.'.join(parts)
print(s.replace('.', 'p'))
"
}

launched=0; skipped=0
for rho in "${RHOS[@]}"; do
    tag=$(make_tag "$rho")
    for seed in "${SEEDS[@]}"; do
        summ="${BASE}/rho_${tag}/seed_${seed}_summary.json"
        if is_done "$summ"; then
            skipped=$((skipped + 1))
            continue
        fi
        while true; do
            running=$(jobs -rp 2>/dev/null | wc -l)
            [ "$running" -lt "$MAX_PARALLEL" ] && break
            sleep 5
        done
        log="${LOG_DIR}/relaunch_rho${rho}_seed${seed}.txt"
        $PYTHON "${SCRIPT}" --rho "${rho}" --seed "${seed}" --max_iter "${MAX_ITER}" \
            > "${log}" 2>&1 &
        launched=$((launched + 1))
        echo "$(date +%H:%M:%S) Launched rho=${rho} seed=${seed} max_iter=${MAX_ITER} (pid=$!)"
    done
done

wait
echo ""
echo "=== All done: launched=${launched}  skipped=${skipped} ==="
