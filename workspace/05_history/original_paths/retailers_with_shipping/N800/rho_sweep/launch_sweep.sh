#!/bin/bash
# launch_sweep.sh  --  Run all (rho, seed) CCG jobs for shipping problem N=800
# ============================================================================
# Usage:  bash launch_sweep.sh [max_parallel]
# Default max_parallel = 1

set -euo pipefail

MAX_PARALLEL=${1:-1}

SCRIPT_DIR="/data/home/aviyac/retailers_with_shipping/N800/rho_sweep"
SWEEP_DIR="/data/home/aviyac/retailers_with_shipping/N800/rho_sweep_minibatch"
SCRIPT="${SCRIPT_DIR}/run_rho_seed.py"
LOG_DIR="${SWEEP_DIR}/logs"
mkdir -p "${LOG_DIR}"

source /data/home/aviyac/ap_env/bin/activate

RHOS=(0.001 0.003 0.005 0.010 0.020 0.050 0.100 0.150 0.200 0.300 0.400 0.500 0.700 1.000 1.500)
SEEDS=(0 1 2 3 4 5 6 7 8 9)
STARTED_AT_UTC=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
SECONDS=0

python3 - "${SWEEP_DIR}/sweep_runtime.json" "${STARTED_AT_UTC}" "${MAX_PARALLEL}" <<'PY'
import json
import sys

path, started, workers = sys.argv[1:]
with open(path, "w") as stream:
    json.dump({
        "experiment": "fixed_shipping_N800_neural_ccg_minibatch_mp",
        "method": "Adam minibatch MP retraining",
        "n_samples": 800,
        "radii": [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15,
                  0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5],
        "seeds": list(range(10)),
        "mp_batch_size": 200,
        "max_parallel_workers": int(workers),
        "started_at_utc": started,
        "status": "running",
    }, stream, indent=2)
PY

total=0; skipped=0; launched=0

echo "======================================================"
echo "Shipping N=800 robustness sweep launcher"
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
            wait -n
        done

        logfile="${LOG_DIR}/rho_${tag}_seed_${seed}.log"
        python3 "${SCRIPT}" --rho "${rho}" --seed "${seed}" > "${logfile}" 2>&1 &
        launched=$((launched + 1))
        echo "  Launched rho=${rho} seed=${seed}  (log: ${logfile})"
    done
done

# Wait for all background jobs
wait

ELAPSED_SECONDS=${SECONDS}
ENDED_AT_UTC=$(date --utc +%Y-%m-%dT%H:%M:%SZ)
python3 - "${SWEEP_DIR}/sweep_runtime.json" "${STARTED_AT_UTC}" "${ENDED_AT_UTC}" \
    "${ELAPSED_SECONDS}" "${MAX_PARALLEL}" "${total}" "${skipped}" "${launched}" <<'PY'
import json
import sys

path, started, ended, elapsed, workers, total, skipped, launched = sys.argv[1:]
with open(path, "w") as stream:
    json.dump({
        "experiment": "fixed_shipping_N800_neural_ccg_minibatch_mp",
        "method": "Adam minibatch MP retraining",
        "n_samples": 800,
        "radii": [0.001, 0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15,
                  0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5],
        "seeds": list(range(10)),
        "mp_batch_size": 200,
        "max_parallel_workers": int(workers),
        "started_at_utc": started,
        "ended_at_utc": ended,
        "sweep_elapsed_time_s": float(elapsed),
        "jobs_total": int(total),
        "jobs_skipped_completed": int(skipped),
        "jobs_launched_current_invocation": int(launched),
        "status": "completed",
    }, stream, indent=2)
PY

echo ""
echo "======================================================"
echo "All jobs done."
echo "  Total : ${total}"
echo "  Skipped (done): ${skipped}"
echo "  Launched: ${launched}"
echo "======================================================"
