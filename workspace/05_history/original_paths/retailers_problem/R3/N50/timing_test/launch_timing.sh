#!/usr/bin/env bash
# launch_timing.sh — Run 9 full timing benchmarks (rho × seed) in parallel
# Each run times: SAA solve + nominal training + CCG iterations
# Output goes to timing_test/rho_XXX/seed_Y_timing.json

PYTHON=/data/home/aviyac/ap_env/bin/python3
SCRIPT=/data/home/aviyac/retailers_problem/R3/N50/timing_test/run_timing_seed.py
LOGDIR=/data/home/aviyac/retailers_problem/R3/N50/timing_test/logs
mkdir -p "$LOGDIR"

RHOS=(0.001 0.300 1.500)
SEEDS=(10 11 12)

for rho in "${RHOS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tag=$(printf "rho%.3f_seed%d" "$rho" "$seed")
    logfile="$LOGDIR/${tag}.log"
    echo "Launching rho=$rho seed=$seed -> $logfile"
    nohup "$PYTHON" "$SCRIPT" --rho "$rho" --seed "$seed" --max_iter 15 \
      > "$logfile" 2>&1 &
  done
done

echo ""
echo "All 9 jobs launched. Monitor with:"
echo "  tail -f $LOGDIR/rho0.001_seed0.log"
echo "  ls -lh $LOGDIR/"
