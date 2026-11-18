#!/usr/bin/env bash
set -u
cd /data/home/aviyac/retailers_moreau_training_selected/green_R3_N200_N800_20260923 || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
stamp=$(date -u +%Y%m%dT%H%M%S)
/data/home/aviyac/ap_env/bin/python -B -u run_campaign.py --preflight "$1" > "campaign_${stamp}.log" 2>&1
code=$?
printf '%s\n' "$code" > "campaign_${stamp}.exit"
exit "$code"