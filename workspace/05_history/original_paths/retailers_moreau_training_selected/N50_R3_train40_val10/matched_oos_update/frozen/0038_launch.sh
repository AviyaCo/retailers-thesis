#!/usr/bin/env bash
set -u
cd /data/home/aviyac/retailers_moreau_training_selected/N50_R3_train40_val10 || exit 1
stamp=$(date -u +%Y%m%dT%H%M%S)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
/data/home/aviyac/ap_env/bin/python -B -u run_campaign.py > "campaign_${stamp}.log" 2>&1
status=$?
printf '%s\n' "$status" > "campaign_${stamp}.exit"
exit "$status"