#!/usr/bin/env bash
set -u
cd /data/home/aviyac/retailers_moreau_training_selected/N50_R3_train40_val10 || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
/data/home/aviyac/ap_env/bin/python -B -u update_comparison.py run > matched_oos_update/campaign.log 2>&1
rc=$?
printf '%s\n' "$rc" > matched_oos_update/campaign.exit
exit "$rc"