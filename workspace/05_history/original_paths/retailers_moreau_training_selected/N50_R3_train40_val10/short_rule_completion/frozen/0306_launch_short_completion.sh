#!/usr/bin/env bash
set -u
cd /data/home/aviyac/retailers_moreau_training_selected/N50_R3_train40_val10 || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1
/data/home/aviyac/ap_env/bin/python -B -u fill_short_completion.py run > short_rule_completion/campaign.log 2>&1
code=$?
printf '%s\n' "$code" > short_rule_completion/campaign.exit
exit "$code"