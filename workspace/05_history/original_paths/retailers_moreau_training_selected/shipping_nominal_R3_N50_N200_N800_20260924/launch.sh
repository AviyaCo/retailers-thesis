#!/usr/bin/env bash
set -euo pipefail
cd /data/home/aviyac/retailers_moreau_training_selected/shipping_nominal_R3_N50_N200_N800_20260924
exec /data/home/aviyac/ap_env/bin/python -B -u launch.py --inside "$@"