#!/usr/bin/env bash
# Explicit invocation only. This file is never called by the preflight.
set -u
cd /data/home/aviyac/retailers_moreau_training_selected/green_R10_N50_N200_N800_20260924 || exit 1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
if [[ ${1:-} == --inside-tmux ]]; then
    stamp=$2
    preflight=$3
    /data/home/aviyac/ap_env/bin/python -B -u run_campaign.py --preflight "$preflight" > "campaign_${stamp}.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "campaign_${stamp}.exit"
    exit "$code"
fi
if [[ $# != 1 || ! -f $1 ]]; then
    printf 'Usage: bash launch.sh COMPLETED_PREFLIGHT_SUMMARY\n' >&2
    exit 2
fi
command -v tmux >/dev/null || { printf 'tmux required; no silent nohup fallback\n' >&2; exit 1; }
stamp=$(date -u +%Y%m%dT%H%M%S%N)
session="green_r10_${stamp}"
printf -v command 'bash %q --inside-tmux %q %q' "$PWD/launch.sh" "$stamp" "$(realpath "$1")"
tmux new-session -d -s "$session" "$command" || exit 1
printf 'session=%s\ncommand=%s\nlog=%s/campaign_%s.log\nexit=%s/campaign_%s.exit\nattach: tmux attach -t %s\nstatus: tmux has-session -t %s\n' "$session" "$command" "$PWD" "$stamp" "$PWD" "$stamp" "$session" "$session"