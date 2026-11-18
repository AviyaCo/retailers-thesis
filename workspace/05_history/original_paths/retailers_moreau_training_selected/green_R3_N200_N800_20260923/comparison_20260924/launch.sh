#!/usr/bin/env bash
# Parent starts this launcher inside a named external tmux session.
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'Usage: bash launch.sh ABSOLUTE_PREFLIGHT_JSON [NEW_OUTPUT_DIRECTORY]\n' >&2
    exit 2
fi
if [[ -z "${TMUX:-}" ]]; then
    printf 'Launch inside a named tmux session; this script does not create one.\n' >&2
    exit 2
fi
pref=$(readlink -f -- "$1")
stamp=$(date -u +%Y%m%dT%H%M%S.%N)
export COMPARISON_LOG="$PWD/campaign_${stamp}.log"
export COMPARISON_EXIT="$PWD/campaign_${stamp}.exit"
export COMPARISON_TMUX_SESSION
COMPARISON_TMUX_SESSION=$(tmux display-message -p '#S')
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
set -o noclobber
: > "$COMPARISON_LOG"
trap 'code=$?; printf "%s\n" "$code" > "$COMPARISON_EXIT"' EXIT
command=(/data/home/aviyac/ap_env/bin/python -B -u "$PWD/run_comparison.py" --preflight "$pref")
if [[ $# -eq 2 ]]; then command+=(--output "$2"); fi
printf 'Session: %s\nLog: %s\nExit: %s\nCommand: ' "$COMPARISON_TMUX_SESSION" "$COMPARISON_LOG" "$COMPARISON_EXIT"
printf '%q ' "${command[@]}"
printf '\n'
{
    printf 'UTC start: %s\nWorking directory: %s\nSession: %s\nCommand: ' "$(date -u --iso-8601=seconds)" "$PWD" "$COMPARISON_TMUX_SESSION"
    printf '%q ' "${command[@]}"
    printf '\n'
    "${command[@]}"
} >> "$COMPARISON_LOG" 2>&1