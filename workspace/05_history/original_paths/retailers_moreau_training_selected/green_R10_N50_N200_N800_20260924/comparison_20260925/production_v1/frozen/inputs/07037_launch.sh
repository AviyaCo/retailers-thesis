#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec /data/home/aviyac/ap_env/bin/python -B -u launch.py "$@"