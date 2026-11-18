#!/bin/bash

set -euo pipefail

ROOT="/data/home/aviyac/retailers_problem/R3/N50"
export WIDTH=11
exec /bin/bash "${ROOT}/launch_width13_pipeline.sh" "$@"
