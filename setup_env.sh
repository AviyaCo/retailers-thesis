#!/bin/bash
# Recreate the Retailers Python environment (see REPRODUCE.md section 2).
# Usage: bash setup_env.sh [ENV_DIR]     default ENV_DIR=/data/home/aviyac/ap_env
# Needs: python3.11, internet access to PyPI. The Gurobi license is NOT installed here: put it at a path of your
# choice and export GRB_LICENSE_FILE=<path> (never commit it).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ENV_DIR=${1:-/data/home/aviyac/ap_env}
BUNDLE="$HERE/archive/hydro_rcr_rescaled/.moreau_diagnostic_deps"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    python3.11 -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$HERE/requirements.txt"

# Moreau is loaded from this exact folder (src/config.py MOREAU_DEPS, checked in src/policy.py).
if [ ! -f "$BUNDLE/moreau/__init__.py" ]; then
    "$ENV_DIR/bin/python" -m pip install --no-deps --target "$BUNDLE" \
        moreau==0.4.0 moreau-cpu==0.4.0 pydantic==2.13.5 pydantic-core==2.46.5 \
        annotated-types==0.8.0 typing-extensions==4.16.0 typing-inspection==0.4.4
fi

cd "$HERE"
OMP_NUM_THREADS=1 "$ENV_DIR/bin/python" -B -m unittest discover -s tests
echo "Environment ready: $ENV_DIR. Remember: export GRB_LICENSE_FILE=<your gurobi.lic>"
