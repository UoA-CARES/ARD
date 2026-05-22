#!/usr/bin/env bash
# Smoke-test the ARD refinement loop on the Cartpole task via the --task flag.
#
# Assumes:
#   - PCS coordinator up with workers registered and the training image present
#   - TOKEN exported (e.g. from ~/.bashrc)
#   - OPENROUTER_API_KEY exported
#   - configs/settings.yaml and configs/refineconfig.yaml already set correctly
#
# Usage:
#   scripts/test_cartpole.sh                 # uses settings/refineconfig defaults
#   scripts/test_cartpole.sh --settings configs/settings.yaml   # forward extra flags
set -euo pipefail
cd "$(dirname "$0")/.."

: "${TOKEN:?TOKEN is not set — export it (or source ~/.bashrc) first}"
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is not set — export it first}"

echo "=== ARD refinement: Cartpole ==="
python main.py --refine --task cartpole "$@"
echo "=== Done: Cartpole ==="
