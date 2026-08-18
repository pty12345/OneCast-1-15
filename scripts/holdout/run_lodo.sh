#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_predictor_holdout.py" "$@" --phase zs_train
"${PYTHON_BIN}" "${SCRIPT_DIR}/run_predictor_holdout.py" "$@" --phase zs_test
