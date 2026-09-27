#!/usr/bin/env bash
# Run API models against a prepared evaluation plan.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Run this file with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail
PHYSALIGN_API_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHYSALIGN_API_PROJECT="${PHYSALIGN_ROOT:-$(cd -- "$PHYSALIGN_API_SCRIPT_DIR/.." && pwd)}"
PHYSALIGN_API_PYTHON="${PHYSALIGN_PY:-$(command -v python3 || command -v python)}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
cd -- "$PHYSALIGN_API_PROJECT"
test -x "$PHYSALIGN_API_PYTHON"
if [[ $# -eq 0 ]]; then set -- all; fi
PHYSALIGN_API_STAGE="$1"
shift
if [[ "$PHYSALIGN_API_STAGE" == 'plot' ]]; then
  exec "$PHYSALIGN_API_PYTHON" -u server_eval/compare_models.py \
    --panel "${PHYSALIGN_API_PANEL:-configs/comparison.example.json}" "$@"
fi
exec "$PHYSALIGN_API_PYTHON" -u server_eval/api_models.py "$PHYSALIGN_API_STAGE" \
  --plan "${PHYSALIGN_API_PLAN:-plans/evaluation.json}" \
  --output "${PHYSALIGN_API_OUTPUT:-runs/api-models}" "$@"
