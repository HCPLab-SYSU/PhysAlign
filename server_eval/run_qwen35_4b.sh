#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Run this file with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail
PHYSALIGN_QWEN4_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHYSALIGN_QWEN4_PROJECT="${PHYSALIGN_ROOT:-$(cd -- "$PHYSALIGN_QWEN4_SCRIPT_DIR/.." && pwd)}"
PHYSALIGN_QWEN4_PYTHON="${PHYSALIGN_PY:-$(command -v python3 || command -v python)}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd -- "$PHYSALIGN_QWEN4_PROJECT"
test -x "$PHYSALIGN_QWEN4_PYTHON"
if [[ $# -eq 0 ]]; then set -- all; fi
PHYSALIGN_QWEN4_STAGE="$1"
shift
if [[ "$PHYSALIGN_QWEN4_STAGE" == 'plot' ]]; then
  exec "$PHYSALIGN_QWEN4_PYTHON" -u server_eval/compare_models.py \
    --panel "${PHYSALIGN_QWEN4_PANEL:-configs/comparison.example.json}" "$@"
fi
if [[ "$PHYSALIGN_QWEN4_STAGE" == 'freeze' ]]; then
  exec "$PHYSALIGN_QWEN4_PYTHON" -u server_eval/qwen35_4b.py freeze "$@"
fi
exec "$PHYSALIGN_QWEN4_PYTHON" -u server_eval/qwen35_4b.py "$PHYSALIGN_QWEN4_STAGE" \
  --plan "${PHYSALIGN_QWEN4_PLAN:-plans/evaluation.json}" \
  --output "${PHYSALIGN_QWEN4_OUTPUT:-runs/local-models/qwen35-4b-nonthinking}" \
  --preflight-output "${PHYSALIGN_QWEN4_PREFLIGHT:-plans/qwen35-4b-nonthinking-v1.preflight.json}" \
  --gpus "${PHYSALIGN_QWEN4_GPUS:-${CUDA_VISIBLE_DEVICES:-}}" "$@"
