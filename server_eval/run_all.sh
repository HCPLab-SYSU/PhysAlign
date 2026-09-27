#!/usr/bin/env bash
# Execute with `bash server_eval/run_all.sh all`; do not source this file.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Run this file with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail

PHYSALIGN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PHYSALIGN_ROOT="${PHYSALIGN_ROOT:-$(cd -- "$PHYSALIGN_SCRIPT_DIR/.." && pwd)}"
export PHYSALIGN_PY="${PHYSALIGN_PY:-$(command -v python3 || command -v python)}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLBACKEND=Agg

cd -- "$PHYSALIGN_ROOT"
test -x "$PHYSALIGN_PY"
test -f evaluate.py
if [[ $# -eq 0 ]]; then set -- all; fi
exec "$PHYSALIGN_PY" -u "$PHYSALIGN_SCRIPT_DIR/two_experiments.py" \
  --project "$PHYSALIGN_ROOT" \
  --dataset "${DATASET_ROOT:-$PHYSALIGN_ROOT/datasets/physalign-final-v1}" \
  --tag "${PHYSALIGN_RUN_TAG:-all-raw-gold-v1}" "$@"
