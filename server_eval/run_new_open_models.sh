#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  printf '%s\n' 'Run this file with bash; do not source it.' >&2
  return 2
fi
set -euo pipefail

PHYSALIGN_NEW_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHYSALIGN_NEW_PROJECT="${PHYSALIGN_ROOT:-$(cd -- "$PHYSALIGN_NEW_SCRIPT_DIR/.." && pwd)}"
PHYSALIGN_NEW_PYTHON="${PHYSALIGN_PY:-$(command -v python3 || command -v python)}"
PHYSALIGN_NEW_PLAN="${PHYSALIGN_NEW_PLAN:-plans/evaluation.json}"
PHYSALIGN_NEW_LOG_DIR="${PHYSALIGN_NEW_LOG_DIR:-logs/new-open-models-v1}"

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd -- "$PHYSALIGN_NEW_PROJECT"
test -x "$PHYSALIGN_NEW_PYTHON"

usage() {
  printf '%s\n' \
    'Usage:' \
    '  bash server_eval/run_new_open_models.sh MODEL [freeze|preflight|run|score|all|status] [options]' \
    '  bash server_eval/run_new_open_models.sh panel [preflight|run|score|all|status]' \
    '' \
    'MODEL: gemma4-26b-a4b | glm46v-flash | molmo2-8b | kimi-vl-a3b'
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

common_args=(--plan "$PHYSALIGN_NEW_PLAN")
if [[ -n "${PHYSALIGN_NEW_DATASET:-}" ]]; then
  common_args+=(--dataset "$PHYSALIGN_NEW_DATASET")
fi

gpu_group() {
  case "$1" in
    gemma4-26b-a4b) printf '%s' "${PHYSALIGN_GEMMA4_GPUS:-0,1,2,3}" ;;
    glm46v-flash) printf '%s' "${PHYSALIGN_GLM46V_GPUS:-4,5}" ;;
    molmo2-8b) printf '%s' "${PHYSALIGN_MOLMO2_GPUS:-6,7}" ;;
    kimi-vl-a3b) printf '%s' "${PHYSALIGN_KIMIVL_GPUS:-4,5}" ;;
    *) printf 'Unknown model key: %s\n' "$1" >&2; return 2 ;;
  esac
}

run_one() {
  local model="$1"
  local stage="$2"
  shift 2
  if [[ "$stage" == 'freeze' ]]; then
    "$PHYSALIGN_NEW_PYTHON" -u server_eval/new_open_models.py "$model" freeze "$@"
  elif [[ "$stage" == 'run' || "$stage" == 'all' ]]; then
    "$PHYSALIGN_NEW_PYTHON" -u server_eval/new_open_models.py "$model" "$stage" \
      "${common_args[@]}" --gpus "$(gpu_group "$model")" "$@"
  else
    "$PHYSALIGN_NEW_PYTHON" -u server_eval/new_open_models.py "$model" "$stage" \
      "${common_args[@]}" "$@"
  fi
}

target="$1"
shift
stage="${1:-all}"
if [[ $# -gt 0 ]]; then shift; fi

if [[ "$target" != 'panel' ]]; then
  case "$target" in
    gemma4-26b-a4b|glm46v-flash|molmo2-8b|kimi-vl-a3b) ;;
    *) usage >&2; exit 2 ;;
  esac
  run_one "$target" "$stage" "$@"
  exit $?
fi

case "$stage" in
  preflight|score|status)
    for model in gemma4-26b-a4b glm46v-flash molmo2-8b kimi-vl-a3b; do
      run_one "$model" "$stage" "$@"
    done
    ;;
  run|all)
    if [[ "$stage" == 'all' ]]; then
      for model in gemma4-26b-a4b glm46v-flash molmo2-8b kimi-vl-a3b; do
        run_one "$model" preflight "$@"
      done
    fi
    mkdir -p -- "$PHYSALIGN_NEW_LOG_DIR"
    run_one gemma4-26b-a4b "$stage" "$@" \
      >"$PHYSALIGN_NEW_LOG_DIR/gemma4-26b-a4b.log" 2>&1 &
    gemma_pid=$!
    (
      run_one glm46v-flash "$stage" "$@"
      run_one kimi-vl-a3b "$stage" "$@"
    ) >"$PHYSALIGN_NEW_LOG_DIR/glm46v-then-kimi.log" 2>&1 &
    pair_pid=$!
    run_one molmo2-8b "$stage" "$@" \
      >"$PHYSALIGN_NEW_LOG_DIR/molmo2-8b.log" 2>&1 &
    molmo_pid=$!
    printf 'Started Gemma PID=%s, GLM->Kimi PID=%s, Molmo2 PID=%s\n' \
      "$gemma_pid" "$pair_pid" "$molmo_pid"
    panel_status=0
    if ! wait "$gemma_pid"; then panel_status=1; fi
    if ! wait "$pair_pid"; then panel_status=1; fi
    if ! wait "$molmo_pid"; then panel_status=1; fi
    exit "$panel_status"
    ;;
  *) usage >&2; exit 2 ;;
esac
