#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

TRAIN_SCRIPT="${ROOT_DIR}/train.sh"
DEFAULT_CONFIG_FILE="${ROOT_DIR}/src/codex/train_configs/debug_model.toml"
CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
EXTRA_ARGS=("$@")
WIDTHS=(256 512 1024)
CUSTOM_IMPORT="src.codex.width_override"

for width in "${WIDTHS[@]}"; do
  echo "============================================================"
  echo "Running Codex width sweep with d_model=${width}"
  echo "Config file: ${CONFIG_FILE}"
  echo "============================================================"

  CODEX_D_MODEL="${width}" \
  WANDB_RUN_NAME="codex_width_${width}" \
  CONFIG_FILE="${CONFIG_FILE}" \
  "${TRAIN_SCRIPT}" \
    --experimental.custom_import "${CUSTOM_IMPORT}" \
    "${EXTRA_ARGS[@]}"
done

