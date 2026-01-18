#!/usr/bin/env bash
# Hyperparameter sweep launcher for Codex model using W&B agent
# Usage: WANDB_PROJECT=codex WANDB_ENTITY=nesla-lab ./run_hp_sweep.sh

set -ex

NUM_RUNS=${NUM_RUNS:-50}
CONFIG_FILE=${CONFIG_FILE:-"sweep_config/optimizer_training.yaml"}
WANDB_PROJECT=${WANDB_PROJECT:-"neslacodex-hp-tuning"}
WANDB_ENTITY=${WANDB_ENTITY:-"tinuade"}

echo "============================================================"
echo "Starting W&B Hyperparameter Sweep"
echo "Config: ${CONFIG_FILE}"
echo "Project: ${WANDB_PROJECT}"
echo "Entity: ${WANDB_ENTITY}"
echo "Max runs: ${NUM_RUNS}"
echo "============================================================"

# Create the sweep and extract the sweep ID
SWEEP_ID=$(
  wandb sweep --project "${WANDB_PROJECT}" --entity "${WANDB_ENTITY}" "${CONFIG_FILE}" 2>&1 \
  | awk '/Run sweep agent with: wandb agent/ {print $NF}'
)

if [ -z "${SWEEP_ID}" ]; then
  echo "ERROR: Failed to create sweep. Check your W&B credentials and config."
  exit 1
fi

echo "Created sweep: ${SWEEP_ID}"
echo "============================================================"
echo "Starting W&B agent with ${NUM_RUNS} runs..."
echo "============================================================"

# Launch the agent
wandb agent "${SWEEP_ID}" --count "${NUM_RUNS}"

