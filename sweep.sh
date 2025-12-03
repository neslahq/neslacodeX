#!/usr/bin/env bash

set -ex

NUM_RUNS=${NUM_RUNS:-1}
CONFIG_FILE=${CONFIG_FILE:-"sweep_config/ffn_scale.yaml"}
WANDB_PROJECT=${WANDB_PROJECT:-"neslacodex"}
WANDB_ENTITY=${WANDB_ENTITY:-"nesla-lab"}

# Show raw wandb output first (for debugging)
# wandb sweep --project "${WANDB_PROJECT}" "${CONFIG_FILE}" 2>&1 | tee /tmp/wandb_sweep.log

# Now extract the ID from the logged output
SWEEP_ID=$(
  wandb sweep --project "${WANDB_PROJECT}" "${CONFIG_FILE}" 2>&1 \
  | awk '/Run sweep agent with: wandb agent/ {print $NF}'
)

echo "Sweep ID: ${SWEEP_ID}"

wandb agent ${SWEEP_ID} --count ${NUM_RUNS}