#!/usr/bin/env bash
# Hyperparameter sweep launcher for Codex model using W&B agent
# Usage: WANDB_PROJECT=codex WANDB_ENTITY=nesla-lab ./run_hp_sweep.sh

set -ex

DEPTHS=(12 16 20)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

RESULTS_DIR="${ROOT_DIR}/hp_tuning_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

NUM_RUNS=${NUM_RUNS:-30}
CONFIG_FILE=${CONFIG_FILE:-"sweep_config/optimizer_training.yaml"}
WANDB_PROJECT=${WANDB_PROJECT:-"neslacodex-scion-hp-transfer"}
WANDB_ENTITY=${WANDB_ENTITY:-"tinuade"}


# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,model_dim,num_params,lr,val_loss" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results
run_exists() {
    local depth=$1
    grep -q "^${depth}," "$RESULTS_FILE" 2>/dev/null
}

for d in "${DEPTHS[@]}"; do
    if run_exists "$d"; then
        log "Skipping d=$d (already in results)"
        continue
    fi

    log "Training d=$d..."

    echo "============================================================"
    echo "Starting W&B Hyperparameter Sweep"
    echo "Depth: ${d}"
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

    if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
            log "WARNING: Training exited with code $TRAIN_EXIT_CODE, attempting to extract partial results..."
        fi

        # Extract training stats from the log (with fallbacks for missing values)
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
        # Use sed to extract the number after the specific label
        NUM_SCALING_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*Number of parameters: //' | grep -oP '^\d+' || echo "0")
        # # Try to get scaling params, fall back to total params if not available
        # NUM_SCALING_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP 'scaling \d+' | grep -oP '\d+' || echo "")
        # if [ -z "$NUM_SCALING_PARAMS" ]; then
        #     NUM_SCALING_PARAMS="$NUM_PARAMS"
        # fi
        # NUM_ITERS=$(grep "Calculated number of iterations:" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*Calculated number of iterations: //' | grep -oP '^\d+' || echo "0")
        # Calculate tokens trained
        # TOKENS_TRAINED=$(grep "Number of training tokens:" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*Number of training tokens: //' | grep -oP '^\d+' || echo "0")
        # Param:data ratio (using scaling params per Kaplan et al.)
        # if [ "$NUM_SCALING_PARAMS" != "0" ] && [ "$TOKENS_TRAINED" != "0" ]; then
        #     PARAM_DATA_RATIO=$(python3 -c "print(f'{$TOKENS_TRAINED / $NUM_SCALING_PARAMS:.2f}')")
        # else
        #     PARAM_DATA_RATIO="N/A"
        # fi
        # Model dim
        MODEL_DIM=$((d * 64))
        # Val BPB from final eval
        VAL_LOSS=$(grep "Final validation loss:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")
        LR=$(grep "optimizer.lr" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")
        # Extract CORE score from training log (evaluated on final step)
        # CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        # if [ -z "$CORE_SCORE" ]; then
        #     log "WARNING: Could not extract CORE score for d=$d"
        #     CORE_SCORE="0.0"
        # fi

        # Handle empty values
        # NUM_PARAMS="${NUM_PARAMS:-0}"
        NUM_SCALING_PARAMS="${NUM_SCALING_PARAMS:-0}"
        # NUM_ITERS="${NUM_ITERS:-0}"
        # TOKENS_TRAINED="${TOKENS_TRAINED:-0}"
        VAL_LOSS="${VAL_LOSS:-N/A}"
        LR="${LR:-N/A}"
        log "  Scaling Params: $NUM_SCALING_PARAMS, LR: $LR, Val Loss: $VAL_LOSS"

        if [ "$TRAIN_EXIT_CODE" -ne 0 ] && [ "$VAL_LOSS" = "N/A" ]; then
            log "WARNING: Training failed and no validation loss found, skipping CSV entry"
            continue
        fi

        # Append to CSV
        echo "$d,$MODEL_DIM,$NUM_SCALING_PARAMS,$LR,$VAL_LOSS" >> "$RESULTS_FILE"

done
