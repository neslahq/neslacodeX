#!/usr/bin/env bash
set -euo pipefail

# EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval (default is ~10M)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

TRAIN_SCRIPT="${ROOT_DIR}/train.sh"
DEFAULT_CONFIG_FILE="${ROOT_DIR}/src/codex/train_configs/debug_model.toml"
CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"

EXTRA_ARGS=("$@")
CUSTOM_IMPORT="src.scripts.override_model_config"

LEARNING_RATES=(
    2.2e-2
    2.2e-4
    2.2e-6
)
# DEPTHS=(20)
DEPTHS=(10 14 18)
# DEPTHS=(10 12 14 16 18 20)
WANDB_RUN="${WANDB_RUN:-hp_transfer_scion}"

# export OMP_NUM_THREADS=1
# export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
# source .venv/bin/activate

RESULTS_DIR="${ROOT_DIR}/scion_hp_transfer_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "lr,depth,model_dim,num_params,val_loss" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results
run_exists() {
    local lr=$1
    local depth=$2
    grep -q "^${lr},${depth}," "$RESULTS_FILE" 2>/dev/null
}



for lr in "${LEARNING_RATES[@]}"; do
    log "=============================================="
    log "Learning rate: $lr"
    log "=============================================="

    for d in "${DEPTHS[@]}"; do

        # Skip if already completed
        if run_exists "$lr" "$d"; then
            log "Skipping d=$d at $lr learning rate (already in results)"
            continue
        fi

        log "Training d=$d at $lr learning rate..."

        # Unique tag for this run
        TAG="hp_transfer_${lr}_d${d}"

        # Record start time
        START_TIME=$(date +%s)

        # Run training, capture exit code (don't exit script on failure)
        set +e
        CODEX_DEPTH="${d}" \
        WANDB_RUN_NAME="${WANDB_RUN}_${TAG}" \
        CONFIG_FILE="${CONFIG_FILE}" \
        "${TRAIN_SCRIPT}" \
        --training.target_flops=1e16 \
        # --optimizer.name="AdamW" \
        # --optimizer.beta1=0.9 \
        # --optimizer.beta2=0.95 \
        # --optimizer.eps=1e-8 \
        # --optimizer.weight_decay=0.1 \
        # --optimizer.lr=${lr} \
        # --lr_scheduler.min_lr_factor=0.0 \
        --experimental.custom_import "${CUSTOM_IMPORT}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"
        TRAIN_EXIT_CODE=${PIPESTATUS[0]}
        set -e

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

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

        log "  Val Loss: $VAL_LOSS"

        if [ "$TRAIN_EXIT_CODE" -ne 0 ] && [ "$VAL_LOSS" = "N/A" ]; then
            log "WARNING: Training failed and no validation loss found, skipping CSV entry"
            continue
        fi

        # Append to CSV
        echo "$lr,$d,$MODEL_DIM,$NUM_SCALING_PARAMS,$VAL_LOSS" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "HP Transfer Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"