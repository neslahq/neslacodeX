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

FLOPS_BUDGETS=(
    6e15
    8e15
    10e15
    1e16
    # 6e18
)
DEPTHS=(10 12 14 16)
# DEPTHS=(10 12 14 16 18 20)
WANDB_RUN="${WANDB_RUN:-scaling}"

export OMP_NUM_THREADS=1

RESULTS_DIR="${ROOT_DIR}/scaling_laws_results_v2"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,num_scaling_params,num_iterations,tokens_trained,param_data_ratio,val_loss,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results
run_exists() {
    local flops=$1
    local depth=$2
    grep -q "^${flops},${depth}," "$RESULTS_FILE" 2>/dev/null
}

# =============================================================================
# Main Loop
# =============================================================================
# for width in "${WIDTHS[@]}"; do
#   echo "============================================================"
#   echo "Running Codex width sweep with d_model=${width}"
#   echo "Config file: ${CONFIG_FILE}"
#   echo "============================================================"

  
# done

for flops in "${FLOPS_BUDGETS[@]}"; do
    log "=============================================="
    log "Compute budget: $flops FLOPs"
    log "=============================================="

    for d in "${DEPTHS[@]}"; do

        # Skip if already completed
        if run_exists "$flops" "$d"; then
            log "Skipping d=$d at $flops FLOPs (already in results)"
            continue
        fi

        log "Training d=$d at $flops FLOPs..."

        # Unique tag for this run
        TAG="scaling_${flops}_d${d}"

        # Record start time
        START_TIME=$(date +%s)

        # Run training, capture exit code (don't exit script on failure)
        set +e
        CODEX_DEPTH="${d}" \
        WANDB_RUN_NAME="${WANDB_RUN}_${TAG}" \
        CONFIG_FILE="${CONFIG_FILE}" \
        "${TRAIN_SCRIPT}" \
        --training.target_flops=$flops \
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
        NUM_ITERS=$(grep "Calculated number of iterations:" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*Calculated number of iterations: //' | grep -oP '^\d+' || echo "0")
        # Calculate tokens trained
        TOKENS_TRAINED=$(grep "Number of training tokens:" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*Number of training tokens: //' | grep -oP '^\d+' || echo "0")
        # Param:data ratio (using scaling params per Kaplan et al.)
        if [ "$NUM_SCALING_PARAMS" != "0" ] && [ "$TOKENS_TRAINED" != "0" ]; then
            PARAM_DATA_RATIO=$(python3 -c "print(f'{$TOKENS_TRAINED / $NUM_SCALING_PARAMS:.2f}')")
        else
            PARAM_DATA_RATIO="N/A"
        fi
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
        NUM_ITERS="${NUM_ITERS:-0}"
        TOKENS_TRAINED="${TOKENS_TRAINED:-0}"
        VAL_LOSS="${VAL_LOSS:-N/A}"

        log "  Scaling Params: $NUM_SCALING_PARAMS, Iters: $NUM_ITERS, Ratio: $PARAM_DATA_RATIO, Val Loss: $VAL_LOSS"

        if [ "$TRAIN_EXIT_CODE" -ne 0 ] && [ "$VAL_LOSS" = "N/A" ]; then
            log "WARNING: Training failed and no validation loss found, skipping CSV entry"
            continue
        fi

        # Append to CSV
        echo "$flops,$d,$MODEL_DIM,$NUM_SCALING_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$PARAM_DATA_RATIO,$VAL_LOSS,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "Scaling Laws Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"