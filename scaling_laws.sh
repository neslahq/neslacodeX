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
    1e18
    3e18
    6e18
)
DEPTHS=(10 12 14 16 18 20)
WANDB_RUN="${WANDB_RUN:-scaling}"

# export OMP_NUM_THREADS=1
# export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
# source .venv/bin/activate

RESULTS_DIR="${ROOT_DIR}/scaling_laws_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,num_params,num_scaling_params,num_iterations,tokens_trained,param_data_ratio,val_loss,train_time_sec" > "$RESULTS_FILE"
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

        # Train the model with fixed flops budget
        # The script will auto-calculate num_iterations to hit target_flops
        # CORE eval happens once at the end (999999 ensures only final step)
        # torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        #     --depth=$d \
        #     --target_flops=$flops \
        #     --target_param_data_ratio=-1 \
        #     --run="${WANDB_RUN}_${TAG}" \
        #     --model_tag="${TAG}" \
        #     --eval_tokens=$EVAL_TOKENS \
        #     --core_metric_every=999999 \
        #     --core_metric_max_per_task=-1 \
        #     --sample_every=-1 \
        #     --save_every=-1 \
        #     2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

        CODEX_DEPTH="${d}" \
        WANDB_RUN_NAME="${WANDB_RUN}_${TAG}" \
        CONFIG_FILE="${CONFIG_FILE}" \
        "${TRAIN_SCRIPT}" \
        --target_flops=$flops \
        --experimental.custom_import "${CUSTOM_IMPORT}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"


        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # Extract training stats from the log
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
        NUM_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | head -1 | tr -d ',')
        NUM_SCALING_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" | tail -1 | grep -oP 'scaling: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
        NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
        # Calculate tokens trained (iterations * batch_size, default 524288)
        TOKENS_TRAINED=$(grep "Number of training tokens:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | head -1 | tr -d ',')
        # Param:data ratio (using scaling params per Kaplan et al.)
        PARAM_DATA_RATIO=$(python -c "print(f'{$TOKENS_TRAINED / $NUM_SCALING_PARAMS:.2f}')")
        # Model dim
        MODEL_DIM=$((d * 64))
        # Val BPB from final eval
        VAL_LOSS=$(grep "Final validation loss:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')

        # Extract CORE score from training log (evaluated on final step)
        # CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        # if [ -z "$CORE_SCORE" ]; then
        #     log "WARNING: Could not extract CORE score for d=$d"
        #     CORE_SCORE="0.0"
        # fi

        log "  Params: $NUM_PARAMS, Iters: $NUM_ITERS, Ratio: $PARAM_DATA_RATIO, Val Loss: $VAL_LOSS"

        # Append to CSV
        echo "$flops,$d,$MODEL_DIM,$NUM_PARAMS,$NUM_SCALING_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$PARAM_DATA_RATIO,$VAL_LOSS,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "Scaling Laws Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"