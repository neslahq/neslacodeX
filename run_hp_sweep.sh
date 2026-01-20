#!/usr/bin/env bash
# Hyperparameter sweep launcher for Codex model - LR grid search per FLOP budget
# Usage: ./run_hp_sweep.sh

set -ex

# FLOP budgets and corresponding depths (1:1 mapping)
FLOPS_BUDGETS=(1e16 3e16 6e16)
DEPTHS=(12 16 20)

# Learning rates: 2^-5.5, 2^-5, 2^-4.5, 2^-4, 2^-3.5, 2^-3  
LR_VALUES=(0.022097087 0.03125 0.044194174 0.0625 0.088388348 0.125)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

TRAIN_SCRIPT="${ROOT_DIR}/train.sh"
DEFAULT_CONFIG_FILE="${ROOT_DIR}/src/codex/train_configs/debug_model.toml"
BASE_CONFIG_FILE="${CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
CUSTOM_IMPORT="src.scripts.override_model_config"

RESULTS_DIR="${ROOT_DIR}/hp_tuning_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

WANDB_PROJECT=${WANDB_PROJECT:-"neslacodex-scion-hp-scaling"}
WANDB_ENTITY=${WANDB_ENTITY:-"tinuade"}
WANDB_RUN="${WANDB_RUN:-hp_sweep}"

EXTRA_ARGS=("$@")

# Write CSV header only if file doesn't exist
if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,num_params,lr,val_loss" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results
run_exists() {
    local flops=$1
    local depth=$2
    local lr=$3
    grep -q "^${flops},${depth},[^,]*,[^,]*,${lr}," "$RESULTS_FILE" 2>/dev/null
}

# Create a temporary config file with updated LR values (including extra_param_group_split_rules)
create_lr_config() {
    local lr=$1
    local temp_config="$RESULTS_DIR/temp_config_lr${lr}.toml"
    
    # Use Python to update LR in both optimizer.lr and extra_param_group_split_rules
    python3 << EOF
import re

with open("${BASE_CONFIG_FILE}", "r") as f:
    content = f.read()

# Update optimizer.lr
content = re.sub(r'^(lr\s*=\s*)[\d.eE+-]+', r'\g<1>${lr}', content, flags=re.MULTILINE)

with open("${temp_config}", "w") as f:
    f.write(content)
EOF

    echo "$temp_config"
}

# Main loop: iterate over flops budgets
for i in "${!FLOPS_BUDGETS[@]}"; do
    flops="${FLOPS_BUDGETS[$i]}"
    d="${DEPTHS[$i]}"
    MODEL_DIM=$((d * 64))

    log "=============================================="
    log "Compute budget: $flops FLOPs, Depth: $d, Model dim: $MODEL_DIM"
    log "=============================================="

    # Iterate over learning rates
    for lr in "${LR_VALUES[@]}"; do
        # Skip if already completed
        if run_exists "$flops" "$d" "$lr"; then
            log "Skipping flops=$flops, d=$d, lr=$lr (already in results)"
            continue
        fi

        log "Training flops=$flops, d=$d, lr=$lr..."

        # Create temporary config with updated LR values
        TEMP_CONFIG=$(create_lr_config "$lr")
        log "Using config: $TEMP_CONFIG"

        # Unique tag for this run
        TAG="hp_${flops}_d${d}_lr${lr}"
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

        # Run training, capture exit code (don't exit script on failure)
        set +e
        CODEX_DEPTH="${d}" \
        WANDB_RUN_NAME="${WANDB_RUN}_${TAG}" \
        CONFIG_FILE="${TEMP_CONFIG}" \
        "${TRAIN_SCRIPT}" \
            --training.target_flops=$flops \
            --experimental.custom_import "${CUSTOM_IMPORT}" \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "$LOG_FILE"
        TRAIN_EXIT_CODE=${PIPESTATUS[0]}
        set -e

        if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
            log "WARNING: Training exited with code $TRAIN_EXIT_CODE, attempting to extract partial results..."
        fi

        # Extract training stats from the log
        NUM_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" 2>/dev/null | tail -1 | sed 's/.*Number of parameters: //' | grep -oP '^\d+' || echo "0")
        VAL_LOSS=$(grep "Final validation loss:" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "N/A")

        # Handle empty values
        NUM_PARAMS="${NUM_PARAMS:-0}"
        VAL_LOSS="${VAL_LOSS:-N/A}"

        log "  Params: $NUM_PARAMS, LR: $lr, Val Loss: $VAL_LOSS"

        if [ "$TRAIN_EXIT_CODE" -ne 0 ] && [ "$VAL_LOSS" = "N/A" ]; then
            log "WARNING: Training failed and no validation loss found, skipping CSV entry"
            continue
        fi

        # Append to CSV
        echo "$flops,$d,$MODEL_DIM,$NUM_PARAMS,$lr,$VAL_LOSS" >> "$RESULTS_FILE"
    done
done

# Cleanup temporary config files
rm -f "$RESULTS_DIR"/temp_config_lr*.toml

log "=============================================="
log "HP Sweep Complete"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
echo ""
echo "Results:"
column -t -s',' "$RESULTS_FILE"
