#!/bin/bash

# Dual GPU Parallel Federated Learning Script
# This script automatically assigns methods to GPUs for parallel execution
# 
# Usage:
#   Single GPU (sequential):  bash run_4methods_dual_gpu.sh --sequential
#   Dual GPU (automatic):     bash run_4methods_dual_gpu.sh
#   Dual GPU (custom pairs):  bash run_4methods_dual_gpu.sh --gpu0 MOON,FedALA --gpu1 StatAvg,DASHA
#   Specific methods only:    bash run_4methods_dual_gpu.sh MOON FedALA

# Default GPU assignments (for dual GPU)
GPU0_METHODS=("MOON" "FedALA")
GPU1_METHODS=("StatAvg" "DASHA")

# Parse command line arguments
SEQUENTIAL_MODE=false
CUSTOM_GPU0=""
CUSTOM_GPU1=""
METHODS_TO_RUN="$@"

while [[ $# -gt 0 ]]; do
    case $1 in
        --sequential)
            SEQUENTIAL_MODE=true
            METHODS_TO_RUN="${METHODS_TO_RUN//--sequential/}"
            shift
            ;;
        --gpu0)
            CUSTOM_GPU0="$2"
            shift 2
            ;;
        --gpu1)
            CUSTOM_GPU1="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Clean up METHODS_TO_RUN
METHODS_TO_RUN=$(echo "$METHODS_TO_RUN" | xargs)

# If custom GPU assignments provided, parse them
if [ -n "$CUSTOM_GPU0" ]; then
    IFS=',' read -ra GPU0_METHODS <<< "$CUSTOM_GPU0"
fi
if [ -n "$CUSTOM_GPU1" ]; then
    IFS=',' read -ra GPU1_METHODS <<< "$CUSTOM_GPU1"
fi

# If specific methods requested, filter GPU assignments
if [ -n "$METHODS_TO_RUN" ]; then
    FILTERED_GPU0=()
    FILTERED_GPU1=()
    
    for method in "${GPU0_METHODS[@]}"; do
        if echo "$METHODS_TO_RUN" | grep -q "\<$method\>"; then
            FILTERED_GPU0+=("$method")
        fi
    done
    
    for method in "${GPU1_METHODS[@]}"; do
        if echo "$METHODS_TO_RUN" | grep -q "\<$method\>"; then
            FILTERED_GPU1+=("$method")
        fi
    done
    
    GPU0_METHODS=("${FILTERED_GPU0[@]}")
    GPU1_METHODS=("${FILTERED_GPU1[@]}")
fi

# Validate at least one method selected
if [ ${#GPU0_METHODS[@]} -eq 0 ] && [ ${#GPU1_METHODS[@]} -eq 0 ]; then
    echo "Error: No valid methods selected. Available: MOON, FedALA, StatAvg, DASHA"
    exit 1
fi

echo "=========================================="
echo "Dual GPU Federated Learning Orchestration"
echo "=========================================="
echo "Mode: $([ "$SEQUENTIAL_MODE" = true ] && echo "Sequential (Single GPU)" || echo "Parallel (Dual GPU)")"
echo "GPU 0 Methods: ${GPU0_METHODS[*]:-None}"
echo "GPU 1 Methods: ${GPU1_METHODS[*]:-None}"
echo "=========================================="

# Experiment parameters
ALPHA_VALUES=(0.001 0.01 0.1 1.0)
TAU_VALUES=(0.01 0.1 1.0)
SLR_VALUES=(0.001 0.01 0.1)

# Calculate total experiments
TOTAL_EXPERIMENTS=0
for method in "${GPU0_METHODS[@]}" "${GPU1_METHODS[@]}"; do
    TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS + 12))
done

TOTAL_COMPLETED=0

# Function to run a method with specific GPU
run_method_on_gpu() {
    local METHOD=$1
    local GPU_ID=$2
    local ALPHA=$3
    local PARAM_VALUE=$4
    local PARAM_NAME=$5
    
    echo "[$METHOD] Starting experiment: alpha=$ALPHA, $PARAM_NAME=$PARAM_VALUE (GPU $GPU_ID)"
    
    # Export GPU_ID for use in Python scripts
    export GPU_ID=$GPU_ID
    
    case $METHOD in
        MOON)
            bash moon_server.sh "$ALPHA" "$PARAM_VALUE" > "LSTM_MOON_alpha_${ALPHA}_tau_${PARAM_VALUE}.txt" 2>&1
            ;;
        FedALA)
            bash fedala_server.sh "$ALPHA" "$PARAM_VALUE" > "LSTM_FedALA_alpha_${ALPHA}_tau_${PARAM_VALUE}.txt" 2>&1
            ;;
        StatAvg)
            bash statavg_server.sh "$ALPHA" "$PARAM_VALUE" > "LSTM_StatAvg_alpha_${ALPHA}_slr_${PARAM_VALUE}.txt" 2>&1
            ;;
        DASHA)
            bash dasha_server.sh "$ALPHA" "$PARAM_VALUE" > "LSTM_DASHA_alpha_${ALPHA}_slr_${PARAM_VALUE}.txt" 2>&1
            ;;
    esac
    
    unset GPU_ID
    
    TOTAL_COMPLETED=$((TOTAL_COMPLETED + 1))
    echo "[$METHOD] Completed ($TOTAL_COMPLETED/$TOTAL_EXPERIMENTS)"
}

# Function to run all experiments for a method
run_method_experiments() {
    local METHOD=$1
    local GPU_ID=$2
    
    case $METHOD in
        MOON|FedALA)
            PARAM_NAME="tau"
            PARAM_VALUES=("${TAU_VALUES[@]}")
            ;;
        StatAvg|DASHA)
            PARAM_NAME="slr"
            PARAM_VALUES=("${SLR_VALUES[@]}")
            ;;
    esac
    
    for ALPHA in "${ALPHA_VALUES[@]}"; do
        for PARAM_VALUE in "${PARAM_VALUES[@]}"; do
            run_method_on_gpu "$METHOD" "$GPU_ID" "$ALPHA" "$PARAM_VALUE" "$PARAM_NAME"
            sleep 2
        done
    done
}

# Sequential mode (single GPU or safe mode)
if [ "$SEQUENTIAL_MODE" = true ]; then
    echo "Running in sequential mode..."
    
    for method in "${GPU0_METHODS[@]}"; do
        run_method_experiments "$method" 0
        sleep 5
    done
    
    for method in "${GPU1_METHODS[@]}"; do
        run_method_experiments "$method" 0
        sleep 5
    done
else
    # Parallel mode (dual GPU)
    echo "Running in parallel dual GPU mode..."
    
    # Find the longer array to determine loop count
    MAX_LENGTH=${#GPU0_METHODS[@]}
    if [ ${#GPU1_METHODS[@]} -gt $MAX_LENGTH ]; then
        MAX_LENGTH=${#GPU1_METHODS[@]}
    fi
    
    # Launch methods in pairs (one per GPU)
    for ((i=0; i<MAX_LENGTH; i++)); do
        PIDS=()
        
        # Launch GPU 0 method if available
        if [ $i -lt ${#GPU0_METHODS[@]} ]; then
            METHOD=${GPU0_METHODS[$i]}
            echo "Launching $METHOD on GPU 0..."
            (run_method_experiments "$METHOD" 0) &
            PIDS+=($!)
        fi
        
        # Launch GPU 1 method if available
        if [ $i -lt ${#GPU1_METHODS[@]} ]; then
            METHOD=${GPU1_METHODS[$i]}
            echo "Launching $METHOD on GPU 1..."
            (run_method_experiments "$METHOD" 1) &
            PIDS+=($!)
        fi
        
        # Wait for both to complete
        for pid in "${PIDS[@]}"; do
            wait $pid
        done
        
        echo "Pair $((i+1)) completed. Pausing before next pair..."
        sleep 10
    done
fi

echo "=========================================="
echo "All experiments completed!"
echo "Total: $TOTAL_COMPLETED experiments"
echo "=========================================="

# Collect results
echo "Collecting results..."
if [ ${#GPU0_METHODS[@]} -gt 0 ] || [ ${#GPU1_METHODS[@]} -gt 0 ]; then
    mkdir -p results_$(date +%Y%m%d_%H%M%S)
    for method in "${GPU0_METHODS[@]}" "${GPU1_METHODS[@]}"; do
        case $method in
            MOON|FedALA)
                mv LSTM_${method}_*.txt results_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
                ;;
            StatAvg|DASHA)
                mv LSTM_${method}_*.txt results_$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
                ;;
        esac
    done
    echo "Results saved to results_$(date +%Y%m%d_%H%M%S)/"
fi

echo "Done!"
