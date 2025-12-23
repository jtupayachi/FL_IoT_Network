#!/bin/bash

# Unified Federated Learning Experiment Runner
# Automatically detects GPU count and optimizes execution
#
# Usage:
#   Auto-detect GPUs:           sudo ./run_experiments.sh
#   Force single GPU:           sudo ./run_experiments.sh --single-gpu
#   Force dual GPU:             sudo ./run_experiments.sh --dual-gpu
#   Specific methods (1 GPU):   sudo ./run_experiments.sh --single-gpu MOON
#   Specific methods (2 GPUs):  sudo ./run_experiments.sh --dual-gpu MOON FedALA
#   With model type:            sudo ./run_experiments.sh --model LSTM MOON FedALA
#   Run both models:            sudo ./run_experiments.sh --model BOTH --dual-gpu

set -e

# Default parameters
MODEL_TYPE="LSTM"  # LSTM, MLP, or BOTH
GPU_MODE="auto"    # auto, single, or dual
METHODS_TO_RUN=""
LOG_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --single-gpu)
            GPU_MODE="single"
            shift
            ;;
        --dual-gpu)
            GPU_MODE="dual"
            shift
            ;;
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        MOON|FedALA|ALA|StatAvg|DASHA)
            # Normalize ALA to FedALA
            if [ "$1" == "ALA" ]; then
                METHODS_TO_RUN="$METHODS_TO_RUN FedALA"
            else
                METHODS_TO_RUN="$METHODS_TO_RUN $1"
            fi
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--single-gpu|--dual-gpu] [--model LSTM|MLP|BOTH] [--log filename] [MOON] [FedALA|ALA] [StatAvg] [DASHA]"
            exit 1
            ;;
    esac
done

# Detect available GPUs if auto mode
if [ "$GPU_MODE" == "auto" ]; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ $GPU_COUNT -ge 2 ]; then
        GPU_MODE="dual"
        echo "Auto-detected: $GPU_COUNT GPUs available, using dual GPU mode"
    else
        GPU_MODE="single"
        echo "Auto-detected: $GPU_COUNT GPU available, using single GPU mode"
    fi
fi

# Set default log file if not specified
if [ -z "$LOG_FILE" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="experiments_${MODEL_TYPE}_${GPU_MODE}gpu_${TIMESTAMP}.log"
fi

echo "=========================================="
echo "Federated Learning Experiment Runner"
echo "=========================================="
echo "Model Type: $MODEL_TYPE"
echo "GPU Mode: $GPU_MODE"
echo "Methods: ${METHODS_TO_RUN:-All (MOON, FedALA, StatAvg, DASHA)}"
echo "Log File: $LOG_FILE"
echo "=========================================="

# Experiment parameters
ALPHA_VALUES=(0.001 0.01 0.1 1.0)
TAU_VALUES=(0.01 0.1 1.0)
SLR_VALUES=(0.001 0.01 0.1)

# Determine which methods to run
if [ -z "$METHODS_TO_RUN" ]; then
    ALL_METHODS=("MOON" "FedALA" "StatAvg" "DASHA")
else
    IFS=' ' read -ra ALL_METHODS <<< "$METHODS_TO_RUN"
fi

# Function to run experiment with GPU assignment
run_experiment() {
    local METHOD=$1
    local GPU_ID=$2
    local ALPHA=$3
    local PARAM_VALUE=$4
    local MODEL=$5
    
    # Determine parameter name and script based on method
    case $METHOD in
        MOON|FedALA)
            PARAM_NAME="tau"
            ;;
        StatAvg|DASHA)
            PARAM_NAME="slr"
            ;;
    esac
    
    local OUTPUT_FILE="${MODEL}_${METHOD}_alpha_${ALPHA}_${PARAM_NAME}_${PARAM_VALUE}.txt"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID: $MODEL $METHOD alpha=$ALPHA ${PARAM_NAME}=$PARAM_VALUE"
    
    # Export GPU_ID for Python scripts
    export GPU_ID=$GPU_ID
    
    # Run the appropriate server script based on method
    case $METHOD in
        MOON)
            bash moon_server.sh "$ALPHA" "$PARAM_VALUE" > "$OUTPUT_FILE" 2>&1
            ;;
        FedALA)
            bash fedala_server.sh "$ALPHA" "$PARAM_VALUE" > "$OUTPUT_FILE" 2>&1
            ;;
        StatAvg)
            bash statavg_server.sh "$ALPHA" "$PARAM_VALUE" > "$OUTPUT_FILE" 2>&1
            ;;
        DASHA)
            bash dasha_server.sh "$ALPHA" "$PARAM_VALUE" > "$OUTPUT_FILE" 2>&1
            ;;
    esac
    
    unset GPU_ID
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID: $MODEL $METHOD alpha=$ALPHA ${PARAM_NAME}=$PARAM_VALUE - COMPLETED"
}

# Function to run all experiments for a method
run_method_all_experiments() {
    local METHOD=$1
    local GPU_ID=$2
    local MODEL=$3
    
    case $METHOD in
        MOON|FedALA)
            PARAM_VALUES=("${TAU_VALUES[@]}")
            ;;
        StatAvg|DASHA)
            PARAM_VALUES=("${SLR_VALUES[@]}")
            ;;
    esac
    
    local TOTAL=$((${#ALPHA_VALUES[@]} * ${#PARAM_VALUES[@]}))
    local COUNT=0
    
    echo "Starting $METHOD on GPU $GPU_ID (${TOTAL} experiments)..."
    
    for ALPHA in "${ALPHA_VALUES[@]}"; do
        for PARAM_VALUE in "${PARAM_VALUES[@]}"; do
            COUNT=$((COUNT + 1))
            echo "[$METHOD GPU$GPU_ID] Progress: $COUNT/$TOTAL"
            run_experiment "$METHOD" "$GPU_ID" "$ALPHA" "$PARAM_VALUE" "$MODEL"
            sleep 2
        done
    done
    
    echo "$METHOD on GPU $GPU_ID completed all ${TOTAL} experiments"
}

# Function to run experiments based on GPU mode
run_all_experiments() {
    local MODEL=$1
    
    echo ""
    echo "=========================================="
    echo "Starting $MODEL experiments"
    echo "=========================================="
    
    if [ "$GPU_MODE" == "single" ]; then
        # Single GPU: Sequential execution
        echo "Running in SINGLE GPU mode (sequential)..."
        for METHOD in "${ALL_METHODS[@]}"; do
            run_method_all_experiments "$METHOD" 0 "$MODEL"
            echo "Pausing 5 seconds between methods..."
            sleep 5
        done
        
    elif [ "$GPU_MODE" == "dual" ]; then
        # Dual GPU: Parallel execution with automatic load balancing
        echo "Running in DUAL GPU mode (parallel)..."
        
        # For single method: use GPU 0
        if [ ${#ALL_METHODS[@]} -eq 1 ]; then
            echo "Single method specified, using GPU 0"
            run_method_all_experiments "${ALL_METHODS[0]}" 0 "$MODEL"
            
        # For two methods: one per GPU
        elif [ ${#ALL_METHODS[@]} -eq 2 ]; then
            echo "Two methods specified, running in parallel:"
            echo "  GPU 0: ${ALL_METHODS[0]}"
            echo "  GPU 1: ${ALL_METHODS[1]}"
            
            (run_method_all_experiments "${ALL_METHODS[0]}" 0 "$MODEL") &
            PID1=$!
            (run_method_all_experiments "${ALL_METHODS[1]}" 1 "$MODEL") &
            PID2=$!
            
            wait $PID1
            wait $PID2
            
        # For 3-4 methods: pair them up
        else
            echo "Multiple methods specified, pairing for dual GPU:"
            
            # Default pairing: MOON+FedALA on GPU0, StatAvg+DASHA on GPU1
            GPU0_METHODS=()
            GPU1_METHODS=()
            
            for METHOD in "${ALL_METHODS[@]}"; do
                if [[ "$METHOD" == "MOON" ]] || [[ "$METHOD" == "FedALA" ]]; then
                    GPU0_METHODS+=("$METHOD")
                else
                    GPU1_METHODS+=("$METHOD")
                fi
            done
            
            echo "  GPU 0: ${GPU0_METHODS[*]}"
            echo "  GPU 1: ${GPU1_METHODS[*]}"
            
            # Run GPU0 methods sequentially in background
            (
                for METHOD in "${GPU0_METHODS[@]}"; do
                    run_method_all_experiments "$METHOD" 0 "$MODEL"
                    sleep 3
                done
            ) &
            PID1=$!
            
            # Run GPU1 methods sequentially in background
            (
                for METHOD in "${GPU1_METHODS[@]}"; do
                    run_method_all_experiments "$METHOD" 1 "$MODEL"
                    sleep 3
                done
            ) &
            PID2=$!
            
            wait $PID1
            wait $PID2
        fi
    fi
    
    echo ""
    echo "=========================================="
    echo "$MODEL experiments completed!"
    echo "=========================================="
}

# Main execution
{
    echo "Experiment started at: $(date)"
    echo ""
    
    if [ "$MODEL_TYPE" == "BOTH" ]; then
        run_all_experiments "LSTM"
        echo ""
        echo "Pausing 10 seconds between LSTM and MLP..."
        sleep 10
        run_all_experiments "MLP"
    else
        run_all_experiments "$MODEL_TYPE"
    fi
    
    echo ""
    echo "=========================================="
    echo "ALL EXPERIMENTS COMPLETED!"
    echo "=========================================="
    echo "Completed at: $(date)"
    
    # Collect results
    RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"
    
    echo "Collecting results to $RESULTS_DIR..."
    mv *_MOON_*.txt "$RESULTS_DIR/" 2>/dev/null || true
    mv *_FedALA_*.txt "$RESULTS_DIR/" 2>/dev/null || true
    mv *_StatAvg_*.txt "$RESULTS_DIR/" 2>/dev/null || true
    mv *_DASHA_*.txt "$RESULTS_DIR/" 2>/dev/null || true
    
    echo "Results saved to: $RESULTS_DIR"
    echo "Log saved to: $LOG_FILE"
    
} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Experiment complete! Check $LOG_FILE for details."
