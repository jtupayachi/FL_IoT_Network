#!/bin/bash

set -e  # Exit on any error

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="$BASE_DIR/results"
SERVER_SCRIPT="$BASE_DIR/server.py"
CLIENT_SCRIPT="$BASE_DIR/client.py"
CONFIG_TEMPLATE="$BASE_DIR/config_template.json"
SERVER_HOST="localhost"
BASE_PORT=8686

# Timestamp for all experiments in this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="run_${TIMESTAMP}"

# Experiment parameters - UPDATED FOR MULTIPLE STRATEGIES
declare -a STRATEGIES=("fedavg" "fedavgm" "fedopt" "fedprox" )  # Flower built-ins
declare -A MODEL_PARAMS=(
    ["fedavg"]=""
    ["fedavgm"]="server_momentum:0.5,0.9"
    ["fedopt"]="tau:0.001,0.01"
    ["qfedavg"]="q_param:0.1,0.5,1.0;qffl_learning_rate:0.01,0.1"
)
declare -a CLIENTS=(25)  # Keep constant
declare -a ALPHAS=(0.5)  # Vary this single parameter (0.1=very non-IID, 1.0=more IID)


# Default values
DEFAULT_ROUNDS=10
DEFAULT_MIN_CLIENTS=2

# Parse command line arguments
PARALLEL=false
CLEAN=false
QUICK_TEST=false






# Parse model-specific parameters for a strategy
parse_model_params() {
    local strategy=$1
    local params_string="${MODEL_PARAMS[$strategy]}"
    
    if [ -z "$params_string" ]; then
        echo ""
        return
    fi
    
    # Parse format: "param1:val1,val2;param2:val3,val4"
    echo "$params_string"
}

# Get all parameter combinations for a strategy
get_param_combinations() {
    local strategy=$1
    local params_string=$(parse_model_params "$strategy")
    
    if [ -z "$params_string" ]; then
        echo "default"
        return
    fi
    
    # Split by semicolon to get individual parameters
    IFS=';' read -ra PARAMS <<< "$params_string"
    
    # For each parameter, split by colon to get name:values
    declare -A param_arrays
    declare -a param_names
    
    for param in "${PARAMS[@]}"; do
        IFS=':' read -r name values <<< "$param"
        param_names+=("$name")
        IFS=',' read -ra param_arrays[$name] <<< "$values"
    done
    
    # Generate all combinations
    # For simplicity, we'll iterate through first param values
    # You can extend this for full cartesian product
    if [ ${#param_names[@]} -eq 1 ]; then
        local pname="${param_names[0]}"
        for val in ${param_arrays[$pname][@]}; do
            echo "${pname}=${val}"
        done
    elif [ ${#param_names[@]} -eq 2 ]; then
        local pname1="${param_names[0]}"
        local pname2="${param_names[1]}"
        for val1 in ${param_arrays[$pname1][@]}; do
            for val2 in ${param_arrays[$pname2][@]}; do
                echo "${pname1}=${val1},${pname2}=${val2}"
            done
        done
    fi
}


usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --parallel          Run experiments in parallel"
    echo "  --clean             Clean previous results before running"
    echo "  --quick-test        Run a quick test with minimal parameters"
    echo "  --help              Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --parallel --clean    # Run all experiments in parallel, clean previous results"
    echo "  $0 --quick-test          # Run a single quick test"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --quick-test)
            QUICK_TEST=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Create config template if it doesn't exist
create_config_template() {
    cat > "$CONFIG_TEMPLATE" << 'EOF'
{
  "experiment_id": "PLACEHOLDER_ID",
  "algorithm": "PLACEHOLDER_ALGORITHM",
  "description": "NASA CMAPs FL Experiment",
  "run_id": "PLACEHOLDER_RUN_ID",
  "timestamp": "PLACEHOLDER_TIMESTAMP",
  "server": {
    "host": "PLACEHOLDER_HOST",
    "port": PLACEHOLDER_PORT,
    "num_rounds": PLACEHOLDER_ROUNDS
  },
  "strategy": {
    "name": "PLACEHOLDER_ALGORITHM",
    "fraction_fit": 1.0,
    "fraction_evaluate": 1.0,
    "min_fit_clients": PLACEHOLDER_MIN_CLIENTS,
    "min_evaluate_clients": PLACEHOLDER_MIN_CLIENTS,
    "min_available_clients": PLACEHOLDER_TOTAL_CLIENTS
  },
  "data": {
    "base_path": "PLACEHOLDER_DATA_PATH",
    "num_clients": PLACEHOLDER_TOTAL_CLIENTS,
    "alpha": PLACEHOLDER_ALPHA
  },
  "model": {
    "type": "dense",
    "learning_rate": 0.001,
    "local_epochs": 1,
    "batch_size": 32
  }
}
EOF
    echo "✅ Created config template at $CONFIG_TEMPLATE"
}

# Initialize experiment runner
initialize_runner() {
    echo "🧪 NASA FL Experiments Runner"
    echo "=============================="
    echo "Run ID: $RUN_ID"
    echo "Timestamp: $(date)"
    echo "Base Directory: $BASE_DIR"
    echo "Results Base: $RESULTS_BASE"
    echo "Mode: $([ "$PARALLEL" = true ] && echo "Parallel" || echo "Sequential")"
    echo "Quick Test: $([ "$QUICK_TEST" = true ] && echo "Yes" || echo "No")"
    echo ""
    
    # Check if required files exist
    if [ ! -f "$SERVER_SCRIPT" ]; then
        echo "❌ Server script not found: $SERVER_SCRIPT"
        exit 1
    fi
    
    if [ ! -f "$CLIENT_SCRIPT" ]; then
        echo "❌ Client script not found: $CLIENT_SCRIPT"
        exit 1
    fi
    
    # Create config template if needed
    if [ ! -f "$CONFIG_TEMPLATE" ]; then
        create_config_template
    fi
    
    # Update parameters for quick test
    if [ "$QUICK_TEST" = true ]; then
        STRATEGIES=("fedavg")
        CLIENTS=(5)
        ALPHAS=(0.1)
        DEFAULT_ROUNDS=3
        echo "🔬 Quick test mode activated"
        echo "   Strategies: ${STRATEGIES[@]}"
        echo "   Clients: ${CLIENTS[@]}"
        echo "   Alphas: ${ALPHAS[@]}"
        echo "   Rounds: $DEFAULT_ROUNDS"
    else
        echo "🔬 Full experiment mode"
        echo "   Strategies: ${STRATEGIES[@]}"
        echo "   Clients: ${CLIENTS[@]}"
        echo "   Alphas: ${ALPHAS[@]}"
        echo "   Rounds: $DEFAULT_ROUNDS"
    fi
    echo ""
}

# Clean previous results
clean_results() {
    if [ -d "$RESULTS_BASE" ]; then
        echo "🧹 Cleaning previous results from $RESULTS_BASE..."
        rm -rf "$RESULTS_BASE"
    fi
    mkdir -p "$RESULTS_BASE"
}

# Create run directory with timestamp
create_run_directory() {
    local run_dir="$RESULTS_BASE/$RUN_ID"
    mkdir -p "$run_dir"
    echo "$run_dir"
}

# Wait for port to be available
wait_for_port() {
    local port=$1
    local max_attempts=30
    local attempt=0
    
    while netstat -tuln 2>/dev/null | grep ":$port " > /dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "❌ Port $port still not available after $max_attempts attempts"
            return 1
        fi
        echo "⏳ Waiting for port $port to be available... ($attempt/$max_attempts)"
        sleep 2
    done
    return 0
}

# Generate experiment configuration
generate_experiment_config() {
    local strategy=$1
    local num_clients=$2
    local alpha=$3
    local port=$4
    local config_dir=$5
    
    local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}"
    local config_file="$config_dir/config.json"
    
    # Calculate min clients
    local min_clients=$((num_clients / 2))
    if [ $min_clients -lt $DEFAULT_MIN_CLIENTS ]; then
        min_clients=$DEFAULT_MIN_CLIENTS
    fi
    if [ $min_clients -gt $num_clients ]; then
        min_clients=$num_clients
    fi
    
    # Generate config file from template
    sed \
        -e "s/PLACEHOLDER_ID/$exp_id/g" \
        -e "s/PLACEHOLDER_ALGORITHM/$strategy/g" \
        -e "s/PLACEHOLDER_RUN_ID/$RUN_ID/g" \
        -e "s/PLACEHOLDER_TIMESTAMP/$TIMESTAMP/g" \
        -e "s/PLACEHOLDER_HOST/$SERVER_HOST/g" \
        -e "s/PLACEHOLDER_PORT/$port/g" \
        -e "s/PLACEHOLDER_ROUNDS/$DEFAULT_ROUNDS/g" \
        -e "s/PLACEHOLDER_MIN_CLIENTS/$min_clients/g" \
        -e "s/PLACEHOLDER_TOTAL_CLIENTS/$num_clients/g" \
        -e "s/PLACEHOLDER_ALPHA/$alpha/g" \
        -e "s|PLACEHOLDER_DATA_PATH|$BASE_DIR/data/nasa_cmaps/pre_split_data|g" \
        "$CONFIG_TEMPLATE" > "$config_file"
    
    echo "$config_file"
}

# Run a single experiment
run_experiment() {
    local strategy=$1
    local num_clients=$2
    local alpha=$3
    local port=$4
    local run_dir=$5
    
    local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}"
    local results_dir="$run_dir/$exp_id"
    
    echo "🚀 Starting experiment: $exp_id"
    echo "   Strategy: $strategy"
    echo "   Clients: $num_clients"
    echo "   Alpha: $alpha"
    echo "   Port: $port"
    echo "   Results: $results_dir"
    
    # Create experiment directories
    mkdir -p "$results_dir"
    mkdir -p "$results_dir/logs"
    mkdir -p "$results_dir/metrics"
    
    # Generate config file in the experiment directory
    local config_file=$(generate_experiment_config "$strategy" "$num_clients" "$alpha" "$port" "$results_dir")
    echo "   Config: $config_file"
    
    # Verify config file was created
    if [ ! -f "$config_file" ]; then
        echo "❌ Config file not created: $config_file"
        return 1
    fi
    
    # Start server with timestamped log
    local server_log="$results_dir/logs/server_$(date +%H%M%S).log"
    echo "   Starting server... Log: $server_log"
    
    # Start the server - FIXED: Use the config file we just created
    python3 "$SERVER_SCRIPT" --config "$config_file" --results-dir "$results_dir" > "$server_log" 2>&1 &
    local server_pid=$!
    
    # Wait for server to start
    echo "   Waiting for server to initialize..."
    sleep 10
    
    # Check if server started successfully
    if ! kill -0 $server_pid 2>/dev/null; then
        echo "❌ Server failed to start. Check log: $server_log"
        tail -20 "$server_log"
        return 1
    fi
    
    # Start clients
    echo "   Starting $num_clients clients..."
    local client_pids=()
    
    for ((i=0; i<num_clients; i++)); do
        local client_log="$results_dir/logs/client_${i}_$(date +%H%M%S).log"
        echo "   Starting client_$i..."
        python3 "$CLIENT_SCRIPT" --client-id "client_$i" --config "$config_file" > "$client_log" 2>&1 &
        client_pids+=($!)
        
        # Stagger client starts
        sleep 2
    done
    
    echo "   Experiment running... (Server PID: $server_pid)"
    echo "   Monitor progress: tail -f $server_log"
    
    # Wait for server to complete
    echo "   Waiting for server to complete..."
    wait $server_pid
    local server_exit_code=$?
    
    # Clean up any remaining clients
    for pid in "${client_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Cleaning up client PID: $pid"
            kill "$pid" 2>/dev/null
        fi
    done
    
    # Check experiment success
    if [ $server_exit_code -eq 0 ] && [ -f "$results_dir/metrics/round_metrics.csv" ]; then
        echo "✅ Experiment $exp_id completed successfully"
        # Compress logs to save space
        # gzip "$results_dir"/logs/*.log 2>/dev/null || true
        return 0
    else
        echo "❌ Experiment $exp_id failed with exit code $server_exit_code"
        echo "   Check logs in: $results_dir/logs/"
        # Show last few lines of server log for debugging
        if [ -f "$server_log" ]; then
            echo "   Last server log lines:"
            tail -10 "$server_log"
        fi
        return 1
    fi
}

# Run experiments in sequence
# run_sequential_experiments() {
#     local run_dir=$1
#     local current_port=$BASE_PORT
#     local total_experiments=$((${#STRATEGIES[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
#     local completed=0
#     local failed=0
    
#     echo "➡️ Running $total_experiments experiments sequentially..."
#     echo ""
    
#     for strategy in "${STRATEGIES[@]}"; do
#         for num_clients in "${CLIENTS[@]}"; do
#             for alpha in "${ALPHAS[@]}"; do
#                 ((completed++))
#                 echo "📊 Progress: $completed/$total_experiments"
#                 echo "🔧 Setting up: $strategy / $num_clients clients / alpha $alpha"
                
#                 if wait_for_port $current_port; then
#                     if run_experiment "$strategy" "$num_clients" "$alpha" $current_port "$run_dir"; then
#                         echo "✅ Completed: $strategy / $num_clients clients / alpha $alpha"
#                     else
#                         ((failed++))
#                         # echo "❌ Failed: $strategy / $num_clients clients / alpha $alpha"
#                     fi
#                 else
#                     ((failed++))
#                     # echo "❌ Port unavailable: $strategy / $num_clients clients / alpha $alpha"
#                 fi
                
#                 current_port=$((current_port + 1))
#                 echo "----------------------------------------"
#                 sleep 5  # Brief pause between experiments
#             done
#         done
#     done
    
#     echo "🎉 Sequential experiments completed: $((completed - failed))/$completed successful"
#     return $failed
# }

run_sequential_experiments() {
    local run_dir=$1
    local current_port=$BASE_PORT
    local total_experiments=0
    
    # Calculate total experiments
    for strategy in "${STRATEGIES[@]}"; do
        local param_combos=($(get_param_combinations "$strategy"))
        total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
    done
    
    local completed=0
    local failed=0
    
    echo "➡️ Running $total_experiments experiments sequentially..."
    echo ""
    
    for strategy in "${STRATEGIES[@]}"; do
        local param_combos=($(get_param_combinations "$strategy"))
        
        for param_combo in "${param_combos[@]}"; do
            for num_clients in "${CLIENTS[@]}"; do
                for alpha in "${ALPHAS[@]}"; do
                    ((completed++))
                    echo "📊 Progress: $completed/$total_experiments"
                    echo "🔧 Setting up: $strategy / $param_combo / $num_clients clients / alpha $alpha"
                    
                    if wait_for_port $current_port; then
                        if run_experiment "$strategy" "$num_clients" "$alpha" $current_port "$run_dir" "$param_combo"; then
                            echo "✅ Completed"
                        else
                            ((failed++))
                        fi
                    else
                        ((failed++))
                    fi
                    
                    current_port=$((current_port + 1))
                    echo "----------------------------------------"
                    sleep 5
                done
            done
        done
    done
    
    echo "🎉 Sequential experiments completed: $((completed - failed))/$completed successful"
    return $failed
}

# Run experiments in parallel (one per strategy)
run_parallel_experiments() {
    local run_dir=$1
    local base_port=$BASE_PORT
    local experiment_pids=()
    local experiment_info=()
    
    echo "🔀 Running experiments in parallel mode..."
    echo ""
    
    for strategy in "${STRATEGIES[@]}"; do
        for num_clients in "${CLIENTS[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                if wait_for_port $base_port; then
                    (
                        echo "🚀 Starting parallel experiment: $strategy / $num_clients / $alpha on port $base_port"
                        if run_experiment "$strategy" "$num_clients" "$alpha" $base_port "$run_dir"; then
                            echo "✅ Parallel experiment completed: $strategy / $num_clients / $alpha"
                        else
                            echo "❌ Parallel experiment failed: $strategy / $num_clients / $alpha"
                            exit 1
                        fi
                    ) &
                    experiment_pids+=($!)
                    experiment_info+=("$strategy:$num_clients:$alpha:$base_port")
                    base_port=$((base_port + 1))
                    
                    # Delay between starting parallel experiments
                    sleep 15
                else
                    echo "❌ Port $base_port unavailable, skipping experiment"
                    base_port=$((base_port + 1))
                fi
            done
        done
    done
    
    # Wait for all experiments to complete
    echo "⏳ Waiting for all parallel experiments to complete..."
    local failed=0
    for i in "${!experiment_pids[@]}"; do
        if wait ${experiment_pids[i]}; then
            echo "✅ ${experiment_info[i]} - Success"
        else
            echo "❌ ${experiment_info[i]} - Failed"
            ((failed++))
        fi
    done
    
    echo "🎉 Parallel experiments completed: $((${#experiment_pids[@]} - failed))/${#experiment_pids[@]} successful"
    return $failed
}

# Generate comprehensive summary
generate_summary() {
    local run_dir=$1
    local summary_file="$run_dir/experiment_summary.md"
    
    local total_experiments=$((${#STRATEGIES[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
    local completed_count=0
    
    cat > "$summary_file" << EOF
# NASA FL Experiments Summary

## Run Information
- **Run ID**: $RUN_ID
- **Timestamp**: $(date)
- **Total Experiments**: $total_experiments
- **Run Directory**: $run_dir

## Experiment Parameters
- **Strategies**: ${STRATEGIES[@]}
- **Client Counts**: ${CLIENTS[@]}
- **Alpha Values**: ${ALPHAS[@]}
- **Server Rounds**: $DEFAULT_ROUNDS

## Experiments Status

| Experiment ID | Strategy | Clients | Alpha | Port | Status | Results |
|---------------|----------|---------|-------|------|--------|---------|
EOF

    # local current_port=$BASE_PORT
    # for strategy in "${STRATEGIES[@]}"; do
    #     for num_clients in "${CLIENTS[@]}"; do
    #         for alpha in "${ALPHAS[@]}"; do
    #             local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}"
    #             local exp_dir="$run_dir/$exp_id"
                
    #             local status="❌ Failed"
    #             local results_link="-"
                
    #             if [ -f "$exp_dir/metrics/round_metrics.csv" ]; then
    #                 status="✅ Completed"
    #                 results_link="[View]($exp_id/metrics/round_metrics.csv)"
    #                 ((completed_count++))
    #             elif [ -d "$exp_dir" ]; then
    #                 status="⚠️ Partial"
    #             fi
                
    #             echo "| $exp_id | $strategy | $num_clients | $alpha | $current_port | $status | $results_link |" >> "$summary_file"
    #             current_port=$((current_port + 1))
    #         done
    #     done
    # done

    cat >> "$summary_file" << EOF

## Summary
- **Completed**: $completed_count/$total_experiments
- **Success Rate**: $((completed_count * 100 / total_experiments))%

## Quick Analysis Commands
\`\`\`bash
# Check completion status
find "$run_dir" -name "round_metrics.csv" | wc -l

# View recent results
find "$run_dir" -name "round_metrics.csv" -exec dirname {} \; | while read dir; do
    echo "=== \$(basename \$dir) ==="
    tail -1 "\$dir/round_metrics.csv"
done

# Monitor disk usage
du -sh "$run_dir"
\`\`\`

## Directory Structure
\`\`\`
$run_dir/
├── experiment_summary.md
└── nasa_[clients]c_alpha_[alpha]_[strategy]/
    ├── config.json
    ├── logs/
    │   ├── server_[time].log.gz
    │   └── client_*_[time].log.gz
    └── metrics/
        ├── round_metrics.csv
        ├── client_metrics.csv
        └── eval_metrics.csv
\`\`\`
EOF

    echo "📊 Summary generated: $summary_file"
    echo "📈 Completion: $completed_count/$total_experiments experiments"
}

# Main execution
main() {
    initialize_runner
    
    # Clean results if requested
    if [ "$CLEAN" = true ]; then
        clean_results
    fi
    
    # Create run directory with timestamp
    local run_dir=$(create_run_directory)
    echo "📁 Run directory: $run_dir"
    
    # Start time
    local start_time=$(date +%s)
    
    # Run experiments
    local failed_experiments=0
    if [ "$PARALLEL" = true ]; then
        run_parallel_experiments "$run_dir" || failed_experiments=$?
    else
        run_sequential_experiments "$run_dir" || failed_experiments=$?
    fi
    
    # End time and duration
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Generate summary
    generate_summary "$run_dir"
    
    # Final report
    echo ""
    echo "🎉 Experiment Run Completed!"
    echo "=============================="
    echo "Run ID: $RUN_ID"
    echo "Duration: $((duration / 60))m $((duration % 60))s"
    echo "Results: $run_dir"
    echo "Summary: $run_dir/experiment_summary.md"
    
    # if [ $failed_experiments -gt 0 ]; then
    #     echo "⚠️  $failed_experiments experiments failed. Check the logs above."
    #     exit 1
    # else
    #     echo "✅ All experiments completed successfully!"
    #     exit 0
    # fi
}

# Run main function
main "$@"