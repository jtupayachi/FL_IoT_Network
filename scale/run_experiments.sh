# # # #!/bin/bash

# # # set -e  # Exit on any error

# # # # Configuration
# # # BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# # # RESULTS_BASE="$BASE_DIR/results"
# # # SERVER_SCRIPT="$BASE_DIR/server.py"
# # # CLIENT_SCRIPT="$BASE_DIR/client.py"
# # # CONFIG_TEMPLATE="$BASE_DIR/config_template.json"
# # # SERVER_HOST="localhost"
# # # BASE_PORT=8686

# # # # Timestamp for all experiments in this run
# # # TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # # RUN_ID="run_${TIMESTAMP}"

# # # # Experiment parameters - UPDATED FOR MULTIPLE STRATEGIES
# # # declare -a STRATEGIES=("fedavg" "fedavgm" "fedopt" "qfedavg")

# # # # Model-specific parameters: format is "param_name:value1,value2,value3"
# # # # For multiple parameters use semicolon: "param1:val1,val2;param2:val3,val4"
# # # MODEL_PARAMS_FEDAVG=""
# # # MODEL_PARAMS_FEDAVGM="server_momentum:0.5,0.9"
# # # MODEL_PARAMS_FEDOPT="tau:0.001,0.01"
# # # MODEL_PARAMS_QFEDAVG="q_param:0.1,0.5,1.0;qffl_learning_rate:0.01,0.1"

# # # declare -a CLIENTS=(25)  # Keep constant
# # # declare -a ALPHAS=(0.5)  # Vary this single parameter (0.1=very non-IID, 1.0=more IID)

# # # # Default values
# # # DEFAULT_ROUNDS=10
# # # DEFAULT_MIN_CLIENTS=2

# # # # Parse command line arguments
# # # PARALLEL=false
# # # CLEAN=false
# # # QUICK_TEST=false

# # # # Parse model-specific parameters for a strategy
# # # parse_model_params() {
# # #     local strategy=$1
# # #     local params_string=""
    
# # #     case "$strategy" in
# # #         "fedavg")
# # #             params_string="$MODEL_PARAMS_FEDAVG"
# # #             ;;
# # #         "fedavgm")
# # #             params_string="$MODEL_PARAMS_FEDAVGM"
# # #             ;;
# # #         "fedopt")
# # #             params_string="$MODEL_PARAMS_FEDOPT"
# # #             ;;
# # #         "qfedavg")
# # #             params_string="$MODEL_PARAMS_QFEDAVG"
# # #             ;;
# # #     esac
    
# # #     echo "$params_string"
# # # }

# # # # Get all parameter combinations for a strategy
# # # get_param_combinations() {
# # #     local strategy=$1
# # #     local params_string=$(parse_model_params "$strategy")
    
# # #     if [ -z "$params_string" ]; then
# # #         echo "default"
# # #         return
# # #     fi
    
# # #     # Split by semicolon to get individual parameters
# # #     IFS=';' read -ra PARAMS <<< "$params_string"
    
# # #     # Simple approach: generate combinations
# # #     if [ ${#PARAMS[@]} -eq 1 ]; then
# # #         # Single parameter with multiple values
# # #         local param="${PARAMS[0]}"
# # #         IFS=':' read -r pname pvalues <<< "$param"
# # #         IFS=',' read -ra values <<< "$pvalues"
        
# # #         for val in "${values[@]}"; do
# # #             echo "${pname}=${val}"
# # #         done
# # #     elif [ ${#PARAMS[@]} -eq 2 ]; then
# # #         # Two parameters - cartesian product
# # #         local param1="${PARAMS[0]}"
# # #         local param2="${PARAMS[1]}"
        
# # #         IFS=':' read -r pname1 pvalues1 <<< "$param1"
# # #         IFS=':' read -r pname2 pvalues2 <<< "$param2"
        
# # #         IFS=',' read -ra values1 <<< "$pvalues1"
# # #         IFS=',' read -ra values2 <<< "$pvalues2"
        
# # #         for val1 in "${values1[@]}"; do
# # #             for val2 in "${values2[@]}"; do
# # #                 echo "${pname1}=${val1},${pname2}=${val2}"
# # #             done
# # #         done
# # #     else
# # #         echo "default"
# # #     fi
# # # }

# # # usage() {
# # #     echo "Usage: $0 [OPTIONS]"
# # #     echo "Options:"
# # #     echo "  --parallel          Run experiments in parallel"
# # #     echo "  --clean             Clean previous results before running"
# # #     echo "  --quick-test        Run a quick test with minimal parameters"
# # #     echo "  --help              Show this help message"
# # #     echo ""
# # #     echo "Example:"
# # #     echo "  $0 --parallel --clean    # Run all experiments in parallel, clean previous results"
# # #     echo "  $0 --quick-test          # Run a single quick test"
# # # }

# # # while [[ $# -gt 0 ]]; do
# # #     case $1 in
# # #         --parallel)
# # #             PARALLEL=true
# # #             shift
# # #             ;;
# # #         --clean)
# # #             CLEAN=true
# # #             shift
# # #             ;;
# # #         --quick-test)
# # #             QUICK_TEST=true
# # #             shift
# # #             ;;
# # #         --help)
# # #             usage
# # #             exit 0
# # #             ;;
# # #         *)
# # #             echo "Unknown option: $1"
# # #             usage
# # #             exit 1
# # #             ;;
# # #     esac
# # # done

# # # # Create config template if it doesn't exist
# # # create_config_template() {
# # #     cat > "$CONFIG_TEMPLATE" << 'EOF'
# # # {
# # #   "experiment_id": "PLACEHOLDER_ID",
# # #   "algorithm": "PLACEHOLDER_ALGORITHM",
# # #   "description": "NASA CMAPs FL Experiment",
# # #   "run_id": "PLACEHOLDER_RUN_ID",
# # #   "timestamp": "PLACEHOLDER_TIMESTAMP",
# # #   "server": {
# # #     "host": "PLACEHOLDER_HOST",
# # #     "port": PLACEHOLDER_PORT,
# # #     "num_rounds": PLACEHOLDER_ROUNDS
# # #   },
# # #   "strategy": {
# # #     "name": "PLACEHOLDER_ALGORITHM",
# # #     "fraction_fit": 1.0,
# # #     "fraction_evaluate": 1.0,
# # #     "min_fit_clients": PLACEHOLDER_MIN_CLIENTS,
# # #     "min_evaluate_clients": PLACEHOLDER_MIN_CLIENTS,
# # #     "min_available_clients": PLACEHOLDER_TOTAL_CLIENTS,
# # #     "params": {}
# # #   },
# # #   "data": {
# # #     "base_path": "PLACEHOLDER_DATA_PATH",
# # #     "num_clients": PLACEHOLDER_TOTAL_CLIENTS,
# # #     "alpha": PLACEHOLDER_ALPHA
# # #   },
# # #   "model": {
# # #     "model_type": "dense",
# # #     "learning_rate": 0.001,
# # #     "local_epochs": 1,
# # #     "batch_size": 32,
# # #     "hidden_dims": [64, 32],
# # #     "dropout": 0.2,
# # #     "n_components": 10
# # #   }
# # # }
# # # EOF
# # #     echo "✅ Created config template at $CONFIG_TEMPLATE"
# # # }

# # # # Initialize experiment runner
# # # initialize_runner() {
# # #     echo "🧪 NASA FL Experiments Runner"
# # #     echo "=============================="
# # #     echo "Run ID: $RUN_ID"
# # #     echo "Timestamp: $(date)"
# # #     echo "Base Directory: $BASE_DIR"
# # #     echo "Results Base: $RESULTS_BASE"
# # #     echo "Mode: $([ "$PARALLEL" = true ] && echo "Parallel" || echo "Sequential")"
# # #     echo "Quick Test: $([ "$QUICK_TEST" = true ] && echo "Yes" || echo "No")"
# # #     echo ""
    
# # #     # Check if required files exist
# # #     if [ ! -f "$SERVER_SCRIPT" ]; then
# # #         echo "❌ Server script not found: $SERVER_SCRIPT"
# # #         exit 1
# # #     fi
    
# # #     if [ ! -f "$CLIENT_SCRIPT" ]; then
# # #         echo "❌ Client script not found: $CLIENT_SCRIPT"
# # #         exit 1
# # #     fi
    
# # #     # Create config template if needed
# # #     if [ ! -f "$CONFIG_TEMPLATE" ]; then
# # #         create_config_template
# # #     fi
    
# # #     # Update parameters for quick test
# # #     if [ "$QUICK_TEST" = true ]; then
# # #         STRATEGIES=("fedavg")
# # #         CLIENTS=(5)
# # #         ALPHAS=(0.5)
# # #         DEFAULT_ROUNDS=3
# # #         echo "🔬 Quick test mode activated"
# # #         echo "   Strategies: ${STRATEGIES[@]}"
# # #         echo "   Clients: ${CLIENTS[@]}"
# # #         echo "   Alphas: ${ALPHAS[@]}"
# # #         echo "   Rounds: $DEFAULT_ROUNDS"
# # #     else
# # #         echo "🔬 Full experiment mode"
# # #         echo "   Strategies: ${STRATEGIES[@]}"
# # #         echo "   Clients: ${CLIENTS[@]}"
# # #         echo "   Alphas: ${ALPHAS[@]}"
# # #         echo "   Rounds: $DEFAULT_ROUNDS"
# # #     fi
# # #     echo ""
# # # }

# # # # Clean previous results
# # # clean_results() {
# # #     if [ -d "$RESULTS_BASE" ]; then
# # #         echo "🧹 Cleaning previous results from $RESULTS_BASE..."
# # #         rm -rf "$RESULTS_BASE"
# # #     fi
# # #     mkdir -p "$RESULTS_BASE"
# # # }

# # # # Create run directory with timestamp
# # # create_run_directory() {
# # #     local run_dir="$RESULTS_BASE/$RUN_ID"
# # #     mkdir -p "$run_dir"
# # #     echo "$run_dir"
# # # }

# # # # Wait for port to be available
# # # wait_for_port() {
# # #     local port=$1
# # #     local max_attempts=30
# # #     local attempt=0
    
# # #     while netstat -tuln 2>/dev/null | grep ":$port " > /dev/null; do
# # #         attempt=$((attempt + 1))
# # #         if [ $attempt -ge $max_attempts ]; then
# # #             echo "❌ Port $port still not available after $max_attempts attempts"
# # #             return 1
# # #         fi
# # #         echo "⏳ Waiting for port $port to be available... ($attempt/$max_attempts)"
# # #         sleep 2
# # #     done
# # #     return 0
# # # }

# # # # Generate experiment configuration
# # # generate_experiment_config() {
# # #     local strategy=$1
# # #     local num_clients=$2
# # #     local alpha=$3
# # #     local port=$4
# # #     local config_dir=$5
# # #     local param_combo=${6:-"default"}
    
# # #     # Generate param suffix for experiment ID
# # #     local param_suffix=""
# # #     if [ "$param_combo" != "default" ]; then
# # #         param_suffix="_${param_combo//[=,]/_}"
# # #     fi
    
# # #     local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}${param_suffix}"
# # #     local config_file="$config_dir/config.json"
    
# # #     # Calculate min clients
# # #     local min_clients=$((num_clients / 2))
# # #     if [ $min_clients -lt $DEFAULT_MIN_CLIENTS ]; then
# # #         min_clients=$DEFAULT_MIN_CLIENTS
# # #     fi
# # #     if [ $min_clients -gt $num_clients ]; then
# # #         min_clients=$num_clients
# # #     fi
    
# # #     # Create directory if it doesn't exist
# # #     mkdir -p "$config_dir"
    
# # #     # Generate config file from template
# # #     sed \
# # #         -e "s/PLACEHOLDER_ID/$exp_id/g" \
# # #         -e "s/PLACEHOLDER_ALGORITHM/$strategy/g" \
# # #         -e "s/PLACEHOLDER_RUN_ID/$RUN_ID/g" \
# # #         -e "s/PLACEHOLDER_TIMESTAMP/$TIMESTAMP/g" \
# # #         -e "s/PLACEHOLDER_HOST/$SERVER_HOST/g" \
# # #         -e "s/PLACEHOLDER_PORT/$port/g" \
# # #         -e "s/PLACEHOLDER_ROUNDS/$DEFAULT_ROUNDS/g" \
# # #         -e "s/PLACEHOLDER_MIN_CLIENTS/$min_clients/g" \
# # #         -e "s/PLACEHOLDER_TOTAL_CLIENTS/$num_clients/g" \
# # #         -e "s/PLACEHOLDER_ALPHA/$alpha/g" \
# # #         -e "s|PLACEHOLDER_DATA_PATH|$BASE_DIR/data/nasa_cmaps/pre_split_data|g" \
# # #         "$CONFIG_TEMPLATE" > "$config_file"
    
# # #     # Add model-specific parameters to config if present
# # #     if [ "$param_combo" != "default" ]; then
# # #         python3 - <<EOF "$config_file" "$param_combo"
# # # import json
# # # import sys

# # # config_file = sys.argv[1]
# # # params_str = sys.argv[2]

# # # with open(config_file, 'r') as f:
# # #     config = json.load(f)

# # # # Ensure strategy params exists
# # # if 'strategy' not in config:
# # #     config['strategy'] = {}
# # # if 'params' not in config['strategy']:
# # #     config['strategy']['params'] = {}

# # # # Parse parameters
# # # for param in params_str.split(','):
# # #     key, val = param.split('=')
# # #     try:
# # #         config['strategy']['params'][key] = float(val)
# # #     except ValueError:
# # #         config['strategy']['params'][key] = val

# # # with open(config_file, 'w') as f:
# # #     json.dump(config, f, indent=2)
# # # EOF
# # #     fi
    
# # #     echo "$config_file"
# # # }

# # # # Run a single experiment
# # # run_experiment() {
# # #     local strategy=$1
# # #     local num_clients=$2
# # #     local alpha=$3
# # #     local port=$4
# # #     local run_dir=$5
# # #     local param_combo=${6:-"default"}
    
# # #     local param_suffix=""
# # #     if [ "$param_combo" != "default" ]; then
# # #         param_suffix="_${param_combo//[=,]/_}"
# # #     fi
    
# # #     local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}${param_suffix}"
# # #     local results_dir="$run_dir/$exp_id"
    
# # #     echo "🚀 Starting experiment: $exp_id"
# # #     echo "   Strategy: $strategy"
# # #     if [ "$param_combo" != "default" ]; then
# # #         echo "   Parameters: $param_combo"
# # #     fi
# # #     echo "   Clients: $num_clients"
# # #     echo "   Alpha: $alpha"
# # #     echo "   Port: $port"
# # #     echo "   Results: $results_dir"
    
# # #     # Create experiment directories
# # #     mkdir -p "$results_dir"
# # #     mkdir -p "$results_dir/logs"
# # #     mkdir -p "$results_dir/metrics"
    
# # #     # Generate config file in the experiment directory
# # #     local config_file=$(generate_experiment_config "$strategy" "$num_clients" "$alpha" "$port" "$results_dir" "$param_combo")
    
# # #     echo "   Config: $config_file"
    
# # #     # Verify config file was created
# # #     if [ ! -f "$config_file" ]; then
# # #         echo "❌ Config file not created: $config_file"
# # #         return 1
# # #     fi
    
# # #     # Start server with timestamped log
# # #     local server_log="$results_dir/logs/server_$(date +%H%M%S).log"
# # #     echo "   Starting server... Log: $server_log"
    
# # #     # Start the server
# # #     python3 "$SERVER_SCRIPT" --config "$config_file" --results-dir "$results_dir" > "$server_log" 2>&1 &
# # #     local server_pid=$!
    
# # #     # Wait for server to start
# # #     echo "   Waiting for server to initialize..."
# # #     sleep 10
    
# # #     # Check if server started successfully
# # #     if ! kill -0 $server_pid 2>/dev/null; then
# # #         echo "❌ Server failed to start. Check log: $server_log"
# # #         echo "   Last 30 lines of server log:"
# # #         tail -30 "$server_log"
# # #         return 1
# # #     fi
    
# # #     # Start clients
# # #     echo "   Starting $num_clients clients..."
# # #     local client_pids=()
    
# # #     for ((i=0; i<num_clients; i++)); do
# # #         local client_log="$results_dir/logs/client_${i}_$(date +%H%M%S).log"
# # #         echo "   Starting client_$i..."
# # #         python3 "$CLIENT_SCRIPT" --client-id "client_$i" --config "$config_file" > "$client_log" 2>&1 &
# # #         client_pids+=($!)
        
# # #         # Stagger client starts
# # #         sleep 2
# # #     done
    
# # #     echo "   Experiment running... (Server PID: $server_pid)"
# # #     echo "   Monitor progress: tail -f $server_log"
    
# # #     # Wait for server to complete
# # #     echo "   Waiting for server to complete..."
# # #     wait $server_pid
# # #     local server_exit_code=$?
    
# # #     # Give clients time to finish
# # #     sleep 5
    
# # #     # Clean up any remaining clients
# # #     for pid in "${client_pids[@]}"; do
# # #         if kill -0 "$pid" 2>/dev/null; then
# # #             echo "   Cleaning up client PID: $pid"
# # #             kill "$pid" 2>/dev/null || true
# # #         fi
# # #     done
    
# # #     # Check experiment success
# # #     if [ $server_exit_code -eq 0 ] && [ -f "$results_dir/metrics/round_metrics.csv" ]; then
# # #         echo "✅ Experiment $exp_id completed successfully"
# # #         return 0
# # #     else
# # #         echo "❌ Experiment $exp_id failed with exit code $server_exit_code"
# # #         echo "   Check logs in: $results_dir/logs/"
# # #         if [ -f "$server_log" ]; then
# # #             echo "   Last 20 server log lines:"
# # #             tail -20 "$server_log"
# # #         fi
# # #         return 1
# # #     fi
# # # }

# # # # Run experiments in sequence
# # # run_sequential_experiments() {
# # #     local run_dir=$1
# # #     local current_port=$BASE_PORT
# # #     local total_experiments=0
    
# # #     # Calculate total experiments
# # #     for strategy in "${STRATEGIES[@]}"; do
# # #         local param_combos=($(get_param_combinations "$strategy"))
# # #         total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
# # #     done
    
# # #     local completed=0
# # #     local failed=0
    
# # #     echo "➡️ Running $total_experiments experiments sequentially..."
# # #     echo ""
    
# # #     for strategy in "${STRATEGIES[@]}"; do
# # #         local param_combos=($(get_param_combinations "$strategy"))
        
# # #         for param_combo in "${param_combos[@]}"; do
# # #             for num_clients in "${CLIENTS[@]}"; do
# # #                 for alpha in "${ALPHAS[@]}"; do
# # #                     ((completed++))
# # #                     echo ""
# # #                     echo "========================================"
# # #                     echo "📊 Progress: $completed/$total_experiments"
# # #                     echo "🔧 Setting up: $strategy / $param_combo / $num_clients clients / alpha $alpha"
# # #                     echo "========================================"
                    
# # #                     if wait_for_port $current_port; then
# # #                         if run_experiment "$strategy" "$num_clients" "$alpha" $current_port "$run_dir" "$param_combo"; then
# # #                             echo "✅ Experiment $completed/$total_experiments completed"
# # #                         else
# # #                             echo "❌ Experiment $completed/$total_experiments failed"
# # #                             ((failed++))
# # #                         fi
# # #                     else
# # #                         echo "❌ Port $current_port unavailable"
# # #                         ((failed++))
# # #                     fi
                    
# # #                     current_port=$((current_port + 1))
# # #                     echo "========================================"
# # #                     echo ""
                    
# # #                     # Cleanup between experiments
# # #                     sleep 5
# # #                 done
# # #             done
# # #         done
# # #     done
    
# # #     echo ""
# # #     echo "🎉 Sequential experiments completed!"
# # #     echo "   Successful: $((completed - failed))/$completed"
# # #     echo "   Failed: $failed"
# # #     return $failed
# # # }

# # # # Run experiments in parallel (one per strategy)
# # # run_parallel_experiments() {
# # #     local run_dir=$1
# # #     local base_port=$BASE_PORT
# # #     local experiment_pids=()
# # #     local experiment_info=()
    
# # #     echo "🔀 Running experiments in parallel mode..."
# # #     echo ""
    
# # #     local exp_count=0
    
# # #     for strategy in "${STRATEGIES[@]}"; do
# # #         local param_combos=($(get_param_combinations "$strategy"))
        
# # #         for param_combo in "${param_combos[@]}"; do
# # #             for num_clients in "${CLIENTS[@]}"; do
# # #                 for alpha in "${ALPHAS[@]}"; do
# # #                     ((exp_count++))
                    
# # #                     if wait_for_port $base_port; then
# # #                         echo "🚀 Starting parallel experiment $exp_count: $strategy / $param_combo / $num_clients / $alpha on port $base_port"
                        
# # #                         (
# # #                             if run_experiment "$strategy" "$num_clients" "$alpha" $base_port "$run_dir" "$param_combo"; then
# # #                                 echo "✅ Parallel experiment $exp_count completed: $strategy / $param_combo / $num_clients / $alpha"
# # #                             else
# # #                                 echo "❌ Parallel experiment $exp_count failed: $strategy / $param_combo / $num_clients / $alpha"
# # #                                 exit 1
# # #                             fi
# # #                         ) &
                        
# # #                         experiment_pids+=($!)
# # #                         experiment_info+=("$exp_count:$strategy:$param_combo:$num_clients:$alpha:$base_port")
# # #                         base_port=$((base_port + 1))
                        
# # #                         # Delay between starting parallel experiments
# # #                         sleep 15
# # #                     else
# # #                         echo "❌ Port $base_port unavailable, skipping experiment"
# # #                         base_port=$((base_port + 1))
# # #                     fi
# # #                 done
# # #             done
# # #         done
# # #     done
    
# # #     # Wait for all experiments to complete
# # #     echo ""
# # #     echo "⏳ Waiting for all $exp_count parallel experiments to complete..."
# # #     echo ""
    
# # #     local failed=0
# # #     for i in "${!experiment_pids[@]}"; do
# # #         if wait ${experiment_pids[i]}; then
# # #             echo "✅ ${experiment_info[i]} - Success"
# # #         else
# # #             echo "❌ ${experiment_info[i]} - Failed"
# # #             ((failed++))
# # #         fi
# # #     done
    
# # #     echo ""
# # #     echo "🎉 Parallel experiments completed!"
# # #     echo "   Successful: $((${#experiment_pids[@]} - failed))/${#experiment_pids[@]}"
# # #     echo "   Failed: $failed"
# # #     return $failed
# # # }

# # # # Generate comprehensive summary
# # # generate_summary() {
# # #     local run_dir=$1
# # #     local summary_file="$run_dir/experiment_summary.md"
    
# # #     # Count total experiments
# # #     local total_experiments=0
# # #     for strategy in "${STRATEGIES[@]}"; do
# # #         local param_combos=($(get_param_combinations "$strategy"))
# # #         total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
# # #     done
    
# # #     local completed_count=$(find "$run_dir" -name "round_metrics.csv" 2>/dev/null | wc -l)
    
# # #     cat > "$summary_file" << EOF
# # # # NASA FL Experiments Summary

# # # ## Run Information
# # # - **Run ID**: $RUN_ID
# # # - **Timestamp**: $(date)
# # # - **Total Experiments**: $total_experiments
# # # - **Completed**: $completed_count
# # # - **Run Directory**: \`$run_dir\`

# # # ## Experiment Parameters
# # # - **Strategies**: ${STRATEGIES[@]}
# # # - **Client Counts**: ${CLIENTS[@]}
# # # - **Alpha Values**: ${ALPHAS[@]}
# # # - **Server Rounds**: $DEFAULT_ROUNDS

# # # ## Model-Specific Parameters
# # # - **FedAvg**: ${MODEL_PARAMS_FEDAVG:-"None"}
# # # - **FedAvgM**: ${MODEL_PARAMS_FEDAVGM:-"None"}
# # # - **FedOpt**: ${MODEL_PARAMS_FEDOPT:-"None"}
# # # - **QFedAvg**: ${MODEL_PARAMS_QFEDAVG:-"None"}

# # # ## Summary
# # # - **Total Experiments**: $total_experiments
# # # - **Completed**: $completed_count
# # # - **Success Rate**: $((completed_count > 0 ? completed_count * 100 / total_experiments : 0))%

# # # ## Results Structure
# # # \`\`\`
# # # $run_dir/
# # # ├── nasa_<clients>c_alpha_<alpha>_<strategy>_<params>/
# # # │   ├── config.json
# # # │   ├── logs/
# # # │   │   ├── server_*.log
# # # │   │   └── client_*_*.log
# # # │   └── metrics/
# # # │       ├── round_metrics.csv
# # # │       ├── client_metrics.csv
# # # │       └── test_metrics.csv
# # # └── experiment_summary.md
# # # \`\`\`

# # # ## Quick Analysis Commands

# # # ### Check completion status
# # # \`\`\`bash
# # # find "$run_dir" -name "round_metrics.csv" | wc -l
# # # \`\`\`

# # # ### View recent results
# # # \`\`\`bash
# # # find "$run_dir" -name "round_metrics.csv" -exec dirname {} \; | while read dir; do
# # #     echo "=== \$(basename \$(dirname \$dir)) ==="
# # #     tail -1 "\$dir/round_metrics.csv"
# # # done
# # # \`\`\`

# # # ### Monitor disk usage
# # # \`\`\`bash
# # # du -sh "$run_dir"
# # # \`\`\`

# # # ### Find failed experiments
# # # \`\`\`bash
# # # for dir in $run_dir/nasa_*/; do
# # #     if [ ! -f "\$dir/metrics/round_metrics.csv" ]; then
# # #         echo "Failed: \$(basename \$dir)"
# # #     fi
# # # done
# # # \`\`\`

# # # ### View error logs
# # # \`\`\`bash
# # # find "$run_dir" -name "server_*.log" -exec grep -l "Error\|Exception\|Traceback" {} \;
# # # \`\`\`

# # # ## Notes
# # # Generated on: $(date)
# # # EOF

# # #     echo "📊 Summary generated: $summary_file"
# # #     echo "📈 Completion: $completed_count/$total_experiments experiments"
# # # }

# # # # Main execution
# # # main() {
# # #     initialize_runner
    
# # #     # Clean results if requested
# # #     if [ "$CLEAN" = true ]; then
# # #         clean_results
# # #     fi
    
# # #     # Create run directory with timestamp
# # #     local run_dir=$(create_run_directory)
# # #     echo "📁 Run directory: $run_dir"
# # #     echo ""
    
# # #     # Start time
# # #     local start_time=$(date +%s)
    
# # #     # Run experiments
# # #     local failed_experiments=0
# # #     if [ "$PARALLEL" = true ]; then
# # #         run_parallel_experiments "$run_dir" || failed_experiments=$?
# # #     else
# # #         run_sequential_experiments "$run_dir" || failed_experiments=$?
# # #     fi
    
# # #     # End time and duration
# # #     local end_time=$(date +%s)
# # #     local duration=$((end_time - start_time))
# # #     local hours=$((duration / 3600))
# # #     local minutes=$(((duration % 3600) / 60))
# # #     local seconds=$((duration % 60))
    
# # #     # Generate summary
# # #     generate_summary "$run_dir"
    
# # #     # Final report
# # #     echo ""
# # #     echo "========================================"
# # #     echo "🎉 Experiment Run Completed!"
# # #     echo "========================================"
# # #     echo "Run ID: $RUN_ID"
# # #     echo "Duration: ${hours}h ${minutes}m ${seconds}s"
# # #     echo "Results: $run_dir"
# # #     echo "Summary: $run_dir/experiment_summary.md"
# # #     echo ""
    
# # #     if [ $failed_experiments -gt 0 ]; then
# # #         echo "⚠️  $failed_experiments experiments failed. Check the logs above."
# # #         echo "========================================"
# # #         exit 1
# # #     else
# # #         echo "✅ All experiments completed successfully!"
# # #         echo "========================================"
# # #         exit 0
# # #     fi
# # # }

# # # # Run main function with all arguments
# # # main "$@"


# # #!/bin/bash

# # set -e  # Exit on any error

# # # Configuration
# # BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# # RESULTS_BASE="$BASE_DIR/results"
# # SERVER_SCRIPT="$BASE_DIR/server.py"
# # CLIENT_SCRIPT="$BASE_DIR/client.py"
# # CONFIG_TEMPLATE="$BASE_DIR/config_template.json"
# # SERVER_HOST="localhost"
# # BASE_PORT=8686

# # # Timestamp for all experiments in this run
# # TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# # RUN_ID="run_${TIMESTAMP}"

# # # Experiment parameters - UPDATED FOR MULTIPLE STRATEGIES
# # declare -a STRATEGIES=("fedavg" "fedavgm" "fedopt" "qfedavg")

# # # Model-specific parameters: format is "param_name:value1,value2,value3"
# # MODEL_PARAMS_FEDAVG=""
# # MODEL_PARAMS_FEDAVGM="server_momentum:0.5,0.9"
# # MODEL_PARAMS_FEDOPT="tau:0.001,0.01"
# # MODEL_PARAMS_QFEDAVG="q_param:0.1,0.5,1.0;qffl_learning_rate:0.01,0.1"

# # declare -a CLIENTS=(25)
# # declare -a ALPHAS=(0.5)

# # # Default values
# # DEFAULT_ROUNDS=10
# # DEFAULT_MIN_CLIENTS=2

# # # Parse command line arguments
# # PARALLEL=false
# # CLEAN=false
# # QUICK_TEST=false

# # # Get all parameter combinations for a strategy
# # get_param_combinations() {
# #     local strategy=$1
# #     local params_string=""
    
# #     # Get parameters for this strategy
# #     case "$strategy" in
# #         "fedavg")
# #             params_string="$MODEL_PARAMS_FEDAVG"
# #             ;;
# #         "fedavgm")
# #             params_string="$MODEL_PARAMS_FEDAVGM"
# #             ;;
# #         "fedopt")
# #             params_string="$MODEL_PARAMS_FEDOPT"
# #             ;;
# #         "qfedavg")
# #             params_string="$MODEL_PARAMS_QFEDAVG"
# #             ;;
# #     esac
    
# #     # If no parameters, return "default"
# #     if [ -z "$params_string" ]; then
# #         echo "default"
# #         return
# #     fi
    
# #     # Split by semicolon to get individual parameters
# #     IFS=';' read -ra PARAMS <<< "$params_string"
    
# #     if [ ${#PARAMS[@]} -eq 0 ]; then
# #         echo "default"
# #         return
# #     fi
    
# #     # Single parameter with multiple values
# #     if [ ${#PARAMS[@]} -eq 1 ]; then
# #         local param="${PARAMS[0]}"
# #         IFS=':' read -r pname pvalues <<< "$param"
# #         IFS=',' read -ra values <<< "$pvalues"
        
# #         for val in "${values[@]}"; do
# #             echo "${pname}=${val}"
# #         done
# #         return
# #     fi
    
# #     # Two parameters - cartesian product
# #     if [ ${#PARAMS[@]} -eq 2 ]; then
# #         local param1="${PARAMS[0]}"
# #         local param2="${PARAMS[1]}"
        
# #         IFS=':' read -r pname1 pvalues1 <<< "$param1"
# #         IFS=':' read -r pname2 pvalues2 <<< "$param2"
        
# #         IFS=',' read -ra values1 <<< "$pvalues1"
# #         IFS=',' read -ra values2 <<< "$pvalues2"
        
# #         for val1 in "${values1[@]}"; do
# #             for val2 in "${values2[@]}"; do
# #                 echo "${pname1}=${val1},${pname2}=${val2}"
# #             done
# #         done
# #         return
# #     fi
    
# #     # Fallback
# #     echo "default"
# # }

# # usage() {
# #     echo "Usage: $0 [OPTIONS]"
# #     echo "Options:"
# #     echo "  --parallel          Run experiments in parallel"
# #     echo "  --clean             Clean previous results before running"
# #     echo "  --quick-test        Run a quick test with minimal parameters"
# #     echo "  --help              Show this help message"
# #     echo ""
# #     echo "Example:"
# #     echo "  $0 --parallel --clean    # Run all experiments in parallel, clean previous results"
# #     echo "  $0 --quick-test          # Run a single quick test"
# # }

# # while [[ $# -gt 0 ]]; do
# #     case $1 in
# #         --parallel)
# #             PARALLEL=true
# #             shift
# #             ;;
# #         --clean)
# #             CLEAN=true
# #             shift
# #             ;;
# #         --quick-test)
# #             QUICK_TEST=true
# #             shift
# #             ;;
# #         --help)
# #             usage
# #             exit 0
# #             ;;
# #         *)
# #             echo "Unknown option: $1"
# #             usage
# #             exit 1
# #             ;;
# #     esac
# # done

# # # Create config template if it doesn't exist
# # create_config_template() {
# #     cat > "$CONFIG_TEMPLATE" << 'EOF'
# # {
# #   "experiment_id": "PLACEHOLDER_ID",
# #   "algorithm": "PLACEHOLDER_ALGORITHM",
# #   "description": "NASA CMAPs FL Experiment",
# #   "run_id": "PLACEHOLDER_RUN_ID",
# #   "timestamp": "PLACEHOLDER_TIMESTAMP",
# #   "server": {
# #     "host": "PLACEHOLDER_HOST",
# #     "port": PLACEHOLDER_PORT,
# #     "num_rounds": PLACEHOLDER_ROUNDS
# #   },
# #   "strategy": {
# #     "name": "PLACEHOLDER_ALGORITHM",
# #     "fraction_fit": 1.0,
# #     "fraction_evaluate": 1.0,
# #     "min_fit_clients": PLACEHOLDER_MIN_CLIENTS,
# #     "min_evaluate_clients": PLACEHOLDER_MIN_CLIENTS,
# #     "min_available_clients": PLACEHOLDER_TOTAL_CLIENTS,
# #     "params": {}
# #   },
# #   "data": {
# #     "base_path": "PLACEHOLDER_DATA_PATH",
# #     "num_clients": PLACEHOLDER_TOTAL_CLIENTS,
# #     "alpha": PLACEHOLDER_ALPHA
# #   },
# #   "model": {
# #     "model_type": "dense",
# #     "learning_rate": 0.001,
# #     "local_epochs": 1,
# #     "batch_size": 32,
# #     "hidden_dims": [64, 32],
# #     "dropout": 0.2,
# #     "n_components": 10
# #   }
# # }
# # EOF
# #     echo "✅ Created config template at $CONFIG_TEMPLATE"
# # }

# # # Initialize experiment runner
# # initialize_runner() {
# #     echo "🧪 NASA FL Experiments Runner"
# #     echo "=============================="
# #     echo "Run ID: $RUN_ID"
# #     echo "Timestamp: $(date)"
# #     echo "Base Directory: $BASE_DIR"
# #     echo "Results Base: $RESULTS_BASE"
# #     echo "Mode: $([ "$PARALLEL" = true ] && echo "Parallel" || echo "Sequential")"
# #     echo "Quick Test: $([ "$QUICK_TEST" = true ] && echo "Yes" || echo "No")"
# #     echo ""
    
# #     if [ ! -f "$SERVER_SCRIPT" ]; then
# #         echo "❌ Server script not found: $SERVER_SCRIPT"
# #         exit 1
# #     fi
    
# #     if [ ! -f "$CLIENT_SCRIPT" ]; then
# #         echo "❌ Client script not found: $CLIENT_SCRIPT"
# #         exit 1
# #     fi
    
# #     if [ ! -f "$CONFIG_TEMPLATE" ]; then
# #         create_config_template
# #     fi
    
# #     if [ "$QUICK_TEST" = true ]; then
# #         STRATEGIES=("fedavg")
# #         CLIENTS=(5)
# #         ALPHAS=(0.5)
# #         DEFAULT_ROUNDS=3
# #         MODEL_PARAMS_FEDAVG=""
# #         MODEL_PARAMS_FEDAVGM=""
# #         MODEL_PARAMS_FEDOPT=""
# #         MODEL_PARAMS_QFEDAVG=""
# #         echo "🔬 Quick test mode activated"
# #         echo "   Strategies: ${STRATEGIES[@]}"
# #         echo "   Clients: ${CLIENTS[@]}"
# #         echo "   Alphas: ${ALPHAS[@]}"
# #         echo "   Rounds: $DEFAULT_ROUNDS"
# #     else
# #         echo "🔬 Full experiment mode"
# #         echo "   Strategies: ${STRATEGIES[@]}"
# #         echo "   Clients: ${CLIENTS[@]}"
# #         echo "   Alphas: ${ALPHAS[@]}"
# #         echo "   Rounds: $DEFAULT_ROUNDS"
        
# #         # Show what will be tested
# #         echo ""
# #         echo "📋 Experiment Plan:"
# #         for strategy in "${STRATEGIES[@]}"; do
# #             local combos=($(get_param_combinations "$strategy"))
# #             echo "   ${strategy^^}:"
# #             for combo in "${combos[@]}"; do
# #                 if [ "$combo" = "default" ]; then
# #                     echo "      - No parameters (default)"
# #                 else
# #                     echo "      - $combo"
# #                 fi
# #             done
# #         done
# #     fi
# #     echo ""
# # }

# # clean_results() {
# #     if [ -d "$RESULTS_BASE" ]; then
# #         echo "🧹 Cleaning previous results from $RESULTS_BASE..."
# #         rm -rf "$RESULTS_BASE"
# #     fi
# #     mkdir -p "$RESULTS_BASE"
# # }

# # create_run_directory() {
# #     local run_dir="$RESULTS_BASE/$RUN_ID"
# #     mkdir -p "$run_dir"
# #     echo "$run_dir"
# # }

# # wait_for_port() {
# #     local port=$1
# #     local max_attempts=30
# #     local attempt=0
    
# #     while netstat -tuln 2>/dev/null | grep ":$port " > /dev/null; do
# #         attempt=$((attempt + 1))
# #         if [ $attempt -ge $max_attempts ]; then
# #             echo "❌ Port $port still not available after $max_attempts attempts"
# #             return 1
# #         fi
# #         echo "⏳ Waiting for port $port to be available... ($attempt/$max_attempts)"
# #         sleep 2
# #     done
# #     return 0
# # }

# # generate_experiment_config() {
# #     local strategy=$1
# #     local num_clients=$2
# #     local alpha=$3
# #     local port=$4
# #     local config_dir=$5
# #     local param_combo=${6:-"default"}
    
# #     local param_suffix=""
# #     if [ "$param_combo" != "default" ]; then
# #         param_suffix="_${param_combo//[=,]/_}"
# #     fi
    
# #     local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}${param_suffix}"
# #     local config_file="$config_dir/config.json"
    
# #     local min_clients=$((num_clients / 2))
# #     if [ $min_clients -lt $DEFAULT_MIN_CLIENTS ]; then
# #         min_clients=$DEFAULT_MIN_CLIENTS
# #     fi
# #     if [ $min_clients -gt $num_clients ]; then
# #         min_clients=$num_clients
# #     fi
    
# #     mkdir -p "$config_dir"
    
# #     sed \
# #         -e "s/PLACEHOLDER_ID/$exp_id/g" \
# #         -e "s/PLACEHOLDER_ALGORITHM/$strategy/g" \
# #         -e "s/PLACEHOLDER_RUN_ID/$RUN_ID/g" \
# #         -e "s/PLACEHOLDER_TIMESTAMP/$TIMESTAMP/g" \
# #         -e "s/PLACEHOLDER_HOST/$SERVER_HOST/g" \
# #         -e "s/PLACEHOLDER_PORT/$port/g" \
# #         -e "s/PLACEHOLDER_ROUNDS/$DEFAULT_ROUNDS/g" \
# #         -e "s/PLACEHOLDER_MIN_CLIENTS/$min_clients/g" \
# #         -e "s/PLACEHOLDER_TOTAL_CLIENTS/$num_clients/g" \
# #         -e "s/PLACEHOLDER_ALPHA/$alpha/g" \
# #         -e "s|PLACEHOLDER_DATA_PATH|$BASE_DIR/data/nasa_cmaps/pre_split_data|g" \
# #         "$CONFIG_TEMPLATE" > "$config_file"
    
# #     if [ "$param_combo" != "default" ]; then
# #         python3 - <<EOF "$config_file" "$param_combo"
# # import json
# # import sys

# # config_file = sys.argv[1]
# # params_str = sys.argv[2]

# # with open(config_file, 'r') as f:
# #     config = json.load(f)

# # if 'strategy' not in config:
# #     config['strategy'] = {}
# # if 'params' not in config['strategy']:
# #     config['strategy']['params'] = {}

# # for param in params_str.split(','):
# #     key, val = param.split('=')
# #     try:
# #         config['strategy']['params'][key] = float(val)
# #     except ValueError:
# #         config['strategy']['params'][key] = val

# # with open(config_file, 'w') as f:
# #     json.dump(config, f, indent=2)
# # EOF
# #     fi
    
# #     echo "$config_file"
# # }

# # run_experiment() {
# #     local strategy=$1
# #     local num_clients=$2
# #     local alpha=$3
# #     local port=$4
# #     local run_dir=$5
# #     local param_combo=${6:-"default"}
    
# #     local param_suffix=""
# #     if [ "$param_combo" != "default" ]; then
# #         param_suffix="_${param_combo//[=,]/_}"
# #     fi
    
# #     local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}${param_suffix}"
# #     local results_dir="$run_dir/$exp_id"
    
# #     echo "🚀 Starting experiment: $exp_id"
# #     echo "   Strategy: $strategy"
# #     if [ "$param_combo" != "default" ]; then
# #         echo "   Parameters: $param_combo"
# #     fi
# #     echo "   Clients: $num_clients"
# #     echo "   Alpha: $alpha"
# #     echo "   Port: $port"
# #     echo "   Results: $results_dir"
    
# #     mkdir -p "$results_dir"
# #     mkdir -p "$results_dir/logs"
# #     mkdir -p "$results_dir/metrics"
    
# #     local config_file=$(generate_experiment_config "$strategy" "$num_clients" "$alpha" "$port" "$results_dir" "$param_combo")
    
# #     echo "   Config: $config_file"
    
# #     if [ ! -f "$config_file" ]; then
# #         echo "❌ Config file not created: $config_file"
# #         return 1
# #     fi
    
# #     local server_log="$results_dir/logs/server_$(date +%H%M%S).log"
# #     echo "   Starting server... Log: $server_log"
    
# #     python3 "$SERVER_SCRIPT" --config "$config_file" --results-dir "$results_dir" > "$server_log" 2>&1 &
# #     local server_pid=$!
    
# #     echo "   Waiting for server to initialize..."
# #     sleep 10
    
# #     if ! kill -0 $server_pid 2>/dev/null; then
# #         echo "❌ Server failed to start. Check log: $server_log"
# #         echo "   Last 30 lines of server log:"
# #         tail -30 "$server_log"
# #         return 1
# #     fi
    
# #     echo "   Starting $num_clients clients..."
# #     local client_pids=()
    
# #     for ((i=0; i<num_clients; i++)); do
# #         local client_log="$results_dir/logs/client_${i}_$(date +%H%M%S).log"
# #         python3 "$CLIENT_SCRIPT" --client-id "client_$i" --config "$config_file" > "$client_log" 2>&1 &
# #         client_pids+=($!)
# #         sleep 2
# #     done
    
# #     echo "   Experiment running... (Server PID: $server_pid)"
# #     echo "   Waiting for server to complete..."
    
# #     wait $server_pid
# #     local server_exit_code=$?
    
# #     sleep 5
    
# #     for pid in "${client_pids[@]}"; do
# #         if kill -0 "$pid" 2>/dev/null; then
# #             kill "$pid" 2>/dev/null || true
# #         fi
# #     done
    
# #     if [ $server_exit_code -eq 0 ] && [ -f "$results_dir/metrics/round_metrics.csv" ]; then
# #         echo "✅ Experiment $exp_id completed successfully"
# #         return 0
# #     else
# #         echo "❌ Experiment $exp_id failed with exit code $server_exit_code"
# #         if [ -f "$server_log" ]; then
# #             echo "   Last 20 server log lines:"
# #             tail -20 "$server_log"
# #         fi
# #         return 1
# #     fi
# # }

# # # FIXED: Sequential experiments with proper iteration
# # run_sequential_experiments() {
# #     local run_dir=$1
# #     local current_port=$BASE_PORT
# #     local total_experiments=0
    
# #     # Calculate total experiments correctly
# #     for strategy in "${STRATEGIES[@]}"; do
# #         readarray -t param_combos < <(get_param_combinations "$strategy")
# #         total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
# #     done
    
# #     local completed=0
# #     local failed=0
    
# #     echo "➡️  Running $total_experiments experiments sequentially..."
# #     echo ""
    
# #     # FIXED: Proper nested loop iteration
# #     for strategy in "${STRATEGIES[@]}"; do
# #         echo ""
# #         echo "🔷 Starting strategy: ${strategy^^}"
# #         echo "========================================" 
        
# #         # Get param combinations as array
# #         readarray -t param_combos < <(get_param_combinations "$strategy")
        
# #         echo "   Found ${#param_combos[@]} parameter combination(s)"
        
# #         for param_combo in "${param_combos[@]}"; do
# #             for num_clients in "${CLIENTS[@]}"; do
# #                 for alpha in "${ALPHAS[@]}"; do
# #                     ((completed++))
# #                     echo ""
# #                     echo "========================================"
# #                     echo "📊 Progress: $completed/$total_experiments"
# #                     echo "🔧 Configuration:"
# #                     echo "   Strategy: $strategy"
# #                     echo "   Parameters: $param_combo"
# #                     echo "   Clients: $num_clients"
# #                     echo "   Alpha: $alpha"
# #                     echo "   Port: $current_port"
# #                     echo "========================================"
                    
# #                     if wait_for_port $current_port; then
# #                         if run_experiment "$strategy" "$num_clients" "$alpha" $current_port "$run_dir" "$param_combo"; then
# #                             echo "✅ Experiment $completed/$total_experiments completed"
# #                         else
# #                             echo "❌ Experiment $completed/$total_experiments failed"
# #                             ((failed++))
# #                         fi
# #                     else
# #                         echo "❌ Port $current_port unavailable"
# #                         ((failed++))
# #                     fi
                    
# #                     current_port=$((current_port + 1))
# #                     echo "========================================"
# #                     echo ""
                    
# #                     sleep 5
# #                 done
# #             done
# #         done
        
# #         echo "✓ Completed all experiments for ${strategy^^}"
# #         echo ""
# #     done
    
# #     echo ""
# #     echo "🎉 Sequential experiments completed!"
# #     echo "   Successful: $((completed - failed))/$completed"
# #     echo "   Failed: $failed"
# #     return $failed
# # }

# # # Generate comprehensive summary
# # generate_summary() {
# #     local run_dir=$1
# #     local summary_file="$run_dir/experiment_summary.md"
    
# #     local total_experiments=0
# #     for strategy in "${STRATEGIES[@]}"; do
# #         readarray -t param_combos < <(get_param_combinations "$strategy")
# #         total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
# #     done
    
# #     local completed_count=$(find "$run_dir" -name "round_metrics.csv" 2>/dev/null | wc -l)
    
# #     cat > "$summary_file" << EOF
# # # NASA FL Experiments Summary

# # ## Run Information
# # - **Run ID**: $RUN_ID
# # - **Timestamp**: $(date)
# # - **Total Experiments**: $total_experiments
# # - **Completed**: $completed_count
# # - **Success Rate**: $((completed_count > 0 ? completed_count * 100 / total_experiments : 0))%
# # - **Run Directory**: \`$run_dir\`

# # ## Experiment Parameters
# # - **Strategies**: ${STRATEGIES[@]}
# # - **Client Counts**: ${CLIENTS[@]}
# # - **Alpha Values**: ${ALPHAS[@]}
# # - **Server Rounds**: $DEFAULT_ROUNDS

# # ## Model-Specific Parameters
# # - **FedAvg**: ${MODEL_PARAMS_FEDAVG:-"None (default)"}
# # - **FedAvgM**: ${MODEL_PARAMS_FEDAVGM:-"None"}
# # - **FedOpt**: ${MODEL_PARAMS_FEDOPT:-"None"}
# # - **QFedAvg**: ${MODEL_PARAMS_QFEDAVG:-"None"}

# # ## Completed Experiments
# # \`\`\`
# # $(find "$run_dir" -type d -name "nasa_*" -exec basename {} \; | sort)
# # \`\`\`

# # ## Quick Analysis Commands

# # ### List all experiments
# # \`\`\`bash
# # ls -1 "$run_dir"/nasa_*
# # \`\`\`

# # ### Check completion
# # \`\`\`bash
# # find "$run_dir" -name "round_metrics.csv" | wc -l
# # \`\`\`

# # ### View final results
# # \`\`\`bash
# # for dir in "$run_dir"/nasa_*/metrics/; do
# #     echo "=== \$(basename \$(dirname \$dir)) ==="
# #     tail -1 "\$dir/round_metrics.csv" 2>/dev/null || echo "No results"
# # done
# # \`\`\`

# # Generated on: $(date)
# # EOF

# #     echo "📊 Summary generated: $summary_file"
# #     echo "📈 Completion: $completed_count/$total_experiments experiments"
# # }

# # # Main execution
# # main() {
# #     initialize_runner
    
# #     if [ "$CLEAN" = true ]; then
# #         clean_results
# #     fi
    
# #     local run_dir=$(create_run_directory)
# #     echo "📁 Run directory: $run_dir"
# #     echo ""
    
# #     local start_time=$(date +%s)
    
# #     local failed_experiments=0
# #     if [ "$PARALLEL" = true ]; then
# #         echo "⚠️  Parallel mode not fully implemented for parameter combinations"
# #         echo "   Running sequentially instead..."
# #         run_sequential_experiments "$run_dir" || failed_experiments=$?
# #     else
# #         run_sequential_experiments "$run_dir" || failed_experiments=$?
# #     fi
    
# #     local end_time=$(date +%s)
# #     local duration=$((end_time - start_time))
# #     local hours=$((duration / 3600))
# #     local minutes=$(((duration % 3600) / 60))
# #     local seconds=$((duration % 60))
    
# #     generate_summary "$run_dir"
    
# #     echo ""
# #     echo "========================================"
# #     echo "🎉 Experiment Run Completed!"
# #     echo "========================================"
# #     echo "Run ID: $RUN_ID"
# #     echo "Duration: ${hours}h ${minutes}m ${seconds}s"
# #     echo "Results: $run_dir"
# #     echo "Summary: $run_dir/experiment_summary.md"
# #     echo ""
    
# #     if [ $failed_experiments -gt 0 ]; then
# #         echo "⚠️  $failed_experiments experiments failed. Check the logs."
# #         echo "========================================"
# #         exit 1
# #     else
# #         echo "✅ All experiments completed successfully!"
# #         echo "========================================"
# #         exit 0
# #     fi
# # }

# # main "$@"

# #!/bin/bash

# set -e  # Exit on any error

# # Configuration
# BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# RESULTS_BASE="$BASE_DIR/results"
# SERVER_SCRIPT="$BASE_DIR/server.py"
# CLIENT_SCRIPT="$BASE_DIR/client.py"
# CONFIG_TEMPLATE="$BASE_DIR/config_template.json"
# SERVER_HOST="localhost"
# BASE_PORT=8686

# # Timestamp for all experiments in this run
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# RUN_ID="run_${TIMESTAMP}"

# # Experiment parameters
# declare -a STRATEGIES=("fedavg" "fedavgm" "fedopt" "qfedavg")

# MODEL_PARAMS_FEDAVG=""
# MODEL_PARAMS_FEDAVGM="server_momentum:0.5,0.9"
# MODEL_PARAMS_FEDOPT="tau:0.001,0.01"
# MODEL_PARAMS_QFEDAVG="q_param:0.1,0.5,1.0;qffl_learning_rate:0.01,0.1"

# declare -a CLIENTS=(25)
# declare -a ALPHAS=(0.5)

# # Default values
# DEFAULT_ROUNDS=10
# DEFAULT_MIN_CLIENTS=2
# EXPERIMENT_TIMEOUT=1800  # 30 minutes per experiment

# # Parse command line arguments
# PARALLEL=false
# CLEAN=false
# QUICK_TEST=false

# get_param_combinations() {
#     local strategy=$1
#     local params_string=""
    
#     case "$strategy" in
#         "fedavg") params_string="$MODEL_PARAMS_FEDAVG" ;;
#         "fedavgm") params_string="$MODEL_PARAMS_FEDAVGM" ;;
#         "fedopt") params_string="$MODEL_PARAMS_FEDOPT" ;;
#         "qfedavg") params_string="$MODEL_PARAMS_QFEDAVG" ;;
#     esac
    
#     if [ -z "$params_string" ]; then
#         echo "default"
#         return
#     fi
    
#     IFS=';' read -ra PARAMS <<< "$params_string"
    
#     if [ ${#PARAMS[@]} -eq 0 ]; then
#         echo "default"
#         return
#     fi
    
#     if [ ${#PARAMS[@]} -eq 1 ]; then
#         local param="${PARAMS[0]}"
#         IFS=':' read -r pname pvalues <<< "$param"
#         IFS=',' read -ra values <<< "$pvalues"
        
#         for val in "${values[@]}"; do
#             echo "${pname}=${val}"
#         done
#         return
#     fi
    
#     if [ ${#PARAMS[@]} -eq 2 ]; then
#         local param1="${PARAMS[0]}"
#         local param2="${PARAMS[1]}"
        
#         IFS=':' read -r pname1 pvalues1 <<< "$param1"
#         IFS=':' read -r pname2 pvalues2 <<< "$param2"
        
#         IFS=',' read -ra values1 <<< "$pvalues1"
#         IFS=',' read -ra values2 <<< "$pvalues2"
        
#         for val1 in "${values1[@]}"; do
#             for val2 in "${values2[@]}"; do
#                 echo "${pname1}=${val1},${pname2}=${val2}"
#             done
#         done
#         return
#     fi
    
#     echo "default"
# }

# usage() {
#     echo "Usage: $0 [OPTIONS]"
#     echo "Options:"
#     echo "  --parallel          Run experiments in parallel"
#     echo "  --clean             Clean previous results before running"
#     echo "  --quick-test        Run a quick test with minimal parameters"
#     echo "  --help              Show this help message"
# }

# while [[ $# -gt 0 ]]; do
#     case $1 in
#         --parallel) PARALLEL=true; shift ;;
#         --clean) CLEAN=true; shift ;;
#         --quick-test) QUICK_TEST=true; shift ;;
#         --help) usage; exit 0 ;;
#         *) echo "Unknown option: $1"; usage; exit 1 ;;
#     esac
# done

# create_config_template() {
#     cat > "$CONFIG_TEMPLATE" << 'EOF'
# {
#   "experiment_id": "PLACEHOLDER_ID",
#   "algorithm": "PLACEHOLDER_ALGORITHM",
#   "description": "NASA CMAPs FL Experiment",
#   "run_id": "PLACEHOLDER_RUN_ID",
#   "timestamp": "PLACEHOLDER_TIMESTAMP",
#   "server": {
#     "host": "PLACEHOLDER_HOST",
#     "port": PLACEHOLDER_PORT,
#     "num_rounds": PLACEHOLDER_ROUNDS
#   },
#   "strategy": {
#     "name": "PLACEHOLDER_ALGORITHM",
#     "fraction_fit": 1.0,
#     "fraction_evaluate": 1.0,
#     "min_fit_clients": PLACEHOLDER_MIN_CLIENTS,
#     "min_evaluate_clients": PLACEHOLDER_MIN_CLIENTS,
#     "min_available_clients": PLACEHOLDER_TOTAL_CLIENTS,
#     "params": {}
#   },
#   "data": {
#     "base_path": "PLACEHOLDER_DATA_PATH",
#     "num_clients": PLACEHOLDER_TOTAL_CLIENTS,
#     "alpha": PLACEHOLDER_ALPHA
#   },
#   "model": {
#     "model_type": "dense",
#     "learning_rate": 0.001,
#     "local_epochs": 1,
#     "batch_size": 32,
#     "hidden_dims": [64, 32],
#     "dropout": 0.2,
#     "n_components": 10
#   }
# }
# EOF
#     echo "✅ Created config template at $CONFIG_TEMPLATE"
# }

# initialize_runner() {
#     echo "🧪 NASA FL Experiments Runner"
#     echo "=============================="
#     echo "Run ID: $RUN_ID"
#     echo "Timestamp: $(date)"
#     echo "Timeout per experiment: ${EXPERIMENT_TIMEOUT}s"
#     echo ""
    
#     if [ ! -f "$SERVER_SCRIPT" ] || [ ! -f "$CLIENT_SCRIPT" ]; then
#         echo "❌ Required scripts not found"
#         exit 1
#     fi
    
#     if [ ! -f "$CONFIG_TEMPLATE" ]; then
#         create_config_template
#     fi
    
#     if [ "$QUICK_TEST" = true ]; then
#         STRATEGIES=("fedavg")
#         CLIENTS=(5)
#         ALPHAS=(0.5)
#         DEFAULT_ROUNDS=3
#         EXPERIMENT_TIMEOUT=300
#         MODEL_PARAMS_FEDAVG=""
#         MODEL_PARAMS_FEDAVGM=""
#         MODEL_PARAMS_FEDOPT=""
#         MODEL_PARAMS_QFEDAVG=""
#         echo "🔬 Quick test mode: 3 rounds, 5 clients, 5min timeout"
#     else
#         echo "🔬 Full experiment mode"
#         echo "   Strategies: ${STRATEGIES[@]}"
#         echo "   Clients: ${CLIENTS[@]}"
#         echo "   Alphas: ${ALPHAS[@]}"
#         echo "   Rounds: $DEFAULT_ROUNDS"
        
#         echo ""
#         echo "📋 Experiment Plan:"
#         for strategy in "${STRATEGIES[@]}"; do
#             readarray -t combos < <(get_param_combinations "$strategy")
#             echo "   ${strategy^^}: ${#combos[@]} configuration(s)"
#         done
#     fi
#     echo ""
# }

# clean_results() {
#     if [ -d "$RESULTS_BASE" ]; then
#         echo "🧹 Cleaning previous results..."
#         rm -rf "$RESULTS_BASE"
#     fi
#     mkdir -p "$RESULTS_BASE"
# }

# create_run_directory() {
#     local run_dir="$RESULTS_BASE/$RUN_ID"
#     mkdir -p "$run_dir"
#     echo "$run_dir"
# }

# wait_for_port() {
#     local port=$1
#     local max_attempts=30
#     local attempt=0
    
#     while netstat -tuln 2>/dev/null | grep ":$port " > /dev/null; do
#         attempt=$((attempt + 1))
#         if [ $attempt -ge $max_attempts ]; then
#             echo "❌ Port $port still not available after $max_attempts attempts"
#             # Force kill anything on this port
#             lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
#             sleep 2
#             return 1
#         fi
#         echo "⏳ Waiting for port $port... ($attempt/$max_attempts)"
#         sleep 2
#     done
#     return 0
# }

# generate_experiment_config() {
#     local strategy=$1
#     local num_clients=$2
#     local alpha=$3
#     local port=$4
#     local config_dir=$5
#     local param_combo=${6:-"default"}
    
#     local param_suffix=""
#     if [ "$param_combo" != "default" ]; then
#         param_suffix="_${param_combo//[=,]/_}"
#     fi
    
#     local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}${param_suffix}"
#     local config_file="$config_dir/config.json"
    
#     local min_clients=$((num_clients / 2))
#     [ $min_clients -lt $DEFAULT_MIN_CLIENTS ] && min_clients=$DEFAULT_MIN_CLIENTS
#     [ $min_clients -gt $num_clients ] && min_clients=$num_clients
    
#     mkdir -p "$config_dir"
    
#     sed \
#         -e "s/PLACEHOLDER_ID/$exp_id/g" \
#         -e "s/PLACEHOLDER_ALGORITHM/$strategy/g" \
#         -e "s/PLACEHOLDER_RUN_ID/$RUN_ID/g" \
#         -e "s/PLACEHOLDER_TIMESTAMP/$TIMESTAMP/g" \
#         -e "s/PLACEHOLDER_HOST/$SERVER_HOST/g" \
#         -e "s/PLACEHOLDER_PORT/$port/g" \
#         -e "s/PLACEHOLDER_ROUNDS/$DEFAULT_ROUNDS/g" \
#         -e "s/PLACEHOLDER_MIN_CLIENTS/$min_clients/g" \
#         -e "s/PLACEHOLDER_TOTAL_CLIENTS/$num_clients/g" \
#         -e "s/PLACEHOLDER_ALPHA/$alpha/g" \
#         -e "s|PLACEHOLDER_DATA_PATH|$BASE_DIR/data/nasa_cmaps/pre_split_data|g" \
#         "$CONFIG_TEMPLATE" > "$config_file"
    
#     if [ "$param_combo" != "default" ]; then
#         python3 - "$config_file" "$param_combo" <<'PYTHON'
# import json
# import sys

# config_file = sys.argv[1]
# params_str = sys.argv[2]

# with open(config_file, 'r') as f:
#     config = json.load(f)

# if 'strategy' not in config:
#     config['strategy'] = {}
# if 'params' not in config['strategy']:
#     config['strategy']['params'] = {}

# for param in params_str.split(','):
#     key, val = param.split('=')
#     try:
#         config['strategy']['params'][key] = float(val)
#     except ValueError:
#         config['strategy']['params'][key] = val

# with open(config_file, 'w') as f:
#     json.dump(config, f, indent=2)
# PYTHON
#     fi
    
#     echo "$config_file"
# }

# # FIXED: Kill process tree function
# kill_process_tree() {
#     local pid=$1
#     local children=$(pgrep -P $pid 2>/dev/null || true)
    
#     for child in $children; do
#         kill_process_tree $child
#     done
    
#     if kill -0 $pid 2>/dev/null; then
#         kill -TERM $pid 2>/dev/null || true
#         sleep 1
#         if kill -0 $pid 2>/dev/null; then
#             kill -KILL $pid 2>/dev/null || true
#         fi
#     fi
# }

# # FIXED: Run experiment with timeout
# run_experiment() {
#     local strategy=$1
#     local num_clients=$2
#     local alpha=$3
#     local port=$4
#     local run_dir=$5
#     local param_combo=${6:-"default"}
    
#     local param_suffix=""
#     [ "$param_combo" != "default" ] && param_suffix="_${param_combo//[=,]/_}"
    
#     local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}${param_suffix}"
#     local results_dir="$run_dir/$exp_id"
    
#     echo "🚀 Starting experiment: $exp_id"
#     echo "   Strategy: $strategy | Params: $param_combo"
#     echo "   Clients: $num_clients | Alpha: $alpha | Port: $port"
    
#     mkdir -p "$results_dir/logs" "$results_dir/metrics"
    
#     local config_file=$(generate_experiment_config "$strategy" "$num_clients" "$alpha" "$port" "$results_dir" "$param_combo")
    
#     if [ ! -f "$config_file" ]; then
#         echo "❌ Config file not created"
#         return 1
#     fi
    
#     local server_log="$results_dir/logs/server_$(date +%H%M%S).log"
#     echo "   Server log: $server_log"
    
#     # Start server
#     python3 "$SERVER_SCRIPT" --config "$config_file" --results-dir "$results_dir" > "$server_log" 2>&1 &
#     local server_pid=$!
    
#     sleep 10
    
#     if ! kill -0 $server_pid 2>/dev/null; then
#         echo "❌ Server failed to start"
#         tail -20 "$server_log"
#         return 1
#     fi
    
#     echo "   Starting $num_clients clients..."
#     local client_pids=()
    
#     for ((i=0; i<num_clients; i++)); do
#         local client_log="$results_dir/logs/client_${i}_$(date +%H%M%S).log"
#         python3 "$CLIENT_SCRIPT" --client-id "client_$i" --config "$config_file" > "$client_log" 2>&1 &
#         client_pids+=($!)
#         sleep 1
#     done
    
#     echo "   Server PID: $server_pid | Timeout: ${EXPERIMENT_TIMEOUT}s"
    
#     # FIXED: Wait with timeout
#     local elapsed=0
#     local check_interval=10
    
#     while kill -0 $server_pid 2>/dev/null; do
#         sleep $check_interval
#         elapsed=$((elapsed + check_interval))
        
#         if [ $elapsed -ge $EXPERIMENT_TIMEOUT ]; then
#             echo "⏰ Experiment timeout after ${elapsed}s"
#             kill_process_tree $server_pid
#             for pid in "${client_pids[@]}"; do
#                 kill_process_tree $pid
#             done
#             echo "❌ Timeout - killed all processes"
#             return 1
#         fi
        
#         # Show progress every 60 seconds
#         if [ $((elapsed % 60)) -eq 0 ]; then
#             echo "   Running for ${elapsed}s..."
#         fi
#     done
    
#     # Server finished naturally
#     wait $server_pid 2>/dev/null
#     local server_exit_code=$?
    
#     echo "   Server finished (exit: $server_exit_code) after ${elapsed}s"
    
#     # Clean up clients
#     sleep 3
#     for pid in "${client_pids[@]}"; do
#         kill_process_tree $pid 2>/dev/null || true
#     done
    
#     # Verify results
#     if [ $server_exit_code -eq 0 ] && [ -f "$results_dir/metrics/round_metrics.csv" ]; then
#         local lines=$(wc -l < "$results_dir/metrics/round_metrics.csv")
#         echo "✅ Success - $((lines - 1)) rounds completed"
#         return 0
#     else
#         echo "❌ Failed (exit: $server_exit_code)"
#         [ -f "$server_log" ] && tail -10 "$server_log"
#         return 1
#     fi
# }

# run_sequential_experiments() {
#     local run_dir=$1
#     local current_port=$BASE_PORT
#     local total_experiments=0
    
#     for strategy in "${STRATEGIES[@]}"; do
#         readarray -t param_combos < <(get_param_combinations "$strategy")
#         total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
#     done
    
#     local completed=0
#     local failed=0
    
#     echo "➡️  Running $total_experiments experiments sequentially"
#     echo ""
    
#     for strategy in "${STRATEGIES[@]}"; do
#         echo ""
#         echo "🔷 Strategy: ${strategy^^}"
#         echo "========================================" 
        
#         readarray -t param_combos < <(get_param_combinations "$strategy")
        
#         for param_combo in "${param_combos[@]}"; do
#             for num_clients in "${CLIENTS[@]}"; do
#                 for alpha in "${ALPHAS[@]}"; do
#                     ((completed++))
#                     echo ""
#                     echo "📊 Experiment $completed/$total_experiments"
#                     echo "========================================"
                    
#                     if ! wait_for_port $current_port; then
#                         echo "⚠️  Port cleanup failed, continuing..."
#                     fi
                    
#                     if run_experiment "$strategy" "$num_clients" "$alpha" $current_port "$run_dir" "$param_combo"; then
#                         echo "✅ Completed"
#                     else
#                         echo "❌ Failed"
#                         ((failed++))
#                     fi
                    
#                     current_port=$((current_port + 1))
#                     echo "========================================"
                    
#                     # Brief pause between experiments
#                     sleep 5
#                 done
#             done
#         done
#     done
    
#     echo ""
#     echo "🎉 All experiments completed"
#     echo "   Success: $((completed - failed))/$completed"
#     echo "   Failed: $failed"
#     return $failed
# }

# generate_summary() {
#     local run_dir=$1
#     local summary_file="$run_dir/experiment_summary.md"
    
#     local total_experiments=0
#     for strategy in "${STRATEGIES[@]}"; do
#         readarray -t param_combos < <(get_param_combinations "$strategy")
#         total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
#     done
    
#     local completed_count=$(find "$run_dir" -name "round_metrics.csv" 2>/dev/null | wc -l)
    
#     cat > "$summary_file" << EOF
# # NASA FL Experiments Summary

# **Run ID**: $RUN_ID  
# **Date**: $(date)  
# **Total**: $total_experiments experiments  
# **Completed**: $completed_count ($((completed_count * 100 / total_experiments))%)

# ## Configurations
# - Strategies: ${STRATEGIES[@]}
# - Clients: ${CLIENTS[@]}
# - Alphas: ${ALPHAS[@]}
# - Rounds: $DEFAULT_ROUNDS

# ## Results
# \`\`\`bash
# find "$run_dir" -name "round_metrics.csv" -exec dirname {} \;
# \`\`\`
# EOF

#     echo "📊 Summary: $summary_file"
# }

# main() {
#     initialize_runner
    
#     [ "$CLEAN" = true ] && clean_results
    
#     local run_dir=$(create_run_directory)
#     echo "📁 Results: $run_dir"
#     echo ""
    
#     local start_time=$(date +%s)
    
#     local failed_experiments=0
#     run_sequential_experiments "$run_dir" || failed_experiments=$?
    
#     local end_time=$(date +%s)
#     local duration=$((end_time - start_time))
    
#     generate_summary "$run_dir"
    
#     echo ""
#     echo "========================================"
#     echo "🎉 Experiment Run Completed"
#     echo "========================================"
#     echo "Duration: $((duration / 60))m $((duration % 60))s"
#     echo "Results: $run_dir"
    
#     [ $failed_experiments -gt 0 ] && exit 1 || exit 0
# }

# main "$@"

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

# Experiment parameters
declare -a STRATEGIES=("fedavg" "fedavgm" "fedopt" "qfedavg")

MODEL_PARAMS_FEDAVG=""
MODEL_PARAMS_FEDAVGM="server_momentum:0.5,0.9"
MODEL_PARAMS_FEDOPT="tau:0.001,0.01"
MODEL_PARAMS_QFEDAVG="q_param:0.1,0.5,1.0;qffl_learning_rate:0.01,0.1"

declare -a CLIENTS=(25)
declare -a ALPHAS=(0.5)

# Default values
DEFAULT_ROUNDS=10
DEFAULT_MIN_CLIENTS=2
EXPERIMENT_TIMEOUT=1800  # 30 minutes per experiment

# Parse command line arguments
PARALLEL=false
CLEAN=false
QUICK_TEST=false

get_param_combinations() {
    local strategy=$1
    local params_string=""
    
    case "$strategy" in
        "fedavg") params_string="$MODEL_PARAMS_FEDAVG" ;;
        "fedavgm") params_string="$MODEL_PARAMS_FEDAVGM" ;;
        "fedopt") params_string="$MODEL_PARAMS_FEDOPT" ;;
        "qfedavg") params_string="$MODEL_PARAMS_QFEDAVG" ;;
    esac
    
    if [ -z "$params_string" ]; then
        echo "default"
        return
    fi
    
    IFS=';' read -ra PARAMS <<< "$params_string"
    
    if [ ${#PARAMS[@]} -eq 0 ]; then
        echo "default"
        return
    fi
    
    if [ ${#PARAMS[@]} -eq 1 ]; then
        local param="${PARAMS[0]}"
        IFS=':' read -r pname pvalues <<< "$param"
        IFS=',' read -ra values <<< "$pvalues"
        
        for val in "${values[@]}"; do
            echo "${pname}=${val}"
        done
        return
    fi
    
    if [ ${#PARAMS[@]} -eq 2 ]; then
        local param1="${PARAMS[0]}"
        local param2="${PARAMS[1]}"
        
        IFS=':' read -r pname1 pvalues1 <<< "$param1"
        IFS=':' read -r pname2 pvalues2 <<< "$param2"
        
        IFS=',' read -ra values1 <<< "$pvalues1"
        IFS=',' read -ra values2 <<< "$pvalues2"
        
        for val1 in "${values1[@]}"; do
            for val2 in "${values2[@]}"; do
                echo "${pname1}=${val1},${pname2}=${val2}"
            done
        done
        return
    fi
    
    echo "default"
}

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --parallel          Run experiments in parallel"
    echo "  --clean             Clean previous results before running"
    echo "  --quick-test        Run a quick test with minimal parameters"
    echo "  --help              Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL=true; shift ;;
        --clean) CLEAN=true; shift ;;
        --quick-test) QUICK_TEST=true; shift ;;
        --help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

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
    "min_available_clients": PLACEHOLDER_TOTAL_CLIENTS,
    "params": {}
  },
  "data": {
    "base_path": "PLACEHOLDER_DATA_PATH",
    "num_clients": PLACEHOLDER_TOTAL_CLIENTS,
    "alpha": PLACEHOLDER_ALPHA
  },
  "model": {
    "model_type": "dense",
    "learning_rate": 0.001,
    "local_epochs": 1,
    "batch_size": 32,
    "hidden_dims": [64, 32],
    "dropout": 0.2,
    "n_components": 10
  }
}
EOF
    echo "✅ Created config template at $CONFIG_TEMPLATE"
}

initialize_runner() {
    echo "🧪 NASA FL Experiments Runner"
    echo "=============================="
    echo "Run ID: $RUN_ID"
    echo "Timestamp: $(date)"
    echo "Timeout per experiment: ${EXPERIMENT_TIMEOUT}s"
    echo ""
    
    if [ ! -f "$SERVER_SCRIPT" ] || [ ! -f "$CLIENT_SCRIPT" ]; then
        echo "❌ Required scripts not found"
        exit 1
    fi
    
    if [ ! -f "$CONFIG_TEMPLATE" ]; then
        create_config_template
    fi
    
    if [ "$QUICK_TEST" = true ]; then
        STRATEGIES=("fedavg")
        CLIENTS=(5)
        ALPHAS=(0.5)
        DEFAULT_ROUNDS=3
        EXPERIMENT_TIMEOUT=300
        MODEL_PARAMS_FEDAVG=""
        MODEL_PARAMS_FEDAVGM=""
        MODEL_PARAMS_FEDOPT=""
        MODEL_PARAMS_QFEDAVG=""
        echo "🔬 Quick test mode: 3 rounds, 5 clients, 5min timeout"
    else
        echo "🔬 Full experiment mode"
        echo "   Strategies: ${STRATEGIES[@]}"
        echo "   Clients: ${CLIENTS[@]}"
        echo "   Alphas: ${ALPHAS[@]}"
        echo "   Rounds: $DEFAULT_ROUNDS"
        
        echo ""
        echo "📋 Experiment Plan:"
        for strategy in "${STRATEGIES[@]}"; do
            readarray -t combos < <(get_param_combinations "$strategy")
            echo "   ${strategy^^}: ${#combos[@]} configuration(s)"
        done
    fi
    echo ""
}

clean_results() {
    if [ -d "$RESULTS_BASE" ]; then
        echo "🧹 Cleaning previous results..."
        rm -rf "$RESULTS_BASE"
    fi
    mkdir -p "$RESULTS_BASE"
}

create_run_directory() {
    local run_dir="$RESULTS_BASE/$RUN_ID"
    mkdir -p "$run_dir"
    echo "$run_dir"
}

wait_for_port() {
    local port=$1
    local max_attempts=30
    local attempt=0
    
    while netstat -tuln 2>/dev/null | grep ":$port " > /dev/null; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "❌ Port $port still not available after $max_attempts attempts"
            lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
            sleep 2
            return 1
        fi
        echo "⏳ Waiting for port $port... ($attempt/$max_attempts)"
        sleep 2
    done
    return 0
}

generate_experiment_config() {
    local strategy=$1
    local num_clients=$2
    local alpha=$3
    local port=$4
    local config_dir=$5
    local param_combo=${6:-"default"}
    
    local param_suffix=""
    if [ "$param_combo" != "default" ]; then
        param_suffix="_${param_combo//[=,]/_}"
    fi
    
    local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}${param_suffix}"
    local config_file="$config_dir/config.json"
    
    local min_clients=$((num_clients / 2))
    [ $min_clients -lt $DEFAULT_MIN_CLIENTS ] && min_clients=$DEFAULT_MIN_CLIENTS
    [ $min_clients -gt $num_clients ] && min_clients=$num_clients
    
    mkdir -p "$config_dir"
    
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
    
    if [ "$param_combo" != "default" ]; then
        python3 - "$config_file" "$param_combo" <<'PYTHON'
import json
import sys

config_file = sys.argv[1]
params_str = sys.argv[2]

with open(config_file, 'r') as f:
    config = json.load(f)

if 'strategy' not in config:
    config['strategy'] = {}
if 'params' not in config['strategy']:
    config['strategy']['params'] = {}

for param in params_str.split(','):
    key, val = param.split('=')
    try:
        config['strategy']['params'][key] = float(val)
    except ValueError:
        config['strategy']['params'][key] = val

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)
PYTHON
    fi
    
    echo "$config_file"
}

kill_process_tree() {
    local pid=$1
    local children=$(pgrep -P $pid 2>/dev/null || true)
    
    for child in $children; do
        kill_process_tree $child
    done
    
    if kill -0 $pid 2>/dev/null; then
        kill -TERM $pid 2>/dev/null || true
        sleep 1
        if kill -0 $pid 2>/dev/null; then
            kill -KILL $pid 2>/dev/null || true
        fi
    fi
}

# NEW: Check completion signal
check_completion_signal() {
    local results_dir=$1
    local signal_file="$results_dir/.COMPLETE"
    
    if [ -f "$signal_file" ]; then
        local completed=$(python3 -c "import json; print(json.load(open('$signal_file'))['completed'])" 2>/dev/null || echo "false")
        if [ "$completed" = "True" ]; then
            return 0  # Success
        else
            return 1  # Failed
        fi
    fi
    return 2  # Signal file not found
}

# UPDATED: Run experiment with completion signal checking
run_experiment() {
    local strategy=$1
    local num_clients=$2
    local alpha=$3
    local port=$4
    local run_dir=$5
    local param_combo=${6:-"default"}
    
    local param_suffix=""
    [ "$param_combo" != "default" ] && param_suffix="_${param_combo//[=,]/_}"
    
    local exp_id="nasa_${num_clients}c_alpha_${alpha}_${strategy}${param_suffix}"
    local results_dir="$run_dir/$exp_id"
    local signal_file="$results_dir/.COMPLETE"
    
    echo "🚀 Starting experiment: $exp_id"
    echo "   Strategy: $strategy | Params: $param_combo"
    echo "   Clients: $num_clients | Alpha: $alpha | Port: $port"
    
    mkdir -p "$results_dir/logs" "$results_dir/metrics"
    
    local config_file=$(generate_experiment_config "$strategy" "$num_clients" "$alpha" "$port" "$results_dir" "$param_combo")
    
    if [ ! -f "$config_file" ]; then
        echo "❌ Config file not created"
        return 1
    fi
    
    local server_log="$results_dir/logs/server_$(date +%H%M%S).log"
    echo "   Server log: $server_log"
    echo "   Signal file: $signal_file"
    
    # Remove old signal file if exists
    rm -f "$signal_file"
    
    # Start server
    python3 "$SERVER_SCRIPT" --config "$config_file" --results-dir "$results_dir" > "$server_log" 2>&1 &
    local server_pid=$!
    
    sleep 10
    
    if ! kill -0 $server_pid 2>/dev/null; then
        echo "❌ Server failed to start"
        tail -20 "$server_log"
        return 1
    fi
    
    echo "   Starting $num_clients clients..."
    local client_pids=()
    
    for ((i=0; i<num_clients; i++)); do
        local client_log="$results_dir/logs/client_${i}_$(date +%H%M%S).log"
        python3 "$CLIENT_SCRIPT" --client-id "client_$i" --config "$config_file" > "$client_log" 2>&1 &
        client_pids+=($!)
        sleep 1
    done
    
    echo "   Server PID: $server_pid | Timeout: ${EXPERIMENT_TIMEOUT}s"
    echo "   Monitoring completion signal..."
    
    # UPDATED: Wait for completion signal or timeout
    local elapsed=0
    local check_interval=10
    local completion_status=2  # 0=success, 1=failed, 2=not found
    
    while [ $completion_status -eq 2 ] && kill -0 $server_pid 2>/dev/null; do
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
        
        # Check for completion signal
        check_completion_signal "$results_dir"
        completion_status=$?
        
        if [ $completion_status -eq 0 ]; then
            echo "   🏁 Completion signal detected (success) after ${elapsed}s"
            break
        elif [ $completion_status -eq 1 ]; then
            echo "   ⚠️  Completion signal detected (failure) after ${elapsed}s"
            break
        fi
        
        # Check timeout
        if [ $elapsed -ge $EXPERIMENT_TIMEOUT ]; then
            echo "⏰ Experiment timeout after ${elapsed}s"
            kill_process_tree $server_pid
            for pid in "${client_pids[@]}"; do
                kill_process_tree $pid
            done
            echo "❌ Timeout - killed all processes"
            return 1
        fi
        
        # Show progress every 60 seconds
        if [ $((elapsed % 60)) -eq 0 ]; then
            echo "   Running for ${elapsed}s... (waiting for signal)"
        fi
    done
    
    # Server may have finished naturally
    if kill -0 $server_pid 2>/dev/null; then
        echo "   Waiting for server to exit..."
        sleep 3
    fi
    
    # Get final exit code
    wait $server_pid 2>/dev/null
    local server_exit_code=$?
    
    echo "   Server finished (exit: $server_exit_code) after ${elapsed}s"
    
    # Clean up clients
    sleep 3
    for pid in "${client_pids[@]}"; do
        kill_process_tree $pid 2>/dev/null || true
    done
    
    # Final verification - check completion signal one more time
    check_completion_signal "$results_dir"
    local final_status=$?
    
    # Verify results using both completion signal and CSV files
    if [ $final_status -eq 0 ] && [ -f "$results_dir/metrics/round_metrics.csv" ]; then
        local lines=$(wc -l < "$results_dir/metrics/round_metrics.csv")
        echo "✅ Success - $((lines - 1)) rounds completed"
        echo "   Completion signal: CONFIRMED"
        return 0
    else
        echo "❌ Failed"
        if [ $final_status -eq 1 ]; then
            echo "   Completion signal: FAILED"
            if [ -f "$signal_file" ]; then
                echo "   Error details:"
                cat "$signal_file"
            fi
        elif [ $final_status -eq 2 ]; then
            echo "   Completion signal: NOT FOUND"
        fi
        [ -f "$server_log" ] && echo "   Last 10 log lines:" && tail -10 "$server_log"
        return 1
    fi
}

run_sequential_experiments() {
    local run_dir=$1
    local current_port=$BASE_PORT
    local total_experiments=0
    
    for strategy in "${STRATEGIES[@]}"; do
        readarray -t param_combos < <(get_param_combinations "$strategy")
        total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
    done
    
    local completed=0
    local failed=0
    
    echo "➡️  Running $total_experiments experiments sequentially"
    echo ""
    
    for strategy in "${STRATEGIES[@]}"; do
        echo ""
        echo "🔷 Strategy: ${strategy^^}"
        echo "========================================" 
        
        readarray -t param_combos < <(get_param_combinations "$strategy")
        
        for param_combo in "${param_combos[@]}"; do
            for num_clients in "${CLIENTS[@]}"; do
                for alpha in "${ALPHAS[@]}"; do
                    ((completed++))
                    echo ""
                    echo "📊 Experiment $completed/$total_experiments"
                    echo "========================================"
                    
                    if ! wait_for_port $current_port; then
                        echo "⚠️  Port cleanup failed, continuing..."
                    fi
                    
                    if run_experiment "$strategy" "$num_clients" "$alpha" $current_port "$run_dir" "$param_combo"; then
                        echo "✅ Completed successfully"
                    else
                        echo "❌ Failed"
                        ((failed++))
                    fi
                    
                    current_port=$((current_port + 1))
                    echo "========================================"
                    
                    # Brief pause between experiments
                    sleep 5
                done
            done
        done
    done
    
    echo ""
    echo "🎉 All experiments completed"
    echo "   Success: $((completed - failed))/$completed"
    echo "   Failed: $failed"
    return $failed
}

generate_summary() {
    local run_dir=$1
    local summary_file="$run_dir/experiment_summary.md"
    
    local total_experiments=0
    for strategy in "${STRATEGIES[@]}"; do
        readarray -t param_combos < <(get_param_combinations "$strategy")
        total_experiments=$((total_experiments + ${#param_combos[@]} * ${#CLIENTS[@]} * ${#ALPHAS[@]}))
    done
    
    local completed_count=$(find "$run_dir" -name ".COMPLETE" -exec grep -l '"completed": true' {} \; 2>/dev/null | wc -l)
    local csv_count=$(find "$run_dir" -name "round_metrics.csv" 2>/dev/null | wc -l)
    
    cat > "$summary_file" << EOF
# NASA FL Experiments Summary

**Run ID**: $RUN_ID  
**Date**: $(date)  
**Total**: $total_experiments experiments  
**Completed (by signal)**: $completed_count  
**Completed (by CSV)**: $csv_count  
**Success Rate**: $((completed_count * 100 / total_experiments))%

## Configurations
- Strategies: ${STRATEGIES[@]}
- Clients: ${CLIENTS[@]}
- Alphas: ${ALPHAS[@]}
- Rounds: $DEFAULT_ROUNDS

## Completion Status
\`\`\`bash
# Check completion signals
find "$run_dir" -name ".COMPLETE" -exec bash -c 'echo -n "{}: "; python3 -c "import json; print(json.load(open(\"{}\"))[\"completed\"])"' \;

# Check CSV results
find "$run_dir" -name "round_metrics.csv" -exec dirname {} \;
\`\`\`
EOF

    echo "📊 Summary: $summary_file"
    echo "📈 Completion: $completed_count/$total_experiments (signals), $csv_count/$total_experiments (CSV)"
}

main() {
    initialize_runner
    
    [ "$CLEAN" = true ] && clean_results
    
    local run_dir=$(create_run_directory)
    echo "📁 Results: $run_dir"
    echo ""
    
    local start_time=$(date +%s)
    
    local failed_experiments=0
    run_sequential_experiments "$run_dir" || failed_experiments=$?
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    generate_summary "$run_dir"
    
    echo ""
    echo "========================================"
    echo "🎉 Experiment Run Completed"
    echo "========================================"
    echo "Duration: $((duration / 60))m $((duration % 60))s"
    echo "Results: $run_dir"
    
    [ $failed_experiments -gt 0 ] && exit 1 || exit 0
}

main "$@"