# # # #!/bin/bash

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

# # Experiment parameters
# declare -a STRATEGIES=("fedavg" "fedavgm" "fedopt" "qfedavg")

# MODEL_PARAMS_FEDAVG=""
# MODEL_PARAMS_FEDAVGM="server_momentum:0.5,0.9"
# MODEL_PARAMS_FEDOPT="tau:0.001,0.01"
# MODEL_PARAMS_QFEDAVG="q_param:0.1,0.5,1.0;qffl_learning_rate:0.01,0.1"

# declare -a CLIENTS=(25)
# declare -a ALPHAS=(0.01, 0.05, 0.1, 0.2, 0.5, 0.075, 1)

# Experiment parameters
declare -a STRATEGIES=("fedavg" "fedavgm" "fedopt" "qfedavg" "moon" "fedala") #

# MODEL_PARAMS_FEDAVG=""
# MODEL_PARAMS_FEDAVGM="server_momentum:0.3,0.6,0.9"
# MODEL_PARAMS_FEDOPT="tau:0.001,0.01,0.1,1"
# MODEL_PARAMS_QFEDAVG="q_param:0.1,0.5,1.0;qffl_learning_rate:0.01,0.1"
# MODEL_PARAMS_MOON="temperature:0.5,0.7;mu:1.0,5.0"
# MODEL_PARAMS_FEDALA="eta:0.5,1.0;eta_l:0.05,0.1"

MODEL_PARAMS_FEDAVG=""

# ✅ Use the correct parameter names that your server expects:
MODEL_PARAMS_FEDAVGM="server_momentum:0.7"          # Or "beta:0.7" if server uses "beta"
MODEL_PARAMS_FEDOPT="tau:1e-8"                      # ✅ Correct
MODEL_PARAMS_QFEDAVG="q_param:0.2"                  # Or "q:0.2" if server uses "q"
MODEL_PARAMS_MOON="temperature:0.5;mu:1.0"                          # ✅ Correct  
MODEL_PARAMS_FEDALA="eta:0.5;eta_l:0.05" 

declare -a CLIENTS=(25 100)
declare -a ALPHAS=(0.05) #0.01 0.02 0.1 0.2 0.5 0.075 1

# 0.05 0.1 0.2 0.5 1.0 0.005 
# 0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0

# Default values
DEFAULT_ROUNDS=1000
DEFAULT_MIN_CLIENTS=25
EXPERIMENT_TIMEOUT=1800  # 30 minutes per experiment

# Parse command line arguments
PARALLEL=false
CLEAN=false
QUICK_TEST=false

get_param_combinations() {
    local strategy=$1
    local params_string=""
    
    # case "$strategy" in
    #     "fedavg") params_string="$MODEL_PARAMS_FEDAVG" ;;
    #     "fedavgm") params_string="$MODEL_PARAMS_FEDAVGM" ;;
    #     "fedopt") params_string="$MODEL_PARAMS_FEDOPT" ;;
    #     "qfedavg") params_string="$MODEL_PARAMS_QFEDAVG" ;;
    # esac

    case "$strategy" in
        "fedavg") params_string="$MODEL_PARAMS_FEDAVG" ;;
        "fedavgm") params_string="$MODEL_PARAMS_FEDAVGM" ;;
        "fedopt") params_string="$MODEL_PARAMS_FEDOPT" ;;
        "qfedavg") params_string="$MODEL_PARAMS_QFEDAVG" ;;
        "moon") params_string="$MODEL_PARAMS_MOON" ;;
        "fedala") params_string="$MODEL_PARAMS_FEDALA" ;;
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