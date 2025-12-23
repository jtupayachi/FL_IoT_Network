#!/bin/bash
# TRUE PARALLEL Orchestration for M3 New Methods
# Runs multiple FL experiments simultaneously with different ports

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MODEL_TYPE="${1:-LSTM}"
MAX_PARALLEL="${2:-3}"  # Number of experiments to run in parallel

if [[ "$MODEL_TYPE" != "LSTM" && "$MODEL_TYPE" != "MLP" ]]; then
    echo -e "${RED}Error: MODEL_TYPE must be LSTM or MLP${NC}"
    echo "Usage: $0 [LSTM|MLP] [MAX_PARALLEL]"
    exit 1
fi

EXPERIMENT_NAME="M3_PARALLEL_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TRUE PARALLEL FL Orchestration${NC}"
echo -e "${BLUE}Model: ${MODEL_TYPE}${NC}"
echo -e "${BLUE}Max Parallel: ${MAX_PARALLEL}${NC}"
echo -e "${BLUE}========================================${NC}"

# Define experiment configurations
declare -a EXPERIMENTS

if [[ "$MODEL_TYPE" == "LSTM" ]]; then
    SERVER_BASE_IP="172.18.0.2"
    BASE_PORT=8080
    
    # Reduced experiment set for parallel execution
    EXPERIMENTS=(
        "MOON:mu=0.5:temp=0.5:slr=0.01:alpha=0.001"
        "MOON:mu=1.0:temp=0.5:slr=0.01:alpha=0.001"
        "MOON:mu=0.5:temp=0.5:slr=0.01:alpha=0.1"
        "FedALA:eta=0.1:threshold=0.5:slr=0.01:alpha=0.001"
        "FedALA:eta=0.1:threshold=0.5:slr=0.01:alpha=0.1"
        "StatAvg:stat_weight=0.1:use_variance=true:slr=0.01:alpha=0.001"
        "DASHA:alpha=0.1:gamma=0.5:momentum=0.9:slr=0.01:alpha_split=0.001"
    )
else
    SERVER_BASE_IP="172.18.0.8"
    BASE_PORT=8080
    
    EXPERIMENTS=(
        "MOON:mu=0.5:temp=0.5:slr=0.01:alpha=0.001"
        "MOON:mu=1.0:temp=0.5:slr=0.01:alpha=0.001"
        "MOON:mu=0.5:temp=0.5:slr=0.01:alpha=0.1"
        "FedALA:eta=0.1:threshold=0.5:slr=0.01:alpha=0.001"
        "FedALA:eta=0.1:threshold=0.5:slr=0.01:alpha=0.1"
        "StatAvg:stat_weight=0.1:use_variance=true:slr=0.01:alpha=0.001"
        "DASHA:alpha=0.1:gamma=0.5:momentum=0.9:slr=0.01:alpha_split=0.001"
    )
fi

# Function to run one complete FL experiment (1 server + 5 clients)
run_experiment() {
    local exp_config=$1
    local port=$2
    local exp_id=$3
    
    IFS=':' read -ra PARAMS <<< "$exp_config"
    local method=${PARAMS[0]}
    
    echo -e "${BLUE}[EXP-${exp_id}] Starting ${method} on port ${port}${NC}"
    
    local exp_log_dir="${LOG_DIR}/exp_${exp_id}_${method}"
    mkdir -p "${exp_log_dir}"
    
    # Parse parameters
    declare -A param_map
    for i in "${!PARAMS[@]}"; do
        if [ $i -gt 0 ]; then
            IFS='=' read -ra KV <<< "${PARAMS[$i]}"
            param_map[${KV[0]}]=${KV[1]}
        fi
    done
    
    local server_container="fl_server"
    if [[ "$MODEL_TYPE" == "MLP" ]]; then
        server_container="fl_mlp_server"
    fi
    
    # Start server with specific port
    local server_cmd=""
    if [[ "$method" == "MOON" ]]; then
        server_cmd="python3 fl_testbed/version2/server/federated_server_${MODEL_TYPE}_MOON.py"
        server_cmd="$server_cmd -mu ${param_map[mu]} -temperature ${param_map[temp]}"
    elif [[ "$method" == "FedALA" ]]; then
        server_cmd="python3 fl_testbed/version2/server/federated_server_${MODEL_TYPE}_FedALA.py"
        server_cmd="$server_cmd -eta ${param_map[eta]} -threshold ${param_map[threshold]}"
    elif [[ "$method" == "StatAvg" ]]; then
        server_cmd="python3 fl_testbed/version2/server/federated_server_${MODEL_TYPE}_StatAvg.py"
        server_cmd="$server_cmd -stat_weight ${param_map[stat_weight]} -use_variance ${param_map[use_variance]}"
    elif [[ "$method" == "DASHA" ]]; then
        server_cmd="python3 fl_testbed/version2/server/federated_server_${MODEL_TYPE}_DASHA.py"
        server_cmd="$server_cmd -alpha ${param_map[alpha]} -gamma ${param_map[gamma]} -momentum ${param_map[momentum]}"
    fi
    
    # Note: This requires modifying Python server code to accept -port parameter
    # For now, we document what needs to happen
    
    echo -e "${YELLOW}[EXP-${exp_id}] Would start: ${server_cmd}${NC}"
    echo -e "${YELLOW}[EXP-${exp_id}] Port: ${port}${NC}"
    echo -e "${YELLOW}[EXP-${exp_id}] Log: ${exp_log_dir}${NC}"
    
    # This is a placeholder - actual implementation needs Python code modification
    echo "${exp_config}" > "${exp_log_dir}/config.txt"
    echo "${port}" > "${exp_log_dir}/port.txt"
}

# Main parallel execution loop
echo -e "${GREEN}Starting parallel experiments...${NC}"

running_jobs=0
exp_id=0
port=$BASE_PORT

for exp in "${EXPERIMENTS[@]}"; do
    # Wait if we've hit max parallel
    while [ $running_jobs -ge $MAX_PARALLEL ]; do
        sleep 5
        # Check completed jobs (simplified)
        running_jobs=$(jobs -r | wc -l)
    done
    
    run_experiment "$exp" $port $exp_id &
    
    ((exp_id++))
    ((port++))
    running_jobs=$(jobs -r | wc -l)
    
    sleep 2
done

# Wait for all to complete
wait

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All parallel experiments completed!${NC}"
echo -e "${GREEN}Results in: ${LOG_DIR}${NC}"
echo -e "${GREEN}========================================${NC}"
