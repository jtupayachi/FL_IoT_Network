#!/bin/bash
# Master Orchestration Script for M3 New Methods (MOON, FedALA, StatAvg, DASHA)
# This script runs federated learning experiments in parallel using Docker containers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_TYPE="${1:-LSTM}"  # LSTM or MLP
EXPERIMENT_NAME="M3_NEW_METHODS_${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${EXPERIMENT_NAME}"

# Validate input
if [[ "$MODEL_TYPE" != "LSTM" && "$MODEL_TYPE" != "MLP" ]]; then
    echo -e "${RED}Error: MODEL_TYPE must be LSTM or MLP${NC}"
    echo "Usage: $0 [LSTM|MLP]"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}FL M3 New Methods Orchestration${NC}"
echo -e "${BLUE}Model Type: ${MODEL_TYPE}${NC}"
echo -e "${BLUE}Experiment: ${EXPERIMENT_NAME}${NC}"
echo -e "${BLUE}========================================${NC}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Function to check if containers are running
check_containers() {
    echo -e "${YELLOW}Checking Docker containers...${NC}"
    
    local required_containers=("fl_server" "fl_client1" "fl_client2" "fl_client3" "fl_client4" "fl_client5")
    if [[ "$MODEL_TYPE" == "MLP" ]]; then
        required_containers+=("fl_mlp_server")
    fi
    
    for container in "${required_containers[@]}"; do
        if ! docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
            echo -e "${RED}Error: Container ${container} is not running${NC}"
            echo -e "${YELLOW}Starting containers with docker-compose...${NC}"
            docker-compose up -d
            sleep 10
            return
        fi
    done
    
    echo -e "${GREEN}✓ All containers are running${NC}"
}

# Function to wait for server readiness
wait_for_server() {
    local server_ip=$1
    local server_name=$2
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}Waiting for ${server_name} to be ready...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if docker exec fl_server nc -z ${server_ip} 5000 2>/dev/null; then
            echo -e "${GREEN}✓ ${server_name} is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo -e "${RED}✗ ${server_name} failed to start${NC}"
    return 1
}

# Function to run server script
run_server() {
    local model_type=$1
    local container_name="fl_server"
    local script_name="server_execution_${model_type}_M3_NEW_METHODS.sh"
    
    if [[ "$model_type" == "MLP" ]]; then
        container_name="fl_mlp_server"
    fi
    
    echo -e "${BLUE}Starting ${model_type} Server...${NC}"
    docker exec -d ${container_name} bash -c "cd /workspace && bash ${script_name} 2>&1 | tee ${LOG_DIR}/server_${model_type}.log"
    
    # Give server time to initialize
    sleep 5
}

# Function to run client scripts in parallel
run_clients_parallel() {
    local model_type=$1
    
    echo -e "${BLUE}Starting ${model_type} Clients in parallel...${NC}"
    
    # Array to store background process PIDs
    local pids=()
    
    # Start all clients in background
    for client_id in 0 1 2 3 4; do
        local container_name="fl_client$((client_id + 1))"
        local script_name="client${client_id}_execution_${model_type}_M3_NEW_METHODS.sh"
        
        echo -e "${YELLOW}  Starting Client ${client_id} in ${container_name}...${NC}"
        docker exec -d ${container_name} bash -c "cd /workspace && bash ${script_name} 2>&1 | tee ${LOG_DIR}/client${client_id}_${model_type}.log" &
        pids+=($!)
        
        # Small delay to avoid simultaneous starts
        sleep 1
    done
    
    echo -e "${GREEN}✓ All ${model_type} clients started${NC}"
    echo -e "${BLUE}Client processes: ${pids[@]}${NC}"
}

# Function to monitor progress
monitor_progress() {
    local model_type=$1
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Monitoring ${model_type} Experiment${NC}"
    echo -e "${BLUE}Press Ctrl+C to stop monitoring (experiments will continue)${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    while true; do
        echo -e "\n${YELLOW}--- Status Update $(date) ---${NC}"
        
        # Check server
        if docker exec fl_server pgrep -f "federated_server_.*${model_type}" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server is running${NC}"
        else
            echo -e "${RED}✗ Server process not found${NC}"
        fi
        
        # Check clients
        local running_clients=0
        for client_id in 0 1 2 3 4; do
            local container_name="fl_client$((client_id + 1))"
            if docker exec ${container_name} pgrep -f "federated_client" > /dev/null 2>&1; then
                running_clients=$((running_clients + 1))
            fi
        done
        echo -e "${GREEN}Active clients: ${running_clients}/5${NC}"
        
        # Show recent activity from logs
        if [ -f "${LOG_DIR}/server_${model_type}.log" ]; then
            echo -e "\n${BLUE}Recent server activity:${NC}"
            tail -n 3 "${LOG_DIR}/server_${model_type}.log" 2>/dev/null || echo "No logs yet"
        fi
        
        sleep 30
    done
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Log files location:${NC}"
    echo -e "${BLUE}${LOG_DIR}${NC}"
    echo -e "${BLUE}========================================${NC}"
    ls -lh "${LOG_DIR}/" 2>/dev/null || echo "No logs yet"
}

# Main execution
main() {
    echo -e "\n${YELLOW}Step 1: Checking containers...${NC}"
    check_containers
    
    echo -e "\n${YELLOW}Step 2: Starting ${MODEL_TYPE} Server...${NC}"
    run_server "${MODEL_TYPE}"
    
    # Determine server IP based on model type
    local server_ip="172.18.0.2"
    if [[ "$MODEL_TYPE" == "MLP" ]]; then
        server_ip="172.18.0.8"
    fi
    
    echo -e "\n${YELLOW}Step 3: Waiting for server readiness...${NC}"
    wait_for_server "${server_ip}" "${MODEL_TYPE} Server"
    
    echo -e "\n${YELLOW}Step 4: Starting clients in parallel...${NC}"
    run_clients_parallel "${MODEL_TYPE}"
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}Experiment Started Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    show_logs
    
    echo -e "\n${BLUE}Options:${NC}"
    echo -e "  ${YELLOW}m)${NC} Monitor progress"
    echo -e "  ${YELLOW}l)${NC} Show logs location"
    echo -e "  ${YELLOW}q)${NC} Quit (experiments continue in background)"
    echo -e ""
    read -p "Choose an option [m/l/q]: " choice
    
    case $choice in
        m|M)
            monitor_progress "${MODEL_TYPE}"
            ;;
        l|L)
            show_logs
            ;;
        q|Q)
            echo -e "${GREEN}Experiments are running in background.${NC}"
            echo -e "Check logs in: ${LOG_DIR}"
            ;;
        *)
            echo -e "${YELLOW}Experiments are running in background.${NC}"
            ;;
    esac
}

# Run main function
main

echo -e "${GREEN}Orchestration script completed.${NC}"
echo -e "${BLUE}Experiments continue running in Docker containers.${NC}"
echo -e "${BLUE}Monitor with: docker logs -f fl_server${NC}"
