# Federated Learning M3 New Methods - Parallel Execution Guide

## Overview
This setup enables parallel execution of federated learning experiments for M3 New Methods (MOON, FedALA, StatAvg, DASHA) using Docker containers.

## Architecture

### Docker Containers
- **fl_server** (172.18.0.2) - LSTM RUL Server
- **fl_mlp_server** (172.18.0.8) - MLP Offset Server
- **fl_client1** (172.18.0.3) - Client 0
- **fl_client2** (172.18.0.4) - Client 1
- **fl_client3** (172.18.0.5) - Client 2
- **fl_client4** (172.18.0.6) - Client 3
- **fl_client5** (172.18.0.7) - Client 4

### Client Scripts
Each client (0-4) has dedicated scripts for both LSTM and MLP:
- `client{0-4}_execution_LSTM_M3_NEW_METHODS.sh` - LSTM RUL experiments
- `client{0-4}_execution_MLP_M3_NEW_METHODS.sh` - MLP Offset experiments

### Server Scripts
- `server_execution_LSTM_M3_NEW_METHODS.sh` - Coordinates LSTM experiments
- `server_execution_MLP_M3_NEW_METHODS.sh` - Coordinates MLP experiments

## Quick Start

### 1. Start Docker Containers
```bash
docker-compose up -d
```

Verify all containers are running:
```bash
docker ps
```

### 2. Run Experiments (Automated)

**For LSTM experiments:**
```bash
bash orchestrate_M3_NEW_METHODS.sh LSTM
```

**For MLP experiments:**
```bash
bash orchestrate_M3_NEW_METHODS.sh MLP
```

The orchestration script will:
- Check all containers are running
- Start the server
- Wait for server to be ready
- Launch all 5 clients in parallel
- Provide monitoring options

### 3. Manual Execution (Alternative)

If you prefer manual control:

**Terminal 1 - Start LSTM Server:**
```bash
docker exec -it fl_server bash
cd /workspace
bash server_execution_LSTM_M3_NEW_METHODS.sh
```

**Terminals 2-6 - Start Clients (one per terminal):**
```bash
# Client 0
docker exec -it fl_client1 bash -c "cd /workspace && bash client0_execution_LSTM_M3_NEW_METHODS.sh"

# Client 1
docker exec -it fl_client2 bash -c "cd /workspace && bash client1_execution_LSTM_M3_NEW_METHODS.sh"

# Client 2
docker exec -it fl_client3 bash -c "cd /workspace && bash client2_execution_LSTM_M3_NEW_METHODS.sh"

# Client 3
docker exec -it fl_client4 bash -c "cd /workspace && bash client3_execution_LSTM_M3_NEW_METHODS.sh"

# Client 4
docker exec -it fl_client5 bash -c "cd /workspace && bash client4_execution_LSTM_M3_NEW_METHODS.sh"
```

## Key Improvements

### 1. Removed Sleep Delays
- **Old:** 500s wait between alphas, 300s wait before each client
- **New:** Smart synchronization using `nc` (netcat) to check server readiness

### 2. Parallel Client Execution
- All 5 clients run simultaneously in separate Docker containers
- Each client connects to the server independently
- Massive speedup in computation time

### 3. Proper Synchronization
Each client script includes:
```bash
while ! nc -z <server_ip> 5000 2>/dev/null; do
    sleep 2
done
```
This ensures clients only start when the server is ready.

### 4. Fixed Client IDs
- **Client 0** → runs in `fl_client1` container (172.18.0.3)
- **Client 1** → runs in `fl_client2` container (172.18.0.4)
- **Client 2** → runs in `fl_client3` container (172.18.0.5)
- **Client 3** → runs in `fl_client4` container (172.18.0.6)
- **Client 4** → runs in `fl_client5` container (172.18.0.7)

## Monitoring

### View Live Logs
```bash
# Server logs
docker logs -f fl_server

# Client logs
docker logs -f fl_client1  # Client 0
docker logs -f fl_client2  # Client 1
# ... etc
```

### Check Running Processes
```bash
# Inside server container
docker exec fl_server ps aux | grep python

# Inside client containers
docker exec fl_client1 ps aux | grep python
```

### View Output Files
All output files (`.txt`) are saved in the mounted `/workspace` directory and are accessible from the host at `/home/jose/FL_IoT_Network/`

## Experiment Methods

### MOON (Model-Contrastive Federated Learning)
Parameters:
- `mu`: 0.1, 0.5, 1.0, 5.0
- `temperature`: 0.1, 0.5, 1.0

### FedALA (Federated Learning with Adaptive Local Aggregation)
Parameters:
- `eta`: 0.01, 0.1, 1.0
- `threshold`: 0.1, 0.5, 1.0

### StatAvg (Statistical Averaging)
Parameters:
- `stat_weight`: 0.01, 0.1, 0.5
- `use_variance`: true, false

### DASHA (Distributed Adaptive SGD with Compression)
Parameters:
- `alpha`: 0.01, 0.1, 0.5
- `gamma`: 0.3, 0.5, 0.7
- `momentum`: 0.7, 0.9, 0.95

## Troubleshooting

### Containers Not Starting
```bash
docker-compose down
docker-compose up -d --build
```

### Network Issues
```bash
# Check network connectivity
docker exec fl_client1 ping -c 3 172.18.0.2
```

### Port Already in Use
```bash
# Check what's using port 5000
sudo lsof -i :5000
```

### Reset Everything
```bash
docker-compose down
docker volume prune -f
docker-compose up -d
```

## Resource Requirements

Each container needs:
- **GPU**: NVIDIA GPU access (configured in docker-compose.yml)
- **CPU**: At least 2 cores per container
- **RAM**: 4GB+ per container
- **Disk**: Sufficient space for model checkpoints and logs

For 7 containers (2 servers + 5 clients), recommended minimum:
- **16 CPU cores**
- **32GB RAM**
- **2 GPUs** (or GPU with sufficient memory for parallel training)

## Notes

- The `sleep` commands were removed to enable true parallel execution
- Synchronization is now handled via network polling (`nc -z`)
- Each client has its own data file (e.g., `M3_5_0_ddf_LSTM.pkl` for client 0)
- All containers share the `/workspace` volume for data access
- Experiments can run for extended periods - use `tmux` or `screen` for long sessions
