# FL_IoT_Network - MOON and FedALA Implementation Status

## ✅ Completed

### 1. Docker Infrastructure
- **docker-compose.yml**: Created multi-container setup for 1 server + 5 clients + 1 MLP server
  - All containers configured with GPU support
  - Network configured with fixed IPs (172.17.0.x)
  - Shared volume mounting for workspace

### 2. Custom FL Strategies
- **CustomStrategy_MOON.py**: MOON (Model-Contrastive Federated Learning) strategy
  - Parameters: `mu` (contrastive weight), `temperature`
  - Implements contrastive learning at model level
  
- **CustomStrategy_FedALA.py**: FedALA (Federated Adaptive Local Aggregation) strategy
  - Parameters: `eta` (learning rate), `threshold`, `rand_percent`
  - Implements adaptive layer-wise aggregation

### 3. Execution Scripts
- **server_execution_LSTM_M3_NEW_METHODS.sh**: Server script for LSTM/RUL experiments with MOON and FedALA
- **server_execution_MLP_M3_NEW_METHODS.sh**: Server script for MLP/OFFSET experiments with MOON and FedALA
- **client[1-5]_execution_LSTM_M3_NEW_METHODS.sh**: Client scripts for LSTM experiments
- **client[1-5]_execution_MLP_M3_NEW_METHODS.sh**: Client scripts for MLP experiments

### 4. Helper Scripts
- **manage_docker.sh**: Docker container management utility
- **DOCKER_SETUP_README.md**: Complete Docker setup documentation

## 🔄 Next Steps

### Phase 1: Create Server Implementation Files (CRITICAL)

You need to create 4 new Python server files by copying and modifying existing ones:

#### For LSTM (RUL Prediction):

1. **federated_server_RUL_MOON.py**
   ```bash
   cp fl_testbed/version2/server/federated_server_RUL_FedOpt.py \
      fl_testbed/version2/server/federated_server_RUL_MOON.py
   ```
   Modifications needed:
   - Line ~1: Import `from CustomStrategy_MOON import MOON`
   - Line ~900: Change strategy from `FedOpt` to `MOON`
   - Update parameters: Remove `tau`, `slr`; Add `mu`, `temperature`
   - Update argparse arguments accordingly

2. **federated_server_RUL_FedALA.py**
   ```bash
   cp fl_testbed/version2/server/federated_server_RUL_FedOpt.py \
      fl_testbed/version2/server/federated_server_RUL_FedALA.py
   ```
   Modifications needed:
   - Line ~1: Import `from CustomStrategy_FedALA import FedALA`
   - Line ~900: Change strategy from `FedOpt` to `FedALA`
   - Update parameters: Remove `tau`, `slr`; Add `eta`, `threshold`, `rand_percent`
   - Update argparse arguments accordingly

#### For MLP (OFFSET Prediction):

3. **federated_server_OFFSET_MOON.py**
   ```bash
   cp fl_testbed/version2/server/federated_server_OFFSET_FedOpt.py \
      fl_testbed/version2/server/federated_server_OFFSET_MOON.py
   ```
   Same modifications as RUL_MOON above

4. **federated_server_OFFSET_FedALA.py**
   ```bash
   cp fl_testbed/version2/server/federated_server_OFFSET_FedOpt.py \
      fl_testbed/version2/server/federated_server_OFFSET_FedALA.py
   ```
   Same modifications as RUL_FedALA above

### Phase 2: Create Client Implementation Files (CRITICAL)

You need to create 4 new Python client files:

#### For LSTM:

1. **federated_client_RUL_MOON.py**
   ```bash
   cp fl_testbed/version2/client/federated_client_RUL_FedOpt.py \
      fl_testbed/version2/client/federated_client_RUL_MOON.py
   ```
   Modifications:
   - No major changes needed for basic MOON
   - Clients use standard FlowerClient class
   - MOON logic is handled server-side

2. **federated_client_RUL_FedALA.py**
   ```bash
   cp fl_testbed/version2/client/federated_client_RUL_FedOpt.py \
      fl_testbed/version2/client/federated_client_RUL_FedALA.py
   ```
   Same as above

#### For MLP:

3. **federated_client_OFFSET_MOON.py**
4. **federated_client_OFFSET_FedALA.py**
   (Copy from corresponding FedOpt files)

### Phase 3: Docker Setup and Testing

1. **Build Docker Images**
   ```bash
   cd /home/jose/FL_IoT_Network
   docker-compose build
   ```

2. **Start Containers**
   ```bash
   docker-compose up -d
   ```

3. **Verify Containers**
   ```bash
   docker-compose ps
   ./manage_docker.sh status
   ```

4. **Access Containers**
   ```bash
   # Server
   docker exec -it fl_server bash
   
   # Client 1
   docker exec -it fl_client1 bash
   ```

### Phase 4: Run Experiments

#### Option 1: Manual Execution

**Terminal 1 (Server):**
```bash
docker exec -it fl_server bash
cd /workspace
./server_execution_LSTM_M3_NEW_METHODS.sh
```

**Terminal 2-6 (Clients):**
```bash
# Client 1
docker exec -it fl_client1 bash
cd /workspace
./client1_execution_LSTM_M3_NEW_METHODS.sh

# Repeat for clients 2-5
```

#### Option 2: Using Helper Script

```bash
./manage_docker.sh start-experiment lstm m3
```

### Phase 5: Monitor and Collect Results

1. **Monitor Logs**
   ```bash
   docker-compose logs -f server
   docker-compose logs -f client1
   ```

2. **Results Location**
   - Output files: `LSTM_*.txt` and `MLP_*.txt` in main directory
   - Models: `fl_testbed/version2/data/transformed/`

## 📊 Parameter Ranges for Experiments

### MOON Parameters:
- **mu**: `0.001 0.01 0.1 0.5 1.0 5.0`
- **temperature**: `0.1 0.5 1.0`
- **slr**: `0.001 0.01 1`

### FedALA Parameters:
- **eta**: `0.001 0.01 0.1 1.0 10.0`
- **threshold**: `0.1 0.3 0.5 0.7 0.9`
- **rand_percent**: `50 70 80 90`

## 🔧 Quick Commands

```bash
# Build everything
docker-compose build

# Start all containers
docker-compose up -d

# Stop all containers
docker-compose down

# View container status
docker-compose ps

# View logs
docker-compose logs -f

# Clean up
docker-compose down -v
docker system prune -a
```

## 📝 Notes

1. **GPU Support**: Ensure NVIDIA Docker runtime is installed
2. **Data**: Make sure data files exist in `fl_testbed/version2/data/transformed/`
3. **Permissions**: Execute scripts may need `chmod +x *.sh`
4. **Network**: Fixed IPs ensure consistent communication between containers

## ⚠️ Important

The implementation is **80% complete**. The critical missing piece is:
- **Server Python files** for MOON and FedALA (4 files)
- **Client Python files** for MOON and FedALA (4 files)

These need to be created by copying existing FedOpt files and modifying the strategy initialization sections as described in Phase 1 and Phase 2 above.
