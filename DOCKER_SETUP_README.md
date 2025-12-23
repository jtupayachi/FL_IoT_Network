# Federated Learning IoT Network - Docker Setup Guide

## Overview

This project implements multiple Federated Learning (FL) algorithms for IoT network fault detection using TensorFlow and Flower framework. The system supports both LSTM (RUL - Remaining Useful Life) and MLP (OFFSET - Offset Detection) models.

## Implemented FL Methods

### Existing Methods
1. **FedAvg** - Federated Averaging (baseline)
2. **FedAvgM** - FedAvg with Momentum
3. **FedOpt** - Adaptive Federated Optimization
4. **QFedAvg** - Quantized Federated Averaging

### New Methods (Added)
5. **MOON** - Model-Contrastive Federated Learning
   - Parameters: `mu` (contrastive loss weight), `temperature`
6. **FedALA** - Federated Averaging with Layer-wise Adaptive Learning
   - Parameters: `eta` (adaptive learning rate), `threshold`

## Architecture

The system uses a Docker-based multi-container architecture:

- **1 Server Container** (LSTM/RUL): `172.17.0.2`
- **1 MLP Server Container** (MLP/OFFSET): `172.17.0.8`
- **5 Client Containers**: `172.17.0.3` - `172.17.0.7`

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2.0+
- NVIDIA Docker Runtime (for GPU support)
- NVIDIA GPU with CUDA support

## Quick Start

### 1. Build and Start Containers

```bash
# Build the Docker image
docker-compose build

# Start all containers
docker-compose up -d

# Verify containers are running
docker-compose ps
```

### 2. Access Containers

```bash
# Access server container
docker exec -it fl_server bash

# Access client containers
docker exec -it fl_client1 bash
docker exec -it fl_client2 bash
# ... and so on
```

## Running Experiments

### Data Preparation

Ensure your data files are in the correct location:
```
fl_testbed/version2/data/transformed/
├── M3_5_0_ddf_LSTM.pkl
├── M3_5_1_ddf_LSTM.pkl
├── ...
└── combined_offset_misalignment_M3.csv
```

### Running LSTM (RUL) Experiments

#### Server Side (in fl_server container):

```bash
# For existing methods (FedAvg, FedAvgM, FedOpt, QFedAvg)
./server_execution_LSTM_M3_ALPHA.sh

# For new methods (MOON, FedALA)
./server_execution_LSTM_M3_NEW_METHODS.sh
```

#### Client Side (in each client container):

```bash
# Client 1 - Existing methods
./client1_execution_LSTM_M3_ALPHA.sh

# Client 1 - New methods
./client1_execution_LSTM_M3_NEW_METHODS.sh

# Repeat for clients 2-5
```

### Running MLP (OFFSET) Experiments

#### Server Side (in fl_mlp_server container):

```bash
# For existing methods
./server_execution_MLP_M3_ALPHA.sh

# For new methods
./server_execution_MLP_M3_NEW_METHODS.sh
```

#### Client Side (in each client container):

```bash
# Client 1 - Existing methods
./client1_execution_MLP_M3_ALPHA.sh

# Client 1 - New methods
./client1_execution_MLP_M3_NEW_METHODS.sh
```

## Parameter Configuration

### Common Parameters

- **alphas**: Dirichlet distribution parameter for non-IID data split
  - Values: `0.001, 0.01, 0.1, 0.02, 0.2, 0.005, 0.05, 0.5, 0.075, 1.0, 1000000.0`
  - Lower values = more non-IID (heterogeneous)
  - Higher values = more IID (homogeneous)

- **slr**: Server learning rate
  - Values: `0.001, 0.01, 1`

### Method-Specific Parameters

#### FedAvgM
- `momentum`: Server-side momentum (0.0, 0.7, 0.9)

#### FedOpt
- `tau`: Adaptive optimizer parameter (1e-7, 1e-8, 1e-9)

#### QFedAvg
- `q`: Fairness parameter (0.1, 0.2, 0.5)

#### MOON (New)
- `mu`: Contrastive loss weight (0.1, 0.5, 1.0, 5.0)
- `temperature`: Temperature for contrastive learning (0.1, 0.5, 1.0)

#### FedALA (New)
- `eta`: Adaptive learning rate (0.01, 0.1, 1.0)
- `threshold`: Layer-wise adaptation threshold (0.1, 0.5, 1.0)

## Docker Commands Reference

```bash
# Start all containers
docker-compose up -d

# Stop all containers
docker-compose down

# View logs
docker-compose logs -f

# View specific container logs
docker logs -f fl_server

# Restart a specific container
docker-compose restart fl_server

# Remove all containers and volumes
docker-compose down -v

# Rebuild containers
docker-compose build --no-cache
```

## Network Configuration

The containers communicate over a custom bridge network:

```yaml
networks:
  fl_network:
    subnet: 172.17.0.0/16
```

### IP Addresses

| Container | IP Address | Purpose |
|-----------|------------|---------|
| fl_server | 172.17.0.2 | LSTM/RUL Server |
| fl_client1 | 172.17.0.3 | Client 1 |
| fl_client2 | 172.17.0.4 | Client 2 |
| fl_client3 | 172.17.0.5 | Client 3 |
| fl_client4 | 172.17.0.6 | Client 4 |
| fl_client5 | 172.17.0.7 | Client 5 |
| fl_mlp_server | 172.17.0.8 | MLP/OFFSET Server |

## File Structure

```
FL_IoT_Network/
├── docker-compose.yml                          # Multi-container orchestration
├── Dockerfile                                  # Container image definition
├── requirements.txt                            # Python dependencies
│
├── server_execution_LSTM_M3_ALPHA.sh          # LSTM server script (existing methods)
├── server_execution_LSTM_M3_NEW_METHODS.sh    # LSTM server script (new methods)
├── server_execution_MLP_M3_ALPHA.sh           # MLP server script (existing methods)
├── server_execution_MLP_M3_NEW_METHODS.sh     # MLP server script (new methods)
│
├── client1_execution_LSTM_M3_ALPHA.sh         # LSTM client 1 (existing)
├── client1_execution_LSTM_M3_NEW_METHODS.sh   # LSTM client 1 (new)
├── client1_execution_MLP_M3_ALPHA.sh          # MLP client 1 (existing)
├── client1_execution_MLP_M3_NEW_METHODS.sh    # MLP client 1 (new)
│
└── fl_testbed/version2/
    ├── server/
    │   ├── federated_server_RUL_FedAvg.py
    │   ├── federated_server_RUL_FedAvgM.py
    │   ├── federated_server_RUL_FedOpt.py
    │   ├── federated_server_RUL_QFedAvg.py
    │   ├── federated_server_RUL_MOON.py       # To be implemented
    │   ├── federated_server_RUL_FedALA.py     # To be implemented
    │   ├── federated_server_OFFSET_*.py       # MLP variants
    │   └── ...
    └── client/
        ├── federated_client_RUL_*.py
        ├── federated_client_OFFSET_*.py
        └── ...
```

## Output Files

Results are saved in the project root with descriptive filenames:

- **Data Split**: `DATASPLIT_{alpha}_{model_type}_M3_{alpha}.txt`
- **Independent**: `out_server_M3_5_{client}_OFFSETM3_idp_{alpha}.txt4`
- **Federated**: `{MODEL}_TESLA_{METHOD}_{alpha}_slr_{slr}_{params}.txt`

Examples:
- `LSTM_TESLA_MOON_0.1_slr_0.01_mu_1.0_temp_0.5.txt`
- `MLP_TESLA_FedALA_0.5_slr_0.1_eta_0.1_threshold_0.5.txt`

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Port Already in Use

```bash
# Check if port 8080 is in use
lsof -i :8080

# Kill process if needed
kill -9 <PID>
```

### Container Communication Issues

```bash
# Check network
docker network inspect fl_iot_network_fl_network

# Ping between containers
docker exec fl_client1 ping 172.17.0.2
```

### Memory Issues

Adjust Docker daemon settings (`/etc/docker/daemon.json`):

```json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-shm-size": "2G"
}
```

## Performance Tips

1. **Reduce Parameter Space**: Start with fewer alpha values for faster testing
2. **Parallel Execution**: Run multiple clients simultaneously in different terminals
3. **Monitor Resources**: Use `docker stats` to monitor container resource usage
4. **GPU Utilization**: Monitor with `nvidia-smi -l 1`

## Citation

If you use this code in your research, please cite the original paper and mention the new FL methods implemented.

## License

See LICENSE file for details.

## Contact

For questions and support, please open an issue in the repository.
