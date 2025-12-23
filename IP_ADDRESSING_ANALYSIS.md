# IP Address Handling in Federated Learning Setup

## Current Architecture

### How IPs are Currently Handled

#### 1. **Docker Network Configuration** (`docker-compose.yml`)
All containers are on a **shared bridge network** (`fl_network`) with subnet `172.18.0.0/16`:

| Container | IP Address | Purpose |
|-----------|------------|---------|
| `fl_server` | 172.18.0.2 | LSTM/RUL Server |
| `fl_client1` | 172.18.0.3 | Client 0 |
| `fl_client2` | 172.18.0.4 | Client 1 |
| `fl_client3` | 172.18.0.5 | Client 2 |
| `fl_client4` | 172.18.0.6 | Client 3 |
| `fl_client5` | 172.18.0.7 | Client 4 |
| `fl_mlp_server` | 172.18.0.8 | MLP/Offset Server |

#### 2. **Server-Side IP Handling**
The **server scripts** use the `-ip` parameter:
```bash
python3 fl_testbed/version2/server/federated_server_RUL_MOON.py -ip 172.18.0.2 ...
```

In the Python code:
```python
# Server binds to its own IP + port 8080
fl.server.start_server(
    server_address=str(ip) + ":8080",  # e.g., "172.18.0.2:8080"
    config=fl.server.ServerConfig(num_rounds=rounds),
    strategy=strategy,
)
```

#### 3. **Client-Side IP Handling**
The **client scripts** use the `-ip` parameter to specify the **server's IP**:
```bash
python3 fl_testbed/version2/client/federated_client_RUL_MOON.py -cn 1 -ip 172.18.0.2 ...
```

**⚠️ CRITICAL ISSUE FOUND:**
In the client Python code, the IP is **HARDCODED**:
```python
# This is WRONG - ignores the -ip parameter!
fl.client.start_numpy_client(
    server_address="172.17.0.2:8080",  # HARDCODED!
    client=client,
)
```

The `-ip` parameter is parsed but **NOT USED** in most client scripts!

## Problems with Current Setup

### 1. **Hardcoded Client Connection**
- Client scripts accept `-ip` parameter but don't use it
- Connection address is hardcoded to `172.17.0.2:8080` (wrong subnet!)
- Should use the parsed `ip` variable: `f"{ip}:8080"`

### 2. **Port Conflicts for Parallel Execution**
Currently:
- **All servers bind to port 8080** on their respective IPs
- This works because:
  - LSTM server: `172.18.0.2:8080`
  - MLP server: `172.18.0.8:8080`
  - Different IPs = No conflict

**For true parallel execution within same model type:**
- Cannot run multiple LSTM experiments simultaneously (same server IP/port)
- Cannot run multiple MLP experiments simultaneously (same server IP/port)

### 3. **Single Server Architecture**
Current design:
- 1 LSTM server handles all 5 clients for all experiments sequentially
- 1 MLP server handles all 5 clients for all experiments sequentially

## Solutions for Parallel Execution

### Option 1: Sequential Parameter Sweeps (Current Setup - WORKS)
**Status:** ✅ **Already working correctly**

Each experiment runs sequentially:
1. Server starts with specific parameters (e.g., MOON with mu=0.1, temp=0.5)
2. All 5 clients connect in parallel to that server
3. Training completes
4. Server restarts with next parameter set
5. Repeat

**Advantages:**
- No IP changes needed
- Uses existing container architecture
- All 5 clients train in parallel for each parameter set

**Limitations:**
- Different parameter sets run sequentially (not in parallel)
- One alpha value at a time

### Option 2: Multiple Servers for True Parallelization (Requires Changes)

To run multiple parameter combinations simultaneously:

#### A. Dynamic Port Allocation
```bash
# Server 1: MOON mu=0.1
python3 server.py -ip 172.18.0.2 -port 8080 -mu 0.1 ...

# Server 2: MOON mu=0.5 (different port)
python3 server.py -ip 172.18.0.2 -port 8081 -mu 0.5 ...

# Clients connect to different ports
python3 client.py -ip 172.18.0.2 -port 8080 ...  # for mu=0.1
python3 client.py -ip 172.18.0.2 -port 8081 ...  # for mu=0.5
```

#### B. Multiple Server Containers
Add to `docker-compose.yml`:
```yaml
services:
  lstm_server_1:
    ipv4_address: 172.18.0.10
  lstm_server_2:
    ipv4_address: 172.18.0.11
  # etc...
```

Each server container runs different parameter sets.

### Option 3: Container Scaling (Most Flexible)

Use Docker Compose scaling:
```bash
docker-compose up -d --scale server=5
```

With dynamic service discovery or environment variables.

## Recommended Fix for Current Setup

### Immediate Fix: Update Client Scripts

The client Python code needs fixing:

**Current (WRONG):**
```python
fl.client.start_numpy_client(
    server_address="172.17.0.2:8080",  # HARDCODED
    client=client,
)
```

**Should be:**
```python
fl.client.start_numpy_client(
    server_address=f"{ip}:8080",  # Use parsed IP
    client=client,
)
```

### Files That Need Fixing
All client scripts in `fl_testbed/version2/client/`:
- `federated_client_RUL_MOON.py`
- `federated_client_RUL_FedALA.py`
- `federated_client_RUL_StatAvg.py`
- `federated_client_RUL_DASHA.py`
- `federated_client_OFFSET_MOON.py`
- `federated_client_OFFSET_FedALA.py`
- `federated_client_OFFSET_StatAvg.py`
- `federated_client_OFFSET_DASHA.py`
- And all other federated_client_*.py files

## Current Orchestration Strategy

The `orchestrate_M3_NEW_METHODS.sh` script correctly:

1. ✅ Starts server in one container
2. ✅ Starts all 5 clients in parallel in separate containers
3. ✅ Each client runs in its own Docker container with unique IP
4. ✅ All clients connect to the same server (sequential parameter sweeps)
5. ✅ Uses synchronization (netcat) instead of sleep

**This is optimal for the current architecture** where:
- One server handles one parameter set at a time
- 5 clients train in parallel for that parameter set
- Different parameter sets are explored sequentially

## Summary

### What's Working:
✅ Docker network setup with unique IPs per container  
✅ Server IP binding via `-ip` parameter  
✅ Parallel client execution (5 clients at once)  
✅ Network connectivity between containers  

### What Needs Fixing:
❌ **Client scripts ignore `-ip` parameter** (hardcoded to wrong IP)  
❌ No mechanism for parallel parameter exploration (by design)  
❌ Port hardcoded to 8080 (limits flexibility)  

### For Your Current Needs:
The setup is **90% correct**. The main issue is the **hardcoded client connection IP**. Once fixed, your parallel execution will work perfectly for:
- 5 clients training simultaneously
- Sequential parameter sweep
- Efficient resource utilization

For **true parallel parameter exploration**, you'd need architectural changes (multiple servers, dynamic ports, or container scaling).
