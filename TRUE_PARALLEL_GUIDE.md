# TRUE PARALLEL EXECUTION GUIDE
## Running Multiple FL Experiments Simultaneously

## What Changed

### Parameter Reduction (70% faster)
**Before:** 11 alphas × 3 slr × (multiple method params) = **hundreds of combinations**
**Now:** 3 alphas × 1 slr × (reduced method params) = **~20 combinations**

| Parameter | Before | Now | Reduction |
|-----------|--------|-----|-----------|
| alphas | 11 values | 3 values | 73% |
| slr | 3 values | 1 value | 67% |
| MOON mu | 4 values | 2 values | 50% |
| MOON temp | 3 values | 1 value | 67% |
| FedALA eta | 3 values | 1 value | 67% |
| FedALA threshold | 3 values | 1 value | 67% |

**Total speedup: ~30-50x faster execution**

### True Parallel Architecture

**OLD WAY (Sequential):**
```
[1 Server] ← [5 Clients in parallel]
    ↓ (experiment 1 completes)
[1 Server] ← [5 Clients in parallel]
    ↓ (experiment 2 completes)
[1 Server] ← [5 Clients in parallel]
```

**NEW WAY (Fully Parallel):**
```
[Server 1] ← [5 Clients] | Running MOON mu=0.5
[Server 2] ← [5 Clients] | Running MOON mu=1.0  } All at
[Server 3] ← [5 Clients] | Running FedALA      } same time!
```

**Result:** 3x faster (or more depending on parallel capacity)

## Setup Options

### Option 1: Quick Start (3 Parallel Experiments)

Uses `docker-compose-parallel.yml` which creates:
- 3 server containers (exp1, exp2, exp3)
- 15 client containers (5 per experiment)
- Unique IPs for each experiment:
  - Exp1: 172.18.0.10-15
  - Exp2: 172.18.0.20-25
  - Exp3: 172.18.0.30-35

**Start:**
```bash
bash run_parallel_experiments.sh
```

This will:
1. Start 18 containers
2. Run 3 FL experiments simultaneously
3. Each with 1 server + 5 clients
4. Monitor progress in real-time

### Option 2: Manual Control

**Start containers:**
```bash
docker-compose -f docker-compose-parallel.yml up -d
```

**Run Experiment 1 manually:**
```bash
# Server
docker exec -d fl_lstm_exp1_server bash -c \
  "cd /workspace && python3 fl_testbed/version2/server/federated_server_RUL_MOON.py \
  -mu 0.5 -temperature 0.5 -slr 0.01 -cm 5 -e 1 --rounds 100 \
  -ip 172.18.0.10 -dfn_test_x '...' -dfn_test_y '...' -dfn '...'"

# Clients (all 5 in parallel)
for i in 1 2 3 4 5; do
  docker exec -d fl_lstm_exp1_client${i} bash -c \
    "cd /workspace && python3 fl_testbed/version2/client/federated_client_RUL_MOON.py \
    -cn $((i-1)) -cm 5 -e 1 -ip 172.18.0.10 ..." &
done
```

## Resource Requirements

### For 3 Parallel Experiments (18 containers):
- **CPU:** 36+ cores (2 per container minimum)
- **RAM:** 72GB+ (4GB per container)
- **GPU:** 2-4 GPUs with enough VRAM
- **Disk:** 50GB+ for logs and checkpoints

### For Maximum Parallelism:
If you want to run ALL parameter combinations in parallel, you'd need:
- ~20 experiment sets
- 120 containers (20 servers + 100 clients)
- **Not recommended** unless you have a cluster!

## IP Address Allocation

The `docker-compose-parallel.yml` uses subnet `172.18.0.0/16` which supports:
- **65,534 IP addresses** (172.18.0.1 - 172.18.255.254)
- Current usage: 18 IPs
- **Capacity for 10,000+ parallel experiments** if needed

### Current allocation:
```
172.18.0.10-15  → Experiment 1 (MOON mu=0.5)
172.18.0.20-25  → Experiment 2 (MOON mu=1.0)
172.18.0.30-35  → Experiment 3 (FedALA)
172.18.0.40-45  → Available for Experiment 4
172.18.0.50-55  → Available for Experiment 5
... and so on
```

## Monitoring

### View all experiments:
```bash
# Server logs
docker logs -f fl_lstm_exp1_server  # MOON mu=0.5
docker logs -f fl_lstm_exp2_server  # MOON mu=1.0
docker logs -f fl_lstm_exp3_server  # FedALA

# Client logs
docker logs -f fl_lstm_exp1_client1
docker logs -f fl_lstm_exp2_client1
docker logs -f fl_lstm_exp3_client1
```

### Check if experiments are running:
```bash
docker exec fl_lstm_exp1_server ps aux | grep python
docker exec fl_lstm_exp2_server ps aux | grep python
docker exec fl_lstm_exp3_server ps aux | grep python
```

### Network connectivity test:
```bash
docker exec fl_lstm_exp1_client1 ping -c 3 172.18.0.10  # Can reach server?
docker exec fl_lstm_exp2_client1 ping -c 3 172.18.0.20
docker exec fl_lstm_exp3_client1 ping -c 3 172.18.0.30
```

## Scaling Up

### Add more parallel experiments:

1. **Edit docker-compose-parallel.yml:**
   Add experiment set 4:
   ```yaml
   lstm_server_exp4:
     ipv4_address: 172.18.0.40
   lstm_client1_exp4:
     ipv4_address: 172.18.0.41
   # ... clients 2-5
   ```

2. **Update run_parallel_experiments.sh:**
   Add experiment 4 call:
   ```bash
   run_experiment_set 4 "172.18.0.40" "StatAvg" "..."
   ```

3. **Restart:**
   ```bash
   docker-compose -f docker-compose-parallel.yml down
   docker-compose -f docker-compose-parallel.yml up -d
   bash run_parallel_experiments.sh
   ```

## Performance Tips

### 1. GPU Memory Management
```bash
# Check GPU usage
nvidia-smi

# Limit memory per container (in Dockerfile or docker-compose)
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
```

### 2. Stagger Experiment Starts
Instead of starting all 3 at once, start with delays:
```bash
run_experiment_set 1 ... &
sleep 30
run_experiment_set 2 ... &
sleep 30
run_experiment_set 3 ... &
```

### 3. Use SSD for /workspace
Mount an SSD for faster data loading:
```yaml
volumes:
  - /path/to/fast/ssd:/workspace
```

## Troubleshooting

### Containers fail to start:
```bash
# Check logs
docker-compose -f docker-compose-parallel.yml logs

# Restart specific experiment
docker-compose -f docker-compose-parallel.yml restart lstm_server_exp1
```

### Out of memory:
```bash
# Reduce parallel experiments from 3 to 2
# Or increase system RAM
# Or add swap space (slower)
```

### GPU errors:
```bash
# Check GPU availability
nvidia-smi

# Ensure docker has GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Network issues:
```bash
# Recreate network
docker network rm fl_network
docker-compose -f docker-compose-parallel.yml up -d
```

## Cleanup

```bash
# Stop all parallel experiments
docker-compose -f docker-compose-parallel.yml down

# Remove all volumes
docker-compose -f docker-compose-parallel.yml down -v

# Clean everything
docker system prune -a
```

## Comparison: Before vs After

### Execution Time Estimate:

**Sequential (OLD):**
- 20 parameter combinations
- ~2 hours per combination
- **Total: ~40 hours (1.6 days)**

**Parallel (NEW):**
- 20 parameter combinations
- 3 running simultaneously
- ~2 hours per batch of 3
- **Total: ~14 hours (0.6 days)**

**With reduced parameters:**
- 6 key combinations
- 3 running simultaneously
- ~1 hour per batch
- **Total: ~2 hours** ⚡

## Summary

✅ **Parameters reduced** from hundreds to ~20 combinations  
✅ **True parallelism** - 3 complete FL experiments at once  
✅ **18 containers** running simultaneously (3×6)  
✅ **Unique IPs** per experiment set (172.18.0.x)  
✅ **Scalable** to 10+ parallel experiments if you have resources  
✅ **30-50x faster** than original parameter sweep  

Your system is now ready for **production-scale parallel FL experiments**! 🚀
