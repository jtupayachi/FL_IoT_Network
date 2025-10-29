# NASA FL Experiments Summary

## Run Information
- **Run ID**: run_20251029_151745
- **Timestamp**: Wed Oct 29 15:19:35 EDT 2025
- **Total Experiments**: 1
- **Run Directory**: /mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251029_151745

## Experiment Parameters
- **Strategies**: fedavg
- **Client Counts**: 25
- **Alpha Values**: 0.1
- **Server Rounds**: 10

## Experiments Status

| Experiment ID | Strategy | Clients | Alpha | Port | Status | Results |
|---------------|----------|---------|-------|------|--------|---------|
| nasa_25c_alpha_0.1_fedavg | fedavg | 25 | 0.1 | 8686 | ⚠️ Partial | - |

## Summary
- **Completed**: 0/1
- **Success Rate**: 0%

## Quick Analysis Commands
```bash
# Check completion status
find "/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251029_151745" -name "round_metrics.csv" | wc -l

# View recent results
find "/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251029_151745" -name "round_metrics.csv" -exec dirname {} \; | while read dir; do
    echo "=== $(basename $dir) ==="
    tail -1 "$dir/round_metrics.csv"
done

# Monitor disk usage
du -sh "/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251029_151745"
```

## Directory Structure
```
/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251029_151745/
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
```
