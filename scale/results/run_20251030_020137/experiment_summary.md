# NASA FL Experiments Summary

## Run Information
- **Run ID**: run_20251030_020137
- **Timestamp**: Thu Oct 30 02:03:18 EDT 2025
- **Total Experiments**: 1
- **Run Directory**: /mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251030_020137

## Experiment Parameters
- **Strategies**: fedavg
- **Client Counts**: 25
- **Alpha Values**: 0.1
- **Server Rounds**: 10

## Experiments Status

| Experiment ID | Strategy | Clients | Alpha | Port | Status | Results |
|---------------|----------|---------|-------|------|--------|---------|

## Summary
- **Completed**: 0/1
- **Success Rate**: 0%

## Quick Analysis Commands
```bash
# Check completion status
find "/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251030_020137" -name "round_metrics.csv" | wc -l

# View recent results
find "/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251030_020137" -name "round_metrics.csv" -exec dirname {} \; | while read dir; do
    echo "=== $(basename $dir) ==="
    tail -1 "$dir/round_metrics.csv"
done

# Monitor disk usage
du -sh "/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251030_020137"
```

## Directory Structure
```
/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251030_020137/
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
