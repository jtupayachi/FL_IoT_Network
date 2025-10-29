#!/bin/bash

# Quick analysis of experiment results

RESULTS_BASE="results"

echo "📊 Experiment Results Analysis"
echo "=============================="

# Count completed experiments
completed=$(find "$RESULTS_BASE" -name "round_metrics.csv" | wc -l)
total=$(find "$RESULTS_BASE" -maxdepth 1 -type d | tail -n +2 | wc -l)

echo "Completed: $completed/$total experiments"

# Show recent results
echo ""
echo "📈 Recent Results (last 5 rounds):"
for exp_dir in "$RESULTS_BASE"/*/; do
    if [ -f "$exp_dir/metrics/round_metrics.csv" ]; then
        exp_name=$(basename "$exp_dir")
        echo ""
        echo "🔬 $exp_name:"
        tail -5 "$exp_dir/metrics/round_metrics.csv" | cut -d',' -f1-6 | column -t -s','
    fi
done