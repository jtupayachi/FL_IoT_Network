#!/bin/bash

# Monitor running experiments

echo "🔍 Monitoring Running Experiments"
echo "================================="

# Check server processes
echo "🖥️  Server Processes:"
ps aux | grep "python.*server.py" | grep -v grep

echo ""
echo "💻 Client Processes:"
ps aux | grep "python.*client.py" | grep -v grep | wc -l

echo ""
echo "🌐 Open Ports:"
netstat -tuln | grep ":808"

echo ""
echo "📁 Recent Log Activity:"
for log_file in results/*/server.log; do
    if [ -f "$log_file" ]; then
        echo "📄 $log_file:"
        tail -1 "$log_file"
    fi
done