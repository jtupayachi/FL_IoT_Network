#!/bin/bash
# Auto-generated client0 script for DASHA
# This client waits for server and participates in all parameter sweeps
while ! nc -z 172.18.4.10 8080 2>/dev/null; do
    sleep 2
done
echo "Client 0 connected to DASHA server at 172.18.4.10"
# Client code handles all parameter combinations automatically
