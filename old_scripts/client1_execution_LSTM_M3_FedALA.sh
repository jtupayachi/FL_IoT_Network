#!/bin/bash
# Auto-generated client1 script for FedALA
# This client waits for server and participates in all parameter sweeps
while ! nc -z 172.18.2.10 8080 2>/dev/null; do
    sleep 2
done
echo "Client 1 connected to FedALA server at 172.18.2.10"
# Client code handles all parameter combinations automatically
