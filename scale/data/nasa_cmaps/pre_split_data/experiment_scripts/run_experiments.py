#!/usr/bin/env python3
"""
NASA CMAPs Federated Learning Experiment Runner
Automatically generated script to run experiments across all configurations
"""

import os
import subprocess
import itertools

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIENT_CONFIGS = [25, 100]
ALPHAS = [0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

def run_experiment(num_clients, alpha):
    """Run experiment for specific configuration"""
    data_dir = os.path.join(BASE_DIR, f"../{num_clients}_clients/alpha_{alpha}")
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    print(f"\\n=== Running experiment: {num_clients} clients, alpha={alpha} ===")
    
    # Your federated learning command here
    # Example:
    # command = f"python your_fl_script.py --data_dir {data_dir} --num_clients {num_clients}"
    # subprocess.run(command, shell=True)
    
    print(f"Would run FL training with data from: {data_dir}")

def main():
    print("NASA CMAPs Federated Learning Experiments")
    
    for num_clients in CLIENT_CONFIGS:
        for alpha in ALPHAS:
            run_experiment(num_clients, alpha)

if __name__ == "__main__":
    main()
