#!/usr/bin/env python3
"""
NASA CMAPs Split Analysis
Analyze the data distribution across clients and configurations
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import glob

def analyze_configuration(config_dir):
    """Analyze a single configuration directory"""
    alpha_dirs = glob.glob(os.path.join(config_dir, "alpha_*"))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, alpha_dir in enumerate(alpha_dirs):
        if i >= len(axes):
            break
            
        alpha = os.path.basename(alpha_dir).replace('alpha_', '')
        mapping_file = os.path.join(alpha_dir, "split_mapping.txt")
        
        if os.path.exists(mapping_file):
            # Read and analyze mapping
            data = []
            with open(mapping_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        data.append({
                            'client_id': int(parts[0]),
                            'unit_id': parts[1],
                            'fd_file': parts[2],
                            'size': int(parts[3])
                        })
            
            if data:
                df = pd.DataFrame(data)
                client_stats = df.groupby('client_id').agg({
                    'size': ['sum', 'count'],
                    'fd_file': lambda x: x.nunique()
                }).round(2)
                
                ax = axes[i]
                client_stats[('size', 'sum')].plot(kind='bar', ax=ax)
                ax.set_title(f'Alpha = {alpha}\\nSamples per Client')
                ax.set_xlabel('Client ID')
                ax.set_ylabel('Number of Samples')
                ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Configuration: {os.path.basename(config_dir)}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(config_dir, 'distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    client_dirs = glob.glob(os.path.join(base_dir, "../*_clients"))
    
    for client_dir in client_dirs:
        print(f"Analyzing {client_dir}...")
        analyze_configuration(client_dir)
    
    print("Analysis complete! Check the PNG files in each configuration directory.")

if __name__ == "__main__":
    main()
