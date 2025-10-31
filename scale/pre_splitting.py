import numpy as np
import pandas as pd
import os
from collections import defaultdict
import random
import shutil




import numpy as np
import pandas as pd
import os
from collections import defaultdict
import random
import shutil

class NASA_CMAPS_PreSplitter:
    def __init__(self, base_path):
        self.base_path = base_path
        self.conditions = [0.005, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.client_configs = [25, 100]
        
        # Dataset characteristics from the documentation
        self.dataset_info = {
            'FD001': {'train_units': 100, 'test_units': 100, 'conditions': 'ONE', 'fault_modes': 'ONE'},
            'FD002': {'train_units': 260, 'test_units': 259, 'conditions': 'SIX', 'fault_modes': 'ONE'},
            'FD003': {'train_units': 100, 'test_units': 100, 'conditions': 'ONE', 'fault_modes': 'TWO'},
            'FD004': {'train_units': 248, 'test_units': 249, 'conditions': 'SIX', 'fault_modes': 'TWO'}
        }
    
    def load_data(self):
        """Load all NASA CMAPs data files"""
        data = {}
        
        # Load training data from all FD files
        train_files = ['train_FD001.txt', 'train_FD002.txt', 'train_FD003.txt', 'train_FD004.txt']
        test_files = ['test_FD001.txt', 'test_FD002.txt', 'test_FD003.txt', 'test_FD004.txt']
        rul_files = ['RUL_FD001.txt', 'RUL_FD002.txt', 'RUL_FD003.txt', 'RUL_FD004.txt']
        
        # Load training data
        train_data = []
        total_cycles = 0
        total_units = 0
        
        for file in train_files:
            file_path = os.path.join(self.base_path, file)
            if os.path.exists(file_path):
                print(f"Loading {file_path}...")
                df = pd.read_csv(file_path, delim_whitespace=True, header=None)
                
                # Add metadata
                fd_code = file.replace('train_', '').replace('.txt', '')
                df['unit_id'] = df[0]  # First column is unit ID
                df['time_cycles'] = df[1]  # Second column is time in cycles
                df['fd_file'] = fd_code
                df['original_file'] = file
                train_data.append(df)
                
                # Calculate statistics
                units_in_file = df[0].nunique()
                cycles_in_file = len(df)
                total_units += units_in_file
                total_cycles += cycles_in_file
                
                print(f"  {fd_code}: {units_in_file} units, {cycles_in_file} cycles")
        
        print(f"\nTOTAL: {total_units} units, {total_cycles} cycles across all datasets")
        
        # Load test data and RUL for reference
        test_data = []
        rul_data = []
        
        for test_file, rul_file in zip(test_files, rul_files):
            test_path = os.path.join(self.base_path, test_file)
            rul_path = os.path.join(self.base_path, rul_file)
            
            if os.path.exists(test_path):
                test_df = pd.read_csv(test_path, delim_whitespace=True, header=None)
                test_df['unit_id'] = test_df[0]
                test_df['fd_file'] = test_file.replace('test_', '').replace('.txt', '')
                test_data.append(test_df)
            
            if os.path.exists(rul_path):
                rul_df = pd.read_csv(rul_path, header=None, names=['RUL'])
                rul_df['unit_id'] = rul_df.index + 1
                rul_df['fd_file'] = rul_file.replace('RUL_', '').replace('.txt', '')
                rul_data.append(rul_df)
        
        return {
            'train': pd.concat(train_data, ignore_index=True) if train_data else None,
            'test': pd.concat(test_data, ignore_index=True) if test_data else None,
            'rul': pd.concat(rul_data, ignore_index=True) if rul_data else None,
            'total_units': total_units,
            'total_cycles': total_cycles
        }
    
    def create_unit_mapping(self, data):
        """Create mapping of units to their characteristics"""
        units_info = {}
        
        # Group by unit_id and fd_file
        for (unit_id, fd_file), group in data['train'].groupby(['unit_id', 'fd_file']):
            # Calculate RUL for each data point
            max_cycle = group['time_cycles'].max()
            rul_series = max_cycle - group['time_cycles']
            
            units_info[(unit_id, fd_file)] = {
                'size': len(group),
                'fd_file': fd_file,
                'unit_id': unit_id,
                'data': group,
                'max_cycle': max_cycle,
                'total_cycles': len(group),
                'operating_conditions': self._extract_operating_conditions(group),
                'sensor_characteristics': self._extract_sensor_characteristics(group),
                'rul_series': rul_series
            }
        
        print(f"Created mapping for {len(units_info)} units")
        return units_info
    
    def _extract_operating_conditions(self, unit_data):
        """Extract operating conditions from unit data"""
        op_conditions = {
            'op_setting_1_mean': unit_data[2].mean(),
            'op_setting_2_mean': unit_data[3].mean(),
            'op_setting_3_mean': unit_data[4].mean(),
            'op_setting_1_std': unit_data[2].std(),
            'op_setting_2_std': unit_data[3].std(),
            'op_setting_3_std': unit_data[4].std(),
        }
        return op_conditions
    
    def _extract_sensor_characteristics(self, unit_data):
        """Extract characteristics from sensor measurements"""
        sensor_stats = {}
        # Sensors are columns 5-25 (sensor measurements 1-21)
        for sensor_idx in range(5, 26):
            sensor_data = unit_data[sensor_idx]
            sensor_stats[f'sensor_{sensor_idx-4}'] = {
                'mean': sensor_data.mean(),
                'std': sensor_data.std(),
                'trend': self._calculate_trend(sensor_data.values)
            }
        return sensor_stats
    
    def _calculate_trend(self, data):
        """Calculate linear trend for degradation analysis"""
        if len(data) < 2:
            return 0
        try:
            x = np.arange(len(data))
            slope = np.polyfit(x, data, 1)[0]
            return slope
        except:
            return 0
    
    def dirichlet_split(self, units_info, alpha, num_clients):
        """Split data using Dirichlet distribution"""
        units_list = list(units_info.keys())
        num_units = len(units_list)
        
        # Create client distributions using Dirichlet
        client_distributions = np.random.dirichlet([alpha] * num_clients, num_units)
        
        # Assign units to clients based on probabilities
        client_assignments = defaultdict(list)
        
        for i, unit in enumerate(units_list):
            probabilities = client_distributions[i]
            client_id = np.random.choice(num_clients, p=probabilities)
            client_assignments[client_id].append(unit)
        
        return client_assignments
    
    def create_all_splits(self, units_info):
        """Create all splits for different client numbers and alpha values"""
        all_splits = {}
        
        for num_clients in self.client_configs:
            print(f"\n=== Creating splits for {num_clients} clients ===")
            client_splits = {}
            
            for alpha in self.conditions:
                print(f"Creating split for alpha={alpha}")
                client_assignments = self.dirichlet_split(units_info, alpha, num_clients)
                
                client_splits[alpha] = {
                    'client_assignments': client_assignments,
                    'alpha': alpha,
                    'num_clients': num_clients
                }
                
                # Print statistics
                self._print_split_statistics(client_assignments, units_info, alpha, num_clients)
            
            all_splits[num_clients] = client_splits
        
        return all_splits
    
    def _print_split_statistics(self, client_assignments, units_info, alpha, num_clients):
        """Print statistics for the split"""
        print(f"\n--- {num_clients} Clients - Alpha = {alpha} ---")
        
        client_samples = []
        client_units = []
        fd_distributions = []
        
        for client_id, units in client_assignments.items():
            total_samples = sum(units_info[unit]['size'] for unit in units)
            total_units = len(units)
            fd_files = [units_info[unit]['fd_file'] for unit in units]
            fd_dist = {fd: fd_files.count(fd) for fd in set(fd_files)}
            
            client_samples.append(total_samples)
            client_units.append(total_units)
            fd_distributions.append(fd_dist)
        
        print(f"Total samples distributed: {sum(client_samples)}")
        print(f"Sample distribution: min={min(client_samples)}, max={max(client_samples)}, mean={np.mean(client_samples):.1f}")
        print(f"Unit distribution: min={min(client_units)}, max={max(client_units)}, mean={np.mean(client_units):.1f}")
    
    def create_directory_structure(self, base_output_dir):
        """Create the organized directory structure"""
        structures = {}
        
        for num_clients in self.client_configs:
            client_dir = os.path.join(base_output_dir, f"{num_clients}_clients")
            
            # Create main directory
            if not os.path.exists(client_dir):
                os.makedirs(client_dir)
            
            # Create directories for each alpha
            for alpha in self.conditions:
                alpha_dir = os.path.join(client_dir, f"alpha_{alpha}")
                if not os.path.exists(alpha_dir):
                    os.makedirs(alpha_dir)
                
                # Create client directories
                for client_id in range(num_clients):
                    client_subdir = os.path.join(alpha_dir, f"client_{client_id}")
                    if not os.path.exists(client_subdir):
                        os.makedirs(client_subdir)
            
            structures[num_clients] = client_dir
        
        return structures
    
    def save_client_data(self, all_splits, units_info, base_output_dir):
        """Save pre-split data for all configurations"""
        directory_structures = self.create_directory_structure(base_output_dir)
        
        for num_clients, splits in all_splits.items():
            client_dir = directory_structures[num_clients]
            print(f"\nSaving data for {num_clients} clients in {client_dir}...")
            
            for alpha, split_data in splits.items():
                alpha_dir = os.path.join(client_dir, f"alpha_{alpha}")
                client_assignments = split_data['client_assignments']
                
                # Save mapping file
                self._save_mapping_file(alpha_dir, client_assignments, units_info, alpha, num_clients)
                
                # Save client data files
                for client_id, units in client_assignments.items():
                    client_data_dir = os.path.join(alpha_dir, f"client_{client_id}")
                    self._save_single_client_data(client_data_dir, units, units_info, client_id)
        
        print(f"\nAll data successfully pre-split and saved!")
        return directory_structures
    
    def _save_mapping_file(self, alpha_dir, client_assignments, units_info, alpha, num_clients):
        """Save mapping file for the split"""
        mapping_file = os.path.join(alpha_dir, "split_mapping.txt")
        
        with open(mapping_file, 'w') as f:
            f.write(f"# NASA CMAPs Pre-Split Data\n")
            f.write(f"# Alpha: {alpha}\n")
            f.write(f"# Number of clients: {num_clients}\n")
            f.write(f"# Total units: {sum(len(units) for units in client_assignments.values())}\n")
            f.write("# Format: client_id,unit_id,fd_file,data_size,original_file\n")
            
            for client_id, units in client_assignments.items():
                for unit in units:
                    unit_info = units_info[unit]
                    f.write(f"{client_id},{unit_info['unit_id']},{unit_info['fd_file']},{unit_info['size']},{unit_info['data']['original_file'].iloc[0]}\n")
    
    # def _save_single_client_data(self, client_dir, units, units_info, client_id):
    #     """Save data for a single client"""
    #     client_data = []
        
    #     for unit in units:
    #         unit_info = units_info[unit]
    #         client_data.append(unit_info['data'])
        
    #     if client_data:
    #         client_df = pd.concat(client_data, ignore_index=True)
    #         # Save without the additional columns we added
    #         data_columns = list(range(26))  # Original 26 columns
    #         client_df[data_columns].to_csv(
    #             os.path.join(client_dir, "train_data.txt"), 
    #             sep=' ', 
    #             header=False, 
    #             index=False
    #         )
            
    #         # Save client info
    #         with open(os.path.join(client_dir, "client_info.txt"), 'w') as f:
    #             f.write(f"client_id: {client_id}\n")
    #             f.write(f"num_units: {len(units)}\n")
    #             f.write(f"total_samples: {len(client_df)}\n")
    #             f.write(f"units: {','.join([f'({u[0]},{u[1]})' for u in units])}\n")
    
    # def _save_single_client_data(self, client_dir, units, units_info, client_id):
    #     """Save data for a single client"""
    #     client_data = []
        
    #     for unit in units:
    #         unit_info = units_info[unit]
    #         client_data.append(unit_info['data'])
        
    #     if client_data:
    #         client_df = pd.concat(client_data, ignore_index=True)
            
    #         # ✅ FIXED: Save ALL 26 original columns
    #         # Do NOT include the metadata columns we added (unit_id, time_cycles, fd_file, original_file)
    #         data_columns = list(range(26))  # Columns 0-25
    #         client_df[data_columns].to_csv(
    #             os.path.join(client_dir, "train_data.txt"), 
    #             sep=' ', 
    #             header=False, 
    #             index=False,
    #             float_format='%.4f'  # Match original precision
    #         )
            
    #         # Save client info with RUL statistics
    #         with open(os.path.join(client_dir, "client_info.txt"), 'w') as f:
    #             f.write(f"client_id: {client_id}\n")
    #             f.write(f"num_units: {len(units)}\n")
    #             f.write(f"total_samples: {len(client_df)}\n")
    #             f.write(f"units: {','.join([f'({u[0]},{u[1]})' for u in units])}\n")
                
    #             # Add RUL statistics
    #             for unit in units:
    #                 unit_info = units_info[unit]
    #                 f.write(f"\nUnit ({unit[0]},{unit[1]}):\n")
    #                 f.write(f"  max_cycle: {unit_info['max_cycle']}\n")
    #                 f.write(f"  total_cycles: {unit_info['total_cycles']}\n")


    def _save_single_client_data(self, client_dir, units, units_info, client_id):
        """Save train/test split for a single client"""
        client_data = []
        
        # Collect all data for this client's units
        for unit in units:
            unit_info = units_info[unit]
            client_data.append(unit_info['data'])
        
        if not client_data:
            return
        
        # Combine all units for this client
        client_df = pd.concat(client_data, ignore_index=True)
        
        # Extract features (columns 2-25: settings + sensors)
        # Column 0: unit_id, Column 1: cycle
        feature_columns = list(range(2, 26))  # Settings + sensors
        X = client_df[feature_columns].values
        
        # Calculate RUL (following notebook's approach)
        unit_ids = client_df[0].values
        time_cycles = client_df[1].values
        y = self._calculate_rul(unit_ids, time_cycles)
        
        # ✅ Split into train (80%) and test (20%)
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            shuffle=True
        )
        
        # Save train data
        train_file = os.path.join(client_dir, "train_data.txt")
        np.savetxt(train_file, X_train, fmt='%.6f', delimiter=' ')
        
        train_labels_file = os.path.join(client_dir, "train_labels.txt")
        np.savetxt(train_labels_file, y_train, fmt='%.6f')
        
        # Save test data
        test_file = os.path.join(client_dir, "test_data.txt")
        np.savetxt(test_file, X_test, fmt='%.6f', delimiter=' ')
        
        test_labels_file = os.path.join(client_dir, "test_labels.txt")
        np.savetxt(test_labels_file, y_test, fmt='%.6f')
        
        # Save metadata
        info_file = os.path.join(client_dir, "client_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Client ID: {client_id}\n")
            f.write(f"Units assigned: {sorted(units)}\n")
            f.write(f"Total samples: {len(client_df)}\n")
            f.write(f"Train samples: {len(X_train)}\n")
            f.write(f"Test samples: {len(X_test)}\n")
            f.write(f"Features: {X_train.shape[1]}\n")
        
        print(f"  ✅ Client {client_id}: {len(units)} units, {len(X_train)} train, {len(X_test)} test samples")


    def _calculate_rul(self, unit_ids, time_cycles):
        """Calculate RUL following notebook's ratio approach"""
        rul = np.zeros(len(unit_ids))
        
        for unit_id in np.unique(unit_ids):
            mask = (unit_ids == unit_id)
            max_cycle = time_cycles[mask].max()
            # Ratio: current_cycle / max_cycle (matches notebook)
            rul[mask] = time_cycles[mask] / max_cycle
        
        return rul
        
    def create_experiment_scripts(self, directory_structures):
        """Create scripts to help with experiment execution"""
        scripts_dir = os.path.join(os.path.dirname(list(directory_structures.values())[0]), "experiment_scripts")
        if not os.path.exists(scripts_dir):
            os.makedirs(scripts_dir)
        
        # Create run script
        run_script = os.path.join(scripts_dir, "run_experiments.py")
        with open(run_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
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
    
    print(f"\\\\n=== Running experiment: {num_clients} clients, alpha={alpha} ===")
    
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
''')
        
        # Create analysis script
        analysis_script = os.path.join(scripts_dir, "analyze_splits.py")
        with open(analysis_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
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
                ax.set_title(f'Alpha = {alpha}\\\\nSamples per Client')
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
''')
        
        # Make scripts executable
        os.chmod(run_script, 0o755)
        os.chmod(analysis_script, 0o755)
        
        print(f"Experiment scripts created in: {scripts_dir}")

def main():
    # Configuration /mnt/ceph_drive/FL_IoT_Network/scale/data/nasa-cmaps.zip
    base_path = "/mnt/ceph_drive/FL_IoT_Network/scale/data/nasa_cmaps/CMaps" # Cjamge this
    output_dir = "/mnt/ceph_drive/FL_IoT_Network/scale/data/nasa_cmaps/pre_split_data" # Cjamge this
    
    # Initialize splitter
    splitter = NASA_CMAPS_PreSplitter(base_path)
    
    # Load data
    print("Loading NASA CMAPs data...")
    data = splitter.load_data()
    
    if data['train'] is None:
        print("Error: Could not load training data")
        return
    
    print(f"Loaded {len(data['train'])} training samples")
    
    # Create unit mapping
    units_info = splitter.create_unit_mapping(data)
    
    # Create all splits
    print("Creating all splits...")
    all_splits = splitter.create_all_splits(units_info)
    
    # Save pre-split data
    directory_structures = splitter.save_client_data(all_splits, units_info, output_dir)
    
    # Create experiment scripts
    splitter.create_experiment_scripts(directory_structures)
    
    print(f"\n=== Pre-splitting Complete ===")
    for num_clients, client_dir in directory_structures.items():
        print(f"{num_clients} clients: {client_dir}")
    
    print(f"\nDirectory structure:")
    print(f"pre_split_data/")
    for num_clients in splitter.client_configs:
        print(f"  {num_clients}_clients/")
        for alpha in splitter.conditions:
            print(f"    alpha_{alpha}/")
            print(f"      client_0/")
            print(f"      client_1/")
            print(f"      ...")
            print(f"      client_{num_clients-1}/")
            print(f"      split_mapping.txt")

if __name__ == "__main__":
    main()

