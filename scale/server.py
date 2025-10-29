
# # # import flwr as fl
# # # import torch
# # # import argparse
# # # import json
# # # import os
# # # from typing import Dict, List, Tuple, Optional
# # # import numpy as np
# # # from flwr.common import Parameters, Scalar
# # # from flwr.server.strategy import FedAvg, FedProx
# # # import torch.nn as nn
# # # import pandas as pd
# # # from datetime import datetime
# # # import csv
# # # import sys

# # # class NASAStrategy:
# # #     """Custom strategy wrapper for NASA experiments"""
    
# # #     def __init__(self, strategy_name: str = "fedavg", fraction_fit: float = 1.0, 
# # #                  fraction_evaluate: float = 1.0, min_fit_clients: int = 2,
# # #                  min_evaluate_clients: int = 2, min_available_clients: int = 2,
# # #                  proximal_mu: float = 0.1):
        
# # #         self.strategy_name = strategy_name.lower()
        
# # #         if self.strategy_name == "fedprox":
# # #             self.strategy = FedProx(
# # #                 fraction_fit=fraction_fit,
# # #                 fraction_evaluate=fraction_evaluate,
# # #                 min_fit_clients=min_fit_clients,
# # #                 min_evaluate_clients=min_evaluate_clients,
# # #                 min_available_clients=min_available_clients,
# # #                 proximal_mu=proximal_mu
# # #             )
# # #         else:  # Default to FedAvg
# # #             self.strategy = FedAvg(
# # #                 fraction_fit=fraction_fit,
# # #                 fraction_evaluate=fraction_evaluate,
# # #                 min_fit_clients=min_fit_clients,
# # #                 min_evaluate_clients=min_evaluate_clients,
# # #                 min_available_clients=min_available_clients
# # #             )
    
# # #     def get_strategy(self):
# # #         return self.strategy

# # # class MetricsCollector:
# # #     """Collects and saves metrics from all rounds"""
    
# # #     def __init__(self, results_dir: str, experiment_id: str):
# # #         self.results_dir = results_dir
# # #         self.experiment_id = experiment_id
# # #         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
# # #         # Create results directory
# # #         os.makedirs(results_dir, exist_ok=True)
        
# # #         # Define file paths
# # #         self.round_metrics_path = os.path.join(results_dir, f"round_metrics_{self.timestamp}.csv")
# # #         self.client_metrics_path = os.path.join(results_dir, f"client_metrics_{self.timestamp}.csv")
# # #         self.eval_metrics_path = os.path.join(results_dir, f"eval_metrics_{self.timestamp}.csv")
        
# # #         # Initialize CSV files
# # #         self._initialize_metrics_files()
        
# # #         # Storage for metrics
# # #         self.round_metrics = []
# # #         self.client_metrics = []
# # #         self.eval_metrics = []
    
# # #     def _initialize_metrics_files(self):
# # #         """Initialize CSV files with headers"""
        
# # #         # Round metrics (aggregated per round)
# # #         round_headers = [
# # #             "timestamp", "round", "total_clients", "fit_clients", "eval_clients",
# # #             "avg_train_loss", "avg_val_loss", "avg_train_rmse", "avg_val_rmse",
# # #             "avg_train_r2", "avg_val_r2", "centralized_loss", "algorithm"
# # #         ]
        
# # #         # Client metrics (per client per round)
# # #         client_headers = [
# # #             "timestamp", "round", "client_id", "samples", "train_loss", "val_loss",
# # #             "train_rmse", "val_rmse", "train_r2", "val_r2", "algorithm"
# # #         ]
        
# # #         # Evaluation metrics (centralized evaluation)
# # #         eval_headers = [
# # #             "timestamp", "round", "loss", "rmse", "mse", "mae", "r2", "algorithm"
# # #         ]
        
# # #         # Write headers
# # #         for path, headers in [
# # #             (self.round_metrics_path, round_headers),
# # #             (self.client_metrics_path, client_headers),
# # #             (self.eval_metrics_path, eval_headers)
# # #         ]:
# # #             with open(path, 'w', newline='') as f:
# # #                 writer = csv.DictWriter(f, fieldnames=headers)
# # #                 writer.writeheader()
    
# # #     def collect_fit_metrics(self, round_num: int, results: List[Tuple], failures: List[BaseException]):
# # #         """Collect metrics from fit results"""
# # #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
# # #         total_samples = 0
# # #         train_losses, val_losses = [], []
# # #         train_rmses, val_rmses = [], []
# # #         train_r2s, val_r2s = [], []
        
# # #         for client_proxy, fit_res in results:
# # #             client_id = fit_res.metrics.get("client_id", "unknown")
# # #             num_samples = fit_res.num_examples
# # #             train_loss = fit_res.metrics.get("train_loss", 0)
# # #             val_loss = fit_res.metrics.get("val_loss", 0)
# # #             train_rmse = fit_res.metrics.get("train_rmse", 0)
# # #             val_rmse = fit_res.metrics.get("val_rmse", 0)
# # #             train_r2 = fit_res.metrics.get("train_r2", 0)
# # #             val_r2 = fit_res.metrics.get("val_r2", 0)
            
# # #             # Store client-level metrics
# # #             client_metrics = {
# # #                 "timestamp": timestamp,
# # #                 "round": round_num,
# # #                 "client_id": client_id,
# # #                 "samples": num_samples,
# # #                 "train_loss": train_loss,
# # #                 "val_loss": val_loss,
# # #                 "train_rmse": train_rmse,
# # #                 "val_rmse": val_rmse,
# # #                 "train_r2": train_r2,
# # #                 "val_r2": val_r2,
# # #                 "algorithm": self.experiment_id
# # #             }
            
# # #             self.client_metrics.append(client_metrics)
            
# # #             # Write immediately to CSV
# # #             with open(self.client_metrics_path, 'a', newline='') as f:
# # #                 writer = csv.DictWriter(f, fieldnames=client_metrics.keys())
# # #                 writer.writerow(client_metrics)
            
# # #             # Accumulate for round averages
# # #             total_samples += num_samples
# # #             train_losses.append(train_loss)
# # #             val_losses.append(val_loss)
# # #             train_rmses.append(train_rmse)
# # #             val_rmses.append(val_rmse)
# # #             train_r2s.append(train_r2)
# # #             val_r2s.append(val_r2)
        
# # #         # Calculate round averages
# # #         if train_losses:
# # #             round_metrics = {
# # #                 "timestamp": timestamp,
# # #                 "round": round_num,
# # #                 "total_clients": len(results),
# # #                 "fit_clients": len(results),
# # #                 "eval_clients": 0,  # Will be updated in evaluate
# # #                 "avg_train_loss": np.mean(train_losses),
# # #                 "avg_val_loss": np.mean(val_losses),
# # #                 "avg_train_rmse": np.mean(train_rmses),
# # #                 "avg_val_rmse": np.mean(val_rmses),
# # #                 "avg_train_r2": np.mean(train_r2s),
# # #                 "avg_val_r2": np.mean(val_r2s),
# # #                 "centralized_loss": 0,  # Will be updated in evaluate
# # #                 "algorithm": self.experiment_id
# # #             }
            
# # #             self.round_metrics.append(round_metrics)
            
# # #             # Write round metrics to CSV
# # #             with open(self.round_metrics_path, 'a', newline='') as f:
# # #                 writer = csv.DictWriter(f, fieldnames=round_metrics.keys())
# # #                 writer.writerow(round_metrics)
            
# # #             print(f"📊 Round {round_num} Fit Metrics:")
# # #             print(f"   Clients: {len(results)}")
# # #             print(f"   Avg Train Loss: {np.mean(train_losses):.4f}")
# # #             print(f"   Avg Val Loss: {np.mean(val_losses):.4f}")
# # #             print(f"   Avg Val R²: {np.mean(val_r2s):.4f}")
    
# # #     def collect_evaluate_metrics(self, round_num: int, loss: float, metrics: Dict[str, Scalar]):
# # #         """Collect metrics from centralized evaluation"""
# # #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
# # #         eval_metrics = {
# # #             "timestamp": timestamp,
# # #             "round": round_num,
# # #             "loss": loss,
# # #             "rmse": metrics.get("rmse", 0),
# # #             "mse": metrics.get("mse", 0),
# # #             "mae": metrics.get("mae", 0),
# # #             "r2": metrics.get("r2", 0),
# # #             "algorithm": self.experiment_id
# # #         }
        
# # #         self.eval_metrics.append(eval_metrics)
        
# # #         # Write eval metrics to CSV
# # #         with open(self.eval_metrics_path, 'a', newline='') as f:
# # #             writer = csv.DictWriter(f, fieldnames=eval_metrics.keys())
# # #             writer.writerow(eval_metrics)
        
# # #         # Update round metrics with centralized loss
# # #         if self.round_metrics and len(self.round_metrics) > round_num - 1:
# # #             self.round_metrics[round_num - 1]["centralized_loss"] = loss
            
# # #             # Update the CSV file
# # #             with open(self.round_metrics_path, 'r') as f:
# # #                 lines = f.readlines()
            
# # #             if len(lines) > round_num:  # Header + rounds
# # #                 # This is a simplified approach - in production you might want a more robust solution
# # #                 with open(self.round_metrics_path, 'w') as f:
# # #                     for i, line in enumerate(lines):
# # #                         if i == 0 or i > round_num:
# # #                             f.write(line)
# # #                         else:
# # #                             # Update the specific round line
# # #                             parts = line.strip().split(',')
# # #                             if len(parts) >= 12:  # Ensure we have enough columns
# # #                                 parts[11] = str(loss)  # centralized_loss position
# # #                                 f.write(','.join(parts) + '\n')
        
# # #         print(f"📈 Round {round_num} Centralized Evaluation:")
# # #         print(f"   Loss: {loss:.4f}")
# # #         print(f"   R²: {metrics.get('r2', 0):.4f}")

# # # def is_server_script():
# # #     """Check if this script is being run as the server"""
# # #     # Check if --client-id is in the arguments (client script)
# # #     return '--client-id' not in ' '.join(sys.argv)

# # # def main():
# # #     """Main function - handles both server and client based on arguments"""
    
# # #     if is_server_script():
# # #         # Server mode - use server argument parser
# # #         server_parser = argparse.ArgumentParser(description="NASA FL Server")
# # #         server_parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
# # #         server_parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
# # #         server_parser.add_argument("--port", type=int, help="Server port (overrides config)")
        
# # #         args = server_parser.parse_args()
# # #         run_server(args)
# # #     else:
# # #         # Client mode - let the client script handle it
# # #         print("❌ This script should be run as server only. Use client.py for clients.")
# # #         sys.exit(1)

# # # def run_server(args):
# # #     """Run the FL server"""
# # #     try:
# # #         # Load configuration
# # #         with open(args.config, 'r') as f:
# # #             config = json.load(f)
        
# # #         # Extract server configuration
# # #         server_config = config.get("server", {})
# # #         strategy_config = config.get("strategy", {})
# # #         data_config = config.get("data", {})
        
# # #         # Get server parameters
# # #         server_host = server_config.get("host", "0.0.0.0")
# # #         server_port = args.port or server_config.get("port", 8080)
# # #         num_rounds = server_config.get("num_rounds", 10)
        
# # #         # Get strategy parameters
# # #         strategy_name = strategy_config.get("name", "fedavg")
# # #         fraction_fit = strategy_config.get("fraction_fit", 1.0)
# # #         fraction_evaluate = strategy_config.get("fraction_evaluate", 1.0)
# # #         min_fit_clients = strategy_config.get("min_fit_clients", 2)
# # #         min_evaluate_clients = strategy_config.get("min_evaluate_clients", 2)
# # #         min_available_clients = strategy_config.get("min_available_clients", 
# # #                                                    data_config.get("num_clients", 2))
# # #         proximal_mu = strategy_config.get("proximal_mu", 0.1)
        
# # #         # Experiment info
# # #         experiment_id = config.get("experiment_id", "nasa_experiment")
# # #         algorithm = config.get("algorithm", "fedavg")
        
# # #         print("🚀 Starting NASA Federated Learning Server")
# # #         print("=" * 50)
# # #         print(f"Experiment ID: {experiment_id}")
# # #         print(f"Algorithm: {algorithm.upper()}")
# # #         print(f"Server: {server_host}:{server_port}")
# # #         print(f"Rounds: {num_rounds}")
# # #         print(f"Strategy: {strategy_name}")
# # #         print(f"Min Clients: {min_available_clients}")
# # #         print(f"Results Directory: {args.results_dir}")
# # #         print("=" * 50)
        
# # #         # Create results directory
# # #         results_dir = os.path.join(args.results_dir, experiment_id)
# # #         os.makedirs(results_dir, exist_ok=True)
        
# # #         # Initialize metrics collector
# # #         metrics_collector = MetricsCollector(results_dir, experiment_id)
        
# # #         # Create strategy
# # #         nasa_strategy = NASAStrategy(
# # #             strategy_name=strategy_name,
# # #             fraction_fit=fraction_fit,
# # #             fraction_evaluate=fraction_evaluate,
# # #             min_fit_clients=min_fit_clients,
# # #             min_evaluate_clients=min_evaluate_clients,
# # #             min_available_clients=min_available_clients,
# # #             proximal_mu=proximal_mu
# # #         )
        
# # #         strategy = nasa_strategy.get_strategy()
        
# # #         # Add metrics collection to the strategy
# # #         class MetricsStrategy(type(strategy)):
# # #             def aggregate_fit(self, server_round, results, failures):
# # #                 # Call parent method
# # #                 aggregated_parameters, aggregated_metrics = super().aggregate_fit(
# # #                     server_round, results, failures
# # #                 )
                
# # #                 # Collect metrics
# # #                 metrics_collector.collect_fit_metrics(server_round, results, failures)
                
# # #                 # Add round number to config for clients
# # #                 config = {"current_round": server_round}
                
# # #                 return aggregated_parameters, aggregated_metrics
            
# # #             def aggregate_evaluate(self, server_round, results, failures):
# # #                 # Call parent method
# # #                 aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
# # #                     server_round, results, failures
# # #                 )
                
# # #                 # For centralized evaluation, we'll handle it separately
# # #                 return aggregated_loss, aggregated_metrics
            
        

# # #             def evaluate(self, server_round, parameters):
# # #                 # This is for centralized evaluation
# # #                 eval_metrics = {}
                
# # #                 if server_round > 0:
# # #                     # In a real scenario, you would evaluate on a central test set here
# # #                     # For now, we'll just log that evaluation occurred
# # #                     eval_metrics = {
# # #                         "rmse": 0.0,
# # #                         "mse": 0.0, 
# # #                         "mae": 0.0,
# # #                         "r2": 0.0
# # #                     }
# # #                     metrics_collector.collect_evaluate_metrics(server_round, 0.0, eval_metrics)
# # #                 else:
# # #                     # For round 0, return empty metrics
# # #                     eval_metrics = {
# # #                         "rmse": 0.0,
# # #                         "mse": 0.0, 
# # #                         "mae": 0.0,
# # #                         "r2": 0.0
# # #                     }
                
# # #                 return 0.0, eval_metrics
        
# # #         # Create the enhanced strategy
# # #         enhanced_strategy = MetricsStrategy(
# # #             fraction_fit=fraction_fit,
# # #             fraction_evaluate=fraction_evaluate,
# # #             min_fit_clients=min_fit_clients,
# # #             min_evaluate_clients=min_evaluate_clients,
# # #             min_available_clients=min_available_clients
# # #         )
        
# # #         # Start Flower server
# # #         print(f"🔄 Starting FL server on {server_host}:{server_port}...")
        
# # #         fl.server.start_server(
# # #             server_address=f"{server_host}:{server_port}",
# # #             config=fl.server.ServerConfig(num_rounds=num_rounds),
# # #             strategy=enhanced_strategy
# # #         )
        
# # #         print("✅ Server completed successfully!")
# # #         print(f"📊 Results saved to: {results_dir}")
        
# # #     except Exception as e:
# # #         print(f"❌ Server error: {e}")
# # #         raise

# # # if __name__ == "__main__":
# # #     main()


# # import flwr as fl
# # import torch
# # import argparse
# # import json
# # import os
# # from typing import Dict, List, Tuple, Optional, Any
# # import numpy as np
# # from flwr.common import Parameters, Scalar
# # from flwr.server.strategy import FedAvg, FedProx
# # import torch.nn as nn
# # import pandas as pd
# # from datetime import datetime
# # import csv
# # import sys
# # import warnings
# # warnings.filterwarnings('ignore')

# # class NASAStrategy:
# #     """Custom strategy wrapper for NASA experiments"""
    
# #     def __init__(self, strategy_name: str = "fedavg", fraction_fit: float = 1.0, 
# #                  fraction_evaluate: float = 1.0, min_fit_clients: int = 2,
# #                  min_evaluate_clients: int = 2, min_available_clients: int = 2,
# #                  proximal_mu: float = 0.1, initial_parameters: Optional[Parameters] = None):
        
# #         self.strategy_name = strategy_name.lower()
# #         self.initial_parameters = initial_parameters
        
# #         if self.strategy_name == "fedprox":
# #             self.strategy = FedProx(
# #                 fraction_fit=fraction_fit,
# #                 fraction_evaluate=fraction_evaluate,
# #                 min_fit_clients=min_fit_clients,
# #                 min_evaluate_clients=min_evaluate_clients,
# #                 min_available_clients=min_available_clients,
# #                 proximal_mu=proximal_mu,
# #                 initial_parameters=initial_parameters
# #             )
# #         else:  # Default to FedAvg
# #             self.strategy = FedAvg(
# #                 fraction_fit=fraction_fit,
# #                 fraction_evaluate=fraction_evaluate,
# #                 min_fit_clients=min_fit_clients,
# #                 min_evaluate_clients=min_evaluate_clients,
# #                 min_available_clients=min_available_clients,
# #                 initial_parameters=initial_parameters
# #             )
    
# #     def get_strategy(self):
# #         return self.strategy

# # class MetricsCollector:
# #     """Collects and saves metrics from all rounds"""
    
# #     def __init__(self, results_dir: str, experiment_id: str):
# #         self.results_dir = results_dir
# #         self.experiment_id = experiment_id
# #         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
# #         # Create results directory
# #         os.makedirs(results_dir, exist_ok=True)
        
# #         # Define file paths
# #         self.round_metrics_path = os.path.join(results_dir, f"round_metrics.csv")
# #         self.client_metrics_path = os.path.join(results_dir, f"client_metrics.csv")
# #         self.eval_metrics_path = os.path.join(results_dir, f"eval_metrics.csv")
        
# #         # Initialize CSV files
# #         self._initialize_metrics_files()
        
# #         # Storage for metrics
# #         self.round_metrics = []
# #         self.client_metrics = []
# #         self.eval_metrics = []
    
# #     def _initialize_metrics_files(self):
# #         """Initialize CSV files with headers"""
        
# #         # Round metrics (aggregated per round)
# #         round_headers = [
# #             "timestamp", "round", "total_clients", "fit_clients", "eval_clients",
# #             "avg_train_loss", "avg_val_loss", "avg_val_r2", "centralized_loss", 
# #             "centralized_r2", "algorithm"
# #         ]
        
# #         # Client metrics (per client per round)
# #         client_headers = [
# #             "timestamp", "round", "client_id", "samples", "train_loss", "val_loss",
# #             "train_r2", "val_r2", "algorithm"
# #         ]
        
# #         # Evaluation metrics (centralized evaluation)
# #         eval_headers = [
# #             "timestamp", "round", "loss", "rmse", "mse", "mae", "r2", "algorithm"
# #         ]
        
# #         # Write headers
# #         for path, headers in [
# #             (self.round_metrics_path, round_headers),
# #             (self.client_metrics_path, client_headers),
# #             (self.eval_metrics_path, eval_headers)
# #         ]:
# #             with open(path, 'w', newline='') as f:
# #                 writer = csv.DictWriter(f, fieldnames=headers)
# #                 writer.writeheader()
    
# #     def collect_fit_metrics(self, round_num: int, results: List[Tuple], failures: List[BaseException]):
# #         """Collect metrics from fit results"""
# #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
# #         total_samples = 0
# #         train_losses, val_losses = [], []
# #         train_r2s, val_r2s = [], []
        
# #         for client_proxy, fit_res in results:
# #             if fit_res.num_examples == 0:
# #                 continue
                
# #             client_id = fit_res.metrics.get("client_id", "unknown")
# #             num_samples = fit_res.num_examples
# #             train_loss = fit_res.metrics.get("train_loss", 0)
# #             val_loss = fit_res.metrics.get("val_loss", 0)
# #             train_r2 = fit_res.metrics.get("train_r2", 0)
# #             val_r2 = fit_res.metrics.get("val_r2", 0)
            
# #             # Store client-level metrics
# #             client_metrics = {
# #                 "timestamp": timestamp,
# #                 "round": round_num,
# #                 "client_id": client_id,
# #                 "samples": num_samples,
# #                 "train_loss": train_loss,
# #                 "val_loss": val_loss,
# #                 "train_r2": train_r2,
# #                 "val_r2": val_r2,
# #                 "algorithm": self.experiment_id
# #             }
            
# #             self.client_metrics.append(client_metrics)
            
# #             # Write immediately to CSV
# #             with open(self.client_metrics_path, 'a', newline='') as f:
# #                 writer = csv.DictWriter(f, fieldnames=client_metrics.keys())
# #                 writer.writerow(client_metrics)
            
# #             # Accumulate for round averages
# #             total_samples += num_samples
# #             train_losses.append(train_loss)
# #             val_losses.append(val_loss)
# #             train_r2s.append(train_r2)
# #             val_r2s.append(val_r2)
        
# #         # Calculate round averages
# #         if train_losses:
# #             round_metrics = {
# #                 "timestamp": timestamp,
# #                 "round": round_num,
# #                 "total_clients": len(results),
# #                 "fit_clients": len(results),
# #                 "eval_clients": 0,
# #                 "avg_train_loss": np.mean(train_losses),
# #                 "avg_val_loss": np.mean(val_losses),
# #                 "avg_val_r2": np.mean(val_r2s),
# #                 "centralized_loss": 0,
# #                 "centralized_r2": 0,
# #                 "algorithm": self.experiment_id
# #             }
            
# #             self.round_metrics.append(round_metrics)
            
# #             # Write round metrics to CSV
# #             with open(self.round_metrics_path, 'a', newline='') as f:
# #                 writer = csv.DictWriter(f, fieldnames=round_metrics.keys())
# #                 writer.writerow(round_metrics)
            
# #             print(f"📊 Round {round_num} Fit Metrics:")
# #             print(f"   Clients: {len(results)}")
# #             print(f"   Avg Train Loss: {np.mean(train_losses):.4f}")
# #             print(f"   Avg Val Loss: {np.mean(val_losses):.4f}")
# #             print(f"   Avg Val R²: {np.mean(val_r2s):.4f}")
    
# #     def collect_evaluate_metrics(self, round_num: int, loss: float, metrics: Dict[str, Scalar]):
# #         """Collect metrics from centralized evaluation"""
# #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
# #         eval_metrics = {
# #             "timestamp": timestamp,
# #             "round": round_num,
# #             "loss": loss,
# #             "rmse": metrics.get("rmse", 0),
# #             "mse": metrics.get("mse", 0),
# #             "mae": metrics.get("mae", 0),
# #             "r2": metrics.get("r2", 0),
# #             "algorithm": self.experiment_id
# #         }
        
# #         self.eval_metrics.append(eval_metrics)
        
# #         # Write eval metrics to CSV
# #         with open(self.eval_metrics_path, 'a', newline='') as f:
# #             writer = csv.DictWriter(f, fieldnames=eval_metrics.keys())
# #             writer.writerow(eval_metrics)
        
# #         # Update round metrics with centralized evaluation results
# #         if self.round_metrics and len(self.round_metrics) >= round_num:
# #             # Read all round metrics
# #             round_data = []
# #             with open(self.round_metrics_path, 'r') as f:
# #                 reader = csv.DictReader(f)
# #                 for row in reader:
# #                     round_data.append(row)
            
# #             # Update the specific round
# #             if len(round_data) >= round_num:
# #                 round_data[round_num-1]["centralized_loss"] = str(loss)
# #                 round_data[round_num-1]["centralized_r2"] = str(metrics.get("r2", 0))
                
# #                 # Rewrite the entire file
# #                 with open(self.round_metrics_path, 'w', newline='') as f:
# #                     if round_data:
# #                         writer = csv.DictWriter(f, fieldnames=round_data[0].keys())
# #                         writer.writeheader()
# #                         writer.writerows(round_data)
        
# #         print(f"📈 Round {round_num} Centralized Evaluation:")
# #         print(f"   Loss: {loss:.4f}")
# #         print(f"   R²: {metrics.get('r2', 0):.4f}")

# # def get_evaluate_fn(testloader=None):
# #     """Create evaluation function for centralized evaluation"""
# #     def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
# #         """
# #         Evaluate the model on a centralized test set.
# #         In a real scenario, you would:
# #         1. Load the central test dataset
# #         2. Update the model with the current parameters
# #         3. Evaluate on the test set
# #         4. Return loss and metrics
# #         """
# #         # For now, return placeholder values
# #         # In practice, you should implement proper evaluation here
# #         if testloader is not None:
# #             # Actual evaluation logic would go here
# #             loss = 0.0
# #             r2 = 0.0
# #         else:
# #             # Placeholder - replace with actual evaluation
# #             loss = 0.0
# #             r2 = 0.0
            
# #         metrics = {
# #             "rmse": 0.0,
# #             "mse": 0.0, 
# #             "mae": 0.0,
# #             "r2": r2
# #         }
# #         return loss, metrics
    
# #     return evaluate_fn

# # class CustomFedAvg(FedAvg):
# #     """Custom FedAvg strategy with enhanced metrics collection"""
    
# #     def __init__(self, metrics_collector: MetricsCollector, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.metrics_collector = metrics_collector
    
# #     def aggregate_fit(self, server_round, results, failures):
# #         """Aggregate fit results and collect metrics"""
# #         # Call parent method first
# #         aggregated_parameters, aggregated_metrics = super().aggregate_fit(
# #             server_round, results, failures
# #         )
        
# #         # Collect metrics
# #         if results:
# #             self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        
# #         return aggregated_parameters, aggregated_metrics

# # def is_server_script():
# #     """Check if this script is being run as the server"""
# #     return '--client-id' not in ' '.join(sys.argv)

# # def main():
# #     """Main function - handles both server and client based on arguments"""
    
# #     if is_server_script():
# #         # Server mode - use server argument parser
# #         server_parser = argparse.ArgumentParser(description="NASA FL Server")
# #         server_parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
# #         server_parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
# #         server_parser.add_argument("--port", type=int, help="Server port (overrides config)")
        
# #         args = server_parser.parse_args()
# #         run_server(args)
# #     else:
# #         # Client mode - let the client script handle it
# #         print("❌ This script should be run as server only. Use client.py for clients.")
# #         sys.exit(1)

# # def run_server(args):
# #     """Run the FL server"""
# #     try:
# #         # Load configuration
# #         with open(args.config, 'r') as f:
# #             config = json.load(f)
        
# #         # Extract server configuration
# #         server_config = config.get("server", {})
# #         strategy_config = config.get("strategy", {})
# #         data_config = config.get("data", {})
        
# #         # Get server parameters
# #         server_host = server_config.get("host", "0.0.0.0")
# #         server_port = args.port or server_config.get("port", 8080)
# #         num_rounds = server_config.get("num_rounds", 10)
        
# #         # Get strategy parameters
# #         strategy_name = strategy_config.get("name", "fedavg")
# #         fraction_fit = strategy_config.get("fraction_fit", 1.0)
# #         fraction_evaluate = strategy_config.get("fraction_evaluate", 1.0)
# #         min_fit_clients = strategy_config.get("min_fit_clients", 2)
# #         min_evaluate_clients = strategy_config.get("min_evaluate_clients", 2)
# #         min_available_clients = strategy_config.get("min_available_clients", 
# #                                                    data_config.get("num_clients", 2))
# #         proximal_mu = strategy_config.get("proximal_mu", 0.1)
        
# #         # Experiment info
# #         experiment_id = config.get("experiment_id", "nasa_experiment")
# #         algorithm = config.get("algorithm", "fedavg")
        
# #         print("🚀 Starting NASA Federated Learning Server")
# #         print("=" * 50)
# #         print(f"Experiment ID: {experiment_id}")
# #         print(f"Algorithm: {algorithm.upper()}")
# #         print(f"Server: {server_host}:{server_port}")
# #         print(f"Rounds: {num_rounds}")
# #         print(f"Strategy: {strategy_name}")
# #         print(f"Min Clients: {min_available_clients}")
# #         print(f"Results Directory: {args.results_dir}")
# #         print("=" * 50)
        
# #         # Create results directory
# #         results_dir = os.path.join(args.results_dir, experiment_id)
# #         os.makedirs(results_dir, exist_ok=True)
        
# #         # Initialize metrics collector
# #         metrics_collector = MetricsCollector(results_dir, experiment_id)
        
# #         # Create evaluation function (you need to implement proper test data loading)
# #         evaluate_fn = get_evaluate_fn()  # Pass testloader if available
        
# #         # Create strategy with proper configuration
# #         strategy = CustomFedAvg(
# #             metrics_collector=metrics_collector,
# #             fraction_fit=fraction_fit,
# #             fraction_evaluate=fraction_evaluate,
# #             min_fit_clients=min_fit_clients,
# #             min_evaluate_clients=min_evaluate_clients,
# #             min_available_clients=min_available_clients,
# #             evaluate_fn=evaluate_fn,  # Add evaluation function
# #             on_fit_config_fn=lambda r: {"current_round": r}  # Send round number to clients
# #         )
        
# #         # Start Flower server
# #         print(f"🔄 Starting FL server on {server_host}:{server_port}...")
        
# #         # Save config for reference
# #         config_save_path = os.path.join(results_dir, "server_config.json")
# #         with open(config_save_path, 'w') as f:
# #             json.dump(config, f, indent=2)
        
# #         fl.server.start_server(
# #             server_address=f"{server_host}:{server_port}",
# #             config=fl.server.ServerConfig(num_rounds=num_rounds),
# #             strategy=strategy,
# #             grpc_max_message_length=1024 * 1024 * 1024  # 1GB max message length
# #         )
        
# #         print("✅ Server completed successfully!")
# #         print(f"📊 Results saved to: {results_dir}")
        
# #     except Exception as e:
# #         print(f"❌ Server error: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         raise

# # if __name__ == "__main__":
# #     main()

# import flwr as fl
# import torch
# import argparse
# import json
# import os
# from typing import Dict, List, Tuple, Optional, Any
# import numpy as np
# from flwr.common import Parameters, Scalar
# from flwr.server.strategy import FedAvg, FedProx
# import torch.nn as nn
# import pandas as pd
# from datetime import datetime
# import csv
# import sys
# import warnings
# warnings.filterwarnings('ignore')

# class NASAStrategy:
#     """Custom strategy wrapper for NASA experiments"""
    
#     def __init__(self, strategy_name: str = "fedavg", fraction_fit: float = 1.0, 
#                  fraction_evaluate: float = 1.0, min_fit_clients: int = 2,
#                  min_evaluate_clients: int = 2, min_available_clients: int = 2,
#                  proximal_mu: float = 0.1, initial_parameters: Optional[Parameters] = None):
        
#         self.strategy_name = strategy_name.lower()
#         self.initial_parameters = initial_parameters
        
#         if self.strategy_name == "fedprox":
#             self.strategy = FedProx(
#                 fraction_fit=fraction_fit,
#                 fraction_evaluate=fraction_evaluate,
#                 min_fit_clients=min_fit_clients,
#                 min_evaluate_clients=min_evaluate_clients,
#                 min_available_clients=min_available_clients,
#                 proximal_mu=proximal_mu,
#                 initial_parameters=initial_parameters
#             )
#         else:  # Default to FedAvg
#             self.strategy = FedAvg(
#                 fraction_fit=fraction_fit,
#                 fraction_evaluate=fraction_evaluate,
#                 min_fit_clients=min_fit_clients,
#                 min_evaluate_clients=min_evaluate_clients,
#                 min_available_clients=min_available_clients,
#                 initial_parameters=initial_parameters
#             )
    
#     def get_strategy(self):
#         return self.strategy

# class MetricsCollector:
#     """Collects and saves metrics from all rounds"""
    
#     def __init__(self, results_dir: str, experiment_id: str):
#         self.results_dir = results_dir
#         self.experiment_id = experiment_id
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Create results directory
#         os.makedirs(results_dir, exist_ok=True)
        
#         # Define file paths
#         self.round_metrics_path = os.path.join(results_dir, f"round_metrics.csv")
#         self.client_metrics_path = os.path.join(results_dir, f"client_metrics.csv")
#         self.eval_metrics_path = os.path.join(results_dir, f"eval_metrics.csv")
#         self.test_metrics_path = os.path.join(results_dir, f"test_metrics.csv")  # New: for client test results
        
#         # Initialize CSV files
#         self._initialize_metrics_files()
        
#         # Storage for metrics
#         self.round_metrics = []
#         self.client_metrics = []
#         self.eval_metrics = []
#         self.test_metrics = []
    
#     def _initialize_metrics_files(self):
#         """Initialize CSV files with headers"""
        
#         # Round metrics (aggregated per round)
#         round_headers = [
#             "timestamp", "round", "total_clients", "fit_clients", "eval_clients",
#             "avg_train_loss", "avg_val_loss", "avg_val_r2", "avg_test_loss", "avg_test_r2",
#             "centralized_loss", "centralized_r2", "algorithm"
#         ]
        
#         # Client metrics (per client per round - from fit)
#         client_headers = [
#             "timestamp", "round", "client_id", "samples", "train_loss", "val_loss",
#             "train_r2", "val_r2", "test_loss", "test_r2", "algorithm"  # Added test metrics
#         ]
        
#         # Evaluation metrics (centralized evaluation)
#         eval_headers = [
#             "timestamp", "round", "loss", "rmse", "mse", "mae", "r2", "algorithm"
#         ]
        
#         # Test metrics (from client evaluation - separate file)
#         test_headers = [
#             "timestamp", "round", "client_id", "loss", "rmse", "mse", "mae", "r2", 
#             "samples", "algorithm"
#         ]
        
#         # Write headers
#         for path, headers in [
#             (self.round_metrics_path, round_headers),
#             (self.client_metrics_path, client_headers),
#             (self.eval_metrics_path, eval_headers),
#             (self.test_metrics_path, test_headers)
#         ]:
#             with open(path, 'w', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=headers)
#                 writer.writeheader()
    
#     def collect_fit_metrics(self, round_num: int, results: List[Tuple], failures: List[BaseException]):
#         """Collect metrics from fit results - including test metrics if available"""
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         total_samples = 0
#         train_losses, val_losses, test_losses = [], [], []
#         train_r2s, val_r2s, test_r2s = [], [], []
        
#         for client_proxy, fit_res in results:
#             if fit_res.num_examples == 0:
#                 continue
                
#             client_id = fit_res.metrics.get("client_id", "unknown")
#             num_samples = fit_res.num_examples
#             train_loss = fit_res.metrics.get("train_loss", 0)
#             val_loss = fit_res.metrics.get("val_loss", 0)
#             train_r2 = fit_res.metrics.get("train_r2", 0)
#             val_r2 = fit_res.metrics.get("val_r2", 0)
            
#             # Extract test metrics if available (from fit results)
#             test_loss = fit_res.metrics.get("test_loss", 0)
#             test_r2 = fit_res.metrics.get("test_r2", 0)
            
#             # Store client-level metrics
#             client_metrics = {
#                 "timestamp": timestamp,
#                 "round": round_num,
#                 "client_id": client_id,
#                 "samples": num_samples,
#                 "train_loss": train_loss,
#                 "val_loss": val_loss,
#                 "train_r2": train_r2,
#                 "val_r2": val_r2,
#                 "test_loss": test_loss,
#                 "test_r2": test_r2,
#                 "algorithm": self.experiment_id
#             }
            
#             self.client_metrics.append(client_metrics)
            
#             # Write immediately to CSV
#             with open(self.client_metrics_path, 'a', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=client_metrics.keys())
#                 writer.writerow(client_metrics)
            
#             # Accumulate for round averages
#             total_samples += num_samples
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#             train_r2s.append(train_r2)
#             val_r2s.append(val_r2)
            
#             # Only include test metrics if they're non-zero (actually computed)
#             if test_loss > 0:
#                 test_losses.append(test_loss)
#                 test_r2s.append(test_r2)
        
#         # Calculate round averages
#         if train_losses:
#             round_metrics = {
#                 "timestamp": timestamp,
#                 "round": round_num,
#                 "total_clients": len(results),
#                 "fit_clients": len(results),
#                 "eval_clients": 0,
#                 "avg_train_loss": np.mean(train_losses),
#                 "avg_val_loss": np.mean(val_losses),
#                 "avg_val_r2": np.mean(val_r2s),
#                 "avg_test_loss": np.mean(test_losses) if test_losses else 0,
#                 "avg_test_r2": np.mean(test_r2s) if test_r2s else 0,
#                 "centralized_loss": 0,
#                 "centralized_r2": 0,
#                 "algorithm": self.experiment_id
#             }
            
#             self.round_metrics.append(round_metrics)
            
#             # Write round metrics to CSV
#             with open(self.round_metrics_path, 'a', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=round_metrics.keys())
#                 writer.writerow(round_metrics)
            
#             print(f"📊 Round {round_num} Fit Metrics:")
#             print(f"   Clients: {len(results)}")
#             print(f"   Avg Train Loss: {np.mean(train_losses):.4f}")
#             print(f"   Avg Val Loss: {np.mean(val_losses):.4f}")
#             print(f"   Avg Val R²: {np.mean(val_r2s):.4f}")
            
#             if test_losses:
#                 print(f"   Avg Test Loss: {np.mean(test_losses):.4f}")
#                 print(f"   Avg Test R²: {np.mean(test_r2s):.4f}")
    
#     def collect_evaluate_metrics(self, round_num: int, results: List[Tuple], failures: List[BaseException]):
#         """Collect metrics from client evaluation results (test set evaluation)"""
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         if not results:
#             return
        
#         total_samples = 0
#         test_losses, test_r2s = [], []
        
#         for client_proxy, evaluate_res in results:
#             if evaluate_res.num_examples == 0:
#                 continue
                
#             client_id = evaluate_res.metrics.get("client_id", "unknown")
#             num_samples = evaluate_res.num_examples
#             test_loss = evaluate_res.loss
#             test_r2 = evaluate_res.metrics.get("test_r2", 0)
            
#             # Store client test metrics
#             test_metrics = {
#                 "timestamp": timestamp,
#                 "round": round_num,
#                 "client_id": client_id,
#                 "loss": test_loss,
#                 "rmse": evaluate_res.metrics.get("test_rmse", 0),
#                 "mse": evaluate_res.metrics.get("test_mse", 0),
#                 "mae": evaluate_res.metrics.get("test_mae", 0),
#                 "r2": test_r2,
#                 "samples": num_samples,
#                 "algorithm": self.experiment_id
#             }
            
#             self.test_metrics.append(test_metrics)
            
#             # Write to test metrics CSV
#             with open(self.test_metrics_path, 'a', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
#                 writer.writerow(test_metrics)
            
#             # Accumulate for averages
#             total_samples += num_samples
#             test_losses.append(test_loss)
#             test_r2s.append(test_r2)
        
#         # Print evaluation summary
#         if test_losses:
#             print(f"🧪 Round {round_num} Client Test Evaluation:")
#             print(f"   Clients: {len(results)}")
#             print(f"   Avg Test Loss: {np.mean(test_losses):.4f}")
#             print(f"   Avg Test R²: {np.mean(test_r2s):.4f}")
            
#             # Update round metrics with test results
#             self._update_round_metrics_with_test(round_num, test_losses, test_r2s)
    
#     def _update_round_metrics_with_test(self, round_num: int, test_losses: List[float], test_r2s: List[float]):
#         """Update round metrics with test results"""
#         try:
#             # Read all round metrics
#             round_data = []
#             with open(self.round_metrics_path, 'r') as f:
#                 reader = csv.DictReader(f)
#                 for row in reader:
#                     round_data.append(row)
            
#             # Update the specific round with test metrics
#             if len(round_data) >= round_num:
#                 round_data[round_num-1]["avg_test_loss"] = str(np.mean(test_losses))
#                 round_data[round_num-1]["avg_test_r2"] = str(np.mean(test_r2s))
#                 round_data[round_num-1]["eval_clients"] = str(len(test_losses))
                
#                 # Rewrite the entire file
#                 with open(self.round_metrics_path, 'w', newline='') as f:
#                     if round_data:
#                         writer = csv.DictWriter(f, fieldnames=round_data[0].keys())
#                         writer.writeheader()
#                         writer.writerows(round_data)
#         except Exception as e:
#             print(f"⚠️ Could not update round metrics with test results: {e}")
    
#     def collect_centralized_eval(self, round_num: int, loss: float, metrics: Dict[str, Scalar]):
#         """Collect metrics from centralized evaluation"""
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         eval_metrics = {
#             "timestamp": timestamp,
#             "round": round_num,
#             "loss": loss,
#             "rmse": metrics.get("rmse", 0),
#             "mse": metrics.get("mse", 0),
#             "mae": metrics.get("mae", 0),
#             "r2": metrics.get("r2", 0),
#             "algorithm": self.experiment_id
#         }
        
#         self.eval_metrics.append(eval_metrics)
        
#         # Write eval metrics to CSV
#         with open(self.eval_metrics_path, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=eval_metrics.keys())
#             writer.writerow(eval_metrics)
        
#         print(f"📈 Round {round_num} Centralized Evaluation:")
#         print(f"   Loss: {loss:.4f}")
#         print(f"   R²: {metrics.get('r2', 0):.4f}")

# def get_evaluate_fn(testloader=None):
#     """Create evaluation function for centralized evaluation"""
#     def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
#         """
#         Evaluate the model on a centralized test set.
#         In a real scenario, you would:
#         1. Load the central test dataset
#         2. Update the model with the current parameters
#         3. Evaluate on the test set
#         4. Return loss and metrics
#         """
#         # For now, return placeholder values
#         # In practice, you should implement proper evaluation here
#         if testloader is not None:
#             # Actual evaluation logic would go here
#             loss = 0.0
#             r2 = 0.0
#         else:
#             # Placeholder - replace with actual evaluation
#             loss = 0.0
#             r2 = 0.0
            
#         metrics = {
#             "rmse": 0.0,
#             "mse": 0.0, 
#             "mae": 0.0,
#             "r2": r2
#         }
#         return loss, metrics
    
#     return evaluate_fn

# class CustomFedAvg(FedAvg):
#     """Custom FedAvg strategy with enhanced metrics collection"""
    
#     def __init__(self, metrics_collector: MetricsCollector, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.metrics_collector = metrics_collector
    
#     def aggregate_fit(self, server_round, results, failures):
#         """Aggregate fit results and collect metrics"""
#         # Call parent method first
#         aggregated_parameters, aggregated_metrics = super().aggregate_fit(
#             server_round, results, failures
#         )
        
#         # Collect metrics
#         if results:
#             self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        
#         return aggregated_parameters, aggregated_metrics
    
#     def aggregate_evaluate(self, server_round, results, failures):
#         """Aggregate evaluation results and collect test metrics"""
#         # Call parent method first
#         aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
#             server_round, results, failures
#         )
        
#         # Collect evaluation metrics
#         if results:
#             self.metrics_collector.collect_evaluate_metrics(server_round, results, failures)
        
#         return aggregated_loss, aggregated_metrics

# def is_server_script():
#     """Check if this script is being run as the server"""
#     return '--client-id' not in ' '.join(sys.argv)

# def main():
#     """Main function - handles both server and client based on arguments"""
    
#     if is_server_script():
#         # Server mode - use server argument parser
#         server_parser = argparse.ArgumentParser(description="NASA FL Server")
#         server_parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
#         server_parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
#         server_parser.add_argument("--port", type=int, help="Server port (overrides config)")
        
#         args = server_parser.parse_args()
#         run_server(args)
#     else:
#         # Client mode - let the client script handle it
#         print("❌ This script should be run as server only. Use client.py for clients.")
#         sys.exit(1)

# def run_server(args):
#     """Run the FL server"""
#     try:
#         # Load configuration
#         with open(args.config, 'r') as f:
#             config = json.load(f)
        
#         # Extract server configuration
#         server_config = config.get("server", {})
#         strategy_config = config.get("strategy", {})
#         data_config = config.get("data", {})
        
#         # Get server parameters
#         server_host = server_config.get("host", "0.0.0.0")
#         server_port = args.port or server_config.get("port", 8080)
#         num_rounds = server_config.get("num_rounds", 10)
        
#         # Get strategy parameters - CRITICAL: Enable evaluation
#         strategy_name = strategy_config.get("name", "fedavg")
#         fraction_fit = strategy_config.get("fraction_fit", 1.0)
#         fraction_evaluate = strategy_config.get("fraction_evaluate", 1.0)  # Enable evaluation
#         min_fit_clients = strategy_config.get("min_fit_clients", 2)
#         min_evaluate_clients = strategy_config.get("min_evaluate_clients", 2)  # Enable evaluation
#         min_available_clients = strategy_config.get("min_available_clients", 
#                                                    data_config.get("num_clients", 2))
#         proximal_mu = strategy_config.get("proximal_mu", 0.1)
        
#         # Experiment info
#         experiment_id = config.get("experiment_id", "nasa_experiment")
#         algorithm = config.get("algorithm", "fedavg")
        
#         print("🚀 Starting NASA Federated Learning Server")
#         print("=" * 50)
#         print(f"Experiment ID: {experiment_id}")
#         print(f"Algorithm: {algorithm.upper()}")
#         print(f"Server: {server_host}:{server_port}")
#         print(f"Rounds: {num_rounds}")
#         print(f"Strategy: {strategy_name}")
#         print(f"Min Clients: {min_available_clients}")
#         print(f"Fit Fraction: {fraction_fit}")
#         print(f"Eval Fraction: {fraction_evaluate}")  # Should be > 0 for test evaluation
#         print(f"Results Directory: {args.results_dir}")
#         print("=" * 50)
        
#         # Create results directory
#         results_dir = os.path.join(args.results_dir, experiment_id)
#         os.makedirs(results_dir, exist_ok=True)
        
#         # Initialize metrics collector
#         metrics_collector = MetricsCollector(results_dir, experiment_id)
        
#         # Create evaluation function (you need to implement proper test data loading)
#         evaluate_fn = get_evaluate_fn()  # Pass testloader if available
        
#         # Create strategy with proper configuration - ENABLE EVALUATION
#         strategy = CustomFedAvg(
#             metrics_collector=metrics_collector,
#             fraction_fit=fraction_fit,
#             fraction_evaluate=fraction_evaluate,  # This enables client evaluation
#             min_fit_clients=min_fit_clients,
#             min_evaluate_clients=min_evaluate_clients,  # This enables client evaluation
#             min_available_clients=min_available_clients,
#             evaluate_fn=evaluate_fn,  # Centralized evaluation
#             on_fit_config_fn=lambda r: {"current_round": r},  # Send round number to clients
#             on_evaluate_config_fn=lambda r: {"current_round": r}  # Also for evaluation
#         )
        
#         # Start Flower server
#         print(f"🔄 Starting FL server on {server_host}:{server_port}...")
#         print(f"📊 Evaluation enabled: {fraction_evaluate > 0 and min_evaluate_clients > 0}")
        
#         # Save config for reference
#         config_save_path = os.path.join(results_dir, "server_config.json")
#         with open(config_save_path, 'w') as f:
#             json.dump(config, f, indent=2)
        
#         # Start server
#         fl.server.start_server(
#             server_address=f"{server_host}:{server_port}",
#             config=fl.server.ServerConfig(num_rounds=num_rounds),
#             strategy=strategy,
#             grpc_max_message_length=1024 * 1024 * 1024  # 1GB max message length
#         )
        
#         print("✅ Server completed successfully!")
#         print(f"📊 Results saved to: {results_dir}")
        
#         # Print final summary
#         print("\n📈 FINAL SUMMARY:")
#         print(f"   Round metrics: {os.path.join(results_dir, 'round_metrics.csv')}")
#         print(f"   Client metrics: {os.path.join(results_dir, 'client_metrics.csv')}")
#         print(f"   Test metrics: {os.path.join(results_dir, 'test_metrics.csv')}")
#         print(f"   Eval metrics: {os.path.join(results_dir, 'eval_metrics.csv')}")
        
#     except Exception as e:
#         print(f"❌ Server error: {e}")
#         import traceback
#         traceback.print_exc()
#         raise

# if __name__ == "__main__":
#     main()


import flwr as fl
import torch
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg, FedProx
import torch.nn as nn
import pandas as pd
from datetime import datetime
import csv
import sys
import warnings
warnings.filterwarnings('ignore')

class NASAStrategy:
    """Custom strategy wrapper for NASA experiments"""
    
    def __init__(self, strategy_name: str = "fedavg", fraction_fit: float = 1.0, 
                 fraction_evaluate: float = 1.0, min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2, min_available_clients: int = 2,
                 proximal_mu: float = 0.1, initial_parameters: Optional[Parameters] = None):
        
        self.strategy_name = strategy_name.lower()
        self.initial_parameters = initial_parameters
        
        if self.strategy_name == "fedprox":
            self.strategy = FedProx(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                proximal_mu=proximal_mu,
                initial_parameters=initial_parameters
            )
        else:  # Default to FedAvg
            self.strategy = FedAvg(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                initial_parameters=initial_parameters
            )
    
    def get_strategy(self):
        return self.strategy

class MetricsCollector:
    """Collects and saves metrics from all rounds"""
    
    def __init__(self, results_dir: str, experiment_id: str):
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create metrics subdirectory
        self.metrics_dir = os.path.join(results_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Define file paths inside metrics/
        self.round_metrics_path = os.path.join(self.metrics_dir, "round_metrics.csv")
        self.client_metrics_path = os.path.join(self.metrics_dir, "client_metrics.csv")
        self.eval_metrics_path = os.path.join(self.metrics_dir, "eval_metrics.csv")
        self.test_metrics_path = os.path.join(self.metrics_dir, "test_metrics.csv")

        # Initialize CSV files
        self._initialize_metrics_files()
        
        # Storage for metrics
        self.round_metrics = []
        self.client_metrics = []
        self.eval_metrics = []
        self.test_metrics = []
    
    def _initialize_metrics_files(self):
        """Initialize CSV files with headers"""
        
        # Round metrics (aggregated per round)
        round_headers = [
            "timestamp", "round", "total_clients", "fit_clients", "eval_clients",
            "avg_train_loss", "avg_val_loss", "avg_val_r2", "avg_test_loss", "avg_test_r2",
            "centralized_loss", "centralized_r2", "algorithm"
        ]
        
        # Client metrics (per client per round - from fit)
        client_headers = [
            "timestamp", "round", "client_id", "samples", "train_loss", "val_loss",
            "train_r2", "val_r2", "test_loss", "test_r2", "algorithm"  # Added test metrics
        ]
        
        # Evaluation metrics (centralized evaluation)
        eval_headers = [
            "timestamp", "round", "loss", "rmse", "mse", "mae", "r2", "algorithm"
        ]
        
        # Test metrics (from client evaluation - separate file)
        test_headers = [
            "timestamp", "round", "client_id", "loss", "rmse", "mse", "mae", "r2", 
            "samples", "algorithm"
        ]
        
        # Write headers
        for path, headers in [
            (self.round_metrics_path, round_headers),
            (self.client_metrics_path, client_headers),
            (self.eval_metrics_path, eval_headers),
            (self.test_metrics_path, test_headers)
        ]:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
    
    def collect_fit_metrics(self, round_num: int, results: List[Tuple], failures: List[BaseException]):
        """Collect metrics from fit results - including test metrics if available"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_samples = 0
        train_losses, val_losses, test_losses = [], [], []
        train_r2s, val_r2s, test_r2s = [], [], []
        
        for client_proxy, fit_res in results:
            if fit_res.num_examples == 0:
                continue
                
            client_id = fit_res.metrics.get("client_id", "unknown")
            num_samples = fit_res.num_examples
            train_loss = fit_res.metrics.get("train_loss", 0)
            val_loss = fit_res.metrics.get("val_loss", 0)
            train_r2 = fit_res.metrics.get("train_r2", 0)
            val_r2 = fit_res.metrics.get("val_r2", 0)
            
            # Extract test metrics if available (from fit results)
            test_loss = fit_res.metrics.get("test_loss", 0)
            test_r2 = fit_res.metrics.get("test_r2", 0)
            
            # Store client-level metrics
            client_metrics = {
                "timestamp": timestamp,
                "round": round_num,
                "client_id": client_id,
                "samples": num_samples,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "test_loss": test_loss,
                "test_r2": test_r2,
                "algorithm": self.experiment_id
            }
            
            self.client_metrics.append(client_metrics)
            
            # Write immediately to CSV
            with open(self.client_metrics_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=client_metrics.keys())
                writer.writerow(client_metrics)
            
            # Accumulate for round averages
            total_samples += num_samples
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_r2s.append(train_r2)
            val_r2s.append(val_r2)
            
            # Only include test metrics if they're non-zero (actually computed)
            if test_loss > 0:
                test_losses.append(test_loss)
                test_r2s.append(test_r2)
        
        # Calculate round averages
        if train_losses:
            round_metrics = {
                "timestamp": timestamp,
                "round": round_num,
                "total_clients": len(results),
                "fit_clients": len(results),
                "eval_clients": 0,
                "avg_train_loss": np.mean(train_losses),
                "avg_val_loss": np.mean(val_losses),
                "avg_val_r2": np.mean(val_r2s),
                "avg_test_loss": np.mean(test_losses) if test_losses else 0,
                "avg_test_r2": np.mean(test_r2s) if test_r2s else 0,
                "centralized_loss": 0,
                "centralized_r2": 0,
                "algorithm": self.experiment_id
            }
            
            self.round_metrics.append(round_metrics)
            
            # Write round metrics to CSV
            with open(self.round_metrics_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=round_metrics.keys())
                writer.writerow(round_metrics)
            
            print(f"📊 Round {round_num} Fit Metrics:")
            print(f"   Clients: {len(results)}")
            print(f"   Avg Train Loss: {np.mean(train_losses):.4f}")
            print(f"   Avg Val Loss: {np.mean(val_losses):.4f}")
            print(f"   Avg Val R²: {np.mean(val_r2s):.4f}")
            
            if test_losses:
                print(f"   Avg Test Loss: {np.mean(test_losses):.4f}")
                print(f"   Avg Test R²: {np.mean(test_r2s):.4f}")
    
    def collect_evaluate_metrics(self, round_num: int, results: List[Tuple], failures: List[BaseException]):
        """Collect metrics from client evaluation results (test set evaluation)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not results:
            return
        
        total_samples = 0
        test_losses, test_r2s = [], []
        
        for client_proxy, evaluate_res in results:
            if evaluate_res.num_examples == 0:
                continue
                
            client_id = evaluate_res.metrics.get("client_id", "unknown")
            num_samples = evaluate_res.num_examples
            test_loss = evaluate_res.loss
            test_r2 = evaluate_res.metrics.get("test_r2", 0)
            
            # Store client test metrics
            test_metrics = {
                "timestamp": timestamp,
                "round": round_num,
                "client_id": client_id,
                "loss": test_loss,
                "rmse": evaluate_res.metrics.get("test_rmse", 0),
                "mse": evaluate_res.metrics.get("test_mse", 0),
                "mae": evaluate_res.metrics.get("test_mae", 0),
                "r2": test_r2,
                "samples": num_samples,
                "algorithm": self.experiment_id
            }
            
            self.test_metrics.append(test_metrics)
            
            # Write to test metrics CSV
            with open(self.test_metrics_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
                writer.writerow(test_metrics)
            
            # Accumulate for averages
            total_samples += num_samples
            test_losses.append(test_loss)
            test_r2s.append(test_r2)
        
        # Print evaluation summary
        if test_losses:
            print(f"🧪 Round {round_num} Client Test Evaluation:")
            print(f"   Clients: {len(results)}")
            print(f"   Avg Test Loss: {np.mean(test_losses):.4f}")
            print(f"   Avg Test R²: {np.mean(test_r2s):.4f}")
            
            # Update round metrics with test results
            self._update_round_metrics_with_test(round_num, test_losses, test_r2s)
    
    def _update_round_metrics_with_test(self, round_num: int, test_losses: List[float], test_r2s: List[float]):
        """Update round metrics with test results"""
        try:
            # Read all round metrics
            round_data = []
            with open(self.round_metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    round_data.append(row)
            
            # Update the specific round with test metrics
            if len(round_data) >= round_num:
                round_data[round_num-1]["avg_test_loss"] = str(np.mean(test_losses))
                round_data[round_num-1]["avg_test_r2"] = str(np.mean(test_r2s))
                round_data[round_num-1]["eval_clients"] = str(len(test_losses))
                
                # Rewrite the entire file
                with open(self.round_metrics_path, 'w', newline='') as f:
                    if round_data:
                        writer = csv.DictWriter(f, fieldnames=round_data[0].keys())
                        writer.writeheader()
                        writer.writerows(round_data)
        except Exception as e:
            print(f"⚠️ Could not update round metrics with test results: {e}")
    
    def collect_centralized_eval(self, round_num: int, loss: float, metrics: Dict[str, Scalar]):
        """Collect metrics from centralized evaluation"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        eval_metrics = {
            "timestamp": timestamp,
            "round": round_num,
            "loss": loss,
            "rmse": metrics.get("rmse", 0),
            "mse": metrics.get("mse", 0),
            "mae": metrics.get("mae", 0),
            "r2": metrics.get("r2", 0),
            "algorithm": self.experiment_id
        }
        
        self.eval_metrics.append(eval_metrics)
        
        # Write eval metrics to CSV
        with open(self.eval_metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=eval_metrics.keys())
            writer.writerow(eval_metrics)
        
        print(f"📈 Round {round_num} Centralized Evaluation:")
        print(f"   Loss: {loss:.4f}")
        print(f"   R²: {metrics.get('r2', 0):.4f}")

def get_evaluate_fn(testloader=None):
    """Create evaluation function for centralized evaluation"""
    def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
        """
        Evaluate the model on a centralized test set.
        In a real scenario, you would:
        1. Load the central test dataset
        2. Update the model with the current parameters
        3. Evaluate on the test set
        4. Return loss and metrics
        """
        # For now, return placeholder values
        # In practice, you should implement proper evaluation here
        if testloader is not None:
            # Actual evaluation logic would go here
            loss = 0.0
            r2 = 0.0
        else:
            # Placeholder - replace with actual evaluation
            loss = 0.0
            r2 = 0.0
            
        metrics = {
            "rmse": 0.0,
            "mse": 0.0, 
            "mae": 0.0,
            "r2": r2
        }
        return loss, metrics
    
    return evaluate_fn

class CustomFedAvg(FedAvg):
    """Custom FedAvg strategy with enhanced metrics collection"""
    
    def __init__(self, metrics_collector: MetricsCollector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
    
    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results and collect metrics"""
        # Call parent method first
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Collect metrics
        if results:
            self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results and collect test metrics"""
        # Call parent method first
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        
        # Collect evaluation metrics
        if results:
            self.metrics_collector.collect_evaluate_metrics(server_round, results, failures)
        
        return aggregated_loss, aggregated_metrics

def is_server_script():
    """Check if this script is being run as the server"""
    return '--client-id' not in ' '.join(sys.argv)

def main():
    """Main function - handles both server and client based on arguments"""
    
    if is_server_script():
        # Server mode - use server argument parser
        server_parser = argparse.ArgumentParser(description="NASA FL Server")
        server_parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
        server_parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
        server_parser.add_argument("--port", type=int, help="Server port (overrides config)")
        
        args = server_parser.parse_args()
        run_server(args)
    else:
        # Client mode - let the client script handle it
        print("❌ This script should be run as server only. Use client.py for clients.")
        sys.exit(1)

def run_server(args):
    """Run the FL server"""
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Extract server configuration
        server_config = config.get("server", {})
        strategy_config = config.get("strategy", {})
        data_config = config.get("data", {})
        
        # Get server parameters
        server_host = server_config.get("host", "0.0.0.0")
        server_port = args.port or server_config.get("port", 8080)
        num_rounds = server_config.get("num_rounds", 10)
        
        # Get strategy parameters - CRITICAL: Enable evaluation
        strategy_name = strategy_config.get("name", "fedavg")
        fraction_fit = strategy_config.get("fraction_fit", 1.0)
        fraction_evaluate = strategy_config.get("fraction_evaluate", 1.0)  # Enable evaluation
        min_fit_clients = strategy_config.get("min_fit_clients", 2)
        min_evaluate_clients = strategy_config.get("min_evaluate_clients", 2)  # Enable evaluation
        min_available_clients = strategy_config.get("min_available_clients", 
                                                   data_config.get("num_clients", 2))
        proximal_mu = strategy_config.get("proximal_mu", 0.1)
        
        # Experiment info
        experiment_id = config.get("experiment_id", "nasa_experiment")
        algorithm = config.get("algorithm", "fedavg")
        
        print("🚀 Starting NASA Federated Learning Server")
        print("=" * 50)
        print(f"Experiment ID: {experiment_id}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Server: {server_host}:{server_port}")
        print(f"Rounds: {num_rounds}")
        print(f"Strategy: {strategy_name}")
        print(f"Min Clients: {min_available_clients}")
        print(f"Fit Fraction: {fraction_fit}")
        print(f"Eval Fraction: {fraction_evaluate}")  # Should be > 0 for test evaluation
        print(f"Results Directory: {args.results_dir}")
        print("=" * 50)
        
        # Create results directory
        results_dir = os.path.join(args.results_dir, experiment_id)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize metrics collector
        metrics_collector = MetricsCollector(results_dir, experiment_id)
        
        # Create evaluation function (you need to implement proper test data loading)
        evaluate_fn = get_evaluate_fn()  # Pass testloader if available
        
        # Create strategy with proper configuration - ENABLE EVALUATION
        strategy = CustomFedAvg(
            metrics_collector=metrics_collector,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,  # This enables client evaluation
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,  # This enables client evaluation
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,  # Centralized evaluation
            on_fit_config_fn=lambda r: {"current_round": r},  # Send round number to clients
            on_evaluate_config_fn=lambda r: {"current_round": r}  # Also for evaluation
        )
        
        # Start Flower server
        print(f"🔄 Starting FL server on {server_host}:{server_port}...")
        print(f"📊 Evaluation enabled: {fraction_evaluate > 0 and min_evaluate_clients > 0}")
        
        # Save config for reference
        config_save_path = os.path.join(results_dir, "server_config.json")
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Start server
        fl.server.start_server(
            server_address=f"{server_host}:{server_port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            grpc_max_message_length=1024 * 1024 * 1024  # 1GB max message length
        )
        

        print("✅ Server completed successfully!")
        print(f"📊 Results saved to: {results_dir}")
        
        # Print final summary
        print("\n📈 FINAL SUMMARY:")
        print(f"   Round metrics: {os.path.join(metrics_collector.metrics_dir, 'round_metrics.csv')}")
        print(f"   Client metrics: {os.path.join(metrics_collector.metrics_dir, 'client_metrics.csv')}")
        print(f"   Test metrics: {os.path.join(metrics_collector.metrics_dir, 'test_metrics.csv')}")
        print(f"   Eval metrics: {os.path.join(metrics_collector.metrics_dir, 'eval_metrics.csv')}")
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()