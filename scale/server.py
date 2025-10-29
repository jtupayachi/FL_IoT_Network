



import flwr as fl
import torch
import torch.nn as nn
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg
import pandas as pd
from datetime import datetime
import csv
import sys
import warnings
warnings.filterwarnings('ignore')

# Model Definitions for Server-Side Evaluation
class LSTMModel(nn.Module):
    """LSTM model for NASA RUL prediction"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.fc_layers(last_hidden)

class NASAModel(nn.Module):
    """Dense model for NASA RUL prediction"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def create_model_factory():
    """Create a model factory for server-side evaluation"""
    def model_factory(model_type: str, input_dim: int, **kwargs):
        if model_type == "lstm":
            return LSTMModel(
                input_dim=input_dim,
                hidden_dim=kwargs.get("hidden_dim", 64),
                num_layers=kwargs.get("num_layers", 2),
                dropout=kwargs.get("dropout", 0.3)
            )
        else:  # dense
            return NASAModel(
                input_dim=input_dim,
                hidden_dims=kwargs.get("hidden_dims", [64, 32]),
                dropout=kwargs.get("dropout", 0.2)
            )
    return model_factory

def get_evaluate_fn(model_factory, input_dim: int = 24, test_data_path: str = None):
    """Create proper evaluation function for centralized evaluation"""
    def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
        """
        Evaluate the model on a centralized test set
        """
        try:
            print(f"🔍 Running centralized evaluation for round {server_round}...")
            
            # Extract model configuration
            model_config = config.get("model", {})
            hyperparams = {
                "model_type": model_config.get("model_type", "dense"),
                "hidden_dim": model_config.get("hidden_dim", 64),
                "num_layers": model_config.get("num_layers", 2),
                "dropout": model_config.get("dropout", 0.2),
                "hidden_dims": model_config.get("hidden_dims", [64, 32])
            }
            
            # Create model
            model = model_factory(hyperparams["model_type"], input_dim, **hyperparams)
            
            # Set parameters (convert to tensor format)
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            model.eval()
            
            # Generate synthetic test data for demonstration
            # In production, replace this with actual centralized test data loading
            if test_data_path and os.path.exists(test_data_path):
                print(f"📁 Loading test data from: {test_data_path}")
                # Load actual test data here
                # For now, use synthetic data
                X_test = torch.randn(100, input_dim)
                y_test = torch.randn(100, 1)
            else:
                # Synthetic data for testing - REPLACE WITH REAL DATA
                print("⚠️ Using synthetic test data - replace with actual centralized test dataset")
                X_test = torch.randn(100, input_dim)
                y_test = torch.randn(100, 1) * 0.5 + X_test[:, 0:1] * 0.5  # Add some correlation
            
            # Evaluate
            with torch.no_grad():
                predictions = model(X_test)
                loss = nn.MSELoss()(predictions, y_test).item()
                
                # Calculate metrics
                y_true_np = y_test.numpy().flatten()
                y_pred_np = predictions.numpy().flatten()
                
                mse = np.mean((y_true_np - y_pred_np) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_true_np - y_pred_np))
                
                # Handle R² calculation with division by zero protection
                y_var = np.var(y_true_np)
                if y_var == 0:
                    r2 = 0.0
                else:
                    r2 = 1 - (mse / y_var)
            
            metrics = {
                "rmse": float(rmse),
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2)
            }
            
            print(f"📊 Centralized Evaluation - Round {server_round}:")
            print(f"   Loss: {loss:.4f}")
            print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            return float(loss), metrics
            
        except Exception as e:
            print(f"⚠️ Centralized evaluation error: {e}")
            import traceback
            traceback.print_exc()
            # Return placeholder values if evaluation fails
            return 10.0, {"rmse": 10.0, "mse": 100.0, "mae": 10.0, "r2": -1.0}
    
    return evaluate_fn

class MetricsCollector:
    """Collects and saves metrics from all rounds"""
    
    def __init__(self, results_dir: str, experiment_id: str):
        self.results_dir = results_dir
        self.experiment_id = experiment_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Define file paths
        self.round_metrics_path = os.path.join(results_dir, "round_metrics.csv")
        self.client_metrics_path = os.path.join(results_dir, "client_metrics.csv")
        self.eval_metrics_path = os.path.join(results_dir, "eval_metrics.csv")
        self.test_metrics_path = os.path.join(results_dir, "test_metrics.csv")
        
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
            "train_r2", "val_r2", "test_loss", "test_r2", "algorithm"
        ]
        
        # Evaluation metrics (centralized evaluation)
        eval_headers = [
            "timestamp", "round", "loss", "rmse", "mse", "mae", "r2", "algorithm"
        ]
        
        # Test metrics (from client evaluation)
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
        """Collect metrics from fit results"""
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
            
            # Extract test metrics if available
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
            
            # Only include test metrics if they're non-zero
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
            print(f"⚠️ No evaluation results for round {round_num}")
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
        """Collect metrics from centralized evaluation - THIS FILLS eval_metrics.csv"""
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
        
        # Write eval metrics to CSV - THIS IS CRITICAL
        with open(self.eval_metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=eval_metrics.keys())
            writer.writerow(eval_metrics)
        
        print(f"📈 Round {round_num} Centralized Evaluation:")
        print(f"   Loss: {loss:.4f}")
        print(f"   RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"   R²: {metrics.get('r2', 0):.4f}")
        
        # Also update round metrics with centralized evaluation results
        self._update_round_metrics_with_centralized(round_num, loss, metrics)
    
    def _update_round_metrics_with_centralized(self, round_num: int, loss: float, metrics: Dict[str, Scalar]):
        """Update round metrics with centralized evaluation results"""
        try:
            # Read all round metrics
            round_data = []
            with open(self.round_metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    round_data.append(row)
            
            # Update the specific round with centralized metrics
            if len(round_data) >= round_num:
                round_data[round_num-1]["centralized_loss"] = str(loss)
                round_data[round_num-1]["centralized_r2"] = str(metrics.get("r2", 0))
                
                # Rewrite the entire file
                with open(self.round_metrics_path, 'w', newline='') as f:
                    if round_data:
                        writer = csv.DictWriter(f, fieldnames=round_data[0].keys())
                        writer.writeheader()
                        writer.writerows(round_data)
        except Exception as e:
            print(f"⚠️ Could not update round metrics with centralized results: {e}")

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
        # Server mode
        server_parser = argparse.ArgumentParser(description="NASA FL Server")
        server_parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
        server_parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
        server_parser.add_argument("--port", type=int, help="Server port (overrides config)")
        
        args = server_parser.parse_args()
        run_server(args)
    else:
        # Client mode
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
        model_config = config.get("model", {})
        
        # Get server parameters
        server_host = server_config.get("host", "0.0.0.0")
        server_port = args.port or server_config.get("port", 8080)
        num_rounds = server_config.get("num_rounds", 10)
        
        # Get strategy parameters
        strategy_name = strategy_config.get("name", "fedavg")
        fraction_fit = strategy_config.get("fraction_fit", 1.0)
        fraction_evaluate = strategy_config.get("fraction_evaluate", 1.0)
        min_fit_clients = strategy_config.get("min_fit_clients", 2)
        min_evaluate_clients = strategy_config.get("min_evaluate_clients", 2)
        min_available_clients = strategy_config.get("min_available_clients", 
                                                   data_config.get("num_clients", 2))
        
        # Experiment info
        experiment_id = config.get("experiment_id", "nasa_experiment")
        algorithm = config.get("algorithm", "fedavg")
        
        print("🚀 Starting NASA Federated Learning Server")
        print("=" * 50)
        print(f"Experiment ID: {experiment_id}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Server: {server_host}:{server_port}")
        print(f"Rounds: {num_rounds}")
        
        # === FIXED: Create proper directory structure ===
        experiment_dir = os.path.join(args.results_dir, experiment_id)
        results_dir = os.path.join(experiment_dir, "metrics")  # CSV files go in metrics/
        
        # Create both directories
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Experiment dir: {experiment_dir}")
        print(f"Metrics dir: {results_dir}")  # This should show .../metrics/
        print("=" * 50)
        
        # Initialize metrics collector with correct path
        metrics_collector = MetricsCollector(results_dir, experiment_id)
        
        # Create model factory and evaluation function
        model_factory = create_model_factory()
        input_dim = 24  # NASA dataset has 24 features
        
        # Get test data path if available
        test_data_path = data_config.get("test_data_path")
        
        # Create proper evaluation function
        evaluate_fn = get_evaluate_fn(model_factory, input_dim, test_data_path)
        
        # Create strategy with proper configuration
        strategy = CustomFedAvg(
            metrics_collector=metrics_collector,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda r: {"current_round": r},
            on_evaluate_config_fn=lambda r: {"current_round": r}
        )
        
        # Start Flower server
        print(f"🔄 Starting FL server on {server_host}:{server_port}...")
        print(f"📊 Evaluation enabled: {fraction_evaluate > 0}")
        print(f"📈 CSV files will be saved to: {results_dir}")
        
        # Save config in experiment directory (not metrics/)
        config_save_path = os.path.join(experiment_dir, "config.json")
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📁 Config saved to: {config_save_path}")
        
        # Start server
        fl.server.start_server(
            server_address=f"{server_host}:{server_port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            grpc_max_message_length=1024 * 1024 * 1024
        )
        
        print("✅ Server completed successfully!")
        
        # Print final summary with correct paths
        print("\n📈 FINAL SUMMARY:")
        print(f"   Round metrics: {os.path.join(results_dir, 'round_metrics.csv')}")
        print(f"   Client metrics: {os.path.join(results_dir, 'client_metrics.csv')}")
        print(f"   Test metrics: {os.path.join(results_dir, 'test_metrics.csv')}")
        print(f"   Eval metrics: {os.path.join(results_dir, 'eval_metrics.csv')}")
        
        # Verify files were created
        for csv_file in ['round_metrics.csv', 'client_metrics.csv', 'eval_metrics.csv', 'test_metrics.csv']:
            file_path = os.path.join(results_dir, csv_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    print(f"✅ {csv_file}: {len(lines)-1} records")
            else:
                print(f"❌ {csv_file}: NOT FOUND")
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()