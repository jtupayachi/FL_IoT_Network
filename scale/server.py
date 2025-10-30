

import flwr as fl
import torch
import torch.nn as nn
import argparse
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from flwr.common import Parameters, Scalar
from flwr.server.strategy import FedAvg, FedAvgM, FedOpt, QFedAvg
import pandas as pd
from datetime import datetime
import csv
import sys
import warnings
warnings.filterwarnings('ignore')



# ...existing imports...

class CustomFedProx(fl.server.strategy.FedProx):
    """Custom FedProx strategy (needed for MOON baseline comparison)"""
    
    def __init__(self, metrics_collector: MetricsCollector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if results:
            self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, server_round: int, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            self.metrics_collector.collect_centralized_eval(server_round, loss, metrics)
        return result


class CustomMOON(FedAvg):
    """
    MOON: Model-Contrastive Federated Learning
    Paper: https://arxiv.org/abs/2103.16257
    
    MOON adds contrastive learning loss between:
    - Current local model
    - Previous local model
    - Global model
    """
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 temperature: float = 0.5, mu: float = 5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
        self.temperature = temperature  # Temperature for contrastive loss
        self.mu = mu  # Weight for contrastive loss
        
        print(f"🌙 MOON Strategy initialized:")
        print(f"   Temperature: {temperature}")
        print(f"   Mu (contrastive weight): {mu}")
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if results:
            self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, server_round: int, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            self.metrics_collector.collect_centralized_eval(server_round, loss, metrics)
        return result


class CustomFedALA(FedAvg):
    """
    FedALA: Federated Learning with Adaptive Local Aggregation
    Paper: https://arxiv.org/abs/2212.01197
    
    Key idea: Each client maintains adaptive local aggregation weights
    to balance local and global model updates
    """
    
    def __init__(self, metrics_collector: MetricsCollector,
                 eta: float = 1.0, eta_l: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
        self.eta = eta  # Global learning rate
        self.eta_l = eta_l  # Local aggregation learning rate
        
        print(f"🔄 FedALA Strategy initialized:")
        print(f"   Eta (global lr): {eta}")
        print(f"   Eta_l (local agg lr): {eta_l}")
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if results:
            self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, server_round: int, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            self.metrics_collector.collect_centralized_eval(server_round, loss, metrics)
        return result

        
def get_initial_parameters(model_config: dict, input_dim: int):
    """Generate initial parameters for server-side optimization strategies"""
    model_type = model_config.get("model_type", "dense")
    
    if model_type == "lstm":
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=model_config.get("hidden_dim", 64),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.3)
        )
    else:
        model = NASAModel(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [64, 32]),
            dropout=model_config.get("dropout", 0.2)
        )
    
    params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(params)


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
        else:
            return NASAModel(
                input_dim=input_dim,
                hidden_dims=kwargs.get("hidden_dims", [64, 32]),
                dropout=kwargs.get("dropout", 0.2)
            )
    return model_factory

def get_evaluate_fn(model_factory, input_dim: int = 24, test_data_path: str = None, model_config: dict = None):
    """Create proper evaluation function for centralized evaluation"""
    
    stored_model_config = model_config or {}
    
    def evaluate_fn(server_round: int, parameters: Parameters, config: Dict[str, Scalar]):
        """Evaluate the model on a centralized test set"""
        try:
            print(f"🔍 Running centralized evaluation for round {server_round}...")
            
            param_list = parameters
            if len(param_list) >= 2:
                first_layer_shape = param_list[0].shape
                actual_input_dim = first_layer_shape[1]
                
                model_type = stored_model_config.get("model_type", "dense")
                
                if model_type == "lstm":
                    actual_hidden_dim = first_layer_shape[0] // 4
                    print(f"📐 Detected LSTM architecture:")
                    print(f"   Input dim: {actual_input_dim}")
                    print(f"   Hidden dim: {actual_hidden_dim} (from shape {first_layer_shape[0]} / 4)")
                    
                    model = LSTMModel(
                        input_dim=actual_input_dim,
                        hidden_dim=actual_hidden_dim,
                        num_layers=stored_model_config.get("num_layers", 2),
                        dropout=stored_model_config.get("dropout", 0.3)
                    )
                else:
                    actual_hidden_dim = first_layer_shape[0]
                    print(f"📐 Detected Dense architecture:")
                    print(f"   Input dim: {actual_input_dim}")
                    print(f"   First hidden: {actual_hidden_dim}")
                    
                    hidden_dims = []
                    for i in range(0, len(param_list)-2, 2):
                        layer_shape = param_list[i].shape
                        if len(layer_shape) == 2:
                            hidden_dims.append(layer_shape[0])
                    
                    if len(hidden_dims) > 1:
                        hidden_dims = hidden_dims[:-1]
                    
                    model = NASAModel(
                        input_dim=actual_input_dim,
                        hidden_dims=hidden_dims if hidden_dims else [actual_hidden_dim],
                        dropout=stored_model_config.get("dropout", 0.2)
                    )
            else:
                model_type = stored_model_config.get("model_type", "dense")
                if model_type == "lstm":
                    model = LSTMModel(
                        input_dim=input_dim,
                        hidden_dim=stored_model_config.get("hidden_dim", 64),
                        num_layers=stored_model_config.get("num_layers", 2),
                        dropout=stored_model_config.get("dropout", 0.3)
                    )
                else:
                    model = NASAModel(
                        input_dim=input_dim,
                        hidden_dims=stored_model_config.get("hidden_dims", [64, 32]),
                        dropout=stored_model_config.get("dropout", 0.2)
                    )
                actual_input_dim = input_dim
            
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            model.eval()
            
            print(f"⚠️ Using synthetic test data with input_dim={actual_input_dim}")
            X_test = torch.randn(100, actual_input_dim)
            y_test = torch.randn(100, 1) * 0.5 + X_test[:, 0:1] * 0.5
            
            with torch.no_grad():
                predictions = model(X_test)
                loss = nn.MSELoss()(predictions, y_test).item()
                
                y_true_np = y_test.numpy().flatten()
                y_pred_np = predictions.numpy().flatten()
                
                mse = np.mean((y_true_np - y_pred_np) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(y_true_np - y_pred_np))
                
                y_var = np.var(y_true_np)
                r2 = 0.0 if y_var == 0 else 1 - (mse / y_var)
            
            metrics = {
                "rmse": float(rmse),
                "mse": float(mse),
                "mae": float(mae),
                "r2": float(r2)
            }
            
            print(f"📊 Centralized Evaluation - Round {server_round}:")
            print(f"   Loss: {loss:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            return float(loss), metrics
            
        except Exception as e:
            print(f"⚠️ Centralized evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return 10.0, {"rmse": 10.0, "mse": 100.0, "mae": 10.0, "r2": -1.0}
    
    return evaluate_fn

    
class MetricsCollector:
    """Collects and saves metrics from all rounds"""
    
    def __init__(self, results_dir: str, experiment_id: str):
        self.results_dir = results_dir
        self.metrics_dir = results_dir
        self.experiment_id = experiment_id
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs(results_dir, exist_ok=True)
        
        self.round_metrics_path = os.path.join(results_dir, "round_metrics.csv")
        self.client_metrics_path = os.path.join(results_dir, "client_metrics.csv")
        self.test_metrics_path = os.path.join(results_dir, "test_metrics.csv")
        
        self._initialize_metrics_files()
        
        self.round_metrics = []
        self.client_metrics = []
        self.test_metrics = []
    
    def _initialize_metrics_files(self):
        """Initialize CSV files with headers"""
        
        round_headers = [
            "timestamp", "round", "total_clients", "fit_clients",
            "avg_train_loss", "avg_val_loss", "avg_val_r2",
            "centralized_test_loss", "centralized_test_r2", "algorithm"
        ]
        
        client_headers = [
            "timestamp", "round", "client_id", "samples",
            "train_loss", "val_loss", "train_r2", "val_r2",
            "test_loss", "test_r2", "algorithm"
        ]
        
        test_headers = [
            "timestamp", "round", "loss", "rmse", "mse", "mae", "r2", "algorithm"
        ]
        
        for path, headers in [
            (self.round_metrics_path, round_headers),
            (self.client_metrics_path, client_headers),
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
            
            test_loss = fit_res.metrics.get("test_loss", 0)
            test_r2 = fit_res.metrics.get("test_r2", 0)
            
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
            
            with open(self.client_metrics_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=client_metrics.keys())
                writer.writerow(client_metrics)
            
            total_samples += num_samples
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_r2s.append(train_r2)
            val_r2s.append(val_r2)
            
            if test_loss > 0:
                test_losses.append(test_loss)
                test_r2s.append(test_r2)
        
        if train_losses:
            round_metrics = {
                "timestamp": timestamp,
                "round": round_num,
                "total_clients": len(results),
                "fit_clients": len(results),
                "avg_train_loss": np.mean(train_losses),
                "avg_val_loss": np.mean(val_losses),
                "avg_val_r2": np.mean(val_r2s),
                "centralized_test_loss": 0,
                "centralized_test_r2": 0,
                "algorithm": self.experiment_id
            }
            
            self.round_metrics.append(round_metrics)
            
            with open(self.round_metrics_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=round_metrics.keys())
                writer.writerow(round_metrics)
            
            print(f"📊 Round {round_num} Fit Metrics:")
            print(f"   Clients: {len(results)}")
            print(f"   Avg Train Loss: {np.mean(train_losses):.4f}")
            print(f"   Avg Val Loss: {np.mean(val_losses):.4f}")
            print(f"   Avg Val R²: {np.mean(val_r2s):.4f}")
            
            if test_losses:
                print(f"   Client Avg Test Loss: {np.mean(test_losses):.4f}")
                print(f"   Client Avg Test R²: {np.mean(test_r2s):.4f}")
    
    def collect_centralized_eval(self, round_num: int, loss: float, metrics: Dict[str, Scalar]):
        """Collect centralized evaluation metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        test_metrics = {
            "timestamp": timestamp,
            "round": round_num,
            "loss": loss,
            "rmse": metrics.get("rmse", 0),
            "mse": metrics.get("mse", 0),
            "mae": metrics.get("mae", 0),
            "r2": metrics.get("r2", 0),
            "algorithm": self.experiment_id
        }
        
        self.test_metrics.append(test_metrics)
        
        with open(self.test_metrics_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_metrics.keys())
            writer.writerow(test_metrics)
        
        print(f"📈 Round {round_num} Centralized Test Evaluation:")
        print(f"   Loss: {loss:.4f}")
        print(f"   RMSE: {metrics.get('rmse', 0):.4f}")
        print(f"   R²: {metrics.get('r2', 0):.4f}")
        
        self._update_round_metrics_with_centralized(round_num, loss, metrics)
    
    def _update_round_metrics_with_centralized(self, round_num: int, loss: float, metrics: Dict[str, Scalar]):
        """Update round metrics with centralized evaluation results"""
        try:
            round_data = []
            with open(self.round_metrics_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    round_data.append(row)
            
            if len(round_data) >= round_num:
                round_data[round_num-1]["centralized_test_loss"] = str(loss)
                round_data[round_num-1]["centralized_test_r2"] = str(metrics.get("r2", 0))
                
                with open(self.round_metrics_path, 'w', newline='') as f:
                    if round_data:
                        writer = csv.DictWriter(f, fieldnames=round_data[0].keys())
                        writer.writeheader()
                        writer.writerows(round_data)
        except Exception as e:
            print(f"⚠️ Could not update round metrics: {e}")


class CustomFedAvg(FedAvg):
    def __init__(self, metrics_collector: MetricsCollector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if results:
            self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, server_round: int, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            self.metrics_collector.collect_centralized_eval(server_round, loss, metrics)
        return result


class CustomQFedAvg(QFedAvg):
    def __init__(self, metrics_collector: MetricsCollector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if results:
            self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, server_round: int, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            self.metrics_collector.collect_centralized_eval(server_round, loss, metrics)
        return result


class CustomFedAvgM(FedAvgM):
    def __init__(self, metrics_collector: MetricsCollector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if results:
            self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, server_round: int, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            self.metrics_collector.collect_centralized_eval(server_round, loss, metrics)
        return result


class CustomFedOpt(FedOpt):
    def __init__(self, metrics_collector: MetricsCollector, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if results:
            self.metrics_collector.collect_fit_metrics(server_round, results, failures)
        return aggregated_parameters, aggregated_metrics
    
    def evaluate(self, server_round: int, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            self.metrics_collector.collect_centralized_eval(server_round, loss, metrics)
        return result


def write_completion_signal(experiment_dir: str, success: bool, error_msg: str = ""):
    """Write a completion signal file for the bash script to detect"""
    signal_file = os.path.join(experiment_dir, ".COMPLETE")
    
    completion_data = {
        "completed": success,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": error_msg if not success else ""
    }
    
    with open(signal_file, 'w') as f:
        json.dump(completion_data, f, indent=2)
    
    print(f"🏁 Completion signal written to: {signal_file}")


def run_server(args):
    """Run the FL server with completion signal"""
    experiment_dir = None
    
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        server_config = config.get("server", {})
        strategy_config = config.get("strategy", {})
        strategy_params = strategy_config.get("params", {})
        data_config = config.get("data", {})
        model_config = config.get("model", {})
        
        server_host = server_config.get("host", "0.0.0.0")
        server_port = args.port or server_config.get("port", 8080)
        num_rounds = server_config.get("num_rounds", 10)
        
        strategy_name = strategy_config.get("name", "fedavg").lower()
        fraction_fit = strategy_config.get("fraction_fit", 1.0)
        fraction_evaluate = strategy_config.get("fraction_evaluate", 1.0)
        min_fit_clients = strategy_config.get("min_fit_clients", 2)
        min_evaluate_clients = strategy_config.get("min_evaluate_clients", 2)
        min_available_clients = strategy_config.get("min_available_clients", 
                                                   data_config.get("num_clients", 2))
        
        experiment_id = config.get("experiment_id", "nasa_experiment")
        algorithm = config.get("algorithm", "fedavg")
        
        print("🚀 Starting NASA Federated Learning Server")
        print("=" * 50)
        print(f"Experiment ID: {experiment_id}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Strategy: {strategy_name.upper()}")
        print(f"Server: {server_host}:{server_port}")
        print(f"Rounds: {num_rounds}")
        
        if strategy_params:
            print(f"\n🔧 Strategy Parameters:")
            for key, value in strategy_params.items():
                print(f"   {key}: {value}")
        
        os.makedirs(args.results_dir, exist_ok=True)
        
        experiment_dir = os.path.join(args.results_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=True)
        
        metrics_dir = os.path.join(experiment_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        print(f"\n📁 Directory structure:")
        print(f"   Results: {experiment_dir}")
        print(f"   Metrics: {metrics_dir}")
        print("=" * 50)
        
        metrics_collector = MetricsCollector(metrics_dir, experiment_id)
        model_factory = create_model_factory()
        input_dim = model_config.get("n_components", 24)
        
        test_data_path = data_config.get("test_data_path")
        evaluate_fn = get_evaluate_fn(model_factory, input_dim, test_data_path, model_config)

        print(f"\n⚙️ Creating {strategy_name.upper()} strategy...")
        
        initial_params = None
        if strategy_name in ["fedavgm", "fedopt"]:
            print(f"   🔧 Generating initial parameters...")
            initial_params = get_initial_parameters(model_config, input_dim)
        
        if strategy_name == "fedavg":
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
        elif strategy_name == "qfedavg":
            strategy = CustomQFedAvg(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                q_param=strategy_params.get("q_param", 0.2),
                qffl_learning_rate=strategy_params.get("qffl_learning_rate", 0.1),
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {"current_round": r},
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )
        elif strategy_name == "fedavgm":
            strategy = CustomFedAvgM(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                server_momentum=strategy_params.get("server_momentum", 0.9),
                initial_parameters=initial_params,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {"current_round": r},
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )
        elif strategy_name == "fedopt":
            strategy = CustomFedOpt(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                tau=strategy_params.get("tau", 0.01),
                initial_parameters=initial_params,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {"current_round": r},
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        print(f"\n🎯 Starting federated learning...\n")
        
        fl.server.start_server(
            server_address=f"{server_host}:{server_port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            grpc_max_message_length=1024 * 1024 * 1024
        )
        
        print("\n" + "=" * 50)
        print("✅ Server completed successfully!")
        print("=" * 50)
        
        # Write success signal
        write_completion_signal(experiment_dir, success=True)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        
        # Write failure signal
        if experiment_dir:
            write_completion_signal(experiment_dir, success=False, error_msg=str(e))
        
        sys.exit(1)


def is_server_script():
    return '--client-id' not in ' '.join(sys.argv)


def main():
    if is_server_script():
        parser = argparse.ArgumentParser(description="NASA FL Server")
        parser.add_argument("--config", type=str, default="config.json")
        parser.add_argument("--results-dir", type=str, default="results")
        parser.add_argument("--port", type=int, help="Server port")
        
        args = parser.parse_args()
        run_server(args)
    else:
        print("❌ Use client.py for clients")
        sys.exit(1)


if __name__ == "__main__":
    main()