

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

# ============================================================================
# HELPER FUNCTIONS AND MODEL DEFINITIONS (FIRST)
# ============================================================================

def get_initial_parameters(model_config: dict, input_dim: int):
    """Generate initial model parameters"""
    model_type = model_config.get("model_type", "dense")
    
    if model_type == "lstm":
        # ✅ Convert config to match client architecture
        hidden_dim = model_config.get("hidden_dim", 64)
        num_layers = model_config.get("num_layers", 2)
        hidden_dims = [hidden_dim] * num_layers  # Convert to list format
        dropout = model_config.get("dropout", 0.3)
        
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
    else:
        model = NASAModel(
            input_dim=input_dim,
            hidden_dims=model_config.get("hidden_dims", [64, 32]),
            dropout=model_config.get("dropout", 0.2)
        )
    
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class LSTMModel(nn.Module):
    """LSTM model for RUL prediction - matches client.py architecture"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 64], 
                 dropout: float = 0.2, output_dim: int = 1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Create stacked LSTM layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.LSTM(
                prev_dim, 
                hidden_dim, 
                batch_first=True, 
                dropout=dropout if i < len(hidden_dims) - 1 else 0
            ))
            prev_dim = hidden_dim
        
        self.lstm_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Pass through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Take last timestep
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.fc(x)
        return x


class NASAModel(nn.Module):
    """Dense model for NASA RUL prediction"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 dropout: float = 0.2):
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
    """Factory to create models"""
    def factory(model_type: str, input_dim: int, **kwargs):
        if model_type == "lstm":
            return LSTMModel(input_dim=input_dim, **kwargs)
        else:
            return NASAModel(input_dim=input_dim, **kwargs)
    return factory


def get_evaluate_fn(model_factory, input_dim: int = 24, test_data_path: str = None, model_config: dict = None):
    """Return evaluation function for server-side evaluation"""
    
    stored_model_config = model_config or {}
    
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
        """Evaluate the model on a centralized test set"""
        try:
            print(f"🔍 Running centralized evaluation for round {server_round}...")
            
            # ✅ Guard against empty parameters (initial round)
            if not parameters or len(parameters) == 0:
                print("⚠️ No parameters received, skipping evaluation")
                return None
            
            model_type = stored_model_config.get("model_type", "dense")
            
            if model_type == "lstm":
                # ✅ Convert config to match client architecture
                hidden_dim = stored_model_config.get("hidden_dim", 64)
                num_layers = stored_model_config.get("num_layers", 2)
                hidden_dims = [hidden_dim] * num_layers
                dropout = stored_model_config.get("dropout", 0.3)
                
                model = LSTMModel(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout
                )
            else:
                model = NASAModel(
                    input_dim=input_dim,
                    hidden_dims=stored_model_config.get("hidden_dims", [64, 32]),
                    dropout=stored_model_config.get("dropout", 0.2)
                )
            
            # ✅ Add parameter count validation
            expected_keys = list(model.state_dict().keys())
            if len(parameters) != len(expected_keys):
                print(f"⚠️ Parameter mismatch: expected {len(expected_keys)}, got {len(parameters)}")
                print("   Skipping evaluation for this round")
                return None
            
            try:
                params_dict = zip(expected_keys, parameters)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                model.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                print(f"⚠️ Failed to load parameters: {exc}")
                return None
            
            model.eval()
            
            # Use synthetic test data
            print(f"⚠️ Using synthetic test data with input_dim={input_dim}")
            X_test = torch.randn(100, input_dim)
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
            return None
    
    return evaluate


# ============================================================================
# METRICS COLLECTOR (DEFINE BEFORE STRATEGIES)
# ============================================================================

# class MetricsCollector:
#     """Collect and save metrics during federated learning"""
#     def __init__(self, save_dir: str, experiment_id: str):
#         self.save_dir = save_dir
#         self.experiment_id = experiment_id
#         self.round_metrics = []
        
#         os.makedirs(save_dir, exist_ok=True)
        
#         self.fit_csv = os.path.join(save_dir, f"{experiment_id}_fit_metrics.csv")
#         self.eval_csv = os.path.join(save_dir, f"{experiment_id}_eval_metrics.csv")
        
#         with open(self.fit_csv, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["round", "client_id", "num_examples", "loss", "mae", 
#                            "train_loss", "val_loss", "train_rmse", "val_rmse", 
#                            "train_r2", "val_r2", "algorithm"])
        
#         with open(self.eval_csv, 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(["round", "loss", "rmse", "timestamp"])
    
#     def collect_fit_metrics(self, server_round: int, results, failures):
#         """Collect metrics from fit round"""
#         for client_proxy, fit_res in results:
#             metrics = fit_res.metrics
            
#             row = [
#                 server_round,
#                 metrics.get("client_id", "unknown"),
#                 fit_res.num_examples,
#                 metrics.get("loss", 0),
#                 metrics.get("mae", 0),
#                 metrics.get("train_loss", 0),
#                 metrics.get("val_loss", 0),
#                 metrics.get("train_rmse", 0),
#                 metrics.get("val_rmse", 0),
#                 metrics.get("train_r2", 0),
#                 metrics.get("val_r2", 0),
#                 metrics.get("algorithm", "unknown")
#             ]
            
#             with open(self.fit_csv, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(row)
    
#     def collect_centralized_eval(self, server_round: int, loss: float, metrics: Dict):
#         """Collect centralized evaluation metrics"""
#         row = [
#             server_round,
#             loss,
#             metrics.get("rmse", 0),
#             datetime.now().isoformat()
#         ]
        
#         with open(self.eval_csv, 'a', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerow(row)

class MetricsCollector:
    """Collect and save metrics during federated learning"""
    def __init__(self, save_dir: str, experiment_id: str):
        self.save_dir = save_dir
        self.experiment_id = experiment_id
        self.round_metrics = []
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.fit_csv = os.path.join(save_dir, f"{experiment_id}_fit_metrics.csv")
        self.eval_csv = os.path.join(save_dir, f"{experiment_id}_eval_metrics.csv")
        
        # ✅ Updated header with MAE
        with open(self.fit_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "client_id", "num_examples", "loss", "mae", 
                "train_loss", "val_loss", "train_rmse", "val_rmse", 
                "train_mse", "val_mse", "train_mae", "val_mae",
                "train_r2", "val_r2", "algorithm"
            ])
        
        # ✅ Updated header with MAE
        with open(self.eval_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "loss", "rmse", "mse", "mae", "r2", "timestamp"
            ])
    
    def collect_fit_metrics(self, server_round: int, results, failures):
        """Collect metrics from fit round"""
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            
            # ✅ Updated row with MAE
            row = [
                server_round,
                metrics.get("client_id", "unknown"),
                fit_res.num_examples,
                metrics.get("loss", 0),
                metrics.get("mae", 0),
                metrics.get("train_loss", 0),
                metrics.get("val_loss", 0),
                metrics.get("train_rmse", 0),
                metrics.get("val_rmse", 0),
                metrics.get("train_mse", 0),
                metrics.get("val_mse", 0),
                metrics.get("train_mae", 0),
                metrics.get("val_mae", 0),
                metrics.get("train_r2", 0),
                metrics.get("val_r2", 0),
                metrics.get("algorithm", "unknown")
            ]
            
            with open(self.fit_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
    
    def collect_centralized_eval(self, server_round: int, loss: float, metrics: Dict):
        """Collect centralized evaluation metrics"""
        # ✅ Updated row with MAE
        row = [
            server_round,
            loss,
            metrics.get("rmse", 0),
            metrics.get("mse", 0),
            metrics.get("mae", 0),
            metrics.get("r2", 0),
            datetime.now().isoformat()
        ]
        
        with open(self.eval_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)


# ============================================================================
# CUSTOM STRATEGIES (AFTER METRICSCOLLECTOR)
# ============================================================================

class CustomFedAvg(FedAvg):
    """Custom FedAvg with metrics collection"""
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
    """Custom QFedAvg with metrics collection"""
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
    """Custom FedAvgM with metrics collection"""
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
    """Custom FedOpt with metrics collection"""
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


class CustomFedProx(fl.server.strategy.FedProx):
    """Custom FedProx with metrics collection"""
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
    """MOON: Model-Contrastive Federated Learning"""
    def __init__(self, metrics_collector: MetricsCollector, 
                 temperature: float = 0.5, mu: float = 5.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
        self.temperature = temperature
        self.mu = mu
        
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
    """FedALA: Federated Learning with Adaptive Local Aggregation"""
    def __init__(self, metrics_collector: MetricsCollector,
                 eta: float = 1.0, eta_l: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_collector = metrics_collector
        self.eta = eta
        self.eta_l = eta_l
        
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
        
        # ✅ Always generate initial parameters for all strategies
        print(f"   🔧 Generating initial parameters...")
        initial_params = get_initial_parameters(model_config, input_dim)
        
        # Strategy creation with all algorithms
        if strategy_name == "fedavg":
            strategy = CustomFedAvg(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {"current_round": r, "algorithm": "fedavg"},
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )
        
        elif strategy_name == "qfedavg":
            q_param = strategy_params.get("q_param", 0.2)
            qffl_lr = strategy_params.get("qffl_learning_rate", 0.1)
            
            print(f"   ✓ q_param: {q_param}")
            print(f"   ✓ qffl_learning_rate: {qffl_lr}")
            
            strategy = CustomQFedAvg(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
                q_param=q_param,
                qffl_learning_rate=qffl_lr,
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {"current_round": r, "algorithm": "qfedavg"},
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )

        elif strategy_name == "fedavgm":
            server_momentum = strategy_params.get("server_momentum", 0.9)
            
            print(f"   ✓ server_momentum: {server_momentum}")
            
            strategy = CustomFedAvgM(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                server_momentum=server_momentum,
                initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {"current_round": r, "algorithm": "fedavgm"},
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )
        
        elif strategy_name == "fedopt":
            tau = strategy_params.get("tau", 0.01)
            
            print(f"   ✓ tau: {tau}")
            
            strategy = CustomFedOpt(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                tau=tau,
                initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {"current_round": r, "algorithm": "fedopt"},
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )
        
        elif strategy_name == "moon":
            temperature = strategy_params.get("temperature", 0.5)
            mu = strategy_params.get("mu", 5.0)
            
            print(f"   ✓ temperature: {temperature}")
            print(f"   ✓ mu: {mu}")
            
            strategy = CustomMOON(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                temperature=temperature,
                mu=mu,
                initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {
                    "current_round": r, 
                    "algorithm": "moon",
                    "temperature": temperature,
                    "mu": mu
                },
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )
        
        elif strategy_name == "fedala":
            eta = strategy_params.get("eta", 1.0)
            eta_l = strategy_params.get("eta_l", 0.1)
            
            print(f"   ✓ eta: {eta}")
            print(f"   ✓ eta_l: {eta_l}")
            
            strategy = CustomFedALA(
                metrics_collector=metrics_collector,
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=min_fit_clients,
                min_evaluate_clients=min_evaluate_clients,
                min_available_clients=min_available_clients,
                eta=eta,
                eta_l=eta_l,
                initial_parameters=fl.common.ndarrays_to_parameters(initial_params),
                evaluate_fn=evaluate_fn,
                on_fit_config_fn=lambda r: {
                    "current_round": r, 
                    "algorithm": "fedala",
                    "eta": eta,
                    "eta_l": eta_l
                },
                on_evaluate_config_fn=lambda r: {"current_round": r}
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}. "
                           f"Supported: fedavg, qfedavg, fedavgm, fedopt, moon, fedala")
        
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