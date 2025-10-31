
import os
import sys
import json
import argparse
import warnings
import time
import csv
from collections import OrderedDict
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM model for RUL prediction"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 dropout: float = 0.2, output_dim: int = 1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.LSTM(
                prev_dim, hidden_dim, 
                batch_first=True, 
                dropout=dropout if i < len(hidden_dims) - 1 else 0
            ))
            prev_dim = hidden_dim
        
        self.lstm_layers = nn.ModuleList(layers)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        x = x[:, -1, :]
        x = self.relu(x)
        x = self.fc(x)
        return x


class NASAModel(nn.Module):
    """Dense feedforward model for RUL prediction"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 dropout: float = 0.2, output_dim: int = 1):
        super(NASAModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class ModelFactory:
    """Factory for creating models"""
    @staticmethod
    def create_model(model_type: str, input_dim: int, hidden_dims: List[int] = [64, 32],
                    dropout: float = 0.2, output_dim: int = 1):
        if model_type.lower() == "lstm":
            return LSTMModel(input_dim, hidden_dims, dropout, output_dim)
        elif model_type.lower() == "dense":
            return NASAModel(input_dim, hidden_dims, dropout, output_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# HYPERPARAMETERS
# ============================================================================



class Hyperparameters:
    """Container for hyperparameters"""
    def __init__(self, config: Dict):
        model_config = config.get("model", {})
        data_config = config.get("data", {})
        
        # Model parameters
        self.model_type = model_config.get("model_type", "dense")
        
        # ✅ FIX: Convert LSTM config from template format to hidden_dims list
        if self.model_type == "lstm":
            # Read from config template: "hidden_dim": 64, "num_layers": 2
            hidden_dim = model_config.get("hidden_dim", 64)
            num_layers = model_config.get("num_layers", 2)
            # Convert to list format: [64, 64]
            self.hidden_dims = [hidden_dim] * num_layers
            print(f"   🔧 LSTM config: hidden_dim={hidden_dim}, num_layers={num_layers}")
            print(f"   ✅ Converted to hidden_dims={self.hidden_dims}")
        else:
            # For dense models, use hidden_dims as-is
            self.hidden_dims = model_config.get("hidden_dims", [64, 32])
        
        self.dropout = model_config.get("dropout", 0.2)
        self.learning_rate = model_config.get("learning_rate", 0.001)
        self.local_epochs = model_config.get("local_epochs", 1)
        self.batch_size = model_config.get("batch_size", 32)
        
        # LSTM specific
        self.sequence_length = model_config.get("sequence_length", 10)
        
        # Dimensionality reduction
        self.reduction_type = model_config.get("reduction_type", "none")
        self.n_components = model_config.get("n_components", 10)
        self.kernel = model_config.get("kernel", "rbf")
        
        # Data parameters
        self.test_size = data_config.get("test_size", 0.2)
        self.k_folds = model_config.get("k_folds", 3)
        
        # Early stopping
        self.early_stopping_enabled = model_config.get("early_stopping_enabled", True)
        self.early_stopping_patience = model_config.get("early_stopping_patience", 5)
        self.early_stopping_min_delta = model_config.get("early_stopping_min_delta", 0.001)
        
        # RUL configuration
        self.rul_mode = model_config.get("rul_mode", "linear")
        self.rul_power = model_config.get("rul_power", 1)
        self.max_rul = model_config.get("max_rul", None)


# ============================================================================
# METRICS LOGGER
# ============================================================================

class MetricsLogger:
    """Logger for client metrics"""
    def __init__(self, client_id: str, log_dir: str = "logs"):
        self.client_id = client_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV files
        self.training_csv = os.path.join(log_dir, f"client_{client_id}_training_metrics_{timestamp}.csv")
        self.test_csv = os.path.join(log_dir, f"client_{client_id}_test_metrics_{timestamp}.csv")
        self.cv_csv = os.path.join(log_dir, f"client_{client_id}_cv_metrics_{timestamp}.csv")
        
        # Initialize CSV files
        self._initialize_csv_files()
        
        # Metrics storage
        self.training_history = []
        self.test_history = []
        self.cv_history = []
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers"""
        # Training metrics
        with open(self.training_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "round", "train_loss", "val_loss", "avg_epoch_loss",
                "train_rmse", "train_mse", "train_mae", "train_r2",
                "val_rmse", "val_mse", "val_mae", "val_r2",
                "samples", "algorithm", "client_id", "model_type",
                "learning_rate", "batch_size"
            ])
        
        # Test metrics
        with open(self.test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "round", "test_loss", "test_rmse", "test_mse",
                "test_mae", "test_r2", "client_id", "algorithm", "model_type"
            ])
        
        # CV metrics
        with open(self.cv_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "round", "fold", "train_loss", "val_loss",
                "rmse", "mse", "mae", "r2", "client_id", "model_type"
            ])
    
    def log_training_metrics(self, round_num: int, metrics: Dict, hyperparams):
        """Log training metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            timestamp, round_num,
            metrics.get("train_loss", 0),
            metrics.get("val_loss", 0),
            metrics.get("avg_epoch_loss", 0),
            metrics.get("train_rmse", 0),
            metrics.get("train_mse", 0),
            metrics.get("train_mae", 0),
            metrics.get("train_r2", 0),
            metrics.get("val_rmse", 0),
            metrics.get("val_mse", 0),
            metrics.get("val_mae", 0),
            metrics.get("val_r2", 0),
            metrics.get("samples", 0),
            metrics.get("algorithm", "unknown"),
            self.client_id,
            hyperparams.model_type,
            hyperparams.learning_rate,
            hyperparams.batch_size
        ]
        
        with open(self.training_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.training_history.append(metrics)
    
    def log_test_metrics(self, round_num: int, metrics: Dict, model_type: str):
        """Log test metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            timestamp, round_num,
            metrics.get("test_loss", 0),
            metrics.get("test_rmse", 0),
            metrics.get("test_mse", 0),
            metrics.get("test_mae", 0),
            metrics.get("test_r2", 0),
            self.client_id,
            metrics.get("algorithm", "unknown"),
            model_type
        ]
        
        with open(self.test_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.test_history.append(metrics)
    
    def log_cv_metrics(self, round_num: int, fold: int, metrics: Dict, model_type: str):
        """Log cross-validation metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = [
            timestamp, round_num, fold,
            metrics.get("train_loss", 0),
            metrics.get("val_loss", 0),
            metrics.get("rmse", 0),
            metrics.get("mse", 0),
            metrics.get("mae", 0),
            metrics.get("r2", 0),
            self.client_id,
            model_type
        ]
        
        with open(self.cv_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        self.cv_history.append(metrics)
    
    def generate_final_summary(self, total_rounds: int, algorithm: str, hyperparams) -> Dict:
        """Generate final summary statistics"""
        if not self.training_history or not self.test_history:
            return {}
        
        # Get final metrics
        final_train = self.training_history[-1] if self.training_history else {}
        final_test = self.test_history[-1] if self.test_history else {}
        
        # Calculate averages
        avg_train_loss = np.mean([m.get("train_loss", 0) for m in self.training_history])
        avg_test_rmse = np.mean([m.get("test_rmse", 0) for m in self.test_history])
        
        summary = {
            "client_id": self.client_id,
            "algorithm": algorithm,
            "model_type": hyperparams.model_type,
            "total_rounds": total_rounds,
            "final_train_loss": final_train.get("train_loss", 0),
            "final_train_rmse": final_train.get("train_rmse", 0),
            "final_train_r2": final_train.get("train_r2", 0),
            "final_test_loss": final_test.get("test_loss", 0),
            "final_test_rmse": final_test.get("test_rmse", 0),
            "final_test_r2": final_test.get("test_r2", 0),
            "avg_train_loss": avg_train_loss,
            "avg_test_rmse": avg_test_rmse
        }
        
        return summary


# ============================================================================
# DATA LOADER
# ============================================================================


class NASADataLoader:
    def __init__(self, data_path: str, hyperparams):
        """
        Args:
            data_path: Path to client directory containing train/test files
                      Example: /path/to/pre_split_data/25_clients/alpha_0.005/client_0/
        """
        self.data_path = data_path
        self.hyperparams = hyperparams
        self.scaler = None
        self.reducer = None
        
        # Extract hyperparameters
        self.rul_mode = getattr(hyperparams, 'rul_mode', 'linear')
        self.rul_power = getattr(hyperparams, 'rul_power', 1)
        self.reduction_type = getattr(hyperparams, 'reduction_type', 'none')
        self.n_components = getattr(hyperparams, 'n_components', 10)
        
        print(f"\n📊 NASADataLoader initialized:")
        print(f"   Data path: {data_path}")
        print(f"   RUL mode: {self.rul_mode}")
        print(f"   RUL power: {self.rul_power}")
        print(f"   Reduction: {self.reduction_type}")





    def apply_dimensionality_reduction(self, X_train, X_test):
        if self.reduction_type == "none":
            return X_train, X_test
        if self.reduction_type == "pca":
            self.reducer = PCA(n_components=self.n_components, random_state=42)
        elif self.reduction_type == "kpca":
            kernel = getattr(self.hyperparams, "kernel", "rbf")
            self.reducer = KernelPCA(
                n_components=self.n_components,
                kernel=kernel,
                fit_inverse_transform=False,
                random_state=42,
            )
        else:
            raise ValueError(f"Unknown reduction type: {self.reduction_type}")
        X_train_red = self.reducer.fit_transform(X_train)
        X_test_red = self.reducer.transform(X_test)
        return X_train_red, X_test_red

    def create_sequences(self, data, targets):
        seq_len = getattr(self.hyperparams, "sequence_length", 10)
        if len(data) <= seq_len:
            raise ValueError(
                f"Not enough samples ({len(data)}) for sequence length {seq_len}"
            )
        X_seq, y_seq = [], []
        for i in range(len(data) - seq_len + 1):
            X_seq.append(data[i : i + seq_len])
            y_seq.append(targets[i + seq_len - 1])
        return np.array(X_seq), np.array(y_seq)

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load pre-split train/test data from client directory"""
        try:
            # ✅ Construct file paths
            train_file = os.path.join(self.data_path, "train_data.txt")
            train_labels_file = os.path.join(self.data_path, "train_labels.txt")
            test_file = os.path.join(self.data_path, "test_data.txt")
            test_labels_file = os.path.join(self.data_path, "test_labels.txt")
            
            print(f"\n📂 Loading data files:")
            print(f"   Train data: {train_file}")
            print(f"   Train labels: {train_labels_file}")
            print(f"   Test data: {test_file}")
            print(f"   Test labels: {test_labels_file}")
            
            # Load files
            X_train = np.loadtxt(train_file, delimiter=' ')
            y_train = np.loadtxt(train_labels_file)
            X_test = np.loadtxt(test_file, delimiter=' ')
            y_test = np.loadtxt(test_labels_file)
            
            print(f"\n📊 Raw data loaded:")
            print(f"   Train: X={X_train.shape}, y={y_train.shape}")
            print(f"   Test:  X={X_test.shape}, y={y_test.shape}")
            
            # Apply RUL transformation if needed
            if self.rul_power != 1:
                print(f"\n🔧 Applying RUL transformation: y^{self.rul_power}")
                print(f"   Before - Train: min={y_train.min():.4f}, max={y_train.max():.4f}")
                y_train = np.power(y_train, self.rul_power)
                y_test = np.power(y_test, self.rul_power)
                print(f"   After  - Train: min={y_train.min():.4f}, max={y_train.max():.4f}")
            
            # Apply StandardScaler
            print(f"\n🔧 Applying StandardScaler...")
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            # Apply dimensionality reduction if specified
            if self.reduction_type != 'none':
                X_train, X_test = self.apply_dimensionality_reduction(X_train, X_test)
            
            # Create sequences for LSTM if needed
            if self.hyperparams.model_type == "lstm":
                print(f"\n🔄 Creating LSTM sequences (length={self.hyperparams.sequence_length})...")
                X_train, y_train = self.create_sequences(X_train, y_train)
                X_test, y_test = self.create_sequences(X_test, y_test)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
            
            # Ensure y has correct shape
            if len(y_train_tensor.shape) == 1:
                y_train_tensor = y_train_tensor.unsqueeze(1)
            if len(y_test_tensor.shape) == 1:
                y_test_tensor = y_test_tensor.unsqueeze(1)
            
            print(f"\n✅ Data loading complete!")
            print(f"   Train: {X_train_tensor.shape[0]} samples, {X_train_tensor.shape[-1]} features")
            print(f"   Test:  {X_test_tensor.shape[0]} samples, {X_test_tensor.shape[-1]} features")
            
            return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
            
        except Exception as e:
            print(f"\n❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            raise
    

# class NASAFlowerClient(fl.client.NumPyClient):
#     def __init__(self, client_id: str, config: Dict):
#         self.client_id = client_id
#         self.config = config
#         self.algorithm = config.get("strategy", {}).get("name", "fedavg").lower()
        
#         # Setup hyperparameters
#         self.hyperparams = Hyperparameters(config)
        
#         # Setup device
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"🖥️  Using device: {self.device}")
        
#         # ✅ FIX: Construct correct data path
#         data_path = self._get_data_path(config, client_id)
#         print(f"📂 Client data path: {data_path}")
        
#         # Load data
#         self.data_loader = NASADataLoader(data_path, self.hyperparams)
#         self.X_train, self.y_train, self.X_test, self.y_test = self.data_loader.load_data()
        
#         # ... rest of __init__ ...
    
#     def _get_data_path(self, config: Dict, client_id: str) -> str:
#         """
#         Construct data path from configuration.
        
#         Expected structure from pre_splitting.py:
#         {base_path}/{num_clients}_clients/alpha_{alpha}/client_{N}/
        
#         Example:
#         /mnt/ceph_drive/FL_IoT_Network/scale/data/nasa_cmaps/pre_split_data/25_clients/alpha_0.1/client_0/
#         """
#         # Extract configuration
#         base_path = config.get("data", {}).get("base_path", "")
#         num_clients = config.get("data", {}).get("num_clients", 25)
#         alpha = config.get("data", {}).get("alpha", 0.1)
        
#         # Extract client number from client_id
#         if client_id.startswith("client_"):
#             client_num = int(client_id.split("_")[1])
#         else:
#             try:
#                 client_num = int(client_id)
#             except ValueError:
#                 raise ValueError(f"Invalid client_id format: {client_id}. Expected 'client_N' or 'N'")
        
#         # ✅ Construct path matching pre_splitting.py structure
#         data_path = os.path.join(
#             base_path,                      # /mnt/ceph_drive/.../pre_split_data
#             f"{num_clients}_clients",       # /25_clients
#             f"alpha_{alpha}",                # /alpha_0.1
#             f"client_{client_num}"          # /client_0
#         )
        
#         # Verify path exists
#         if not os.path.exists(data_path):
#             raise FileNotFoundError(
#                 f"❌ Client data directory not found: {data_path}\n"
#                 f"   Expected structure: {base_path}/{num_clients}_clients/alpha_{alpha}/client_{client_num}/\n"
#                 f"   💡 Run 'python pre_splitting.py' first to generate pre-split data!"
#             )
        
#         # Verify required files exist
#         required_files = ["train_data.txt", "train_labels.txt", "test_data.txt", "test_labels.txt"]
#         missing_files = []
        
#         for file_name in required_files:
#             file_path = os.path.join(data_path, file_name)
#             if not os.path.exists(file_path):
#                 missing_files.append(file_name)
        
#         if missing_files:
#             raise FileNotFoundError(
#                 f"❌ Missing required files in {data_path}:\n"
#                 f"   {', '.join(missing_files)}\n"
#                 f"   💡 Run 'python pre_splitting.py' to create these files!"
#             )
        
#         print(f"✅ Found client data directory with all required files")
#         return data_path




class NASAFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, config: Dict):
        self.client_id = client_id
        self.config = config
        self.algorithm = config.get("strategy", {}).get("name", "fedavg").lower()
        
        # Setup hyperparameters
        self.hyperparams = Hyperparameters(config)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        # Construct data path
        data_path = self._get_data_path(config, client_id)
        
        # Load data
        self.data_loader = NASADataLoader(data_path, self.hyperparams)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_loader.load_data()
        
        # Move data to device
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        
        # Determine input dimension
        input_dim = self.X_train.shape[-1]
        
        # Create model
        self.model = ModelFactory.create_model(
            self.hyperparams.model_type,
            input_dim,
            self.hyperparams.hidden_dims,
            self.hyperparams.dropout
        ).to(self.device)
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparams.learning_rate
        )
        self.criterion = nn.MSELoss()
        
        # Metrics logger
        log_dir = config.get("log_dir", "logs")
        self.metrics_logger = MetricsLogger(client_id, log_dir)
        
        # Training state
        self.current_round = 0
        self.total_rounds = config.get("server", {}).get("num_rounds", 100)
        self.best_val_rmse = float('inf')
        self.epochs_trained = 0
        self.total_early_stops = 0
    
    def _get_data_path(self, config: Dict, client_id: str) -> str:
        """Construct data path from configuration"""
        base_path = config.get("data", {}).get("base_path", "")
        num_clients = config.get("data", {}).get("num_clients", 25)
        alpha = config.get("data", {}).get("alpha", 0.1)
        
        if client_id.startswith("client_"):
            client_num = int(client_id.split("_")[1])
        else:
            try:
                client_num = int(client_id)
            except ValueError:
                raise ValueError(f"Invalid client_id format: {client_id}")
        
        data_path = os.path.join(
            base_path,
            f"{num_clients}_clients",
            f"alpha_{alpha}",
            f"client_{client_num}"
        )
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Client data directory not found: {data_path}")
        
        print(f"✅ Found client data directory with all required files")
        return data_path
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return model parameters"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    # def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
    #     """Train the model (simplified version)"""
    #     self.set_parameters(parameters)
    #     self.current_round = config.get("current_round", 0)
        
    #     # Simple training loop
    #     self.model.train()
    #     epochs = config.get("local_epochs", self.hyperparams.local_epochs)
    #     batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
    #     for epoch in range(epochs):
    #         for i in range(0, len(self.X_train), batch_size):
    #             batch_X = self.X_train[i:i+batch_size]
    #             batch_y = self.y_train[i:i+batch_size]
                
    #             self.optimizer.zero_grad()
    #             outputs = self.model(batch_X)
    #             loss = self.criterion(outputs, batch_y)
    #             loss.backward()
    #             self.optimizer.step()
        
    #     # Return results
    #     return self.get_parameters({}), len(self.X_train), {"loss": float(loss.item())}
    



    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model with comprehensive metrics"""
        self.set_parameters(parameters)
        self.current_round = config.get("current_round", 0)
        
        # Training parameters
        epochs = config.get("local_epochs", self.hyperparams.local_epochs)
        batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
        # Split training data into train/val
        val_size = int(0.2 * len(self.X_train))
        indices = torch.randperm(len(self.X_train))
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train_split = self.X_train[train_indices]
        y_train_split = self.y_train[train_indices]
        X_val_split = self.X_train[val_indices]
        y_val_split = self.y_train[val_indices]
        
        # Training loop
        self.model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train_split), batch_size):
                batch_X = X_train_split[i:i+batch_size]
                batch_y = y_train_split[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_losses.append(avg_epoch_loss)
        
        # ✅ Calculate comprehensive training metrics
        self.model.eval()
        with torch.no_grad():
            # Training set metrics
            train_pred = self.model(X_train_split)
            train_loss = self.criterion(train_pred, y_train_split).item()
            
            train_pred_np = train_pred.cpu().numpy().flatten()
            train_true_np = y_train_split.cpu().numpy().flatten()
            
            train_mse = mean_squared_error(train_true_np, train_pred_np)
            train_rmse = np.sqrt(train_mse)
            train_mae = mean_absolute_error(train_true_np, train_pred_np)
            train_r2 = r2_score(train_true_np, train_pred_np)
            
            # Validation set metrics
            val_pred = self.model(X_val_split)
            val_loss = self.criterion(val_pred, y_val_split).item()
            
            val_pred_np = val_pred.cpu().numpy().flatten()
            val_true_np = y_val_split.cpu().numpy().flatten()
            
            val_mse = mean_squared_error(val_true_np, val_pred_np)
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(val_true_np, val_pred_np)
            val_r2 = r2_score(val_true_np, val_pred_np)
        
        # ✅ Return comprehensive metrics
        metrics = {
            "client_id": self.client_id,
            "algorithm": self.algorithm,
            "loss": train_loss,
            "mae": train_mae,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "train_r2": train_r2,
            "val_r2": val_r2,
            "avg_epoch_loss": np.mean(epoch_losses) if epoch_losses else 0.0,
            "samples": len(X_train_split)
        }
        
        # Log metrics
        self.metrics_logger.log_training_metrics(
            self.current_round, 
            metrics, 
            self.hyperparams
        )
        
        print(f"\n📊 Round {self.current_round} Training Metrics:")
        print(f"   Train Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        print(f"   Val   Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
        
        return self.get_parameters({}), len(self.X_train), metrics



    # def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
    #     """Evaluate the model"""
    #     self.set_parameters(parameters)
    #     self.model.eval()
        
    #     with torch.no_grad():
    #         outputs = self.model(self.X_test)
    #         loss = self.criterion(outputs, self.y_test)
        
    #     return float(loss.item()), len(self.X_test), {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model with comprehensive metrics"""
        self.set_parameters(parameters)
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(self.X_test)
            loss = self.criterion(predictions, self.y_test).item()
            
            # ✅ Calculate comprehensive test metrics
            pred_np = predictions.cpu().numpy().flatten()
            true_np = self.y_test.cpu().numpy().flatten()
            
            mse = mean_squared_error(true_np, pred_np)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_np, pred_np)
            r2 = r2_score(true_np, pred_np)
        
        # ✅ Return comprehensive metrics
        metrics = {
            "client_id": self.client_id,
            "algorithm": self.algorithm,
            "test_loss": loss,
            "test_rmse": rmse,
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2
        }
        
        # Log metrics
        self.metrics_logger.log_test_metrics(
            self.current_round,
            metrics,
            self.hyperparams.model_type
        )
        
        print(f"\n📊 Round {self.current_round} Test Metrics:")
        print(f"   Test Loss: {loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return loss, len(self.X_test), metrics


    def generate_final_report(self):
        """Generate final training report"""
        print("\n" + "="*80)
        print("🎯 FINAL TRAINING REPORT")
        print("="*80)
        print(f"Client ID: {self.client_id}")
        print(f"Algorithm: {self.algorithm.upper()}")
        print(f"Total Rounds: {self.total_rounds}")
        print(f"Training Samples: {len(self.X_train)}")
        print(f"Test Samples: {len(self.X_test)}")
        print("="*80)
        
        summary = self.metrics_logger.generate_final_summary(
            self.total_rounds, self.algorithm, self.hyperparams
        )
        
        if summary:
            print(f"\n📊 Performance Summary:")
            print(f"   Final Train Loss: {summary.get('final_train_loss', 0):.4f}")
            print(f"   Final Test Loss: {summary.get('final_test_loss', 0):.4f}")
            print(f"   Final RMSE: {summary.get('final_test_rmse', 0):.4f}")
            print(f"   Final R²: {summary.get('final_test_r2', 0):.4f}")





# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NASA C-MAPSS Flower Client")
    parser.add_argument("--client-id", type=str, required=True, help="Client ID")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Add log directory
    results_dir = os.path.dirname(os.path.dirname(args.config))
    config["log_dir"] = os.path.join(results_dir, "logs")
    
    # Extract server info
    server_address = f"{config['server']['host']}:{config['server']['port']}"
    
    print(f"\n{'='*80}")
    print(f"🚀 NASA C-MAPSS Federated Learning Client")
    print(f"{'='*80}")
    print(f"Client ID: {args.client_id}")
    print(f"Server: {server_address}")
    print(f"Algorithm: {config.get('strategy', {}).get('name', 'fedavg').upper()}")
    print(f"{'='*80}\n")
    
    # Create client
    client = NASAFlowerClient(args.client_id, config)
    
    # Start client
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        
        # Generate final report
        client.generate_final_report()
        
        print(f"\n✅ Client {args.client_id} completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Client {args.client_id} interrupted by user")
    except Exception as e:
        print(f"\n❌ Client {args.client_id} error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
    