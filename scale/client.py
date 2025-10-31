



# import flwr as fl
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# import argparse
# import json
# from sklearn.decomposition import KernelPCA, PCA
# from sklearn.preprocessing import StandardScaler
# import os
# import time
# from typing import Dict, Tuple, List, Optional
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import csv
# from datetime import datetime
# import torch.nn.functional as F  # Add this with other imports

# class LSTMModel(nn.Module):
#     """LSTM model for NASA RUL prediction with sequence data"""
#     def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
#                  dropout: float = 0.3, use_attention: bool = False):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
#         self.use_attention = use_attention
        
#         # LSTM layer
#         self.lstm = nn.LSTM(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             dropout=dropout if num_layers > 1 else 0,
#             batch_first=True,
#             bidirectional=False
#         )
        
#         # Attention mechanism (optional)
#         if use_attention:
#             self.attention = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim // 2),
#                 nn.Tanh(),
#                 nn.Linear(hidden_dim // 2, 1),
#                 nn.Softmax(dim=1)
#             )
        
#         # Output layers
#         self.fc_layers = nn.Sequential(
#             nn.Linear(hidden_dim, 32),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(32, 16),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(16, 1)
#         )
        
#     def forward(self, x):
#         # x shape: (batch_size, sequence_length, input_dim)
#         # For single time step, we'll add sequence dimension in data loader
#         if len(x.shape) == 2:
#             x = x.unsqueeze(1)  # Add sequence dimension
        
#         # LSTM forward
#         lstm_out, (hidden, cell) = self.lstm(x)
        
#         if self.use_attention:
#             # Apply attention
#             attention_weights = self.attention(lstm_out)
#             context_vector = torch.sum(attention_weights * lstm_out, dim=1)
#             output = self.fc_layers(context_vector)
#         else:
#             # Use last hidden state
#             last_hidden = hidden[-1]  # Take the last layer's hidden state
#             output = self.fc_layers(last_hidden)
        
#         return output

# class NASAModel(nn.Module):
#     """Original Dense model for NASA RUL prediction"""
#     def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
#                  dropout: float = 0.2, activation: str = "relu"):
#         super().__init__()
        
#         layers = []
#         prev_dim = input_dim
        
#         for hidden_dim in hidden_dims:
#             layers.extend([
#                 nn.Linear(prev_dim, hidden_dim),
#                 nn.ReLU() if activation == "relu" else nn.Tanh(),
#                 nn.Dropout(dropout)
#             ])
#             prev_dim = hidden_dim
        
#         layers.append(nn.Linear(prev_dim, 1))
        
#         self.network = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.network(x)

# class ModelFactory:
#     """Factory class to create different model architectures"""
#     @staticmethod
#     def create_model(model_type: str, input_dim: int, **kwargs):
#         if model_type == "lstm":
#             return LSTMModel(
#                 input_dim=input_dim,
#                 hidden_dim=kwargs.get("hidden_dim", 64),
#                 num_layers=kwargs.get("num_layers", 2),
#                 dropout=kwargs.get("dropout", 0.3),
#                 use_attention=kwargs.get("use_attention", False)
#             )
#         elif model_type == "dense":
#             return NASAModel(
#                 input_dim=input_dim,
#                 hidden_dims=kwargs.get("hidden_dims", [64, 32]),
#                 dropout=kwargs.get("dropout", 0.2),
#                 activation=kwargs.get("activation", "relu")
#             )
#         else:
#             raise ValueError(f"Unknown model type: {model_type}")

# class Hyperparameters:
#     """Hyperparameter configuration class"""
#     # def __init__(self, config: Dict):
#     #     self.model_type = config.get("model_type", "dense")
        
#     #     # Training parameters
#     #     self.learning_rate = config.get("learning_rate", 0.001)
#     #     self.batch_size = config.get("batch_size", 32)
#     #     self.local_epochs = config.get("local_epochs", 1)
#     #     self.optimizer = config.get("optimizer", "adam")
#     #     self.weight_decay = config.get("weight_decay", 0.0)
        
#     #     # Model architecture parameters
#     #     self.hidden_dims = config.get("hidden_dims", [64, 32])  # For dense
#     #     self.hidden_dim = config.get("hidden_dim", 64)  # For LSTM
#     #     self.num_layers = config.get("num_layers", 2)  # For LSTM
#     #     self.dropout = config.get("dropout", 0.2)
#     #     self.activation = config.get("activation", "relu")
#     #     self.use_attention = config.get("use_attention", False)  # For LSTM
        
#     #     # Data parameters
#     #     self.sequence_length = config.get("sequence_length", 10)  # For LSTM
#     #     self.test_size = config.get("test_size", 0.2)
#     #     self.val_size = config.get("val_size", 0.2)
        
#     #     # Dimensionality reduction parameters
#     #     self.reduction_type = config.get("reduction_type", "none")
#     #     self.n_components = config.get("n_components", 10)
#     #     self.kernel = config.get("kernel", "rbf")  # For KernelPCA
        
#     #     # Cross-validation
#     #     self.k_folds = config.get("k_folds", 5)
        

#     def __init__(self, config: Dict):
#         self.model_type = config.get("model_type", "dense")
        
#         # Training parameters
#         self.learning_rate = config.get("learning_rate", 0.001)
#         self.batch_size = config.get("batch_size", 32)
#         self.local_epochs = config.get("local_epochs", 1)
#         self.optimizer = config.get("optimizer", "adam")
#         self.weight_decay = config.get("weight_decay", 0.0)
        
#         # Early stopping parameters - NEW
#         self.early_stopping_patience = config.get("early_stopping_patience", 3)
#         self.early_stopping_min_delta = config.get("early_stopping_min_delta", 0.001)
#         self.early_stopping_enabled = config.get("early_stopping_enabled", True)
        
#         # Model architecture parameters
#         self.hidden_dims = config.get("hidden_dims", [64, 32])
#         self.hidden_dim = config.get("hidden_dim", 64)
#         self.num_layers = config.get("num_layers", 2)
#         self.dropout = config.get("dropout", 0.2)
#         self.activation = config.get("activation", "relu")
#         self.use_attention = config.get("use_attention", False)
        
#         # Data parameters
#         self.sequence_length = config.get("sequence_length", 10)
#         self.test_size = config.get("test_size", 0.2)
#         self.val_size = config.get("val_size", 0.2)
        
#         # NEW: RUL transformation parameters (from notebook)
#         self.rul_mode = config.get("rul_mode", "ratio")  # "ratio" or "absolute"
#         self.rul_power = config.get("rul_power", 8)  # Notebook uses 8
        
#         # Update reduction defaults to match notebook
#         self.reduction_type = config.get("reduction_type", "kpca")  # Notebook uses KernelPCA
#         self.n_components = config.get("n_components", 5)  # Notebook uses 5
#         self.kernel = config.get("kernel", "poly")  # Notebook uses poly kernel
        
#         # Cross-validation
#         self.k_folds = config.get("k_folds", 5)

#     # def to_dict(self) -> Dict:
#     #     """Convert hyperparameters to dictionary - FIXED VERSION"""
#     #     return {
#     #         "model_type": self.model_type,
#     #         "learning_rate": self.learning_rate,
#     #         "batch_size": self.batch_size,
#     #         "local_epochs": self.local_epochs,
#     #         "optimizer": self.optimizer,
#     #         "weight_decay": self.weight_decay,
#     #         "hidden_dims": self.hidden_dims,
#     #         "hidden_dim": self.hidden_dim,  # FIXED: was self.hyperparams.hidden_dim
#     #         "num_layers": self.num_layers,  # FIXED: was self.hyperparams.num_layers
#     #         "dropout": self.dropout,
#     #         "activation": self.activation,
#     #         "use_attention": self.use_attention,
#     #         "sequence_length": self.sequence_length,
#     #         "test_size": self.test_size,
#     #         "val_size": self.val_size,
#     #         "reduction_type": self.reduction_type,
#     #         "n_components": self.n_components,
#     #         "kernel": self.kernel,
#     #         "k_folds": self.k_folds
#     #     }

#     def to_dict(self) -> Dict:
#         """Convert hyperparameters to dictionary"""
#         return {
#             "model_type": self.model_type,
#             "learning_rate": self.learning_rate,
#             "batch_size": self.batch_size,
#             "local_epochs": self.local_epochs,
#             "optimizer": self.optimizer,
#             "weight_decay": self.weight_decay,
#             "early_stopping_patience": self.early_stopping_patience,  # NEW
#             "early_stopping_min_delta": self.early_stopping_min_delta,  # NEW
#             "early_stopping_enabled": self.early_stopping_enabled,  # NEW
#             "hidden_dims": self.hidden_dims,
#             "hidden_dim": self.hidden_dim,
#             "num_layers": self.num_layers,
#             "dropout": self.dropout,
#             "activation": self.activation,
#             "use_attention": self.use_attention,
#             "sequence_length": self.sequence_length,
#             "test_size": self.test_size,
#             "val_size": self.val_size,
#             "reduction_type": self.reduction_type,
#             "n_components": self.n_components,
#             "kernel": self.kernel,
#             "k_folds": self.k_folds
#         }

# class MetricsLogger:
#     """Handles logging metrics to CSV files"""
#     def __init__(self, client_id: str, log_dir: str = "logs"):
#         self.client_id = client_id
#         self.log_dir = log_dir
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Create logs directory if it doesn't exist
#         os.makedirs(self.log_dir, exist_ok=True)
        
#         # Define CSV file paths
#         self.cv_csv_path = os.path.join(self.log_dir, f"{client_id}_cv_metrics_{self.timestamp}.csv")
#         self.training_csv_path = os.path.join(self.log_dir, f"{client_id}_training_metrics_{self.timestamp}.csv")
#         self.test_csv_path = os.path.join(self.log_dir, f"{client_id}_test_metrics_{self.timestamp}.csv")
#         self.hyperparams_csv_path = os.path.join(self.log_dir, f"{client_id}_hyperparams_{self.timestamp}.csv")
#         self.final_summary_path = os.path.join(self.log_dir, f"{client_id}_final_summary_{self.timestamp}.csv")
        
#         # Initialize CSV files with headers
#         self._initialize_csv_files()
        
#         # Store metrics for final summary
#         self.all_training_metrics = []
#         self.all_test_metrics = []
    
#     def _initialize_csv_files(self):
#         """Initialize CSV files with headers"""
#         # CV metrics headers
#         cv_headers = [
#             "timestamp", "round", "fold", "train_loss", "val_loss", 
#             "rmse", "mse", "mae", "r2", "client_id", "model_type"
#         ]
        
#         # Training metrics headers
#         training_headers = [
#             "timestamp", "round", "train_loss", "val_loss", "avg_epoch_loss",
#             "train_rmse", "train_mse", "train_mae", "train_r2",
#             "val_rmse", "val_mse", "val_mae", "val_r2",
#             "samples", "algorithm", "client_id", "model_type", "learning_rate", "batch_size"
#         ]
        
#         # Test metrics headers
#         test_headers = [
#             "timestamp", "round", "test_loss", "test_rmse", "test_mse", "test_mae", "test_r2",
#             "val_rmse", "val_r2", "client_id", "algorithm", "model_type"
#         ]
        
#         # Hyperparameters headers
#         hyperparams_headers = [
#             "timestamp", "client_id", "model_type", "learning_rate", "batch_size",
#             "local_epochs", "hidden_dims", "hidden_dim", "num_layers", "dropout",
#             "activation", "use_attention", "sequence_length", "optimizer", "weight_decay"
#         ]
        
#         # Final summary headers
#         summary_headers = [
#             "timestamp", "client_id", "algorithm", "model_type", "total_rounds",
#             "final_train_loss", "final_val_loss", "final_test_loss",
#             "final_train_rmse", "final_val_rmse", "final_test_rmse",
#             "final_train_r2", "final_val_r2", "final_test_r2",
#             "avg_train_loss", "std_train_loss", "avg_val_loss", "std_val_loss",
#             "avg_test_loss", "std_test_loss", "avg_train_rmse", "std_train_rmse",
#             "avg_val_rmse", "std_val_rmse", "avg_test_rmse", "std_test_rmse",
#             "avg_train_r2", "std_train_r2", "avg_val_r2", "std_val_r2",
#             "avg_test_r2", "std_test_r2", "best_round", "best_test_r2",
#             "learning_rate", "batch_size", "hidden_dim", "num_layers", "dropout"
#         ]
        
#         # Write headers
#         for path, headers in [
#             (self.cv_csv_path, cv_headers),
#             (self.training_csv_path, training_headers),
#             (self.test_csv_path, test_headers),
#             (self.hyperparams_csv_path, hyperparams_headers),
#             (self.final_summary_path, summary_headers)
#         ]:
#             with open(path, 'w', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=headers)
#                 writer.writeheader()
    
#     def log_hyperparameters(self, hyperparams: Hyperparameters):
#         """Log hyperparameters to CSV"""
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         params_dict = hyperparams.to_dict()
        
#         row = {
#             "timestamp": timestamp,
#             "client_id": self.client_id,
#             **params_dict
#         }
        
#         with open(self.hyperparams_csv_path, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=row.keys())
#             writer.writerow(row)
        
#         print(f"💾 Hyperparameters saved to: {self.hyperparams_csv_path}")
    
#     def log_cv_metrics(self, round_num: int, fold_metrics: Dict, model_type: str):
#         """Log cross-validation metrics to CSV"""
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         for i in range(len(fold_metrics["fold"])):
#             row = {
#                 "timestamp": timestamp,
#                 "round": round_num,
#                 "fold": fold_metrics["fold"][i],
#                 "train_loss": fold_metrics["train_loss"][i],
#                 "val_loss": fold_metrics["val_loss"][i],
#                 "rmse": fold_metrics["rmse"][i],
#                 "mse": fold_metrics["mse"][i],
#                 "mae": fold_metrics["mae"][i],
#                 "r2": fold_metrics["r2"][i],
#                 "client_id": self.client_id,
#                 "model_type": model_type
#             }
            
#             with open(self.cv_csv_path, 'a', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=row.keys())
#                 writer.writerow(row)
        
#         print(f"💾 CV metrics saved to: {self.cv_csv_path}")
    
#     def log_training_metrics(self, round_num: int, metrics: Dict, hyperparams: Hyperparameters):
#         """Log training round metrics to CSV"""
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         row = {
#             "timestamp": timestamp,
#             "round": round_num,
#             "train_loss": metrics.get("train_loss", 0),
#             "val_loss": metrics.get("val_loss", 0),
#             "avg_epoch_loss": metrics.get("avg_epoch_loss", 0),
#             "train_rmse": metrics.get("train_rmse", 0),
#             "train_mse": metrics.get("train_mse", 0),
#             "train_mae": metrics.get("train_mae", 0),
#             "train_r2": metrics.get("train_r2", 0),
#             "val_rmse": metrics.get("val_rmse", 0),
#             "val_mse": metrics.get("val_mse", 0),
#             "val_mae": metrics.get("val_mae", 0),
#             "val_r2": metrics.get("val_r2", 0),
#             "samples": metrics.get("samples", 0),
#             "algorithm": metrics.get("algorithm", ""),
#             "client_id": self.client_id,
#             "model_type": hyperparams.model_type,
#             "learning_rate": hyperparams.learning_rate,
#             "batch_size": hyperparams.batch_size
#         }
        
#         with open(self.training_csv_path, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=row.keys())
#             writer.writerow(row)
        
#         # Store for final summary
#         self.all_training_metrics.append({
#             "round": round_num,
#             "train_loss": metrics.get("train_loss", 0),
#             "val_loss": metrics.get("val_loss", 0),
#             "train_rmse": metrics.get("train_rmse", 0),
#             "val_rmse": metrics.get("val_rmse", 0),
#             "train_r2": metrics.get("train_r2", 0),
#             "val_r2": metrics.get("val_r2", 0)
#         })
        
#         print(f"💾 Training metrics saved to: {self.training_csv_path}")
    
#     def log_test_metrics(self, round_num: int, metrics: Dict, model_type: str):
#         """Log test metrics to CSV"""
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
#         row = {
#             "timestamp": timestamp,
#             "round": round_num,
#             "test_loss": metrics.get("test_loss", 0),
#             "test_rmse": metrics.get("test_rmse", 0),
#             "test_mse": metrics.get("test_mse", 0),
#             "test_mae": metrics.get("test_mae", 0),
#             "test_r2": metrics.get("test_r2", 0),
#             "val_rmse": metrics.get("val_rmse", 0),
#             "val_r2": metrics.get("val_r2", 0),
#             "client_id": self.client_id,
#             "algorithm": metrics.get("algorithm", ""),
#             "model_type": model_type
#         }
        
#         with open(self.test_csv_path, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=row.keys())
#             writer.writerow(row)
        
#         # Store for final summary
#         self.all_test_metrics.append({
#             "round": round_num,
#             "test_loss": metrics.get("test_loss", 0),
#             "test_rmse": metrics.get("test_rmse", 0),
#             "test_r2": metrics.get("test_r2", 0)
#         })
        
#         print(f"💾 Test metrics saved to: {self.test_csv_path}")
    
#     def generate_final_summary(self, total_rounds: int, algorithm: str, hyperparams: Hyperparameters):
#         """Generate and save final summary with statistics"""
#         if not self.all_training_metrics or not self.all_test_metrics:
#             print("⚠️ No metrics available for final summary")
#             return
        
#         # Get final round metrics
#         final_train = self.all_training_metrics[-1]
#         final_test = self.all_test_metrics[-1]
        
#         # Calculate statistics across all rounds
#         train_losses = [m["train_loss"] for m in self.all_training_metrics]
#         val_losses = [m["val_loss"] for m in self.all_training_metrics]
#         test_losses = [m["test_loss"] for m in self.all_test_metrics]
        
#         train_rmses = [m["train_rmse"] for m in self.all_training_metrics]
#         val_rmses = [m["val_rmse"] for m in self.all_training_metrics]
#         test_rmses = [m["test_rmse"] for m in self.all_test_metrics]
        
#         train_r2s = [m["train_r2"] for m in self.all_training_metrics]
#         val_r2s = [m["val_r2"] for m in self.all_training_metrics]
#         test_r2s = [m["test_r2"] for m in self.all_test_metrics]
        
#         # Find best round based on test R²
#         best_round_idx = np.argmax(test_r2s)
#         best_round = self.all_test_metrics[best_round_idx]["round"]
#         best_test_r2 = test_r2s[best_round_idx]
        
#         # Create summary row
#         summary = {
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "client_id": self.client_id,
#             "algorithm": algorithm,
#             "model_type": hyperparams.model_type,
#             "total_rounds": total_rounds,
#             # Final round metrics
#             "final_train_loss": final_train["train_loss"],
#             "final_val_loss": final_train["val_loss"],
#             "final_test_loss": final_test["test_loss"],
#             "final_train_rmse": final_train["train_rmse"],
#             "final_val_rmse": final_train["val_rmse"],
#             "final_test_rmse": final_test["test_rmse"],
#             "final_train_r2": final_train["train_r2"],
#             "final_val_r2": final_train["val_r2"],
#             "final_test_r2": final_test["test_r2"],
#             # Statistics across all rounds
#             "avg_train_loss": np.mean(train_losses),
#             "std_train_loss": np.std(train_losses),
#             "avg_val_loss": np.mean(val_losses),
#             "std_val_loss": np.std(val_losses),
#             "avg_test_loss": np.mean(test_losses),
#             "std_test_loss": np.std(test_losses),
#             "avg_train_rmse": np.mean(train_rmses),
#             "std_train_rmse": np.std(train_rmses),
#             "avg_val_rmse": np.mean(val_rmses),
#             "std_val_rmse": np.std(val_rmses),
#             "avg_test_rmse": np.mean(test_rmses),
#             "std_test_rmse": np.std(test_rmses),
#             "avg_train_r2": np.mean(train_r2s),
#             "std_train_r2": np.std(train_r2s),
#             "avg_val_r2": np.mean(val_r2s),
#             "std_val_r2": np.std(val_r2s),
#             "avg_test_r2": np.mean(test_r2s),
#             "std_test_r2": np.std(test_r2s),
#             "best_round": best_round,
#             "best_test_r2": best_test_r2,
#             # Hyperparameters
#             "learning_rate": hyperparams.learning_rate,
#             "batch_size": hyperparams.batch_size,
#             "hidden_dim": hyperparams.hidden_dim,
#             "num_layers": hyperparams.num_layers,
#             "dropout": hyperparams.dropout
#         }
        
#         # Save to CSV
#         with open(self.final_summary_path, 'a', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=summary.keys())
#             writer.writerow(summary)
        
#         print(f"💾 Final summary saved to: {self.final_summary_path}")
        
#         return summary

# # class NASADataLoader:
# #     def __init__(self, data_path: str, hyperparams: Hyperparameters, random_state: int = 42):
# #         self.data_path = data_path
# #         self.hyperparams = hyperparams
# #         self.random_state = random_state
# #         self.scaler = None
# #         self.pca = None
        
# #     def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
# #         """Create sequences for LSTM training"""
# #         sequence_length = self.hyperparams.sequence_length
# #         sequences = []
# #         target_sequences = []
        
# #         for i in range(len(data) - sequence_length):
# #             sequences.append(data[i:(i + sequence_length)])
# #             target_sequences.append(targets[i + sequence_length])
        
# #         return np.array(sequences), np.array(target_sequences)
    
# #     def apply_dimensionality_reduction(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# #         """Apply PCA/KPCA dimensionality reduction"""
# #         reduction_type = getattr(self.hyperparams, 'reduction_type', 'none')
# #         n_components = getattr(self.hyperparams, 'n_components', 10)
        
# #         if reduction_type == 'none':
# #             print("🔍 No dimensionality reduction applied")
# #             return X_train, X_val, X_test
        
# #         # Standardize the data first (important for PCA)
# #         self.scaler = StandardScaler()
# #         X_train_scaled = self.scaler.fit_transform(X_train)
# #         X_val_scaled = self.scaler.transform(X_val)
# #         X_test_scaled = self.scaler.transform(X_test)
        
# #         if reduction_type == 'pca':
# #             self.pca = PCA(n_components=n_components, random_state=self.random_state)
# #             print(f"🔍 Applying PCA with {n_components} components")
# #         elif reduction_type == 'kpca':
# #             kernel = getattr(self.hyperparams, 'kernel', 'rbf')
# #             self.pca = KernelPCA(n_components=n_components, kernel=kernel, 
# #                                random_state=self.random_state)
# #             print(f"🔍 Applying KernelPCA with {n_components} components, kernel={kernel}")
# #         else:
# #             print(f"⚠️ Unknown reduction type: {reduction_type}")
# #             return X_train, X_val, X_test
        
# #         # Fit and transform
# #         X_train_reduced = self.pca.fit_transform(X_train_scaled)
# #         X_val_reduced = self.pca.transform(X_val_scaled)
# #         X_test_reduced = self.pca.transform(X_test_scaled)
        
# #         # Print variance explained for PCA
# #         if reduction_type == 'pca' and hasattr(self.pca, 'explained_variance_ratio_'):
# #             explained_variance = self.pca.explained_variance_ratio_.sum()
# #             print(f"📊 Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
# #         print(f"📊 Data shape - Before: {X_train.shape}, After: {X_train_reduced.shape}")
        
# #         return X_train_reduced, X_val_reduced, X_test_reduced
        
# #     def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
# #         """Load and preprocess client data with proper train/val/test split"""
# #         try:
# #             # Check if path exists, if not try alternative structure
# #             if not os.path.exists(self.data_path):
# #                 alt_paths = [
# #                     self.data_path,
# #                     self.data_path.replace("pre_split_data", "data"),
# #                     os.path.join(os.path.dirname(self.data_path), "train_data.txt"),
# #                     "data/train_data.txt"
# #                 ]
                
# #                 for alt_path in alt_paths:
# #                     if os.path.exists(alt_path):
# #                         self.data_path = alt_path
# #                         break
# #                 else:
# #                     raise FileNotFoundError(f"No data file found in any expected location")
            
# #             data_file = os.path.join(self.data_path, "train_data.txt") if os.path.isdir(self.data_path) else self.data_path
            
# #             # Load data - FIXED: use raw string for regex
# #             data = pd.read_csv(data_file, sep=r'\s+', header=None)
            
# #             # Features: columns 2-25 (operational settings + sensors)
# #             X = data.iloc[:, 2:26].values.astype(np.float32)
# #             y = self._calculate_rul(data).astype(np.float32)
            
# #             print(f"📊 Loaded {len(X)} total samples from {self.data_path}")
# #             print(f"🔢 Original feature dimension: {X.shape[1]}")
            
# #             # Split into train+val and test sets
# #             X_temp, X_test, y_temp, y_test = train_test_split(
# #                 X, y, test_size=self.hyperparams.test_size, random_state=self.random_state, shuffle=True
# #             )
            
# #             # Split train+val into train and validation sets
# #             val_ratio = self.hyperparams.val_size / (1 - self.hyperparams.test_size)
# #             X_train, X_val, y_train, y_val = train_test_split(
# #                 X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, shuffle=True
# #             )
            
# #             # Apply dimensionality reduction if specified in hyperparams
# #             if hasattr(self.hyperparams, 'reduction_type') and self.hyperparams.reduction_type != 'none':
# #                 X_train, X_val, X_test = self.apply_dimensionality_reduction(X_train, X_val, X_test)
            
# #             # Create sequences for LSTM if needed
# #             if self.hyperparams.model_type == "lstm":
# #                 X_train, y_train = self.create_sequences(X_train, y_train)
# #                 X_val, y_val = self.create_sequences(X_val, y_val)
# #                 X_test, y_test = self.create_sequences(X_test, y_test)
# #                 print(f"🔄 Created sequences with length {self.hyperparams.sequence_length}")
# #                 print(f"   Final dataset shape: X {X_train.shape}, y {y_train.shape}")
            
# #             # Convert to PyTorch tensors
# #             X_train_tensor = torch.tensor(X_train)
# #             y_train_tensor = torch.tensor(y_train).unsqueeze(1) if len(y_train.shape) == 1 else torch.tensor(y_train)
# #             X_val_tensor = torch.tensor(X_val)
# #             y_val_tensor = torch.tensor(y_val).unsqueeze(1) if len(y_val.shape) == 1 else torch.tensor(y_val)
# #             X_test_tensor = torch.tensor(X_test)
# #             y_test_tensor = torch.tensor(y_test).unsqueeze(1) if len(y_test.shape) == 1 else torch.tensor(y_test)
            
# #             print(f"✅ Data split completed:")
# #             print(f"   Training samples: {len(X_train)}")
# #             print(f"   Validation samples: {len(X_val)}")
# #             print(f"   Test samples: {len(X_test)}")
# #             print(f"   Model type: {self.hyperparams.model_type}")
# #             print(f"   Final input dimension: {X_train_tensor.shape[-1]}")
            
# #             return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
            
# #         except Exception as e:
# #             print(f"❌ Error loading data from {self.data_path}: {e}")
# #             print("🔄 Using synthetic data for testing")
# #             # Return synthetic data for testing
# #             input_dim = getattr(self.hyperparams, 'n_components', 24) if hasattr(self.hyperparams, 'reduction_type') and self.hyperparams.reduction_type != 'none' else 24
            
# #             if self.hyperparams.model_type == "lstm":
# #                 seq_len = self.hyperparams.sequence_length
# #                 X_train = torch.randn(60, seq_len, input_dim)
# #                 y_train = torch.randn(60, 1)
# #                 X_val = torch.randn(20, seq_len, input_dim)
# #                 y_val = torch.randn(20, 1)
# #                 X_test = torch.randn(20, seq_len, input_dim)
# #                 y_test = torch.randn(20, 1)
# #             else:
# #                 X_train = torch.randn(60, input_dim)
# #                 y_train = torch.randn(60, 1)
# #                 X_val = torch.randn(20, input_dim)
# #                 y_val = torch.randn(20, 1)
# #                 X_test = torch.randn(20, input_dim)
# #                 y_test = torch.randn(20, 1)
# #             return X_train, y_train, X_val, y_val, X_test, y_test
    
# #     def _calculate_rul(self, data: pd.DataFrame) -> np.ndarray:
# #         """Calculate RUL labels"""
# #         unit_ids = data[0].values
# #         time_cycles = data[1].values
        
# #         rul_labels = []
# #         for unit_id in np.unique(unit_ids):
# #             unit_mask = (unit_ids == unit_id)
# #             max_cycle = time_cycles[unit_mask].max()
# #             unit_rul = max_cycle - time_cycles[unit_mask]
# #             rul_labels.extend(unit_rul)
        
# #         return np.array(rul_labels)


# # Find the NASADataLoader class and replace with this complete version:

# class NASADataLoader:
#     def __init__(self, data_path: str, hyperparams: Hyperparameters, random_state: int = 42):
#         self.data_path = data_path
#         self.hyperparams = hyperparams
#         self.random_state = random_state
#         self.scaler = None
#         self.pca = None
        
#         # NEW: Add configuration for RUL transformation
#         self.rul_power = hyperparams.to_dict().get("rul_power", 1)
#         self.rul_mode = hyperparams.to_dict().get("rul_mode", "ratio")
    
#     def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """Create sequences for LSTM training"""
#         sequence_length = self.hyperparams.sequence_length
#         sequences = []
#         target_sequences = []
        
#         for i in range(len(data) - sequence_length):
#             sequences.append(data[i:(i + sequence_length)])
#             target_sequences.append(targets[i + sequence_length])
        
#         return np.array(sequences), np.array(target_sequences)
    
#     def apply_dimensionality_reduction(self, X_train: np.ndarray, X_val: np.ndarray, 
#                                        X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Apply PCA/KPCA dimensionality reduction"""
#         reduction_type = getattr(self.hyperparams, 'reduction_type', 'none')
#         n_components = getattr(self.hyperparams, 'n_components', 10)
        
#         if reduction_type == 'none':
#             print("🔍 No dimensionality reduction applied")
#             return X_train, X_val, X_test
        
#         if reduction_type == 'pca':
#             from sklearn.decomposition import PCA
#             self.pca = PCA(n_components=n_components, random_state=self.random_state)
#             print(f"🔍 Applying PCA with {n_components} components")
#         elif reduction_type == 'kpca':
#             from sklearn.decomposition import KernelPCA
#             kernel = getattr(self.hyperparams, 'kernel', 'rbf')
#             self.pca = KernelPCA(n_components=n_components, kernel=kernel, 
#                                random_state=self.random_state)
#             print(f"🔍 Applying KernelPCA with {n_components} components, kernel={kernel}")
#         else:
#             print(f"⚠️ Unknown reduction type: {reduction_type}")
#             return X_train, X_val, X_test
        
#         # Fit and transform (data is already scaled)
#         X_train_reduced = self.pca.fit_transform(X_train)
#         X_val_reduced = self.pca.transform(X_val)
#         X_test_reduced = self.pca.transform(X_test)
        
#         # Print variance explained for PCA
#         if reduction_type == 'pca' and hasattr(self.pca, 'explained_variance_ratio_'):
#             explained_variance = self.pca.explained_variance_ratio_.sum()
#             print(f"📊 Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
#         print(f"📊 Data shape - Before: {X_train.shape}, After: {X_train_reduced.shape}")
        
#         return X_train_reduced, X_val_reduced, X_test_reduced
    
#     def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
#                                   torch.Tensor, torch.Tensor, torch.Tensor]:
#         """Load data with proper feature engineering from notebook"""
#         try:
#             # Load data file
#             if os.path.isdir(self.data_path):
#                 data_file = os.path.join(self.data_path, "train_data.txt")
#             else:
#                 data_file = self.data_path
            
#             if not os.path.exists(data_file):
#                 raise FileNotFoundError(f"Data file not found: {data_file}")
            
#             print(f"📂 Loading data from: {data_file}")
            
#             # Load raw data (26 columns: unit_id, time_cycle, 3 settings, 21 sensors)
#             data = pd.read_csv(data_file, sep=r'\s+', header=None)
            
#             print(f"📊 Raw data shape: {data.shape}")
            
#             # Extract unit IDs and time cycles
#             unit_ids = data.iloc[:, 0].values  # Column 0: unit_id
#             time_cycles = data.iloc[:, 1].values  # Column 1: time_cycle
            
#             # Extract ALL features first (settings + sensors)
#             # Columns 2-25: 3 operational settings + 21 sensors = 24 features
#             X_all = data.iloc[:, 2:26].values.astype(np.float32)
            
#             print(f"📊 All features shape: {X_all.shape} (24 features)")
            
#             # Remove constant features (based on notebook analysis)
#             # Adjust indices since we're now working with columns 2-25 (0-indexed as 0-23)
#             constant_feature_indices = [
#                 0,   # setting_3 (column 2 -> index 0)
#                 3,   # s_1 (column 5 -> index 3)
#                 7,   # s_5 (column 9 -> index 7)
#                 8,   # s_6 (column 10 -> index 8)
#                 12,  # s_10 (column 14 -> index 12)
#                 18,  # s_16 (column 20 -> index 18)
#                 20,  # s_18 (column 22 -> index 20)
#                 21   # s_19 (column 23 -> index 21)
#             ]
            
#             # Keep only non-constant features
#             all_indices = set(range(X_all.shape[1]))
#             keep_indices = sorted(all_indices - set(constant_feature_indices))
#             X = X_all[:, keep_indices]
            
#             print(f"✅ After removing constant features: {X.shape} (16 features)")
#             print(f"   Removed: setting_3, s_1, s_5, s_6, s_10, s_16, s_18, s_19")
            
#             # Calculate RUL (following notebook's approach)
#             y = self._calculate_rul_notebook_style(unit_ids, time_cycles)
            
#             print(f"\n📊 RUL STATISTICS (before transformation):")
#             print(f"   Mode: {self.rul_mode}")
#             print(f"   Min: {y.min():.4f}, Max: {y.max():.4f}")
#             print(f"   Mean: {y.mean():.4f}, Std: {y.std():.4f}")
            
#             # Apply power transformation (notebook uses y^8)
#             if self.rul_power != 1:
#                 y = np.power(y, self.rul_power)
#                 print(f"\n📊 RUL AFTER y^{self.rul_power} TRANSFORMATION:")
#                 print(f"   Min: {y.min():.4f}, Max: {y.max():.4f}")
#                 print(f"   Mean: {y.mean():.4f}, Std: {y.std():.4f}")
            
#             # Standard train/val/test split
#             X_temp, X_test, y_temp, y_test = train_test_split(
#                 X, y, test_size=self.hyperparams.test_size, 
#                 random_state=self.random_state, shuffle=True
#             )
            
#             val_ratio = self.hyperparams.val_size / (1 - self.hyperparams.test_size)
#             X_train, X_val, y_train, y_val = train_test_split(
#                 X_temp, y_temp, test_size=val_ratio, 
#                 random_state=self.random_state, shuffle=True
#             )
            
#             print(f"\n📊 DATA SPLIT:")
#             print(f"   Train: {X_train.shape[0]} samples")
#             print(f"   Val: {X_val.shape[0]} samples")
#             print(f"   Test: {X_test.shape[0]} samples")
            
#             # Apply StandardScaler (notebook uses StandardScaler)
#             print(f"\n🔧 Applying StandardScaler...")
#             self.scaler = StandardScaler()
#             X_train = self.scaler.fit_transform(X_train)
#             X_val = self.scaler.transform(X_val)
#             X_test = self.scaler.transform(X_test)
            
#             # Apply dimensionality reduction if specified
#             if hasattr(self.hyperparams, 'reduction_type') and self.hyperparams.reduction_type != 'none':
#                 X_train, X_val, X_test = self.apply_dimensionality_reduction(X_train, X_val, X_test)
            
#             # Create sequences for LSTM if needed
#             if self.hyperparams.model_type == "lstm":
#                 print(f"\n🔄 Creating LSTM sequences (length={self.hyperparams.sequence_length})...")
#                 X_train, y_train = self.create_sequences(X_train, y_train)
#                 X_val, y_val = self.create_sequences(X_val, y_val)
#                 X_test, y_test = self.create_sequences(X_test, y_test)
            
#             # Convert to PyTorch tensors
#             X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#             y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) if len(y_train.shape) == 1 else torch.tensor(y_train, dtype=torch.float32)
#             X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
#             y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1) if len(y_val.shape) == 1 else torch.tensor(y_val, dtype=torch.float32)
#             X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#             y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1) if len(y_test.shape) == 1 else torch.tensor(y_test, dtype=torch.float32)
            
#             print(f"\n✅ Data loading complete!")
#             print(f"   Final input dimension: {X_train_tensor.shape[-1]}")
            
#             return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
            
#         except Exception as e:
#             print(f"❌ Error loading data: {e}")
#             import traceback
#             traceback.print_exc()
#             raise
    
#     def _calculate_rul_notebook_style(self, unit_ids: np.ndarray, time_cycles: np.ndarray) -> np.ndarray:
#         """
#         Calculate RUL following the notebook's approach:
#         RUL = (current_cycle / max_cycle_for_unit)
        
#         This creates a ratio from 0 to 1 representing progression through engine life
#         """
#         rul_labels = []
        
#         for unit_id in np.unique(unit_ids):
#             unit_mask = (unit_ids == unit_id)
#             unit_cycles = time_cycles[unit_mask]
#             max_cycle = unit_cycles.max()
            
#             if self.rul_mode == "ratio":
#                 # Notebook approach: ratio (0 to 1)
#                 unit_rul = unit_cycles / max_cycle
#             else:
#                 # Traditional: remaining cycles
#                 unit_rul = max_cycle - unit_cycles
            
#             rul_labels.extend(unit_rul)
        
#         return np.array(rul_labels, dtype=np.float32)


# # class NASADataLoader:
# #     def __init__(self, data_path: str, hyperparams: Hyperparameters, random_state: int = 42):
# #         self.data_path = data_path
# #         self.hyperparams = hyperparams
# #         self.random_state = random_state
# #         self.scaler = None
# #         self.pca = None
        
# #         # Features to drop (based on notebook analysis)
# #         self.constant_features = [
# #             2,   # setting_3 (index 2 in 0-indexed columns after unit_id, cycle)
# #             5,   # s_1
# #             9,   # s_5
# #             10,  # s_6
# #             14,  # s_10
# #             20,  # s_16
# #             22,  # s_18
# #             23   # s_19
# #         ]
        
# #         # NEW: Add configuration for RUL transformation
# #         self.rul_power = hyperparams.to_dict().get("rul_power", 1)
# #         self.rul_mode = hyperparams.to_dict().get("rul_mode", "ratio")
    
# #     def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
# #                                   torch.Tensor, torch.Tensor, torch.Tensor]:
# #         """Load data with proper feature engineering from notebook"""
# #         try:
# #             # Load data file
# #             if os.path.isdir(self.data_path):
# #                 data_file = os.path.join(self.data_path, "train_data.txt")
# #             else:
# #                 data_file = self.data_path
            
# #             if not os.path.exists(data_file):
# #                 raise FileNotFoundError(f"Data file not found: {data_file}")
            
# #             print(f"📂 Loading data from: {data_file}")
            
# #             # Load raw data (26 columns: unit_id, time_cycle, 3 settings, 21 sensors)
# #             data = pd.read_csv(data_file, sep=r'\s+', header=None)
            
# #             print(f"📊 Raw data shape: {data.shape}")
            
# #             # Extract unit IDs and time cycles
# #             unit_ids = data.iloc[:, 0].values  # Column 0: unit_id
# #             time_cycles = data.iloc[:, 1].values  # Column 1: time_cycle
            
# #             # Extract ALL features first (settings + sensors)
# #             # Columns 2-25: 3 operational settings + 21 sensors = 24 features
# #             X_all = data.iloc[:, 2:26].values.astype(np.float32)
            
# #             print(f"📊 All features shape: {X_all.shape} (24 features)")
            
# #             # Remove constant features (based on notebook analysis)
# #             # Adjust indices since we're now working with columns 2-25 (0-indexed as 0-23)
# #             constant_feature_indices = [
# #                 0,   # setting_3 (column 2 -> index 0)
# #                 3,   # s_1 (column 5 -> index 3)
# #                 7,   # s_5 (column 9 -> index 7)
# #                 8,   # s_6 (column 10 -> index 8)
# #                 12,  # s_10 (column 14 -> index 12)
# #                 18,  # s_16 (column 20 -> index 18)
# #                 20,  # s_18 (column 22 -> index 20)
# #                 21   # s_19 (column 23 -> index 21)
# #             ]
            
# #             # Keep only non-constant features
# #             all_indices = set(range(X_all.shape[1]))
# #             keep_indices = sorted(all_indices - set(constant_feature_indices))
# #             X = X_all[:, keep_indices]
            
# #             print(f"✅ After removing constant features: {X.shape} (16 features)")
# #             print(f"   Removed: setting_3, s_1, s_5, s_6, s_10, s_16, s_18, s_19")
            
# #             # Calculate RUL (following notebook's approach)
# #             y = self._calculate_rul_notebook_style(unit_ids, time_cycles)
            
# #             print(f"\n📊 RUL STATISTICS (before transformation):")
# #             print(f"   Mode: {self.rul_mode}")
# #             print(f"   Min: {y.min():.4f}, Max: {y.max():.4f}")
# #             print(f"   Mean: {y.mean():.4f}, Std: {y.std():.4f}")
            
# #             # Apply power transformation (notebook uses y^8)
# #             y_original = y.copy()
# #             if self.rul_power != 1:
# #                 y = np.power(y, self.rul_power)
# #                 print(f"\n📊 RUL AFTER y^{self.rul_power} TRANSFORMATION:")
# #                 print(f"   Min: {y.min():.4f}, Max: {y.max():.4f}")
# #                 print(f"   Mean: {y.mean():.4f}, Std: {y.std():.4f}")
            
# #             # Standard train/val/test split (notebook uses 70/30 split, then separate test)
# #             # We'll use your configured split sizes
# #             X_temp, X_test, y_temp, y_test = train_test_split(
# #                 X, y, test_size=self.hyperparams.test_size, 
# #                 random_state=self.random_state, shuffle=True
# #             )
            
# #             val_ratio = self.hyperparams.val_size / (1 - self.hyperparams.test_size)
# #             X_train, X_val, y_train, y_val = train_test_split(
# #                 X_temp, y_temp, test_size=val_ratio, 
# #                 random_state=self.random_state, shuffle=True
# #             )
            
# #             print(f"\n📊 DATA SPLIT:")
# #             print(f"   Train: {X_train.shape[0]} samples")
# #             print(f"   Val: {X_val.shape[0]} samples")
# #             print(f"   Test: {X_test.shape[0]} samples")
            
# #             # Apply StandardScaler (notebook uses StandardScaler)
# #             print(f"\n🔧 Applying StandardScaler...")
# #             self.scaler = StandardScaler()
# #             X_train = self.scaler.fit_transform(X_train)
# #             X_val = self.scaler.transform(X_val)
# #             X_test = self.scaler.transform(X_test)
            
# #             # Apply dimensionality reduction if specified
# #             if hasattr(self.hyperparams, 'reduction_type') and self.hyperparams.reduction_type != 'none':
# #                 X_train, X_val, X_test = self.apply_dimensionality_reduction(X_train, X_val, X_test)
            
# #             # Create sequences for LSTM if needed
# #             if self.hyperparams.model_type == "lstm":
# #                 print(f"\n🔄 Creating LSTM sequences (length={self.hyperparams.sequence_length})...")
# #                 X_train, y_train = self.create_sequences(X_train, y_train)
# #                 X_val, y_val = self.create_sequences(X_val, y_val)
# #                 X_test, y_test = self.create_sequences(X_test, y_test)
            
# #             # Convert to PyTorch tensors
# #             X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# #             y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) if len(y_train.shape) == 1 else torch.tensor(y_train, dtype=torch.float32)
# #             X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
# #             y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1) if len(y_val.shape) == 1 else torch.tensor(y_val, dtype=torch.float32)
# #             X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# #             y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1) if len(y_test.shape) == 1 else torch.tensor(y_test, dtype=torch.float32)
            
# #             print(f"\n✅ Data loading complete!")
            
# #             return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
            
# #         except Exception as e:
# #             print(f"❌ Error loading data: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             raise
    
# #     def _calculate_rul_notebook_style(self, unit_ids: np.ndarray, time_cycles: np.ndarray) -> np.ndarray:
# #         """
# #         Calculate RUL following the notebook's approach:
# #         RUL = (current_cycle / max_cycle_for_unit)
        
# #         This creates a ratio from 0 to 1 representing progression through engine life
# #         """
# #         rul_labels = []
        
# #         for unit_id in np.unique(unit_ids):
# #             unit_mask = (unit_ids == unit_id)
# #             unit_cycles = time_cycles[unit_mask]
# #             max_cycle = unit_cycles.max()
            
# #             if self.rul_mode == "ratio":
# #                 # Notebook approach: ratio (0 to 1)
# #                 unit_rul = unit_cycles / max_cycle
# #             else:
# #                 # Traditional: remaining cycles
# #                 unit_rul = max_cycle - unit_cycles
            
# #             rul_labels.extend(unit_rul)
        
# #         return np.array(rul_labels, dtype=np.float32)

        
# def _compute_moon_loss(self, z: torch.Tensor, z_prev: torch.Tensor, 
#                        z_global: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
#     """
#     Compute MOON contrastive loss
    
#     Args:
#         z: Current model representation
#         z_prev: Previous local model representation
#         z_global: Global model representation
#         temperature: Temperature parameter
    
#     Returns:
#         Contrastive loss
#     """
#     import torch.nn.functional as F
    
#     # Normalize representations
#     z = F.normalize(z, dim=1)
#     z_prev = F.normalize(z_prev, dim=1)
#     z_global = F.normalize(z_global, dim=1)
    
#     # Positive similarity (with global model)
#     pos_sim = torch.exp(torch.sum(z * z_global, dim=1) / temperature)
    
#     # Negative similarity (with previous local model)
#     neg_sim = torch.exp(torch.sum(z * z_prev, dim=1) / temperature)
    
#     # Contrastive loss
#     loss = -torch.log(pos_sim / (pos_sim + neg_sim))
    
#     return loss.mean()


# def _get_model_representation(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
#     """Extract intermediate representation from model for MOON"""
#     if hasattr(model, 'lstm'):  # LSTM model
#         lstm_out, (hidden, _) = model.lstm(x if len(x.shape) == 3 else x.unsqueeze(1))
#         return hidden[-1]  # Last layer hidden state
#     else:  # Dense model
#         # Get second-to-last layer output
#         for i, layer in enumerate(model.network):
#             x = layer(x)
#             if i == len(model.network) - 2:  # Before final layer
#                 return x
#         return x


# class NASAFlowerClient(fl.client.NumPyClient):
#     # def __init__(self, client_id: str, config: Dict):
#     #     self.client_id = client_id
#     #     self.config = config
#     #     self.algorithm = config.get("algorithm", "fedavg")
#     #     self.current_round = 0
#     #     self.cv_completed = False
#     #     self.total_rounds = config.get("server", {}).get("num_rounds", 10)
        
#     #     # GPU optimization
#     #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #     print(f"🔄 Using device: {self.device}")
#     #     if torch.cuda.is_available():
#     #         print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        
#     #     # Initialize hyperparameters
#     #     model_config = config.get("model", {})
#     #     self.hyperparams = Hyperparameters(model_config)
        
#     #     # Initialize metrics logger
#     #     log_dir = config.get("logging", {}).get("log_dir", "logs")
#     #     self.metrics_logger = MetricsLogger(client_id, log_dir)
        
#     #     # Log hyperparameters
#     #     self.metrics_logger.log_hyperparameters(self.hyperparams)
        
#     #     # Load data with proper splits
#     #     data_path = self._get_data_path()
        
#     #     self.data_loader = NASADataLoader(data_path, self.hyperparams)
#     #     self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.data_loader.load_data()
        
#     #     # Move data to GPU
#     #     self.X_train = self.X_train.to(self.device)
#     #     self.y_train = self.y_train.to(self.device)
#     #     self.X_val = self.X_val.to(self.device)
#     #     self.y_val = self.y_val.to(self.device)
#     #     self.X_test = self.X_test.to(self.device)
#     #     self.y_test = self.y_test.to(self.device)
        
#     #     # Create model
#     #     input_dim = self.X_train.shape[-1]
#     #     self.model_kwargs = self.hyperparams.to_dict().copy()
#     #     self.model_kwargs.pop('model_type', None)

#     #     self.model = ModelFactory.create_model(
#     #         self.hyperparams.model_type, 
#     #         input_dim, 
#     #         **self.model_kwargs
#     #     )
        
#     #     # Move model to GPU
#     #     self.model = self.model.to(self.device)
        
#     #     # Initialize optimizer
#     #     if self.hyperparams.optimizer == "adam":
#     #         self.optimizer = optim.Adam(
#     #             self.model.parameters(), 
#     #             lr=self.hyperparams.learning_rate,
#     #             weight_decay=self.hyperparams.weight_decay
#     #         )
#     #     elif self.hyperparams.optimizer == "sgd":
#     #         self.optimizer = optim.SGD(
#     #             self.model.parameters(), 
#     #             lr=self.hyperparams.learning_rate,
#     #             weight_decay=self.hyperparams.weight_decay,
#     #             momentum=0.9
#     #         )
#     #     else:
#     #         self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams.learning_rate)
            
#     #     self.criterion = nn.MSELoss()
        
#     #     print(f"✅ Client {client_id} ready:")
#     #     print(f"   Model: {self.hyperparams.model_type.upper()}")
#     #     print(f"   Training: {len(self.X_train)} samples")
#     #     print(f"   Device: {self.device}")
#     #     print(f"   Validation: {len(self.X_val)} samples") 
#     #     print(f"   Test: {len(self.X_test)} samples")
#     #     print(f"   Algorithm: {self.algorithm}")
#     #     print(f"   Total Rounds: {self.total_rounds}")
#     #     print(f"   Learning Rate: {self.hyperparams.learning_rate}")
#     #     print(f"   Batch Size: {self.hyperparams.batch_size}")
#     #     print(f"   Logging to: {log_dir}")

#     # def __init__(self, client_id: str, config: Dict):
#     #     self.client_id = client_id
#     #     self.config = config
#     #     self.algorithm = config.get("algorithm", "fedavg")
#     #     self.current_round = 0
#     #     self.cv_completed = False
#     #     self.total_rounds = config.get("server", {}).get("num_rounds", 10)
        
#     #     # Early stopping tracking - NEW
#     #     self.early_stop_counter = 0
#     #     self.best_val_rmse = float('inf')
#     #     self.epochs_trained = 0
#     #     self.total_early_stops = 0
        
#     #     # GPU optimization
#     #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     #     print(f"🔄 Using device: {self.device}")
#     #     if torch.cuda.is_available():
#     #         print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        
#     #     # Initialize hyperparameters
#     #     model_config = config.get("model", {})
#     #     self.hyperparams = Hyperparameters(model_config)
        
#     #     # Print early stopping config - NEW
#     #     if self.hyperparams.early_stopping_enabled:
#     #         print(f"🛑 Early stopping enabled:")
#     #         print(f"   Patience: {self.hyperparams.early_stopping_patience} epochs")
#     #         print(f"   Min delta: {self.hyperparams.early_stopping_min_delta}")

#     def __init__(self, client_id: str, config: Dict):
#         self.client_id = client_id
#         self.config = config
#         self.algorithm = config.get("algorithm", "fedavg")
#         self.current_round = 0
#         self.cv_completed = False
#         self.total_rounds = config.get("server", {}).get("num_rounds", 10)
        
#         # Early stopping tracking - NEW
#         self.early_stop_counter = 0
#         self.best_val_rmse = float('inf')
#         self.epochs_trained = 0
#         self.total_early_stops = 0
        
#         # GPU optimization
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"🔄 Using device: {self.device}")
#         if torch.cuda.is_available():
#             print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        
#         # Initialize hyperparameters
#         model_config = config.get("model", {})
#         self.hyperparams = Hyperparameters(model_config)
        
#         # Print early stopping config - NEW
#         if self.hyperparams.early_stopping_enabled:
#             print(f"🛑 Early stopping enabled:")
#             print(f"   Patience: {self.hyperparams.early_stopping_patience} epochs")
#             print(f"   Min delta: {self.hyperparams.early_stopping_min_delta}")
        
#         # Initialize metrics logger - RESTORED
#         log_dir = config.get("logging", {}).get("log_dir", "logs")
#         self.metrics_logger = MetricsLogger(client_id, log_dir)
        
#         # Log hyperparameters - RESTORED
#         self.metrics_logger.log_hyperparameters(self.hyperparams)
        
#         # Load data with proper splits - RESTORED
#         data_path = self._get_data_path()
        
#         self.data_loader = NASADataLoader(data_path, self.hyperparams)
#         self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.data_loader.load_data()
        
#         # Move data to GPU - RESTORED
#         self.X_train = self.X_train.to(self.device)
#         self.y_train = self.y_train.to(self.device)
#         self.X_val = self.X_val.to(self.device)
#         self.y_val = self.y_val.to(self.device)
#         self.X_test = self.X_test.to(self.device)
#         self.y_test = self.y_test.to(self.device)
        
#         # Create model - RESTORED
#         input_dim = self.X_train.shape[-1]
#         self.model_kwargs = self.hyperparams.to_dict().copy()
#         self.model_kwargs.pop('model_type', None)

#         self.model = ModelFactory.create_model(
#             self.hyperparams.model_type, 
#             input_dim, 
#             **self.model_kwargs
#         )
        
#         # Move model to GPU - RESTORED
#         self.model = self.model.to(self.device)
        
#         # Initialize optimizer - RESTORED
#         if self.hyperparams.optimizer == "adam":
#             self.optimizer = optim.Adam(
#                 self.model.parameters(), 
#                 lr=self.hyperparams.learning_rate,
#                 weight_decay=self.hyperparams.weight_decay
#             )
#         elif self.hyperparams.optimizer == "sgd":
#             self.optimizer = optim.SGD(
#                 self.model.parameters(), 
#                 lr=self.hyperparams.learning_rate,
#                 weight_decay=self.hyperparams.weight_decay,
#                 momentum=0.9
#             )
#         else:
#             self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams.learning_rate)
            
#         self.criterion = nn.MSELoss()

#         # MOON: Store previous models for contrastive learning
#         self.prev_model = None  # Previous local model (for MOON)
#         self.global_model = None  # Current global model (for MOON)

#         # FedALA: Adaptive local aggregation weights (placeholder)
#         self.ala_weights = None  # Learned aggregation weights (for FedALA)




        
#         print(f"✅ Client {client_id} ready:")
#         print(f"   Model: {self.hyperparams.model_type.upper()}")
#         print(f"   Training: {len(self.X_train)} samples")
#         print(f"   Device: {self.device}")
#         print(f"   Validation: {len(self.X_val)} samples") 
#         print(f"   Test: {len(self.X_test)} samples")
#         print(f"   Algorithm: {self.algorithm}")
#         print(f"   Total Rounds: {self.total_rounds}")
#         print(f"   Learning Rate: {self.hyperparams.learning_rate}")
#         print(f"   Batch Size: {self.hyperparams.batch_size}")
#         print(f"   Logging to: {log_dir}")


#     def get_parameters(self, config):
#         """Return model weights"""
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

#     def set_parameters(self, parameters):
#         """Set model parameters - GPU compatible"""
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
#         self.model.load_state_dict(state_dict, strict=True)

#     # def fit(self, parameters, config):
#     #     """Train the model for one round with proper validation"""
#     #     self.current_round = config.get("current_round", 0)
        
#     #     # Run CV only periodically to save computation time
#     #     run_cv = (
#     #         self.current_round == 0 or
#     #         self.current_round % 5 == 0 or
#     #         self.current_round == self.total_rounds - 1
#     #     )
        
#     #     if run_cv:
#     #         k_folds = min(3, self.hyperparams.k_folds)
#     #         print(f"\n🔍 Running {k_folds}-fold cross-validation (Round {self.current_round})")
#     #         cv_metrics = self._k_fold_cross_validation(parameters, config, k_folds)
#     #     else:
#     #         cv_metrics = {
#     #             "fold": [0],
#     #             "train_loss": [0], 
#     #             "val_loss": [0],
#     #             "rmse": [0],
#     #             "mse": [0],
#     #             "mae": [0],
#     #             "r2": [0]
#     #         }
#     #         print(f"⏩ Skipping CV for Round {self.current_round} (runs every 5 rounds)")
        
#     #     # Train on full training dataset for federated learning
#     #     self.set_parameters(parameters)
        
#     #     epochs = config.get("local_epochs", self.hyperparams.local_epochs)
#     #     batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
#     #     # Training loop
#     #     self.model.train()
#     #     train_losses = []
        
#     #     for epoch in range(epochs):
#     #         epoch_loss = 0
#     #         num_batches = 0
            
#     #         for i in range(0, len(self.X_train), batch_size):
#     #             batch_X = self.X_train[i:i+batch_size]
#     #             batch_y = self.y_train[i:i+batch_size]
                
#     #             self.optimizer.zero_grad()
#     #             outputs = self.model(batch_X)
#     #             loss = self.criterion(outputs, batch_y)
#     #             loss.backward()
#     #             self.optimizer.step()
                
#     #             epoch_loss += loss.item()
#     #             num_batches += 1
            
#     #         if num_batches > 0:
#     #             train_losses.append(epoch_loss / num_batches)
        
#     #     # Evaluate on validation set
#     #     with torch.no_grad():
#     #         self.model.eval()
#     #         train_predictions = self.model(self.X_train)
#     #         train_loss = self.criterion(train_predictions, self.y_train).item()
#     #         train_metrics = self._calculate_metrics(self.y_train, train_predictions)
            
#     #         val_predictions = self.model(self.X_val)
#     #         val_loss = self.criterion(val_predictions, self.y_val).item()
#     #         val_metrics = self._calculate_metrics(self.y_val, val_predictions)
        
#     #     # Reduced logging frequency
#     #     if run_cv or self.current_round % 3 == 0 or self.current_round == self.total_rounds - 1:
#     #         print(f"\n🎯 Round {self.current_round} Training Results:")
#     #         print(f"   Training - Loss: {train_loss:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
#     #         print(f"   Validation - Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
#     #     else:
#     #         print(f"Round {self.current_round}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_r2={val_metrics['r2']:.4f}")
        
#     #     # Prepare results
#     #     results = {
#     #         "loss": float(val_loss),
#     #         "mae": float(val_metrics["mae"]),
#     #         "train_loss": float(train_loss),
#     #         "val_loss": float(val_loss),
#     #         "train_rmse": float(train_metrics["rmse"]),
#     #         "val_rmse": float(val_metrics["rmse"]),
#     #         "train_r2": float(train_metrics["r2"]),
#     #         "val_r2": float(val_metrics["r2"]),
#     #         "client_id": self.client_id,
#     #         "samples": len(self.X_train),
#     #         "algorithm": self.algorithm,
#     #         "avg_epoch_loss": float(np.mean(train_losses) if train_losses else train_loss),
#     #     }
        
#     #     # Reduced CSV logging frequency
#     #     if run_cv or self.current_round % 2 == 0 or self.current_round == self.total_rounds - 1:
#     #         self.metrics_logger.log_training_metrics(self.current_round, results, self.hyperparams)
        
#     #     return self.get_parameters({}), len(self.X_train), results


#     # def fit(self, parameters, config):
#     #     """Train the model with local early stopping"""
#     #     self.current_round = config.get("current_round", 0)
        
#     #     # Run CV only periodically
#     #     run_cv = (
#     #         self.current_round == 0 or
#     #         self.current_round % 5 == 0 or
#     #         self.current_round == self.total_rounds - 1
#     #     )
        
#     #     if run_cv:
#     #         k_folds = min(3, self.hyperparams.k_folds)
#     #         print(f"\n🔍 Running {k_folds}-fold cross-validation (Round {self.current_round})")
#     #         cv_metrics = self._k_fold_cross_validation(parameters, config, k_folds)
#     #     else:
#     #         cv_metrics = {
#     #             "fold": [0], "train_loss": [0], "val_loss": [0],
#     #             "rmse": [0], "mse": [0], "mae": [0], "r2": [0]
#     #         }
#     #         print(f"⏩ Skipping CV for Round {self.current_round}")
        
#     #     # Train with early stopping
#     #     self.set_parameters(parameters)
        
#     #     epochs = config.get("local_epochs", self.hyperparams.local_epochs)
#     #     batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
#     #     # Reset early stopping for this round
#     #     epochs_without_improvement = 0
#     #     best_val_rmse_this_round = float('inf')
#     #     best_model_state = None
        
#     #     self.model.train()
#     #     train_losses = []
#     #     val_rmses_per_epoch = []
        
#     #     print(f"\n🔄 Starting local training (Round {self.current_round})...")
        
#     #     for epoch in range(epochs):
#     #         epoch_loss = 0
#     #         num_batches = 0
            
#     #         # Training
#     #         for i in range(0, len(self.X_train), batch_size):
#     #             batch_X = self.X_train[i:i+batch_size]
#     #             batch_y = self.y_train[i:i+batch_size]
                
#     #             self.optimizer.zero_grad()
#     #             outputs = self.model(batch_X)
#     #             loss = self.criterion(outputs, batch_y)
#     #             loss.backward()
#     #             self.optimizer.step()
                
#     #             epoch_loss += loss.item()
#     #             num_batches += 1
            
#     #         avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
#     #         train_losses.append(avg_train_loss)
            
#     #         # Validation after each epoch for early stopping
#     #         self.model.eval()
#     #         with torch.no_grad():
#     #             val_predictions = self.model(self.X_val)
#     #             val_loss = self.criterion(val_predictions, self.y_val).item()
#     #             val_metrics = self._calculate_metrics(self.y_val, val_predictions)
#     #             val_rmse = val_metrics["rmse"]
#     #             val_rmses_per_epoch.append(val_rmse)
            
#     #         self.model.train()
            
#     #         # Early stopping check
#     #         if self.hyperparams.early_stopping_enabled:
#     #             improvement = best_val_rmse_this_round - val_rmse
                
#     #             if improvement > self.hyperparams.early_stopping_min_delta:
#     #                 best_val_rmse_this_round = val_rmse
#     #                 epochs_without_improvement = 0
#     #                 # Save best model state
#     #                 best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
#     #                 print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f} ⭐ (improved by {improvement:.4f})")
#     #             else:
#     #                 epochs_without_improvement += 1
#     #                 print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f} (no improvement: {epochs_without_improvement}/{self.hyperparams.early_stopping_patience})")
                
#     #             # Early stop if no improvement
#     #             if epochs_without_improvement >= self.hyperparams.early_stopping_patience:
#     #                 print(f"   🛑 Early stopping triggered after {epoch+1} epochs (no improvement for {self.hyperparams.early_stopping_patience} epochs)")
#     #                 self.total_early_stops += 1
#     #                 self.epochs_trained = epoch + 1
#     #                 # Restore best model
#     #                 if best_model_state is not None:
#     #                     self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
#     #                     print(f"   ↩️  Restored best model (val_rmse={best_val_rmse_this_round:.4f})")
#     #                 break
#     #         else:
#     #             print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f}")
#     #     else:
#     #         # Completed all epochs without early stopping
#     #         self.epochs_trained = epochs
#     #         if best_model_state is not None and self.hyperparams.early_stopping_enabled:
#     #             self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
#     #             print(f"   ✅ Completed {epochs} epochs, using best model (val_rmse={best_val_rmse_this_round:.4f})")
        
#     #     # Final evaluation on validation set
#     #     self.model.eval()
#     #     with torch.no_grad():
#     #         train_predictions = self.model(self.X_train)
#     #         train_loss = self.criterion(train_predictions, self.y_train).item()
#     #         train_metrics = self._calculate_metrics(self.y_train, train_predictions)
            
#     #         val_predictions = self.model(self.X_val)
#     #         val_loss = self.criterion(val_predictions, self.y_val).item()
#     #         val_metrics = self._calculate_metrics(self.y_val, val_predictions)
        
#     #     # Print summary
#     #     if run_cv or self.current_round % 3 == 0 or self.current_round == self.total_rounds - 1:
#     #         print(f"\n🎯 Round {self.current_round} Training Summary:")
#     #         print(f"   Epochs trained: {self.epochs_trained}/{epochs}")
#     #         if self.hyperparams.early_stopping_enabled:
#     #             print(f"   Total early stops so far: {self.total_early_stops}")
#     #         print(f"   Training   - Loss: {train_loss:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
#     #         print(f"   Validation - Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
#     #     else:
#     #         print(f"Round {self.current_round}: epochs={self.epochs_trained}, train_loss={train_loss:.4f}, val_rmse={val_metrics['rmse']:.4f}, val_r2={val_metrics['r2']:.4f}")
        
#     #     # Prepare results
#     #     results = {
#     #         "loss": float(val_loss),
#     #         "mae": float(val_metrics["mae"]),
#     #         "train_loss": float(train_loss),
#     #         "val_loss": float(val_loss),
#     #         "train_rmse": float(train_metrics["rmse"]),
#     #         "val_rmse": float(val_metrics["rmse"]),
#     #         "train_r2": float(train_metrics["r2"]),
#     #         "val_r2": float(val_metrics["r2"]),
#     #         "client_id": self.client_id,
#     #         "samples": len(self.X_train),
#     #         "algorithm": self.algorithm,
#     #         "avg_epoch_loss": float(np.mean(train_losses) if train_losses else train_loss),
#     #         "epochs_trained": self.epochs_trained,  # NEW
#     #         "early_stopped": self.epochs_trained < epochs if self.hyperparams.early_stopping_enabled else False  # NEW
#     #     }
        
#     #     # Log metrics
#     #     if run_cv or self.current_round % 2 == 0 or self.current_round == self.total_rounds - 1:
#     #         self.metrics_logger.log_training_metrics(self.current_round, results, self.hyperparams)
        
#     #     return self.get_parameters({}), len(self.X_train), results
        
#     def fit(self, parameters, config):
#         """Train with support for FedAvg, MOON, and FedALA"""
#         self.current_round = config.get("current_round", 0)
#         algorithm = config.get("algorithm", self.algorithm).lower()
        
#         # Run CV periodically
#         run_cv = (
#             self.current_round == 0 or
#             self.current_round % 5 == 0 or
#             self.current_round == self.total_rounds - 1
#         )
        
#         if run_cv:
#             k_folds = min(3, self.hyperparams.k_folds)
#             print(f"\n🔍 Running {k_folds}-fold CV (Round {self.current_round})")
#             cv_metrics = self._k_fold_cross_validation(parameters, config, k_folds)
#         else:
#             print(f"⏩ Skipping CV for Round {self.current_round}")
        
#         # MOON: Store global model for contrastive learning
#         if algorithm == "moon":
#             self.global_model = ModelFactory.create_model(
#                 self.hyperparams.model_type, 
#                 self.X_train.shape[-1],
#                 **{k: v for k, v in self.hyperparams.to_dict().items() if k != 'model_type'}
#             ).to(self.device)
            
#             # Load global parameters
#             params_dict = zip(self.global_model.state_dict().keys(), parameters)
#             state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
#             self.global_model.load_state_dict(state_dict, strict=True)
#             self.global_model.eval()
            
#             print(f"   🌙 MOON: Global model loaded for contrastive learning")
        
#         # Set local model parameters
#         self.set_parameters(parameters)
        
#         epochs = config.get("local_epochs", self.hyperparams.local_epochs)
#         batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
#         # MOON parameters
#         moon_temperature = config.get("temperature", 0.5) if algorithm == "moon" else None
#         moon_mu = config.get("mu", 5.0) if algorithm == "moon" else None
        
#         # Training loop with early stopping
#         epochs_without_improvement = 0
#         best_val_rmse_this_round = float('inf')
#         best_model_state = None
        
#         self.model.train()
#         train_losses = []
        
#         print(f"\n🔄 Starting local training (Round {self.current_round}, {algorithm.upper()})...")
        
#         for epoch in range(epochs):
#             epoch_loss = 0
#             num_batches = 0
            
#             for i in range(0, len(self.X_train), batch_size):
#                 batch_X = self.X_train[i:i+batch_size]
#                 batch_y = self.y_train[i:i+batch_size]
                
#                 self.optimizer.zero_grad()
                
#                 # Standard prediction loss
#                 outputs = self.model(batch_X)
#                 loss = self.criterion(outputs, batch_y)
                
#                 # MOON: Add contrastive loss
#                 if algorithm == "moon" and self.prev_model is not None:
#                     with torch.no_grad():
#                         z_global = self._get_model_representation(self.global_model, batch_X)
#                         z_prev = self._get_model_representation(self.prev_model, batch_X)
                    
#                     z_current = self._get_model_representation(self.model, batch_X)
                    
#                     contrastive_loss = self._compute_moon_loss(
#                         z_current, z_prev, z_global, moon_temperature
#                     )
                    
#                     loss = loss + moon_mu * contrastive_loss
                
#                 loss.backward()
#                 self.optimizer.step()
                
#                 epoch_loss += loss.item()
#                 num_batches += 1
            
#             avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
#             train_losses.append(avg_train_loss)
            
#             # Validation for early stopping
#             self.model.eval()
#             with torch.no_grad():
#                 val_predictions = self.model(self.X_val)
#                 val_loss = self.criterion(val_predictions, self.y_val).item()
#                 val_metrics = self._calculate_metrics(self.y_val, val_predictions)
#                 val_rmse = val_metrics["rmse"]
#             self.model.train()
            
#             # Early stopping check
#             if self.hyperparams.early_stopping_enabled:
#                 improvement = best_val_rmse_this_round - val_rmse
                
#                 if improvement > self.hyperparams.early_stopping_min_delta:
#                     best_val_rmse_this_round = val_rmse
#                     epochs_without_improvement = 0
#                     best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
#                     print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f} ⭐")
#                 else:
#                     epochs_without_improvement += 1
#                     print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f} ({epochs_without_improvement}/{self.hyperparams.early_stopping_patience})")
                
#                 if epochs_without_improvement >= self.hyperparams.early_stopping_patience:
#                     print(f"   🛑 Early stopping triggered")
#                     self.total_early_stops += 1
#                     self.epochs_trained = epoch + 1
#                     if best_model_state is not None:
#                         self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
#                     break
#             else:
#                 print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f}")
#         else:
#             self.epochs_trained = epochs
#             if best_model_state is not None and self.hyperparams.early_stopping_enabled:
#                 self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
        
#         # MOON: Store current model as previous for next round
#         if algorithm == "moon":
#             self.prev_model = ModelFactory.create_model(
#                 self.hyperparams.model_type,
#                 self.X_train.shape[-1],
#                 **{k: v for k, v in self.hyperparams.to_dict().items() if k != 'model_type'}
#             ).to(self.device)
#             self.prev_model.load_state_dict(self.model.state_dict())
#             self.prev_model.eval()
#             print(f"   📦 MOON: Current model stored as previous model for next round")
        
#         # Final evaluation
#         self.model.eval()
#         with torch.no_grad():
#             train_predictions = self.model(self.X_train)
#             train_loss = self.criterion(train_predictions, self.y_train).item()
#             train_metrics = self._calculate_metrics(self.y_train, train_predictions)
            
#             val_predictions = self.model(self.X_val)
#             val_loss = self.criterion(val_predictions, self.y_val).item()
#             val_metrics = self._calculate_metrics(self.y_val, val_predictions)
        
#         # Print summary
#         if run_cv or self.current_round % 3 == 0 or self.current_round == self.total_rounds - 1:
#             print(f"\n🎯 Round {self.current_round} Summary ({algorithm.upper()}):")
#             print(f"   Epochs: {self.epochs_trained}/{epochs}")
#             if self.hyperparams.early_stopping_enabled:
#                 print(f"   Early stops: {self.total_early_stops}")
#             print(f"   Train - Loss: {train_loss:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
#             print(f"   Val   - Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        
#         # Prepare results
#         results = {
#             "loss": float(val_loss),
#             "mae": float(val_metrics["mae"]),
#             "train_loss": float(train_loss),
#             "val_loss": float(val_loss),
#             "train_rmse": float(train_metrics["rmse"]),
#             "val_rmse": float(val_metrics["rmse"]),
#             "train_r2": float(train_metrics["r2"]),
#             "val_r2": float(val_metrics["r2"]),
#             "client_id": self.client_id,
#             "samples": len(self.X_train),
#             "algorithm": algorithm,
#             "avg_epoch_loss": float(np.mean(train_losses) if train_losses else train_loss),
#             "epochs_trained": self.epochs_trained,
#             "early_stopped": self.epochs_trained < epochs if self.hyperparams.early_stopping_enabled else False
#         }
        
#         # Log metrics
#         if run_cv or self.current_round % 2 == 0 or self.current_round == self.total_rounds - 1:
#             self.metrics_logger.log_training_metrics(self.current_round, results, self.hyperparams)
        
#         return self.get_parameters({}), len(self.X_train), results
    
#     def evaluate(self, parameters, config):
#         """Evaluate the model on test set - FIXED AND COMPLETE VERSION"""
#         # Set the received parameters
#         self.set_parameters(parameters)
        
#         # Set model to evaluation mode
#         self.model.eval()
        
#         test_loss = 0.0
#         test_predictions = []
#         test_targets = []
        
#         # Evaluate in batches to handle large datasets
#         batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
#         with torch.no_grad():
#             for i in range(0, len(self.X_test), batch_size):
#                 batch_X = self.X_test[i:i + batch_size]
#                 batch_y = self.y_test[i:i + batch_size]
                
#                 outputs = self.model(batch_X)
#                 loss = self.criterion(outputs, batch_y)
                
#                 test_loss += loss.item() * len(batch_X)
#                 test_predictions.append(outputs.cpu().numpy())
#                 test_targets.append(batch_y.cpu().numpy())
        
#         # Calculate overall metrics
#         test_loss = test_loss / len(self.X_test)
#         test_predictions = np.concatenate(test_predictions).flatten()
#         test_targets = np.concatenate(test_targets).flatten()
        
#         # Calculate regression metrics
#         mse = mean_squared_error(test_targets, test_predictions)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(test_targets, test_predictions)
#         r2 = r2_score(test_targets, test_predictions)
        
#         print(f"\n🧪 Round {self.current_round} Evaluation Results:")
#         print(f"   Test Loss: {test_loss:.4f}")
#         print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
#         # Also get validation metrics for comparison
#         with torch.no_grad():
#             val_predictions = self.model(self.X_val)
#             val_metrics = self._calculate_metrics(self.y_val, val_predictions)
        
#         # Log to test metrics file
#         test_results = {
#             "test_loss": test_loss,
#             "test_rmse": rmse,
#             "test_mse": mse,
#             "test_mae": mae,
#             "test_r2": r2,
#             "val_rmse": val_metrics["rmse"],
#             "val_r2": val_metrics["r2"],
#             "algorithm": self.algorithm
#         }
        
#         self.metrics_logger.log_test_metrics(self.current_round, test_results, self.hyperparams.model_type)
        
#         # Return metrics in Flower format
#         return float(test_loss), len(self.X_test), {
#             "rmse": float(rmse),
#             "mse": float(mse),
#             "mae": float(mae),
#             "r2": float(r2)
#         }

#     def _calculate_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
#         """Calculate RMSE, MSE, MAE, and R² metrics"""
#         y_true_np = y_true.cpu().numpy().flatten()
#         y_pred_np = y_pred.cpu().numpy().flatten()
        
#         mse = mean_squared_error(y_true_np, y_pred_np)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(y_true_np, y_pred_np)
#         r2 = r2_score(y_true_np, y_pred_np)
        
#         return {
#             "rmse": rmse,
#             "mse": mse,
#             "mae": mae,
#             "r2": r2
#         }

#     def _k_fold_cross_validation(self, parameters, config, k_folds: int = 3) -> Dict[str, List[float]]:
#         """Perform k-fold cross-validation on TRAINING data only"""
#         self.set_parameters(parameters)
        
#         epochs = min(2, config.get("local_epochs", self.hyperparams.local_epochs))
#         batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
#         # Use only training data for cross-validation
#         kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
#         fold_metrics = {
#             "fold": [],
#             "train_loss": [],
#             "val_loss": [],
#             "rmse": [],
#             "mse": [],
#             "mae": [],
#             "r2": []
#         }
        
#         if not self.cv_completed:
#             print(f"\n🔍 Starting {k_folds}-fold cross-validation on TRAINING data for client {self.client_id}")
        
#         for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
#             if not self.cv_completed:
#                 print(f"\n📊 Fold {fold + 1}/{k_folds}")
            
#             # Split training data for this fold
#             X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
#             y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
#             # Create fresh model for this fold
#             input_dim = self.X_train.shape[-1]
#             fold_model = ModelFactory.create_model(
#                 self.hyperparams.model_type, 
#                 input_dim, 
#                 **self.model_kwargs
#             )
            
#             fold_model = fold_model.to(self.device)
            
#             if self.hyperparams.optimizer == "adam":
#                 fold_optimizer = optim.Adam(
#                     fold_model.parameters(), 
#                     lr=self.hyperparams.learning_rate,
#                     weight_decay=self.hyperparams.weight_decay
#                 )
#             else:
#                 fold_optimizer = optim.Adam(fold_model.parameters(), lr=self.hyperparams.learning_rate)
            
#             # Copy initial parameters
#             fold_model.load_state_dict(self.model.state_dict())
            
#             # Training loop for this fold
#             fold_model.train()
#             train_losses = []
            
#             for epoch in range(epochs):
#                 epoch_loss = 0
#                 num_batches = 0
                
#                 cv_batch_size = min(batch_size * 2, len(X_fold_train))
                
#                 for i in range(0, len(X_fold_train), cv_batch_size):
#                     batch_X = X_fold_train[i:i+cv_batch_size]
#                     batch_y = y_fold_train[i:i+cv_batch_size]
                    
#                     fold_optimizer.zero_grad()
#                     outputs = fold_model(batch_X)
#                     loss = self.criterion(outputs, batch_y)
#                     loss.backward()
#                     fold_optimizer.step()
                    
#                     epoch_loss += loss.item()
#                     num_batches += 1
                
#                 if num_batches > 0:
#                     train_losses.append(epoch_loss / num_batches)
            
#             # Evaluate on fold validation set
#             fold_model.eval()
#             with torch.no_grad():
#                 val_predictions = fold_model(X_fold_val)
#                 val_loss = self.criterion(val_predictions, y_fold_val).item()
#                 val_metrics = self._calculate_metrics(y_fold_val, val_predictions)
            
#             # Store metrics for this fold
#             fold_metrics["fold"].append(fold + 1)
#             fold_metrics["train_loss"].append(np.mean(train_losses) if train_losses else 0)
#             fold_metrics["val_loss"].append(val_loss)
#             fold_metrics["rmse"].append(val_metrics["rmse"])
#             fold_metrics["mse"].append(val_metrics["mse"])
#             fold_metrics["mae"].append(val_metrics["mae"])
#             fold_metrics["r2"].append(val_metrics["r2"])
            
#             if not self.cv_completed:
#                 print(f"   Fold {fold + 1} Results:")
#                 print(f"     Val Loss: {val_loss:.4f}")
#                 print(f"     Val RMSE: {val_metrics['rmse']:.4f}, Val R²: {val_metrics['r2']:.4f}")
        
#         self.metrics_logger.log_cv_metrics(self.current_round, fold_metrics, self.hyperparams.model_type)
        
#         if not self.cv_completed:
#             print(f"\n📈 {k_folds}-Fold CV Summary (Training Data):")
#             for metric in ["val_loss", "rmse", "r2"]:
#                 values = fold_metrics[metric]
#                 print(f"   {metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
#             self.cv_completed = True
        
#         return fold_metrics

#     def _get_data_path(self) -> str:
#         """Construct data path from configuration"""
#         base_path = self.config["data"]["base_path"]
#         num_clients = self.config["data"]["num_clients"]
#         alpha = self.config["data"]["alpha"]
        
#         if self.client_id.startswith("client_"):
#             client_num = int(self.client_id.split("_")[1])
#         else:
#             client_num = int(self.client_id)
        
#         path = f"{base_path}/{num_clients}_clients/alpha_{alpha}/client_{client_num}"
#         print(f"🔍 Looking for data at: {path}")
#         return path

#     def generate_final_report(self):
#         """Generate and display final comprehensive report"""
#         print("\n" + "="*80)
#         print("🎯 FINAL COMPREHENSIVE REPORT")
#         print("="*80)
        
#         # Generate final summary
#         summary = self.metrics_logger.generate_final_summary(self.total_rounds, self.algorithm, self.hyperparams)
        
#         if summary:
#             print(f"\n📊 CLIENT: {self.client_id} | ALGORITHM: {self.algorithm} | MODEL: {self.hyperparams.model_type.upper()}")
#             print(f"📈 TOTAL ROUNDS: {self.total_rounds}")
            
#             # Hyperparameters
#             print(f"\n⚙️  HYPERPARAMETERS:")
#             print(f"   Learning Rate: {self.hyperparams.learning_rate}")
#             print(f"   Batch Size: {self.hyperparams.batch_size}")
#             print(f"   Local Epochs: {self.hyperparams.local_epochs}")
#             if self.hyperparams.model_type == "lstm":
#                 print(f"   Hidden Dim: {self.hyperparams.hidden_dim}")
#                 print(f"   Num Layers: {self.hyperparams.num_layers}")
#                 print(f"   Sequence Length: {self.hyperparams.sequence_length}")
#                 print(f"   Use Attention: {self.hyperparams.use_attention}")
#             else:
#                 print(f"   Hidden Dims: {self.hyperparams.hidden_dims}")
#             print(f"   Dropout: {self.hyperparams.dropout}")
#             print(f"   Optimizer: {self.hyperparams.optimizer}")

#             # Early stopping summary - NEW SECTION
#             if self.hyperparams.early_stopping_enabled:
#                 print(f"\n🛑 EARLY STOPPING SUMMARY:")
#                 print(f"   Enabled: Yes")
#                 print(f"   Patience: {self.hyperparams.early_stopping_patience} epochs")
#                 print(f"   Min delta: {self.hyperparams.early_stopping_min_delta}")
#                 print(f"   Total early stops: {self.total_early_stops}/{self.total_rounds} rounds")
#                 early_stop_rate = (self.total_early_stops / self.total_rounds) * 100
#                 print(f"   Early stop rate: {early_stop_rate:.1f}%")
#             else:
#                 print(f"\n🛑 EARLY STOPPING: Disabled")

                        
#             # Final Round Performance
#             print(f"\n🏁 FINAL ROUND PERFORMANCE:")
#             print(f"   Training   - Loss: {summary['final_train_loss']:8.2f} | RMSE: {summary['final_train_rmse']:6.2f} | R²: {summary['final_train_r2']:7.4f}")
#             print(f"   Validation - Loss: {summary['final_val_loss']:8.2f} | RMSE: {summary['final_val_rmse']:6.2f} | R²: {summary['final_val_r2']:7.4f}")
#             print(f"   Test       - Loss: {summary['final_test_loss']:8.2f} | RMSE: {summary['final_test_rmse']:6.2f} | R²: {summary['final_test_r2']:7.4f}")
            
#             # Statistics Across All Rounds
#             print(f"\n📊 STATISTICS ACROSS ALL ROUNDS (Mean ± Std):")
#             print(f"   Training Loss:   {summary['avg_train_loss']:8.2f} ± {summary['std_train_loss']:6.2f}")
#             print(f"   Validation Loss: {summary['avg_val_loss']:8.2f} ± {summary['std_val_loss']:6.2f}")
#             print(f"   Test Loss:       {summary['avg_test_loss']:8.2f} ± {summary['std_test_loss']:6.2f}")
            
#             print(f"\n   Training RMSE:   {summary['avg_train_rmse']:6.2f} ± {summary['std_train_rmse']:5.2f}")
#             print(f"   Validation RMSE: {summary['avg_val_rmse']:6.2f} ± {summary['std_val_rmse']:5.2f}")
#             print(f"   Test RMSE:       {summary['avg_test_rmse']:6.2f} ± {summary['std_test_rmse']:5.2f}")
            
#             print(f"\n   Training R²:     {summary['avg_train_r2']:7.4f} ± {summary['std_train_r2']:6.4f}")
#             print(f"   Validation R²:   {summary['avg_val_r2']:7.4f} ± {summary['std_val_r2']:6.4f}")
#             print(f"   Test R²:         {summary['avg_test_r2']:7.4f} ± {summary['std_test_r2']:6.4f}")
            
#             # Best Performance
#             print(f"\n⭐ BEST PERFORMANCE:")
#             print(f"   Best Round: {summary['best_round']} (Test R²: {summary['best_test_r2']:.4f})")
            
#             # Data Summary
#             print(f"\n📋 DATA SUMMARY:")
#             print(f"   Training samples:   {len(self.X_train)}")
#             print(f"   Validation samples: {len(self.X_val)}")
#             print(f"   Test samples:       {len(self.X_test)}")
#             print(f"   Total samples:      {len(self.X_train) + len(self.X_val) + len(self.X_test)}")
            
#         print("="*80)

# def main():
#     """Main function for client"""
#     parser = argparse.ArgumentParser(description="NASA FL Client")
#     parser.add_argument("--client-id", type=str, required=True, help="Client ID (e.g., client_0)")
#     parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
#     parser.add_argument("--server-address", type=str, help="Server address (host:port)")
#     parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
#     parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save CSV logs")
    
#     args = parser.parse_args()
    
#     try:
#         with open(args.config, 'r') as f:
#             config = json.load(f)
        
#         # Update config with command line arguments
#         if "logging" not in config:
#             config["logging"] = {}
#         config["logging"]["log_dir"] = args.log_dir
        
#         algorithm = config.get("algorithm", "fedavg")
#         server_host = config['server']['host']
#         server_port = config['server']['port']
#         server_address = args.server_address or f"{server_host}:{server_port}"
        
#         print(f"🚀 Starting NASA FL Client: {args.client_id}")
#         print(f"Algorithm: {algorithm.upper()}")
#         print(f"Server: {server_address}")
#         print(f"K-Folds: {args.k_folds}")
#         print(f"Log Directory: {args.log_dir}")
        
#         # Create client instance
#         client = NASAFlowerClient(args.client_id, config)
        
#         # Use modern client API
#         fl.client.start_client(
#             server_address=server_address,
#             client=client.to_client(),
#         )
        
#         # Generate final comprehensive report
#         client.generate_final_report()
        
#         print(f"✅ Client {args.client_id} completed | Algorithm: {algorithm.upper()}")
        
#     except Exception as e:
#         print(f"❌ Client error: {e}")
#         raise

# if __name__ == "__main__":
#     main()


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
    """Data loader for NASA C-MAPSS dataset"""
    def __init__(self, data_path: str, hyperparams: Hyperparameters, random_state: int = 42):
        self.data_path = data_path
        self.hyperparams = hyperparams
        self.random_state = random_state
        self.scaler = None
        self.pca = None
        
        # RUL configuration
        self.rul_mode = hyperparams.rul_mode
        self.rul_power = hyperparams.rul_power
        self.max_rul = hyperparams.max_rul
    
    def _calculate_rul_notebook_style(self, unit_ids: np.ndarray, time_cycles: np.ndarray) -> np.ndarray:
        """Calculate RUL following notebook implementation"""
        rul = np.zeros(len(unit_ids))
        
        for unit_id in np.unique(unit_ids):
            unit_mask = unit_ids == unit_id
            unit_cycles = time_cycles[unit_mask]
            max_cycle = unit_cycles.max()
            
            unit_rul = max_cycle - unit_cycles
            
            if self.max_rul is not None:
                unit_rul = np.minimum(unit_rul, self.max_rul)
            
            rul[unit_mask] = unit_rul
        
        return rul
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        sequence_length = self.hyperparams.sequence_length
        sequences = []
        target_sequences = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
            target_sequences.append(targets[i + sequence_length])
        
        return np.array(sequences), np.array(target_sequences)
    
    def apply_dimensionality_reduction(self, X_train: np.ndarray, 
                                       X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply PCA/KPCA with only 2 splits"""
        reduction_type = getattr(self.hyperparams, 'reduction_type', 'none')
        
        if reduction_type == 'none':
            print("🔍 No dimensionality reduction applied")
            return X_train, X_test
        
        n_components = getattr(self.hyperparams, 'n_components', 10)
        
        if reduction_type == 'pca':
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            print(f"🔍 Applying PCA with {n_components} components")
        elif reduction_type == 'kpca':
            from sklearn.decomposition import KernelPCA
            kernel = getattr(self.hyperparams, 'kernel', 'rbf')
            self.pca = KernelPCA(n_components=n_components, kernel=kernel, 
                               random_state=self.random_state)
            print(f"🔍 Applying KernelPCA with {n_components} components, kernel={kernel}")
        else:
            print(f"⚠️ Unknown reduction type: {reduction_type}")
            return X_train, X_test
        
        # Fit and transform (data is already scaled)
        X_train_reduced = self.pca.fit_transform(X_train)
        X_test_reduced = self.pca.transform(X_test)
        
        # Print variance explained for PCA
        if reduction_type == 'pca' and hasattr(self.pca, 'explained_variance_ratio_'):
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"📊 Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
        print(f"📊 Data shape - Before: {X_train.shape}, After: {X_train_reduced.shape}")
        
        return X_train_reduced, X_test_reduced
    
    # def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Load data with ONLY train/test split - use K-Fold for validation"""
    #     try:
    #         # Load data file
    #         if os.path.isdir(self.data_path):
    #             data_file = os.path.join(self.data_path, "train_data.txt")
    #         else:
    #             data_file = self.data_path
            
    #         if not os.path.exists(data_file):
    #             raise FileNotFoundError(f"Data file not found: {data_file}")
            
    #         print(f"📂 Loading data from: {data_file}")
            
    #         # Load raw data
    #         data = pd.read_csv(data_file, sep=r'\s+', header=None)
    #         print(f"📊 Raw data shape: {data.shape}")
            
    #         # Extract unit IDs and time cycles
    #         unit_ids = data.iloc[:, 0].values
    #         time_cycles = data.iloc[:, 1].values
            
    #         # Extract ALL features (settings + sensors)
    #         X_all = data.iloc[:, 2:26].values.astype(np.float32)
    #         print(f"📊 All features shape: {X_all.shape} (24 features)")
            
    #         # Remove constant features
    #         constant_feature_indices = [0, 3, 7, 8, 12, 18, 20, 21]
    #         all_indices = set(range(X_all.shape[1]))
    #         keep_indices = sorted(all_indices - set(constant_feature_indices))
    #         X = X_all[:, keep_indices]
            
    #         print(f"✅ After removing constant features: {X.shape} (16 features)")
            
    #         # Calculate RUL
    #         y = self._calculate_rul_notebook_style(unit_ids, time_cycles)
            
    #         print(f"\n📊 RUL STATISTICS (before transformation):")
    #         print(f"   Mode: {self.rul_mode}")
    #         print(f"   Min: {y.min():.4f}, Max: {y.max():.4f}")
    #         print(f"   Mean: {y.mean():.4f}, Std: {y.std():.4f}")
            
    #         # Apply power transformation
    #         if self.rul_power != 1:
    #             y = np.power(y, self.rul_power)
    #             print(f"\n📊 RUL AFTER y^{self.rul_power} TRANSFORMATION:")
    #             print(f"   Min: {y.min():.4f}, Max: {y.max():.4f}")
            
    #         # ONLY 2-WAY SPLIT (Train 80% / Test 20%)
    #         X_train, X_test, y_train, y_test = train_test_split(
    #             X, y, 
    #             test_size=self.hyperparams.test_size,
    #             random_state=self.random_state, 
    #             shuffle=True
    #         )
            
    #         print(f"\n📊 DATA SPLIT:")
    #         print(f"   Train: {X_train.shape[0]} samples ({(1-self.hyperparams.test_size)*100:.0f}%) - used for training + K-fold CV")
    #         print(f"   Test: {X_test.shape[0]} samples ({self.hyperparams.test_size*100:.0f}%) - held out for final evaluation")
            
    #         # Apply StandardScaler
    #         print(f"\n🔧 Applying StandardScaler...")
    #         self.scaler = StandardScaler()
    #         X_train = self.scaler.fit_transform(X_train)
    #         X_test = self.scaler.transform(X_test)
            
    #         # Apply dimensionality reduction if specified
    #         if hasattr(self.hyperparams, 'reduction_type') and self.hyperparams.reduction_type != 'none':
    #             X_train, X_test = self.apply_dimensionality_reduction(X_train, X_test)
            
    #         # Create sequences for LSTM if needed
    #         if self.hyperparams.model_type == "lstm":
    #             print(f"\n🔄 Creating LSTM sequences (length={self.hyperparams.sequence_length})...")
    #             X_train, y_train = self.create_sequences(X_train, y_train)
    #             X_test, y_test = self.create_sequences(X_test, y_test)
            
    #         # Convert to PyTorch tensors
    #         X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    #         y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) if len(y_train.shape) == 1 else torch.tensor(y_train, dtype=torch.float32)
    #         X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    #         y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1) if len(y_test.shape) == 1 else torch.tensor(y_test, dtype=torch.float32)
            
    #         print(f"\n✅ Data loading complete!")
    #         print(f"   Final input dimension: {X_train_tensor.shape[-1]}")
    #         print(f"   K-Fold CV will split training data into folds during training")
            
    #         return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
            
    #     except Exception as e:
    #         print(f"❌ Error loading data: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         raise


    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load pre-split train/test data"""
        try:
            # ✅ Construct correct paths to pre-split data
            if os.path.isdir(self.data_path):
                train_file = os.path.join(self.data_path, "train_data.txt")
                train_labels_file = os.path.join(self.data_path, "train_labels.txt")
                test_file = os.path.join(self.data_path, "test_data.txt")
                test_labels_file = os.path.join(self.data_path, "test_labels.txt")
            else:
                raise ValueError(f"Data path must be a directory: {self.data_path}")
            
            # Verify all files exist
            for file_path, file_name in [
                (train_file, "train_data.txt"),
                (train_labels_file, "train_labels.txt"),
                (test_file, "test_data.txt"),
                (test_labels_file, "test_labels.txt")
            ]:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(
                        f"❌ {file_name} not found: {file_path}\n"
                        f"   💡 Run 'python pre_splitting.py' to generate pre-split data!"
                    )
            
            print(f"📂 Loading data from: {self.data_path}")
            
            # Load train data
            X_train = np.loadtxt(train_file, delimiter=' ')
            y_train = np.loadtxt(train_labels_file)
            
            # Load test data
            X_test = np.loadtxt(test_file, delimiter=' ')
            y_test = np.loadtxt(test_labels_file)
            
            print(f"📊 Loaded - Train: {X_train.shape}, Test: {X_test.shape}")
            
            # Remove constant features (indices: 0, 3, 7, 8, 12, 18, 20, 21)
            constant_indices = [0, 3, 7, 8, 12, 18, 20, 21]
            all_indices = set(range(X_train.shape[1]))
            keep_indices = sorted(all_indices - set(constant_indices))
            
            X_train = X_train[:, keep_indices]
            X_test = X_test[:, keep_indices]
            
            print(f"✅ After removing constant features: Train {X_train.shape}, Test {X_test.shape}")
            
            # Apply RUL transformation
            print(f"\n📊 RUL STATISTICS (before transformation):")
            print(f"   Mode: {self.rul_mode}")
            print(f"   Train - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}")
            print(f"   Test  - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}")
            
            if self.rul_power != 1:
                y_train = np.power(y_train, self.rul_power)
                y_test = np.power(y_test, self.rul_power)
                print(f"\n📊 RUL AFTER y^{self.rul_power} TRANSFORMATION:")
                print(f"   Train - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}")
                print(f"   Test  - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}")
            
            # Apply StandardScaler
            print(f"\n🔧 Applying StandardScaler...")
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            # Apply dimensionality reduction if specified
            if hasattr(self.hyperparams, 'reduction_type') and self.hyperparams.reduction_type != 'none':
                X_train, X_test = self.apply_dimensionality_reduction(X_train, X_test)
            
            # Create sequences for LSTM if needed
            if self.hyperparams.model_type == "lstm":
                print(f"\n🔄 Creating LSTM sequences (length={self.hyperparams.sequence_length})...")
                X_train, y_train = self.create_sequences(X_train, y_train)
                X_test, y_test = self.create_sequences(X_test, y_test)
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) if len(y_train.shape) == 1 else torch.tensor(y_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1) if len(y_test.shape) == 1 else torch.tensor(y_test, dtype=torch.float32)
            
            print(f"\n✅ Data loading complete!")
            print(f"   Train: {X_train_tensor.shape[0]} samples")
            print(f"   Test: {X_test_tensor.shape[0]} samples")
            print(f"   Final input dimension: {X_train_tensor.shape[-1]}")
            print(f"   K-Fold CV will split training data into folds during training")
            
            return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            import traceback
            traceback.print_exc()
            raise

# ============================================================================
# FLOWER CLIENT
# ============================================================================

class NASAFlowerClient(fl.client.NumPyClient):
    """Flower client for NASA C-MAPSS RUL prediction"""
    
    def __init__(self, client_id: str, config: Dict):
        self.client_id = client_id
        self.config = config
        self.algorithm = config.get("strategy", {}).get("name", "fedavg").lower()
        
        # Setup hyperparameters
        self.hyperparams = Hyperparameters(config)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")
        
        # Load data
        data_path = config["data"]["base_path"]
        self.data_loader = NASADataLoader(data_path, self.hyperparams)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data_loader.load_data()
        
        # Move data to GPU
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        
        # Determine input dimension
        if self.hyperparams.model_type == "lstm":
            self.input_dim = self.X_train.shape[-1]
        else:
            self.input_dim = self.X_train.shape[-1]
        
        # Create model
        self.model = ModelFactory.create_model(
            self.hyperparams.model_type,
            self.input_dim,
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
        
        # MOON specific
        self.prev_model = None
        self.global_model = None
        
        # FedALA specific
        self.local_adaptation_layer = None
        
        print(f"✅ Client {client_id} ready:")
        print(f"   Model: {self.hyperparams.model_type.upper()}")
        print(f"   Training: {len(self.X_train)} samples (will use K-Fold for validation)")
        print(f"   Test: {len(self.X_test)} samples")
        print(f"   Device: {self.device}")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return model parameters as a list of NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from a list of NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def _calculate_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict:
        """Calculate regression metrics"""
        y_true_np = y_true.detach().cpu().numpy().flatten()
        y_pred_np = y_pred.detach().cpu().numpy().flatten()
        
        mse = mean_squared_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
    
    def _k_fold_cross_validation(self, parameters: NDArrays, config: Dict, k_folds: int = 3) -> Dict:
        """Perform k-fold cross-validation on training data"""
        print(f"\n{'='*60}")
        print(f"🔍 K-FOLD CROSS-VALIDATION (k={k_folds})")
        print(f"{'='*60}")
        
        self.set_parameters(parameters)
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        epochs = config.get("local_epochs", self.hyperparams.local_epochs)
        batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train), 1):
            print(f"\n📊 Fold {fold}/{k_folds}")
            print(f"{'='*60}")
            
            # Split data
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_fold_val = self.X_train[val_idx]
            y_fold_val = self.y_train[val_idx]
            
            print(f"   Train: {len(X_fold_train)} samples")
            print(f"   Val: {len(X_fold_val)} samples")
            
            # Reset model for this fold
            self.set_parameters(parameters)
            
            # Train for specified epochs
            self.model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                num_batches = 0
                
                for i in range(0, len(X_fold_train), batch_size):
                    batch_X = X_fold_train[i:i+batch_size]
                    batch_y = y_fold_train[i:i+batch_size]
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            # Evaluate on fold validation set
            self.model.eval()
            with torch.no_grad():
                train_pred = self.model(X_fold_train)
                val_pred = self.model(X_fold_val)
                
                train_loss = self.criterion(train_pred, y_fold_train).item()
                val_loss = self.criterion(val_pred, y_fold_val).item()
                
                val_metrics = self._calculate_metrics(y_fold_val, val_pred)
            
            print(f"\n   Fold {fold} Results:")
            print(f"     Train Loss: {train_loss:.4f}")
            print(f"     Val Loss: {val_loss:.4f}")
            print(f"     Val RMSE: {val_metrics['rmse']:.4f}, Val R²: {val_metrics['r2']:.4f}")
            
            fold_results.append({
                "fold": fold,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "rmse": val_metrics["rmse"],
                "mse": val_metrics["mse"],
                "mae": val_metrics["mae"],
                "r2": val_metrics["r2"]
            })
            
            # Log fold results
            self.metrics_logger.log_cv_metrics(
                self.current_round, fold, fold_results[-1], self.hyperparams.model_type
            )
        
        # Calculate summary statistics
        avg_metrics = {
            "val_loss": np.mean([r["val_loss"] for r in fold_results]),
            "val_loss_std": np.std([r["val_loss"] for r in fold_results]),
            "rmse": np.mean([r["rmse"] for r in fold_results]),
            "rmse_std": np.std([r["rmse"] for r in fold_results]),
            "r2": np.mean([r["r2"] for r in fold_results]),
            "r2_std": np.std([r["r2"] for r in fold_results])
        }
        
        print(f"\n{'='*60}")
        print(f"📈 {k_folds}-Fold CV Summary (Training Data):")
        print(f"{'='*60}")
        print(f"   VAL_LOSS: {avg_metrics['val_loss']:.4f} ± {avg_metrics['val_loss_std']:.4f}")
        print(f"   RMSE: {avg_metrics['rmse']:.4f} ± {avg_metrics['rmse_std']:.4f}")
        print(f"   R2: {avg_metrics['r2']:.4f} ± {avg_metrics['r2_std']:.4f}")
        print(f"{'='*60}\n")
        
        return avg_metrics
    
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train with K-Fold validation for early stopping"""
        self.current_round = config.get("current_round", 0)
        algorithm = config.get("algorithm", self.algorithm).lower()
        
        # Run CV periodically
        run_cv = (
            self.current_round == 0 or
            self.current_round % 5 == 0 or
            self.current_round == self.total_rounds - 1
        )
        
        if run_cv:
            k_folds = 3
            print(f"\n🔍 Running {k_folds}-fold cross-validation (Round {self.current_round})")
            cv_metrics = self._k_fold_cross_validation(parameters, config, k_folds)
        else:
            print(f"\n⏩ Skipping CV for Round {self.current_round}")
        
        # Start regular training
        print(f"\n🔄 Starting local training (Round {self.current_round}, {algorithm.upper()})...")
        
        self.set_parameters(parameters)
        
        epochs = config.get("local_epochs", self.hyperparams.local_epochs)
        batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
        # Create ONE fold from K-Fold for early stopping validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        train_idx, val_idx = next(kf.split(self.X_train))
        
        X_train_fold = self.X_train[train_idx]
        y_train_fold = self.y_train[train_idx]
        X_val_fold = self.X_train[val_idx]
        y_val_fold = self.y_train[val_idx]
        
        print(f"   Train: {len(X_train_fold)} samples")
        print(f"   Validation: {len(X_val_fold)} samples (from K-Fold)")
        
        # MOON: Store global model if needed
        if algorithm == "moon":
            self.global_model = ModelFactory.create_model(
                self.hyperparams.model_type,
                self.input_dim,
                self.hyperparams.hidden_dims,
                self.hyperparams.dropout
            ).to(self.device)
            self.global_model.load_state_dict(self.model.state_dict())
            self.global_model.eval()
        
        # Training loop with early stopping
        epochs_without_improvement = 0
        best_val_rmse_this_round = float('inf')
        best_model_state = None
        
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Train on fold training data
            for i in range(0, len(X_train_fold), batch_size):
                batch_X = X_train_fold[i:i+batch_size]
                batch_y = y_train_fold[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # MOON: Add contrastive loss if needed
                if algorithm == "moon" and self.prev_model is not None:
                    temperature = config.get("temperature", 0.5)
                    mu = config.get("mu", 1.0)
                    
                    with torch.no_grad():
                        prev_outputs = self.prev_model(batch_X)
                        global_outputs = self.global_model(batch_X)
                    
                    cos_sim_prev = nn.functional.cosine_similarity(outputs, prev_outputs)
                    cos_sim_global = nn.functional.cosine_similarity(outputs, global_outputs)
                    
                    contrastive_loss = -torch.log(
                        torch.exp(cos_sim_global / temperature) /
                        (torch.exp(cos_sim_global / temperature) + torch.exp(cos_sim_prev / temperature))
                    ).mean()
                    
                    loss = loss + mu * contrastive_loss
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)
            
            # Validate on fold validation data
            self.model.eval()
            with torch.no_grad():
                val_predictions = self.model(X_val_fold)
                val_metrics = self._calculate_metrics(y_val_fold, val_predictions)
                val_rmse = val_metrics["rmse"]
            self.model.train()
            
            # Early stopping check
            if self.hyperparams.early_stopping_enabled:
                improvement = best_val_rmse_this_round - val_rmse
                
                if improvement > self.hyperparams.early_stopping_min_delta:
                    best_val_rmse_this_round = val_rmse
                    epochs_without_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f} ⭐")
                else:
                    epochs_without_improvement += 1
                    print(f"   Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_rmse={val_rmse:.4f} ({epochs_without_improvement}/{self.hyperparams.early_stopping_patience})")
                
                if epochs_without_improvement >= self.hyperparams.early_stopping_patience:
                    print(f"   🛑 Early stopping triggered")
                    self.total_early_stops += 1
                    self.epochs_trained = epoch + 1
                    if best_model_state is not None:
                        self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
                    break
        else:
            self.epochs_trained = epochs
            if best_model_state is not None and self.hyperparams.early_stopping_enabled:
                self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
        
        # Update MOON previous model
        if algorithm == "moon":
            if self.prev_model is None:
                self.prev_model = ModelFactory.create_model(
                    self.hyperparams.model_type,
                    self.input_dim,
                    self.hyperparams.hidden_dims,
                    self.hyperparams.dropout
                ).to(self.device)
            self.prev_model.load_state_dict(self.model.state_dict())
            self.prev_model.eval()
        
        # Final evaluation on FULL training data
        self.model.eval()
        with torch.no_grad():
            train_predictions = self.model(self.X_train)
            train_loss = self.criterion(train_predictions, self.y_train).item()
            train_metrics = self._calculate_metrics(self.y_train, train_predictions)
            
            # Use last fold validation for reporting
            val_predictions = self.model(X_val_fold)
            val_loss = self.criterion(val_predictions, y_val_fold).item()
            val_metrics = self._calculate_metrics(y_val_fold, val_predictions)
        
        print(f"\n✅ Round {self.current_round} Training Complete:")
        print(f"   Epochs: {self.epochs_trained}/{epochs}")
        print(f"   Train - Loss: {train_loss:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"   Val   - Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        
        # Update best validation RMSE
        if val_metrics["rmse"] < self.best_val_rmse:
            improvement = self.best_val_rmse - val_metrics["rmse"]
            self.best_val_rmse = val_metrics["rmse"]
            print(f"   🎯 New best validation RMSE! Improved by {improvement:.4f}")
        
        # Prepare results
        results = {
            "loss": float(val_loss),
            "mae": float(val_metrics["mae"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_rmse": float(train_metrics["rmse"]),
            "val_rmse": float(val_metrics["rmse"]),
            "train_r2": float(train_metrics["r2"]),
            "val_r2": float(val_metrics["r2"]),
            "client_id": self.client_id,
            "samples": len(self.X_train),
            "algorithm": algorithm,
            "avg_epoch_loss": float(np.mean(train_losses) if train_losses else train_loss),
            "epochs_trained": self.epochs_trained,
            "early_stopped": self.epochs_trained < epochs if self.hyperparams.early_stopping_enabled else False
        }
        
        # Log metrics selectively (every 5 rounds or first/last)
        if self.current_round == 0 or self.current_round % 5 == 0 or self.current_round == self.total_rounds - 1:
            self.metrics_logger.log_training_metrics(self.current_round, results, self.hyperparams)
        
        return self.get_parameters({}), len(self.X_train), results
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on test set"""
        self.set_parameters(parameters)
        self.model.eval()
        
        test_loss = 0.0
        test_predictions = []
        test_targets = []
        
        batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
        with torch.no_grad():
            for i in range(0, len(self.X_test), batch_size):
                batch_X = self.X_test[i:i + batch_size]
                batch_y = self.y_test[i:i + batch_size]
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                test_loss += loss.item() * len(batch_X)
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(batch_y.cpu().numpy())
        
        test_loss = test_loss / len(self.X_test)
        test_predictions = np.concatenate(test_predictions).flatten()
        test_targets = np.concatenate(test_targets).flatten()
        
        mse = mean_squared_error(test_targets, test_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)
        
        print(f"\n🧪 Round {self.current_round} Evaluation Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        test_results = {
            "test_loss": test_loss,
            "test_rmse": rmse,
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2,
            "algorithm": self.algorithm
        }
        
        self.metrics_logger.log_test_metrics(self.current_round, test_results, self.hyperparams.model_type)
        
        return float(test_loss), len(self.X_test), {
            "rmse": float(rmse),
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2)
        }
    
    def generate_final_report(self):
        """Generate final report without validation set"""
        print("\n" + "="*80)
        print("🎯 FINAL COMPREHENSIVE REPORT")
        print("="*80)
        
        summary = self.metrics_logger.generate_final_summary(
            self.total_rounds, self.algorithm, self.hyperparams
        )
        
        if summary:
            print(f"\n📊 CLIENT: {summary['client_id']}")
            print(f"   Algorithm: {summary['algorithm'].upper()}")
            print(f"   Model Type: {summary['model_type'].upper()}")
            print(f"   Total Rounds: {summary['total_rounds']}")
            
            print(f"\n🏁 FINAL ROUND PERFORMANCE:")
            print(f"   Training - Loss: {summary['final_train_loss']:8.4f} | RMSE: {summary['final_train_rmse']:6.4f} | R²: {summary['final_train_r2']:7.4f}")
            print(f"   Test     - Loss: {summary['final_test_loss']:8.4f} | RMSE: {summary['final_test_rmse']:6.4f} | R²: {summary['final_test_r2']:7.4f}")
            
            print(f"\n📈 AVERAGE METRICS:")
            print(f"   Avg Train Loss: {summary['avg_train_loss']:.4f}")
            print(f"   Avg Test RMSE: {summary['avg_test_rmse']:.4f}")
            
            print(f"\n🛑 EARLY STOPPING:")
            print(f"   Total early stops: {self.total_early_stops}")
            print(f"   Best validation RMSE: {self.best_val_rmse:.4f}")
            
            print(f"\n📋 DATA SUMMARY:")
            print(f"   Training samples: {len(self.X_train)} (used for training + K-fold CV)")
            print(f"   Test samples: {len(self.X_test)} (held out)")
            print(f"   Total samples: {len(self.X_train) + len(self.X_test)}")
            
            print(f"\n📁 LOGS SAVED TO:")
            print(f"   Training: {self.metrics_logger.training_csv}")
            print(f"   Test: {self.metrics_logger.test_csv}")
            print(f"   CV: {self.metrics_logger.cv_csv}")
        
        print("="*80)


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
    