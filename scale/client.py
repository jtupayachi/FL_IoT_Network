



import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import json
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import StandardScaler
import os
import time
from typing import Dict, Tuple, List, Optional
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import csv
from datetime import datetime

class LSTMModel(nn.Module):
    """LSTM model for NASA RUL prediction with sequence data"""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3, use_attention: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softmax(dim=1)
            )
        
        # Output layers
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
        # x shape: (batch_size, sequence_length, input_dim)
        # For single time step, we'll add sequence dimension in data loader
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # Apply attention
            attention_weights = self.attention(lstm_out)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)
            output = self.fc_layers(context_vector)
        else:
            # Use last hidden state
            last_hidden = hidden[-1]  # Take the last layer's hidden state
            output = self.fc_layers(last_hidden)
        
        return output

class NASAModel(nn.Module):
    """Original Dense model for NASA RUL prediction"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 dropout: float = 0.2, activation: str = "relu"):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if activation == "relu" else nn.Tanh(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class ModelFactory:
    """Factory class to create different model architectures"""
    @staticmethod
    def create_model(model_type: str, input_dim: int, **kwargs):
        if model_type == "lstm":
            return LSTMModel(
                input_dim=input_dim,
                hidden_dim=kwargs.get("hidden_dim", 64),
                num_layers=kwargs.get("num_layers", 2),
                dropout=kwargs.get("dropout", 0.3),
                use_attention=kwargs.get("use_attention", False)
            )
        elif model_type == "dense":
            return NASAModel(
                input_dim=input_dim,
                hidden_dims=kwargs.get("hidden_dims", [64, 32]),
                dropout=kwargs.get("dropout", 0.2),
                activation=kwargs.get("activation", "relu")
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class Hyperparameters:
    """Hyperparameter configuration class"""
    def __init__(self, config: Dict):
        self.model_type = config.get("model_type", "dense")
        
        # Training parameters
        self.learning_rate = config.get("learning_rate", 0.001)
        self.batch_size = config.get("batch_size", 32)
        self.local_epochs = config.get("local_epochs", 1)
        self.optimizer = config.get("optimizer", "adam")
        self.weight_decay = config.get("weight_decay", 0.0)
        
        # Model architecture parameters
        self.hidden_dims = config.get("hidden_dims", [64, 32])  # For dense
        self.hidden_dim = config.get("hidden_dim", 64)  # For LSTM
        self.num_layers = config.get("num_layers", 2)  # For LSTM
        self.dropout = config.get("dropout", 0.2)
        self.activation = config.get("activation", "relu")
        self.use_attention = config.get("use_attention", False)  # For LSTM
        
        # Data parameters
        self.sequence_length = config.get("sequence_length", 10)  # For LSTM
        self.test_size = config.get("test_size", 0.2)
        self.val_size = config.get("val_size", 0.2)
        
        # Dimensionality reduction parameters
        self.reduction_type = config.get("reduction_type", "none")
        self.n_components = config.get("n_components", 10)
        self.kernel = config.get("kernel", "rbf")  # For KernelPCA
        
        # Cross-validation
        self.k_folds = config.get("k_folds", 5)
        
    def to_dict(self) -> Dict:
        """Convert hyperparameters to dictionary - FIXED VERSION"""
        return {
            "model_type": self.model_type,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "hidden_dims": self.hidden_dims,
            "hidden_dim": self.hidden_dim,  # FIXED: was self.hyperparams.hidden_dim
            "num_layers": self.num_layers,  # FIXED: was self.hyperparams.num_layers
            "dropout": self.dropout,
            "activation": self.activation,
            "use_attention": self.use_attention,
            "sequence_length": self.sequence_length,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "reduction_type": self.reduction_type,
            "n_components": self.n_components,
            "kernel": self.kernel,
            "k_folds": self.k_folds
        }

class MetricsLogger:
    """Handles logging metrics to CSV files"""
    def __init__(self, client_id: str, log_dir: str = "logs"):
        self.client_id = client_id
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Define CSV file paths
        self.cv_csv_path = os.path.join(self.log_dir, f"{client_id}_cv_metrics_{self.timestamp}.csv")
        self.training_csv_path = os.path.join(self.log_dir, f"{client_id}_training_metrics_{self.timestamp}.csv")
        self.test_csv_path = os.path.join(self.log_dir, f"{client_id}_test_metrics_{self.timestamp}.csv")
        self.hyperparams_csv_path = os.path.join(self.log_dir, f"{client_id}_hyperparams_{self.timestamp}.csv")
        self.final_summary_path = os.path.join(self.log_dir, f"{client_id}_final_summary_{self.timestamp}.csv")
        
        # Initialize CSV files with headers
        self._initialize_csv_files()
        
        # Store metrics for final summary
        self.all_training_metrics = []
        self.all_test_metrics = []
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers"""
        # CV metrics headers
        cv_headers = [
            "timestamp", "round", "fold", "train_loss", "val_loss", 
            "rmse", "mse", "mae", "r2", "client_id", "model_type"
        ]
        
        # Training metrics headers
        training_headers = [
            "timestamp", "round", "train_loss", "val_loss", "avg_epoch_loss",
            "train_rmse", "train_mse", "train_mae", "train_r2",
            "val_rmse", "val_mse", "val_mae", "val_r2",
            "samples", "algorithm", "client_id", "model_type", "learning_rate", "batch_size"
        ]
        
        # Test metrics headers
        test_headers = [
            "timestamp", "round", "test_loss", "test_rmse", "test_mse", "test_mae", "test_r2",
            "val_rmse", "val_r2", "client_id", "algorithm", "model_type"
        ]
        
        # Hyperparameters headers
        hyperparams_headers = [
            "timestamp", "client_id", "model_type", "learning_rate", "batch_size",
            "local_epochs", "hidden_dims", "hidden_dim", "num_layers", "dropout",
            "activation", "use_attention", "sequence_length", "optimizer", "weight_decay"
        ]
        
        # Final summary headers
        summary_headers = [
            "timestamp", "client_id", "algorithm", "model_type", "total_rounds",
            "final_train_loss", "final_val_loss", "final_test_loss",
            "final_train_rmse", "final_val_rmse", "final_test_rmse",
            "final_train_r2", "final_val_r2", "final_test_r2",
            "avg_train_loss", "std_train_loss", "avg_val_loss", "std_val_loss",
            "avg_test_loss", "std_test_loss", "avg_train_rmse", "std_train_rmse",
            "avg_val_rmse", "std_val_rmse", "avg_test_rmse", "std_test_rmse",
            "avg_train_r2", "std_train_r2", "avg_val_r2", "std_val_r2",
            "avg_test_r2", "std_test_r2", "best_round", "best_test_r2",
            "learning_rate", "batch_size", "hidden_dim", "num_layers", "dropout"
        ]
        
        # Write headers
        for path, headers in [
            (self.cv_csv_path, cv_headers),
            (self.training_csv_path, training_headers),
            (self.test_csv_path, test_headers),
            (self.hyperparams_csv_path, hyperparams_headers),
            (self.final_summary_path, summary_headers)
        ]:
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
    
    def log_hyperparameters(self, hyperparams: Hyperparameters):
        """Log hyperparameters to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        params_dict = hyperparams.to_dict()
        
        row = {
            "timestamp": timestamp,
            "client_id": self.client_id,
            **params_dict
        }
        
        with open(self.hyperparams_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        
        print(f"💾 Hyperparameters saved to: {self.hyperparams_csv_path}")
    
    def log_cv_metrics(self, round_num: int, fold_metrics: Dict, model_type: str):
        """Log cross-validation metrics to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for i in range(len(fold_metrics["fold"])):
            row = {
                "timestamp": timestamp,
                "round": round_num,
                "fold": fold_metrics["fold"][i],
                "train_loss": fold_metrics["train_loss"][i],
                "val_loss": fold_metrics["val_loss"][i],
                "rmse": fold_metrics["rmse"][i],
                "mse": fold_metrics["mse"][i],
                "mae": fold_metrics["mae"][i],
                "r2": fold_metrics["r2"][i],
                "client_id": self.client_id,
                "model_type": model_type
            }
            
            with open(self.cv_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)
        
        print(f"💾 CV metrics saved to: {self.cv_csv_path}")
    
    def log_training_metrics(self, round_num: int, metrics: Dict, hyperparams: Hyperparameters):
        """Log training round metrics to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = {
            "timestamp": timestamp,
            "round": round_num,
            "train_loss": metrics.get("train_loss", 0),
            "val_loss": metrics.get("val_loss", 0),
            "avg_epoch_loss": metrics.get("avg_epoch_loss", 0),
            "train_rmse": metrics.get("train_rmse", 0),
            "train_mse": metrics.get("train_mse", 0),
            "train_mae": metrics.get("train_mae", 0),
            "train_r2": metrics.get("train_r2", 0),
            "val_rmse": metrics.get("val_rmse", 0),
            "val_mse": metrics.get("val_mse", 0),
            "val_mae": metrics.get("val_mae", 0),
            "val_r2": metrics.get("val_r2", 0),
            "samples": metrics.get("samples", 0),
            "algorithm": metrics.get("algorithm", ""),
            "client_id": self.client_id,
            "model_type": hyperparams.model_type,
            "learning_rate": hyperparams.learning_rate,
            "batch_size": hyperparams.batch_size
        }
        
        with open(self.training_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        
        # Store for final summary
        self.all_training_metrics.append({
            "round": round_num,
            "train_loss": metrics.get("train_loss", 0),
            "val_loss": metrics.get("val_loss", 0),
            "train_rmse": metrics.get("train_rmse", 0),
            "val_rmse": metrics.get("val_rmse", 0),
            "train_r2": metrics.get("train_r2", 0),
            "val_r2": metrics.get("val_r2", 0)
        })
        
        print(f"💾 Training metrics saved to: {self.training_csv_path}")
    
    def log_test_metrics(self, round_num: int, metrics: Dict, model_type: str):
        """Log test metrics to CSV"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = {
            "timestamp": timestamp,
            "round": round_num,
            "test_loss": metrics.get("test_loss", 0),
            "test_rmse": metrics.get("test_rmse", 0),
            "test_mse": metrics.get("test_mse", 0),
            "test_mae": metrics.get("test_mae", 0),
            "test_r2": metrics.get("test_r2", 0),
            "val_rmse": metrics.get("val_rmse", 0),
            "val_r2": metrics.get("val_r2", 0),
            "client_id": self.client_id,
            "algorithm": metrics.get("algorithm", ""),
            "model_type": model_type
        }
        
        with open(self.test_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)
        
        # Store for final summary
        self.all_test_metrics.append({
            "round": round_num,
            "test_loss": metrics.get("test_loss", 0),
            "test_rmse": metrics.get("test_rmse", 0),
            "test_r2": metrics.get("test_r2", 0)
        })
        
        print(f"💾 Test metrics saved to: {self.test_csv_path}")
    
    def generate_final_summary(self, total_rounds: int, algorithm: str, hyperparams: Hyperparameters):
        """Generate and save final summary with statistics"""
        if not self.all_training_metrics or not self.all_test_metrics:
            print("⚠️ No metrics available for final summary")
            return
        
        # Get final round metrics
        final_train = self.all_training_metrics[-1]
        final_test = self.all_test_metrics[-1]
        
        # Calculate statistics across all rounds
        train_losses = [m["train_loss"] for m in self.all_training_metrics]
        val_losses = [m["val_loss"] for m in self.all_training_metrics]
        test_losses = [m["test_loss"] for m in self.all_test_metrics]
        
        train_rmses = [m["train_rmse"] for m in self.all_training_metrics]
        val_rmses = [m["val_rmse"] for m in self.all_training_metrics]
        test_rmses = [m["test_rmse"] for m in self.all_test_metrics]
        
        train_r2s = [m["train_r2"] for m in self.all_training_metrics]
        val_r2s = [m["val_r2"] for m in self.all_training_metrics]
        test_r2s = [m["test_r2"] for m in self.all_test_metrics]
        
        # Find best round based on test R²
        best_round_idx = np.argmax(test_r2s)
        best_round = self.all_test_metrics[best_round_idx]["round"]
        best_test_r2 = test_r2s[best_round_idx]
        
        # Create summary row
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "client_id": self.client_id,
            "algorithm": algorithm,
            "model_type": hyperparams.model_type,
            "total_rounds": total_rounds,
            # Final round metrics
            "final_train_loss": final_train["train_loss"],
            "final_val_loss": final_train["val_loss"],
            "final_test_loss": final_test["test_loss"],
            "final_train_rmse": final_train["train_rmse"],
            "final_val_rmse": final_train["val_rmse"],
            "final_test_rmse": final_test["test_rmse"],
            "final_train_r2": final_train["train_r2"],
            "final_val_r2": final_train["val_r2"],
            "final_test_r2": final_test["test_r2"],
            # Statistics across all rounds
            "avg_train_loss": np.mean(train_losses),
            "std_train_loss": np.std(train_losses),
            "avg_val_loss": np.mean(val_losses),
            "std_val_loss": np.std(val_losses),
            "avg_test_loss": np.mean(test_losses),
            "std_test_loss": np.std(test_losses),
            "avg_train_rmse": np.mean(train_rmses),
            "std_train_rmse": np.std(train_rmses),
            "avg_val_rmse": np.mean(val_rmses),
            "std_val_rmse": np.std(val_rmses),
            "avg_test_rmse": np.mean(test_rmses),
            "std_test_rmse": np.std(test_rmses),
            "avg_train_r2": np.mean(train_r2s),
            "std_train_r2": np.std(train_r2s),
            "avg_val_r2": np.mean(val_r2s),
            "std_val_r2": np.std(val_r2s),
            "avg_test_r2": np.mean(test_r2s),
            "std_test_r2": np.std(test_r2s),
            "best_round": best_round,
            "best_test_r2": best_test_r2,
            # Hyperparameters
            "learning_rate": hyperparams.learning_rate,
            "batch_size": hyperparams.batch_size,
            "hidden_dim": hyperparams.hidden_dim,
            "num_layers": hyperparams.num_layers,
            "dropout": hyperparams.dropout
        }
        
        # Save to CSV
        with open(self.final_summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            writer.writerow(summary)
        
        print(f"💾 Final summary saved to: {self.final_summary_path}")
        
        return summary

class NASADataLoader:
    def __init__(self, data_path: str, hyperparams: Hyperparameters, random_state: int = 42):
        self.data_path = data_path
        self.hyperparams = hyperparams
        self.random_state = random_state
        self.scaler = None
        self.pca = None
        
    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        sequence_length = self.hyperparams.sequence_length
        sequences = []
        target_sequences = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:(i + sequence_length)])
            target_sequences.append(targets[i + sequence_length])
        
        return np.array(sequences), np.array(target_sequences)
    
    def apply_dimensionality_reduction(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply PCA/KPCA dimensionality reduction"""
        reduction_type = getattr(self.hyperparams, 'reduction_type', 'none')
        n_components = getattr(self.hyperparams, 'n_components', 10)
        
        if reduction_type == 'none':
            print("🔍 No dimensionality reduction applied")
            return X_train, X_val, X_test
        
        # Standardize the data first (important for PCA)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        if reduction_type == 'pca':
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            print(f"🔍 Applying PCA with {n_components} components")
        elif reduction_type == 'kpca':
            kernel = getattr(self.hyperparams, 'kernel', 'rbf')
            self.pca = KernelPCA(n_components=n_components, kernel=kernel, 
                               random_state=self.random_state)
            print(f"🔍 Applying KernelPCA with {n_components} components, kernel={kernel}")
        else:
            print(f"⚠️ Unknown reduction type: {reduction_type}")
            return X_train, X_val, X_test
        
        # Fit and transform
        X_train_reduced = self.pca.fit_transform(X_train_scaled)
        X_val_reduced = self.pca.transform(X_val_scaled)
        X_test_reduced = self.pca.transform(X_test_scaled)
        
        # Print variance explained for PCA
        if reduction_type == 'pca' and hasattr(self.pca, 'explained_variance_ratio_'):
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"📊 Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
        
        print(f"📊 Data shape - Before: {X_train.shape}, After: {X_train_reduced.shape}")
        
        return X_train_reduced, X_val_reduced, X_test_reduced
        
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and preprocess client data with proper train/val/test split"""
        try:
            # Check if path exists, if not try alternative structure
            if not os.path.exists(self.data_path):
                alt_paths = [
                    self.data_path,
                    self.data_path.replace("pre_split_data", "data"),
                    os.path.join(os.path.dirname(self.data_path), "train_data.txt"),
                    "data/train_data.txt"
                ]
                
                for alt_path in alt_paths:
                    if os.path.exists(alt_path):
                        self.data_path = alt_path
                        break
                else:
                    raise FileNotFoundError(f"No data file found in any expected location")
            
            data_file = os.path.join(self.data_path, "train_data.txt") if os.path.isdir(self.data_path) else self.data_path
            
            # Load data - FIXED: use raw string for regex
            data = pd.read_csv(data_file, sep=r'\s+', header=None)
            
            # Features: columns 2-25 (operational settings + sensors)
            X = data.iloc[:, 2:26].values.astype(np.float32)
            y = self._calculate_rul(data).astype(np.float32)
            
            print(f"📊 Loaded {len(X)} total samples from {self.data_path}")
            print(f"🔢 Original feature dimension: {X.shape[1]}")
            
            # Split into train+val and test sets
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.hyperparams.test_size, random_state=self.random_state, shuffle=True
            )
            
            # Split train+val into train and validation sets
            val_ratio = self.hyperparams.val_size / (1 - self.hyperparams.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, shuffle=True
            )
            
            # Apply dimensionality reduction if specified in hyperparams
            if hasattr(self.hyperparams, 'reduction_type') and self.hyperparams.reduction_type != 'none':
                X_train, X_val, X_test = self.apply_dimensionality_reduction(X_train, X_val, X_test)
            
            # Create sequences for LSTM if needed
            if self.hyperparams.model_type == "lstm":
                X_train, y_train = self.create_sequences(X_train, y_train)
                X_val, y_val = self.create_sequences(X_val, y_val)
                X_test, y_test = self.create_sequences(X_test, y_test)
                print(f"🔄 Created sequences with length {self.hyperparams.sequence_length}")
                print(f"   Final dataset shape: X {X_train.shape}, y {y_train.shape}")
            
            # Convert to PyTorch tensors
            X_train_tensor = torch.tensor(X_train)
            y_train_tensor = torch.tensor(y_train).unsqueeze(1) if len(y_train.shape) == 1 else torch.tensor(y_train)
            X_val_tensor = torch.tensor(X_val)
            y_val_tensor = torch.tensor(y_val).unsqueeze(1) if len(y_val.shape) == 1 else torch.tensor(y_val)
            X_test_tensor = torch.tensor(X_test)
            y_test_tensor = torch.tensor(y_test).unsqueeze(1) if len(y_test.shape) == 1 else torch.tensor(y_test)
            
            print(f"✅ Data split completed:")
            print(f"   Training samples: {len(X_train)}")
            print(f"   Validation samples: {len(X_val)}")
            print(f"   Test samples: {len(X_test)}")
            print(f"   Model type: {self.hyperparams.model_type}")
            print(f"   Final input dimension: {X_train_tensor.shape[-1]}")
            
            return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
            
        except Exception as e:
            print(f"❌ Error loading data from {self.data_path}: {e}")
            print("🔄 Using synthetic data for testing")
            # Return synthetic data for testing
            input_dim = getattr(self.hyperparams, 'n_components', 24) if hasattr(self.hyperparams, 'reduction_type') and self.hyperparams.reduction_type != 'none' else 24
            
            if self.hyperparams.model_type == "lstm":
                seq_len = self.hyperparams.sequence_length
                X_train = torch.randn(60, seq_len, input_dim)
                y_train = torch.randn(60, 1)
                X_val = torch.randn(20, seq_len, input_dim)
                y_val = torch.randn(20, 1)
                X_test = torch.randn(20, seq_len, input_dim)
                y_test = torch.randn(20, 1)
            else:
                X_train = torch.randn(60, input_dim)
                y_train = torch.randn(60, 1)
                X_val = torch.randn(20, input_dim)
                y_val = torch.randn(20, 1)
                X_test = torch.randn(20, input_dim)
                y_test = torch.randn(20, 1)
            return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _calculate_rul(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate RUL labels"""
        unit_ids = data[0].values
        time_cycles = data[1].values
        
        rul_labels = []
        for unit_id in np.unique(unit_ids):
            unit_mask = (unit_ids == unit_id)
            max_cycle = time_cycles[unit_mask].max()
            unit_rul = max_cycle - time_cycles[unit_mask]
            rul_labels.extend(unit_rul)
        
        return np.array(rul_labels)

class NASAFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, config: Dict):
        self.client_id = client_id
        self.config = config
        self.algorithm = config.get("algorithm", "fedavg")
        self.current_round = 0
        self.cv_completed = False
        self.total_rounds = config.get("server", {}).get("num_rounds", 10)
        
        # GPU optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔄 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"🎯 GPU: {torch.cuda.get_device_name()}")
        
        # Initialize hyperparameters
        model_config = config.get("model", {})
        self.hyperparams = Hyperparameters(model_config)
        
        # Initialize metrics logger
        log_dir = config.get("logging", {}).get("log_dir", "logs")
        self.metrics_logger = MetricsLogger(client_id, log_dir)
        
        # Log hyperparameters
        self.metrics_logger.log_hyperparameters(self.hyperparams)
        
        # Load data with proper splits
        data_path = self._get_data_path()
        
        self.data_loader = NASADataLoader(data_path, self.hyperparams)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.data_loader.load_data()
        
        # Move data to GPU
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.X_val = self.X_val.to(self.device)
        self.y_val = self.y_val.to(self.device)
        self.X_test = self.X_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        
        # Create model
        input_dim = self.X_train.shape[-1]
        self.model_kwargs = self.hyperparams.to_dict().copy()
        self.model_kwargs.pop('model_type', None)

        self.model = ModelFactory.create_model(
            self.hyperparams.model_type, 
            input_dim, 
            **self.model_kwargs
        )
        
        # Move model to GPU
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        if self.hyperparams.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.hyperparams.learning_rate,
                weight_decay=self.hyperparams.weight_decay
            )
        elif self.hyperparams.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.hyperparams.learning_rate,
                weight_decay=self.hyperparams.weight_decay,
                momentum=0.9
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparams.learning_rate)
            
        self.criterion = nn.MSELoss()
        
        print(f"✅ Client {client_id} ready:")
        print(f"   Model: {self.hyperparams.model_type.upper()}")
        print(f"   Training: {len(self.X_train)} samples")
        print(f"   Device: {self.device}")
        print(f"   Validation: {len(self.X_val)} samples") 
        print(f"   Test: {len(self.X_test)} samples")
        print(f"   Algorithm: {self.algorithm}")
        print(f"   Total Rounds: {self.total_rounds}")
        print(f"   Learning Rate: {self.hyperparams.learning_rate}")
        print(f"   Batch Size: {self.hyperparams.batch_size}")
        print(f"   Logging to: {log_dir}")

    def get_parameters(self, config):
        """Return model weights"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters - GPU compatible"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train the model for one round with proper validation"""
        self.current_round = config.get("current_round", 0)
        
        # Run CV only periodically to save computation time
        run_cv = (
            self.current_round == 0 or
            self.current_round % 5 == 0 or
            self.current_round == self.total_rounds - 1
        )
        
        if run_cv:
            k_folds = min(3, self.hyperparams.k_folds)
            print(f"\n🔍 Running {k_folds}-fold cross-validation (Round {self.current_round})")
            cv_metrics = self._k_fold_cross_validation(parameters, config, k_folds)
        else:
            cv_metrics = {
                "fold": [0],
                "train_loss": [0], 
                "val_loss": [0],
                "rmse": [0],
                "mse": [0],
                "mae": [0],
                "r2": [0]
            }
            print(f"⏩ Skipping CV for Round {self.current_round} (runs every 5 rounds)")
        
        # Train on full training dataset for federated learning
        self.set_parameters(parameters)
        
        epochs = config.get("local_epochs", self.hyperparams.local_epochs)
        batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(self.X_train), batch_size):
                batch_X = self.X_train[i:i+batch_size]
                batch_y = self.y_train[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if num_batches > 0:
                train_losses.append(epoch_loss / num_batches)
        
        # Evaluate on validation set
        with torch.no_grad():
            self.model.eval()
            train_predictions = self.model(self.X_train)
            train_loss = self.criterion(train_predictions, self.y_train).item()
            train_metrics = self._calculate_metrics(self.y_train, train_predictions)
            
            val_predictions = self.model(self.X_val)
            val_loss = self.criterion(val_predictions, self.y_val).item()
            val_metrics = self._calculate_metrics(self.y_val, val_predictions)
        
        # Reduced logging frequency
        if run_cv or self.current_round % 3 == 0 or self.current_round == self.total_rounds - 1:
            print(f"\n🎯 Round {self.current_round} Training Results:")
            print(f"   Training - Loss: {train_loss:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
            print(f"   Validation - Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f}, R²: {val_metrics['r2']:.4f}")
        else:
            print(f"Round {self.current_round}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_r2={val_metrics['r2']:.4f}")
        
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
            "algorithm": self.algorithm,
            "avg_epoch_loss": float(np.mean(train_losses) if train_losses else train_loss),
        }
        
        # Reduced CSV logging frequency
        if run_cv or self.current_round % 2 == 0 or self.current_round == self.total_rounds - 1:
            self.metrics_logger.log_training_metrics(self.current_round, results, self.hyperparams)
        
        return self.get_parameters({}), len(self.X_train), results

    def evaluate(self, parameters, config):
        """Evaluate the model on test set - FIXED AND COMPLETE VERSION"""
        # Set the received parameters
        self.set_parameters(parameters)
        
        # Set model to evaluation mode
        self.model.eval()
        
        test_loss = 0.0
        test_predictions = []
        test_targets = []
        
        # Evaluate in batches to handle large datasets
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
        
        # Calculate overall metrics
        test_loss = test_loss / len(self.X_test)
        test_predictions = np.concatenate(test_predictions).flatten()
        test_targets = np.concatenate(test_targets).flatten()
        
        # Calculate regression metrics
        mse = mean_squared_error(test_targets, test_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_targets, test_predictions)
        r2 = r2_score(test_targets, test_predictions)
        
        print(f"\n🧪 Round {self.current_round} Evaluation Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Also get validation metrics for comparison
        with torch.no_grad():
            val_predictions = self.model(self.X_val)
            val_metrics = self._calculate_metrics(self.y_val, val_predictions)
        
        # Log to test metrics file
        test_results = {
            "test_loss": test_loss,
            "test_rmse": rmse,
            "test_mse": mse,
            "test_mae": mae,
            "test_r2": r2,
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "algorithm": self.algorithm
        }
        
        self.metrics_logger.log_test_metrics(self.current_round, test_results, self.hyperparams.model_type)
        
        # Return metrics in Flower format
        return float(test_loss), len(self.X_test), {
            "rmse": float(rmse),
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2)
        }

    def _calculate_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        """Calculate RMSE, MSE, MAE, and R² metrics"""
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        
        mse = mean_squared_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)
        
        return {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }

    def _k_fold_cross_validation(self, parameters, config, k_folds: int = 3) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation on TRAINING data only"""
        self.set_parameters(parameters)
        
        epochs = min(2, config.get("local_epochs", self.hyperparams.local_epochs))
        batch_size = config.get("batch_size", self.hyperparams.batch_size)
        
        # Use only training data for cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_metrics = {
            "fold": [],
            "train_loss": [],
            "val_loss": [],
            "rmse": [],
            "mse": [],
            "mae": [],
            "r2": []
        }
        
        if not self.cv_completed:
            print(f"\n🔍 Starting {k_folds}-fold cross-validation on TRAINING data for client {self.client_id}")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X_train)):
            if not self.cv_completed:
                print(f"\n📊 Fold {fold + 1}/{k_folds}")
            
            # Split training data for this fold
            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # Create fresh model for this fold
            input_dim = self.X_train.shape[-1]
            fold_model = ModelFactory.create_model(
                self.hyperparams.model_type, 
                input_dim, 
                **self.model_kwargs
            )
            
            fold_model = fold_model.to(self.device)
            
            if self.hyperparams.optimizer == "adam":
                fold_optimizer = optim.Adam(
                    fold_model.parameters(), 
                    lr=self.hyperparams.learning_rate,
                    weight_decay=self.hyperparams.weight_decay
                )
            else:
                fold_optimizer = optim.Adam(fold_model.parameters(), lr=self.hyperparams.learning_rate)
            
            # Copy initial parameters
            fold_model.load_state_dict(self.model.state_dict())
            
            # Training loop for this fold
            fold_model.train()
            train_losses = []
            
            for epoch in range(epochs):
                epoch_loss = 0
                num_batches = 0
                
                cv_batch_size = min(batch_size * 2, len(X_fold_train))
                
                for i in range(0, len(X_fold_train), cv_batch_size):
                    batch_X = X_fold_train[i:i+cv_batch_size]
                    batch_y = y_fold_train[i:i+cv_batch_size]
                    
                    fold_optimizer.zero_grad()
                    outputs = fold_model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    fold_optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                if num_batches > 0:
                    train_losses.append(epoch_loss / num_batches)
            
            # Evaluate on fold validation set
            fold_model.eval()
            with torch.no_grad():
                val_predictions = fold_model(X_fold_val)
                val_loss = self.criterion(val_predictions, y_fold_val).item()
                val_metrics = self._calculate_metrics(y_fold_val, val_predictions)
            
            # Store metrics for this fold
            fold_metrics["fold"].append(fold + 1)
            fold_metrics["train_loss"].append(np.mean(train_losses) if train_losses else 0)
            fold_metrics["val_loss"].append(val_loss)
            fold_metrics["rmse"].append(val_metrics["rmse"])
            fold_metrics["mse"].append(val_metrics["mse"])
            fold_metrics["mae"].append(val_metrics["mae"])
            fold_metrics["r2"].append(val_metrics["r2"])
            
            if not self.cv_completed:
                print(f"   Fold {fold + 1} Results:")
                print(f"     Val Loss: {val_loss:.4f}")
                print(f"     Val RMSE: {val_metrics['rmse']:.4f}, Val R²: {val_metrics['r2']:.4f}")
        
        self.metrics_logger.log_cv_metrics(self.current_round, fold_metrics, self.hyperparams.model_type)
        
        if not self.cv_completed:
            print(f"\n📈 {k_folds}-Fold CV Summary (Training Data):")
            for metric in ["val_loss", "rmse", "r2"]:
                values = fold_metrics[metric]
                print(f"   {metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
            self.cv_completed = True
        
        return fold_metrics

    def _get_data_path(self) -> str:
        """Construct data path from configuration"""
        base_path = self.config["data"]["base_path"]
        num_clients = self.config["data"]["num_clients"]
        alpha = self.config["data"]["alpha"]
        
        if self.client_id.startswith("client_"):
            client_num = int(self.client_id.split("_")[1])
        else:
            client_num = int(self.client_id)
        
        path = f"{base_path}/{num_clients}_clients/alpha_{alpha}/client_{client_num}"
        print(f"🔍 Looking for data at: {path}")
        return path

    def generate_final_report(self):
        """Generate and display final comprehensive report"""
        print("\n" + "="*80)
        print("🎯 FINAL COMPREHENSIVE REPORT")
        print("="*80)
        
        # Generate final summary
        summary = self.metrics_logger.generate_final_summary(self.total_rounds, self.algorithm, self.hyperparams)
        
        if summary:
            print(f"\n📊 CLIENT: {self.client_id} | ALGORITHM: {self.algorithm} | MODEL: {self.hyperparams.model_type.upper()}")
            print(f"📈 TOTAL ROUNDS: {self.total_rounds}")
            
            # Hyperparameters
            print(f"\n⚙️  HYPERPARAMETERS:")
            print(f"   Learning Rate: {self.hyperparams.learning_rate}")
            print(f"   Batch Size: {self.hyperparams.batch_size}")
            print(f"   Local Epochs: {self.hyperparams.local_epochs}")
            if self.hyperparams.model_type == "lstm":
                print(f"   Hidden Dim: {self.hyperparams.hidden_dim}")
                print(f"   Num Layers: {self.hyperparams.num_layers}")
                print(f"   Sequence Length: {self.hyperparams.sequence_length}")
                print(f"   Use Attention: {self.hyperparams.use_attention}")
            else:
                print(f"   Hidden Dims: {self.hyperparams.hidden_dims}")
            print(f"   Dropout: {self.hyperparams.dropout}")
            print(f"   Optimizer: {self.hyperparams.optimizer}")
            
            # Final Round Performance
            print(f"\n🏁 FINAL ROUND PERFORMANCE:")
            print(f"   Training   - Loss: {summary['final_train_loss']:8.2f} | RMSE: {summary['final_train_rmse']:6.2f} | R²: {summary['final_train_r2']:7.4f}")
            print(f"   Validation - Loss: {summary['final_val_loss']:8.2f} | RMSE: {summary['final_val_rmse']:6.2f} | R²: {summary['final_val_r2']:7.4f}")
            print(f"   Test       - Loss: {summary['final_test_loss']:8.2f} | RMSE: {summary['final_test_rmse']:6.2f} | R²: {summary['final_test_r2']:7.4f}")
            
            # Statistics Across All Rounds
            print(f"\n📊 STATISTICS ACROSS ALL ROUNDS (Mean ± Std):")
            print(f"   Training Loss:   {summary['avg_train_loss']:8.2f} ± {summary['std_train_loss']:6.2f}")
            print(f"   Validation Loss: {summary['avg_val_loss']:8.2f} ± {summary['std_val_loss']:6.2f}")
            print(f"   Test Loss:       {summary['avg_test_loss']:8.2f} ± {summary['std_test_loss']:6.2f}")
            
            print(f"\n   Training RMSE:   {summary['avg_train_rmse']:6.2f} ± {summary['std_train_rmse']:5.2f}")
            print(f"   Validation RMSE: {summary['avg_val_rmse']:6.2f} ± {summary['std_val_rmse']:5.2f}")
            print(f"   Test RMSE:       {summary['avg_test_rmse']:6.2f} ± {summary['std_test_rmse']:5.2f}")
            
            print(f"\n   Training R²:     {summary['avg_train_r2']:7.4f} ± {summary['std_train_r2']:6.4f}")
            print(f"   Validation R²:   {summary['avg_val_r2']:7.4f} ± {summary['std_val_r2']:6.4f}")
            print(f"   Test R²:         {summary['avg_test_r2']:7.4f} ± {summary['std_test_r2']:6.4f}")
            
            # Best Performance
            print(f"\n⭐ BEST PERFORMANCE:")
            print(f"   Best Round: {summary['best_round']} (Test R²: {summary['best_test_r2']:.4f})")
            
            # Data Summary
            print(f"\n📋 DATA SUMMARY:")
            print(f"   Training samples:   {len(self.X_train)}")
            print(f"   Validation samples: {len(self.X_val)}")
            print(f"   Test samples:       {len(self.X_test)}")
            print(f"   Total samples:      {len(self.X_train) + len(self.X_val) + len(self.X_test)}")
            
        print("="*80)

def main():
    """Main function for client"""
    parser = argparse.ArgumentParser(description="NASA FL Client")
    parser.add_argument("--client-id", type=str, required=True, help="Client ID (e.g., client_0)")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--server-address", type=str, help="Server address (host:port)")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save CSV logs")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Update config with command line arguments
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["log_dir"] = args.log_dir
        
        algorithm = config.get("algorithm", "fedavg")
        server_host = config['server']['host']
        server_port = config['server']['port']
        server_address = args.server_address or f"{server_host}:{server_port}"
        
        print(f"🚀 Starting NASA FL Client: {args.client_id}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Server: {server_address}")
        print(f"K-Folds: {args.k_folds}")
        print(f"Log Directory: {args.log_dir}")
        
        # Create client instance
        client = NASAFlowerClient(args.client_id, config)
        
        # Use modern client API
        fl.client.start_client(
            server_address=server_address,
            client=client.to_client(),
        )
        
        # Generate final comprehensive report
        client.generate_final_report()
        
        print(f"✅ Client {args.client_id} completed | Algorithm: {algorithm.upper()}")
        
    except Exception as e:
        print(f"❌ Client error: {e}")
        raise

if __name__ == "__main__":
    main()

    