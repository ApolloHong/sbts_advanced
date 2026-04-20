"""
Neural Drift Estimators Module

Implements LSTM and Transformer-based drift estimators for
learning the drift function μ(t, x) from data.

Key Design Principle:
    The drift estimator MUST be trained on PURIFIED data (jumps removed)
    to prevent it from trying to fit jumps with drift, which causes instability.

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings

# Note: DriftEstimator base class is defined locally to avoid circular imports
from abc import ABC, abstractmethod

class DriftEstimator(ABC):
    """Abstract base class for drift estimators."""
    
    ESTIMATOR_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: np.ndarray, time_grid: np.ndarray) -> 'DriftEstimator':
        pass
    
    @abstractmethod
    def predict(self, t: Union[float, np.ndarray], x: np.ndarray) -> np.ndarray:
        pass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================
# Neural Network Architectures
# ============================================

if TORCH_AVAILABLE:
    
    class LSTMDriftNetwork(nn.Module):
        """
        LSTM network for drift estimation.
        
        Architecture:
            Input -> LSTM -> FC -> Output
        
        The network learns to predict the next-step drift given
        the current state and recent history.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            n_layers: int = 2,
            dropout: float = 0.3,
            bidirectional: bool = False
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.n_layers = n_layers
            self.bidirectional = bidirectional
            
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0,
                bidirectional=bidirectional
            )
            
            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
            
            self.fc = nn.Sequential(
                nn.Linear(lstm_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim)
            )
        
        def forward(
            self,
            x: torch.Tensor,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Forward pass.
            
            Args:
                x: Input tensor (batch, seq_len, input_dim)
                hidden: Optional initial hidden state
            
            Returns:
                Tuple of (output, hidden_state)
                - output: Drift predictions (batch, seq_len, input_dim)
                - hidden_state: Final hidden state
            """
            lstm_out, hidden = self.lstm(x, hidden)
            output = self.fc(lstm_out)
            return output, hidden
        
        def predict_drift(
            self,
            x: torch.Tensor,
            hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        ) -> torch.Tensor:
            """
            Predict drift at the last time step.
            
            Args:
                x: Input tensor (batch, seq_len, input_dim)
                hidden: Optional hidden state
            
            Returns:
                Drift prediction (batch, input_dim)
            """
            output, _ = self.forward(x, hidden)
            return output[:, -1, :]
    
    
    class TransformerDriftNetwork(nn.Module):
        """
        Transformer network for drift estimation.
        
        Uses self-attention to capture long-range dependencies
        in the time series.
        """
        
        def __init__(
            self,
            input_dim: int,
            d_model: int = 128,
            n_heads: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
            max_seq_len: int = 100
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.d_model = d_model
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, d_model)
            
            # Positional encoding
            self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            
            # Output projection
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, input_dim)
            )
        
        def _create_positional_encoding(
            self,
            max_len: int,
            d_model: int
        ) -> torch.Tensor:
            """Create sinusoidal positional encoding."""
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            return pe.unsqueeze(0)  # (1, max_len, d_model)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor (batch, seq_len, input_dim)
            
            Returns:
                Drift predictions (batch, seq_len, input_dim)
            """
            batch_size, seq_len, _ = x.shape
            
            # Project input
            x = self.input_proj(x)
            
            # Add positional encoding
            x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
            
            # Create causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
            
            # Transformer forward
            x = self.transformer(x, mask=mask)
            
            # Output projection
            output = self.output_proj(x)
            
            return output
        
        def predict_drift(self, x: torch.Tensor) -> torch.Tensor:
            """Predict drift at the last time step."""
            output = self.forward(x)
            return output[:, -1, :]


# ============================================
# LSTM Drift Estimator
# ============================================

class LSTMDriftEstimator(DriftEstimator):
    """
    LSTM-based Drift Estimator.
    
    CRITICAL: Must be trained on PURIFIED data (jumps removed and interpolated).
    Training on raw data with jumps causes the LSTM to try to fit jumps with drift,
    leading to instability and poor generalization.
    
    Usage:
        estimator = LSTMDriftEstimator(config)
        estimator.fit(purified_data, time_grid)  # NOT raw_data!
        drift = estimator.predict(t, x)
    """
    
    ESTIMATOR_TYPE = "lstm"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LSTM drift estimator.
        
        Args:
            config: Configuration with keys:
                - lstm_hidden: Hidden dimension (default: 128)
                - lstm_epochs: Training epochs (default: 50)
                - lstm_lr: Learning rate (default: 0.005)
                - lstm_dropout: Dropout rate (default: 0.3)
                - lstm_use_huber: Use Huber loss (default: True)
                - lstm_weight_decay: Weight decay (default: 0.001)
                - lstm_seq_len: Sequence length for training (default: 20)
                - lstm_drift_dampening: Dampening factor (default: 0.9)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for LSTMDriftEstimator")
        
        self.hidden_dim = config.get('lstm_hidden', 128)
        self.epochs = config.get('lstm_epochs', 50)
        self.lr = config.get('lstm_lr', 0.005)
        self.dropout = config.get('lstm_dropout', 0.3)
        self.use_huber = config.get('lstm_use_huber', True)
        self.weight_decay = config.get('lstm_weight_decay', 0.001)
        self.seq_len = config.get('lstm_seq_len', 20)
        self.dampening = config.get('lstm_drift_dampening', 0.9)
        
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Model (to be created during fit)
        self.model = None
        self.n_features = None
        self.time_grid = None
        self.data_mean = None
        self.data_std = None
        
        # Training history
        self.train_losses = []
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'LSTMDriftEstimator':
        """
        Fit LSTM drift estimator to PURIFIED data.
        
        IMPORTANT: The data should be purified (jumps removed and interpolated)
        before calling this method.
        
        Args:
            data: PURIFIED time series data (n_samples, seq_len, n_features)
            time_grid: Time points (seq_len,)
            verbose: Whether to print training progress
        
        Returns:
            self for method chaining
        """
        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.time_grid = time_grid
        
        # Normalize data
        self.data_mean = np.mean(data)
        self.data_std = np.std(data) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        
        # Compute returns (drift targets)
        returns = np.diff(data_norm, axis=1)
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        drift_targets = returns / dt  # Approximate drift
        
        # Create training sequences
        X_sequences = []
        y_targets = []
        
        for i in range(n_samples):
            for j in range(self.seq_len, seq_len - 1):
                X_sequences.append(data_norm[i, j-self.seq_len:j, :])
                y_targets.append(drift_targets[i, j-1, :])
        
        if len(X_sequences) == 0:
            warnings.warn("Not enough data for LSTM training")
            self.is_fitted = True
            return self
        
        X = torch.tensor(np.array(X_sequences), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y_targets), dtype=torch.float32).to(self.device)
        
        # Create model
        self.model = LSTMDriftNetwork(
            input_dim=n_features,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout
        ).to(self.device)
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        if self.use_huber:
            criterion = nn.HuberLoss(delta=1.0)
        else:
            criterion = nn.MSELoss()
        
        # Learning rate scheduler
        scheduler_kwargs = {
            'mode': 'min',
            'factor': 0.5,
            'patience': 10,
        }
        # Some PyTorch versions expose a verbose flag while newer ones do not.
        if 'verbose' in torch.optim.lr_scheduler.ReduceLROnPlateau.__init__.__code__.co_varnames:
            scheduler_kwargs['verbose'] = verbose
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **scheduler_kwargs
        )
        
        # Training loop
        self.model.train()
        batch_size = min(128, len(X))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.train_losses = []
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                pred, _ = self.model(batch_X)
                pred_drift = pred[:, -1, :]  # Last time step
                
                # Apply dampening
                pred_drift = pred_drift * self.dampening
                
                loss = criterion(pred_drift, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [LSTM Drift] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        self.model.eval()
        self.is_fitted = True
        return self
    
    def predict(
        self,
        t: Union[float, np.ndarray],
        x: np.ndarray,
        history: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict drift at given time and state.
        
        Args:
            t: Time point(s)
            x: State(s) of shape (n_samples, n_features) or (n_features,)
            history: Optional recent history for LSTM context
        
        Returns:
            Predicted drift of same shape as x
        """
        if not self.is_fitted or self.model is None:
            return np.zeros_like(x)
        
        self.model.eval()
        
        # Ensure 2D
        if x.ndim == 1:
            x = x[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False
        
        n_samples = x.shape[0]
        
        # Normalize
        x_norm = (x - self.data_mean) / self.data_std
        
        # Create input sequence
        if history is not None:
            # Use provided history
            history_norm = (history - self.data_mean) / self.data_std
            X = torch.tensor(history_norm, dtype=torch.float32).to(self.device)
        else:
            # Create synthetic history (repeat current state)
            X = torch.tensor(
                np.tile(x_norm[:, np.newaxis, :], (1, self.seq_len, 1)),
                dtype=torch.float32
            ).to(self.device)
        
        with torch.no_grad():
            pred_drift = self.model.predict_drift(X)
            drift = pred_drift.cpu().numpy() * self.dampening
        
        # Denormalize (drift is in normalized space, need to scale back)
        drift = drift * self.data_std
        
        if squeeze_output:
            drift = drift[0]
        
        return drift
    
    def get_training_history(self) -> List[float]:
        """Get training loss history."""
        return self.train_losses


# ============================================
# Transformer Drift Estimator
# ============================================

class TransformerDriftEstimator(DriftEstimator):
    """
    Transformer-based Drift Estimator.
    
    Uses self-attention to capture long-range dependencies.
    """
    
    ESTIMATOR_TYPE = "transformer"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Transformer drift estimator.
        
        Args:
            config: Configuration with keys:
                - transformer_d_model: Model dimension (default: 128)
                - transformer_n_heads: Number of attention heads (default: 4)
                - transformer_n_layers: Number of layers (default: 2)
                - transformer_epochs: Training epochs (default: 50)
                - transformer_lr: Learning rate (default: 0.001)
                - transformer_dropout: Dropout rate (default: 0.1)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TransformerDriftEstimator")
        
        self.d_model = config.get('transformer_d_model', 128)
        self.n_heads = config.get('transformer_n_heads', 4)
        self.n_layers = config.get('transformer_n_layers', 2)
        self.epochs = config.get('transformer_epochs', 50)
        self.lr = config.get('transformer_lr', 0.001)
        self.dropout = config.get('transformer_dropout', 0.1)
        self.seq_len = config.get('transformer_seq_len', 20)
        
        self.device = torch.device(config.get('device', 'cuda'))
        
        self.model = None
        self.n_features = None
        self.data_mean = None
        self.data_std = None
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True
    ) -> 'TransformerDriftEstimator':
        """Fit Transformer drift estimator to PURIFIED data."""
        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        
        # Normalize
        self.data_mean = np.mean(data)
        self.data_std = np.std(data) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        
        # Compute drift targets
        returns = np.diff(data_norm, axis=1)
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
        drift_targets = returns / dt
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        for i in range(n_samples):
            for j in range(self.seq_len, seq_len - 1):
                X_sequences.append(data_norm[i, j-self.seq_len:j, :])
                y_targets.append(drift_targets[i, j-1, :])
        
        if len(X_sequences) == 0:
            self.is_fitted = True
            return self
        
        X = torch.tensor(np.array(X_sequences), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y_targets), dtype=torch.float32).to(self.device)
        
        # Create model
        self.model = TransformerDriftNetwork(
            input_dim=n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            max_seq_len=seq_len
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.HuberLoss(delta=1.0)
        
        # Training
        self.model.train()
        batch_size = min(64, len(X))
        
        for epoch in range(self.epochs):
            perm = torch.randperm(len(X))
            X, y = X[perm], y[perm]
            
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                pred = self.model.predict_drift(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  [Transformer Drift] Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/n_batches:.6f}")
        
        self.model.eval()
        self.is_fitted = True
        return self
    
    def predict(
        self,
        t: Union[float, np.ndarray],
        x: np.ndarray,
        history: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict drift at given time and state."""
        if not self.is_fitted or self.model is None:
            return np.zeros_like(x)
        
        self.model.eval()
        
        if x.ndim == 1:
            x = x[np.newaxis, :]
            squeeze = True
        else:
            squeeze = False
        
        x_norm = (x - self.data_mean) / self.data_std
        
        if history is not None:
            history_norm = (history - self.data_mean) / self.data_std
            X = torch.tensor(history_norm, dtype=torch.float32).to(self.device)
        else:
            X = torch.tensor(
                np.tile(x_norm[:, np.newaxis, :], (1, self.seq_len, 1)),
                dtype=torch.float32
            ).to(self.device)
        
        with torch.no_grad():
            drift = self.model.predict_drift(X).cpu().numpy()
        
        drift = drift * self.data_std
        
        if squeeze:
            drift = drift[0]
        
        return drift


# ============================================
# Factory Function
# ============================================

def get_neural_drift_estimator(config: Dict[str, Any]) -> DriftEstimator:
    """
    Factory function to create neural drift estimator.
    
    Args:
        config: Configuration with 'drift_estimator' key
    
    Returns:
        DriftEstimator instance
    """
    estimator_type = config.get('drift_estimator', 'lstm')
    
    if estimator_type == 'lstm':
        return LSTMDriftEstimator(config)
    elif estimator_type == 'transformer':
        return TransformerDriftEstimator(config)
    else:
        raise ValueError(f"Unknown drift estimator type: {estimator_type}")
