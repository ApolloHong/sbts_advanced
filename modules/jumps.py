"""
Jump Detection Module

Implements static and neural jump detectors for Merton Jump-Diffusion models.
Includes the "Filter & Interpolate" strategy for data purification.

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
from numba import njit, prange
import warnings

# Note: JumpDetector base class is defined locally to avoid circular imports
from abc import ABC, abstractmethod

class JumpDetector(ABC):
    """Abstract base class for jump detectors."""
    
    DETECTOR_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False
        self.jump_intensity = None
        self.jump_mean = None
        self.jump_std = None
    
    @abstractmethod
    def fit(
        self,
        data: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> 'JumpDetector':
        pass
    
    @abstractmethod
    def detect(self, data: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def filter_and_interpolate(self, data: np.ndarray) -> np.ndarray:
        pass
    
    def sample_jumps(self, n_samples: int, n_steps: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("JumpDetector must be fitted before sampling")
        jump_probs = 1 - np.exp(-self.jump_intensity * dt)
        jump_mask = np.random.random((n_samples, n_steps)) < jump_probs
        jump_sizes = np.random.normal(self.jump_mean, self.jump_std, (n_samples, n_steps))
        jump_sizes = jump_sizes * jump_mask
        return jump_mask, jump_sizes

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================
# Numba-Accelerated Jump Detection
# ============================================

@njit(cache=True, parallel=True)
def _detect_jumps_threshold_numba(
    returns: np.ndarray,
    threshold: float,
    rolling_window: int = 20
) -> np.ndarray:
    """
    Detect jumps using rolling window threshold method.
    
    Args:
        returns: Returns array of shape (n_samples, seq_len)
        threshold: Number of standard deviations for threshold
        rolling_window: Window size for rolling statistics
    
    Returns:
        Boolean jump mask of same shape as returns
    """
    n_samples, seq_len = returns.shape
    jump_mask = np.zeros((n_samples, seq_len), dtype=np.bool_)
    
    for i in prange(n_samples):
        for j in range(rolling_window, seq_len):
            # Rolling mean and std
            window = returns[i, j-rolling_window:j]
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 1e-10:
                z_score = np.abs(returns[i, j] - mean) / std
                if z_score > threshold:
                    jump_mask[i, j] = True
    
    return jump_mask


@njit(cache=True)
def _interpolate_jumps_numba(
    data: np.ndarray,
    jump_mask: np.ndarray
) -> np.ndarray:
    """
    Interpolate over detected jumps using linear interpolation.
    
    Args:
        data: Original data of shape (n_samples, seq_len)
        jump_mask: Boolean jump mask
    
    Returns:
        Purified data with jumps interpolated
    """
    n_samples, seq_len = data.shape
    purified = data.copy()
    
    for i in range(n_samples):
        j = 0
        while j < seq_len:
            if jump_mask[i, j]:
                # Find start and end of jump region
                start = j
                while j < seq_len and jump_mask[i, j]:
                    j += 1
                end = j
                
                # Interpolate
                if start > 0 and end < seq_len:
                    # Linear interpolation
                    val_start = purified[i, start - 1]
                    val_end = purified[i, end]
                    n_points = end - start + 1
                    for k in range(start, end):
                        alpha = (k - start + 1) / n_points
                        purified[i, k] = val_start * (1 - alpha) + val_end * alpha
                elif start == 0 and end < seq_len:
                    # Use end value
                    for k in range(start, end):
                        purified[i, k] = purified[i, end]
                elif start > 0 and end >= seq_len:
                    # Use start value
                    for k in range(start, seq_len):
                        purified[i, k] = purified[i, start - 1]
            else:
                j += 1
    
    return purified


# ============================================
# Static Jump Detector
# ============================================

class StaticJumpDetector(JumpDetector):
    """
    Static Jump Detector using threshold-based detection.
    
    Implements the "Filter & Interpolate" strategy:
    1. Detect jumps using rolling z-score threshold
    2. Estimate jump parameters (λ, μ_J, σ_J)
    3. Interpolate over jumps to create purified data
    """
    
    DETECTOR_TYPE = "static"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize static jump detector.
        
        Args:
            config: Configuration with keys:
                - jump_threshold_std: Z-score threshold (default: 4.0)
                - jump_rolling_window: Rolling window size (default: 20)
        """
        super().__init__(config)
        
        self.threshold = config.get('jump_threshold_std', 4.0)
        self.rolling_window = config.get('jump_rolling_window', 20)
        self.jump_size_scale = float(config.get('jump_size_scale', 1.0))
        self.jump_size_clip = config.get('jump_size_clip', None)
        if self.jump_size_clip is not None:
            self.jump_size_clip = float(self.jump_size_clip)
        
        # Per-feature parameters
        self.jump_intensities = None
        self.jump_means = None
        self.jump_stds = None
        self.reference_sigma = None
        self.n_features = None
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: Optional[np.ndarray] = None
    ) -> 'StaticJumpDetector':
        """
        Fit jump detector and estimate parameters.
        
        Args:
            data: Time series data of shape (n_samples, seq_len, n_features)
                  or (n_samples, seq_len) for univariate
        
        Returns:
            self for method chaining
        """
        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        if time_grid is not None and len(time_grid) > 1:
            dt = float(np.mean(np.diff(np.asarray(time_grid, dtype=np.float64))))
        else:
            dt = float(self.config.get('dt', self.config.get('jump_dt', 1.0)))
        if dt <= 0:
            raise ValueError("time_grid must be strictly increasing for jump calibration")
        
        # Compute returns
        returns = np.diff(data, axis=1)
        
        # Initialize parameter arrays
        self.jump_intensities = np.zeros(n_features)
        self.jump_means = np.zeros(n_features)
        self.jump_stds = np.zeros(n_features)
        
        # Fit for each feature
        for k in range(n_features):
            ret_k = returns[:, :, k]
            
            # Detect jumps
            jump_mask = _detect_jumps_threshold_numba(
                ret_k.astype(np.float64),
                self.threshold,
                self.rolling_window
            )
            
            # Estimate parameters
            n_jumps = np.sum(jump_mask)
            total_time = n_samples * (seq_len - 1) * dt
            
            self.jump_intensities[k] = n_jumps / total_time if total_time > 0 else 0.0
            
            jump_sizes = ret_k[jump_mask]
            if len(jump_sizes) > 0:
                self.jump_means[k] = np.mean(jump_sizes)
                self.jump_stds[k] = np.std(jump_sizes)
            else:
                self.jump_means[k] = 0.0
                self.jump_stds[k] = np.std(ret_k)
        
        # Set aggregate parameters for backward compatibility
        self.jump_intensity = np.mean(self.jump_intensities)
        self.jump_mean = np.mean(self.jump_means)
        self.jump_std = np.mean(self.jump_stds)
        
        self.is_fitted = True
        return self

    def set_reference_params(
        self,
        lambda0: Union[float, np.ndarray],
        c: Union[float, np.ndarray],
        gamma: Union[float, np.ndarray],
        sigma: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        """
        Override jump sampler parameters with calibrated SBJTS reference values.

        This keeps the detector API unchanged while allowing the three-stage
        calibration pipeline to provide component-wise Merton jump parameters.
        """
        if self.n_features is None:
            self.n_features = 1

        def per_feature(value, default):
            if value is None:
                return np.full(self.n_features, default, dtype=np.float64)
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim == 0:
                return np.full(self.n_features, float(arr), dtype=np.float64)
            if arr.size != self.n_features:
                return np.resize(arr, self.n_features).astype(np.float64)
            return arr.astype(np.float64)

        self.jump_intensities = np.clip(per_feature(lambda0, 0.0), 0.0, 1000.0)
        self.jump_means = per_feature(c, 0.0)
        self.jump_stds = np.maximum(per_feature(gamma, 1e-4), 1e-4)
        self.reference_sigma = None if sigma is None else np.maximum(per_feature(sigma, 1e-4), 1e-4)

        self.jump_intensity = float(np.mean(self.jump_intensities))
        self.jump_mean = float(np.mean(self.jump_means))
        self.jump_std = float(np.mean(self.jump_stds))
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        Detect jumps in data.
        
        Args:
            data: Time series data of shape (n_samples, seq_len, n_features)
                  or (n_samples, seq_len)
        
        Returns:
            Boolean mask indicating jump locations
        """
        # Ensure 3D
        original_ndim = data.ndim
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        
        # Compute returns
        returns = np.diff(data, axis=1)
        
        # Detect for each feature
        jump_mask = np.zeros((n_samples, seq_len - 1, n_features), dtype=bool)
        
        for k in range(n_features):
            jump_mask[:, :, k] = _detect_jumps_threshold_numba(
                returns[:, :, k].astype(np.float64),
                self.threshold,
                self.rolling_window
            )
        
        # Pad to match original shape
        jump_mask_full = np.zeros((n_samples, seq_len, n_features), dtype=bool)
        jump_mask_full[:, 1:, :] = jump_mask
        
        if original_ndim == 2:
            return jump_mask_full[:, :, 0]
        return jump_mask_full
    
    def filter_and_interpolate(self, data: np.ndarray) -> np.ndarray:
        """
        Remove jumps and interpolate to create purified data.
        
        This is the "Filter & Interpolate" strategy that fixes the
        volatility calibration issue.
        
        Args:
            data: Raw time series data
        
        Returns:
            Purified data with jumps removed and interpolated
        """
        # Ensure 3D
        original_ndim = data.ndim
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        
        # Detect jumps
        jump_mask = self.detect(data)
        
        # Interpolate for each feature
        purified = np.zeros_like(data)
        
        for k in range(n_features):
            purified[:, :, k] = _interpolate_jumps_numba(
                data[:, :, k].astype(np.float64),
                jump_mask[:, :, k] if jump_mask.ndim == 3 else jump_mask
            )
        
        if original_ndim == 2:
            return purified[:, :, 0]
        return purified
    
    def sample_jumps(
        self,
        n_samples: int,
        n_steps: int,
        dt: float,
        feature_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample jump occurrences and sizes.
        
        Args:
            n_samples: Number of paths
            n_steps: Number of time steps
            dt: Time step size
            feature_idx: Optional feature index (for multivariate)
        
        Returns:
            Tuple of (jump_mask, jump_sizes)
        """
        if not self.is_fitted:
            raise RuntimeError("JumpDetector must be fitted before sampling")
        
        if feature_idx is not None:
            intensity = self.jump_intensities[feature_idx]
            mean = self.jump_means[feature_idx]
            std = self.jump_stds[feature_idx]
        elif self.n_features is not None and self.n_features > 1:
            jump_sizes = np.zeros((n_samples, n_steps, self.n_features))
            jump_mask = np.zeros((n_samples, n_steps, self.n_features), dtype=bool)
            for k in range(self.n_features):
                jump_probs = 1 - np.exp(-self.jump_intensities[k] * dt)
                jump_mask[:, :, k] = np.random.random((n_samples, n_steps)) < jump_probs
                feature_jumps = np.random.normal(
                    self.jump_means[k],
                    self.jump_stds[k],
                    (n_samples, n_steps)
                )
                feature_jumps = self._stabilize_jump_sizes(feature_jumps)
                jump_sizes[:, :, k] = feature_jumps * jump_mask[:, :, k]
            return jump_mask, jump_sizes
        else:
            intensity = self.jump_intensity
            mean = self.jump_mean
            std = self.jump_std
        
        # Sample from Poisson process
        jump_probs = 1 - np.exp(-intensity * dt)
        jump_mask = np.random.random((n_samples, n_steps)) < jump_probs
        
        # Sample jump sizes
        jump_sizes = np.random.normal(mean, std, (n_samples, n_steps))
        jump_sizes = self._stabilize_jump_sizes(jump_sizes)
        jump_sizes = jump_sizes * jump_mask

        return jump_mask, jump_sizes

    def _stabilize_jump_sizes(self, jump_sizes: np.ndarray) -> np.ndarray:
        """Apply optional generation-time jump size shrinkage and clipping."""
        if self.jump_size_scale != 1.0:
            jump_sizes = jump_sizes * self.jump_size_scale
        if self.jump_size_clip is not None and self.jump_size_clip > 0:
            jump_sizes = np.clip(jump_sizes, -self.jump_size_clip, self.jump_size_clip)
        return jump_sizes
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get estimated jump parameters."""
        return {
            'intensity': self.jump_intensity,
            'mean': self.jump_mean,
            'std': self.jump_std,
            'intensities': self.jump_intensities,
            'means': self.jump_means,
            'stds': self.jump_stds,
            'reference_sigma': self.reference_sigma
        }


# ============================================
# Neural Jump Detector
# ============================================

if TORCH_AVAILABLE:
    
    class IntensityNetwork(nn.Module):
        """
        Neural network for predicting time-varying jump intensity λ(t).
        
        Uses LSTM to capture temporal patterns in jump occurrence.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            n_layers: int = 2,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0
            )
            
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Ensure positive intensity
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor of shape (batch, seq_len, input_dim)
            
            Returns:
                Intensity predictions of shape (batch, seq_len, 1)
            """
            lstm_out, _ = self.lstm(x)
            intensity = self.fc(lstm_out)
            return intensity
    
    
    class FocalLoss(nn.Module):
        """
        Focal Loss for handling class imbalance in jump detection.
        
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
        Useful because jumps are rare events.
        """
        
        def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
        
        def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute focal loss.
            
            Args:
                inputs: Predicted probabilities
                targets: Binary targets
            
            Returns:
                Focal loss value
            """
            bce = nn.functional.binary_cross_entropy(
                inputs, targets, reduction='none'
            )
            
            p_t = inputs * targets + (1 - inputs) * (1 - targets)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * (1 - p_t) ** self.gamma
            
            return (focal_weight * bce).mean()


class NeuralJumpDetector(JumpDetector):
    """
    Neural Jump Detector with time-varying intensity.
    
    Uses a neural network to predict λ(t, h_t) where h_t is the hidden state.
    This allows the jump intensity to be "endogenous" - dependent on market conditions.
    
    Key improvement: Uses Focal Loss to handle class imbalance (jumps are rare).
    """
    
    DETECTOR_TYPE = "neural"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize neural jump detector.
        
        Args:
            config: Configuration with keys:
                - neural_jump_hidden_dim: Hidden dimension (default: 64)
                - neural_jump_epochs: Training epochs (default: 30)
                - neural_jump_lr: Learning rate (default: 0.001)
                - neural_jump_seq_len: Sequence length for LSTM (default: 10)
                - jump_threshold_std: Threshold for initial detection (default: 4.0)
                - focal_alpha: Focal loss alpha (default: 0.25)
                - focal_gamma: Focal loss gamma (default: 2.0)
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for NeuralJumpDetector")
        
        self.hidden_dim = config.get('neural_jump_hidden_dim', 64)
        self.epochs = config.get('neural_jump_epochs', 30)
        self.lr = config.get('neural_jump_lr', 0.001)
        self.seq_len = config.get('neural_jump_seq_len', 10)
        self.threshold = config.get('jump_threshold_std', 4.0)
        self.focal_alpha = config.get('focal_alpha', 0.25)
        self.focal_gamma = config.get('focal_gamma', 2.0)
        self.verbose = config.get('verbose', True)
        
        self.device = torch.device(config.get('device', 'cuda'))
        
        # Models (to be created during fit)
        self.intensity_net = None
        self.static_detector = None
        self.n_features = None
    
    def fit(
        self,
        data: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
        verbose: Optional[bool] = None
    ) -> 'NeuralJumpDetector':
        """
        Fit neural jump detector.
        
        Training process:
        1. Use static detector to get initial jump labels
        2. Train intensity network to predict jump probability
        3. Use Focal Loss to handle class imbalance
        
        Args:
            data: Time series data of shape (n_samples, seq_len, n_features)
        
        Returns:
            self for method chaining
        """
        if verbose is None:
            verbose = self.verbose

        # Ensure 3D
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        
        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        
        # First, use static detector to get jump labels
        self.static_detector = StaticJumpDetector(self.config)
        self.static_detector.fit(data, time_grid=time_grid)
        
        # Get jump parameters from static detector
        self.jump_intensity = self.static_detector.jump_intensity
        self.jump_mean = self.static_detector.jump_mean
        self.jump_std = self.static_detector.jump_std
        
        # Get jump labels
        jump_mask = self.static_detector.detect(data)
        
        # Prepare training data for intensity network
        returns = np.diff(data, axis=1)
        
        # Create sequences for LSTM
        X_sequences = []
        y_labels = []
        
        for i in range(n_samples):
            for j in range(self.seq_len, seq_len - 1):
                # Input: returns and absolute returns (volatility proxy)
                seq = returns[i, j-self.seq_len:j, :]
                abs_seq = np.abs(seq)
                features = np.concatenate([seq, abs_seq], axis=-1)
                X_sequences.append(features)
                
                # Label: whether a jump occurs at this point
                y_labels.append(float(np.any(jump_mask[i, j+1, :])))
        
        if len(X_sequences) == 0:
            warnings.warn("Not enough data for neural jump detector training")
            self.is_fitted = True
            return self
        
        X = torch.tensor(np.array(X_sequences), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(y_labels), dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        # Create and train intensity network
        input_dim = n_features * 2  # returns + abs_returns
        self.intensity_net = IntensityNetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.intensity_net.parameters(), lr=self.lr)
        criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        
        # Training loop
        self.intensity_net.train()
        batch_size = min(128, len(X))
        
        for epoch in range(self.epochs):
            # Shuffle
            perm = torch.randperm(len(X))
            X = X[perm]
            y = y[perm]
            
            total_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass - get last time step prediction
                intensity = self.intensity_net(batch_X)[:, -1, :]
                
                # Convert intensity to probability
                prob = 1 - torch.exp(-intensity)
                prob = torch.clamp(prob, 1e-7, 1 - 1e-7)
                
                loss = criterion(prob, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / n_batches
                print(f"  [Neural Jump] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        self.intensity_net.eval()
        self.is_fitted = True
        return self
    
    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        Detect jumps using neural intensity prediction.
        
        Args:
            data: Time series data
        
        Returns:
            Boolean jump mask
        """
        # Fall back to static detection
        return self.static_detector.detect(data)
    
    def filter_and_interpolate(self, data: np.ndarray) -> np.ndarray:
        """
        Remove jumps and interpolate.
        
        Args:
            data: Raw time series data
        
        Returns:
            Purified data
        """
        return self.static_detector.filter_and_interpolate(data)
    
    def predict_intensity(
        self,
        history: np.ndarray
    ) -> np.ndarray:
        """
        Predict jump intensity given recent history.
        
        Args:
            history: Recent returns of shape (batch, seq_len, n_features)
        
        Returns:
            Predicted intensity of shape (batch,)
        """
        if self.intensity_net is None:
            return np.full(history.shape[0], self.jump_intensity)
        
        self.intensity_net.eval()
        
        with torch.no_grad():
            # Prepare features
            abs_history = np.abs(history)
            features = np.concatenate([history, abs_history], axis=-1)
            
            X = torch.tensor(features, dtype=torch.float32).to(self.device)
            intensity = self.intensity_net(X)[:, -1, 0]
            
            return intensity.cpu().numpy()
    
    def sample_jumps_neural(
        self,
        history: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample jumps using neural intensity prediction.
        
        Args:
            history: Recent returns of shape (batch, seq_len, n_features)
            dt: Time step size
        
        Returns:
            Tuple of (jump_mask, jump_sizes)
        """
        n_samples = history.shape[0]
        
        # Predict intensity
        intensity = self.predict_intensity(history)
        
        # Sample from Poisson process with predicted intensity
        jump_probs = 1 - np.exp(-intensity * dt)
        jump_mask = np.random.random(n_samples) < jump_probs
        
        # Sample jump sizes
        jump_sizes = np.random.normal(self.jump_mean, self.jump_std, n_samples)
        jump_sizes = jump_sizes * jump_mask
        
        return jump_mask, jump_sizes


# ============================================
# Factory Function
# ============================================

def get_jump_detector(config: Dict[str, Any]) -> JumpDetector:
    """
    Factory function to create jump detector based on config.
    
    Args:
        config: Configuration dictionary with 'use_neural_jumps' key
    
    Returns:
        JumpDetector instance
    """
    use_neural = config.get('use_neural_jumps', False)
    
    if use_neural:
        if not TORCH_AVAILABLE:
            warnings.warn("PyTorch not available, falling back to static detector")
            return StaticJumpDetector(config)
        return NeuralJumpDetector(config)
    else:
        return StaticJumpDetector(config)
