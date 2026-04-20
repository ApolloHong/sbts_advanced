"""
Neural Jump-Diffusion Module for JD-SBTS

This module implements "Endogenous Jumps" where the jump intensity λ is a function
of the hidden state h_t (from LSTM/Transformer), rather than a constant.

Inspired by DSL-Lab/neural-MJD implementation.

Key Components:
1. IntensityNet: Neural network that predicts time-varying jump intensity
2. NeuralJumpDetector: Extends JumpDetector with learned intensity
3. NeuralMJDParameters: Predicts full MJD parameters (μ, σ, λ, ν, γ)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math


class IntensityNet(nn.Module):
    """
    Neural network that predicts time-varying jump intensity λ(t, h_t).
    
    The intensity is modeled as a function of:
    - Current state x_t
    - Hidden state h_t from an LSTM encoder
    - Time t
    
    Architecture inspired by DSL-Lab/neural-MJD.
    """
    def __init__(self, input_dim, hidden_dim=64, output_dim=1, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM encoder for temporal patterns
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Time embedding (sinusoidal)
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        
        # Intensity prediction head
        # Output: log(λ) to ensure λ > 0
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Bounds for numerical stability
        self.max_log_lambda = math.log(1.0)  # λ ≤ 1.0 (max 1 jump per unit time)
        self.min_log_lambda = math.log(1e-6)  # λ ≥ 1e-6
        
    def forward(self, x_seq, t):
        """
        Predict jump intensity given historical sequence and time.
        
        Args:
            x_seq: (batch, seq_len, input_dim) historical observations
            t: (batch,) or scalar, current time
            
        Returns:
            lambda_t: (batch, 1) predicted jump intensity
        """
        batch_size = x_seq.size(0)
        
        # Encode sequence with LSTM
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        h_t = h_n[-1]  # Last layer hidden state: (batch, hidden_dim)
        
        # Time embedding
        if isinstance(t, (int, float)):
            t = torch.full((batch_size,), t, device=x_seq.device, dtype=x_seq.dtype)
        t_emb = self.time_embed(t)  # (batch, hidden_dim)
        
        # Concatenate hidden state and time embedding
        combined = torch.cat([h_t, t_emb], dim=-1)  # (batch, hidden_dim * 2)
        
        # Predict log intensity
        log_lambda = self.intensity_head(combined)  # (batch, 1)
        
        # Clamp for numerical stability
        log_lambda = log_lambda.clamp(self.min_log_lambda, self.max_log_lambda)
        
        # Convert to intensity
        lambda_t = torch.exp(log_lambda)
        
        return lambda_t


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for continuous time representation."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)
        sinusoid_inp = torch.outer(t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return emb


class NeuralMJDParameters(nn.Module):
    """
    Neural network that predicts full MJD parameters.
    
    Outputs for each time step:
    - μ: drift
    - σ: diffusion volatility  
    - λ: jump intensity (time-varying)
    - ν: mean jump size
    - γ: jump size volatility
    
    Based on DSL-Lab/neural-MJD architecture.
    """
    def __init__(self, input_dim, hidden_dim=64, n_future=1, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_future = n_future
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Parameter prediction heads (5 parameters per future step)
        # Output dimensions: (μ, σ, log_λ, ν, γ) for each asset
        self.param_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 5 * input_dim * n_future)
        )
        
        # Parameter bounds (from neural-MJD)
        self.bound_mus = 10.0
        self.bound_variance = 1.0
        self.bound_lambdas = 1.0
        self.bound_nus = 0.5
        
    def forward(self, x_seq):
        """
        Predict MJD parameters from historical sequence.
        
        Args:
            x_seq: (batch, seq_len, input_dim) historical observations
            
        Returns:
            params: dict with keys 'mu', 'sigma', 'lambda', 'nu', 'gamma'
                   each of shape (batch, n_future, input_dim)
        """
        batch_size = x_seq.size(0)
        
        # Encode
        _, (h_n, _) = self.encoder(x_seq)
        h = h_n[-1]  # (batch, hidden_dim)
        
        # Predict raw parameters
        raw_params = self.param_head(h)  # (batch, 5 * input_dim * n_future)
        raw_params = raw_params.view(batch_size, self.n_future, self.input_dim, 5)
        
        # Split and constrain parameters
        mus = raw_params[..., 0].clamp(-self.bound_mus, self.bound_mus)
        sigmas = raw_params[..., 1].clamp(1e-6, self.bound_variance)
        log_lambdas = raw_params[..., 2].clamp(max=math.log(self.bound_lambdas))
        lambdas = torch.exp(log_lambdas).clamp(1e-6, self.bound_lambdas)
        nus = raw_params[..., 3].clamp(-self.bound_nus, self.bound_nus)
        gammas = raw_params[..., 4].clamp(1e-6, self.bound_variance)
        
        return {
            'mu': mus,
            'sigma': sigmas,
            'lambda': lambdas,
            'nu': nus,
            'gamma': gammas
        }


class NeuralJumpDetector:
    """
    Neural Jump Detector with learned time-varying intensity.
    
    Extends the static JumpDetector to support "Endogenous Jumps" where
    λ(t) is predicted by a neural network based on historical data.
    
    Usage:
        detector = NeuralJumpDetector(dt=1/252, input_dim=5)
        detector.fit(trajectories)  # Train intensity network
        lambda_t = detector.get_intensity(x_history, t)  # Get current intensity
        jump = detector.sample_jump(batch_size, x_history, t)  # Sample with learned λ
    """
    def __init__(self, dt, input_dim, hidden_dim=64, threshold_multiplier=3.0,
                 lr=0.001, epochs=50, device=None):
        self.dt = dt
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.c = threshold_multiplier
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device or "cuda")
        
        # Static jump parameters (calibrated from data)
        self.static_params = {
            "lambda": 0.0,
            "mu": 0.0,
            "sigma": 0.0
        }
        
        # Neural intensity network
        self.intensity_net = IntensityNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1
        ).to(self.device)
        
        self.is_fitted = False
        self._use_neural = True
        
    def fit(self, trajectories, seq_len=10):
        """
        Fit the neural jump detector.
        
        1. Detect jumps using threshold method (for labels)
        2. Train intensity network to predict jump occurrence
        
        Args:
            trajectories: (N, T, D) array of paths
            seq_len: Length of historical sequence for intensity prediction
        """
        N, T, D = trajectories.shape
        
        # 1. Static calibration (same as JumpDetector)
        flat_returns = np.diff(trajectories, axis=1).flatten()
        abs_ret = np.abs(flat_returns)
        sigma_hat = np.median(abs_ret) / 0.6745
        threshold = self.c * sigma_hat * np.sqrt(self.dt)
        
        jump_mask = np.abs(flat_returns) > threshold
        detected_jumps = flat_returns[jump_mask]
        
        n_jumps = len(detected_jumps)
        total_time = len(flat_returns) * self.dt
        
        self.static_params["lambda"] = n_jumps / total_time if total_time > 0 else 0
        self.static_params["mu"] = np.mean(detected_jumps) if n_jumps > 0 else 0
        self.static_params["sigma"] = np.std(detected_jumps) if n_jumps > 0 else 0
        
        print(f"   [NeuralJumps] Static calibration: λ={self.static_params['lambda']:.4f}")
        
        # 2. Prepare training data for intensity network
        # Create sequences and labels (1 if jump occurred, 0 otherwise)
        returns = np.diff(trajectories, axis=1)  # (N, T-1, D)
        jump_labels = (np.abs(returns) > threshold).any(axis=-1).astype(np.float32)  # (N, T-1)
        
        X_seqs = []
        y_labels = []
        
        for i in range(N):
            for t in range(seq_len, T-1):
                X_seqs.append(trajectories[i, t-seq_len:t, :])
                y_labels.append(jump_labels[i, t])
        
        X_seqs = np.array(X_seqs)  # (n_samples, seq_len, D)
        y_labels = np.array(y_labels)  # (n_samples,)
        
        # 3. Train intensity network
        X_tensor = torch.FloatTensor(X_seqs).to(self.device)
        y_tensor = torch.FloatTensor(y_labels).unsqueeze(-1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        optimizer = optim.Adam(self.intensity_net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        self.intensity_net.train()
        print(f"   [NeuralJumps] Training intensity network...")
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Predict intensity (probability of jump)
                t_dummy = torch.zeros(batch_x.size(0), device=self.device)
                pred_lambda = self.intensity_net(batch_x, t_dummy)
                
                # Binary cross-entropy loss
                loss = criterion(pred_lambda, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        self.intensity_net.eval()
        self.is_fitted = True
        print(f"   [NeuralJumps] Training complete. Final loss: {total_loss/len(dataloader):.4f}")
        
        return self
    
    def get_intensity(self, x_history, t):
        """
        Get predicted jump intensity λ(t) given historical data.
        
        Args:
            x_history: (batch, seq_len, D) historical observations
            t: current time
            
        Returns:
            lambda_t: (batch,) predicted intensity
        """
        if not self._use_neural:
            return self.static_params["lambda"]
        
        if not self.is_fitted:
            return self.static_params["lambda"]
        
        with torch.no_grad():
            if isinstance(x_history, np.ndarray):
                x_history = torch.FloatTensor(x_history).to(self.device)
            
            if x_history.dim() == 2:
                x_history = x_history.unsqueeze(0)
            
            lambda_t = self.intensity_net(x_history, t)
            return lambda_t.squeeze(-1).cpu().numpy()
    
    def sample_jump(self, batch_size, x_history=None, t=0):
        """
        Sample jumps using neural intensity (if available) or static intensity.
        
        Args:
            batch_size: number of samples
            x_history: (batch, seq_len, D) historical data for neural intensity
            t: current time
            
        Returns:
            jump_sizes: (batch_size,) array of jump sizes
        """
        # Get intensity
        if x_history is not None and self._use_neural and self.is_fitted:
            lambda_t = self.get_intensity(x_history, t)
            if isinstance(lambda_t, np.ndarray):
                prob_jump = lambda_t * self.dt
            else:
                prob_jump = np.full(batch_size, lambda_t * self.dt)
        else:
            prob_jump = np.full(batch_size, self.static_params["lambda"] * self.dt)
        
        # Sample jump occurrence
        jump_mask = np.random.random(batch_size) < prob_jump
        
        if not np.any(jump_mask):
            return np.zeros(batch_size)
        
        # Sample jump sizes from N(μ, σ)
        jump_sizes = np.random.normal(
            self.static_params["mu"],
            max(self.static_params["sigma"], 1e-6),
            size=batch_size
        )
        
        return jump_sizes * jump_mask
    
    def set_neural_mode(self, use_neural):
        """Enable or disable neural intensity prediction."""
        self._use_neural = use_neural
        
    def get_static_params(self):
        """Return static MJD parameters."""
        return self.static_params.copy()


class NeuralPointProcess:
    """
    Alias for NeuralJumpDetector for compatibility with prompt requirements.
    
    This class provides the same functionality as NeuralJumpDetector
    but with a name that matches the prompt specification.
    """
    def __init__(self, *args, **kwargs):
        self._detector = NeuralJumpDetector(*args, **kwargs)
        
    def __getattr__(self, name):
        return getattr(self._detector, name)
    
    def fit(self, *args, **kwargs):
        return self._detector.fit(*args, **kwargs)
    
    def sample_jump(self, *args, **kwargs):
        return self._detector.sample_jump(*args, **kwargs)
    
    def get_intensity(self, *args, **kwargs):
        return self._detector.get_intensity(*args, **kwargs)
