"""
Autoregressive Transformer baseline for time series generation.

This module implements a compact decoder-only / causal Transformer that
performs next-step prediction on sliding windows and rolls out future steps
autoregressively during generation.
"""

from typing import Any, Dict, Optional

import numpy as np

from models.base import GenerativeModel as TimeSeriesGenerator

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _build_sinusoidal_encoding(max_seq_len: int, d_model: int) -> torch.Tensor:
    """Create standard sinusoidal positional encodings."""
    positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-np.log(10000.0) / d_model)
    )

    pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe.unsqueeze(0)


if TORCH_AVAILABLE:

    class CausalTransformerForecaster(nn.Module):
        """Decoder-only Transformer for continuous autoregressive prediction."""

        def __init__(
            self,
            input_dim: int,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
            d_ff: int = 128,
            dropout: float = 0.1,
            max_seq_len: int = 128,
        ):
            super().__init__()

            self.max_seq_len = max_seq_len
            self.input_proj = nn.Linear(input_dim, d_model)
            self.output_proj = nn.Linear(d_model, input_dim)
            self.dropout = nn.Dropout(dropout)

            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)

            positional_encoding = _build_sinusoidal_encoding(max_seq_len, d_model)
            self.register_buffer("positional_encoding", positional_encoding, persistent=False)

        def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
            """Upper-triangular mask for causal attention."""
            return torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device),
                diagonal=1,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq_len = x.size(1)
            if seq_len > self.max_seq_len:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
                )

            h = self.input_proj(x)
            h = h + self.positional_encoding[:, :seq_len, :].to(x.device)
            h = self.dropout(h)
            h = self.encoder(h, mask=self._causal_mask(seq_len, x.device))
            h = self.norm(h)
            return self.output_proj(h)


class TransformerARBaseline(TimeSeriesGenerator):
    """
    Causal autoregressive Transformer baseline.

    Training uses teacher-forced next-step prediction on windowed data and
    generation rolls out one step at a time from a real prefix.
    """

    MODEL_TYPE = "transformer_ar"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TransformerARBaseline")

        self.d_model = config.get("transformer_ar_d_model", 64)
        self.n_heads = config.get("transformer_ar_n_heads", 4)
        self.n_layers = config.get("transformer_ar_n_layers", 2)
        self.d_ff = config.get("transformer_ar_d_ff", 128)
        self.dropout = config.get("transformer_ar_dropout", 0.1)
        self.max_seq_len = config.get("transformer_ar_max_seq_len")
        self.context_len = config.get("transformer_ar_context_len")
        self.epochs = config.get("transformer_ar_epochs", 50)
        self.lr = config.get("transformer_ar_lr", 1e-3)
        self.batch_size = config.get("transformer_ar_batch_size", 128)
        self.weight_decay = config.get("transformer_ar_weight_decay", 0.0)

        if self.d_model % self.n_heads != 0:
            raise ValueError("transformer_ar_d_model must be divisible by transformer_ar_n_heads")

        self.device = torch.device(config.get("device", "cuda"))

        self.model = None
        self.seq_len = None
        self.n_features = None
        self.data_mean = None
        self.data_std = None
        self.prefix_bank = None
        self.train_losses = []

    @staticmethod
    def _ensure_3d(data: np.ndarray) -> np.ndarray:
        """Convert univariate arrays to (N, T, 1)."""
        if data.ndim == 2:
            return data[:, :, np.newaxis]
        return data

    def _resolve_context_len(self, seq_len: int) -> int:
        """Pick a safe prefix length for generation."""
        context_len = self.context_len or max(5, seq_len // 2)
        return max(1, min(context_len, seq_len - 1))

    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True,
    ) -> "TransformerARBaseline":
        """
        Train the causal Transformer baseline.

        Args:
            data: Windowed data of shape (n_samples, seq_len, n_features)
            time_grid: Unused, kept for API compatibility
            verbose: Whether to print progress
        """
        del time_grid

        data = self._ensure_3d(data)
        n_samples, seq_len, n_features = data.shape
        if seq_len < 2:
            raise ValueError("TransformerARBaseline requires seq_len >= 2")

        self.seq_len = seq_len
        self.n_features = n_features
        self.context_len = self._resolve_context_len(seq_len)
        self.max_seq_len = self.max_seq_len or max(seq_len, 64)

        if self.max_seq_len < seq_len - 1:
            raise ValueError("transformer_ar_max_seq_len must be at least seq_len - 1")

        self.data_mean = data.mean(axis=(0, 1), keepdims=True)
        self.data_std = data.std(axis=(0, 1), keepdims=True) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std

        self.prefix_bank = data_norm[:, :self.context_len, :].copy()

        inputs = torch.tensor(data_norm[:, :-1, :], dtype=torch.float32)
        targets = torch.tensor(data_norm[:, 1:, :], dtype=torch.float32)

        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = CausalTransformerForecaster(
            input_dim=n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        if verbose:
            print("=" * 60)
            print("Transformer AR Baseline Training")
            print("=" * 60)

        self.train_losses = []
        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0

            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_inputs)
                loss = criterion(predictions, batch_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"  [Transformer-AR] Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )

        self.model.eval()
        self.is_fitted = True

        if verbose:
            print("=" * 60)
            print("Transformer AR Baseline Training Complete!")
            print("=" * 60)

        return self

    def _prepare_seed(
        self,
        n_samples: int,
        n_steps: int,
        x0: Optional[np.ndarray],
    ) -> np.ndarray:
        """Prepare normalized prefixes for autoregressive decoding."""
        if x0 is None:
            idx = np.random.choice(len(self.prefix_bank), size=n_samples, replace=True)
            seed = self.prefix_bank[idx]
        else:
            seed = np.asarray(x0, dtype=np.float32)
            if seed.ndim == 2:
                seed = seed[np.newaxis, :, :]
            if seed.ndim != 3:
                raise ValueError("x0 must have shape (context_len, n_features) or (n_samples, context_len, n_features)")
            if seed.shape[-1] != self.n_features:
                raise ValueError("x0 feature dimension does not match training data")
            if seed.shape[0] == 1 and n_samples > 1:
                seed = np.repeat(seed, n_samples, axis=0)
            elif seed.shape[0] != n_samples:
                raise ValueError("x0 batch size must be 1 or match n_samples")
            seed = (seed - self.data_mean) / self.data_std

        return seed[:, : max(1, min(seed.shape[1], n_steps)), :]

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generate windows autoregressively from a seed prefix.

        Args:
            n_samples: Number of windows to generate
            n_steps: Total output length. Defaults to training seq_len
            x0: Optional seed prefix in original scale
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("TransformerARBaseline must be fitted before generation")

        n_steps = n_steps or self.seq_len
        if n_steps < 1:
            raise ValueError("n_steps must be positive")

        current = self._prepare_seed(n_samples, n_steps, x0)
        self.model.eval()

        while current.shape[1] < n_steps:
            model_input = torch.tensor(
                current[:, -self.max_seq_len :, :], dtype=torch.float32, device=self.device
            )
            predictions = self.model(model_input)
            next_step = predictions[:, -1:, :].cpu().numpy()
            current = np.concatenate([current, next_step], axis=1)

        generated = current[:, :n_steps, :] * self.data_std + self.data_mean
        return generated

    def sample(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Alias for generate()."""
        return self.generate(n_samples=n_samples, n_steps=n_steps, x0=x0)
