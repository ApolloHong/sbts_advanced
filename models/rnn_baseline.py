"""
RNN Baseline for Time Series Generation.

Provides a compact autoregressive RNN baseline that can be trained on the
same sliding-window data used by the other models in this repository.

Training:
    - next-step prediction with teacher forcing
    - MSE loss on normalized windows

Generation:
    - sample a real prefix from the training windows (or use x0)
    - roll out future steps autoregressively
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


if TORCH_AVAILABLE:

    class RNNForecastNet(nn.Module):
        """Simple autoregressive recurrent predictor."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
            bidirectional: bool = False,
            rnn_type: str = "lstm",
        ):
            super().__init__()

            rnn_type = rnn_type.lower()
            rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}.get(rnn_type)
            if rnn_cls is None:
                raise ValueError(f"Unsupported rnn_type: {rnn_type}")

            self.rnn_type = rnn_type
            self.rnn = rnn_cls(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=bidirectional,
            )
            output_dim = hidden_dim * (2 if bidirectional else 1)
            self.readout = nn.Linear(output_dim, input_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.rnn(x)
            return self.readout(h)


class RNNBaseline(TimeSeriesGenerator):
    """
    Minimal autoregressive RNN baseline.

    The model is trained on windows with a one-step-ahead objective:
        x_t -> x_{t+1}

    During generation it uses a real prefix sampled from the training set
    and produces the remaining steps autoregressively.
    """

    MODEL_TYPE = "rnn"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for RNNBaseline")

        self.hidden_dim = config.get("rnn_hidden_dim", 64)
        self.num_layers = config.get("rnn_num_layers", 2)
        self.dropout = config.get("rnn_dropout", 0.1)
        self.bidirectional = config.get("rnn_bidirectional", False)
        self.rnn_type = config.get("rnn_type", "lstm")
        self.epochs = config.get("rnn_epochs", 50)
        self.lr = config.get("rnn_lr", 1e-3)
        self.batch_size = config.get("rnn_batch_size", 128)
        self.weight_decay = config.get("rnn_weight_decay", 0.0)
        self.context_len = config.get("rnn_context_len")

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
        """Convert univariate data to (N, T, 1)."""
        if data.ndim == 2:
            return data[:, :, np.newaxis]
        return data

    def _resolve_context_len(self, seq_len: int) -> int:
        """Pick a safe context length for autoregressive rollout."""
        context_len = self.context_len or max(5, seq_len // 2)
        return max(1, min(context_len, seq_len - 1))

    def fit(
        self,
        data: np.ndarray,
        time_grid: np.ndarray,
        verbose: bool = True,
    ) -> "RNNBaseline":
        """
        Train the RNN baseline on sliding-window data.

        Args:
            data: Windowed data of shape (n_samples, seq_len, n_features)
            time_grid: Unused, kept for API compatibility
            verbose: Whether to print progress
        """
        del time_grid

        data = self._ensure_3d(data)
        n_samples, seq_len, n_features = data.shape
        if seq_len < 2:
            raise ValueError("RNNBaseline requires seq_len >= 2")

        self.seq_len = seq_len
        self.n_features = n_features
        self.context_len = self._resolve_context_len(seq_len)

        self.data_mean = data.mean(axis=(0, 1), keepdims=True)
        self.data_std = data.std(axis=(0, 1), keepdims=True) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std

        self.prefix_bank = data_norm[:, :self.context_len, :].copy()

        inputs = torch.tensor(data_norm[:, :-1, :], dtype=torch.float32)
        targets = torch.tensor(data_norm[:, 1:, :], dtype=torch.float32)

        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = RNNForecastNet(
            input_dim=n_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            rnn_type=self.rnn_type,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        if verbose:
            print("=" * 60)
            print("RNN Baseline Training")
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
                    f"  [RNN] Epoch {epoch + 1}/{self.epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )

        self.model.eval()
        self.is_fitted = True

        if verbose:
            print("=" * 60)
            print("RNN Baseline Training Complete!")
            print("=" * 60)

        return self

    def _prepare_seed(
        self,
        n_samples: int,
        n_steps: int,
        x0: Optional[np.ndarray],
    ) -> np.ndarray:
        """Prepare normalized prefix windows for rollout."""
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
            raise RuntimeError("RNNBaseline must be fitted before generation")

        n_steps = n_steps or self.seq_len
        if n_steps < 1:
            raise ValueError("n_steps must be positive")

        current = self._prepare_seed(n_samples, n_steps, x0)
        self.model.eval()

        while current.shape[1] < n_steps:
            model_input = torch.tensor(
                current[:, -self.context_len :, :], dtype=torch.float32, device=self.device
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
