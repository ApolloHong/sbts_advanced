"""
Diffusion-TS baseline model.

This file contains the active diffusion baseline used by the current
`main.py` experiment pipeline.
"""

from typing import Any, Dict, Optional

import numpy as np

from models.base import GenerativeModel as TimeSeriesGenerator

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DiffusionUNet(nn.Module):
        """Simple MLP-style denoiser for diffusion over time-series windows."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            time_embed_dim: int = 64,
        ):
            super().__init__()

            self.time_embed = nn.Sequential(
                nn.Linear(1, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )
            self.enc1 = nn.Linear(input_dim + time_embed_dim, hidden_dim)
            self.enc2 = nn.Linear(hidden_dim, hidden_dim * 2)
            self.dec1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.dec2 = nn.Linear(hidden_dim * 2, input_dim)
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            batch, seq_len, _ = x.shape

            if t.dim() == 1:
                t = t.unsqueeze(-1)

            t_embed = self.time_embed(t)
            t_embed = t_embed.unsqueeze(1).expand(-1, seq_len, -1)

            h = torch.cat([x, t_embed], dim=-1)
            h1 = self.act(self.enc1(h))
            h2 = self.act(self.enc2(h1))
            h3 = self.act(self.dec1(h2))
            h4 = torch.cat([h3, h1], dim=-1)
            return self.dec2(h4)


class DiffusionTS(TimeSeriesGenerator):
    """Simplified DDPM-style diffusion baseline for time-series windows."""

    MODEL_TYPE = "diffusion_ts"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for Diffusion-TS")

        self.hidden_dim = config.get("diffusion_hidden_dim", 128)
        self.n_diffusion_steps = config.get("diffusion_n_steps", 100)
        self.epochs = config.get("diffusion_epochs", 100)
        self.lr = config.get("diffusion_lr", 0.001)
        self.batch_size = config.get("diffusion_batch_size", 64)
        self.beta_start = config.get("diffusion_beta_start", 0.0001)
        self.beta_end = config.get("diffusion_beta_end", 0.02)

        self.device = torch.device(config.get("device", "cuda"))

        self.betas = None
        self.alphas = None
        self.alpha_bars = None

        self.model = None
        self.n_features = None
        self.seq_len = None
        self.data_mean = None
        self.data_std = None

    def _setup_noise_schedule(self) -> None:
        """Setup a linear noise schedule."""
        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.n_diffusion_steps,
            device=self.device,
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def _q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion step."""
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar = self.alpha_bars[t]
        while alpha_bar.dim() < x0.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)

        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    def fit(
        self,
        data: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "DiffusionTS":
        """Fit the diffusion baseline on sliding-window data."""
        del time_grid

        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.seq_len = seq_len

        self.data_mean = np.mean(data)
        self.data_std = np.std(data) + 1e-8
        data_norm = (data - self.data_mean) / self.data_std
        X = torch.tensor(data_norm, dtype=torch.float32).to(self.device)

        self._setup_noise_schedule()
        self.model = DiffusionUNet(
            input_dim=n_features,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if verbose:
            print("=" * 60)
            print("Diffusion-TS Training")
            print("=" * 60)

        for epoch in range(self.epochs):
            perm = torch.randperm(n_samples)
            X = X[perm]

            total_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch = X[i : i + self.batch_size]
                batch_size = len(batch)

                optimizer.zero_grad()
                t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=self.device)
                noise = torch.randn_like(batch)
                x_t = self._q_sample(batch, t, noise)
                t_normalized = t.float() / self.n_diffusion_steps
                pred_noise = self.model(x_t, t_normalized)
                loss = F.mse_loss(pred_noise, noise)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 20 == 0:
                print(f"  [Diffusion-TS] Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / n_batches:.6f}")

        self.model.eval()
        self.is_fitted = True

        if verbose:
            print("=" * 60)
            print("Diffusion-TS Training Complete!")
            print("=" * 60)

        return self

    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate synthetic windows via reverse diffusion."""
        del x0, kwargs

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")

        if n_steps is None:
            n_steps = self.seq_len

        self.model.eval()
        x = torch.randn(n_samples, n_steps, self.n_features, device=self.device)

        with torch.no_grad():
            for t in reversed(range(self.n_diffusion_steps)):
                t_batch = torch.full((n_samples,), t, device=self.device)
                t_normalized = t_batch.float() / self.n_diffusion_steps
                pred_noise = self.model(x, t_normalized)

                alpha = self.alphas[t]
                alpha_bar = self.alpha_bars[t]

                if t > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(self.betas[t])
                else:
                    noise = 0
                    sigma = 0

                x = (1 / torch.sqrt(alpha)) * (
                    x - (self.betas[t] / torch.sqrt(1 - alpha_bar)) * pred_noise
                ) + sigma * noise

        x = x.cpu().numpy()
        x = x * self.data_std + self.data_mean
        return x
