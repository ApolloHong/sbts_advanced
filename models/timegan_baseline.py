"""
TimeGAN baseline model.

This module implements the GRU-based TimeGAN training recipe from
Yoon et al. (NeurIPS 2019) behind the repository's common generator
interface.
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

    class EmbeddingNetwork(nn.Module):
        """Embed real sequences into the TimeGAN latent space."""

        def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h, _ = self.rnn(x)
            return torch.sigmoid(self.fc(h))


    class RecoveryNetwork(nn.Module):
        """Recover data-space sequences from latent states."""

        def __init__(
            self,
            hidden_dim: int,
            output_dim: int,
            n_layers: int = 2,
            output_activation: str = "linear",
        ):
            super().__init__()
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.output_activation = output_activation

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(h)
            out = self.fc(out)
            if self.output_activation == "sigmoid":
                return torch.sigmoid(out)
            return out


    class GeneratorNetwork(nn.Module):
        """Map random noise sequences into latent states."""

        def __init__(self, z_dim: int, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            self.rnn = nn.GRU(
                input_size=z_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            h, _ = self.rnn(z)
            return torch.sigmoid(self.fc(h))


    class SupervisorNetwork(nn.Module):
        """Learn one-step latent dynamics."""

        def __init__(self, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=max(n_layers - 1, 1),
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(h)
            return torch.sigmoid(self.fc(out))


    class DiscriminatorNetwork(nn.Module):
        """Classify real and generated latent trajectories."""

        def __init__(self, hidden_dim: int, n_layers: int = 2):
            super().__init__()
            self.rnn = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(h)
            return self.fc(out)


class TimeGAN(TimeSeriesGenerator):
    """GRU TimeGAN baseline used by the active experiment pipeline."""

    MODEL_TYPE = "timegan"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TimeGAN")

        self.hidden_dim = config.get("timegan_hidden_dim", 64)
        self.z_dim = config.get("timegan_z_dim", 32)
        self.n_layers = config.get("timegan_n_layers", 2)
        if self.n_layers < 2:
            raise ValueError("timegan_n_layers must be >= 2")

        self.epochs = config.get("timegan_epochs", 50)
        self.lr = config.get("timegan_lr", 0.001)
        self.batch_size = config.get("timegan_batch_size", 128)
        self.normalization = config.get("timegan_normalization", "standard")
        if self.normalization not in {"standard", "minmax"}:
            raise ValueError("timegan_normalization must be 'standard' or 'minmax'")
        self.gamma = config.get("timegan_gamma", 1.0)
        self.moment_loss_weight = config.get("timegan_moment_loss_weight", 100.0)
        self.supervised_loss_weight = config.get("timegan_supervised_loss_weight", 100.0)
        self.reconstruction_loss_weight = config.get(
            "timegan_reconstruction_loss_weight", 10.0
        )
        self.embedding_supervised_weight = config.get(
            "timegan_embedding_supervised_weight", 0.1
        )
        self.discriminator_threshold = config.get("timegan_discriminator_threshold", 0.15)
        self.generator_steps = config.get("timegan_generator_steps", 2)
        self.clip_output = config.get("timegan_clip_output", True)

        self.device = torch.device(config.get("device", "cuda"))

        self.embedder = None
        self.recovery = None
        self.generator = None
        self.supervisor = None
        self.discriminator = None

        self.n_features = None
        self.seq_len = None
        self.data_mean = None
        self.data_std = None
        self.data_min = None
        self.data_max = None
        self.data_scale = None

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        if self.normalization == "minmax":
            return (data - self.data_min) / self.data_scale
        return (data - self.data_mean) / self.data_std

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        if self.normalization == "minmax":
            return data * self.data_scale + self.data_min
        return data * self.data_std + self.data_mean

    @staticmethod
    def _sqrt_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.mse_loss(pred, target) + 1e-8)

    @staticmethod
    def _moment_loss(x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        fake_mean = x_fake.mean(dim=(0, 1))
        real_mean = x_real.mean(dim=(0, 1))
        fake_std = torch.sqrt(x_fake.var(dim=(0, 1), unbiased=False) + 1e-6)
        real_std = torch.sqrt(x_real.var(dim=(0, 1), unbiased=False) + 1e-6)
        mean_loss = torch.mean(torch.abs(fake_mean - real_mean))
        std_loss = torch.mean(torch.abs(fake_std - real_std))
        return mean_loss + std_loss

    def _sample_noise(self, batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.rand(batch_size, seq_len, self.z_dim, device=self.device)

    def _iter_batches(self, x: torch.Tensor) -> torch.Tensor:
        n_samples = x.shape[0]
        perm = torch.randperm(n_samples, device=self.device)
        for start in range(0, n_samples, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield x[idx]

    def _embedder_loss(self, batch: torch.Tensor) -> torch.Tensor:
        h = self.embedder(batch)
        x_tilde = self.recovery(h)
        return self._sqrt_mse(x_tilde, batch)

    def _supervised_loss(self, h: torch.Tensor) -> torch.Tensor:
        h_hat_supervise = self.supervisor(h)
        return F.mse_loss(h_hat_supervise[:, :-1, :], h[:, 1:, :])

    def fit(
        self,
        data: np.ndarray,
        time_grid: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "TimeGAN":
        """Fit TimeGAN on windowed time-series data."""
        del time_grid

        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        data = data.astype(np.float32, copy=False)

        n_samples, seq_len, n_features = data.shape
        self.n_features = n_features
        self.seq_len = seq_len

        if self.normalization == "minmax":
            self.data_min = data.min(axis=(0, 1), keepdims=True)
            self.data_max = data.max(axis=(0, 1), keepdims=True)
            self.data_scale = np.maximum(self.data_max - self.data_min, 1e-6)
            recovery_activation = "sigmoid"
        else:
            self.data_mean = data.mean(axis=(0, 1), keepdims=True)
            self.data_std = data.std(axis=(0, 1), keepdims=True) + 1e-8
            recovery_activation = "linear"
        data_norm = self._normalize(data)
        X = torch.tensor(data_norm, dtype=torch.float32, device=self.device)

        self.embedder = EmbeddingNetwork(n_features, self.hidden_dim, self.n_layers).to(
            self.device
        )
        self.recovery = RecoveryNetwork(
            self.hidden_dim,
            n_features,
            self.n_layers,
            output_activation=recovery_activation,
        ).to(self.device)
        self.generator = GeneratorNetwork(self.z_dim, self.hidden_dim, self.n_layers).to(
            self.device
        )
        self.supervisor = SupervisorNetwork(self.hidden_dim, self.n_layers).to(self.device)
        self.discriminator = DiscriminatorNetwork(self.hidden_dim, self.n_layers).to(
            self.device
        )

        if verbose:
            print("=" * 60)
            print("TimeGAN Training")
            print("=" * 60)
            print("[Phase 1] Embedding Network Training...")

        ae_optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.lr,
        )

        ae_losses = []
        for epoch in range(self.epochs):
            epoch_losses = []
            for batch in self._iter_batches(X):
                ae_optimizer.zero_grad()
                e_loss_t0 = self._embedder_loss(batch)
                e_loss = self.reconstruction_loss_weight * e_loss_t0
                e_loss.backward()
                ae_optimizer.step()
                epoch_losses.append(e_loss_t0.item())

            ae_losses.append(float(np.mean(epoch_losses)))
            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{self.epochs}, E Loss: {ae_losses[-1]:.6f}")

        if verbose:
            print("\n[Phase 2] Supervised Latent Dynamics Training...")

        sup_optimizer = torch.optim.Adam(self.supervisor.parameters(), lr=self.lr)

        sup_losses = []
        for epoch in range(self.epochs):
            epoch_losses = []
            for batch in self._iter_batches(X):
                sup_optimizer.zero_grad()
                with torch.no_grad():
                    h = self.embedder(batch)
                g_loss_s = self._supervised_loss(h)
                g_loss_s.backward()
                sup_optimizer.step()
                epoch_losses.append(g_loss_s.item())

            sup_losses.append(float(np.mean(epoch_losses)))
            if verbose and (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch + 1}/{self.epochs}, S Loss: {sup_losses[-1]:.6f}")

        if verbose:
            print("\n[Phase 3] Joint Adversarial Training...")

        g_optimizer = torch.optim.Adam(
            list(self.generator.parameters()) + list(self.supervisor.parameters()),
            lr=self.lr,
        )
        e_optimizer = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=self.lr,
        )
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        g_losses = []
        d_losses = []
        e_losses = []
        for epoch in range(self.epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            epoch_e_losses = []

            for batch in self._iter_batches(X):
                batch_size = batch.shape[0]

                for _ in range(self.generator_steps):
                    g_optimizer.zero_grad()

                    with torch.no_grad():
                        h = self.embedder(batch)
                    h_supervise = self.supervisor(h)
                    z = self._sample_noise(batch_size, seq_len)
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)
                    x_hat = self.recovery(h_hat)

                    y_fake = self.discriminator(h_hat)
                    y_fake_e = self.discriminator(e_hat)
                    g_loss_u = F.binary_cross_entropy_with_logits(
                        y_fake, torch.ones_like(y_fake)
                    )
                    g_loss_u_e = F.binary_cross_entropy_with_logits(
                        y_fake_e, torch.ones_like(y_fake_e)
                    )
                    g_loss_s = F.mse_loss(h_supervise[:, :-1, :], h[:, 1:, :])
                    g_loss_v = self._moment_loss(x_hat, batch)

                    g_loss = (
                        g_loss_u
                        + self.gamma * g_loss_u_e
                        + self.supervised_loss_weight * torch.sqrt(g_loss_s + 1e-8)
                        + self.moment_loss_weight * g_loss_v
                    )
                    g_loss.backward()
                    g_optimizer.step()
                    epoch_g_losses.append(g_loss.item())

                    e_optimizer.zero_grad()
                    h = self.embedder(batch)
                    x_tilde = self.recovery(h)
                    h_supervise = self.supervisor(h)
                    e_loss_t0 = self._sqrt_mse(x_tilde, batch)
                    g_loss_s = F.mse_loss(h_supervise[:, :-1, :], h[:, 1:, :])
                    e_loss = (
                        self.reconstruction_loss_weight * e_loss_t0
                        + self.embedding_supervised_weight * g_loss_s
                    )
                    e_loss.backward()
                    e_optimizer.step()
                    epoch_e_losses.append(e_loss.item())

                d_optimizer.zero_grad()
                with torch.no_grad():
                    h = self.embedder(batch)
                    z = self._sample_noise(batch_size, seq_len)
                    e_hat = self.generator(z)
                    h_hat = self.supervisor(e_hat)

                y_real = self.discriminator(h.detach())
                y_fake = self.discriminator(h_hat.detach())
                y_fake_e = self.discriminator(e_hat.detach())
                d_loss_real = F.binary_cross_entropy_with_logits(
                    y_real, torch.ones_like(y_real)
                )
                d_loss_fake = F.binary_cross_entropy_with_logits(
                    y_fake, torch.zeros_like(y_fake)
                )
                d_loss_fake_e = F.binary_cross_entropy_with_logits(
                    y_fake_e, torch.zeros_like(y_fake_e)
                )
                d_loss = d_loss_real + d_loss_fake + self.gamma * d_loss_fake_e

                if d_loss.item() > self.discriminator_threshold:
                    d_loss.backward()
                    d_optimizer.step()
                epoch_d_losses.append(d_loss.item())

            g_losses.append(float(np.mean(epoch_g_losses)))
            d_losses.append(float(np.mean(epoch_d_losses)))
            e_losses.append(float(np.mean(epoch_e_losses)))
            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"    Epoch {epoch + 1}/{self.epochs}, "
                    f"G Loss: {g_losses[-1]:.4f}, "
                    f"E Loss: {e_losses[-1]:.4f}, "
                    f"D Loss: {d_losses[-1]:.4f}"
                )

        self._training_metrics = {
            "timegan_embedding_loss": ae_losses[-1] if ae_losses else None,
            "timegan_supervised_loss": sup_losses[-1] if sup_losses else None,
            "timegan_generator_loss": g_losses[-1] if g_losses else None,
            "timegan_joint_embedding_loss": e_losses[-1] if e_losses else None,
            "timegan_discriminator_loss": d_losses[-1] if d_losses else None,
        }
        self.is_fitted = True

        if verbose:
            print("=" * 60)
            print("TimeGAN Training Complete!")
            print("=" * 60)

        return self

    def generate(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate synthetic time series samples."""
        del x0, kwargs

        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generation")

        if n_steps is None:
            n_steps = self.seq_len

        self.generator.eval()
        self.supervisor.eval()
        self.recovery.eval()

        with torch.no_grad():
            z = self._sample_noise(n_samples, n_steps)
            e_hat = self.generator(z)
            h_hat = self.supervisor(e_hat)
            x_fake = self.recovery(h_hat)

        x_fake = x_fake.cpu().numpy()
        if self.normalization == "minmax" and self.clip_output:
            x_fake = np.clip(x_fake, 0.0, 1.0)
        return self._denormalize(x_fake)
