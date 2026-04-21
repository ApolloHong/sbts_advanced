"""
Feedback Mechanism Module

Implements the Transient Stress Factor dynamics for JD-SBTS-F.

The feedback mechanism captures "Volatility Clustering" by introducing
a Shot Noise process that amplifies volatility after jumps.

Mathematical Model:
    dS_t = -κ * S_t * dt + γ * |dJ_t|
    σ_eff(t, x) = σ_LV(t, x) * √(1 + S_t)

Author: Manus AI
"""

import numpy as np
from typing import Callable, Dict, Any, Optional, Tuple, Union
from numba import njit, prange
from dataclasses import dataclass


@dataclass
class FeedbackConfig:
    """Configuration for feedback mechanism."""
    kappa: float = 0.8      # Mean reversion speed
    gamma: float = 1.0      # Jump impact multiplier
    s0: float = 0.0         # Initial stress level
    max_stress: float = 10.0  # Maximum stress cap (for stability)


@njit(cache=True)
def _update_stress_scalar(
    s_prev: float,
    dt: float,
    jump_size: float,
    kappa: float,
    gamma: float,
    max_stress: float = 10.0
) -> float:
    """
    Update stress factor for a single step.
    
    S_t = S_{t-1} * exp(-κ*dt) + γ * |dJ|
    
    Args:
        s_prev: Previous stress value
        dt: Time step
        jump_size: Absolute jump size
        kappa: Mean reversion speed
        gamma: Jump impact multiplier
        max_stress: Maximum stress cap
    
    Returns:
        Updated stress value
    """
    decay = np.exp(-kappa * dt)
    s_new = s_prev * decay + gamma * np.abs(jump_size)
    return min(s_new, max_stress)


@njit(cache=True, parallel=True)
def _simulate_stress_trajectory(
    jump_sizes: np.ndarray,
    time_grid: np.ndarray,
    kappa: float,
    gamma: float,
    s0: float = 0.0,
    max_stress: float = 10.0
) -> np.ndarray:
    """
    Simulate stress factor trajectory given jump sizes.
    
    Args:
        jump_sizes: Jump sizes (n_samples, n_steps)
        time_grid: Time points (n_steps,)
        kappa: Mean reversion speed
        gamma: Jump impact multiplier
        s0: Initial stress level
        max_stress: Maximum stress cap
    
    Returns:
        Stress trajectory (n_samples, n_steps)
    """
    n_samples, n_steps = jump_sizes.shape
    stress = np.zeros((n_samples, n_steps))
    
    for i in prange(n_samples):
        stress[i, 0] = s0
        
        for t in range(1, n_steps):
            dt = time_grid[t] - time_grid[t-1]
            stress[i, t] = _update_stress_scalar(
                stress[i, t-1],
                dt,
                jump_sizes[i, t-1],
                kappa,
                gamma,
                max_stress
            )
    
    return stress


@njit(cache=True)
def _compute_effective_volatility(
    base_vol: np.ndarray,
    stress: np.ndarray
) -> np.ndarray:
    """
    Compute effective volatility with stress factor.
    
    σ_eff = σ_base * √(1 + S)
    
    Args:
        base_vol: Base volatility (n_samples, n_steps, n_features)
        stress: Stress factor (n_samples, n_steps)
    
    Returns:
        Effective volatility (n_samples, n_steps, n_features)
    """
    n_samples, n_steps, n_features = base_vol.shape
    eff_vol = np.zeros_like(base_vol)
    
    for i in range(n_samples):
        for t in range(n_steps):
            multiplier = np.sqrt(1.0 + stress[i, t])
            for k in range(n_features):
                eff_vol[i, t, k] = base_vol[i, t, k] * multiplier
    
    return eff_vol


class StressFactor:
    """
    Transient Stress Factor for Jump-Volatility Feedback.
    
    Models the temporary increase in volatility following jumps,
    capturing the "Volatility Clustering" phenomenon.
    
    Usage:
        stress = StressFactor(kappa=0.8, gamma=1.0)
        stress_trajectory = stress.simulate(jump_sizes, time_grid)
        eff_vol = stress.apply_to_volatility(base_vol, stress_trajectory)
    """
    
    def __init__(
        self,
        kappa: float = 0.8,
        gamma: float = 1.0,
        s0: float = 0.0,
        max_stress: float = 10.0
    ):
        """
        Initialize stress factor.
        
        Args:
            kappa: Mean reversion speed (higher = faster decay)
            gamma: Jump impact multiplier (higher = stronger impact)
            s0: Initial stress level
            max_stress: Maximum stress cap for numerical stability
        """
        self.kappa = kappa
        self.gamma = gamma
        self.s0 = s0
        self.max_stress = max_stress
        
        # Derived quantities
        self.half_life = np.log(2) / kappa if kappa > 0 else np.inf
    
    def simulate(
        self,
        jump_sizes: np.ndarray,
        time_grid: np.ndarray
    ) -> np.ndarray:
        """
        Simulate stress factor trajectory.
        
        Args:
            jump_sizes: Jump sizes (n_samples, n_steps) or (n_samples, n_steps, n_features)
            time_grid: Time points (n_steps,)
        
        Returns:
            Stress trajectory (n_samples, n_steps)
        """
        # If multivariate, sum absolute jumps across features
        if jump_sizes.ndim == 3:
            jump_sizes_agg = np.sum(np.abs(jump_sizes), axis=-1)
        else:
            jump_sizes_agg = np.abs(jump_sizes)
        
        return _simulate_stress_trajectory(
            jump_sizes_agg.astype(np.float64),
            time_grid.astype(np.float64),
            self.kappa,
            self.gamma,
            self.s0,
            self.max_stress
        )
    
    def apply_to_volatility(
        self,
        base_vol: np.ndarray,
        stress: np.ndarray
    ) -> np.ndarray:
        """
        Apply stress factor to base volatility.
        
        Args:
            base_vol: Base volatility (n_samples, n_steps, n_features)
            stress: Stress factor (n_samples, n_steps)
        
        Returns:
            Effective volatility (n_samples, n_steps, n_features)
        """
        # Ensure 3D
        if base_vol.ndim == 2:
            base_vol = base_vol[:, :, np.newaxis]
        
        return _compute_effective_volatility(
            base_vol.astype(np.float64),
            stress.astype(np.float64)
        )
    
    def get_multiplier(self, stress: np.ndarray) -> np.ndarray:
        """
        Get volatility multiplier from stress.
        
        Args:
            stress: Stress values
        
        Returns:
            Volatility multiplier √(1 + S)
        """
        return np.sqrt(1.0 + stress)
    
    def expected_stress(self, jump_intensity: float) -> float:
        """
        Compute expected long-run stress level.
        
        E[S] = γ * λ * E[|J|] / κ
        
        For a compound Poisson process with intensity λ.
        
        Args:
            jump_intensity: Jump intensity λ
        
        Returns:
            Expected stress level
        """
        # Assuming E[|J|] ≈ 1 for normalized jumps
        return self.gamma * jump_intensity / self.kappa if self.kappa > 0 else 0.0
    
    def get_config(self) -> Dict[str, float]:
        """Get configuration as dictionary."""
        return {
            'kappa': self.kappa,
            'gamma': self.gamma,
            's0': self.s0,
            'max_stress': self.max_stress,
            'half_life': self.half_life
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'StressFactor':
        """Create from configuration dictionary."""
        return cls(
            kappa=config.get('feedback_kappa', 0.8),
            gamma=config.get('feedback_gamma', 1.0),
            s0=config.get('feedback_s0', 0.0),
            max_stress=config.get('feedback_max_stress', 10.0)
        )


def _coerce_rng(
    rng: Optional[Union[int, np.random.Generator]]
) -> np.random.Generator:
    """Create or validate a NumPy random generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _evaluate_path_function(
    fn: Union[float, Callable[[float, np.ndarray], np.ndarray]],
    t: float,
    x: np.ndarray,
    name: str,
) -> np.ndarray:
    """Evaluate a scalar or vectorized path function and broadcast to x."""
    if callable(fn):
        value = fn(t, x)
    else:
        value = fn

    if np.isscalar(value):
        return np.full_like(x, float(value), dtype=np.float64)

    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == x.shape:
        return arr
    try:
        return np.broadcast_to(arr, x.shape).astype(np.float64, copy=True)
    except ValueError as exc:
        raise ValueError(
            f"{name} must return a scalar or an array broadcastable to {x.shape}; "
            f"got shape {arr.shape}"
        ) from exc


def simulate_coupled_jump_diffusion_feedback(
    X_0: Union[float, np.ndarray],
    T: float,
    N_steps: int,
    mu: Union[float, Callable[[float, np.ndarray], np.ndarray]],
    sigma_LV_func: Union[float, Callable[[float, np.ndarray], np.ndarray]],
    kappa: float,
    gamma: float,
    lambda_jump: float,
    mu_J: float,
    delta_J: float,
    num_paths: int = 1,
    rng: Optional[Union[int, np.random.Generator]] = None,
    include_initial: bool = True,
    max_stress: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a jump-diffusion with transient volatility feedback.

    This implements the synchronous coupled operator splitting scheme:

    1. Diffusion phase:
       S* = S_{n-1} exp(-kappa dt)
       X* = X_{n-1} + mu dt
            + sigma_LV(t, X_{n-1}) sqrt(1 + S_{n-1}) sqrt(dt) Z
    2. Coupled jump phase:
       J_X is sampled from a compound Poisson Merton jump distribution.
       J_S = gamma |J_X| is computed from the same jump realization.
    3. Recombination:
       X_n = X* + J_X
       S_n = S* + J_S

    Args:
        X_0: Scalar initial log-price or one value per path.
        T: Terminal time.
        N_steps: Number of time intervals.
        mu: Constant drift or callable ``mu(t, x)`` returning pathwise drift.
        sigma_LV_func: Constant local volatility or callable
            ``sigma_LV_func(t, x)`` returning pathwise volatility.
        kappa: Exponential stress decay rate.
        gamma: Jump-to-stress impact coefficient.
        lambda_jump: Poisson jump intensity.
        mu_J: Mean of each Gaussian jump size.
        delta_J: Standard deviation of each Gaussian jump size.
        num_paths: Number of Monte Carlo paths.
        rng: Optional NumPy ``Generator`` or integer seed.
        include_initial: If True, return arrays with shape
            ``(num_paths, N_steps + 1)`` including ``X_0`` and ``S_0``.
            If False, return only simulated post-step values with shape
            ``(num_paths, N_steps)``.
        max_stress: Optional cap for numerical stability.

    Returns:
        Tuple ``(X, S)`` containing log-price and stress trajectories.
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if N_steps <= 0:
        raise ValueError("N_steps must be positive")
    if num_paths <= 0:
        raise ValueError("num_paths must be positive")
    if kappa < 0:
        raise ValueError("kappa must be non-negative")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if lambda_jump < 0:
        raise ValueError("lambda_jump must be non-negative")
    if delta_J < 0:
        raise ValueError("delta_J must be non-negative")

    generator = _coerce_rng(rng)
    dt = float(T) / int(N_steps)
    sqrt_dt = np.sqrt(dt)
    decay = np.exp(-float(kappa) * dt)

    x0 = np.asarray(X_0, dtype=np.float64)
    if x0.ndim == 0:
        x_curr = np.full(num_paths, float(x0), dtype=np.float64)
    else:
        x_curr = np.broadcast_to(x0.reshape(-1), (num_paths,)).astype(np.float64, copy=True)
    s_curr = np.zeros(num_paths, dtype=np.float64)

    out_steps = N_steps + 1 if include_initial else N_steps
    X = np.zeros((num_paths, out_steps), dtype=np.float64)
    S = np.zeros((num_paths, out_steps), dtype=np.float64)
    if include_initial:
        X[:, 0] = x_curr
        S[:, 0] = s_curr

    for step in range(N_steps):
        t_prev = step * dt

        # Coupled Operator Splitting, phase 1:
        # evolve the continuous diffusion from the pre-jump state.
        drift = _evaluate_path_function(mu, t_prev, x_curr, "mu")
        local_vol = _evaluate_path_function(sigma_LV_func, t_prev, x_curr, "sigma_LV_func")
        z = generator.standard_normal(num_paths)
        s_star = s_curr * decay
        x_star = (
            x_curr
            + drift * dt
            + local_vol * np.sqrt(1.0 + np.maximum(s_curr, 0.0)) * sqrt_dt * z
        )

        # Coupled Operator Splitting, phase 2:
        # sample the compound Poisson jump once and reuse it for price and stress.
        jump_counts = generator.poisson(float(lambda_jump) * dt, size=num_paths)
        jump_x = np.zeros(num_paths, dtype=np.float64)
        active = jump_counts > 0
        if np.any(active):
            counts = jump_counts[active].astype(np.float64)
            jump_x[active] = (
                counts * float(mu_J)
                + np.sqrt(counts) * float(delta_J) * generator.standard_normal(counts.size)
            )
        jump_s = float(gamma) * np.abs(jump_x)

        # Coupled Operator Splitting, phase 3:
        # recombine both state variables inside the same discrete time step.
        x_curr = x_star + jump_x
        s_curr = s_star + jump_s
        if max_stress is not None:
            s_curr = np.minimum(s_curr, float(max_stress))

        out_idx = step + 1 if include_initial else step
        X[:, out_idx] = x_curr
        S[:, out_idx] = s_curr

    return X, S


def calibrate_feedback_params(
    returns: np.ndarray,
    jump_mask: np.ndarray,
    time_grid: np.ndarray,
    method: str = 'moment_matching'
) -> Tuple[float, float]:
    """
    Calibrate feedback parameters from historical data.
    
    Args:
        returns: Return series (n_samples, n_steps)
        jump_mask: Boolean jump mask
        time_grid: Time points
        method: Calibration method ('moment_matching' or 'mle')
    
    Returns:
        Tuple of (kappa, gamma)
    """
    if method == 'moment_matching':
        # Simple moment matching approach
        
        # Estimate volatility around jumps vs. normal times
        vol_at_jumps = np.std(returns[jump_mask]) if np.any(jump_mask) else np.std(returns)
        vol_normal = np.std(returns[~jump_mask]) if np.any(~jump_mask) else np.std(returns)
        
        # Estimate gamma from volatility ratio
        vol_ratio = vol_at_jumps / vol_normal if vol_normal > 0 else 1.0
        gamma = max(0.1, (vol_ratio ** 2 - 1))  # From √(1 + S) = vol_ratio
        
        # Estimate kappa from autocorrelation decay
        # Use squared returns as volatility proxy
        sq_returns = returns ** 2
        
        # Compute autocorrelation at lag 1
        if len(sq_returns.flatten()) > 10:
            acf1 = np.corrcoef(sq_returns.flatten()[:-1], sq_returns.flatten()[1:])[0, 1]
            acf1 = max(0.01, min(0.99, acf1))  # Bound for stability
            
            # From AR(1): acf(1) ≈ exp(-κ*dt)
            dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 1.0
            kappa = -np.log(acf1) / dt
            kappa = max(0.1, min(50.0, kappa))  # Reasonable bounds
        else:
            kappa = 0.8  # Default
        
        return kappa, gamma
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")


def analyze_volatility_clustering(
    returns: np.ndarray,
    stress: np.ndarray,
    time_grid: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze volatility clustering in generated data.
    
    Args:
        returns: Return series (n_samples, n_steps)
        stress: Stress factor trajectory (n_samples, n_steps)
        time_grid: Time points
    
    Returns:
        Dictionary with analysis results
    """
    # Compute realized volatility
    realized_vol = np.abs(returns)
    
    # Correlation between stress and realized volatility
    stress_vol_corr = np.corrcoef(
        stress.flatten(),
        realized_vol.flatten()
    )[0, 1]
    
    # Autocorrelation of squared returns (volatility clustering measure)
    sq_returns = returns ** 2
    acf_sq = []
    for lag in range(1, min(20, returns.shape[1])):
        acf = np.corrcoef(
            sq_returns[:, :-lag].flatten(),
            sq_returns[:, lag:].flatten()
        )[0, 1]
        acf_sq.append(acf)
    
    # Compute ARCH effect (variance of variance)
    vol_of_vol = np.std(realized_vol, axis=1).mean()
    
    return {
        'stress_vol_correlation': stress_vol_corr,
        'acf_squared_returns': acf_sq,
        'volatility_of_volatility': vol_of_vol,
        'mean_stress': np.mean(stress),
        'max_stress': np.max(stress),
        'stress_std': np.std(stress)
    }
