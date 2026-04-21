"""
SDE Solver Module with Jump-Volatility Feedback Mechanism

Implements the JD-SBTS-F (Feedback) solver that captures the
"Volatility Clustering" phenomenon through a Transient Stress Factor.

Mathematical Model:
    dX_t = μ(t, X_t)dt + σ_LV(t, X_t) * √(1 + S_t) * dW_t + dJ_t
    dS_t = -κ * S_t * dt + γ * |dJ_t|

Where:
    - S_t: Transient Stress Factor (Shot Noise process)
    - κ: Mean reversion speed of stress
    - γ: Jump impact multiplier

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable, Union
from numba import njit, prange
import warnings

# Note: SDESolver base class is defined locally to avoid circular imports
from abc import ABC, abstractmethod

class SDESolver(ABC):
    """Abstract base class for SDE solvers."""
    
    SOLVER_TYPE: str = "base"
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def solve(self, x0: np.ndarray, time_grid: np.ndarray, drift_fn, volatility_fn, jump_sampler=None, **kwargs) -> np.ndarray:
        pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================
# Numba-Accelerated Euler-Maruyama Solver
# ============================================

@njit(cache=True, parallel=True)
def _euler_maruyama_numba(
    x0: np.ndarray,
    time_grid: np.ndarray,
    drift_values: np.ndarray,
    vol_values: np.ndarray,
    jump_sizes: np.ndarray,
    dW: np.ndarray
) -> np.ndarray:
    """
    Numba-accelerated Euler-Maruyama solver (without feedback).
    
    Args:
        x0: Initial conditions (n_samples, n_features)
        time_grid: Time points (n_steps,)
        drift_values: Pre-computed drift values (n_samples, n_steps, n_features)
        vol_values: Pre-computed volatility values (n_samples, n_steps, n_features)
        jump_sizes: Jump sizes (n_samples, n_steps, n_features)
        dW: Brownian increments (n_samples, n_steps, n_features)
    
    Returns:
        Solution paths (n_samples, n_steps, n_features)
    """
    n_samples, n_steps, n_features = drift_values.shape
    paths = np.zeros((n_samples, n_steps, n_features))
    
    for i in prange(n_samples):
        paths[i, 0, :] = x0[i, :]
        
        for t in range(1, n_steps):
            dt = time_grid[t] - time_grid[t-1]
            sqrt_dt = np.sqrt(dt)
            
            for k in range(n_features):
                drift = drift_values[i, t-1, k]
                vol = vol_values[i, t-1, k]
                
                paths[i, t, k] = (
                    paths[i, t-1, k] +
                    drift * dt +
                    vol * sqrt_dt * dW[i, t-1, k] +
                    jump_sizes[i, t-1, k]
                )
    
    return paths


@njit(cache=True, parallel=True)
def _euler_maruyama_feedback_numba(
    x0: np.ndarray,
    time_grid: np.ndarray,
    drift_values: np.ndarray,
    vol_values: np.ndarray,
    jump_sizes: np.ndarray,
    dW: np.ndarray,
    kappa: float,
    gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated Euler-Maruyama solver WITH feedback mechanism.
    
    Implements the JD-SBTS-F model:
        dX_t = μ(t, X_t)dt + σ_LV(t, X_t) * √(1 + S_t) * dW_t + dJ_t
        dS_t = -κ * S_t * dt + γ * |dJ_t|
    
    Args:
        x0: Initial conditions (n_samples, n_features)
        time_grid: Time points (n_steps,)
        drift_values: Pre-computed drift values (n_samples, n_steps, n_features)
        vol_values: Pre-computed volatility values (n_samples, n_steps, n_features)
        jump_sizes: Jump sizes (n_samples, n_steps, n_features)
        dW: Brownian increments (n_samples, n_steps, n_features)
        kappa: Mean reversion speed of stress factor
        gamma: Jump impact multiplier
    
    Returns:
        Tuple of (paths, stress_factor):
            - paths: Solution paths (n_samples, n_steps, n_features)
            - stress_factor: Stress factor trajectory (n_samples, n_steps)
    """
    n_samples, n_steps, n_features = drift_values.shape
    paths = np.zeros((n_samples, n_steps, n_features))
    stress = np.zeros((n_samples, n_steps))
    
    for i in prange(n_samples):
        paths[i, 0, :] = x0[i, :]
        stress[i, 0] = 0.0  # Initialize stress at zero
        
        for t in range(1, n_steps):
            dt = time_grid[t] - time_grid[t-1]
            sqrt_dt = np.sqrt(dt)
            
            # Coupled operator splitting:
            # the continuous diffusion uses pre-jump stress S_{t-1}; the
            # same jump realization then updates both X_t and S_t.
            vol_multiplier = np.sqrt(1.0 + stress[i, t-1])
            
            for k in range(n_features):
                drift = drift_values[i, t-1, k]
                vol = vol_values[i, t-1, k] * vol_multiplier
                
                paths[i, t, k] = (
                    paths[i, t-1, k] +
                    drift * dt +
                    vol * sqrt_dt * dW[i, t-1, k] +
                    jump_sizes[i, t-1, k]
                )

            total_jump = 0.0
            for k in range(n_features):
                total_jump += np.abs(jump_sizes[i, t-1, k])
            stress[i, t] = stress[i, t-1] * np.exp(-kappa * dt) + gamma * total_jump
    
    return paths, stress


# ============================================
# PyTorch Vectorized Solver
# ============================================

if TORCH_AVAILABLE:
    
    class JumpDiffusionEulerSolverTorch:
        """
        PyTorch-based Euler-Maruyama solver with feedback mechanism.
        
        Fully vectorized for GPU acceleration and gradient compatibility.
        """
        
        def __init__(
            self,
            use_feedback: bool = True,
            kappa: float = 5.0,
            gamma: float = 0.5,
            device: str = 'auto'
        ):
            """
            Initialize solver.
            
            Args:
                use_feedback: Whether to use feedback mechanism
                kappa: Mean reversion speed of stress factor
                gamma: Jump impact multiplier
                device: Device to use ('auto', 'cpu', 'cuda')
            """
            self.use_feedback = use_feedback
            self.kappa = kappa
            self.gamma = gamma
            
            if device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)
        
        def solve(
            self,
            x0: torch.Tensor,
            time_grid: torch.Tensor,
            drift_fn: Callable,
            volatility_fn: Callable,
            jump_sampler: Optional[Callable] = None,
            return_stress: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Solve the SDE using Euler-Maruyama scheme.
            
            Args:
                x0: Initial conditions (n_samples, n_features)
                time_grid: Time points (n_steps,)
                drift_fn: Drift function μ(t, x) -> (n_samples, n_features)
                volatility_fn: Volatility function σ(t, x) -> (n_samples, n_features)
                jump_sampler: Optional jump sampler (dt) -> (jump_mask, jump_sizes)
                return_stress: Whether to return stress factor trajectory
            
            Returns:
                Solution paths (n_samples, n_steps, n_features)
                Optionally also stress factor (n_samples, n_steps)
            """
            n_samples, n_features = x0.shape
            n_steps = len(time_grid)
            
            # Initialize paths and stress
            paths = torch.zeros(n_samples, n_steps, n_features, device=self.device)
            paths[:, 0, :] = x0
            
            if self.use_feedback:
                stress = torch.zeros(n_samples, n_steps, device=self.device)
            
            # Pre-generate Brownian increments
            dW = torch.randn(n_samples, n_steps - 1, n_features, device=self.device)
            
            # Time stepping
            for t in range(1, n_steps):
                dt = time_grid[t] - time_grid[t-1]
                sqrt_dt = torch.sqrt(dt)
                
                x_prev = paths[:, t-1, :]
                t_prev = time_grid[t-1]
                
                # Compute drift and volatility
                drift = drift_fn(t_prev, x_prev)
                vol = volatility_fn(t_prev, x_prev)
                
                # Sample jumps
                if jump_sampler is not None:
                    jump_mask, jump_sizes = jump_sampler(dt.item())
                    jump_sizes = torch.tensor(jump_sizes, dtype=torch.float32, device=self.device)
                    if jump_sizes.ndim == 1:
                        jump_sizes = jump_sizes.unsqueeze(-1).expand(-1, n_features)
                else:
                    jump_sizes = torch.zeros(n_samples, n_features, device=self.device)
                
                # Coupled operator splitting: diffusion uses S_{t-1};
                # the sampled jump updates X_t and S_t in the same step.
                if self.use_feedback:
                    vol_multiplier = torch.sqrt(1.0 + stress[:, t-1]).unsqueeze(-1)
                    vol = vol * vol_multiplier
                
                # Euler-Maruyama update
                paths[:, t, :] = (
                    x_prev +
                    drift * dt +
                    vol * sqrt_dt * dW[:, t-1, :] +
                    jump_sizes
                )

                if self.use_feedback:
                    total_jump = torch.abs(jump_sizes).sum(dim=-1)
                    stress[:, t] = stress[:, t-1] * torch.exp(-self.kappa * dt) + self.gamma * total_jump
            
            if return_stress and self.use_feedback:
                return paths, stress
            return paths


# ============================================
# Main Solver Class
# ============================================

class JumpDiffusionEulerSolver(SDESolver):
    """
    Jump-Diffusion Euler-Maruyama Solver with optional Feedback mechanism.
    
    This is the main solver for JD-SBTS and JD-SBTS-F models.
    
    Features:
        - Numba acceleration for CPU-bound operations
        - PyTorch backend for GPU and gradient support
        - Feedback mechanism for volatility clustering
    """
    
    SOLVER_TYPE = "euler_maruyama"
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize solver.
        
        Args:
            config: Configuration with keys:
                - use_feedback: Whether to use feedback mechanism (default: True)
                - feedback_kappa: Mean reversion speed (default: 0.8)
                - feedback_gamma: Jump impact multiplier (default: 1.0)
                - solver_backend: 'numba' or 'torch' (default: 'numba')
        """
        super().__init__(config)
        
        self.use_feedback = config.get('use_feedback', True)
        self.kappa = config.get('feedback_kappa', 0.8)
        self.gamma = config.get('feedback_gamma', 1.0)
        self.max_stress = config.get('feedback_max_stress', 10.0)
        self.backend = config.get('solver_backend', 'numba')
        
        # Initialize PyTorch solver if needed
        if self.backend == 'torch' and TORCH_AVAILABLE:
            self.torch_solver = JumpDiffusionEulerSolverTorch(
                use_feedback=self.use_feedback,
                kappa=self.kappa,
                gamma=self.gamma
            )
        else:
            self.torch_solver = None
    
    def solve(
        self,
        x0: np.ndarray,
        time_grid: np.ndarray,
        drift_fn: Callable,
        volatility_fn: Callable,
        jump_sampler: Optional[Callable] = None,
        return_stress: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Solve the SDE.
        
        Args:
            x0: Initial conditions (n_samples, n_features)
            time_grid: Time points (n_steps,)
            drift_fn: Drift function μ(t, x)
            volatility_fn: Volatility function σ(t, x)
            jump_sampler: Optional jump sampler
            return_stress: Whether to return stress factor trajectory
            **kwargs: Additional parameters
        
        Returns:
            Solution paths (n_samples, n_steps, n_features)
            Optionally also stress factor (n_samples, n_steps)
        """
        # Ensure x0 is 2D
        if x0.ndim == 1:
            x0 = x0[:, np.newaxis]
        n_samples = x0.shape[0]
        n_features = x0.shape[1]
        n_steps = len(time_grid)
        
        paths = np.zeros((n_samples, n_steps, n_features), dtype=np.float64)
        paths[:, 0, :] = x0
        stress = np.zeros((n_samples, n_steps), dtype=np.float64)
        dW = np.random.randn(n_samples, n_steps - 1, n_features)

        history_len = int(self.config.get(
            'lstm_seq_len',
            self.config.get('transformer_seq_len', 20)
        ))
        history_len = max(1, history_len)

        for t in range(1, n_steps):
            dt = float(time_grid[t] - time_grid[t - 1])
            if dt <= 0:
                raise ValueError("time_grid must be strictly increasing")

            x_prev = paths[:, t - 1, :]
            t_prev = float(time_grid[t - 1])
            history = self._build_history(paths, t, history_len)

            drift = self._evaluate_function(
                drift_fn,
                t_prev,
                x_prev,
                history=history
            )
            vol = self._evaluate_function(volatility_fn, t_prev, x_prev)

            if jump_sampler is not None:
                _, js = jump_sampler(n_samples, 1, dt)
                jump_step = self._coerce_jump_sizes(js, n_samples, n_features)
            else:
                jump_step = np.zeros((n_samples, n_features), dtype=np.float64)

            if self.use_feedback:
                # Coupled operator splitting: the diffusion increment is
                # evaluated with S_{t-1}; jump_step then updates X_t and S_t
                # synchronously below.
                vol = vol * np.sqrt(1.0 + stress[:, t - 1])[:, np.newaxis]

            paths[:, t, :] = (
                x_prev
                + drift * dt
                + vol * np.sqrt(dt) * dW[:, t - 1, :]
                + jump_step
            )

            if self.use_feedback:
                total_jump = np.abs(jump_step).sum(axis=1)
                stress[:, t] = (
                    stress[:, t - 1] * np.exp(-self.kappa * dt)
                    + self.gamma * total_jump
                )
                stress[:, t] = np.minimum(stress[:, t], self.max_stress)

        if return_stress:
            return paths, stress
        return paths

    def _build_history(
        self,
        paths: np.ndarray,
        step_idx: int,
        history_len: int
    ) -> np.ndarray:
        """Build a fixed-length state history ending at step_idx - 1."""
        n_samples, _, n_features = paths.shape
        history = np.zeros((n_samples, history_len, n_features), dtype=np.float64)
        start = max(0, step_idx - history_len)
        available = paths[:, start:step_idx, :]
        if available.shape[1] == 0:
            return history
        history[:, -available.shape[1]:, :] = available
        if available.shape[1] < history_len:
            history[:, :history_len - available.shape[1], :] = available[:, :1, :]
        return history

    def _coerce_jump_sizes(
        self,
        jump_sizes: np.ndarray,
        n_samples: int,
        n_features: int
    ) -> np.ndarray:
        """Normalize jump sampler output to (n_samples, n_features)."""
        js = np.asarray(jump_sizes, dtype=np.float64)
        if js.ndim == 3:
            if js.shape[1] != 1:
                js = js[:, 0, :]
            else:
                js = js[:, 0, :]
        elif js.ndim == 2:
            if js.shape == (n_samples, n_features):
                pass
            elif js.shape[0] == n_samples and js.shape[1] == 1:
                js = np.repeat(js, n_features, axis=1)
            elif js.shape[0] == n_samples:
                if js.shape[1] >= n_features:
                    js = js[:, :n_features]
                else:
                    reps = int(np.ceil(n_features / js.shape[1]))
                    js = np.tile(js, (1, reps))[:, :n_features]
            else:
                js = np.broadcast_to(js, (n_samples, n_features))
        elif js.ndim == 1:
            if len(js) != n_samples:
                js = np.broadcast_to(js, (n_samples,))
            js = np.repeat(js[:, np.newaxis], n_features, axis=1)
        else:
            js = np.full((n_samples, n_features), float(js))

        if js.shape != (n_samples, n_features):
            js = np.broadcast_to(js, (n_samples, n_features)).copy()
        return js
    
    def _evaluate_function(
        self,
        fn: Callable,
        t: float,
        x: np.ndarray,
        history: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Evaluate a function, handling different return shapes.
        
        Args:
            fn: Function to evaluate
            t: Time point
            x: State values (n_samples, n_features)
        
        Returns:
            Function values (n_samples, n_features)
        """
        try:
            if history is None:
                result = fn(t, x)
            else:
                try:
                    result = fn(t, x, history=history)
                except TypeError:
                    result = fn(t, x)
            
            # Handle scalar return
            if np.isscalar(result):
                return np.full_like(x, result)
            
            result = np.asarray(result)
            if result.shape == x.shape:
                return result
            
            # Handle 1D return
            if result.ndim == 1:
                if len(result) == x.shape[0]:
                    return result[:, np.newaxis] * np.ones((1, x.shape[1]))
                elif len(result) == x.shape[1]:
                    return np.ones((x.shape[0], 1)) * result[np.newaxis, :]
            elif result.ndim == 2:
                if result.shape[0] == x.shape[0] and result.shape[1] == 1:
                    return result * np.ones((1, x.shape[1]))
                if result.shape[0] == 1 and result.shape[1] == x.shape[1]:
                    return np.ones((x.shape[0], 1)) * result
            
            return np.broadcast_to(result, x.shape).copy()
            
        except Exception as e:
            warnings.warn(f"Error evaluating function: {e}")
            return np.zeros_like(x)
    
    def solve_with_neural_drift(
        self,
        x0: np.ndarray,
        time_grid: np.ndarray,
        drift_model,  # PyTorch model
        volatility_fn: Callable,
        jump_sampler: Optional[Callable] = None,
        return_stress: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Solve SDE with neural drift estimator (gradient-compatible).
        
        Uses PyTorch backend for gradient flow through drift model.
        
        Args:
            x0: Initial conditions
            time_grid: Time points
            drift_model: PyTorch drift model
            volatility_fn: Volatility function
            jump_sampler: Optional jump sampler
            return_stress: Whether to return stress factor
        
        Returns:
            Solution paths (and optionally stress factor)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for neural drift solver")
        
        if self.torch_solver is None:
            self.torch_solver = JumpDiffusionEulerSolverTorch(
                use_feedback=self.use_feedback,
                kappa=self.kappa,
                gamma=self.gamma
            )
        
        # Convert to torch tensors
        x0_torch = torch.tensor(x0, dtype=torch.float32, device=self.torch_solver.device)
        time_grid_torch = torch.tensor(time_grid, dtype=torch.float32, device=self.torch_solver.device)
        
        # Wrap drift model
        def drift_fn_torch(t, x):
            return drift_model(t, x)
        
        # Wrap volatility function
        def vol_fn_torch(t, x):
            vol = volatility_fn(t, x.cpu().numpy())
            return torch.tensor(vol, dtype=torch.float32, device=self.torch_solver.device)
        
        # Solve
        result = self.torch_solver.solve(
            x0_torch,
            time_grid_torch,
            drift_fn_torch,
            vol_fn_torch,
            jump_sampler,
            return_stress
        )
        
        if return_stress:
            paths, stress = result
            return paths.cpu().numpy(), stress.cpu().numpy()
        return result.cpu().numpy()


# ============================================
# Feedback Analysis Utilities
# ============================================

def analyze_feedback_dynamics(
    stress_trajectory: np.ndarray,
    jump_times: np.ndarray,
    time_grid: np.ndarray,
    kappa: float,
    gamma: float
) -> Dict[str, Any]:
    """
    Analyze the feedback dynamics from a simulation.
    
    Args:
        stress_trajectory: Stress factor trajectory (n_samples, n_steps)
        jump_times: Boolean array indicating jump times
        time_grid: Time points
        kappa: Mean reversion speed
        gamma: Jump impact multiplier
    
    Returns:
        Dictionary with analysis results
    """
    n_samples, n_steps = stress_trajectory.shape
    
    # Compute statistics
    mean_stress = np.mean(stress_trajectory, axis=0)
    std_stress = np.std(stress_trajectory, axis=0)
    max_stress = np.max(stress_trajectory, axis=0)
    
    # Compute half-life of stress decay
    half_life = np.log(2) / kappa
    
    # Compute average stress level
    avg_stress = np.mean(stress_trajectory)
    
    # Compute correlation between jumps and subsequent stress
    if jump_times is not None:
        jump_stress_corr = np.corrcoef(
            jump_times.flatten(),
            stress_trajectory[:, 1:].flatten()
        )[0, 1] if stress_trajectory.shape[1] > 1 else 0.0
    else:
        jump_stress_corr = 0.0
    
    return {
        'mean_stress': mean_stress,
        'std_stress': std_stress,
        'max_stress': max_stress,
        'half_life': half_life,
        'avg_stress': avg_stress,
        'jump_stress_correlation': jump_stress_corr,
        'kappa': kappa,
        'gamma': gamma
    }
