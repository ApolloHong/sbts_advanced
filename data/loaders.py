"""
Data Loaders

Provides data loading and preprocessing utilities.

Author: Manus AI
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import warnings

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


def load_etf_data(
    tickers: List[str] = ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD'],
    start_date: str = '2020-01-01',
    end_date: str = '2024-01-01',
    interval: str = '1d',
    normalize: bool = True,
    return_type: str = 'log_returns',
    auto_adjust: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load ETF price data from Yahoo Finance.
    
    Args:
        tickers: List of ETF ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: yfinance interval, e.g. '1d' or '1h'
        normalize: Whether to normalize prices to start at 100
        return_type: 'prices', 'returns', or 'log_returns'
        auto_adjust: Forwarded to yfinance download.
    
    Returns:
        Tuple of (data, time_grid, metadata)
        - data: (1, n_steps, n_features) array
        - time_grid: (n_steps,) array
        - metadata: Dictionary with additional info
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance required for ETF data loading. Install with: pip install yfinance")
    
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required for ETF data loading. Install with: pip install pandas")
    
    print(f"[Data] Loading ETF data: {tickers}")
    print(f"[Data] Period: {start_date} to {end_date} (interval={interval})")
    
    # Download data
    data_dict = {}
    price_column_used = {}
    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=auto_adjust,
            )
            if len(df) > 0:
                if 'Close' in df.columns:
                    series = df['Close']
                    price_column_used[ticker] = 'Close'
                elif 'Adj Close' in df.columns:
                    warnings.warn(
                        f"{ticker}: 'Close' not found, falling back to 'Adj Close'. "
                        "If this is unexpected, check the installed yfinance version "
                        "and auto_adjust setting."
                    )
                    series = df['Adj Close']
                    price_column_used[ticker] = 'Adj Close'
                else:
                    warnings.warn(
                        f"{ticker}: neither 'Adj Close' nor 'Close' columns are available"
                    )
                    continue

                data_dict[ticker] = series
                print(f"  {ticker}: {len(df)} data points")
            else:
                warnings.warn(f"No data for {ticker}")
        except Exception as e:
            warnings.warn(f"Failed to load {ticker}: {e}")
    
    if len(data_dict) == 0:
        raise ValueError("No data loaded")
    
    # Align on the intersection of trading dates.
    aligned = pd.concat(data_dict, axis=1, join='inner').dropna()
    if aligned.empty:
        raise ValueError("No overlapping dates across loaded tickers")

    prices = aligned.to_numpy()
    aligned_tickers = list(aligned.columns.get_level_values(0)) if hasattr(aligned.columns, "get_level_values") else list(aligned.columns)
    
    # Normalize to start at 100
    if normalize:
        prices = prices / prices[0] * 100
    
    # Create time grid
    time_grid = np.linspace(0, 1, len(prices))
    
    # Convert to returns if requested
    if return_type == 'returns':
        data = np.diff(prices, axis=0) / prices[:-1]
        time_grid = time_grid[1:]
    elif return_type == 'log_returns':
        data = np.diff(np.log(prices), axis=0)
        time_grid = time_grid[1:]
    else:
        data = prices
    
    # Reshape to (1, n_steps, n_features)
    data = data[np.newaxis, :, :]
    
    metadata = {
        'tickers': aligned_tickers,
        'n_features': len(aligned_tickers),
        'n_steps': data.shape[1],
        'start_date': start_date,
        'end_date': end_date,
        'interval': interval,
        'return_type': return_type,
        'auto_adjust': auto_adjust,
        'price_column_used': price_column_used,
        'aligned_dates': [str(idx) for idx in aligned.index],
        'prices': prices if return_type != 'prices' else None
    }
    
    print(f"[Data] Loaded shape: {data.shape}")
    
    return data, time_grid, metadata


def load_synthetic_data(
    n_samples: int = 100,
    n_steps: int = 60,
    n_features: int = 5,
    data_type: str = 'gbm_jump',
    seed: Optional[int] = None,
    return_type: str = 'prices'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic time series data for testing.
    
    Args:
        n_samples: Number of samples
        n_steps: Number of time steps
        n_features: Number of features (assets)
        data_type: Type of synthetic data:
            - 'gbm': Geometric Brownian Motion
            - 'gbm_jump': GBM with jumps (Merton model)
            - 'heston': Heston stochastic volatility
            - 'ou': Ornstein-Uhlenbeck process
        seed: Random seed for reproducibility
        return_type: 'prices', 'returns', or 'log_returns'
    
    Returns:
        Tuple of (data, time_grid, metadata)
    """
    if seed is not None:
        np.random.seed(seed)
    
    time_grid = np.linspace(0, 1, n_steps)
    dt = time_grid[1] - time_grid[0]
    
    if data_type == 'gbm':
        prices = _generate_gbm(n_samples, n_steps, n_features, dt)
    elif data_type == 'gbm_jump':
        prices = _generate_gbm_jump(n_samples, n_steps, n_features, dt)
    elif data_type == 'heston':
        prices = _generate_heston(n_samples, n_steps, n_features, dt)
    elif data_type == 'ou':
        prices = _generate_ou(n_samples, n_steps, n_features, dt)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if return_type == 'returns':
        data = np.diff(prices, axis=1) / (prices[:, :-1, :] + 1e-8)
        time_grid = time_grid[1:]
    elif return_type == 'log_returns':
        data = np.diff(np.log(np.maximum(prices, 1e-8)), axis=1)
        time_grid = time_grid[1:]
    elif return_type == 'prices':
        data = prices
    else:
        raise ValueError(f"Unknown synthetic return_type: {return_type}")

    metadata = {
        'data_type': data_type,
        'n_samples': n_samples,
        'n_steps': n_steps,
        'n_features': n_features,
        'seed': seed,
        'return_type': return_type,
        'prices': prices if return_type != 'prices' else None,
    }
    
    return data, time_grid, metadata


def _generate_gbm(
    n_samples: int,
    n_steps: int,
    n_features: int,
    dt: float,
    mu: float = 0.05,
    sigma: float = 0.2
) -> np.ndarray:
    """Generate Geometric Brownian Motion."""
    data = np.zeros((n_samples, n_steps, n_features))
    data[:, 0, :] = 100  # Start at 100
    
    for t in range(1, n_steps):
        dW = np.random.randn(n_samples, n_features) * np.sqrt(dt)
        data[:, t, :] = data[:, t-1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW
        )
    
    return data


def _generate_gbm_jump(
    n_samples: int,
    n_steps: int,
    n_features: int,
    dt: float,
    mu: float = 0.05,
    sigma: float = 0.2,
    lambda_jump: float = 0.1,
    mu_jump: float = -0.02,
    sigma_jump: float = 0.05
) -> np.ndarray:
    """Generate GBM with Merton jumps."""
    data = np.zeros((n_samples, n_steps, n_features))
    data[:, 0, :] = 100
    
    for t in range(1, n_steps):
        # Diffusion
        dW = np.random.randn(n_samples, n_features) * np.sqrt(dt)
        
        # Jumps
        n_jumps = np.random.poisson(lambda_jump * dt, (n_samples, n_features))
        jump_sizes = np.where(
            n_jumps > 0,
            np.random.normal(mu_jump, sigma_jump, (n_samples, n_features)),
            0
        )
        
        data[:, t, :] = data[:, t-1, :] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW + jump_sizes
        )
    
    return data


def _generate_heston(
    n_samples: int,
    n_steps: int,
    n_features: int,
    dt: float,
    mu: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.3,
    rho: float = -0.7,
    v0: float = 0.04
) -> np.ndarray:
    """Generate Heston stochastic volatility model."""
    data = np.zeros((n_samples, n_steps, n_features))
    data[:, 0, :] = 100
    
    v = np.full((n_samples, n_features), v0)
    
    for t in range(1, n_steps):
        # Correlated Brownian motions
        dW1 = np.random.randn(n_samples, n_features) * np.sqrt(dt)
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.randn(n_samples, n_features) * np.sqrt(dt)
        
        # Variance process
        v = np.maximum(v + kappa * (theta - v) * dt + xi * np.sqrt(v) * dW2, 0)
        
        # Price process
        data[:, t, :] = data[:, t-1, :] * np.exp(
            (mu - 0.5 * v) * dt + np.sqrt(v) * dW1
        )
    
    return data


def _generate_ou(
    n_samples: int,
    n_steps: int,
    n_features: int,
    dt: float,
    theta: float = 0.5,
    mu: float = 0.0,
    sigma: float = 0.1
) -> np.ndarray:
    """Generate Ornstein-Uhlenbeck process."""
    data = np.zeros((n_samples, n_steps, n_features))
    data[:, 0, :] = mu
    
    for t in range(1, n_steps):
        dW = np.random.randn(n_samples, n_features) * np.sqrt(dt)
        data[:, t, :] = data[:, t-1, :] + theta * (mu - data[:, t-1, :]) * dt + sigma * dW
    
    return data


def create_sliding_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1
) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Input data (n_samples, n_steps, n_features) or (n_steps, n_features)
        window_size: Size of each window
        stride: Step size between windows
    
    Returns:
        Windowed data (n_windows, window_size, n_features)
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    n_samples, n_steps, n_features = data.shape
    
    windows = []
    for i in range(n_samples):
        for start in range(0, n_steps - window_size + 1, stride):
            windows.append(data[i, start:start + window_size, :])
    
    return np.array(windows)


def normalize_data(
    data: np.ndarray,
    method: str = 'standard'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize time series data.
    
    Args:
        data: Input data
        method: Normalization method:
            - 'standard': Zero mean, unit variance
            - 'minmax': Scale to [0, 1]
            - 'initial': Normalize by initial value
    
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    params = {'method': method}
    
    if method == 'standard':
        params['mean'] = np.mean(data)
        params['std'] = np.std(data) + 1e-8
        normalized = (data - params['mean']) / params['std']
    
    elif method == 'minmax':
        params['min'] = np.min(data)
        params['max'] = np.max(data)
        normalized = (data - params['min']) / (params['max'] - params['min'] + 1e-8)
    
    elif method == 'initial':
        if data.ndim == 3:
            params['initial'] = data[:, 0:1, :]
        else:
            params['initial'] = data[0]
        normalized = data / (params['initial'] + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(
    data: np.ndarray,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Denormalize time series data.
    
    Args:
        data: Normalized data
        params: Normalization parameters from normalize_data()
    
    Returns:
        Denormalized data
    """
    method = params['method']
    
    if method == 'standard':
        return data * params['std'] + params['mean']
    
    elif method == 'minmax':
        return data * (params['max'] - params['min']) + params['min']
    
    elif method == 'initial':
        return data * params['initial']
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
