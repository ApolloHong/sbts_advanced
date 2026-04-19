"""
Numba-Accelerated Evaluation Metrics

Provides high-performance implementations of common evaluation metrics
using Numba JIT compilation.

Author: Manus AI
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from numba import njit, prange


# ============================================
# Wasserstein Distance
# ============================================

@njit(cache=True)
def _sort_array(arr: np.ndarray) -> np.ndarray:
    """Simple sorting for Numba (uses bubble sort for small arrays)."""
    n = len(arr)
    result = arr.copy()
    
    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
    
    return result


@njit(cache=True)
def wasserstein_distance_1d_numba(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance).
    
    Uses the closed-form solution for 1D distributions:
    W_1(P, Q) = ∫ |F_P^{-1}(u) - F_Q^{-1}(u)| du
    
    Args:
        x: First distribution samples
        y: Second distribution samples
    
    Returns:
        Wasserstein distance
    """
    # Sort both arrays
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    
    n = len(x_sorted)
    m = len(y_sorted)
    
    # Interpolate to common grid
    if n == m:
        # Direct comparison
        return np.mean(np.abs(x_sorted - y_sorted))
    else:
        # Interpolate shorter to longer
        if n < m:
            x_interp = np.interp(
                np.linspace(0, 1, m),
                np.linspace(0, 1, n),
                x_sorted
            )
            return np.mean(np.abs(x_interp - y_sorted))
        else:
            y_interp = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, m),
                y_sorted
            )
            return np.mean(np.abs(x_sorted - y_interp))


@njit(cache=True, parallel=True)
def wasserstein_distance_multivariate_numba(
    X: np.ndarray,
    Y: np.ndarray
) -> float:
    """
    Compute multivariate Wasserstein distance (sliced approximation).
    
    Uses sliced Wasserstein distance for efficiency.
    
    Args:
        X: First distribution samples (n, d)
        Y: Second distribution samples (m, d)
    
    Returns:
        Sliced Wasserstein distance
    """
    n, d = X.shape
    m = Y.shape[0]
    
    # Average over marginals (simple approximation)
    total_dist = 0.0
    
    for k in prange(d):
        total_dist += wasserstein_distance_1d_numba(X[:, k], Y[:, k])
    
    return total_dist / d


# ============================================
# Autocorrelation Function
# ============================================

@njit(cache=True)
def acf_numba(
    x: np.ndarray,
    max_lag: int = 20
) -> np.ndarray:
    """
    Compute autocorrelation function.
    
    Args:
        x: Time series (1D)
        max_lag: Maximum lag to compute
    
    Returns:
        ACF values for lags 0 to max_lag
    """
    n = len(x)
    max_lag = min(max_lag, n - 1)
    
    # Demean
    x_mean = np.mean(x)
    x_centered = x - x_mean
    
    # Variance
    var = np.sum(x_centered ** 2) / n
    
    if var < 1e-10:
        return np.zeros(max_lag + 1)
    
    acf = np.zeros(max_lag + 1)
    acf[0] = 1.0
    
    for lag in range(1, max_lag + 1):
        cov = np.sum(x_centered[:-lag] * x_centered[lag:]) / n
        acf[lag] = cov / var
    
    return acf


@njit(cache=True, parallel=True)
def acf_batch_numba(
    X: np.ndarray,
    max_lag: int = 20
) -> np.ndarray:
    """
    Compute ACF for batch of time series.
    
    Args:
        X: Batch of time series (n_samples, seq_len)
        max_lag: Maximum lag
    
    Returns:
        ACF values (n_samples, max_lag + 1)
    """
    n_samples, seq_len = X.shape
    acf_batch = np.zeros((n_samples, max_lag + 1))
    
    for i in prange(n_samples):
        acf_batch[i, :] = acf_numba(X[i, :], max_lag)
    
    return acf_batch


@njit(cache=True)
def acf_mse_numba(
    real_acf: np.ndarray,
    gen_acf: np.ndarray
) -> float:
    """
    Compute MSE between ACF of real and generated data.
    
    Args:
        real_acf: ACF of real data
        gen_acf: ACF of generated data
    
    Returns:
        MSE value
    """
    n = min(len(real_acf), len(gen_acf))
    return np.mean((real_acf[:n] - gen_acf[:n]) ** 2)


# ============================================
# Correlation Matrix
# ============================================

@njit(cache=True)
def correlation_matrix_numba(
    X: np.ndarray
) -> np.ndarray:
    """
    Compute correlation matrix.
    
    Args:
        X: Data matrix (n_samples, n_features)
    
    Returns:
        Correlation matrix (n_features, n_features)
    """
    n, d = X.shape
    
    # Compute means
    means = np.zeros(d)
    for j in range(d):
        means[j] = np.mean(X[:, j])
    
    # Compute covariance matrix
    cov = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov[i, j] = np.sum((X[:, i] - means[i]) * (X[:, j] - means[j])) / n
    
    # Compute correlation matrix
    corr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            std_i = np.sqrt(cov[i, i])
            std_j = np.sqrt(cov[j, j])
            if std_i > 1e-10 and std_j > 1e-10:
                corr[i, j] = cov[i, j] / (std_i * std_j)
            else:
                corr[i, j] = 0.0 if i != j else 1.0
    
    return corr


@njit(cache=True)
def correlation_distance_numba(
    corr_real: np.ndarray,
    corr_gen: np.ndarray
) -> float:
    """
    Compute Frobenius distance between correlation matrices.
    
    Args:
        corr_real: Real data correlation matrix
        corr_gen: Generated data correlation matrix
    
    Returns:
        Frobenius norm of difference
    """
    diff = corr_real - corr_gen
    return np.sqrt(np.sum(diff ** 2))


# ============================================
# Marginal Statistics
# ============================================

@njit(cache=True)
def marginal_statistics_numba(
    x: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute marginal statistics.
    
    Args:
        x: Data array
    
    Returns:
        Tuple of (mean, std, skewness, kurtosis)
    """
    n = len(x)
    
    mean = np.mean(x)
    
    # Variance and std
    var = np.sum((x - mean) ** 2) / n
    std = np.sqrt(var)
    
    if std < 1e-10:
        return mean, std, 0.0, 0.0
    
    # Standardized moments
    z = (x - mean) / std
    
    skewness = np.mean(z ** 3)
    kurtosis = np.mean(z ** 4) - 3  # Excess kurtosis
    
    return mean, std, skewness, kurtosis


@njit(cache=True, parallel=True)
def marginal_statistics_batch_numba(
    X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute marginal statistics for batch.
    
    Args:
        X: Batch of data (n_samples, seq_len)
    
    Returns:
        Tuple of arrays (means, stds, skewnesses, kurtoses)
    """
    n_samples = X.shape[0]
    
    means = np.zeros(n_samples)
    stds = np.zeros(n_samples)
    skews = np.zeros(n_samples)
    kurts = np.zeros(n_samples)
    
    for i in prange(n_samples):
        means[i], stds[i], skews[i], kurts[i] = marginal_statistics_numba(X[i, :])
    
    return means, stds, skews, kurts


# ============================================
# Combined Metrics
# ============================================

def compute_all_metrics_numba(
    real_data: np.ndarray,
    gen_data: np.ndarray,
    max_acf_lag: int = 15
) -> Dict[str, float]:
    """
    Compute all statistical metrics using Numba acceleration.
    
    Args:
        real_data: Real time series (n_samples, seq_len) or (n_samples, seq_len, n_features)
        gen_data: Generated time series (same shape)
        max_acf_lag: Maximum ACF lag
    
    Returns:
        Dictionary of metric values
    """
    real_data = np.asarray(real_data)
    gen_data = np.asarray(gen_data)

    # Keep features separate for multivariate series. Earlier versions flattened
    # (time, feature) into one axis, which mixed time increments with adjacent
    # feature differences.
    if real_data.ndim == 3:
        n_samples, seq_len, n_features = real_data.shape
        real_returns_by_feature = np.diff(real_data, axis=1).reshape(-1, n_features)
        gen_returns_by_feature = np.diff(gen_data, axis=1).reshape(-1, n_features)
        real_acf_series = np.diff(real_data, axis=1).transpose(0, 2, 1).reshape(-1, seq_len - 1)
        gen_acf_series = np.diff(gen_data, axis=1).transpose(0, 2, 1).reshape(-1, seq_len - 1)
        real_marginal_by_feature = real_data.reshape(-1, n_features)
        gen_marginal_by_feature = gen_data.reshape(-1, n_features)
    else:
        n_features = 1
        real_returns_by_feature = np.diff(real_data, axis=1).reshape(-1, 1)
        gen_returns_by_feature = np.diff(gen_data, axis=1).reshape(-1, 1)
        real_acf_series = np.diff(real_data, axis=1)
        gen_acf_series = np.diff(gen_data, axis=1)
        real_marginal_by_feature = real_data.reshape(-1, 1)
        gen_marginal_by_feature = gen_data.reshape(-1, 1)
    
    metrics = {}
    
    # Wasserstein distance averaged over feature-wise return distributions.
    metrics['wasserstein_distance'] = wasserstein_distance_multivariate_numba(
        real_returns_by_feature.astype(np.float64),
        gen_returns_by_feature.astype(np.float64)
    )
    
    # ACF MSE averaged over all sample-feature return series.
    real_acf_batch = acf_batch_numba(
        real_acf_series.astype(np.float64),
        max_acf_lag
    )
    gen_acf_batch = acf_batch_numba(
        gen_acf_series.astype(np.float64),
        max_acf_lag
    )
    
    real_acf_mean = np.mean(real_acf_batch, axis=0)
    gen_acf_mean = np.mean(gen_acf_batch, axis=0)
    
    metrics['acf_mse'] = acf_mse_numba(real_acf_mean, gen_acf_mean)
    
    # Correlation distance (if multivariate)
    if real_data.ndim == 3 and n_features > 1:
        # Compute correlation on terminal values
        real_terminal = real_data[:, -1, :]
        gen_terminal = gen_data[:, -1, :]
        
        real_corr = correlation_matrix_numba(real_terminal.astype(np.float64))
        gen_corr = correlation_matrix_numba(gen_terminal.astype(np.float64))
        
        metrics['correlation_distance'] = correlation_distance_numba(real_corr, gen_corr)
    else:
        metrics['correlation_distance'] = 0.0
    
    # Marginal statistics comparison, averaged over features.
    mean_diffs = []
    std_diffs = []
    skew_diffs = []
    kurt_diffs = []
    for feature_idx in range(n_features):
        real_mean, real_std, real_skew, real_kurt = marginal_statistics_numba(
            real_marginal_by_feature[:, feature_idx].astype(np.float64)
        )
        gen_mean, gen_std, gen_skew, gen_kurt = marginal_statistics_numba(
            gen_marginal_by_feature[:, feature_idx].astype(np.float64)
        )
        mean_diffs.append(abs(real_mean - gen_mean))
        std_diffs.append(abs(real_std - gen_std))
        skew_diffs.append(abs(real_skew - gen_skew))
        kurt_diffs.append(abs(real_kurt - gen_kurt))

    metrics['mean_diff'] = float(np.mean(np.asarray(mean_diffs)))
    metrics['std_diff'] = float(np.mean(np.asarray(std_diffs)))
    metrics['skewness_diff'] = float(np.mean(np.asarray(skew_diffs)))
    metrics['kurtosis_diff'] = float(np.mean(np.asarray(kurt_diffs)))
    
    return metrics


# ============================================
# Stylized Facts Metrics
# ============================================

@njit(cache=True)
def volatility_clustering_score_numba(
    returns: np.ndarray,
    max_lag: int = 10
) -> float:
    """
    Compute volatility clustering score.
    
    Measures autocorrelation of absolute returns (ARCH effect).
    
    Args:
        returns: Return series
        max_lag: Maximum lag for ACF
    
    Returns:
        Average ACF of absolute returns (higher = more clustering)
    """
    abs_returns = np.abs(returns)
    acf = acf_numba(abs_returns, max_lag)
    
    # Average ACF excluding lag 0
    return np.mean(acf[1:])


@njit(cache=True)
def fat_tails_score_numba(
    returns: np.ndarray
) -> float:
    """
    Compute fat tails score.
    
    Uses excess kurtosis as a measure of fat tails.
    
    Args:
        returns: Return series
    
    Returns:
        Excess kurtosis (positive = fatter tails than normal)
    """
    _, _, _, kurtosis = marginal_statistics_numba(returns)
    return kurtosis


@njit(cache=True)
def leverage_effect_score_numba(
    returns: np.ndarray,
    max_lag: int = 5
) -> float:
    """
    Compute leverage effect score.
    
    Measures correlation between returns and future volatility.
    Negative correlation indicates leverage effect.
    
    Args:
        returns: Return series
        max_lag: Maximum lag
    
    Returns:
        Average correlation (negative = leverage effect present)
    """
    n = len(returns)
    abs_returns = np.abs(returns)
    
    total_corr = 0.0
    count = 0
    
    for lag in range(1, min(max_lag + 1, n)):
        # Correlation between r_t and |r_{t+lag}|
        r = returns[:-lag]
        vol = abs_returns[lag:]
        
        # Compute correlation
        r_mean = np.mean(r)
        vol_mean = np.mean(vol)
        
        r_std = np.std(r)
        vol_std = np.std(vol)
        
        if r_std > 1e-10 and vol_std > 1e-10:
            corr = np.mean((r - r_mean) * (vol - vol_mean)) / (r_std * vol_std)
            total_corr += corr
            count += 1
    
    return total_corr / count if count > 0 else 0.0


def compute_stylized_facts_numba(
    returns: np.ndarray
) -> Dict[str, float]:
    """
    Compute all stylized facts metrics.
    
    Args:
        returns: Return series. Supported shapes are 1D, 2D, or
            (n_samples, seq_len, n_features). For 3D input, metrics are
            computed feature-wise and averaged.
    
    Returns:
        Dictionary of stylized facts scores
    """
    returns = np.asarray(returns)

    if returns.ndim == 3:
        n_features = returns.shape[-1]
        volatility = np.zeros(n_features)
        fat_tails = np.zeros(n_features)
        leverage = np.zeros(n_features)

        for feature_idx in range(n_features):
            feature_returns = returns[:, :, feature_idx].flatten().astype(np.float64)
            volatility[feature_idx] = volatility_clustering_score_numba(feature_returns)
            fat_tails[feature_idx] = fat_tails_score_numba(feature_returns)
            leverage[feature_idx] = leverage_effect_score_numba(feature_returns)

        return {
            'volatility_clustering': float(np.mean(volatility)),
            'fat_tails': float(np.mean(fat_tails)),
            'leverage_effect': float(np.mean(leverage))
        }

    if returns.ndim == 2:
        returns = returns.flatten()

    returns = returns.astype(np.float64)
    
    return {
        'volatility_clustering': volatility_clustering_score_numba(returns),
        'fat_tails': fat_tails_score_numba(returns),
        'leverage_effect': leverage_effect_score_numba(returns)
    }
