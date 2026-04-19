"""
Evaluation Metrics for Time Series Generation

This module provides comprehensive evaluation metrics for
assessing the quality of generated time series.

Metrics Categories:
    1. Statistical Metrics: Wasserstein Distance, ACF MSE, Correlation
    2. Discriminative Score: GRU classifier distinguishing real/fake
    3. Predictive Score: Sequence prediction quality
    4. Financial Metrics: Stylized facts preservation

Author: Manus AI
"""

from metrics.discriminative_score import discriminative_score_metrics, discriminative_score_cnn_metrics
from metrics.predictive_score import predictive_score_metrics
from metrics.statistical_metrics import compute_all_metrics
from metrics.numba_metrics import (
    wasserstein_distance_1d_numba,
    acf_numba,
    acf_mse_numba,
    correlation_matrix_numba,
    correlation_distance_numba,
    compute_all_metrics_numba
)

__all__ = [
    # Original metrics
    'discriminative_score_metrics',
    'discriminative_score_cnn_metrics',
    'predictive_score_metrics', 
    'compute_all_metrics',
    
    # Numba-accelerated metrics
    'wasserstein_distance_1d_numba',
    'acf_numba',
    'acf_mse_numba',
    'correlation_matrix_numba',
    'correlation_distance_numba',
    'compute_all_metrics_numba',
]
