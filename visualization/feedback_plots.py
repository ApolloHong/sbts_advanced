"""
Feedback Mechanism Visualization

Provides specialized visualizations for the JD-SBTS-F feedback mechanism,
including stress factor dynamics and volatility clustering analysis.

Author: Manus AI
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from typing import Dict, Any, Optional, List, Tuple

# Style configuration for black-and-white printing compatibility
STYLE_CONFIG = {
    'JD-SBTS': {'color': '#E64B35', 'linestyle': '-', 'marker': 'o', 'hatch': ''},
    'JD-SBTS-F': {'color': '#4DBBD5', 'linestyle': '--', 'marker': 's', 'hatch': '//'},
    'JD-SBTS-Neural': {'color': '#00A087', 'linestyle': '-.', 'marker': '^', 'hatch': '\\\\'},
    'JD-SBTS-F-Neural': {'color': '#3C5488', 'linestyle': ':', 'marker': 'D', 'hatch': 'xx'},
    'LightSB': {'color': '#F39B7F', 'linestyle': '-', 'marker': 'v', 'hatch': '..'},
    'Numba-SB': {'color': '#8491B4', 'linestyle': '--', 'marker': '<', 'hatch': '++'},
    'TimeGAN': {'color': '#91D1C2', 'linestyle': '-.', 'marker': '>', 'hatch': '--'},
    'Diffusion-TS': {'color': '#DC9FB4', 'linestyle': ':', 'marker': 'p', 'hatch': '||'},
    'rnn': {'color': '#D55E00', 'linestyle': '--', 'marker': 's', 'hatch': 'oo'},
    'transformer_ar': {'color': '#009E73', 'linestyle': '-.', 'marker': '^', 'hatch': '**'},
    'Real': {'color': 'black', 'linestyle': '-', 'marker': 'o', 'hatch': ''},
}


def _ensure_dir(path: str):
    """Ensure directory exists."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def _get_style(name: str) -> Dict[str, Any]:
    """Get style configuration for a method."""
    return STYLE_CONFIG.get(name, {
        'color': 'gray',
        'linestyle': '-',
        'marker': 'o',
        'hatch': ''
    })


def set_publication_style():
    """Set publication-quality style."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300,
        'legend.frameon': False,
        'legend.fontsize': 9,
    })


def plot_stress_factor_dynamics(
    time_grid: np.ndarray,
    stress_trajectory: np.ndarray,
    jump_times: Optional[np.ndarray] = None,
    save_path: str = "outputs/stress_dynamics.png"
):
    """
    Plot stress factor S_t dynamics over time.
    
    Shows how the stress factor responds to jumps and decays.
    
    Args:
        time_grid: Time points
        stress_trajectory: Stress factor values (n_samples, n_steps) or (n_steps,)
        jump_times: Optional array of jump occurrence times
        save_path: Path to save the figure
    """
    set_publication_style()
    _ensure_dir(save_path)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if stress_trajectory.ndim == 2:
        # Plot individual trajectories
        n_samples = min(20, stress_trajectory.shape[0])
        for i in range(n_samples):
            ax.plot(time_grid, stress_trajectory[i], 
                   color='#4DBBD5', alpha=0.3, linewidth=0.8)
        
        # Plot mean
        mean_stress = np.mean(stress_trajectory, axis=0)
        ax.plot(time_grid, mean_stress, 
               color='#E64B35', linewidth=2.5, label='Mean $S_t$')
        
        # Plot confidence interval
        std_stress = np.std(stress_trajectory, axis=0)
        ax.fill_between(time_grid, 
                       mean_stress - std_stress,
                       mean_stress + std_stress,
                       color='#E64B35', alpha=0.2, label='±1 Std')
    else:
        ax.plot(time_grid, stress_trajectory, 
               color='#E64B35', linewidth=2, label='$S_t$')
    
    # Mark jump times
    if jump_times is not None and len(jump_times) > 0:
        for jt in jump_times[:20]:  # Limit to first 20 jumps
            ax.axvline(x=jt, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.axvline(x=jump_times[0], color='gray', linestyle='--', 
                  alpha=0.5, linewidth=0.8, label='Jump Events')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Stress Factor $S_t$')
    ax.set_title('Transient Stress Factor Dynamics (Jump-Volatility Feedback)')
    ax.legend(loc='upper right')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")
    plt.close()


def plot_feedback_comparison(
    time_grid: np.ndarray,
    paths_without_feedback: np.ndarray,
    paths_with_feedback: np.ndarray,
    stress_trajectory: np.ndarray,
    save_path: str = "outputs/feedback_comparison.png"
):
    """
    Compare generated paths with and without feedback mechanism.
    
    Shows how feedback affects volatility dynamics.
    
    Args:
        time_grid: Time points
        paths_without_feedback: Generated paths without feedback (n_samples, n_steps)
        paths_with_feedback: Generated paths with feedback (n_samples, n_steps)
        stress_trajectory: Stress factor trajectory (n_samples, n_steps)
        save_path: Path to save the figure
    """
    set_publication_style()
    _ensure_dir(save_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: Sample paths comparison
    ax = axes[0, 0]
    n_plot = min(10, paths_without_feedback.shape[0])
    
    for i in range(n_plot):
        ax.plot(time_grid, paths_without_feedback[i], 
               color='#4DBBD5', alpha=0.4, linewidth=0.8)
        ax.plot(time_grid, paths_with_feedback[i], 
               color='#E64B35', alpha=0.4, linewidth=0.8)
    
    ax.plot([], [], color='#4DBBD5', linewidth=2, label='Without Feedback')
    ax.plot([], [], color='#E64B35', linewidth=2, label='With Feedback')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Sample Path Comparison')
    ax.legend()
    
    # Top-right: Rolling volatility comparison
    ax = axes[0, 1]
    window = max(5, len(time_grid) // 20)
    
    returns_no_fb = np.diff(paths_without_feedback, axis=1)
    returns_fb = np.diff(paths_with_feedback, axis=1)
    
    # Compute rolling volatility
    def rolling_vol(returns, window):
        n_samples, n_steps = returns.shape
        vol = np.zeros((n_samples, n_steps - window + 1))
        for i in range(n_steps - window + 1):
            vol[:, i] = np.std(returns[:, i:i+window], axis=1)
        return vol
    
    vol_no_fb = rolling_vol(returns_no_fb, window)
    vol_fb = rolling_vol(returns_fb, window)
    
    time_vol = time_grid[window:-1] if len(time_grid) > window else time_grid[:-1]
    
    ax.plot(time_vol[:len(np.mean(vol_no_fb, axis=0))], 
           np.mean(vol_no_fb, axis=0), 
           color='#4DBBD5', linewidth=2, linestyle='--', label='Without Feedback')
    ax.plot(time_vol[:len(np.mean(vol_fb, axis=0))], 
           np.mean(vol_fb, axis=0), 
           color='#E64B35', linewidth=2, linestyle='-', label='With Feedback')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rolling Volatility')
    ax.set_title('Volatility Dynamics Comparison')
    ax.legend()
    
    # Bottom-left: Stress factor
    ax = axes[1, 0]
    if stress_trajectory.ndim == 2:
        mean_stress = np.mean(stress_trajectory, axis=0)
        std_stress = np.std(stress_trajectory, axis=0)
        ax.plot(time_grid, mean_stress, color='#E64B35', linewidth=2)
        ax.fill_between(time_grid, 
                       mean_stress - std_stress,
                       mean_stress + std_stress,
                       color='#E64B35', alpha=0.2)
    else:
        ax.plot(time_grid, stress_trajectory, color='#E64B35', linewidth=2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Stress Factor $S_t$')
    ax.set_title('Stress Factor Evolution')
    ax.set_ylim(bottom=0)
    
    # Bottom-right: Volatility clustering (ACF of absolute returns)
    ax = axes[1, 1]
    
    def compute_acf(x, max_lag=15):
        acf = np.zeros(max_lag)
        x_centered = x - np.mean(x)
        var = np.var(x)
        if var < 1e-10:
            return acf
        for lag in range(max_lag):
            if lag >= len(x) - 1:
                break
            acf[lag] = np.mean(x_centered[:-lag-1] * x_centered[lag+1:]) / var
        return acf
    
    abs_returns_no_fb = np.abs(returns_no_fb).flatten()
    abs_returns_fb = np.abs(returns_fb).flatten()
    
    acf_no_fb = compute_acf(abs_returns_no_fb)
    acf_fb = compute_acf(abs_returns_fb)
    
    lags = np.arange(len(acf_no_fb))
    ax.plot(lags, acf_no_fb, color='#4DBBD5', marker='s', 
           linestyle='--', linewidth=2, label='Without Feedback')
    ax.plot(lags, acf_fb, color='#E64B35', marker='o', 
           linestyle='-', linewidth=2, label='With Feedback')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF of |Returns|')
    ax.set_title('Volatility Clustering (ARCH Effect)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")
    plt.close()


def plot_jump_volatility_interaction(
    returns: np.ndarray,
    stress_trajectory: np.ndarray,
    jump_mask: np.ndarray,
    save_path: str = "outputs/jump_vol_interaction.png"
):
    """
    Visualize the interaction between jumps and volatility.
    
    Shows how volatility responds after jump events.
    
    Args:
        returns: Return series (n_samples, n_steps)
        stress_trajectory: Stress factor (n_samples, n_steps)
        jump_mask: Boolean mask of jump locations (n_samples, n_steps)
        save_path: Path to save the figure
    """
    set_publication_style()
    _ensure_dir(save_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Scatter plot of jump size vs subsequent volatility
    ax = axes[0]
    
    # Find jump locations and compute subsequent volatility
    jump_sizes = []
    subsequent_vols = []
    
    for i in range(returns.shape[0]):
        jump_idx = np.where(jump_mask[i])[0]
        for j in jump_idx:
            if j < returns.shape[1] - 5:  # Need at least 5 steps after
                jump_sizes.append(np.abs(returns[i, j]))
                subsequent_vols.append(np.std(returns[i, j+1:j+6]))
    
    if len(jump_sizes) > 0:
        ax.scatter(jump_sizes, subsequent_vols, alpha=0.5, s=30, c='#E64B35')
        
        # Fit trend line
        z = np.polyfit(jump_sizes, subsequent_vols, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(jump_sizes), max(jump_sizes), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
        ax.legend()
    
    ax.set_xlabel('|Jump Size|')
    ax.set_ylabel('Subsequent Volatility')
    ax.set_title('Jump Size vs Subsequent Volatility')
    
    # Middle: Event study around jumps
    ax = axes[1]
    
    window = 10  # Steps before and after jump
    vol_around_jump = []
    
    for i in range(returns.shape[0]):
        jump_idx = np.where(jump_mask[i])[0]
        for j in jump_idx:
            if j >= window and j < returns.shape[1] - window:
                vol_window = np.abs(returns[i, j-window:j+window+1])
                vol_around_jump.append(vol_window)
    
    if len(vol_around_jump) > 0:
        vol_around_jump = np.array(vol_around_jump)
        mean_vol = np.mean(vol_around_jump, axis=0)
        std_vol = np.std(vol_around_jump, axis=0)
        
        x_axis = np.arange(-window, window + 1)
        ax.plot(x_axis, mean_vol, color='#E64B35', linewidth=2, marker='o')
        ax.fill_between(x_axis, mean_vol - std_vol, mean_vol + std_vol,
                       color='#E64B35', alpha=0.2)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Jump Event')
        ax.legend()
    
    ax.set_xlabel('Steps Relative to Jump')
    ax.set_ylabel('|Return|')
    ax.set_title('Event Study: Volatility Around Jumps')
    
    # Right: Stress factor response
    ax = axes[2]
    
    stress_around_jump = []
    
    for i in range(stress_trajectory.shape[0]):
        jump_idx = np.where(jump_mask[i])[0]
        for j in jump_idx:
            if j >= window and j < stress_trajectory.shape[1] - window:
                stress_window = stress_trajectory[i, j-window:j+window+1]
                stress_around_jump.append(stress_window)
    
    if len(stress_around_jump) > 0:
        stress_around_jump = np.array(stress_around_jump)
        mean_stress = np.mean(stress_around_jump, axis=0)
        std_stress = np.std(stress_around_jump, axis=0)
        
        ax.plot(x_axis, mean_stress, color='#4DBBD5', linewidth=2, marker='s')
        ax.fill_between(x_axis, mean_stress - std_stress, mean_stress + std_stress,
                       color='#4DBBD5', alpha=0.2)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, label='Jump Event')
        ax.legend()
    
    ax.set_xlabel('Steps Relative to Jump')
    ax.set_ylabel('Stress Factor $S_t$')
    ax.set_title('Stress Factor Response to Jumps')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")
    plt.close()


def plot_model_comparison_grid(
    real_data: np.ndarray,
    generated_dict: Dict[str, np.ndarray],
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str = "outputs/model_comparison.png"
):
    """
    Create comprehensive model comparison grid.
    
    Uses different colors, line styles, and markers for black-and-white compatibility.
    
    Args:
        real_data: Real time series (n_samples, n_steps, n_features)
        generated_dict: Dictionary of generated data by model name
        metrics_dict: Dictionary of metrics by model name
        save_path: Path to save the figure
    """
    set_publication_style()
    _ensure_dir(save_path)
    
    n_models = len(generated_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols + 2  # Extra rows for metrics
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(n_rows, n_cols)
    
    # Collapse multifeature paths for path overlays by averaging features.
    if real_data.ndim == 3:
        real_flat = np.mean(real_data, axis=2)
    else:
        real_flat = real_data
    
    # Row 1-2: Sample paths for each model
    for idx, (name, gen_data) in enumerate(generated_dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        style = _get_style(name)
        
        # Collapse multifeature generated paths the same way as real paths.
        if gen_data.ndim == 3:
            gen_flat = np.mean(gen_data, axis=2)
        else:
            gen_flat = gen_data
        
        # Plot real data (gray background)
        n_plot = min(20, real_flat.shape[0])
        for i in range(n_plot):
            ax.plot(real_flat[i], color='gray', alpha=0.2, linewidth=0.5)
        
        # Plot generated data
        n_plot = min(10, gen_flat.shape[0])
        for i in range(n_plot):
            ax.plot(gen_flat[i], color=style['color'], alpha=0.4, 
                   linewidth=0.8, linestyle=style['linestyle'])
        
        # Plot means
        ax.plot(np.mean(real_flat, axis=0), 'k--', linewidth=2, label='Real Mean')
        ax.plot(np.mean(gen_flat, axis=0), color=style['color'], 
               linewidth=2, linestyle=style['linestyle'], 
               marker=style['marker'], markevery=max(1, len(gen_flat[0])//10),
               label=f'{name} Mean')
        
        ax.set_title(name, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
    
    # Fill empty cells
    for idx in range(len(generated_dict), n_cols * (n_rows - 2)):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # Bottom rows: Metrics comparison
    used_metric_axes = set()
    if metrics_dict:
        # Bar chart of WD
        ax = axes[-2, 0]
        used_metric_axes.add((n_rows - 2, 0))
        models = list(metrics_dict.keys())
        wd_values = [metrics_dict[m].get('wasserstein_distance', 0) for m in models]
        colors = [_get_style(m)['color'] for m in models]
        hatches = [_get_style(m)['hatch'] for m in models]
        
        bars = ax.bar(range(len(models)), wd_values, color=colors)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Wasserstein Distance')
        ax.set_title('Distribution Quality (↓ better)')
        
        # Bar chart of ACF MSE
        ax = axes[-2, 1] if n_cols > 1 else axes[-1, 0]
        used_metric_axes.add((n_rows - 2, 1) if n_cols > 1 else (n_rows - 1, 0))
        acf_values = [metrics_dict[m].get('acf_mse', 0) for m in models]
        
        bars = ax.bar(range(len(models)), acf_values, color=colors)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('ACF MSE')
        ax.set_title('Temporal Dynamics (↓ better)')
        
        # Radar chart or additional metrics
        if n_cols > 2:
            ax = axes[-2, 2]
            used_metric_axes.add((n_rows - 2, 2))
            disc_values = [metrics_dict[m].get('discriminative_score', 0.5) for m in models]
            
            bars = ax.bar(range(len(models)), disc_values, color=colors)
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)
            
            ax.axhline(y=0.0, color='green', linestyle='--', label='Ideal')
            ax.axhline(y=0.5, color='red', linestyle='--', label='Worst')
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel('Discriminative Score')
            ax.set_title('Realism (↓ better)')
            ax.legend()
    
    # Hide remaining axes
    for row in range(n_rows - 2, n_rows):
        for col in range(n_cols):
            if (row, col) not in used_metric_axes:
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")
    plt.close()


def create_legend_figure(
    save_path: str = "outputs/method_legend.png"
):
    """
    Create a standalone legend figure for all methods.
    
    Useful for papers where legend needs to be separate.
    
    Args:
        save_path: Path to save the figure
    """
    set_publication_style()
    _ensure_dir(save_path)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    
    handles = []
    labels = []
    
    for name, style in STYLE_CONFIG.items():
        line = plt.Line2D([0], [0], 
                         color=style['color'],
                         linestyle=style['linestyle'],
                         marker=style['marker'],
                         linewidth=2,
                         markersize=8)
        handles.append(line)
        labels.append(name)
    
    ax.legend(handles, labels, loc='center', ncol=4, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")
    plt.close()
