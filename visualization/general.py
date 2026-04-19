import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os
import pandas as pd

# --- Nature-Style Configuration ---
def set_style():
    """Configures Matplotlib for Nature-journal quality."""
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    global STYLE_MAP
    STYLE_MAP = {
        'Real':    {'color': 'black',   'ls': '--', 'marker': 'o', 'label': 'Real'},
        'Kernel':  {'color': '#4DBBD5', 'ls': '-',  'marker': 's', 'label': 'Kernel'}, 
        'SBTS-LSTM': {'color': '#E64B35', 'ls': '--', 'marker': '^', 'label': 'LSTM'},   
        'LSTM':      {'color': '#E64B35', 'ls': '--', 'marker': '^', 'label': 'LSTM'},   
        'LIGHTSB': {'color': '#00A087', 'ls': ':',  'marker': 'x', 'label': 'LightSB'}, 
        'LightSB': {'color': '#00A087', 'ls': ':',  'marker': 'x', 'label': 'LightSB'},
        'lightsb': {'color': '#00A087', 'ls': ':',  'marker': 'x', 'label': 'LightSB'},
        'timegan': {'color': '#3C5488', 'ls': '-',  'marker': 'D', 'label': 'TimeGAN'},
        'diffusion_ts': {'color': '#9467BD', 'ls': '-', 'marker': 'P', 'label': 'Diffusion-TS'},
        'rnn': {'color': '#D55E00', 'ls': '--', 'marker': 's', 'label': 'RNN'},
        'transformer_ar': {'color': '#009E73', 'ls': '-.', 'marker': '^', 'label': 'Transformer-AR'},
    }
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300,
        'legend.frameon': False,
    })

def _ensure_dir(path):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

def _as_plot_series(paths):
    """Collapse multifeature paths to one display series by averaging features."""
    paths = np.asarray(paths)
    if paths.ndim == 3:
        return np.mean(paths, axis=2)
    return paths

def _terminal_values(paths):
    """Return terminal values across all features for distribution plots."""
    paths = np.asarray(paths)
    if paths.ndim == 3:
        return paths[:, -1, :].reshape(-1)
    return paths[:, -1]

def _returns_values(paths):
    """Return flattened time returns across all features."""
    paths = np.asarray(paths)
    return np.diff(paths, axis=1).reshape(-1)

def _calc_autocorr(paths, max_lag=20):
    if paths.ndim == 3:
        series = paths.reshape(-1, paths.shape[1])
    else:
        series = paths
        
    acfs = []
    for i in range(max_lag):
        if i >= series.shape[1] - 1: acfs.append(0); continue
        val_t = series[:, :-i-1].flatten()
        val_tk = series[:, i+1:].flatten()
        mask = ~np.isnan(val_t) & ~np.isnan(val_tk)
        if np.sum(mask) > 10:
            acfs.append(np.corrcoef(val_t[mask], val_tk[mask])[0, 1])
        else:
            acfs.append(0)
    return np.array(acfs)

# --- Performance Metrics Bar Chart ---
def plot_performance_metrics(df_metrics, save_path="outputs/performance_metrics.png"):
    """
    Plots 1x4 bar charts for Training Time, Inference Time, WD, and ACF Error.
    """
    set_style()
    _ensure_dir(save_path)
    df_plot = df_metrics.copy()
    if 'WD' not in df_plot.columns and 'wasserstein_distance' in df_plot.columns:
        df_plot['WD'] = df_plot['wasserstein_distance']
    if 'ACF_MSE' not in df_plot.columns and 'acf_mse' in df_plot.columns:
        df_plot['ACF_MSE'] = df_plot['acf_mse']
    
    metrics = ['train_time', 'gen_time', 'WD', 'ACF_MSE']
    titles = ['Training Time (s)', 'Generation Time (s)', 'Wasserstein Dist (Lower Better)', 'ACF MSE (Lower Better)']
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    palette = [STYLE_MAP.get(idx, {'color': 'gray'})['color'] for idx in df_plot.index]
    
    for i, metric in enumerate(metrics):
        if metric in df_plot.columns:
            # --- FIX: Added hue and legend=False to silence warning ---
            sns.barplot(
                x=df_plot.index, 
                y=df_plot[metric], 
                hue=df_plot.index, # Assign x to hue
                ax=axes[i], 
                palette=palette, 
                legend=False          # Disable legend to mimic old behavior
            )
            axes[i].set_title(titles[i], fontweight='bold')
            axes[i].set_ylabel('')
            axes[i].set_xlabel('')
            
            for p in axes[i].patches:
                axes[i].annotate(f'{p.get_height():.4f}', 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                xytext = (0, 9), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

# --- Comprehensive Comparison ---
def plot_comprehensive_comparison(time_grid, real_returns, results_store, real_prices=None, gen_prices_dict=None, save_path="outputs/full_comparison.png"):
    set_style()
    _ensure_dir(save_path)
    
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
    methods = list(results_store.keys())
    plot_methods = methods[:3] 
    
    if real_prices is not None and gen_prices_dict is not None:
        for i, method in enumerate(plot_methods):
            ax = axes[0, i]
            style = STYLE_MAP.get(method, {'color':'blue'})
            gen_data = gen_prices_dict.get(method)
            
            real_plot = _as_plot_series(real_prices)
            ax.plot(real_plot[:50].T, color='gray', alpha=0.1, linewidth=0.8)
            if gen_data is not None and not np.isnan(gen_data).any():
                gen_plot = _as_plot_series(gen_data)
                ax.plot(gen_plot[:15].T, color=style['color'], alpha=0.6, linewidth=1.0)
                ax.plot(np.mean(gen_plot, axis=0), color=style['color'], linewidth=2.5, label=f'{method} Mean')
            
            ax.plot(np.mean(real_plot, axis=0), 'k--', linewidth=2.5, label='Real Mean')
            ax.set_title(f"Price Reconstruction: Real vs {method}", fontweight='bold')
            ax.legend(loc='upper left')
            
            y_min, y_max = np.min(real_prices), np.max(real_prices)
            margin = (y_max - y_min) * 0.5
            ax.set_ylim(y_min - margin, y_max + margin)

    ax_dens = axes[1, 0]
    sns.kdeplot(_terminal_values(real_returns), ax=ax_dens, color='gray', fill=True, alpha=0.3, label='Real')
    
    for method in methods:
        paths = results_store[method]
        if np.isnan(paths).any(): continue
        style = STYLE_MAP.get(method, {'color':'blue', 'ls':'-'})
        sns.kdeplot(_terminal_values(paths), ax=ax_dens, color=style['color'], linestyle=style['ls'], linewidth=2, label=method)
    
    ax_dens.set_title("Log-Returns Density (Fat Tails Check)", fontweight='bold')
    ax_dens.legend()

    ax_acf = axes[1, 1]
    lag = np.arange(15)
    real_acf = _calc_autocorr(real_returns, 15)
    ax_acf.plot(lag, real_acf, color='black', marker='o', linestyle='-', label='Real')
    
    for method in methods:
        paths = results_store[method]
        if np.isnan(paths).any(): continue
        style = STYLE_MAP.get(method, {'color':'blue', 'marker':'x', 'ls':'--'})
        gen_acf = _calc_autocorr(paths, 15)
        ax_acf.plot(lag, gen_acf, color=style['color'], marker=style['marker'], linestyle=style['ls'], label=method)
        
    ax_acf.set_title("Temporal Dynamics (ACF)", fontweight='bold')
    ax_acf.legend()

    ax_vol = axes[1, 2]
    real_abs_acf = _calc_autocorr(np.abs(real_returns), 15)
    ax_vol.plot(lag, real_abs_acf, color='black', marker='o', linestyle='-', label='Real |r|')
    
    method = 'SBTS-LSTM'
    if method in results_store:
        paths = results_store[method]
        style = STYLE_MAP.get(method, {'color':'red'})
        gen_abs_acf = _calc_autocorr(np.abs(paths), 15)
        ax_vol.plot(lag, gen_abs_acf, color=style['color'], marker='^', linestyle='--', label=f'{method} |r|')
        
    ax_vol.set_title("Volatility Clustering (Stylized Fact)", fontweight='bold')
    ax_vol.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

def plot_bandwidth_optimization(mse_history, selected_h, save_path="outputs/bandwidth_tuning.png"):
    set_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    hs = sorted(list(mse_history.keys()))
    mses = [mse_history[h] for h in hs]
    ax.plot(hs, mses, marker='o', markersize=5, color='#3C5488', label='CV MSE')
    ax.axvline(x=selected_h, color='#E64B35', linestyle='--', label=f'Optimal h={selected_h:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel("Bandwidth (h)")
    ax.set_ylabel("MSE")
    ax.set_title("Bandwidth Optimization")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_correlation_distribution(real_data, gen_dict, save_path="outputs/corr_dist.png"):
    set_style()
    _ensure_dir(save_path)
    N, T, D = real_data.shape
    if D < 2: return

    real_flat = real_data.reshape(-1, D)
    real_corr = np.corrcoef(real_flat, rowvar=False)
    real_vals = real_corr[np.triu_indices_from(real_corr, k=1)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(real_vals, color='gray', fill=True, alpha=0.3, label='Real')
    
    for name, paths in gen_dict.items():
        if np.isnan(paths).any(): continue
        flat = paths.reshape(-1, D)
        corr = np.corrcoef(flat, rowvar=False)
        vals = corr[np.triu_indices_from(corr, k=1)]
        style = STYLE_MAP.get(name, {'color':'blue'})
        sns.kdeplot(vals, color=style['color'], label=name)
        
    ax.set_title(f"Pairwise Correlation Distribution ({D} Assets)")
    ax.set_xlabel("Correlation Coefficient")
    ax.set_xlim(-1, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_price_index_comparison(real_prices, gen_prices_dict, save_path="outputs/price_index.png"):
    set_style()
    _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    real_index = np.mean(real_prices, axis=2) 
    ax.plot(real_index[:50].T, color='gray', alpha=0.1)
    ax.plot(np.mean(real_index, axis=0), color='black', linestyle='--', linewidth=2.5, label='Real Index Mean')
    
    for name, paths in gen_prices_dict.items():
        if np.isnan(paths).any(): continue
        gen_index = np.mean(paths, axis=2)
        style = STYLE_MAP.get(name, {'color':'blue'})
        ax.plot(np.mean(gen_index, axis=0), color=style['color'], linewidth=2.5, label=name)
        
    ax.set_title("Market Index Reconstruction (Equal Weighted)", fontweight='bold')
    ax.set_ylabel("Price (Normalized)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

# --- Volatility Surface Visualization ---
def plot_volatility_surface(vol_model, data_range, T=1.0, save_path="outputs/vol_surface.png"):
    """
    Plots the calibrated Local Volatility Surface sigma(t, x).
    Handles multi-asset models by setting non-target assets to mean (0).
    """
    set_style()
    _ensure_dir(save_path)

    if hasattr(vol_model, 'get_surface'):
        try:
            t_grid_model, x_grid_model, vol_surface = vol_model.get_surface()
            Vol_mesh = np.mean(vol_surface, axis=2) if vol_surface.ndim == 3 else vol_surface
            T_mesh, X_mesh = np.meshgrid(t_grid_model, x_grid_model, indexing='ij')

            fig, ax = plt.subplots(figsize=(8, 6))
            cp = ax.contourf(T_mesh, X_mesh, Vol_mesh, 20, cmap='magma')
            cbar = fig.colorbar(cp)
            cbar.set_label("Local Volatility $\\sigma(t, x)$")
            ax.set_title("Calibrated Local Volatility Surface (Asset 1 Slice)", fontweight='bold')
            ax.set_xlabel("Time (t)")
            ax.set_ylabel("State (x)")
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"[Plot Saved] {save_path}")
            plt.close()
            return
        except Exception as e:
            print(f"[Warning] Could not plot active vol surface directly: {e}")
    
    # 1. Create Grid
    t_grid = np.linspace(0, T, 50)
    x_min, x_max = np.min(data_range), np.max(data_range)
    x_grid = np.linspace(x_min, x_max, 50)
    
    T_mesh, X_mesh = np.meshgrid(t_grid, x_grid)
    
    t_flat = T_mesh.ravel()
    x_flat = X_mesh.ravel() # shape (N_grid,)
    
    try:
        # Check expected input dimension from the scaler
        # n_features_in_ = 1 (time) + D (assets)
        expected_dim = vol_model.scaler_X.n_features_in_
        n_assets = expected_dim - 1
        
        # Prepare x input: (N_grid, n_assets)
        # We vary the first asset (or all assuming homogeneity) and set others to 0 (mean return)
        x_input = np.zeros((len(x_flat), n_assets))
        
        # Set the first dimension to our grid values
        # This visualizes the volatility slice w.r.t the first asset's movement
        x_input[:, 0] = x_flat
        
        # Predict
        # predict handles t as vector (N_grid,) and x as (N_grid, n_assets)
        # It returns (N_grid, n_assets) volatility vector
        vol_pred = vol_model.predict(t_flat, x_input)
        
        # Take the volatility of the first asset
        vol_flat = vol_pred[:, 0]
        
        Vol_mesh = vol_flat.reshape(T_mesh.shape)
        
    except Exception as e:
        print(f"[Warning] Could not plot vol surface: {e}")
        return

    # 3. Plot Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    cp = ax.contourf(T_mesh, X_mesh, Vol_mesh, 20, cmap='magma')
    cbar = fig.colorbar(cp)
    cbar.set_label("Local Volatility $\\sigma(t, x)$")
    
    ax.set_title("Calibrated Local Volatility Surface (Asset 1 Slice)", fontweight='bold')
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Log Return State (x)")
    
    ax.axvline(T/2, color='white', linestyle='--', alpha=0.3)
    ax.text(T/2, x_min, " T/2 Slice", color='white', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

def plot_multi_asset_jumps(dates, continuous_returns, tickers, threshold_std=4.0, save_path="outputs/jump_visualization.png"):
    """
    Visualizes continuous log-returns and highlights detected jumps with REAL DATES.
    """
    set_style()
    _ensure_dir(save_path)
    
    n_assets = continuous_returns.shape[1]
    n_plot = min(n_assets, 5) # Plot max 5 assets
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True)
    if n_plot == 1: axes = [axes]
    
    # Calculate thresholds based on the full continuous series
    flat_data = continuous_returns.flatten()
    sigma_robust = np.median(np.abs(flat_data)) / 0.6745
    # Assuming daily data approx
    dt = 1.0/252.0
    threshold_val = threshold_std * sigma_robust * np.sqrt(dt)
    
    for i in range(n_plot):
        ax = axes[i]
        ticker = tickers[i]
        
        # Data for this asset
        series = continuous_returns[:, i]
        
        # Identify Jumps
        is_jump = np.abs(series) > threshold_val
        
        # Plot Returns
        ax.plot(dates, series, color='#333333', linewidth=0.8, label=f'{ticker} Returns')
        
        # Plot Threshold Bounds
        ax.axhline(threshold_val, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Threshold')
        ax.axhline(-threshold_val, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        
        # Highlight Jumps
        if np.any(is_jump):
            ax.scatter(dates[is_jump], series[is_jump], 
                       color='#E64B35', s=20, zorder=5, label='Detected Jump')
            
        ax.set_title(f"{ticker} Jump Detection (Threshold = {threshold_std}$\\sigma$)", fontweight='bold')
        ax.set_ylabel("Log Return")
        
        # Format X-Axis as Years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        
        if i == 0:
            ax.legend(loc='upper right', frameon=True)
        ax.grid(True, alpha=0.2)
        
    plt.xlabel("Time (Year)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

def plot_zoomed_crisis(dates, returns, tickers, threshold_std=4.0, save_path="outputs/jump_crisis_zoom.png"):
    """
    Zooms in on the 2020 COVID-19 period to show jumps clearly.
    """
    set_style()
    _ensure_dir(save_path)
    
    # Filter Data for 2020
    # Assuming dates is a pandas Index or array of datetimes
    if isinstance(dates, pd.DatetimeIndex) or isinstance(dates, np.ndarray):
        df_dates = pd.to_datetime(dates)
        mask = (df_dates >= "2020-01-01") & (df_dates <= "2020-06-30")
        
        if np.sum(mask) == 0:
            print("[Warning] No data found in 2020 for zoom plot.")
            return
            
        dates_zoom = dates[mask]
        returns_zoom = returns[mask]
    else:
        return

    n_assets = returns.shape[1]
    n_plot = min(n_assets, 3) # Only plot top 3 assets to save space
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True)
    if n_plot == 1: axes = [axes]
    
    # Recalculate threshold
    flat_data = returns.flatten()
    sigma_robust = np.median(np.abs(flat_data)) / 0.6745
    threshold_val = threshold_std * sigma_robust * np.sqrt(1.0/252.0)
    
    for i in range(n_plot):
        ax = axes[i]
        ticker = tickers[i]
        series = returns_zoom[:, i]
        
        # Plot Bar chart for returns (better for daily view)
        # Colors: Red for negative, Green for positive
        colors = np.where(series >= 0, '#00A087', '#E64B35')
        ax.bar(dates_zoom, series, color=colors, alpha=0.6, width=1.0)
        
        # Plot Threshold lines
        ax.axhline(threshold_val, color='orange', linestyle='--', linewidth=1)
        ax.axhline(-threshold_val, color='orange', linestyle='--', linewidth=1)
        
        # Highlight Jumps
        is_jump = np.abs(series) > threshold_val
        if np.any(is_jump):
            ax.scatter(dates_zoom[is_jump], series[is_jump], 
                       color='black', s=20, marker='x', label='Jump Event', zorder=10)

        ax.set_title(f"{ticker} COVID-19 Crisis View (2020 H1)", fontweight='bold')
        ax.set_ylabel("Daily Log Return")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        
    plt.xlabel("Date (2020)")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

def plot_jumps_on_price(dates, returns, tickers, threshold_std=4.0, save_path="outputs/jump_on_price.png"):
    """
    Visualizes Price Paths and highlights where jumps occurred.
    Much more intuitive than looking at raw returns.
    """
    set_style()
    _ensure_dir(save_path)
    
    n_assets = returns.shape[1]
    n_plot = min(n_assets, 5)
    
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 4 * n_plot), sharex=True)
    if n_plot == 1: axes = [axes]
    
    # Global threshold calculation
    flat_data = returns.flatten()
    sigma_robust = np.median(np.abs(flat_data)) / 0.6745
    # Daily vol approx
    dt = 1.0/252.0
    threshold_val = threshold_std * sigma_robust * np.sqrt(dt)
    
    for i in range(n_plot):
        ax = axes[i]
        ticker = tickers[i]
        
        # 1. Reconstruct Price (Normalized to 100)
        r_series = returns[:, i]
        price_series = 100 * np.exp(np.cumsum(r_series))
        
        # 2. Identify Jump Indices
        is_jump = np.abs(r_series) > threshold_val
        
        # 3. Plot Price Line
        ax.plot(dates, price_series, color='#333333', linewidth=1.5, label=f'{ticker} Price')
        
        # 4. Highlight Jump Days
        # We plot dots ON the price line where jumps happened
        if np.any(is_jump):
            # Split into positive and negative jumps for color
            pos_jumps = (r_series > threshold_val)
            neg_jumps = (r_series < -threshold_val)
            
            if np.any(pos_jumps):
                ax.scatter(dates[pos_jumps], price_series[pos_jumps], 
                           color='#00A087', s=40, zorder=5, marker='^', label='Positive Jump')
            if np.any(neg_jumps):
                ax.scatter(dates[neg_jumps], price_series[neg_jumps], 
                           color='#E64B35', s=40, zorder=5, marker='v', label='Negative Jump')

        ax.set_title(f"{ticker} Price Evolution with Jumps (Threshold={threshold_std}$\\sigma$)", fontweight='bold')
        ax.set_ylabel("Price ($)")
        
        # Highlight 2020 Crisis Area
        # Convert date strings to datetime if needed, or rely on matplotlib handling
        # Assuming dates are datetime objects or recognizable
        try:
            crisis_start = pd.Timestamp("2020-02-20")
            crisis_end = pd.Timestamp("2020-04-30")
            ax.axvspan(crisis_start, crisis_end, color='red', alpha=0.1, label='COVID-19 Crash')
        except:
            pass # Skip shading if date format issues
            
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.2)
        
        # Date Formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot Saved] {save_path}")
    plt.close()

# ==============================================================================
# NEW: SOTA Comparison Visualizations
# ==============================================================================

# Extended style map for SOTA methods
EXTENDED_STYLE_MAP = {
    'Real':           {'color': 'black',   'ls': '--', 'marker': 'o', 'label': 'Real'},
    'JD-SBTS-Neural': {'color': '#E64B35', 'ls': '-',  'marker': '^', 'label': 'JD-SBTS (Neural)'},
    'JD-SBTS-Static': {'color': '#F39B7F', 'ls': '--', 'marker': 'v', 'label': 'JD-SBTS (Static)'},
    'Numba-SB':       {'color': '#4DBBD5', 'ls': '-',  'marker': 's', 'label': 'Numba-SB'},
    'LightSB':        {'color': '#00A087', 'ls': ':',  'marker': 'x', 'label': 'LightSB'},
    'TimeGAN':        {'color': '#3C5488', 'ls': '-',  'marker': 'D', 'label': 'TimeGAN'},
    'Diffusion-TS':   {'color': '#9467BD', 'ls': '-',  'marker': 'P', 'label': 'Diffusion-TS'},
    'rnn':            {'color': '#D55E00', 'ls': '--', 'marker': 's', 'label': 'RNN'},
    'transformer_ar': {'color': '#009E73', 'ls': '-.', 'marker': '^', 'label': 'Transformer-AR'},
    'Kernel':         {'color': '#8C564B', 'ls': '--', 'marker': 'o', 'label': 'Kernel'},
    'SBTS-LSTM':      {'color': '#E64B35', 'ls': '--', 'marker': '^', 'label': 'LSTM'},
}


def plot_sota_comparison(real_data, results_store, metrics_store, save_path="outputs/sota_comparison.png"):
    """
    Comprehensive SOTA comparison visualization.
    
    Creates a 3x3 grid showing:
    - Row 1: Sample paths for top 3 methods
    - Row 2: Distribution comparisons (terminal, returns, volatility)
    - Row 3: Metrics comparison (bar charts)
    
    Args:
        real_data: (N, T, D) real time series
        results_store: Dict of {method_name: generated_paths}
        metrics_store: Dict of {method_name: {metric: value}}
        save_path: Path to save the figure
    """
    set_style()
    _ensure_dir(save_path)
    
    fig = plt.figure(figsize=(20, 16))
    
    methods = list(results_store.keys())
    n_methods = len(methods)
    
    # =========================================
    # Row 1: Sample Paths (3 columns)
    # =========================================
    top_methods = methods[:min(3, n_methods)]
    
    for i, method in enumerate(top_methods):
        ax = fig.add_subplot(3, 3, i + 1)
        style = EXTENDED_STYLE_MAP.get(method, {'color': 'blue', 'ls': '-'})
        gen_data = results_store[method]
        real_plot = _as_plot_series(real_data)
        gen_plot = _as_plot_series(gen_data)
        
        # Plot real paths (gray background)
        n_plot = min(30, len(real_plot))
        for j in range(n_plot):
            ax.plot(real_plot[j], color='gray', alpha=0.1, linewidth=0.5)
        
        # Plot generated paths
        if not np.isnan(gen_data).any():
            n_gen_plot = min(15, len(gen_plot))
            for j in range(n_gen_plot):
                ax.plot(gen_plot[j], color=style['color'], alpha=0.4, linewidth=0.8)
            
            # Mean paths
            ax.plot(np.mean(real_plot, axis=0), 'k--', linewidth=2, label='Real Mean')
            ax.plot(np.mean(gen_plot, axis=0), color=style['color'], linewidth=2, label=f'{method} Mean')
        
        ax.set_title(f'{method}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper left', fontsize=8)
    
    # =========================================
    # Row 2: Distribution Comparisons
    # =========================================
    
    # 2.1 Terminal Distribution
    ax_term = fig.add_subplot(3, 3, 4)
    sns.kdeplot(_terminal_values(real_data), ax=ax_term, color='black', fill=True, alpha=0.3, label='Real', linewidth=2)
    
    for method in methods:
        gen_data = results_store[method]
        if np.isnan(gen_data).any():
            continue
        style = EXTENDED_STYLE_MAP.get(method, {'color': 'blue', 'ls': '-'})
        sns.kdeplot(_terminal_values(gen_data), ax=ax_term, color=style['color'], 
                    linestyle=style['ls'], linewidth=2, label=method)
    
    ax_term.set_title('Terminal Distribution', fontweight='bold')
    ax_term.set_xlabel('Value')
    ax_term.legend(fontsize=8)
    
    # 2.2 Returns Distribution (Fat Tails)
    ax_ret = fig.add_subplot(3, 3, 5)
    
    # Compute returns
    real_returns = _returns_values(real_data)
    sns.kdeplot(real_returns, ax=ax_ret, color='black', fill=True, alpha=0.3, label='Real', linewidth=2)
    
    for method in methods:
        gen_data = results_store[method]
        if np.isnan(gen_data).any():
            continue
        style = EXTENDED_STYLE_MAP.get(method, {'color': 'blue', 'ls': '-'})
        gen_returns = _returns_values(gen_data)
        sns.kdeplot(gen_returns, ax=ax_ret, color=style['color'], 
                    linestyle=style['ls'], linewidth=2, label=method)
    
    ax_ret.set_title('Returns Distribution (Fat Tails)', fontweight='bold')
    ax_ret.set_xlabel('Return')
    ax_ret.legend(fontsize=8)
    
    # 2.3 ACF Comparison
    ax_acf = fig.add_subplot(3, 3, 6)
    max_lag = 15
    lags = np.arange(max_lag)
    
    real_acf = _calc_autocorr(real_data, max_lag)
    ax_acf.plot(lags, real_acf, 'ko-', linewidth=2, markersize=6, label='Real')
    
    for method in methods:
        gen_data = results_store[method]
        if np.isnan(gen_data).any():
            continue
        style = EXTENDED_STYLE_MAP.get(method, {'color': 'blue', 'marker': 'x', 'ls': '-'})
        gen_acf = _calc_autocorr(gen_data, max_lag)
        ax_acf.plot(lags, gen_acf, color=style['color'], marker=style['marker'], 
                    linestyle=style['ls'], linewidth=1.5, markersize=5, label=method)
    
    ax_acf.set_title('Autocorrelation Function', fontweight='bold')
    ax_acf.set_xlabel('Lag')
    ax_acf.set_ylabel('ACF')
    ax_acf.legend(fontsize=8)
    
    # =========================================
    # Row 3: Metrics Bar Charts
    # =========================================
    
    # Convert metrics to DataFrame
    df_metrics = pd.DataFrame(metrics_store).T
    
    # 3.1 Wasserstein Distance
    ax_wd = fig.add_subplot(3, 3, 7)
    if 'WD' in df_metrics.columns:
        colors = [EXTENDED_STYLE_MAP.get(m, {'color': 'gray'})['color'] for m in df_metrics.index]
        bars = ax_wd.bar(range(len(df_metrics)), df_metrics['WD'].values, color=colors)
        ax_wd.set_xticks(range(len(df_metrics)))
        ax_wd.set_xticklabels(df_metrics.index, rotation=45, ha='right', fontsize=9)
        ax_wd.set_title('Wasserstein Distance ↓', fontweight='bold')
        ax_wd.set_ylabel('WD')
        
        # Add value labels
        for bar, val in zip(bars, df_metrics['WD'].values):
            if not np.isnan(val):
                ax_wd.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                          f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 3.2 ACF MSE
    ax_acf_mse = fig.add_subplot(3, 3, 8)
    if 'ACF_MSE' in df_metrics.columns:
        colors = [EXTENDED_STYLE_MAP.get(m, {'color': 'gray'})['color'] for m in df_metrics.index]
        bars = ax_acf_mse.bar(range(len(df_metrics)), df_metrics['ACF_MSE'].values, color=colors)
        ax_acf_mse.set_xticks(range(len(df_metrics)))
        ax_acf_mse.set_xticklabels(df_metrics.index, rotation=45, ha='right', fontsize=9)
        ax_acf_mse.set_title('ACF MSE ↓', fontweight='bold')
        ax_acf_mse.set_ylabel('MSE')
        
        for bar, val in zip(bars, df_metrics['ACF_MSE'].values):
            if not np.isnan(val):
                ax_acf_mse.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                               f'{val:.6f}', ha='center', va='bottom', fontsize=8)
    
    # 3.3 Discriminative Score (if available)
    ax_disc = fig.add_subplot(3, 3, 9)
    if 'Disc_Score' in df_metrics.columns:
        colors = [EXTENDED_STYLE_MAP.get(m, {'color': 'gray'})['color'] for m in df_metrics.index]
        bars = ax_disc.bar(range(len(df_metrics)), df_metrics['Disc_Score'].values, color=colors)
        ax_disc.set_xticks(range(len(df_metrics)))
        ax_disc.set_xticklabels(df_metrics.index, rotation=45, ha='right', fontsize=9)
        ax_disc.set_title('Discriminative Score ↓', fontweight='bold')
        ax_disc.set_ylabel('Score')
        ax_disc.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Ideal')
        
        for bar, val in zip(bars, df_metrics['Disc_Score'].values):
            if not np.isnan(val):
                ax_disc.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                            f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    else:
        # Show training time instead
        if 'train_time' in df_metrics.columns:
            colors = [EXTENDED_STYLE_MAP.get(m, {'color': 'gray'})['color'] for m in df_metrics.index]
            bars = ax_disc.bar(range(len(df_metrics)), df_metrics['train_time'].values, color=colors)
            ax_disc.set_xticks(range(len(df_metrics)))
            ax_disc.set_xticklabels(df_metrics.index, rotation=45, ha='right', fontsize=9)
            ax_disc.set_title('Training Time (s)', fontweight='bold')
            ax_disc.set_ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")
    plt.close()


def plot_discriminative_results(metrics_store, save_path="outputs/discriminative_results.png"):
    """
    Plot discriminative and predictive score comparison.
    
    Args:
        metrics_store: Dict of {method_name: {metric: value}}
        save_path: Path to save the figure
    """
    set_style()
    _ensure_dir(save_path)
    
    df_metrics = pd.DataFrame(metrics_store).T
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Discriminative Score
    ax1 = axes[0]
    if 'Disc_Score' in df_metrics.columns:
        colors = [EXTENDED_STYLE_MAP.get(m, {'color': 'gray'})['color'] for m in df_metrics.index]
        bars = ax1.bar(range(len(df_metrics)), df_metrics['Disc_Score'].values, color=colors)
        ax1.set_xticks(range(len(df_metrics)))
        ax1.set_xticklabels(df_metrics.index, rotation=45, ha='right')
        ax1.set_title('Discriminative Score (Lower = Better)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Score')
        ax1.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Ideal (0.0)')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Worst (0.5)')
        ax1.legend()
        
        for bar, val in zip(bars, df_metrics['Disc_Score'].values):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Predictive Score
    ax2 = axes[1]
    if 'Pred_Score' in df_metrics.columns:
        colors = [EXTENDED_STYLE_MAP.get(m, {'color': 'gray'})['color'] for m in df_metrics.index]
        bars = ax2.bar(range(len(df_metrics)), df_metrics['Pred_Score'].values, color=colors)
        ax2.set_xticks(range(len(df_metrics)))
        ax2.set_xticklabels(df_metrics.index, rotation=45, ha='right')
        ax2.set_title('Predictive Score (Lower = Better)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('MAE')
        
        for bar, val in zip(bars, df_metrics['Pred_Score'].values):
            if not np.isnan(val):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")
    plt.close()


def plot_method_ranking(metrics_store, save_path="outputs/method_ranking.png"):
    """
    Create a radar chart comparing all methods across metrics.
    
    Args:
        metrics_store: Dict of {method_name: {metric: value}}
        save_path: Path to save the figure
    """
    set_style()
    _ensure_dir(save_path)
    
    df_metrics = pd.DataFrame(metrics_store).T
    
    # Select metrics for radar chart
    radar_metrics = ['WD', 'ACF_MSE', 'Disc_Score', 'Pred_Score']
    available_metrics = [m for m in radar_metrics if m in df_metrics.columns]
    
    if len(available_metrics) < 3:
        print("[Warning] Not enough metrics for radar chart")
        return
    
    # Normalize metrics (lower is better, so invert)
    df_norm = df_metrics[available_metrics].copy()
    for col in df_norm.columns:
        max_val = df_norm[col].max()
        min_val = df_norm[col].min()
        if max_val > min_val:
            df_norm[col] = 1 - (df_norm[col] - min_val) / (max_val - min_val)
        else:
            df_norm[col] = 1.0
    
    # Create radar chart
    n_metrics = len(available_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for method in df_norm.index:
        values = df_norm.loc[method].values.tolist()
        values += values[:1]  # Close the polygon
        
        style = EXTENDED_STYLE_MAP.get(method, {'color': 'gray'})
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=style['color'])
        ax.fill(angles, values, alpha=0.1, color=style['color'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_metrics, fontsize=11)
    ax.set_title('Method Comparison (Higher = Better)', fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Plot Saved] {save_path}")
    plt.close()
