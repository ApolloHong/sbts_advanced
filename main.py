#!/usr/bin/env python3
"""
JD-SBTS: Jump-Diffusion Schrödinger Bridge for Time Series Generation

Main entry point for running experiments and benchmarks.

FEATURES:
- JD-SBTS-F: Feedback mechanism with Jump-Volatility Interaction
- Neural MJD: Time-varying jump intensity prediction
- Numba acceleration for CPU-intensive computations
- OOP architecture with Factory/Strategy patterns
- Comprehensive evaluation metrics
- Publication-quality visualizations

Usage:
    python main.py                          # Run with default config
    python main.py --config config.json     # Run with custom config
    python main.py --model jd_sbts_f        # Run specific model
    python main.py --model rnn              # Run RNN baseline
    python main.py --model transformer_ar   # Run causal Transformer baseline
    python main.py --benchmark              # Run full benchmark

Author: Manus AI
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================
# Default Configuration
# ============================================

DEFAULT_CONFIG = {
    # Experiment settings
    'experiment_name': 'jdsbts_experiment',
    'seed': 42,
    'verbose': True,
    
    # Data settings
    'data_source': 'etf',  # 'etf' or 'synthetic'
    'tickers': ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD'],
    'start_date': '2020-01-01',
    'end_date': '2024-01-01',
    'window_size': 60,
    'stride': 10,
    'fallback_to_synthetic_on_data_error': False,
    
    # Synthetic data settings
    'synthetic_type': 'gbm_jump',
    'n_synthetic_samples': 500,
    'synthetic_total_steps': 240,
    'synthetic_return_type': 'log_returns',
    
    # Model settings
    'models_to_run': ['jd_sbts', 'jd_sbts_f'],
    
    # JD-SBTS settings
    'use_neural_jumps': False,
    'use_feedback': True,
    'jump_threshold_std': 4.0,
    'feedback_kappa': 5.0,
    'feedback_gamma': 0.5,
    
    # Drift estimation
    'drift_estimator': 'lstm',
    'lstm_hidden': 128,
    'lstm_epochs': 50,
    'lstm_lr': 0.005,
    'lstm_dropout': 0.3,
    
    # Generation settings
    'n_generate': 100,
    'n_steps': None,  # None = same as training
    
    # Evaluation settings
    'run_discriminative': True,
    'run_predictive': True,
    'discriminative_iterations': 500,
    'predictive_iterations': 500,
    'acf_max_lag': 15,
    
    # Output settings
    'output_dir': 'experiments',
    'save_models': True,
    'save_plots': True,
}

CLI_MODEL_CHOICES = [
    'jd_sbts',
    'jd_sbts_f',
    'jd_sbts_neural',
    'jd_sbts_f_neural',
    'lightsb',
    'lightsb_path',
    'numba_sb',
    'timegan',
    'diffusion_ts',
    'rnn',
    'transformer_ar',
    'sbts',
    'sbts_f',
    'sbts_neural',
    'sbts_f_neural',
    'light_sb',
    'path_lightsb',
    'light_sb_path',
    'step_lightsb',
    'numbasb',
    'time_gan',
    'diffusion',
    'rnn_baseline',
    'transformer',
    'ar_transformer',
]


# ============================================
# Import Handlers (with fallbacks)
# ============================================

def import_new_modules():
    """Import new modular architecture."""
    try:
        from models.factory import get_model, list_models, get_default_config, create_model_comparison
        from data.loaders import load_etf_data, load_synthetic_data, create_sliding_windows
        from utils.experiment_manager import ExperimentManager
        from metrics import compute_all_metrics, discriminative_score_metrics, predictive_score_metrics
        from metrics.numba_metrics import compute_all_metrics_numba, compute_stylized_facts_numba
        
        return {
            'get_model': get_model,
            'list_models': list_models,
            'get_default_config': get_default_config,
            'create_model_comparison': create_model_comparison,
            'load_etf_data': load_etf_data,
            'load_synthetic_data': load_synthetic_data,
            'create_sliding_windows': create_sliding_windows,
            'ExperimentManager': ExperimentManager,
            'compute_all_metrics': compute_all_metrics,
            'discriminative_score_metrics': discriminative_score_metrics,
            'predictive_score_metrics': predictive_score_metrics,
            'compute_all_metrics_numba': compute_all_metrics_numba,
            'compute_stylized_facts_numba': compute_stylized_facts_numba,
        }
    except ImportError as e:
        warnings.warn(f"New modules not available: {e}")
        return None


def import_visualization():
    """Import visualization modules."""
    try:
        os.environ.setdefault('MPLCONFIGDIR', os.path.join('/tmp', 'matplotlib'))
        os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
        from visualization import (
            set_style, plot_performance_metrics, plot_comprehensive_comparison,
            plot_correlation_distribution, plot_volatility_surface,
            plot_stress_factor_dynamics, plot_feedback_comparison,
            plot_model_comparison_grid, create_legend_figure
        )
        return {
            'set_style': set_style,
            'plot_performance_metrics': plot_performance_metrics,
            'plot_comprehensive_comparison': plot_comprehensive_comparison,
            'plot_correlation_distribution': plot_correlation_distribution,
            'plot_volatility_surface': plot_volatility_surface,
            'plot_stress_factor_dynamics': plot_stress_factor_dynamics,
            'plot_feedback_comparison': plot_feedback_comparison,
            'plot_model_comparison_grid': plot_model_comparison_grid,
            'create_legend_figure': create_legend_figure,
        }
    except ImportError as e:
        warnings.warn(f"Visualization not fully available: {e}")
        return None


def _extract_metric_value(result: Any, key: str, default: float) -> float:
    """Normalize metric outputs that may be dicts or scalar values."""
    if isinstance(result, dict):
        return float(result.get(key, default))
    if np.isscalar(result):
        return float(result)
    return float(default)


def _match_eval_shapes(
    real_data: np.ndarray,
    gen_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim real and generated batches to a shared evaluation size."""
    n_eval = min(len(real_data), len(gen_data))
    if n_eval <= 0:
        raise ValueError("Need at least one sample for evaluation")
    return real_data[:n_eval], gen_data[:n_eval]


LEGACY_MODEL_NAME_MAP = {
    'kernel': 'jd_sbts',
    'sbts-lstm': 'jd_sbts',
    'lstm': 'jd_sbts',
    'lightsb': 'lightsb',
    'light_sb': 'lightsb',
    'light-sb': 'lightsb',
    'lightsb_path': 'lightsb_path',
    'path_lightsb': 'lightsb_path',
    'light_sb_path': 'lightsb_path',
    'step_lightsb': 'lightsb_path',
    'timegan': 'timegan',
    'diffusion-ts': 'diffusion_ts',
    'diffusion_ts': 'diffusion_ts',
    'rnn': 'rnn',
    'transformer': 'transformer_ar',
    'transformer_ar': 'transformer_ar',
}

CONDITIONED_X0_MODELS = {
    'jd_sbts',
    'jd_sbts_f',
    'jd_sbts_neural',
    'jd_sbts_f_neural',
    'lightsb_path',
}
CONDITIONED_PREFIX_MODELS = {'rnn', 'transformer_ar'}


def _normalize_model_name(model_name: str) -> str:
    """Normalize legacy display names into active factory keys."""
    key = str(model_name).strip().lower().replace(' ', '_')
    return LEGACY_MODEL_NAME_MAP.get(key, key)


def _merge_legacy_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the old nested uppercase config schema to the active flat schema."""
    if not any(k in user_config for k in ('data', 'models', 'physics', 'lstm', 'lightsb')):
        return user_config

    flat = {
        k: v for k, v in user_config.items()
        if k not in {'project', 'data', 'models', 'physics', 'neural_jumps', 'kernel', 'lstm', 'lightsb', 'generation', '_comments'}
    }

    data_cfg = user_config.get('data', {})
    if data_cfg:
        if 'USE_REAL_DATA' in data_cfg:
            flat['data_source'] = 'etf' if data_cfg['USE_REAL_DATA'] else 'synthetic'
        if 'START_DATE' in data_cfg:
            flat['start_date'] = data_cfg['START_DATE']
        if 'END_DATE' in data_cfg:
            flat['end_date'] = data_cfg['END_DATE']
        if 'SEQ_LEN' in data_cfg:
            flat['window_size'] = data_cfg['SEQ_LEN']

    models_cfg = user_config.get('models', {})
    if 'MODELS_TO_RUN' in models_cfg:
        flat['models_to_run'] = [
            _normalize_model_name(name) for name in models_cfg['MODELS_TO_RUN']
        ]

    physics_cfg = user_config.get('physics', {})
    if physics_cfg:
        if 'USE_NEURAL_JUMPS' in physics_cfg:
            flat['use_neural_jumps'] = physics_cfg['USE_NEURAL_JUMPS']
        if 'JUMP_THRESHOLD_STD' in physics_cfg:
            flat['jump_threshold_std'] = physics_cfg['JUMP_THRESHOLD_STD']
        if 'VOL_BANDWIDTH' in physics_cfg:
            flat['vol_bandwidth'] = physics_cfg['VOL_BANDWIDTH']

    neural_jump_cfg = user_config.get('neural_jumps', {})
    if neural_jump_cfg:
        mapping = {
            'NEURAL_JUMP_HIDDEN_DIM': 'neural_jump_hidden_dim',
            'NEURAL_JUMP_LR': 'neural_jump_lr',
            'NEURAL_JUMP_EPOCHS': 'neural_jump_epochs',
            'NEURAL_JUMP_SEQ_LEN': 'neural_jump_seq_len',
            'NEURAL_JUMP_DROPOUT': 'neural_jump_dropout',
        }
        flat.update({dst: neural_jump_cfg[src] for src, dst in mapping.items() if src in neural_jump_cfg})

    lstm_cfg = user_config.get('lstm', {})
    if lstm_cfg:
        mapping = {
            'LSTM_HIDDEN': 'lstm_hidden',
            'LSTM_LR': 'lstm_lr',
            'LSTM_EPOCHS': 'lstm_epochs',
            'LSTM_WEIGHT_DECAY': 'lstm_weight_decay',
            'LSTM_DROPOUT': 'lstm_dropout',
            'LSTM_DRIFT_DAMPENING': 'lstm_drift_dampening',
            'LSTM_USE_HUBER_LOSS': 'lstm_use_huber',
            'LSTM_HUBER_DELTA': 'lstm_huber_delta',
        }
        flat.update({dst: lstm_cfg[src] for src, dst in mapping.items() if src in lstm_cfg})

    generation_cfg = user_config.get('generation', {})
    if generation_cfg:
        if 'N_GEN_PATHS' in generation_cfg:
            flat['n_generate'] = generation_cfg['N_GEN_PATHS']
        if 'SYNTHETIC_SAMPLES' in generation_cfg:
            flat['n_synthetic_samples'] = generation_cfg['SYNTHETIC_SAMPLES']

    return flat


def _build_generation_kwargs(
    model_name: str,
    model: Any,
    reference_windows: np.ndarray,
    n_samples: int,
    n_steps: int,
    fair_generation: bool = True,
) -> Dict[str, Any]:
    """Build model-specific generation kwargs for fair comparisons."""
    kwargs: Dict[str, Any] = {'n_samples': n_samples, 'n_steps': n_steps}
    if not fair_generation or len(reference_windows) == 0:
        return kwargs

    reference = reference_windows[:n_samples]
    if model_name in CONDITIONED_X0_MODELS:
        kwargs['x0'] = reference[:, 0, :].astype(np.float32)
    elif model_name in CONDITIONED_PREFIX_MODELS:
        context_len = int(getattr(model, 'context_len', max(1, n_steps // 2)))
        context_len = max(1, min(context_len, reference.shape[1] - 1))
        kwargs['x0'] = reference[:, :context_len, :].astype(np.float32)
    return kwargs


# ============================================
# Main Experiment Runner (New Architecture)
# ============================================

def run_experiment_new(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run experiment using new modular architecture.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Dictionary of results
    """
    modules = import_new_modules()
    viz = import_visualization()
    
    if modules is None:
        raise ImportError("Model modules not available in the unified architecture.")
    
    # Set random seed
    np.random.seed(config.get('seed', 42))
    
    verbose = config.get('verbose', True)
    
    # Initialize experiment manager
    models_to_run = config.get('models_to_run', ['jd_sbts'])
    if len(models_to_run) == 1:
        run_label = models_to_run[0]
    else:
        run_label = config.get('experiment_name', 'multi_model')

    exp_manager = modules['ExperimentManager'](
        model_name=run_label,
        base_dir=config.get('output_dir', 'experiments'),
        config=config,
        verbose=verbose
    )
    
    if verbose:
        print("=" * 70)
        print("JD-SBTS Experiment (New Architecture)")
        print("=" * 70)
        print(f"Experiment: {exp_manager.run_name}")
        print(f"Output dir: {exp_manager.run_dir}")
        print()
    
    # ========================================
    # Load Data
    # ========================================
    if verbose:
        print("[Phase 1] Loading Data...")
    
    data_source = config.get('data_source', 'etf')
    
    if data_source == 'etf':
        try:
            raw_data, time_grid, metadata = modules['load_etf_data'](
                tickers=config.get('tickers', ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD']),
                start_date=config.get('start_date', '2020-01-01'),
                end_date=config.get('end_date', '2024-01-01'),
                return_type='log_returns'
            )
        except Exception as e:
            if config.get('fallback_to_synthetic_on_data_error', False):
                warnings.warn(f"Failed to load ETF data: {e}. Using synthetic data.")
                data_source = 'synthetic'
            else:
                raise RuntimeError(
                    "ETF data loading failed. "
                    "Set 'fallback_to_synthetic_on_data_error': True if you want "
                    "the experiment to continue on synthetic data."
                ) from e
    
    if data_source == 'synthetic':
        synthetic_total_steps = config.get(
            'synthetic_total_steps',
            max(config.get('window_size', 60) * 4, config.get('window_size', 60) + 1)
        )
        raw_data, time_grid, metadata = modules['load_synthetic_data'](
            n_samples=config.get('n_synthetic_samples', 500),
            n_steps=synthetic_total_steps,
            n_features=len(config.get('tickers', ['SPY', 'QQQ', 'IWM', 'EEM', 'GLD'])),
            data_type=config.get('synthetic_type', 'gbm_jump'),
            seed=config.get('seed', 42),
            return_type=config.get('synthetic_return_type', 'log_returns'),
        )
    
    # Create sliding windows
    window_size = config.get('window_size', 60)
    stride = config.get('stride', 10)
    
    if raw_data.shape[1] > window_size:
        data = modules['create_sliding_windows'](raw_data, window_size, stride)
    else:
        data = raw_data
    
    # Update time grid
    time_grid = np.linspace(0, 1, data.shape[1])
    
    if verbose:
        print(f"  Data shape: {data.shape}")
        print(f"  Time grid: {len(time_grid)} steps")
    
    exp_manager.log(f"Data loaded: shape={data.shape}")
    
    # ========================================
    # Train Models
    # ========================================
    if verbose:
        print("\n[Phase 2] Training Models...")
    
    trained_models = {}
    training_times = {}
    failures = {}
    
    for model_name in models_to_run:
        if verbose:
            print(f"\n  Training {model_name}...")
        
        try:
            # Get model config
            model_config = modules['get_default_config'](model_name)
            model_config.update(config)  # Override with experiment config
            if _normalize_model_name(model_name) == 'transformer_ar':
                required_len = max(1, data.shape[1] - 1)
                current_max_len = model_config.get('transformer_ar_max_seq_len')
                if current_max_len is None or current_max_len < required_len:
                    model_config['transformer_ar_max_seq_len'] = required_len
            
            # Create model
            model = modules['get_model'](model_name, model_config)
            
            # Train
            start_time = time.perf_counter()
            model.fit(data, time_grid, verbose=verbose)
            train_time = time.perf_counter() - start_time
            
            trained_models[model_name] = model
            training_times[model_name] = train_time
            
            exp_manager.log(f"{model_name} trained in {train_time:.2f}s")
            
            if verbose:
                print(f"  {model_name} training time: {train_time:.2f}s")
        
        except Exception as e:
            failures[model_name] = {'phase': 'training', 'error': str(e)}
            warnings.warn(f"Failed to train {model_name}: {e}")
            exp_manager.log(f"ERROR: {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

    if not trained_models:
        exp_manager.save_artifact("failures.json", failures)
        raise RuntimeError(f"No models trained successfully: {failures}")
    
    # ========================================
    # Generate Samples
    # ========================================
    if verbose:
        print("\n[Phase 3] Generating Samples...")
    
    n_generate = int(config.get('n_generate', 100))
    if config.get('fair_generation', True):
        n_generate = min(n_generate, len(data))
    n_steps = config.get('n_steps') or data.shape[1]
    
    generated_data = {}
    generation_times = {}
    stress_trajectories = {}
    
    for model_name, model in trained_models.items():
        if verbose:
            print(f"\n  Generating from {model_name}...")
        
        try:
            start_time = time.perf_counter()
            reference_windows = data[:min(n_generate, len(data))]
            gen_kwargs = _build_generation_kwargs(
                model_name=_normalize_model_name(model_name),
                model=model,
                reference_windows=reference_windows,
                n_samples=n_generate,
                n_steps=n_steps,
                fair_generation=config.get('fair_generation', True),
            )
            
            # Check if model supports stress factor output
            if hasattr(model, 'use_feedback') and model.use_feedback:
                gen_paths, stress = model.generate(return_stress=True, **gen_kwargs)
                stress_trajectories[model_name] = stress
            else:
                gen_paths = model.generate(**gen_kwargs)
            
            gen_time = time.perf_counter() - start_time
            
            generated_data[model_name] = gen_paths
            generation_times[model_name] = gen_time
            
            exp_manager.log(f"{model_name} generated {n_generate} samples in {gen_time:.2f}s")
            
            if verbose:
                print(f"  {model_name} generation time: {gen_time:.2f}s")
                print(f"  Generated shape: {gen_paths.shape}")
        
        except Exception as e:
            failures[model_name] = {'phase': 'generation', 'error': str(e)}
            warnings.warn(f"Failed to generate from {model_name}: {e}")
            exp_manager.log(f"ERROR: {model_name} generation failed: {e}")
            import traceback
            traceback.print_exc()

    if not generated_data:
        exp_manager.save_artifact("failures.json", failures)
        raise RuntimeError(f"No models generated successfully: {failures}")
    
    # ========================================
    # Evaluate Models
    # ========================================
    if verbose:
        print("\n[Phase 4] Evaluating Models...")
    
    metrics_results = {}
    
    for model_name, gen_data in generated_data.items():
        if verbose:
            print(f"\n  Evaluating {model_name}...")
        
        try:
            real_eval, gen_eval = _match_eval_shapes(data, gen_data)

            # Statistical metrics (Numba-accelerated)
            stats = modules['compute_all_metrics_numba'](
                real_eval,
                gen_eval,
                max_acf_lag=config.get('acf_max_lag', 15)
            )
            
            # Stylized facts
            real_returns = np.diff(real_eval, axis=1)
            gen_returns = np.diff(gen_eval, axis=1)
            real_stylized = modules['compute_stylized_facts_numba'](real_returns)
            stylized = modules['compute_stylized_facts_numba'](gen_returns)
            
            metrics = {
                'train_time': training_times.get(model_name, 0),
                'gen_time': generation_times.get(model_name, 0),
                **stats,
                **{f'stylized_real_{k}': v for k, v in real_stylized.items()},
                **{f'stylized_gen_{k}': v for k, v in stylized.items()},
                **{
                    f'stylized_abs_error_{k}': abs(stylized[k] - real_stylized[k])
                    for k in stylized
                }
            }
            
            # Discriminative score (optional)
            if config.get('run_discriminative', True):
                try:
                    disc_result = modules['discriminative_score_metrics'](
                        real_eval,
                        gen_eval,
                        iterations=config.get('discriminative_iterations', 500),
                    )
                    metrics['discriminative_score'] = _extract_metric_value(
                        disc_result,
                        'discriminative_score',
                        0.5
                    )
                except Exception as e:
                    warnings.warn(f"Discriminative score failed: {e}")
                    metrics['discriminative_score'] = 0.5
            
            # Predictive score (optional)
            if config.get('run_predictive', True):
                try:
                    pred_result = modules['predictive_score_metrics'](
                        real_eval,
                        gen_eval,
                        iterations=config.get('predictive_iterations', 500),
                    )
                    metrics['predictive_score'] = _extract_metric_value(
                        pred_result,
                        'predictive_score',
                        0.0
                    )
                except Exception as e:
                    warnings.warn(f"Predictive score failed: {e}")
                    metrics['predictive_score'] = 0
            
            metrics_results[model_name] = metrics
            
            if verbose:
                print(f"    WD: {metrics.get('wasserstein_distance', 0):.6f}")
                print(f"    ACF MSE: {metrics.get('acf_mse', 0):.6f}")
                if 'discriminative_score' in metrics:
                    print(f"    Disc Score: {metrics['discriminative_score']:.4f}")
        
        except Exception as e:
            failures[model_name] = {'phase': 'evaluation', 'error': str(e)}
            warnings.warn(f"Evaluation failed for {model_name}: {e}")
            exp_manager.log(f"ERROR: {model_name} evaluation failed: {e}")
    
    # ========================================
    # Save Results
    # ========================================
    if verbose:
        print("\n[Phase 5] Saving Results...")
    
    # Save metrics
    exp_manager.save_metrics(metrics_results)
    if failures:
        exp_manager.save_artifact("failures.json", failures)
    
    # Save models
    if config.get('save_models', True):
        for model_name, model in trained_models.items():
            exp_manager.save_model(model, model_name)
    
    # ========================================
    # Generate Plots
    # ========================================
    if config.get('save_plots', True) and viz is not None:
        if verbose:
            print("\n[Phase 6] Generating Plots...")
        
        try:
            # Create metrics DataFrame
            import pandas as pd
            df_metrics = pd.DataFrame(metrics_results).T
            df_metrics.index.name = 'Model'
            
            # Model comparison grid
            viz['plot_model_comparison_grid'](
                data[:min(n_generate, len(data))],
                generated_data,
                metrics_results,
                save_path=os.path.join(str(exp_manager.run_dir), 'model_comparison.png')
            )
            
            # Feedback analysis (if applicable)
            for model_name, stress in stress_trajectories.items():
                viz['plot_stress_factor_dynamics'](
                    time_grid,
                    stress,
                    save_path=os.path.join(str(exp_manager.run_dir), f'{model_name}_stress.png')
                )
            
            # Legend
            viz['create_legend_figure'](
                save_path=os.path.join(str(exp_manager.run_dir), 'legend.png')
            )
            
        except Exception as e:
            warnings.warn(f"Plot generation failed: {e}")
            exp_manager.log(f"ERROR: Plot generation failed: {e}")
    
    # ========================================
    # Summary
    # ========================================
    if verbose:
        print("\n" + "=" * 70)
        print("Experiment Complete!")
        print("=" * 70)
        print(f"\nResults saved to: {str(exp_manager.run_dir)}")
        print("\nMetrics Summary:")
        for model_name, metrics in metrics_results.items():
            print(f"\n  {model_name}:")
            print(f"    Training Time: {metrics.get('train_time', 0):.2f}s")
            print(f"    Generation Time: {metrics.get('gen_time', 0):.2f}s")
            print(f"    Wasserstein Distance: {metrics.get('wasserstein_distance', 0):.6f}")
            print(f"    ACF MSE: {metrics.get('acf_mse', 0):.6f}")
    
    return {
        'config': config,
        'metrics': metrics_results,
        'generated_data': generated_data,
        'stress_trajectories': stress_trajectories,
        'experiment_dir': str(exp_manager.run_dir),
        'failures': failures,
    }


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full benchmark comparing all available models.
    
    Args:
        config: Base configuration
    
    Returns:
        Benchmark results
    """
    modules = import_new_modules()
    if modules is None:
        raise ImportError("Model modules not available in the unified architecture.")

    # Keep benchmark coverage aligned with the active model registry.
    all_models = list(modules['list_models']().keys())

    config['models_to_run'] = all_models
    config['experiment_name'] = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return run_experiment_new(config)


# ============================================
# CLI Interface
# ============================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='JD-SBTS: Jump-Diffusion Schrödinger Bridge for Time Series'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to JSON configuration file'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        choices=CLI_MODEL_CHOICES,
        help='Specific model to run (e.g., jd_sbts_f, rnn, transformer_ar)'
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run full benchmark with all models'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory'
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help='Random seed'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data instead of ETF data'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # List models and exit
    if args.list_models:
        modules = import_new_modules()
        if modules:
            print("\nAvailable Models:")
            print("-" * 50)
            for name, desc in modules['list_models']().items():
                print(f"  {name}: {desc}")
            print()
        else:
            print("Model modules not available in the unified architecture.")
        return
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    if args.config:
        with open(args.config, 'r') as f:
            user_config = json.load(f)
        user_config = _merge_legacy_config(user_config)
        config.update(user_config)
    
    # Override with CLI arguments
    if args.model:
        config['models_to_run'] = [_normalize_model_name(args.model)]
    
    if args.output:
        config['output_dir'] = args.output
    
    if args.seed is not None:
        config['seed'] = args.seed
    
    if args.quiet:
        config['verbose'] = False
    
    if args.synthetic:
        config['data_source'] = 'synthetic'

    config['models_to_run'] = [
        _normalize_model_name(name)
        for name in config.get('models_to_run', ['jd_sbts'])
    ]
    
    # Run experiment or benchmark
    try:
        if args.benchmark:
            results = run_benchmark(config)
        else:
            results = run_experiment_new(config)
        
        print(f"\nExperiment completed. Results saved to: {results['experiment_dir']}")
    
    except ImportError as e:
        print(f"\nNew architecture not available: {e}")
        print("Install the missing dependencies and rerun the unified main pipeline.")


if __name__ == '__main__':
    main()
