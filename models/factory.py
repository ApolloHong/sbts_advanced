"""
Model Factory Module

Provides a unified interface for creating time series generation models.
Uses Factory pattern for clean model instantiation.

Author: Manus AI
"""

from typing import Dict, Any, Type, Optional
import warnings

from models.base import TimeSeriesGenerator
from models.sbts_variants import (
    JDSBTS, JDSBTSF, JDSBTSNeural, JDSBTSFNeural,
    get_sbts_model
)
from models.lightsb import LightSB, PathLightSB, NumbaSB
from models.timegan_baseline import TimeGAN
from models.diffusion_ts_baseline import DiffusionTS
from models.rnn_baseline import RNNBaseline
from models.transformer_ar_baseline import TransformerARBaseline


# ============================================
# Model Registry
# ============================================

MODEL_REGISTRY: Dict[str, Type[TimeSeriesGenerator]] = {
    # JD-SBTS variants
    'jd_sbts': JDSBTS,
    'jd_sbts_f': JDSBTSF,
    'jd_sbts_neural': JDSBTSNeural,
    'jd_sbts_f_neural': JDSBTSFNeural,
    
    # Schrödinger Bridge variants
    'lightsb': LightSB,
    'lightsb_path': PathLightSB,
    'numba_sb': NumbaSB,
    
    # Baselines
    'timegan': TimeGAN,
    'diffusion_ts': DiffusionTS,
    'rnn': RNNBaseline,
    'transformer_ar': TransformerARBaseline,
}

# Aliases for convenience
MODEL_ALIASES = {
    'sbts': 'jd_sbts',
    'sbts_f': 'jd_sbts_f',
    'sbts_neural': 'jd_sbts_neural',
    'sbts_f_neural': 'jd_sbts_f_neural',
    'light_sb': 'lightsb',
    'path_lightsb': 'lightsb_path',
    'light_sb_path': 'lightsb_path',
    'step_lightsb': 'lightsb_path',
    'numbasb': 'numba_sb',
    'time_gan': 'timegan',
    'diffusion': 'diffusion_ts',
    'rnn_baseline': 'rnn',
    'transformer': 'transformer_ar',
    'ar_transformer': 'transformer_ar',
}


# ============================================
# Factory Functions
# ============================================

def get_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None
) -> TimeSeriesGenerator:
    """
    Create a time series generation model.
    
    Args:
        model_type: Type of model to create. Options:
            - 'jd_sbts': Base JD-SBTS model
            - 'jd_sbts_f': JD-SBTS with feedback mechanism
            - 'jd_sbts_neural': JD-SBTS with neural jump detection
            - 'jd_sbts_f_neural': JD-SBTS with feedback and neural jumps
            - 'lightsb': Light Schrödinger Bridge
            - 'lightsb_path': Path-level Light Schrödinger Bridge
            - 'numba_sb': Numba-accelerated Markovian SB
            - 'timegan': TimeGAN baseline
            - 'diffusion_ts': Diffusion-TS baseline
            - 'rnn': Autoregressive RNN baseline
            - 'transformer_ar': Causal autoregressive Transformer baseline
        config: Configuration dictionary (optional)
    
    Returns:
        TimeSeriesGenerator instance
    
    Example:
        >>> model = get_model('jd_sbts_f', {'feedback_kappa': 5.0})
        >>> model.fit(data, time_grid)
        >>> generated = model.generate(100)
    """
    if config is None:
        config = {}
    
    # Resolve aliases
    model_type = model_type.lower()
    if model_type in MODEL_ALIASES:
        model_type = MODEL_ALIASES[model_type]
    
    # Check if model type is registered
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys()) + list(MODEL_ALIASES.keys())
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available models: {available}"
        )
    
    # Add model_type to config
    config['model_type'] = model_type
    
    # Create model
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config)


def list_models() -> Dict[str, str]:
    """
    List all available models with descriptions.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    descriptions = {
        'jd_sbts': 'Jump-Diffusion Schrödinger Bridge (base model)',
        'jd_sbts_f': 'JD-SBTS with Feedback mechanism (volatility clustering)',
        'jd_sbts_neural': 'JD-SBTS with Neural jump detection',
        'jd_sbts_f_neural': 'JD-SBTS with Feedback + Neural jumps',
        'lightsb': 'Window-level Light Schrödinger Bridge (sum-exp quadratic potentials)',
        'lightsb_path': 'Path-level Light Schrödinger Bridge (one-step potential bridge)',
        'numba_sb': 'Numba-accelerated Markovian SB (fast baseline)',
        'timegan': 'TimeGAN (Yoon et al., NeurIPS 2019)',
        'diffusion_ts': 'Diffusion model for time series',
        'rnn': 'Autoregressive RNN baseline (GRU/LSTM)',
        'transformer_ar': 'Causal autoregressive Transformer baseline',
    }
    return descriptions


def get_default_config(model_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a model type.
    
    Args:
        model_type: Type of model
    
    Returns:
        Default configuration dictionary
    """
    # Resolve aliases
    model_type = model_type.lower()
    if model_type in MODEL_ALIASES:
        model_type = MODEL_ALIASES[model_type]
    
    # Base config
    base_config = {
        'model_type': model_type,
    }
    
    # Model-specific defaults
    if model_type in ['jd_sbts', 'jd_sbts_f', 'jd_sbts_neural', 'jd_sbts_f_neural']:
        base_config.update({
            # Jump detection
            'jump_threshold_std': 3.0,
            'jump_rolling_window': 20,
            'jump_size_scale': 1.0,
            'jump_size_clip': None,
            'input_type': 'path',
            'generate_returns_from_price': False,
            'return_generation_use_reference_vol': True,
            'return_drift_scale': 0.35,
            'return_vol_scale': 1.2,
            'return_feedback_gamma_scale': 1.0,
            'use_sbjts_calibration': True,
            'sbjts_fix_c': True,
            'sbjts_threshold_r': 3.0,
            'sbjts_qv_n_synth': 200,
            'sbjts_qv_n_grid': 10,
            'sbjts_variance_tolerance': 0.25,
            'use_neural_jumps': 'neural' in model_type,
            'neural_jump_hidden_dim': 64,
            'neural_jump_epochs': 30,
            'neural_jump_lr': 0.001,
            'neural_jump_seq_len': 10,
            'neural_jump_calibrate_intensity': True,
            'neural_jump_intensity_cap_multiplier': 2.0,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            
            # Volatility calibration
            'vol_bandwidth': 0.5,
            'vol_n_t_grid': 50,
            'vol_n_x_grid': 100,
            
            # Drift estimation
            'drift_estimator': 'lstm',
            'lstm_hidden': 128,
            'lstm_epochs': 50,
            'lstm_lr': 0.005,
            'lstm_dropout': 0.3,
            'lstm_use_huber': True,
            
            # Feedback (for _f variants)
            'use_feedback': '_f' in model_type,
            'feedback_kappa': 0.8,
            'feedback_gamma': 1.0,
            
            # Solver
            'solver_backend': 'numba',
        })
    
    elif model_type == 'lightsb':
        base_config.update({
            'lightsb_n_potentials': 20,
            'lightsb_epsilon': 1.0,
            'lightsb_s_diagonal_init': 0.1,
            'lightsb_sampling_batch_size': 512,
            'lightsb_init_centers_from_data': True,
            'lightsb_epochs': 100,
            'lightsb_lr': 0.001,
            'lightsb_batch_size': 256,
            'lightsb_weight_decay': 0.0,
            'lightsb_grad_clip': 1.0,
            'lightsb_source_std': 1.0,
        })

    elif model_type == 'lightsb_path':
        base_config.update({
            'lightsb_path_n_potentials': 20,
            'lightsb_path_epsilon': 1.0,
            'lightsb_path_s_diagonal_init': 0.1,
            'lightsb_path_sampling_batch_size': 512,
            'lightsb_path_init_centers_from_data': True,
            'lightsb_path_epochs': 100,
            'lightsb_path_lr': 0.001,
            'lightsb_path_batch_size': 256,
            'lightsb_path_weight_decay': 0.0,
            'lightsb_path_grad_clip': 1.0,
            'lightsb_path_state_clip': 8.0,
        })
    
    elif model_type == 'numba_sb':
        base_config.update({
            'numba_sb_sigma': 0.1,
        })
    
    elif model_type == 'timegan':
        base_config.update({
            'timegan_hidden_dim': 64,
            'timegan_z_dim': 32,
            'timegan_n_layers': 2,
            'timegan_epochs': 50,
            'timegan_lr': 0.001,
            'timegan_batch_size': 128,
            'timegan_normalization': 'standard',
            'timegan_gamma': 1.0,
            'timegan_moment_loss_weight': 100.0,
            'timegan_supervised_loss_weight': 100.0,
            'timegan_reconstruction_loss_weight': 10.0,
            'timegan_embedding_supervised_weight': 0.1,
            'timegan_discriminator_threshold': 0.15,
            'timegan_generator_steps': 2,
            'timegan_clip_output': False,
        })
    
    elif model_type == 'diffusion_ts':
        base_config.update({
            'diffusion_hidden_dim': 128,
            'diffusion_n_steps': 100,
            'diffusion_epochs': 100,
            'diffusion_lr': 0.001,
            'diffusion_batch_size': 64,
            'diffusion_beta_start': 0.0001,
            'diffusion_beta_end': 0.02,
        })

    elif model_type == 'rnn':
        base_config.update({
            'rnn_hidden_dim': 64,
            'rnn_num_layers': 2,
            'rnn_dropout': 0.1,
            'rnn_bidirectional': False,
            'rnn_type': 'lstm',
            'rnn_epochs': 50,
            'rnn_lr': 0.001,
            'rnn_batch_size': 128,
            'rnn_context_len': None,
        })

    elif model_type == 'transformer_ar':
        base_config.update({
            'transformer_ar_d_model': 64,
            'transformer_ar_n_heads': 4,
            'transformer_ar_n_layers': 2,
            'transformer_ar_d_ff': 128,
            'transformer_ar_dropout': 0.1,
            'transformer_ar_epochs': 50,
            'transformer_ar_lr': 0.001,
            'transformer_ar_batch_size': 128,
            'transformer_ar_max_seq_len': None,
            'transformer_ar_context_len': None,
        })
    
    return base_config


def create_model_comparison(
    model_types: list,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, TimeSeriesGenerator]:
    """
    Create multiple models for comparison.
    
    Args:
        model_types: List of model types to create
        config: Shared configuration (optional)
    
    Returns:
        Dictionary mapping model names to model instances
    
    Example:
        >>> models = create_model_comparison(
        ...     ['jd_sbts', 'jd_sbts_f', 'timegan'],
        ...     {'lstm_epochs': 30}
        ... )
        >>> for name, model in models.items():
        ...     model.fit(data, time_grid)
    """
    if config is None:
        config = {}
    
    models = {}
    for model_type in model_types:
        try:
            model_config = get_default_config(model_type)
            model_config.update(config)
            models[model_type] = get_model(model_type, model_config)
        except Exception as e:
            warnings.warn(f"Failed to create model '{model_type}': {e}")
    
    return models


# ============================================
# Model Information
# ============================================

def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get detailed information about a model type.
    
    Args:
        model_type: Type of model
    
    Returns:
        Dictionary with model information
    """
    # Resolve aliases
    model_type = model_type.lower()
    if model_type in MODEL_ALIASES:
        model_type = MODEL_ALIASES[model_type]
    
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = MODEL_REGISTRY[model_type]
    
    info = {
        'name': model_type,
        'class': model_class.__name__,
        'module': model_class.__module__,
        'docstring': model_class.__doc__,
        'default_config': get_default_config(model_type),
    }
    
    # Add model-specific info
    if hasattr(model_class, 'MODEL_TYPE'):
        info['model_type'] = model_class.MODEL_TYPE
    
    return info


# ============================================
# Convenience Functions
# ============================================

def create_jdsbts(config: Optional[Dict[str, Any]] = None) -> JDSBTS:
    """Create JD-SBTS model."""
    return get_model('jd_sbts', config)


def create_jdsbts_f(config: Optional[Dict[str, Any]] = None) -> JDSBTSF:
    """Create JD-SBTS-F model with feedback."""
    return get_model('jd_sbts_f', config)


def create_lightsb(config: Optional[Dict[str, Any]] = None) -> LightSB:
    """Create the window-level LightSB model."""
    return get_model('lightsb', config)


def create_lightsb_path(config: Optional[Dict[str, Any]] = None) -> PathLightSB:
    """Create the path-level LightSB model."""
    return get_model('lightsb_path', config)


def create_timegan(config: Optional[Dict[str, Any]] = None) -> TimeGAN:
    """Create TimeGAN model."""
    return get_model('timegan', config)


def create_diffusion_ts(config: Optional[Dict[str, Any]] = None) -> DiffusionTS:
    """Create Diffusion-TS model."""
    return get_model('diffusion_ts', config)


def create_rnn(config: Optional[Dict[str, Any]] = None) -> RNNBaseline:
    """Create the autoregressive RNN baseline."""
    return get_model('rnn', config)


def create_transformer_ar(
    config: Optional[Dict[str, Any]] = None
) -> TransformerARBaseline:
    """Create the causal autoregressive Transformer baseline."""
    return get_model('transformer_ar', config)
