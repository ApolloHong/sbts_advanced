"""
Models Package

Contains all time series generation models.

Models:
    - JD-SBTS: Jump-Diffusion Schrödinger Bridge (our method)
    - JD-SBTS-F: JD-SBTS with Feedback mechanism (our innovation)
    - LightSB: Window-level Light Schrödinger Bridge
    - PathLightSB: Path-level Light Schrödinger Bridge
    - Numba-SB: Numba-accelerated Markovian SB
    - TimeGAN: Time-series Generative Adversarial Network
    - Diffusion-TS: Diffusion-based time series generation
    - RNN: Autoregressive recurrent baseline
    - Transformer-AR: Causal autoregressive Transformer baseline

Author: Manus AI
"""

from models.base import BaseTimeSeriesGenerator
from models.sbts_variants import JDSBTS, JDSBTSF
from models.lightsb import LightSB, PathLightSB
from models.timegan_baseline import TimeGAN
from models.diffusion_ts_baseline import DiffusionTS
from models.rnn_baseline import RNNBaseline
from models.transformer_ar_baseline import TransformerARBaseline
from models.factory import get_model, list_models, get_default_config

__all__ = [
    # Base
    'BaseTimeSeriesGenerator',
    
    # Our methods
    'JDSBTS',
    'JDSBTSF',
    
    # Baselines
    'LightSB',
    'PathLightSB',

    'TimeGAN',
    'DiffusionTS',
    'RNNBaseline',
    'TransformerARBaseline',
    
    # Factory
    'get_model',
    'list_models',
    'get_default_config',
]
