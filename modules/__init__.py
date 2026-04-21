"""
Modules Package

Contains core computational modules for JD-SBTS.

Modules:
    - volatility: Local volatility surface calibration
    - jumps: Jump detection and filtering
    - solver: SDE solvers with feedback mechanism
    - feedback: Stress factor dynamics
    - drift_neural: Neural network drift estimators
    - drift_kernel: Kernel regression drift estimators

Author: Manus AI
"""

from modules.volatility import LocalVolatilityCalibrator
from modules.jumps import StaticJumpDetector, NeuralJumpDetector
from modules.solver import JumpDiffusionEulerSolver
from modules.feedback import (
    StressFactor,
    FeedbackConfig,
    simulate_coupled_jump_diffusion_feedback,
)
from modules.drift_neural import LSTMDriftEstimator, TransformerDriftEstimator
from modules.drift_kernel import KernelDriftEstimator

__all__ = [
    # Volatility
    'LocalVolatilityCalibrator',
    
    # Jumps
    'StaticJumpDetector',
    'NeuralJumpDetector',
    
    # Solver
    'JumpDiffusionEulerSolver',
    
    # Feedback
    'StressFactor',
    'FeedbackConfig',
    'simulate_coupled_jump_diffusion_feedback',
    
    # Drift
    'LSTMDriftEstimator',
    'TransformerDriftEstimator',
    'KernelDriftEstimator',
]
