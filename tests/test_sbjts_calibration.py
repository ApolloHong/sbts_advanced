import unittest

import numpy as np

from models.calibration import (
    calibrate_all,
    calibrate_lambda_qv,
    estimate_jump_params_ecf,
    estimate_jump_params_threshold,
)


def simulate_merton(n_paths, n_steps, dt, sigma, lambda0, c, gamma, seed=123):
    rng = np.random.default_rng(seed)
    increments = sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps - 1))
    counts = rng.poisson(lambda0 * dt, size=(n_paths, n_steps - 1))
    active = counts > 0
    jumps = np.zeros_like(increments)
    if np.any(active):
        active_counts = counts[active]
        jumps[active] = active_counts * c + gamma * np.sqrt(active_counts) * rng.standard_normal(active_counts.size)
    paths = np.zeros((n_paths, n_steps), dtype=np.float64)
    paths[:, 1:] = np.cumsum(increments + jumps, axis=1)
    return paths


class DummySBJTSModel:
    def __init__(self):
        self.config = {
            "sbjts_threshold_r": 3.0,
            "sbjts_qv_n_synth": 120,
            "sbjts_qv_n_grid": 8,
            "sbjts_variance_tolerance": 1.0,
        }

    def generate_reference_paths(self, n_samples, n_steps, dt, sigma, lambda0, c, gamma, x0=None, feature_idx=None):
        seed = int(1000 * lambda0) % (2**32 - 1)
        paths = simulate_merton(n_samples, n_steps, dt, sigma, lambda0, c, gamma, seed=seed)
        if x0 is not None:
            paths = paths + np.asarray(x0).reshape(-1, 1)
        return paths


class SBJTSCalibrationTests(unittest.TestCase):
    def test_threshold_and_ecf_recover_merton_parameters(self):
        sigma_true = 0.25
        lambda_true = 5.0
        c_true = 0.0
        gamma_true = 0.7
        dt = 1.0 / 80.0
        X = simulate_merton(900, 121, dt, sigma_true, lambda_true, c_true, gamma_true, seed=7)

        stage1 = estimate_jump_params_threshold(X, dt, r=3.0)
        sigma1, lambda1, c1, gamma1 = stage1
        self.assertLess(abs(sigma1 - sigma_true), 0.12)
        self.assertLess(abs(lambda1 - lambda_true) / lambda_true, 0.45)
        self.assertLess(abs(c1 - c_true), 0.15)
        self.assertLess(abs(gamma1 - gamma_true) / gamma_true, 0.35)

        sigma2, lambda2, c2, gamma2 = estimate_jump_params_ecf(X, dt, stage1, fix_c=True)
        self.assertLess(abs(sigma2 - sigma_true), 0.12)
        self.assertLess(abs(lambda2 - lambda_true) / lambda_true, 0.55)
        self.assertEqual(c2, 0.0)
        self.assertLess(abs(gamma2 - gamma_true) / gamma_true, 0.45)

    def test_multidimensional_outputs_have_component_shape(self):
        dt = 1.0 / 60.0
        x0 = simulate_merton(500, 91, dt, 0.2, 4.0, 0.0, 0.6, seed=10)
        x1 = simulate_merton(500, 91, dt, 0.35, 7.0, 0.0, 0.8, seed=11)
        X = np.stack([x0, x1], axis=-1)

        params = estimate_jump_params_threshold(X, dt)
        for value in params:
            self.assertEqual(np.asarray(value).shape, (2,))

    def test_qv_calibration_and_master_pipeline_return_finite_values(self):
        sigma_true = 0.2
        lambda_true = 4.0
        gamma_true = 0.6
        dt = 1.0 / 60.0
        X = simulate_merton(650, 101, dt, sigma_true, lambda_true, 0.0, gamma_true, seed=21)
        model = DummySBJTSModel()

        lambda_star = calibrate_lambda_qv(
            X,
            dt,
            sigma_true,
            gamma_true,
            model,
            n_synth=120,
            lambda_init=lambda_true,
            c=0.0,
            n_grid=8,
            variance_tolerance=1.0,
        )
        self.assertTrue(np.isfinite(lambda_star))
        self.assertLess(abs(lambda_star - lambda_true) / lambda_true, 0.8)

        sigma, lambda0, c, gamma = calibrate_all(X, dt, model, fix_c=True)
        for value in (sigma, lambda0, c, gamma):
            self.assertTrue(np.all(np.isfinite(value)))
        self.assertEqual(c, 0.0)
        self.assertGreater(lambda0, 0.0)


if __name__ == "__main__":
    unittest.main()
