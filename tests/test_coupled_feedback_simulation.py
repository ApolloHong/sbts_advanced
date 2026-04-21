import unittest

import numpy as np

from modules.feedback import simulate_coupled_jump_diffusion_feedback


class CoupledFeedbackSimulationTests(unittest.TestCase):
    def test_no_jumps_keeps_stress_zero(self):
        X, S = simulate_coupled_jump_diffusion_feedback(
            X_0=0.0,
            T=1.0,
            N_steps=8,
            mu=0.01,
            sigma_LV_func=0.2,
            kappa=0.8,
            gamma=1.0,
            lambda_jump=0.0,
            mu_J=0.0,
            delta_J=0.1,
            num_paths=5,
            rng=123,
        )

        self.assertEqual(X.shape, (5, 9))
        self.assertEqual(S.shape, (5, 9))
        np.testing.assert_allclose(S, 0.0)

    def test_jump_and_stress_use_same_step_realization(self):
        gamma = 1.7
        kappa = 0.8
        dt = 1.0 / 4
        X, S = simulate_coupled_jump_diffusion_feedback(
            X_0=0.0,
            T=1.0,
            N_steps=4,
            mu=0.0,
            sigma_LV_func=0.0,
            kappa=kappa,
            gamma=gamma,
            lambda_jump=20.0,
            mu_J=0.0,
            delta_J=0.05,
            num_paths=10,
            rng=7,
        )

        increments = np.diff(X, axis=1)
        expected_stress = S[:, :-1] * np.exp(-kappa * dt) + gamma * np.abs(increments)
        np.testing.assert_allclose(S[:, 1:], expected_stress, rtol=1e-12, atol=1e-12)
        self.assertGreater(np.count_nonzero(increments), 0)

    def test_vectorized_callable_coefficients(self):
        def mu(t, x):
            return 0.01 + 0.02 * x

        def sigma_lv(t, x):
            return np.full_like(x, 0.15 + 0.01 * t)

        X, S = simulate_coupled_jump_diffusion_feedback(
            X_0=np.array([0.0, 0.1, -0.1]),
            T=0.5,
            N_steps=5,
            mu=mu,
            sigma_LV_func=sigma_lv,
            kappa=1.0,
            gamma=0.5,
            lambda_jump=2.0,
            mu_J=0.0,
            delta_J=0.02,
            num_paths=3,
            rng=99,
            include_initial=False,
        )

        self.assertEqual(X.shape, (3, 5))
        self.assertEqual(S.shape, (3, 5))
        self.assertTrue(np.all(np.isfinite(X)))
        self.assertTrue(np.all(S >= 0.0))


if __name__ == "__main__":
    unittest.main()
