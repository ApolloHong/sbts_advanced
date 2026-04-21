import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from scipy.optimize import minimize
    from scipy.stats import wasserstein_distance
except ImportError:  # pragma: no cover - scipy is a project dependency
    minimize = None
    wasserstein_distance = None


_PARAM_FLOOR = 1e-12


def _as_3d_paths(X):
    """Return path data as (n_paths, n_steps, n_features)."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        return X[:, :, np.newaxis], True
    if X.ndim == 3:
        return X, False
    raise ValueError("X must have shape (M, N) or (M, N, d)")


def _maybe_scalar(values, was_2d):
    values = np.asarray(values, dtype=np.float64)
    if was_2d:
        return float(values[0])
    return values


def _normalize_params(params, n_features):
    """Coerce tuple/dict/scalar parameter containers to per-feature arrays."""
    if isinstance(params, dict):
        sigma = params.get("sigma", params.get("sigma_hat", 0.1))
        lambda0 = params.get("lambda0", params.get("lambda", params.get("intensity", 1e-4)))
        c = params.get("c", params.get("mean", 0.0))
        gamma = params.get("gamma", params.get("std", 0.1))
    else:
        sigma, lambda0, c, gamma = params

    def arr(v, default):
        if v is None:
            out = np.full(n_features, default, dtype=np.float64)
        else:
            out = np.asarray(v, dtype=np.float64)
            if out.ndim == 0:
                out = np.full(n_features, float(out), dtype=np.float64)
            elif out.size != n_features:
                out = np.resize(out, n_features).astype(np.float64)
        return out

    return (
        arr(sigma, 0.1),
        arr(lambda0, 1e-4),
        arr(c, 0.0),
        arr(gamma, 0.1),
    )


def estimate_jump_params_threshold(X, dt, r=3.0):
    """
    Stage 1: Mancini-style threshold jump detection with bipower variation.

    Args:
        X: Path array of shape (M, N) or (M, N, d).
        dt: Positive time step size.
        r: Threshold multiplier.

    Returns:
        Tuple (sigma, lambda0, c, gamma). For multidimensional input each item
        is an array of shape (d,); for 2D input each item is a float.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    paths, was_2d = _as_3d_paths(X)
    n_paths, n_steps, n_features = paths.shape
    if n_steps < 2:
        raise ValueError("X must contain at least two time steps")

    increments = np.diff(paths, axis=1)
    n_intervals = n_steps - 1

    sigma_hat = np.zeros(n_features, dtype=np.float64)
    lambda_hat = np.zeros(n_features, dtype=np.float64)
    c_hat = np.zeros(n_features, dtype=np.float64)
    gamma_hat = np.zeros(n_features, dtype=np.float64)

    for k in range(n_features):
        dX = increments[:, :, k]
        finite = np.isfinite(dX)
        clean = dX[finite]
        if clean.size == 0:
            sigma_hat[k] = 1e-4
            gamma_hat[k] = 1e-3
            continue

        if dX.shape[1] > 1:
            pair_mask = finite[:, 1:] & finite[:, :-1]
            bv_terms = np.abs(dX[:, 1:]) * np.abs(dX[:, :-1])
            bv_terms = bv_terms[pair_mask]
            if bv_terms.size:
                bv = (np.pi / 2.0) * float(np.mean(bv_terms))
            else:
                bv = float(np.nanvar(clean))
        else:
            bv = float(np.nanvar(clean))

        sigma2 = max(bv / dt, 1e-8)
        sigma = np.sqrt(sigma2)
        is_jump = np.zeros_like(dX, dtype=bool)

        for _ in range(2):
            threshold = float(r) * sigma * np.sqrt(dt)
            is_jump = finite & (np.abs(dX) > threshold)
            non_jump = finite & ~is_jump
            if np.any(non_jump):
                truncated_second_moment = np.mean(np.where(non_jump, dX ** 2, 0.0))
                sigma2 = max(truncated_second_moment / dt, 1e-8)
                sigma = np.sqrt(sigma2)
            else:
                sigma = max(sigma, 1e-4)

        jump_increments = dX[is_jump]
        n_jumps = int(jump_increments.size)
        total_time = n_paths * n_intervals * dt

        sigma_hat[k] = max(float(sigma), 1e-4)
        lambda_hat[k] = max(float(n_jumps / total_time), 0.0) if total_time > 0 else 0.0
        if n_jumps > 0:
            c_hat[k] = float(np.mean(jump_increments))
            jump_var = float(np.var(jump_increments))
            gamma2 = max(jump_var - sigma_hat[k] ** 2 * dt, 1e-6)
            gamma_hat[k] = np.sqrt(gamma2)
        else:
            c_hat[k] = 0.0
            gamma_hat[k] = 1e-3

    return (
        _maybe_scalar(sigma_hat, was_2d),
        _maybe_scalar(lambda_hat, was_2d),
        _maybe_scalar(c_hat, was_2d),
        _maybe_scalar(gamma_hat, was_2d),
    )


def estimate_jump_params_ecf(X, dt, init_params, fix_c=False):
    """
    Stage 2: Empirical characteristic-function refinement.

    Args:
        X: Path array of shape (M, N) or (M, N, d).
        dt: Positive time step size.
        init_params: Stage 1 tuple/dict (sigma, lambda0, c, gamma).
        fix_c: If True, hold c at zero during optimization.

    Returns:
        Refined tuple (sigma, lambda0, c, gamma).
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    paths, was_2d = _as_3d_paths(X)
    _, n_steps, n_features = paths.shape
    if n_steps < 2:
        raise ValueError("X must contain at least two time steps")

    init_sigma, init_lambda, init_c, init_gamma = _normalize_params(init_params, n_features)
    increments = np.diff(paths, axis=1)
    u_grid = np.geomspace(0.5, 10.0, 30)
    weights = np.exp(-(u_grid ** 2) / 20.0)

    sigma_out = np.zeros(n_features, dtype=np.float64)
    lambda_out = np.zeros(n_features, dtype=np.float64)
    c_out = np.zeros(n_features, dtype=np.float64)
    gamma_out = np.zeros(n_features, dtype=np.float64)

    for k in range(n_features):
        dX = increments[:, :, k].reshape(-1)
        dX = dX[np.isfinite(dX)]
        if dX.size == 0 or minimize is None:
            sigma_out[k] = max(init_sigma[k], 1e-4)
            lambda_out[k] = max(init_lambda[k], 1e-4)
            c_out[k] = 0.0 if fix_c else float(np.clip(init_c[k], -5.0, 5.0))
            gamma_out[k] = max(init_gamma[k], 1e-4)
            continue

        phi_hat = np.array([np.mean(np.exp(1j * u * dX)) for u in u_grid])
        start_sigma = float(np.clip(init_sigma[k], 1e-4, 10.0))
        start_lambda = float(np.clip(max(init_lambda[k], 1e-4), 1e-4, 1000.0))
        start_c = 0.0 if fix_c else float(np.clip(init_c[k], -5.0, 5.0))
        start_gamma = float(np.clip(init_gamma[k], 1e-4, 10.0))

        if fix_c:
            theta0 = np.array([np.log(start_sigma), np.log(start_lambda), np.log(start_gamma)])
            bounds = [(np.log(1e-4), np.log(10.0)), (np.log(1e-4), np.log(1000.0)), (np.log(1e-4), np.log(10.0))]
        else:
            theta0 = np.array([np.log(start_sigma), np.log(start_lambda), start_c, np.log(start_gamma)])
            bounds = [(np.log(1e-4), np.log(10.0)), (np.log(1e-4), np.log(1000.0)), (-5.0, 5.0), (np.log(1e-4), np.log(10.0))]

        def unpack(theta):
            if fix_c:
                sigma = np.exp(theta[0])
                lambda0 = np.exp(theta[1])
                c = 0.0
                gamma = np.exp(theta[2])
            else:
                sigma = np.exp(theta[0])
                lambda0 = np.exp(theta[1])
                c = theta[2]
                gamma = np.exp(theta[3])
            return sigma, lambda0, c, gamma

        def objective(theta):
            sigma, lambda0, c, gamma = unpack(theta)
            exponent = (
                -0.5 * (sigma ** 2) * (u_grid ** 2) * dt
                + 1j * u_grid * (lambda0 * dt * c)
                + lambda0 * dt * (np.exp(1j * u_grid * c - 0.5 * (gamma ** 2) * (u_grid ** 2)) - 1.0)
            )
            phi_theory = np.exp(exponent)
            diff = phi_hat - phi_theory
            loss = np.sum(weights * (diff.real ** 2 + diff.imag ** 2))
            if not np.isfinite(loss):
                return 1e12
            return float(loss)

        try:
            result = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)
            theta = result.x if result.success or np.isfinite(result.fun) else theta0
        except Exception as exc:  # pragma: no cover - optimizer failure fallback
            warnings.warn(f"ECF jump calibration failed for feature {k}: {exc}")
            theta = theta0

        sigma, lambda0, c, gamma = unpack(theta)
        sigma_out[k] = float(np.clip(sigma, 1e-4, 10.0))
        lambda_out[k] = float(np.clip(lambda0, 1e-4, 1000.0))
        c_out[k] = 0.0 if fix_c else float(np.clip(c, -5.0, 5.0))
        gamma_out[k] = float(np.clip(gamma, 1e-4, 10.0))

    return (
        _maybe_scalar(sigma_out, was_2d),
        _maybe_scalar(lambda_out, was_2d),
        _maybe_scalar(c_out, was_2d),
        _maybe_scalar(gamma_out, was_2d),
    )


def _simulate_merton_paths(n_synth, n_steps, dt, sigma, lambda0, c, gamma, x0):
    paths = np.zeros((n_synth, n_steps), dtype=np.float64)
    paths[:, 0] = x0
    if n_steps <= 1:
        return paths

    diffusion = sigma * np.sqrt(dt) * np.random.randn(n_synth, n_steps - 1)
    jump_counts = np.random.poisson(lambda0 * dt, size=(n_synth, n_steps - 1))
    jump_sizes = np.zeros_like(diffusion)
    active = jump_counts > 0
    if np.any(active):
        counts = jump_counts[active]
        jump_sizes[active] = counts * c + gamma * np.sqrt(counts) * np.random.randn(counts.size)
    increments = diffusion + jump_sizes
    paths[:, 1:] = x0[:, np.newaxis] + np.cumsum(increments, axis=1)
    return paths


def calibrate_lambda_qv(
    X,
    dt,
    sigma,
    gamma,
    sbjts_model,
    n_synth=200,
    lambda_init=None,
    c=0.0,
    n_grid=10,
    variance_tolerance=0.25,
):
    """
    Stage 3: Indirect lambda calibration via quadratic-variation matching.

    The function uses ``sbjts_model.generate_reference_paths`` when available.
    Otherwise it falls back to simulating the Merton reference measure with the
    supplied parameters.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    paths, was_2d = _as_3d_paths(X)
    n_paths, n_steps, n_features = paths.shape
    sigma_arr, lambda_arr, c_arr, gamma_arr = _normalize_params(
        (sigma, lambda_init, c, gamma),
        n_features,
    )

    lambda_out = np.zeros(n_features, dtype=np.float64)
    increments = np.diff(paths, axis=1)

    for k in range(n_features):
        real_dX = increments[:, :, k]
        qv_real = np.sum(real_dX ** 2, axis=1)

        empirical_var = float(np.mean(np.var(real_dX, axis=0)) / dt)
        model_var = float(sigma_arr[k] ** 2 + lambda_arr[k] * (c_arr[k] ** 2 + gamma_arr[k] ** 2))
        denom = max(abs(empirical_var), _PARAM_FLOOR)
        rel_error = abs(model_var - empirical_var) / denom
        if rel_error > variance_tolerance:
            warnings.warn(
                "SBJTS variance check exceeded tolerance for feature "
                f"{k}: model={model_var:.6g}, empirical={empirical_var:.6g}, "
                f"relative_error={rel_error:.3f}"
            )

        center = float(np.clip(lambda_arr[k], 1e-4, 1000.0))
        lower = max(center / 5.0, 1e-4)
        upper = min(center * 5.0, 1000.0)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            lower, upper = 1e-4, 1.0
        candidates = np.geomspace(lower, upper, max(2, int(n_grid)))

        best_lambda = center
        best_distance = np.inf
        for lambda_candidate in candidates:
            x0 = paths[np.random.choice(n_paths, int(n_synth), replace=True), 0, k]
            generated = None
            if sbjts_model is not None and hasattr(sbjts_model, "generate_reference_paths"):
                try:
                    generated = sbjts_model.generate_reference_paths(
                        n_samples=int(n_synth),
                        n_steps=n_steps,
                        dt=dt,
                        sigma=sigma_arr[k],
                        lambda0=float(lambda_candidate),
                        c=c_arr[k],
                        gamma=gamma_arr[k],
                        x0=x0,
                        feature_idx=k,
                    )
                except TypeError:
                    generated = None
            if generated is None:
                generated = _simulate_merton_paths(
                    int(n_synth),
                    n_steps,
                    dt,
                    sigma_arr[k],
                    float(lambda_candidate),
                    c_arr[k],
                    gamma_arr[k],
                    x0,
                )

            generated = np.asarray(generated, dtype=np.float64)
            if generated.ndim == 3:
                generated = generated[:, :, k if generated.shape[2] > k else 0]
            qv_synth = np.sum(np.diff(generated, axis=1) ** 2, axis=1)

            if wasserstein_distance is not None:
                distance = wasserstein_distance(np.sort(qv_real), np.sort(qv_synth))
            else:  # pragma: no cover - scipy is a project dependency
                m = min(len(qv_real), len(qv_synth))
                distance = float(np.mean(np.abs(np.sort(qv_real)[:m] - np.sort(qv_synth)[:m])))

            if distance < best_distance:
                best_distance = float(distance)
                best_lambda = float(lambda_candidate)

        lambda_out[k] = np.clip(best_lambda, 1e-4, 1000.0)

    return _maybe_scalar(lambda_out, was_2d)


def calibrate_all(X, dt, sbjts_model, fix_c=True):
    """
    Run the three-stage SBJTS reference parameter calibration pipeline.

    Returns:
        Tuple (sigma, lambda0, c, gamma). Multidimensional inputs return one
        array per parameter with shape (d,).
    """
    paths, was_2d = _as_3d_paths(X)
    config = getattr(sbjts_model, "config", {}) if sbjts_model is not None else {}
    r = float(config.get("sbjts_threshold_r", config.get("jump_threshold_std", 3.0)))
    n_synth = int(config.get("sbjts_qv_n_synth", 200))
    n_grid = int(config.get("sbjts_qv_n_grid", 10))
    variance_tolerance = float(config.get("sbjts_variance_tolerance", 0.25))

    print("   [SBJTS] Stage 1/3: threshold jump parameter estimation")
    stage1 = estimate_jump_params_threshold(paths, dt, r=r)
    s1_sigma, s1_lambda, s1_c, s1_gamma = _normalize_params(stage1, paths.shape[2])
    if fix_c:
        s1_c = np.zeros_like(s1_c)
        stage1 = (s1_sigma, s1_lambda, s1_c, s1_gamma)
    print(
        "   [SBJTS] Stage 1 estimates: "
        f"sigma={np.array2string(s1_sigma, precision=4)}, "
        f"lambda={np.array2string(s1_lambda, precision=4)}, "
        f"c={np.array2string(s1_c, precision=4)}, "
        f"gamma={np.array2string(s1_gamma, precision=4)}"
    )

    print("   [SBJTS] Stage 2/3: ECF refinement")
    stage2 = estimate_jump_params_ecf(paths, dt, stage1, fix_c=fix_c)
    s2_sigma, s2_lambda, s2_c, s2_gamma = _normalize_params(stage2, paths.shape[2])
    print(
        "   [SBJTS] Stage 2 estimates: "
        f"sigma={np.array2string(s2_sigma, precision=4)}, "
        f"lambda={np.array2string(s2_lambda, precision=4)}, "
        f"c={np.array2string(s2_c, precision=4)}, "
        f"gamma={np.array2string(s2_gamma, precision=4)}"
    )

    if sbjts_model is not None:
        setattr(
            sbjts_model,
            "_sbjts_stage2_params",
            {
                "sigma": s2_sigma.copy(),
                "lambda0": s2_lambda.copy(),
                "c": s2_c.copy(),
                "gamma": s2_gamma.copy(),
            },
        )

    print("   [SBJTS] Stage 3/3: QV lambda calibration")
    lambda_star = calibrate_lambda_qv(
        paths,
        dt,
        s2_sigma,
        s2_gamma,
        sbjts_model,
        n_synth=n_synth,
        lambda_init=s2_lambda,
        c=s2_c,
        n_grid=n_grid,
        variance_tolerance=variance_tolerance,
    )
    _, lambda_arr, _, _ = _normalize_params((s2_sigma, lambda_star, s2_c, s2_gamma), paths.shape[2])
    print(
        "   [SBJTS] Final estimates: "
        f"sigma={np.array2string(s2_sigma, precision=4)}, "
        f"lambda={np.array2string(lambda_arr, precision=4)}, "
        f"c={np.array2string(s2_c, precision=4)}, "
        f"gamma={np.array2string(s2_gamma, precision=4)}"
    )

    return (
        _maybe_scalar(s2_sigma, was_2d),
        _maybe_scalar(lambda_arr, was_2d),
        _maybe_scalar(s2_c, was_2d),
        _maybe_scalar(s2_gamma, was_2d),
    )

class VolatilityCalibrator:
    """
    Implements Local Volatility Calibration (Phase 3).
    Fits a surface σ_LV(t, x) based on realized volatility of training paths.
    
    REFACTORED (Step 1): 
    - Now accepts purified returns from JumpDetector
    - Uses Filter & Interpolate purification instead of Clipping
    - Results in proper "Smile/Skew" volatility surface shape
    
    The volatility surface should exhibit:
    - Higher volatility at extreme values (tails)
    - Lower volatility near the mean
    - NOT an "inverted U" shape (which indicates incorrect purification)
    """
    def __init__(self, dt, method='kernel', bandwidth=0.1):
        self.dt = dt
        self.method = method
        self.bandwidth = bandwidth 
        self.model = None
        self.scaler_X = StandardScaler() 
        self.min_vol = 1e-4
        self._surface_diagnostics = {}  # Store diagnostics for validation

    def _compute_instantaneous_vol(self, trajectories):
        """
        Compute instantaneous realized volatility from trajectory data.
        
        Args:
            trajectories: (N, T, D) array of price/return paths
            
        Returns:
            X_features: (N*(T-1), 1+D) array of (t, x) features
            Y_target: (N*(T-1), D) array of realized volatility
        """
        X_current = trajectories[:, :-1, :]
        X_next = trajectories[:, 1:, :]
        dX = X_next - X_current
        
        # Realized variance: (dX)^2 / dt
        realized_var = (dX ** 2) / self.dt
        
        N, T_minus_1, D = realized_var.shape
        t_grid = np.linspace(0, self.dt * T_minus_1, T_minus_1)
        
        X_features = []
        Y_target = []
        
        for i in range(T_minus_1):
            t = t_grid[i]
            current_x = X_current[:, i, :]
            current_var = realized_var[:, i, :]
            
            t_col = np.full((N, 1), t)
            
            feat = np.hstack([t_col, current_x])
            X_features.append(feat)
            Y_target.append(current_var)
            
        return np.vstack(X_features), np.vstack(Y_target)

    def fit(self, trajectories, purified_trajectories=None):
        """
        Fit the Local Volatility surface using Kernel Ridge Regression.
        
        IMPORTANT: For correct volatility smile/skew, use purified_trajectories
        from JumpDetector.get_purified_returns() instead of raw data.
        
        Args:
            trajectories: (N, T, D) array of original paths (for feature extraction)
            purified_trajectories: (N, T, D) array of purified paths (for vol estimation)
                                   If None, uses trajectories directly
                                   
        Returns:
            self: Fitted calibrator
        """
        # Use purified data for volatility estimation if provided
        data_for_vol = purified_trajectories if purified_trajectories is not None else trajectories
        
        print("   [Calibration] Computing realized volatility surface...")
        if purified_trajectories is not None:
            print("   [Calibration] Using Filter & Interpolate purified returns")
        
        X_train, Y_train = self._compute_instantaneous_vol(data_for_vol)
        
        # Convert variance to volatility (standard deviation)
        Y_vol = np.sqrt(np.maximum(Y_train, 1e-8))
        
        # Subsample for efficiency
        limit = 20000
        if len(X_train) > limit:
            idx = np.random.choice(len(X_train), limit, replace=False)
            X_train = X_train[idx]
            Y_vol = Y_vol[idx]

        # Standardize features (t, x)
        X_train_scaled = self.scaler_X.fit_transform(X_train)

        if self.method == 'kernel':
            gamma = 1.0 / (2 * self.bandwidth ** 2)
            self.model = KernelRidge(kernel='rbf', gamma=gamma, alpha=0.1)
        elif self.method == 'knn':
            n_neighbors = int(max(5, 100 * self.bandwidth))
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_train_scaled, Y_vol)
        
        # Store diagnostics for validation
        self._compute_surface_diagnostics(X_train, Y_vol)
            
        print("   [Calibration] Local Volatility Surface fitted (Normalized).")
        return self

    def _compute_surface_diagnostics(self, X_train, Y_vol):
        """
        Compute diagnostics to verify volatility surface shape.
        
        A correct surface should show:
        - Higher volatility at extreme x values (smile/skew)
        - NOT highest volatility at x=0 (inverted U)
        
        Args:
            X_train: Feature array (t, x)
            Y_vol: Volatility targets
        """
        # Extract x values (assuming 1D for simplicity)
        x_values = X_train[:, 1] if X_train.shape[1] > 1 else X_train[:, 0]
        
        # Compute volatility by x-quantile
        quantiles = np.percentile(x_values, [10, 25, 50, 75, 90])
        vol_by_quantile = []
        
        for i in range(len(quantiles) - 1):
            mask = (x_values >= quantiles[i]) & (x_values < quantiles[i+1])
            if np.any(mask):
                vol_by_quantile.append(np.mean(Y_vol[mask]))
            else:
                vol_by_quantile.append(np.nan)
        
        # Check for inverted U shape (center > edges)
        if len(vol_by_quantile) >= 3:
            center_vol = vol_by_quantile[len(vol_by_quantile)//2]
            edge_vol = (vol_by_quantile[0] + vol_by_quantile[-1]) / 2
            
            self._surface_diagnostics = {
                "center_volatility": center_vol,
                "edge_volatility": edge_vol,
                "is_smile_shape": edge_vol > center_vol,
                "vol_by_quantile": vol_by_quantile
            }
            
            if edge_vol > center_vol:
                print("   [Calibration] ✓ Volatility surface exhibits Smile/Skew shape")
            else:
                print("   [Calibration] ⚠ Warning: Surface may have inverted U shape")

    def predict(self, t, x):
        """
        Predicts volatility σ_LV(t, x).
        
        Args:
            x: (N, D) array of state values
            t: scalar OR array of shape (N,) for time
            
        Returns:
            vol: (N, D) array of predicted volatilities
        """
        if self.model is None:
            raise ValueError("Calibrator not fitted.")
            
        if x.ndim == 1:
            x = x[np.newaxis, :]
            
        # Handle Vectorized Time Input
        t = np.asarray(t)
        if t.ndim == 0:
            # Scalar case: Broadcast to (N, 1)
            t_col = np.full((x.shape[0], 1), t.item())
        else:
            # Vector case: Reshape to (N, 1)
            if t.shape[0] != x.shape[0]:
                raise ValueError(f"Time dimension {t.shape} does not match State dimension {x.shape}")
            t_col = t.reshape(-1, 1)

        query = np.hstack([t_col, x])
        
        query_scaled = self.scaler_X.transform(query)
        pred_vol = self.model.predict(query_scaled)
        
        return np.maximum(pred_vol, self.min_vol)

    def get_surface_diagnostics(self):
        """
        Returns diagnostics about the fitted volatility surface.
        
        Returns:
            dict: Surface shape diagnostics
        """
        return self._surface_diagnostics
