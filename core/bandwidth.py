import numpy as np
from sklearn.model_selection import KFold
import logging
import optuna
import sys

# Use the project-wide logger.
logger = logging.getLogger("SBTS")

# Silence Optuna's verbose logs unless debugging is needed.
optuna.logging.set_verbosity(optuna.logging.WARNING)

def silverman_rule(data):
    """
    Calculates bandwidth using Silverman's Rule of Thumb.
    """
    n = len(data)
    sigma = np.std(data)
    # Avoid a zero bandwidth when the sample variance is degenerate.
    if sigma < 1e-8: sigma = 1.0
    h = 1.06 * sigma * (n ** (-0.2))
    return h

class BandwidthSelector:
    """
    Implements Adaptive Bandwidth Selection via Bayesian Optimization (Optuna).
    """
    def __init__(self, n_trials=20, n_splits=3):
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.best_h = None
        self.study = None # Store the Optuna study object.

    def compute_terminal_mse(self, generated_terminal, actual_terminal):
        """
        Metric: Minimize Mean Squared Error (MSE) of terminal values.
        """
        # generated: (N_val, N_paths_per_sample, D)
        gen_mean = np.mean(generated_terminal, axis=1) # (N_val, D)
        squared_diffs = (gen_mean - actual_terminal) ** 2
        return np.mean(squared_diffs)

    def fit(self, trajectories, drift_estimator_cls, simulator_func, dt):
        """
        Bayesian Optimization to find optimal h.
        """
        # 1. Determine Search Space using Silverman
        flat_data = trajectories.flatten()
        h_base = silverman_rule(flat_data)
        
        # Search range: [0.1 * silverman, 10 * silverman]
        search_low = h_base * 0.1
        search_high = h_base * 10.0
        
        logger.info(f"   [BayesOpt] Base h={h_base:.4f}. Search Space: [{search_low:.5f}, {search_high:.5f}]")
        
        # 2. Prepare Data (Subsample if too large)
        n_total = len(trajectories)
        if n_total > 500:
            indices = np.random.choice(n_total, 500, replace=False)
            subset_traj = trajectories[indices]
        else:
            subset_traj = trajectories
            
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # 3. Define Objective Function for Optuna
        def objective(trial):
            # Suggest a value for h in log domain (smart sampling)
            h = trial.suggest_float("h", search_low, search_high, log=True)
            
            fold_mses = []
            
            for train_idx, val_idx in kf.split(subset_traj):
                train_data = subset_traj[train_idx]
                val_data = subset_traj[val_idx]
                
                val_start = val_data[:, 0, :]
                val_actual = val_data[:, -1, :]
                
                # A. Fit
                try:
                    estimator = drift_estimator_cls(bandwidth=h)
                    estimator.fit(train_data, dt)
                    
                    # B. Simulate
                    # simulator_func handles batch generation
                    gen_paths = simulator_func(val_start, estimator.predict)
                    
                    # C. Evaluate
                    mse = self.compute_terminal_mse(gen_paths[:, -1, :][:, np.newaxis, :], val_actual)
                    fold_mses.append(mse)
                except Exception as e:
                    # If simulation fails (e.g. divergence), return infinity
                    return float('inf')

            avg_mse = np.mean(fold_mses)
            
            # Simple progress logging
            # print(f"      Trial {trial.number}: h={h:.5f}, MSE={avg_mse:.6f}")
            return avg_mse

        # 4. Run Optimization
        logger.info(f"   [BayesOpt] Starting optimization with {self.n_trials} trials...")
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_h = self.study.best_params["h"]
        best_mse = self.study.best_value
        
        logger.info(f"   [BayesOpt] Best Bandwidth Found: {self.best_h:.5f} (MSE: {best_mse:.6f})")
        
        return self.best_h
    
    def get_history(self):
        """Returns history for visualization: dict {h: mse}"""
        if self.study is None:
            return {}
        # Extract trials
        trials = self.study.trials
        # Filter completed trials
        valid_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        # Sort by h for plotting
        history = {t.params['h']: t.value for t in valid_trials}
        return history
