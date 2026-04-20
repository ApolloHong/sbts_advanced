# Model and Benchmark Explanation

This document summarizes the active model registry, benchmark datasets, generation protocols, and the main evaluation settings used by the current project.

The maintained experiment entry points are:

- `main.py`: CLI training, generation, evaluation, and plotting.
- `main_pipeline.ipynb`: notebook benchmark pipeline.
- `models/factory.py`: source of truth for model names, aliases, and default configuration.

## Active Model Registry

`models/factory.py` currently registers 11 models:

| Model key | Class | Main idea |
|---|---|---|
| `jd_sbts` | `JDSBTS` | Base jump-diffusion SBTS model with static jump detection, local volatility, LSTM drift, and Euler generation. |
| `jd_sbts_f` | `JDSBTSF` | Adds a transient stress factor for post-jump volatility clustering. |
| `jd_sbts_neural` | `JDSBTSNeural` | Replaces static-only jump intensity with a neural jump-intensity module. |
| `jd_sbts_f_neural` | `JDSBTSFNeural` | Combines neural jump modeling and feedback volatility amplification. |
| `lightsb` | `LightSB` | Window-level Light Schrodinger Bridge using sum-exp quadratic potentials. |
| `lightsb_path` | `PathLightSB` | Path-level Light Schrodinger Bridge using one-step potential bridges. |
| `numba_sb` | `NumbaSB` | Fast Markovian SB/Brownian bridge baseline. |
| `timegan` | `TimeGAN` | GRU-based TimeGAN-style baseline. |
| `diffusion_ts` | `DiffusionTS` | DDPM-style time-series diffusion baseline. |
| `rnn` | `RNNBaseline` | Autoregressive GRU/LSTM baseline. |
| `transformer_ar` | `TransformerARBaseline` | Causal autoregressive Transformer baseline. |

Aliases currently point to canonical model names. For example, `path_lightsb`, `light_sb_path`, and `step_lightsb` resolve to `lightsb_path`.

## Current Notebook Benchmark

The current `main_pipeline.ipynb` source is configured for:

```text
BENCHMARK_DATASET = stock
STOCK_TICKER = QQQ
window_length = 60
normalization = log_return
n_generate = 512
MODELS_TO_RUN = jd_sbts, jd_sbts_f, lightsb, lightsb_path, timegan, rnn, transformer_ar
```

For the cached QQQ stock dataset, the generated and evaluated sample shape is expected to be:

```text
(512, 60, 6)
```

The 512 value is a cap. The notebook uses:

```python
n_generate = min(TRAINING_OVERRIDES["n_generate"], len(eval_pool))
shared_eval_subset = eval_pool[:n_generate]
```

So every metric uses the same `shared_eval_subset` and the same generated sample count, unless the test split has fewer than 512 windows.

## Metric Sample Counts

For the current stock benchmark with `n_generate = 512`:

| Metric family | Real samples | Generated samples | Notes |
|---|---:|---:|---|
| Numba statistical metrics | 512 | 512 | Wasserstein, ACF MSE, and correlation distance compare `real_eval` and `gen_eval`. |
| Stylized facts | 0 direct real samples | 512 generated paths | Current code computes stylized facts from generated returns only. |
| Discriminative diagnostics | 512 | 512 | Repeated 10 times. Each repeat uses an 80/20 balanced train/test split. |
| Predictive score | 512 | 512 | Trains a predictor on generated windows and evaluates on real windows. |

The discriminative score uses:

```text
discriminative_iterations = 500
discriminative_repeats = 10
discriminative_train_rate = 0.8
discriminative_hidden_dim = 32
```

With 512 real and 512 generated windows, each repeat trains on:

```text
409 real windows + 409 generated windows
```

and tests on:

```text
103 real windows + 103 generated windows
```

The predictive score uses:

```text
predictive_iterations = 500
```

It trains on all 512 generated windows and evaluates on all 512 real windows. Internally, each window contributes one sequence input of length `T - 1`, and the GRU predictor is trained to predict the last target step.

## Generation Protocols

The benchmark uses protocol labels to make conditioning differences explicit:

| Protocol | Meaning | Current users |
|---|---|---|
| `shared_x0` | The model receives the same real initial states as conditioning. | `jd_sbts`, `jd_sbts_f`, `jd_sbts_neural`, `jd_sbts_f_neural`, `lightsb_path` |
| `shared_seed_only` | The model is unconditional with respect to the evaluation initial state; only the random seed is reset. | `lightsb`, `numba_sb`, `timegan`, `diffusion_ts` |
| `shared_prefix_len_*` | The model receives the same real prefix and autoregressively completes the suffix. | `rnn`, `transformer_ar` |

This distinction is important for LightSB:

- `lightsb` is window-level. It models an entire flattened window as one vector, so it does not naturally condition on the evaluation path's first point.
- `lightsb_path` is path-level. It models one-step transitions and can generate step by step from a supplied `x0`.

## Dataset Configurations

`main_pipeline.ipynb` defines four benchmark dataset names:

| Dataset | Type | Default shape |
|---|---|---:|
| `merton` | Simulated Merton jump-diffusion paths | `(1000, 60, 1)` |
| `ou_standard` | Simulated Ornstein-Uhlenbeck paths | `(1000, 60, 1)` |
| `ou_high_frequency` | Higher-frequency Ornstein-Uhlenbeck paths | `(1000, 60, 1)` |
| `stock` | Yahoo Finance close-price windows | ticker-dependent, processed windows of length 60 |

`google` remains as a backward-compatible alias for `stock`.

The simulated datasets use:

```text
train_frac = 0.70
val_frac = 0.15
test_frac = 0.15  # implicit remainder
shuffle = True
```

The stock dataset uses the same split fractions. With `window_length = 60`, each processed stock sample has 60 time steps over:

```text
Close
```

## Model Details

### `jd_sbts`

`jd_sbts` is the base Jump-Diffusion Schrodinger Bridge Time Series model. It combines:

- `StaticJumpDetector`: rolling z-score jump detection.
- `LocalVolatilityCalibrator`: KDE-style local volatility estimation.
- `LSTMDriftEstimator`: drift estimation on purified trajectories.
- `JumpDiffusionEulerSolver`: Euler-Maruyama generation with jump sampling.

Default key parameters include:

```text
jump_threshold_std = 4.0
jump_rolling_window = 20
vol_bandwidth = 0.5
vol_n_t_grid = 50
vol_n_x_grid = 100
lstm_hidden = 128
lstm_epochs = 50
lstm_lr = 0.005
lstm_dropout = 0.3
solver_backend = numba
```

### `jd_sbts_f`

`jd_sbts_f` adds a feedback stress factor to `jd_sbts`. When a jump occurs, the stress state increases; between jumps it decays. Effective volatility is amplified by this stress state, which gives the model an explicit volatility-clustering mechanism.

Default feedback parameters:

```text
feedback_kappa = 5.0
feedback_gamma = 0.5
```

### `jd_sbts_neural`

`jd_sbts_neural` keeps the SBTS local volatility and drift components but enables neural jump modeling. Static jump labels are used as supervision, and a neural intensity model is trained with focal loss to handle sparse jump events.

Default neural-jump parameters:

```text
neural_jump_hidden_dim = 64
neural_jump_epochs = 30
neural_jump_lr = 0.001
neural_jump_seq_len = 10
focal_alpha = 0.25
focal_gamma = 2.0
```

### `jd_sbts_f_neural`

`jd_sbts_f_neural` is the most complete SBTS variant. It combines:

- static jump statistics,
- neural jump intensity,
- local volatility calibration,
- LSTM drift estimation,
- feedback stress amplification.

Its default configuration is the union of the neural-jump and feedback settings above.

### `lightsb`

`lightsb` is the window-level Light Schrodinger Bridge baseline. It flattens each full time-series window and trains a potential-based bridge between a Gaussian source distribution and the empirical window distribution.

The current implementation uses sum-exp quadratic potentials with diagonal quadratic terms. The key point is that the input dimension is:

```text
flat_dim = sequence_length * n_features
```

For stock data with shape `(512, 60, 1)`, `flat_dim = 60`.

Default key parameters:

```text
lightsb_n_potentials = 20
lightsb_epsilon = 1.0
lightsb_s_diagonal_init = 0.1
lightsb_epochs = 100
lightsb_lr = 0.001
lightsb_batch_size = 256
lightsb_sampling_batch_size = 512
lightsb_source_std = 1.0
```

Because this is a window-level model, the benchmark labels it as `shared_seed_only`.

### `lightsb_path`

`lightsb_path` is the path-level Light Schrodinger Bridge variant added for this project. Instead of flattening full windows, it trains on one-step augmented transitions:

```text
[X_t, t] -> [X_{t+1}, t + dt]
```

The bridge dimension is:

```text
bridge_dim = n_features + 1
```

For stock data with one feature, `bridge_dim = 2`.

Generation is step-by-step:

1. start from provided `x0`, or sample training initial states if `x0` is not provided;
2. augment the current state with time;
3. sample the next augmented state through the one-step LightSB bridge;
4. keep the state coordinates and advance to the next time step.

Default key parameters:

```text
lightsb_path_n_potentials = 20
lightsb_path_epsilon = 1.0
lightsb_path_s_diagonal_init = 0.1
lightsb_path_epochs = 100
lightsb_path_lr = 0.001
lightsb_path_batch_size = 256
lightsb_path_sampling_batch_size = 512
lightsb_path_state_clip = 8.0
```

Because it supports supplied initial states, the benchmark labels it as `shared_x0`.

### `numba_sb`

`numba_sb` is a fast Markovian SB/Brownian bridge baseline. It does not train a neural network. During fitting it stores empirical initial and terminal states and estimates a scalar bridge noise level.

Default key parameter:

```text
numba_sb_sigma = 0.1
```

The fitted sigma may overwrite the configured initial value.

### `timegan`

`timegan` is a GRU-based TimeGAN-style baseline. It contains embedder, recovery, generator, supervisor, and discriminator networks.

Default key parameters:

```text
timegan_hidden_dim = 64
timegan_z_dim = 32
timegan_n_layers = 2
timegan_epochs = 50
timegan_lr = 0.001
timegan_batch_size = 128
timegan_normalization = standard
timegan_generator_steps = 2
timegan_clip_output = False
```

The current default uses standard normalization and a linear recovery output. This avoids forcing heavy-tailed or return-like data into a strict min-max interval. A min-max mode remains configurable when needed.

### `diffusion_ts`

`diffusion_ts` is a simplified DDPM-style time-series diffusion baseline. Its denoiser is named `DiffusionUNet`, but the current architecture is an MLP-style per-time-step denoiser rather than a convolutional U-Net.

Default key parameters:

```text
diffusion_hidden_dim = 128
diffusion_n_steps = 100
diffusion_epochs = 50
diffusion_lr = 0.001
diffusion_batch_size = 64
diffusion_beta_start = 0.0001
diffusion_beta_end = 0.02
```

Generation cost scales with `diffusion_n_steps`, because sampling iterates the reverse chain.

### `rnn`

`rnn` is an autoregressive recurrent baseline. It trains next-step prediction and generates by rolling the network forward. In the notebook benchmark it receives a real prefix and completes the remaining path.

Default key parameters:

```text
rnn_hidden_dim = 64
rnn_num_layers = 2
rnn_dropout = 0.1
rnn_epochs = 50
rnn_lr = 0.001
rnn_batch_size = 64
rnn_context_len = None
rnn_cell_type = lstm
```

The current notebook overrides `rnn_context_len = 20`.

### `transformer_ar`

`transformer_ar` is a causal autoregressive Transformer baseline. It uses `TransformerEncoderLayer` with a causal mask, so its behavior is decoder-only/autoregressive.

Default key parameters:

```text
transformer_ar_d_model = 64
transformer_ar_n_heads = 4
transformer_ar_n_layers = 2
transformer_ar_d_ff = 128
transformer_ar_dropout = 0.1
transformer_ar_epochs = 50
transformer_ar_lr = 0.001
transformer_ar_batch_size = 64
transformer_ar_context_len = None
transformer_ar_max_seq_len = 64
```

The current notebook overrides:

```text
transformer_ar_d_model = 128
transformer_ar_n_layers = 3
transformer_ar_d_ff = 256
transformer_ar_context_len = 20
```

For sequence lengths above 64, set `transformer_ar_max_seq_len` to at least the full window length before model construction.

## Parameter Count Caveats

Parameter counts are not directly comparable across all models:

- SBTS variants have trainable neural drift or jump modules, but also calibrated non-neural state such as jump statistics and local volatility surfaces.
- Window-level LightSB parameter count scales with `sequence_length * n_features`.
- Path-level LightSB parameter count scales with `n_features + 1`.
- Diffusion models may have moderate parameter count but high sampling cost because they iterate over reverse diffusion steps.
- NumbaSB has no neural parameters but stores empirical endpoint samples.

For final reporting, compare at least:

1. trainable neural or potential parameters;
2. calibrated non-neural state;
3. training wall time;
4. generation wall time;
5. generation protocol;
6. metric values on the same shared evaluation subset.
