# SBTS Advanced: Jump-Diffusion Schrodinger Bridges for Time Series Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ecole Polytechnique](https://img.shields.io/badge/Institution-Ecole%20Polytechnique-red)](https://www.polytechnique.edu/)

> Projet Scientifique Collectif (PSC)  
> Ecole Polytechnique, IP Paris  
> Academic year: 2025-2026

## Authors

This project was developed by Lizhan Hong, Sheng Wan, Haoyu Luo, Cyprien Kutek, and Dimitri Krawczyk under the supervision of Prof. Huyen Pham.

## Overview

SBTS Advanced is a research codebase for generating financial time series with Schrodinger Bridge methods, jump-diffusion dynamics, and neural sequence baselines. The project focuses on market-like properties such as heavy tails, jump behavior, autocorrelation structure, volatility clustering, and realistic terminal distributions.

The active workflow has two entry points:

- `main.py`: command-line experiment runner for model training, generation, evaluation, and plot export.
- `main_pipeline.ipynb`: the current benchmark notebook, dataset builder, diagnostic dashboard, and result exporter.

The current benchmark configuration is centered on QQQ daily close-price log returns and compares several Schrodinger Bridge, jump-diffusion, GAN, diffusion, and fast Markovian baselines under a shared evaluation protocol.

## Model Registry

All active models are constructed through `models/factory.py` and follow the shared `TimeSeriesGenerator` interface:

```python
model.fit(data, time_grid=None, verbose=True)
samples = model.generate(n_samples, n_steps=None, x0=None)
```

| Key | Model |
|---|---|
| `jd_sbts` | Base Jump-Diffusion Schrodinger Bridge Time Series model |
| `jd_sbts_f` | JD-SBTS with feedback for jump-volatility interaction |
| `jd_sbts_neural` | JD-SBTS with neural jump detection/intensity modeling |
| `jd_sbts_f_neural` | JD-SBTS with both feedback and neural jumps |
| `lightsb` | Window-level Light Schrodinger Bridge |
| `lightsb_path` | Path-level Light Schrodinger Bridge |
| `numba_sb` | Numba-accelerated Markovian Schrodinger Bridge baseline |
| `timegan` | GRU-based TimeGAN baseline |
| `diffusion_ts` | DDPM-style diffusion baseline for time series |
| `rnn` | Autoregressive RNN baseline |
| `transformer_ar` | Causal autoregressive Transformer baseline |

Aliases such as `sbts`, `light_sb`, `diffusion`, `rnn_baseline`, and `transformer` are normalized by the factory and CLI.

## Current Benchmark Snapshot

The latest saved run is in `results_QQQ/` and was produced from `main_pipeline.ipynb`.

| Setting | Value |
|---|---|
| Dataset | Yahoo Finance stock data |
| Ticker | `QQQ` |
| Date range | `2006-01-01` to `2026-03-31` |
| Feature | `Close` |
| Normalization | `log_return` |
| Processed window shape | `(5068, 24, 1)` |
| Window length | `24` |
| Stride | `1` |
| Evaluation cap | `512` windows |
| Models in latest run | `jd_sbts`, `jd_sbts_f`, `jd_sbts_neural`, `lightsb`, `timegan`, `diffusion_ts`, `numba_sb` |

The notebook uses a fair-comparison protocol:

- all models train on the same windows;
- all metrics use the same shared evaluation subset;
- conditioned models receive shared initial states when supported;
- unconditional models share the evaluation subset and reset RNG seed;
- model failures are recorded in `failures.json` instead of silently ignored.

### Latest QQQ Ranking

Lower values are better for `main_rank_mean`, Wasserstein distance, ACF MSE, predictive score, and the secondary diagnostic scores.

| Rank | Model | Protocol | Main rank mean | Wasserstein | ACF MSE | Predictive |
|---:|---|---|---:|---:|---:|---:|
| 1 | `lightsb` | shared seed only | 1.667 | 0.002362 | 0.000110 | 0.009248 |
| 2 | `diffusion_ts` | shared seed only | 2.667 | 0.001524 | 0.000124 | 0.009580 |
| 3 | `numba_sb` | shared seed only | 4.333 | 0.029273 | 0.000160 | 0.009300 |
| 4 | `jd_sbts` | shared x0 | 4.333 | 0.006975 | 0.002000 | 0.009482 |
| 5 | `timegan` | shared seed only | 4.667 | 0.008002 | 0.005408 | 0.009240 |
| 6 | `jd_sbts_neural` | shared x0 | 5.000 | 0.006820 | 0.002041 | 0.010904 |
| 7 | `jd_sbts_f` | shared x0 | 5.333 | 0.006936 | 0.002398 | 0.009606 |

The full tables are saved under:

- `results_QQQ/tables/df_metrics.csv`
- `results_QQQ/tables/comparison_df.csv`
- `results_QQQ/tables/ranking_summary.csv`
- `results_QQQ/metrics_results.json`

## Repository Structure

```text
sbts_advanced/
├── main.py                         # CLI experiment runner
├── main_pipeline.ipynb             # Main notebook benchmark and result exporter
├── experiment_comparison.ipynb     # Additional experiment notebook
├── turing_test.ipynb               # Turing-test diagnostics
├── config/
│   └── config.json                 # Default JSON config for CLI runs
├── data/
│   └── loaders.py                  # Yahoo Finance, synthetic data, sliding windows
├── models/
│   ├── base.py                     # Shared generator interface
│   ├── factory.py                  # Active model registry and defaults
│   ├── sbts_variants.py            # JD-SBTS model family
│   ├── lightsb.py                  # LightSB, PathLightSB, NumbaSB
│   ├── timegan_baseline.py         # TimeGAN baseline
│   ├── diffusion_ts_baseline.py    # Diffusion baseline
│   ├── rnn_baseline.py             # RNN baseline
│   └── transformer_ar_baseline.py  # Autoregressive Transformer baseline
├── core/                           # LightSB / SB numerical components
├── modules/                        # Drift, volatility, jump, solver components
├── metrics/                        # Statistical, discriminative, predictive metrics
├── visualization/                  # Plotting utilities
├── utils/                          # Experiment manager and compatibility helpers
├── tests/                          # Calibration and feedback simulation tests
├── notebook_outputs/               # Cached notebook datasets
└── results_QQQ/                    # Latest exported QQQ benchmark artifacts
```

## Installation

Python 3.10+ is recommended.

```bash
git clone https://github.com/ApolloHong/sbts_advanced.git
cd sbts_advanced
pip install -r requirements.txt
```

Core dependencies include `numpy`, `scipy`, `torch`, `scikit-learn`, `pandas`, `numba`, `matplotlib`, `seaborn`, `yfinance`, `optuna`, and `einops`.

The default config uses `device = "cuda"`. If CUDA is unavailable, change the config or notebook setting to CPU before running experiments.

## CLI Usage

List registered models:

```bash
python main.py --list-models
```

Run the default experiment from `main.py`:

```bash
python main.py
```

Run a single model:

```bash
python main.py --model jd_sbts_f
python main.py --model lightsb
python main.py --model diffusion_ts
python main.py --model rnn
python main.py --model transformer_ar
```

Run the full registry benchmark:

```bash
python main.py --benchmark
```

Run on synthetic data instead of ETF data:

```bash
python main.py --synthetic
```

Use a custom config:

```bash
python main.py --config config/config.json --output experiments
```

## Notebook Workflow

Use `main_pipeline.ipynb` for the current paper-style benchmark workflow:

1. Configure `BENCHMARK_DATASET`, `STOCK_TICKER`, `MODELS_TO_RUN`, and `TRAINING_OVERRIDES`.
2. Build or load cached datasets under `notebook_outputs/benchmark_main_pipeline/datasets/`.
3. Train the selected models.
4. Generate samples with shared conditioning where supported.
5. Compute statistical, discriminative, predictive, Turing-test, stylized-fact, jump, and timing diagnostics.
6. Inspect plots, including combined Gaussian QQ plots and per-model terminal distributions.
7. Export arrays, tables, JSON summaries, pickled models, and figures to `results_QQQ/`.

The notebook currently sets:

```python
BENCHMARK_DATASET = "stock"
STOCK_TICKER = "QQQ"
MODELS_TO_RUN = [
    "jd_sbts",
    "jd_sbts_f",
    "jd_sbts_neural",
    "lightsb",
    "timegan",
    "diffusion_ts",
    "numba_sb",
]
```

## Metrics

The active benchmark reports:

- Wasserstein distance
- ACF MSE
- GRU discriminative score
- CNN discriminative score
- predictive score
- Turing-test accuracy, score, and AUC
- stylized facts: volatility clustering, fat tails, leverage effect
- jump frequency and magnitude diagnostics
- training and generation times

`correlation_distance`, `overall_rank_score`, and `secondary_rank_mean` are no longer part of the current notebook ranking/report display.

## Result Artifacts

The latest exported result directory contains:

```text
results_QQQ/
├── arrays/                 # train/eval windows and split arrays
├── generated/              # generated samples per model
├── stress/                 # feedback stress trajectories where available
├── tables/                 # CSV and JSON metric/ranking tables
├── models/                 # best-effort model pickles
├── figures/                # real-vs-generated comparison figures
├── metrics_results.json
├── training_times.json
├── generation_times.json
├── generation_protocols.json
├── fairness_notes.json
├── failures.json
└── run_snapshot.json
```

## Notes

- Yahoo Finance availability can change over time, especially for intraday data. Daily QQQ data is the current stable benchmark setting.
- For `transformer_ar`, the pipeline expands `transformer_ar_max_seq_len` when needed so it can handle the benchmark window length.
- `main_old.py` exists as a compatibility shim. New experiments should use `main.py` or `main_pipeline.ipynb`.

## License

This repository is released under the MIT License.
