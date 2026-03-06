<p align="center">
  <b>Monte Carlo Simulation | Rolling Backtest | Risk Metrics</b>
</p>

# Monte-Carlo-Stock-Simulator
A reproducible Python project that builds and validates a Monte Carlo-based probabilistic forecasting system for equity prices using Geometric Brownian Motion (GBM).
The project estimates the future distribution of stock prices over a 5-year horizon and evaluates model reliability via rolling-window backtesting and tail risk metrics.
## Key Features
* Vectorized Monte Carlo simulation of GBM price paths
* Probabilistic forecasting of future equity price distributions
* Rolling-window backtesting for model calibration
* Tail risk analysis including Value-at-Risk (VaR) and Expected Shortfall (ES)
* Fully reproducible experiments with fixed random seed
## Problem Statement
The goal of this project is to:
* Estimate the distribution of future stock prices
* Quantify downside risk (VaR, Expected Shortfall, probability of loss)
* Validate forecast reliability using rolling backtesting
* Analyze model calibration
The target asset used in this study is Amazon (AMZN).
## Data
* Source: `yfinance`
* Estimation window: 01/01/2020 to 01/01/2025
* Backtest window: 01/01/2020 to 12/31/2026
* Trading days per year: 252

## Model Assumptions
Daily log returns are modeled as:

$$
r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)
$$

We assume: 

$$
r_t \sim \mathcal{N}(\mu, \sigma^2)(i.i.d.).
$$

Under the Geometric Brownian Motion (GBM) framework, this implies that future prices are lognormally distributed. 

Parameter estimation is performed using Maximum Likelihood Estimation (MLE):
* $\mu$ = sample mean of log returns
* $\sigma$ = sample standard deviation of log returns

Time step: 
$\Delta t$ is $1/252$

## Simulation Setup
* Forecast horizon: 5 years (1260 trading days)
* Simulation paths: 1000
* Random seed: 422
* Monte Carlo engine implemented with vectorized Numpy operations
* Price paths simulated under a log-return GBM model

## Rolling Backtest
A rolling-window backtest is implemented to evaluate model calibration:
* Rolling training window: 756 trading days(~3 years)
* Forecast horizon: 63 trading days(~3 months)
* Step size: 21 trading days
* 90% prediction interval (5%-95%)

Evaluation metrics:
* Empirical coverage rate
* Average interval width
This allows us to assess whether realized returns fall within the model's predicted confidence bands.

## Risk Metrics
At the 5-year horizon, the project computes:
* Expected end price
* Median end price
* 5th and 95th percentiles
* Probability of loss
* Value-at-Risk (VaR)
* Expected Shortfall (ES)
Both log-payoff and simple-return risk measures are evaluated.

## Outputs
### Running:
```bash
python main.py
```
Produces:
```bash
outputs/
  rolling_backtest.csv
  backtest_summary.csv
  simulation_metrics.csv
  returns_descriptive_stats.csv
  future_price_paths.png
  log_daily_returns_hist.png
  log_daily_returns_qq.png
  horizon_end_prices_hist.png
  horizon_logpayoff_scale_hist.png
```
All experiments are fully reproducible via a fixed random seed.

## Project Structure
```bash
src/
  data.py
  returns.py
  simulation.py
  backtest.py
  metrics.py
  plots.py
main.py
outputs/
```
* `src/` contains modular, reusable components
* `main.py` orchestrates the full pipeline
* `outputs/` stores generated artifacts

## Requirements
* Python >= 3.10
* numpy
* pandas
* scipy
* matplotlib
* yfinance

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Reproducible](https://img.shields.io/badge/Experiments-Reproducible-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
