# Monte Carlo Stock Price Simulation with Rolling Backtest

## Project Summary

- **Model:** Geometric Brownian Motion (GBM)
- **Forecast Horizon:** 5 years (1260 trading days)
- **Simulation Paths:** 1000
- **Backtest:** Rolling window (756-day training, 63-day horizon)
- **Risk Metrics:** Value-at-Risk (VaR), Expected Shortfall (ES)

## 1. Introduction


## 2. Data

## 3. Model

## 4. Monte Carlo Simulation

<p align="center">
  <img src="outputs/future_price_paths.png" width="700">
</p>

*Figure 1: Monte Carlo simulated price paths for AMZN over a five-year horizon. The shaded region represents the 5–95% prediction interval.*

## 5. Risk Distribution
<p align="center">
  <img src="outputs/horizon_end_prices_hist.png" width="700">
</p>

## 6. Rolling Backtest
<p align="center">
  <img src="outputs/backtest_simulated_returns_vs_actual_returns.png" width="700">
</p>

## 7. Model Diagnostics
<p align="center">
  <img src="outputs/rolling_volatility.png" width="700">
</p>


## 8. Risk Metrics

## 9. Limitations

## 10. Conclusion
