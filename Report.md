# Monte Carlo Stock Price Simulation with Rolling Backtest

## Project Summary

- **Model:** Geometric Brownian Motion (GBM), Monte Carlo Simulation, Maximum Likelihood Estimation
- **Forecast Horizon:** 5 years (1260 trading days)
- **Simulation Paths:** 1000
- **Backtest:** Rolling window (756-day training, 63-day horizon)
- **Risk Metrics:** Value-at-Risk (VaR), Expected Shortfall (ES)

## 1. Introduction
In financial markets, stock prices are influenced by many factors, including macroeconomic policies, geopolitical events, and firm-level performance. Modeling the future behavior of stock prices is therefore challenging due to the complexity and uncertainty of these influences.

Traditional approaches, such as linear regression, can help analyze relationships between variables, but they may not fully capture the stochastic dynamics and inherent uncertainty of asset prices.

To address this, the Geometric Brownian Motion (GBM) framework provides a mathematically robust approach, assuming that asset prices follow a lognormal distribution. This preserves critical real-world properties such as non-negativity and multiplicative compounding. Under this framework, Monte Carlo simulation can be used to generate multiple possible future price paths and analyze the resulting distribution of returns.

In this project, we implement a Monte Carlo simulation to model the future price behavior of Amazon (AMZN) stock and evaluate its risk characteristics, including Value-at-Risk (VaR) and Expected Shortfall (ES). 

## 2. Data
Our target asset in this study is Amazon (AMZN). Historical stock price data are obtained using the `yfinance` API, covering the period from January 1, 2020 to the most recent available date. In this project, we use the adjusted closing price, which accounts for corporate actions such as dividends and stock splits and therefore better reflects the true return of the asset.

Based on the adjusted closing prices, we compute daily log returns, which are commonly used in financial modeling due to their desirable statistical properties, including time additivity.

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

