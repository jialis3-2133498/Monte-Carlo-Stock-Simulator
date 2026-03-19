# Monte Carlo Stock Price Simulation with Rolling Backtest

## Project Summary

- **Model:** Geometric Brownian Motion (GBM), Monte Carlo Simulation, Maximum Likelihood Estimation
- **Forecast Horizon:** 5 years (1260 trading days)
- **Simulation Paths:** 1000
- **Backtest:** Rolling window (756-day training, 63-day testing, 21-day step)
- **Risk Metrics:** Value-at-Risk (VaR), Expected Shortfall (ES)

## 1. Introduction
In financial markets, stock prices are influenced by many factors, including macroeconomic policies, geopolitical events, and firm-level performance. Modeling the future behavior of stock prices is therefore challenging due to the complexity and uncertainty of these influences.

Traditional approaches, such as linear regression, can help analyze relationships between variables, but they may not fully capture the stochastic dynamics and inherent uncertainty of asset prices.

To address this, the Geometric Brownian Motion (GBM) framework provides a mathematically robust approach, assuming that asset prices follow a lognormal distribution. This preserves critical real-world properties such as non-negativity and multiplicative compounding. Under this framework, Monte Carlo simulation can be used to generate multiple possible future price paths and analyze the resulting distribution of returns.

In this project, we implement a Monte Carlo simulation to model the future price behavior of Amazon (AMZN) stock and evaluate its risk characteristics, including Value-at-Risk (VaR) and Expected Shortfall (ES). 

## 2. Data
Our target asset in this study is Amazon (AMZN). Historical stock price data are obtained using the `yfinance` API, covering the period from January 1, 2020 to the most recent available trading date. In this project, we use the adjusted closing price, which accounts for corporate actions such as dividends and stock splits and therefore better reflects the true return of the asset.

Based on the adjusted closing prices, we compute daily log returns, which are commonly used in financial modeling due to their desirable statistical properties, including time additivity.

Daily log returns are computed as:

$$r_t = \ln\left(\frac{S_t}{S_{t-1}}\right)$$

where $S_t$ and $S_{t-1}$ denote the adjusted closing prices at time $t$ and ${t-1}$.

<p align="center">
  <img src="outputs/returns_descriptive_stats.png" width="700">
</p>

*Table 1: Descriptive statistics of daily log returns.*

The mean daily log return is approximately 0.00067, while the standard deviation is about 0.02265, indicating that short-term price movements are primarily driven by volatility rather than average drift. These estimated parameters are later used in the Monte Carlo simulation. 


<p align="center">
  <img src="outputs/log_daily_returns_hist.png" width="700">
</p>

*Figure 1: Distribution of Amazon Daily Log Returns*

The histogram indicates that the distribution of daily log returns is approximately symmetric and centered around zero, which is broadly consistent with the normality assumption used in the Geometric Brownian Motion framework.

The distribution also exhibits several extreme observations in the tails, which reflect the presence of occasional large market movements.

<p align="center">
  <img src="outputs/log_daily_returns_qq.png" width="700">
</p>

*Figure 2: Q-Q plot of Amazon daily log returns compared with a normal distribution*

The Q-Q plot suggests that the distribution of daily log returns is approximately normal in the central region, as most observations lie close to the 45-degree reference line. However, noticeable deviations appear in the extreme quantiles, indicating the presence of fat tails and occasional extreme return events. Such behavior is commonly observed in financial return series.

Although financial returns often exhibit fat tails, the normality assumption provides a reasonable first-order approximation for modeling price dynamics under the Geometric Brownian Motion framework.

## 3. Methodology
In this project, the Geometric Brownian Motion (GBM) framework is used, and the stochastic continuous-time model is:

$$dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$$

where $S_t$ denotes the asset price at time $t$, $\mu$ represents the drift (expected return), $\sigma$ denotes volatility, and $W_t$ is a standard Brownian motion.

The discrete solution of GBM is:

$$S_{t+\Delta t} = S_t \exp\left((\mu - \tfrac{1}{2}\sigma^2)\Delta t + \sigma \sqrt{\Delta t} Z_t\right)$$

where 

$$Z_t \sim N(0,1)$$

The discrete-time solution of the GBM process is used to simulate stock price paths. In this equation, $Z_t$ represents a standard normal random variable. At each time step, a random shock is drawn from the normal distribution and applied to the price dynamics. By repeatedly sampling these shocks, the Monte Carlo simulation generates multiple possible future price paths.

The parameters $\mu$ and $\sigma$ are estimated from historical daily log returns and are used as inputs in the Monte Carlo simulation. 

## 4. Monte Carlo Simulation

Using the estimated parameters from historical data, $\mu$ and $\sigma$, we generate possible future price paths for Amazon stock. In each simulation, random shocks are drawn from a standard normal distribution to represent unpredictable market movements.

For each simulation path, cumulative log returns are calculated by summing the simulated daily log returns over time. The cumulative returns are then converted into simulated future prices by applying the exponential transformation to the initial stock price.

By repeating this process for many independent simulation paths, the Monte Carlo approach produces a distribution of possible future stock prices. This distribution allows us to analyze potential future outcomes and evaluate risk metrics such as Value-at-Risk (VaR) and Expected Shortfall (ES).

A total of 1000 simulation paths are generated over a horizon of 1260 trading days, corresponding to approximately five years of future price evolution.

<p align="center">
  <img src="outputs/future_price_paths.png" width="700">
</p>

*Figure 3: Monte Carlo simulated future price distribution for Amazon stock under the GBM model. The blue and orange lines represent the mean and median simulated price paths. The shaded region represents the 5-95% prediction interval across simulation paths. The black line shows the observed stock price trajectory from 2025-2026.*

The observed price trajectory from 2025 to 2026 is overlaid on the simulated distribution for comparison. Although the realized path is only partially observed, it remains within the simulated 5-95% prediction interval. This suggests that the model captures a plausible range of potential future outcomes under the estimated parameters. However, the observed path fluctuates differently from the simulated central tendency, reflecting the unpredictable nature of real market dynamics.

## 5. Risk Distribution and Risk Metrics
<p align="center">
  <img src="outputs/horizon_end_prices_hist.png" width="700">
</p>
At the end of the simulation horizon, each Monte Carlo path produces a possible terminal stock price and corresponding return. Taken together, these simulated outcomes from an empirical distribution of future prices and losses. This distribution provides a convenient basis for summarizing downside risk, including the probability of loss, Value-at-Risk (VaR), and Expected Shortfall (ES).
To evaluate downside risk, we compute both return-based and price-based summaries. In particular, we examine the probability that the ending price falls below the initial price, as well as tail-risk measures derived from the simulated return distribution.

## 6. Rolling Backtest
<p align="center">
  <img src="outputs/backtest_simulated_returns_vs_actual_returns.png" width="700">
</p>

## 7. Limitation
<p align="center">
  <img src="outputs/rolling_volatility.png" width="700">
</p>


## 8. Conclusion







