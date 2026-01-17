import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy
from scipy import stats

# Amazon ticker is AMZN
amzn_df = yf.Ticker("AMZN")
amzn_df = amzn_df.history(
    start='2020-01-01',
    end='2025-01-01',
    interval='1d')
prices = amzn_df['Close'].dropna()
# use tz_localize to make the index more clean and tidy
prices.index = prices.index.tz_localize(None)
# use series.shift() to create values that are a day before
# We use log daily returns
log_daily_returns = np.log(prices / prices.shift(1)).dropna()

# Descriptive Anaylsis
r = log_daily_returns
descriptive_data = r.describe()
table = descriptive_data.to_frame(name='Log Return')

# Plot a histgram to observe the log daily returns distribution
fig, ax = plt.subplots()
ax.hist(
    log_daily_returns,
    bins='fd',
    linewidth=0.5,
    edgecolor="white"
    )
ax.set_title("Amazon's Stock Log Daily Returns Distribution")
ax.set_xlabel("Log Daily Returns")
ax.set_ylabel("Counts")
plt.show()

# Plot a QQ plot
ax2 = stats.probplot(
    log_daily_returns,
    dist='norm',
    plot=plt
    )
plt.title('Q-Q Plot of Daily Log Returns vs Normal')
plt.xlabel("Theoretical Quantiles (Normal Distribution)")
plt.ylabel("Sample Quantiles (Observed Data)")
plt.show()

# Daily log returns are approximately normal in the center, but the
# QQ plot shows heavier tails than a Normal distribution,
# especially in extremes.

# MLE to get estimated mean and standard deviation
mu_hat = log_daily_returns.mean()
sigma_hat = log_daily_returns.std(ddof=0)

# Monte Carlo Simulation for the future price paths
T = 1260  # Simulate the price in the future 1260 days
N = 1000  # For each day, we will simulate 1000 times/paths
S_0 = prices.iloc[-1]  # The last close price in our dataset
Z = stats.norm.rvs(size=(N, T), random_state=422)  # standard shock for adding volatility
returns = mu_hat + sigma_hat * Z 
cum_r = np.cumsum(returns, axis=1)
S_T = np.empty((N, T+1))
S_T[:, 0] = S_0
S_T[:, 1:] = S_0 * np.exp(cum_r)  # Future 5 years prices, 1000 paths for each day

# Analyze the future prices matrix
mean_prices = S_T.mean(axis=0)  # Get the means of the matrix in row-way
median_prices = np.median(S_T, axis=0)
p05 = np.percentile(S_T, 5, axis=0)
p95 = np.percentile(S_T, 95, axis=0)
prob_price_below_S0 = (S_T < S_0).mean(axis=0)  # (S_T<S_0) is a boolean matrix

# mean_prices vs. median_prices
days = np.arange(len(mean_prices))
fig3, ax3 = plt.subplots()
ax3.plot(days, mean_prices, label='mean')
ax3.plot(days, median_prices, label='median')
ax3.fill_between(days, p05, p95, alpha=0.2, label="5-95% band")
ax3.set_yscale("log")
ax3.set_xlabel("Day")
ax3.set_ylabel("Simulated price (log-scale)")
ax3.axhline(S_0, linestyle="--", linewidth=1, label="Initial price(S0)")
ax3.legend()
plt.show()

# End-of-horizon metrics
end_prices = S_T[:, -1] # Last day simulated prices
expected_end_price = end_prices.mean()
median_end_price = np.median(end_prices)
prob_price_below_S0_end_day = (end_prices < S_0).mean()

print(median_end_price, expected_end_price, p05[-1], p95[-1])






