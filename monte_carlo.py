import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.stats as sp
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
log_daily_returns = np.log(prices / prices.shift(1)).dropna() 

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
# plt.show()

# Plot a QQ plot 
ax2 = stats.probplot(
    log_daily_returns,
    dist='norm',
    plot=plt
    )
plt.title('Q-Q Plot of Daily Log Returns vs Normal')
plt.xlabel("Theoretical Quantiles (Normal Distribution)")
plt.ylabel("Sample Quantiles (Observed Data)")
# plt.show()

# Daily log returns are approximately normal in the center, but the
# QQ plot shows heavier tails than a Normal distribution,
# especially in extremes.

# MLE to get estimated mean and standard deviation
mu_hat = log_daily_returns.mean()
sigma_hat = log_daily_returns.std()
