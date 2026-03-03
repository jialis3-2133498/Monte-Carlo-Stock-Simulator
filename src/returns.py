import numpy as np


def calculate_log_daily_returns(prices):
    log_daily_returns = np.log(prices / prices.shift(1)).dropna()
    return log_daily_returns
