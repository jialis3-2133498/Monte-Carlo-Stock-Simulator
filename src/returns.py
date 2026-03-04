import numpy as np


def calculate_log_daily_returns(prices):
    """
    Calculate log daily returns of adjusted prices.

    Paramters
    ---------
    prices: pd.Series
        Time-indexed Series of adjusted prices.

    Returns
    -------
    log_daily_returns: pd.Series
        Daily log returns defined as:
            r_t = ln(S_t / S_{t-1})

    Notes
    -----
    Log returns are used because they transform the multiplicative
    price process into an additive return process, which aligns with
    the Geometric Brownian Motion (GBM) assumption that log returns
    are i.i.d. normally distributed.
    The first observation is dropped due to the shift operation.
    """
    # We use .shift(1) to create S_{t-1}.
    log_daily_returns = np.log(prices / prices.shift(1)).dropna()
    return log_daily_returns
