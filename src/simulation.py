import numpy as np
import pandas as pd


def monte_carlo_cumlog_paths(
        train_returns,
        horizon,
        n_sims = 1000,
        seed = 422):
    """
    """
    rng = np.random.default_rng(seed)
    mu = train_returns.mean()
    sigma = train_returns.std(ddof=0)   
    shocks = rng.normal(loc=mu, scale=sigma, size=(horizon, n_sims))
    cum_log_returns = np.cumsum(shocks, axis=0)
    return cum_log_returns