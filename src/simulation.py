import numpy as np


def monte_carlo_cumlog_paths(
        train_returns,
        horizon,
        n_sims=1000,
        seed=422):
    """
    Simulate Monte Carlo paths of cumulative log returns.

    Parameters
    ---------
    train_returns : pd.Series
        Historical log returns used to estimate model parameters.
    horizon : int
        Number of days to simulate into the future.
    n_sims : int
        Number of Monte Carlo simulation paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    cum_log_returns : np.ndarray
        Simulated cumulative log returns with shape (horizon, n_sims).
        Each column represents one simulated return path.

    Notes
    -----
    Assumes daily log returns follow a normal distribution:
        r_t ~ N(mu, sigma)
    """
    # Create a random number generator object
    rng = np.random.default_rng(seed)
    # Estimate mean and volatility from training returns (MLE)
    mu = train_returns.mean()
    sigma = train_returns.std(ddof=0)
    # Generate simulated daily log returns
    simulated_daily_returns = rng.normal(
        loc=mu,
        scale=sigma,
        size=(horizon, n_sims))
    # Convert daily returns to cumulative log returns
    cum_log_returns = np.cumsum(simulated_daily_returns, axis=0)
    return cum_log_returns
