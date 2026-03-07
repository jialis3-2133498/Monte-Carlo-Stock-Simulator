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
    n_sims : int, default=1000
        Number of Monte Carlo simulation paths.
    seed : int, default=422
        Random seed for reproducibility.

    Returns
    -------
    cum_log_returns : np.ndarray
        shape (horizon, n_sims)
        Rows -> time steps
        Cols -> simulation paths

    Notes
    -----
    Assumes daily log returns follow a normal distribution:
        r_t ~ N(mu, sigma)
    """
    # Initialize a random number generator object
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


def monte_carlo_price_paths(
        train_prices,
        train_returns,
        horizon,
        n_sims=1000,
        seed=422):
    """
    Simulate Monte Carlo paths for future stock prices

    Parameters
    ---------
    train_prices : pd.Series
        Historical price series used to obtain the last observed price S_0.
    train_returns : pd.Series
        Historical log returns used to estimate model parameters.
    horizon : int
        Number of trading days to simulate into the future.
    n_sims : int, default=1000
        Number of Monte Carlo simulation paths.
    seed : int, default=422
        Random seed for reproducibility.

    Returns
    -------
    price_paths : nd.ndarray
        Simulated price paths with shape (horizon + 1, n_sims).
        Rows -> time steps (t=0,....,horizon)
        Cols -> simulation paths
        The first row contains the last observed price S_0.

    Notes
    -----
    Price dynamics: S_t = S_0 * exp(cumulative_log_return)
    """
    S_0 = train_prices.iloc[-1]
    cum_log_returns = monte_carlo_cumlog_paths(
        train_returns=train_returns,
        horizon=horizon,
        n_sims=n_sims,
        seed=seed
    )
    price_paths = np.empty((horizon+1, n_sims))
    price_paths[0, :] = S_0
    price_paths[1:, :] = S_0 * np.exp(cum_log_returns)
    return price_paths
