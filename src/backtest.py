import numpy as np
import pandas as pd
from src.returns import calculate_log_daily_returns
from src.simulation import monte_carlo_cumlog_paths


def rolling_backtest(
        prices,
        window=756,
        horizon=63,
        step=21,
        n_sims=1000,
        seed=422):
    """
    Performing a rolling-window backtest for a Monte Carlo return forecast.

    For each rolling iteration:
        1) Fit (estimate mu, sigma) on a training window of historical
            log returns.
        2) Simulate the distribution of cumulative log return over the
            next 'horizon' days.
        3) Compare the realized cumulative log return to the simulated
            prediction interval (5th to 95th percentile), and record whether
            it falls inside the band.

    Parameters
    ----------
    prices: pd.Series
        Adjusted closing prices indexed by date.
    window: int
        Length of the training in trading days.(e.g., 756 ~ 3 years).
    horizon: int
        Forecast horizon in trading days. (e.g., 63 ~ 3 months).
    step: int
        Step size in trading days between rolling iterations (e.g., 21 ~ 1 month)
    n_sims: int
        Number of Monte Carlo simulation paths per iteration.
    seed: int
        Random seed to ensure reproducibility.
    
    Returns
    -------
    out: pd.DataFrame
        Row-per-iteration results including training/testing
        date ranges, realized cumulative log return, prediction
        interval bounds (p05, p95), whether realized is inside the band, and
        summary columns:
            - coverage: mean(inside_band) across all iterations
            - ave_width: mean(band_width) across all iterations
    
    Notes
    -----
    This backtest evaluates calibration of the simulated return distribution, not point
    forecast accuracy. A well-calibrated 90% prediction interval should have coverage
    close to 90% over many rolling iterations.
    """
    r = calculate_log_daily_returns(prices)
    r = r.dropna()
    idx = r.index
    rows = []
    for start in range(0, len(r) - window - horizon + 1, step):
        # Train on 3 years, e.g., 2020-01-01 to 2023-01-01
        train = r.iloc[start:start + window]
        # Test on next 3 month, e.g., 2023-01-01 to 2023-04-01
        test = r.iloc[start + window: start + window + horizon]
        # Simulate distribution of cumulative return for the horizon
        # sims is in (horizon, n_sims) shape containing cumulative log returns
        # XXX   Path 1 | Path 2 | Path 3 | ~ | Path 1000
        # Day1   0.01
        # Day2   0.02
        # Day3   0.003
        # ~~~
        # Day63  0.05
        sims = monte_carlo_cumlog_paths(
            train,
            horizon,
            n_sims=n_sims,
            seed=seed)
        # Last row in sims is the predicted cumsum log returns after 63 days
        # We can compare it with realized cumsum log returns
        sim_final = sims[-1, :]
        realized = test.sum()
        p05 = np.percentile(sim_final, 5)
        p95 = np.percentile(sim_final, 95)

        rows.append({
            "train_start": idx[start],
            "train_end": idx[start + window - 1],
            "test_start": idx[start + window],
            "test_end": idx[start + window + horizon - 1],
            "horizon_days": horizon,
            "realized_cumlog": realized,
            "p05": p05,
            "p95": p95,
            "inside_band": (p05 <= realized <= p95),
            "band_width": (p95 - p05)
        })
    out = pd.DataFrame(rows)
    out["coverage"] = out["inside_band"].mean()
    out["ave_width"] = out["band_width"].mean()
    return out
