import numpy as np
import pandas as pd
from src.returns import calculate_log_daily_returns
from src.simulation import monte_carlo_cumlog_paths


def rolling_backtest(
        prices, window=756, horizon=63, step=21, n_sims=1000, seed=422):
    """
    """
    r = calculate_log_daily_returns(prices)
    r = r.dropna()
    idx = r.index
    rows = []
    for start in range(0, len(r) - window - horizon + 1, step):
        train = r.iloc[start:start + window]
        test = r.iloc[start + window: start + window + horizon]

        sims = monte_carlo_cumlog_paths(
            train,
            horizon,
            n_sims=n_sims,
            seed=seed)
        sim_final = sims[-1, :]
        realized = test.sum()
        p05 = np.percentile(sim_final, 5)
        p95 = np.percentile(sim_final, 95)

        rows.append({
            "train_start": idx[start],
            "train_end": idx[start + window - 1],
            "test_start" : idx[start + window],
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