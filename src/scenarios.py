import numpy as np
import pandas as pd

from src.simulation import monte_carlo_price_paths
from src.metrics import compute_payoffs, compute_var_es


SCENARIOS = {
    "Base Case": 1.0,
    "High Volatility": 1.5,
    "Stress Case": 2.0,
}


def run_scenario_analysis(
    train_prices,
    train_returns,
    horizon=1260,
    n_sims=1000,
    seed=422,
    alpha=0.05
):
    """
    Compare Monte Carlo risk metrics under different
    volatility assumptions.
    """
    S0 = float(train_prices.iloc[-1])
    historical_sigma = float(train_returns.std(ddof=0))

    results = []

    for scenario_name, multiplier in SCENARIOS.items():

        price_paths = monte_carlo_price_paths(
            train_prices=train_prices,
            train_returns=train_returns,
            horizon=horizon,
            n_sims=n_sims,
            seed=seed,
            volatility_multiplier=multiplier
        )

        end_prices = price_paths[-1, :]
        _, simple_returns = compute_payoffs(end_prices, S0)

        var_simple, es_simple = compute_var_es(
            simple_returns,
            alpha=alpha
        )

        results.append({
            "scenario": scenario_name,
            "volatility_multiplier": multiplier,
            "daily_volatility": historical_sigma * multiplier,
            "horizon": horizon,
            "n_sims": n_sims,
            "seed": seed,
            "alpha": alpha,
            "mean_end_price": float(np.mean(end_prices)),
            "median_end_price": float(np.median(end_prices)),
            "p05_end_price": float(np.quantile(end_prices, 0.05)),
            "p95_end_price": float(np.quantile(end_prices, 0.95)),
            "probability_of_loss": float(np.mean(end_prices < S0)),
            "var_simple": var_simple,
            "es_simple": es_simple,
        })

    return pd.DataFrame(results)