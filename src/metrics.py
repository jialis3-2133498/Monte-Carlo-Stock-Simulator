import numpy as np
import pandas as pd
from typing import Tuple


def compute_var_es(
        values: np.ndarray,
        alpha: float = 0.05) -> Tuple[float, float]:
    """
    """
    values = np.asarray(values).reshape(-1)
    var = float(np.quantile(values, alpha))
    tail = values[values <= var]
    if tail.size > 0:
        es = float(tail.mean())
    else:
        es = float("nan")
    return var, es


def compute_payoffs(
        end_prices: np.ndarray,
        S0: float) -> Tuple[np.ndarray, np.ndarray]:
    end_prices = np.asarray(end_prices).reshape(-1)
    log_payoff = np.log(end_prices / S0)
    simple_return = end_prices / S0 - 1.0
    return log_payoff, simple_return


def summarize_horizon_metrics(
        end_prices: np.ndarray,
        S0: float,
        alpha: float = 0.05) -> pd.DataFrame:
    end_prices = np.asarray(end_prices).reshape(-1)
    log_payoff, simple_return = compute_payoffs(end_prices, S0)
    var_log, es_log = compute_var_es(log_payoff, alpha=alpha)
    var_simple, es_simple = compute_var_es(simple_return, alpha=alpha)
    metrics = {
        "S0": float(S0),
        "mean_end_price": float(end_prices.mean()),
        "median_end_price": float(np.median(end_prices)),
        "p05_end_price": float(np.percentile(end_prices, 5)),
        "p95_end_price": float(np.percentile(end_prices, 95)),
        "prob_end_price_below_S0": float((end_prices < S0).mean()),
        "prob_loss_log": float((log_payoff < 0).mean()),
        "prob_loss_simple": float((simple_return < 0).mean()),
        "alpha": float(alpha),
        "VaR_log": float(var_log),
        "ES_log": float(es_log),
        "VaR_simple": float(var_simple),
        "ES_simple": float(es_simple),
    }
    return pd.DataFrame([metrics])
