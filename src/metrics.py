import numpy as np
import pandas as pd


def compute_var_es(
        values,
        alpha=0.05):
    """
    Compute Value-at-Risk (VaR) and Expected Shortfall (ES)
    for a one-dimensional array of outcomes.

    Parameters
    ----------
    values : np.ndarray
        One-dimensional array of simulated outcomes, such as
        log payoffs or simple returns.
    alpha : float, default=0.05
        Left-tail probability level used to define risk.
        For example, alpha=0.05 corresponds to the 5% tail.

    Returns
    -------
    var : float
        Value-at-Risk at the given alpha level, defined as the
        alpha-quantile of the input values.
    es : float
        Expected Shortfall at the given alpha level, defined as
        the mean of all observations less than or equal to VaR.
        Returns NaN if the tail is empty.

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
        end_prices,
        S0):
    """
    Compute terminal log payoff and simple return
    from simulated horizon prices.

    Parameters
    ----------
    end_prices : np.ndarray
        Simulated asset prices at the forecast horizon across
        all paths.
    S0 : float
        Initial asset price at the start of the simulation.

    Returns
    -------
    log_payoff : np.ndarray
        Log payoff defined as log(S_T / S_0).
    simple_return : np.ndarray
        Simple return defined as (S_T / S_0) - 1.
    """
    end_prices = np.asarray(end_prices).reshape(-1)
    log_payoff = np.log(end_prices / S0)
    simple_return = end_prices / S0 - 1.0
    return log_payoff, simple_return


def summarize_horizon_metrics(
        end_prices,
        S0,
        alpha=0.05):
    """
    Summarize horizon-level simulation and risk metrics.

    Parameters
    ----------
    end_prices : np.ndarray
        Simulated asset prices at the forecast horizon across
        all paths.
    S0 : float
        Initial asset price at the start of the simulation.
    alpha : float, default=0.05
        Tail probability level used to compute VaR and ES.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame containing summary statistics of terminal prices,
        loss probabilities, and tail-risk metrics. Reported fields include:

        - S0
        - mean_end_price
        - median_end_price
        - p05_end_price
        - p95_end_price
        - prob_end_price_below_S0
        - prob_loss_log
        - prob_loss_simple
        - alpha
        - VaR_log
        - ES_log
        - VaR_simple
        - ES_simple
    """
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
