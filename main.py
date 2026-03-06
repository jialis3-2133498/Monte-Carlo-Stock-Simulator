import numpy as np
import os

from src.data import access_stock_price
from src.returns import calculate_log_daily_returns
from src.backtest import rolling_backtest
from src.plots import hist_plot, qq_plot, plot_future_price_paths
from src.metrics import compute_payoffs, summarize_horizon_metrics
from src.simulation import monte_carlo_price_paths


def main():
    # ================   DATA ================
    TICKER = "AMZN"
    START = "2020-01-01"
    END = "2027-01-01"
    TRAIN_END = "2025-01-01"
    BACKTEST_END = "2026-12-31"
    SEED = 422
    os.makedirs("outputs", exist_ok=True)
    prices = access_stock_price(TICKER, START, END, "1d")
    train_prices = prices.loc[: TRAIN_END]
    train_r = calculate_log_daily_returns(train_prices)
    descriptive_data = train_r.describe()
    descriptive_data.to_frame(
        name="log_return").to_csv("outputs/returns_descriptive_stats.csv")
    # Plot the Histgram of the Log Daily Returns
    hist_plot(
        train_r,
        title="Amazon's Stock Log Daily Returns Distribution",
        xlabel="Log Daily Returns",
        ylabel="Counts",
        log=False,
        save_path="outputs/log_daily_returns_hist.png")
    # Plot the QQ-plot of the Log Daily Returns
    qq_plot(
        train_r,
        title="Q-Q Plot of Daily Log Returns vs Normal",
        xlabel="Theoretical Quantiles (Normal Distribution)",
        ylabel="Sample Quantiles (Observed Data)",
        save_path="outputs/log_daily_returns_qq.png")

    # ================ Rolling Backtest ================
    # Perform a Rolling Backtest on 2020-2026 data
    bt = rolling_backtest(
        prices.loc[START:BACKTEST_END], window=756, horizon=63, step=21)
    summary = bt[["coverage", "ave_width"]].iloc[0].to_frame().T
    summary.to_csv("outputs/backtest_summary.csv", index=False)

    # ================ Simulation ================
    # Model A: daily-log-return GBM (Euler exact in log space)
    T = 1260  # Simulate the price in the future 1260 days
    N = 1000  # For each day, we will simulate 1000 paths/day
    S_0 = train_prices.iloc[-1]
    S_T_daily_para_sim = monte_carlo_price_paths(
        train_prices=train_prices,
        train_returns=train_r,
        horizon=T,
        n_sims=N,
        seed=SEED)

    # Summary statistics across simulation paths
    mean_prices = S_T_daily_para_sim.mean(axis=1)
    median_prices = np.median(S_T_daily_para_sim, axis=1)
    p05 = np.percentile(S_T_daily_para_sim, 5, axis=1)
    p95 = np.percentile(S_T_daily_para_sim, 95, axis=1)

    # mean_prices vs. median_prices
    plot_future_price_paths(
        mean_prices,
        median_prices,
        p05,
        p95,
        S_0,
        save_path="outputs/future_price_paths.png"
    )
    # End-of-horizon metrics
    end_prices = S_T_daily_para_sim[:, -1]  # Last day simulated prices
    log_payoff, _ = compute_payoffs(end_prices, S_0)

    # Visualize Horizon payoff and prices
    hist_plot(
        end_prices,
        title="Prices at 5Y Horizon",
        xlabel="Prices at Horizon",
        ylabel="Count",
        log=True,
        save_path="outputs/horizon_end_prices_hist.png")

    hist_plot(
        log_payoff,
        title="Log Payoff at 5Y Horizon",
        xlabel="Payoff in the horizon (Log-scale)",
        ylabel="Count",
        log=False,
        save_path="outputs/horizon_logpayoff_scale_hist.png"
    )
    # ================ Metrics + Outputs ================
    # Create a result metric
    results_df = summarize_horizon_metrics(
        end_prices=end_prices,
        S0=S_0,
        alpha=0.05)
    bt.to_csv("outputs/rolling_backtest.csv", index=False)
    results_df.to_csv("outputs/simulation_metrics.csv", index=False)
    print("Results saved to outputs folder.")

    print("Backtest summary:")
    print(summary)

    print("Simulation metrics:")
    print(results_df)


if __name__ == "__main__":
    main()
