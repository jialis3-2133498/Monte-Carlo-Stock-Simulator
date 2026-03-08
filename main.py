import numpy as np
import os
import matplotlib.pyplot as plt

from src.data import access_stock_price
from src.returns import calculate_log_daily_returns
from src.backtest import rolling_backtest
from src.plots import hist_plot, qq_plot, plot_future_price_paths
from src.plots import plot_backtest_prediction_vs_actual_returns
from src.plots import plot_rolling_volatility, plot_backtest_interval_width
from src.plots import plot_csv_table
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
    plot_csv_table(
        "outputs/returns_descriptive_stats.csv",
        "outputs/returns_descriptive_stats.png"
    )
    # Plot the Histogram of the Log Daily Returns
    hist_plot(
        data=train_r,
        title="Amazon's Stock Log Daily Returns Distribution",
        xlabel="Log Daily Returns",
        ylabel="Counts",
        log=False,
        save_path="outputs/log_daily_returns_hist.png")
    # Plot the QQ-plot of the Log Daily Returns
    qq_plot(
        data=train_r,
        title="Q-Q Plot of Daily Log Returns vs Normal",
        xlabel="Theoretical Quantiles (Normal Distribution)",
        ylabel="Sample Quantiles (Observed Data)",
        save_path="outputs/log_daily_returns_qq.png")

    plot_rolling_volatility(
        log_returns=train_r,
        window=30,
        save_path="outputs/rolling_volatility.png")
    # ================ Rolling Backtest ================
    # Perform a Rolling Backtest on 2020-2026 data
    backtest_df = rolling_backtest(
        prices.loc[START:BACKTEST_END], window=756, horizon=63, step=21)
    plot_backtest_prediction_vs_actual_returns(
        backtest_df=backtest_df,
        save_path="outputs/backtest_simulated_returns_vs_actual_returns.png")
    plot_backtest_interval_width(
        backtest_df=backtest_df,
        save_path="outputs/backtest_interval_width.png"
    )

    # ================ Simulation ================
    # Monte Carlo simulation under daily log-return GBM
    T = 1260  # Simulate the price in the future 1260 days
    N = 1000  # For each day, we will simulate 1000 paths/day
    S_0 = train_prices.iloc[-1]
    sim_price_paths = monte_carlo_price_paths(
        train_prices=train_prices,
        train_returns=train_r,
        horizon=T,
        n_sims=N,
        seed=SEED)

    # Summary statistics across simulation paths
    mean_prices = sim_price_paths.mean(axis=1)
    median_prices = np.median(sim_price_paths, axis=1)
    p05 = np.percentile(sim_price_paths, 5, axis=1)
    p95 = np.percentile(sim_price_paths, 95, axis=1)

    # mean_prices vs. median_prices
    plot_future_price_paths(
        mean_prices=mean_prices,
        median_prices=median_prices,
        p05=p05,
        p95=p95,
        S0=S_0,
        save_path="outputs/future_price_paths.png"
    )
    # End-of-horizon metrics
    end_prices = sim_price_paths[-1, :]  # Last day simulated prices
    log_payoff, _ = compute_payoffs(end_prices, S_0)

    # Visualize Horizon payoff and prices
    fig, ax = hist_plot(
        data=end_prices,
        title="Prices at 5Y Horizon",
        xlabel="Prices at Horizon",
        ylabel="Count",
        log=True)
    ax.axvline(
        S_0, color="red", linestyle="--", linewidth=2, label="Current Price")
    p05_price = np.percentile(end_prices, 5)
    ax.axvline(
        p05_price,
        color="black",
        linestyle="--",
        linewidth=2,
        label="5th Percentile")
    ax.legend()
    fig.savefig(
        "outputs/horizon_end_prices_hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    hist_plot(
        data=log_payoff,
        title="Log Payoff Distribution at 5Y Horizon",
        xlabel="Log Payoff at 5-Year Horizon",
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
    backtest_df.to_csv("outputs/rolling_backtest.csv", index=False)
    results_df.to_csv("outputs/simulation_metrics.csv", index=False)
    print("Results saved to outputs folder.")

    print("Backtest summary:")
    print(backtest_df)

    print("Simulation metrics:")
    print(results_df)


if __name__ == "__main__":
    main()
