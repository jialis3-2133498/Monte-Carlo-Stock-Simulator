import numpy as np
import pandas as pd
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
from src.database import load_historical_prices, load_simulation_runs
from src.database import load_simulation_price_summary, load_simulation_metrics
from src.database import load_backtest_results, load_scenario_metrics
from src.scenarios import run_scenario_analysis


def main():
    # ================   DATA ================
    TICKER = "AMZN"
    START = "2020-01-01"
    END = "2027-01-01"
    TRAIN_END = "2025-01-01"
    BACKTEST_END = "2026-12-31"
    SEED = 422
    RUN_ID = "AMZN_GBM_2025_422"
    os.makedirs("outputs", exist_ok=True)
    prices = access_stock_price(TICKER, START, END, "1d")
    # Export historical adjusted closing prices for SQL
    historical_prices_df = prices.rename("adjusted_close").reset_index()
    historical_prices_df.columns = ["trade_date", "adjusted_close"]
    historical_prices_df.insert(0, "ticker", TICKER)
    historical_prices_df.to_csv(
        "outputs/historical_prices.csv",
        index=False
    )
    load_historical_prices(historical_prices_df)

    prices_2025_2026 = prices["2025":"2026"]
    train_prices = prices.loc[: TRAIN_END]
    train_r = calculate_log_daily_returns(train_prices)
    descriptive_data = train_r.describe().to_frame().T
    descriptive_data = descriptive_data.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std Dev",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Max"
        }
    )
    descriptive_data["Count"] = descriptive_data["Count"].round(0).astype(int)
    for col in ["Mean", "Std Dev", "Min", "Q1", "Median", "Q3", "Max"]:
        descriptive_data[col] = descriptive_data[col].round(6)
    descriptive_data.to_csv(
        "outputs/returns_descriptive_stats.csv", index=False)
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

    # Export simulation run metadata
    simulation_runs_df = pd.DataFrame([{
        "run_id": RUN_ID,
        "ticker": TICKER,
        "model": "GBM",
        "train_start": str(train_prices.index.min().date()),
        "train_end": str(train_prices.index.max().date()),
        "forecast_horizon": T,
        "n_sims": N,
        "seed": SEED,
        "initial_price": float(S_0)
    }])
    simulation_runs_df.to_csv(
        "outputs/simulation_runs.csv",
        index=False
    )
    load_simulation_runs(simulation_runs_df)

    sim_price_paths = monte_carlo_price_paths(
        train_prices=train_prices,
        train_returns=train_r,
        horizon=T,
        n_sims=N,
        seed=SEED)
    # ================ Scenario-Based Risk Analysis ================
    scenario_results = run_scenario_analysis(
        train_prices=train_prices,
        train_returns=train_r,
        horizon=T,
        n_sims=N,
        seed=SEED,
        alpha=0.05
    )
    scenario_results.insert(0, "run_id", RUN_ID)

    scenario_results.to_csv(
        "outputs/scenario_metrics.csv",
        index=False
    )

    print("\nScenario-Based Risk Analysis:")
    print(scenario_results.to_string(index=False))
    assert np.isclose(
        scenario_results.loc[0, "mean_end_price"],
        sim_price_paths[-1, :].mean()
    ), "Scenario Base Case does not match original simulation"

    print("AMZN Base Case consistency: PASS")
    load_scenario_metrics(scenario_results)

    # Summary statistics across simulation paths
    mean_prices = sim_price_paths.mean(axis=1)
    median_prices = np.median(sim_price_paths, axis=1)
    p05 = np.percentile(sim_price_paths, 5, axis=1)
    p95 = np.percentile(sim_price_paths, 95, axis=1)

    # Export daily simulation summary for SQL and Tableau
    simulation_summary_df = pd.DataFrame({
        "run_id": RUN_ID,
        "day": np.arange(T+1),
        "mean_price": mean_prices,
        "median_price": median_prices,
        "p05_price": p05,
        "p95_price": p95
    })
    simulation_summary_df.to_csv(
        "outputs/simulation_price_summary.csv",
        index=False
    )
    load_simulation_price_summary(simulation_summary_df)


    # mean_prices vs. median_prices
    plot_future_price_paths(
        real_prices=prices_2025_2026,
        real_prices_label="2025-2026 Observed Prices",
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
    # Link risk metrics to the simulation run
    results_df.insert(0, "run_id", RUN_ID)
    # Rename columns to match MySQL schema
    results_df = results_df.rename(columns={
        "S0": "initial_price",
        "Mean": "mean_end_price",
        "Median": "median_end_price",
        "5th pct": "p05_end_price",
        "95th pct": "p95_end_price",
        "Prob(ST < S0)": "prob_loss",
        "Log_loss": "log_loss",
        "Simp_loss": "simple_loss",
        "VaR_log": "var_log",
        "ES_log": "es_log",
        "VaR_simp": "var_simple",
        "ES_simp": "es_simple"
    })
    # Covert boolean values to integers for MySQL compatibility
    backtest_df["inside_band"] = backtest_df["inside_band"].astype(int)
    backtest_df.to_csv("outputs/rolling_backtest.csv", index=False)
    results_df.to_csv("outputs/simulation_metrics.csv", index=False)
    load_simulation_metrics(results_df)
    load_backtest_results(backtest_df)
    plot_csv_table(
        "outputs/simulation_metrics.csv",
        "outputs/simulation_metrics.png"
    )
    print("Results saved to outputs folder.")

    print("Backtest summary:")
    print(backtest_df)

    print("Simulation metrics:")
    print(results_df)


if __name__ == "__main__":
    main()
