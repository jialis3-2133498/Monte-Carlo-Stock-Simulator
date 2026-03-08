import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd


def hist_plot(
        data,
        title=None,
        xlabel=None,
        ylabel=None,
        log=False,
        save_path=None):
    """
    Helper function to plot a histogram.

    Parameters
    ----------
    data : pd.Series
        Numeric data to be plotted.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    log : bool, default=False
        If true, use a logarithmic scale on the x-axis.
    save_path : str, optional
        File path where the figure will be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The generated matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        data,
        bins="fd",
        linewidth=0.5,
        edgecolor="white"
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_xscale("log")
    else:
        ax.axvline(0, linestyle="--", linewidth=1)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def qq_plot(
        data,
        title=None,
        xlabel=None,
        ylabel=None,
        save_path=None):
    """
    Helper function to plot a QQ plot
    against the normal distribution.

    Parameters
    ----------
    data : pd.Series
        Numeric data to be compared
        with the normal distribution.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    save_path : str, optional
        File path where the figure will be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The generated matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(
        data,
        dist="norm",
        plot=ax
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def plot_future_price_paths(
        mean_prices,
        median_prices,
        p05,
        p95,
        S0,
        save_path=None):
    """
    Plot simulated future price statistics.

    Parameters
    ----------
    mean_prices : np.ndarray
        Mean simulated price at each time step.
    median_prices : np.ndarray
        Median simulated price at each time step.
    p05 : np.ndarray
        5th percentile of simulated prices.
    p95 : np.ndarray
        95th percentile of simulated prices.
    S0 : float
        Initial price at the start of the simulation.
    save_path : str, optional
        File path where the figure will be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        Price path plot.
    """
    days = np.arange(len(mean_prices))
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(days, mean_prices, label="Mean path", linewidth=2)
    ax.plot(days, median_prices, label="Median path", linewidth=2)
    ax.fill_between(days, p05, p95, alpha=0.2, label="5-95% band")
    ax.set_yscale("log")
    ax.set_xlabel("Day")
    ax.set_ylabel("Simulated price (log-scale)")
    ax.axhline(S0, linestyle="--", linewidth=1, label="Initial price (S0)")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def plot_backtest_prediction_vs_actual_returns(
        backtest_df,
        save_path=None):
    """
    Helper function to plot
    backtest prediction vs. actual returns

    Parameters
    ----------
    backtest_df : pd.DataFrame
        DataFrame containing rolling backtest results.
    save_path : str, optional
        File path where the figure will be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The generated matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    backtest_df = backtest_df.sort_values("test_start")
    dates = backtest_df["test_start"]
    inside = backtest_df["inside_band"]
    realized = backtest_df["realized_cumlog"]
    outside = ~inside
    coverage = backtest_df["inside_band"].mean()
    ax.fill_between(
        dates,
        backtest_df["p05"],
        backtest_df["p95"],
        alpha=0.25,
        label="90% prediction interval")
    ax.scatter(
        dates[inside],
        realized[inside],
        color="black",
        s=35,
        zorder=3,
        label="Realized return (inside)")
    ax.scatter(
        dates[outside],
        realized[outside],
        color="red",
        s=35,
        zorder=3,
        label="Realized return (outside)"
    )
    ax.text(
        0.02,
        0.95,
        f"Empirical coverage: {coverage:.2%}",
        transform=ax.transAxes
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title(
        "Rolling Backtest of 63-Day Forecast Intervals"
    )
    ax.set_xlabel("Forecast window")
    ax.set_ylabel("63-day cumulative log return")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0.)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def plot_backtest_interval_width(
        backtest_df,
        save_path=None):
    """
    Helper function to plot
    backtest interval width over time.

    Parameters
    ----------
    backtest_df : pd.DataFrame
        DataFrame containing rolling backtest results.
    save_path : str, optional
        File path where the figure will be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The generated matplotlib figure and axes.
    """
    backtest_df = backtest_df.sort_values("test_start")
    dates = backtest_df["test_start"]
    band_width = backtest_df["band_width"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, band_width, linewidth=2)
    ax.set_title("Forecast Interval Width Over Time")
    ax.set_xlabel("Forecast Window")
    ax.set_ylabel("Prediction Interval Width")
    ax.axhline(
        band_width.mean(),
        linestyle="--",
        linewidth=1,
        label="Mean interval width")
    ax.legend()
    ax.grid(alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def plot_rolling_volatility(
        log_returns,
        window=30,
        save_path=None):
    """
    Helper function to plot
    rolling volatility.

    Parameters
    ----------
    log_returns : pd.Series
        Series of daily log returns.
    window : int, default=30
        Rolling window length in trading days.
    save_path : str, optional
        File path where the figure will be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The generated matplotlib figure and axes.
    """
    rolling_vol = log_returns.rolling(window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        rolling_vol.index,
        rolling_vol,
        label="Rolling Volatility",
        linewidth=2)
    ax.set_title(f"Rolling Volatility ({window}-Day Window)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility")
    ax.grid(alpha=0.3)
    ax.axhline(
        rolling_vol.mean(),
        linestyle="--",
        linewidth=1,
        label="Mean volatility"
    )
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def plot_csv_table(
        path,
        save_path):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Statistic"})
    for i in range(len(df)):
        value = float(df.iloc[i, 1])
        if df.iloc[i, 0] == "count":
            df.iloc[i, 1] = int(value)
        else:
            df.iloc[i, 1] = f"{value:.6f}"
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.scale(1.1, 1.6)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0 and row > 0:
            cell.set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    return fig, ax
