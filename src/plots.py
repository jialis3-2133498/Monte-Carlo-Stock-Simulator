import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def hist_plot(df,
              title=None,
              xlabel=None,
              ylabel=None,
              log=False,
              save_path=None):
    """
    Helper function to plot a histgram
    :param df: Description
    :param title: Description
    :param xlabel: Description
    :param ylabel: Description
    """
    fig, ax = plt.subplots()
    ax.hist(
        df,
        bins="fd",
        linewidth=0.5,
        edgecolor="white"
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        plt.xscale("log")
    else:
        plt.axvline(0, linestyle="--", linewidth=1)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
    return fig, ax


def qq_plot(
        df,
        title=None,
        xlabel=None,
        ylabel=None,
        save_path=None):
    ax = stats.probplot(
        df,
        dist="norm",
        plot=plt
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return ax


def plot_future_price_paths(
        mean_prices: np.ndarray,
        median_prices: np.ndarray,
        p05: np.ndarray,
        p95: np.ndarray,
        S0: np.ndarray,
        save_path: str | None = None):
    days = np.arange(len(mean_prices))
    fig, ax = plt.subplots()
    ax.plot(days, mean_prices, label="mean")
    ax.plot(days, median_prices, label="median")
    ax.fill_between(days, p05, p95, alpha=0.2, label="5-95 band")
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
