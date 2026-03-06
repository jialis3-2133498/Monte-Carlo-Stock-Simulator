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
    Helper function to plot a histgram.

    Parameters
    ----------
    df : pd.Series
        A Series of log returns.
    title : None
        Title of the plot.
    xlabel : None
        x-axis label
    ylabel : None
        y-axis label
    log : Boolean
        If true, the x scale will be in log form.
    save_path : str
        An address to save the plot.
    
    Returns
    -------
    fig, ax : plt.subplots
        Histgram plot
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
    
    """
    Helper function to plot a QQ plot.

    Parameters
    ----------
    df : pd.Series
        A Series of log returns.
    title : None
        Title of the plot.
    xlabel : None
        x-axis label
    ylabel : None
        y-axis label
    log : Boolean
        If true, the x scale will be in log form.
    save_path : str
        An address to save the plot.
    
    Returns
    -------
    ax : stats.probplot
        QQ plot
    """
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
        mean_prices,
        median_prices,
        p05,
        p95,
        S0,
        save_path=None):
    """
    Plot future price paths

    Parameters
    ----------
    mean_prices : np.ndarray
    """
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
