import yfinance as yf


def access_stock_price(
        company: str,
        start_date: str,
        end_date: str,
        time_interval: str):
    """
    Retrieve historical adjusted closing prices for a given asset
    using the yfinance API.

    Parameters
    ----------
    company: str
        Ticker symbol (e.g., "AMZN").
    start_date: str
        A starting date in the format of YYYY-MM-DD
    end_date: str
        An end date in the format of YYYY-MM-DD
    time_interval: str
        Data frequency, like '1d', '1wk'.

    Returns
    -------
    prices: pd.Series
        From yfinance, we access a dataframe containing a
        ticker's opening, closing, adjusted prices, etc.
        And only extract one time-indexed Series
        from this DF, and it is 'Adjusted Close' prices
        with dropped NAs. Timezone info will be stripped.
    Notes
    -----
    The function uses auto-adjusted prices to account for
    dividends and stock splits, ensuring consistency in return
    calculations.

    """
    # We initialize an AMZN ticker object
    company_df = yf.Ticker(company)
    # We use history() to create a DF that
    # contains desired ticker's info.
    company_df = company_df.history(
        start=start_date,
        end=end_date,
        interval=time_interval,
        auto_adjust=True)
    prices = company_df["Close"].dropna()
    # use tz_localize to make the index more clean and tidy
    prices.index = prices.index.tz_localize(None)
    return prices
