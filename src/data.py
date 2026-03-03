import yfinance as yf
import pandas as pd


def access_stock_price(company: str,
                start_date: str,
                end_date: str,
                time_interval: str):
    """
    Helper function to access the stock data of a
    single company by using yfinance API.
    
    :param company: Description
    :type company: str
    :param start_date: Description
    :type start_date: str
    :param end_date: Description
    :type end_date: str
    :param time_interval: Description
    :type time_interval: str
    """
    company_df = yf.Ticker(company)
    company_df = company_df.history(
        start=start_date,
        end=end_date,
        interval=time_interval,
        auto_adjust=True)
    prices = company_df["Close"].dropna()
    
    # use tz_localize to make the index more clean and tidy
    prices.index = prices.index.tz_localize(None)
    return prices