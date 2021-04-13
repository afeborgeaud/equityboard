import requests
from datetime import datetime, time, timezone, date
import pandas as pd
import os
from tqdm import tqdm
import logging
from time import sleep

logging.basicConfig(level=logging.DEBUG,
                    filename='log_download.txt',
                    filemode='w')


def investopedia_url(ticker: str, from_day: str, to_day: str):
    from_date = date.fromisoformat(from_day)
    to_date = date.fromisoformat(to_day)
    format = '%b+%d%%2C+%Y'
    return (
        "https://www.investopedia.com/markets/api/partial/historical/?"
        f"Symbol={ticker.upper()}&Type=Historical+Prices&Timeframe=Daily&"
        f"StartDate={from_date.strftime(format)}&"
        f"EndDate={to_date.strftime(format)}"
    )
    # "https://www.investopedia.com/markets/api/partial/historical/?Symbol=[SYMBOL NAME]&Type="
    # "Historical+Prices&Timeframe=Daily&StartDate=Nov+28%2C+2017&EndDate=Dec+05%2C+2017"


def yahoo_url(ticker: str, from_day: str, to_day: str) -> str:
    """Get url to request yahoo finance."""
    from_dt = int(
        datetime.combine(
            date.fromisoformat(from_day), time(), tzinfo=timezone.utc)
            .timestamp()
    )
    to_dt = int(
        datetime.combine(
            date.fromisoformat(to_day), time(), tzinfo=timezone.utc)
            .timestamp()
    )
    return (
        "https://query1.finance.yahoo.com/v7/finance/download/"
        f"{ticker.upper()}?period1={from_dt}&period2={to_dt}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )


def rename_ticker(ticker: str) -> str:
    return ticker.replace('/', '-')


def yahoo_to_series(ticker: str, from_day: str, to_day: str) -> pd.Series:
    """Return closing prices for ticker from Yahoo finance."""
    def to_float(str):
        if str == 'null':
            return None
        else:
            return float(str)
    url = yahoo_url(ticker, from_day, to_day)
    logging.info(f"requesting {ticker}")
    r = requests.get(url)
    # if r.status_code != 200:
    #     raise ValueError(f'Could not fetch {url}')
    str = r.content.decode('utf-8')
    try:
        d = {s.split(',')[0]: to_float(s.split(',')[4])
            for s in str.split('\n')[1:]}
    except IndexError:
        logging.debug(url)
        logging.debug(f'request returned {r}\ncould not parse {ticker}')
        return None
    ser = pd.Series(data=d,
                    name=rename_ticker(ticker).upper(),
                    dtype='float32')
    ser.index = pd.to_datetime(ser.index)
    ser.index.name = 'date'
    return ser


def yahoo_to_dataframe(tickers: list[str],
        from_day: str, to_day: str) -> pd.DataFrame:
    """Get closing prices from tickers from Yahoo finance."""
    series = []
    for t in tqdm(tickers):
        fname = f'{rename_ticker(t).lower()}.gzip'
        if os.path.exists(fname):
            ser = pd.read_pickle(fname)
            series.append(ser)
        else:
            ser = yahoo_to_series(t, from_day, to_day)
            sleep(1)
            if ser is not None:
                ser.to_pickle(f'{rename_ticker(t).lower()}.gzip')
                series.append(ser)
    if len(series) == 0:
        return None
    df = pd.concat(series, axis=1, join='outer')
    return df


def us_tickers() -> pd.DataFrame:
    """Returns a list of US-listed companies tickers."""
    fname = os.path.join('resources',
                         'nasdaq_screener_1618019870209.csv')
    df = pd.read_csv(fname, header=0, index_col=0)
    return df


if __name__ == '__main__':
    companies = us_tickers()
    companies = companies[
        ~companies.index.str.contains('^', regex=False)
    ]
    tickers = companies.index.to_numpy()
    df = yahoo_to_dataframe(tickers, '1990-01-01', '2021-04-11')
    df.to_parquet(os.path.join('resources', 'stock_closes.pq'))

