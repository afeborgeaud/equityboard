import requests
from datetime import datetime, time, timezone, date, timedelta
import time as tt
import pandas as pd
import os
from tqdm import tqdm
import logging
from time import sleep
from pkg_resources import resource_stream, resource_filename
import pickle


logging.basicConfig(level=logging.DEBUG,
                    filename='log_download.txt',
                    filemode='w')

df_fname_tmp = resource_filename(
                'resources', 'stock_closes_tmp.pq')
df_fname = df_fname_tmp.replace('_tmp', '')

oldest_day = '1990-01-02'
crypto_tickers = [
    'ETH-USD','BTC-USD','XLM-USD','LINK-USD','XRP-USD','XTZ-USD',
    'REP-USD','ZRX-USD','ADA-USD','UNI3-USD','AAVE-USD','ATOM1-USD',
    'ALGO-USD','MKR-USD','COMP-USD','YFI-USD'
]


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
        f"{rename_ticker(ticker).upper()}"
        f"?period1={from_dt}&period2={to_dt}&interval=1d"
        "&events=history&includeAdjustedClose=true"
    )


def rename_ticker(ticker: str) -> str:
    return (
        ticker.replace('/', '-')
            .strip()
    )


def yahoo_to_series(ticker: str, from_day: str, to_day: str) -> pd.Series:
    """Return closing prices for ticker from Yahoo finance."""
    def to_float(str):
        if str == 'null':
            return None
        else:
            return float(str)
    url = yahoo_url(ticker, from_day, to_day)
    logging.info(url)
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


def yahoo_to_dataframe(tickers: list,
        from_day: str, to_day: str,
        current_tickers=None) -> pd.DataFrame:
    """Get closing prices from tickers from Yahoo finance."""
    series = []
    for t in tqdm(tickers):
        if (current_tickers is not None) and (t not in current_tickers):
            from_day = oldest_day
        fname = f'{rename_ticker(t).lower()}.gzip'
        fpath = resource_filename('resources', fname)
        if os.path.exists(fpath):
            ser = pd.read_pickle(fpath)
            series.append(ser)
        else:
            ser = yahoo_to_series(t, from_day, to_day)
            sleep(1)
            if ser is not None:
                ser.to_pickle(fpath)
                series.append(ser)
    if len(series) == 0:
        return None
    df = pd.concat(series, axis=1, join='outer')
    return df


def us_tickers() -> pd.DataFrame:
    """Returns a list of US-listed companies tickers."""
    fname = os.path.join('resources',
                         'nasdaq_screener_1618019870209.csv')
    df = pd.read_csv(
        resource_stream('resources', 'nasdaq_screener_1618019870209.csv'),
        header=0, index_col=0
    )
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    df_proc = df.interpolate(method='backfill', axis=0)
    full_index = pd.date_range(df_proc.index.min(), df_proc.index.max(),
                               freq='D')
    ser = pd.Series(range(len(full_index)), index=full_index, name='dummy')
    df_proc = df_proc.join(ser, how='right')
    df_proc.index.name = 'Date'
    df_proc.interpolate('index', inplace=True)
    df_proc.drop(['dummy'], axis=1, inplace=True)
    df_proc.dropna(axis=1, how='all', inplace=True)
    df_proc['day'] = df_proc.index.date
    df_proc.drop_duplicates(subset='day', inplace=True)
    df_proc.drop('day', axis=1, inplace=True)
    return df_proc


def check(df: pd.DataFrame) -> bool:
    logging.info(df.info())
    na_columns = df.isna().all(axis=0)
    na_columns = na_columns[na_columns]
    na_tickers = ' '.join([t for t in na_columns.index])
    logging.info(f'NaN tickers: {na_tickers}')
    logging.info(f'Number of NaN tickers: {len(na_columns)}')
    return True


def most_recent_date():
    df = pd.read_parquet(
        df_fname,
        columns=['AAPL']
    )
    day = df.index[-1].strftime("%Y-%m-%d")
    next_day = ((date.fromisoformat(day) + timedelta(days=1))
                    )
    today = datetime.utcfromtimestamp(tt.time()).date()
    return next_day, today


def get_tickers() -> list:
    companies = us_tickers()
    companies = companies[
        ~companies.index.str.contains('^', regex=False)
    ]
    return companies.index.to_numpy()


def write_to_parquet(df: pd.DataFrame) -> None:
    if check(df):
        try:
            os.rename(df_fname, df_fname + f'_backup_{date.today()}')
            process(df).to_parquet(df_fname_tmp)
            os.rename(df_fname_tmp, df_fname)
            logging.info(f'wrote new dataframe to {df_fname}')

            all_tickers = process(df).columns.tolist()
            fname_ticker = resource_filename(
                'resources', 'tickers.pkl')
            with open(fname_ticker, 'wb') as f:
                pickle.dump(all_tickers, f)
            logging.info(f'wrote tickers to {fname_ticker}')
        except:
            logging.debug('failed to write new dataframe')


def add_tickers(df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    from_day = df.index[0].strftime("%Y-%m-%d")
    to_day = df.index[-1].strftime("%Y-%m-%d")
    df_add = yahoo_to_dataframe(tickers, from_day, to_day)
    return df.join(df_add, how='left')


def request_data() -> None:
    try:
        df = pd.read_parquet(
            df_fname,
        )
    except:
        df = None
    next_day, today = most_recent_date()
    if (today - next_day) >= timedelta(days=6):
        tickers = get_tickers()
        df_new = yahoo_to_dataframe(tickers,
                                    next_day.isoformat(),
                                    today.isoformat(),
                                    df.columns)
        if df is not None:
            df = process(
                pd.concat([df, df_new])
            )
        else:
            df = df_new
        write_to_parquet(df)


if __name__ == '__main__':
    request_data()

