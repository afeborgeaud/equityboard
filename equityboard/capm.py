import pandas as pd
import numpy as np
from download import us_tickers
from datetime import date, timedelta
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
import os
from pkg_resources import resource_stream


def stock_prices(tickers=None):
    df = pd.read_parquet(
        resource_stream('resources', 'stock_closes.pq'),
        columns=tickers
    )
    df = df.interpolate(method='backfill', axis=0)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq='D')
    ser = pd.Series(range(len(full_index)), index=full_index, name='dummy')
    df = df.join(ser, how='right')
    df.index.name = 'Date'
    df.interpolate('index', inplace=True)
    df.drop(['dummy'], axis=1, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    return df


def _n_year(from_day: str, to_day: str) -> float:
    return (date.fromisoformat(to_day) - date.fromisoformat(
        from_day)).days / 365.


def risk(
        df_daily: pd.DataFrame, from_day: str, to_day: str,
        annual=True) -> pd.Series:
    r = np.sqrt(
        (
            df_daily[from_day:to_day]
            - df_daily[from_day:to_day].mean(axis=0)
        ).var(axis=0))
    r.name = 'risk'
    if annual:
        n_year = _n_year(from_day, to_day)
        r = r / n_year
    return r


def profit(
        df: pd.DataFrame, from_day: str, to_day: str) -> pd.DataFrame:
    # TODO use lambda
    df_profit = (df[from_day:to_day].interpolate(method='backfill', axis=0)
                 .subtract(df.loc[from_day], axis=1)
                 .divide(df.loc[from_day], axis=1)) * 100.
    return df_profit[from_day:to_day]


def daily_return(df, from_day, to_day):
    previous_day = ((date.fromisoformat(from_day) - timedelta(days=1))
                     .isoformat())
    previous_to_day = ((date.fromisoformat(to_day) - timedelta(days=1))
                    .isoformat())
    daily = df[previous_day:to_day].diff(axis=0)[from_day:to_day]
    daily = daily / df[previous_day:previous_to_day].to_numpy() * 100
    return daily


def capm(
        risk: pd.Series, profit: pd.Series) -> pd.Series:
    df = pd.concat([risk, profit], axis=1, join='inner')
    df_ticker = us_tickers()
    df_ticker.index.name = 'Symbol'
    df.index.name = 'Symbol'
    df = df.join(df_ticker, on='Symbol', how='left')
    # df = df[~df[['risk', 'return', 'Market Cap']].isna().any(axis=1)]
    return df


def efficient_frontier(
        df_daily: pd.DataFrame, profits: pd.Series,
        target_profits: np.array, from_day: str, to_day: str):
    def total_risk(weights):
        df = df_daily[from_day:to_day] - df_daily[from_day:to_day].mean(axis=0)
        return np.sqrt(
            np.sum(
                (df.to_numpy() * weights.reshape(1, -1)),
                axis=1)
            .var()) / _n_year(from_day, to_day)

    n_ticker = len(df_daily.columns)
    if n_ticker == 1:
        return None

    C1 = profits.to_numpy().reshape(1, -1)
    C2 = np.ones((1, n_ticker), dtype='float')
    lc1 = None
    lc2 = LinearConstraint(C2, 1., 1.)
    bounds = [(0., 1.) for i in range(n_ticker)]
    w0 = np.ones(n_ticker, dtype='float') / n_ticker

    weights = np.zeros((len(target_profits), n_ticker), dtype='float')
    total_risks = np.zeros(len(target_profits), dtype='float')
    max_it = 4
    for i, target_profit in enumerate(target_profits):
        lc1 = LinearConstraint(C1, target_profit, target_profit)
        res = minimize(
            total_risk, w0,
            bounds=bounds, constraints=(lc1, lc2),
            options={'maxiter': max_it}
        )
        weights[i] = res.x
        total_risks[i] = total_risk(weights[i])
    total_profits = np.dot(weights, profits.to_numpy())

    return weights, total_risks, total_profits


def result_df(weights: np.array,
                 risks: np.array,
                 profits: np.array,
                 tickers: list[str]) -> pd.DataFrame:
    assert weights.shape[1] == len(tickers)
    d = {'ann. return (%)': profits, 'ann. risk': risks}
    d.update({tickers[i]: weights[:, i] for i in range(len(tickers))})
    df = pd.DataFrame(data=d)
    return df


if __name__ == '__main__':
    df = stock_prices()
    df_filt = df[['AAC', 'AAPL', 'AA']]
    df_capm = capm(df_filt, '2020-04-09', '2021-04-01')
    df_profit = profit(df_filt, '2020-04-09', '2021-04-01')
    df_daily = daily_return(df_filt, '2020-04-09', '2021-04-01')
    profits = df_profit.iloc[-1].to_numpy()

    weights, risks, profits = efficient_frontier(
        df_daily, profits, [float(i) for i in range(1, 100, 2)],
        '2020-04-09', '2021-04-01')
    # risks = np.dot(weights, df_capm_filt.risk.to_numpy())
    # profits = np.dot(weights, df_capm_filt['return'].to_numpy())
    # print(weights)
    # plt.plot(risks, profits)
    # plt.scatter(df_capm['risk'], df_capm['return'])
    # plt.show()

    # print(df.loc[:, 'AAC'])
    # print(df_capm.head())
    # print(df_capm.loc['AAC'])
