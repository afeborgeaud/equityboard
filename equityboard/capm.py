import pandas as pd
import numpy as np
from download import us_tickers
from datetime import date, timedelta
from scipy.optimize import minimize, LinearConstraint
import matplotlib.pyplot as plt
import os


def stock_prices():
    df = pd.read_parquet(os.path.join('resources', 'stock_closes.pq'))
    df = df.interpolate(method='backfill', axis=0)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq='D')
    ser = pd.Series(range(len(full_index)), index=full_index, name='dummy')
    df = df.join(ser, how='right')
    df.index.name = 'Date'
    df.interpolate('index', inplace=True)
    df.drop(['dummy'], axis=1, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    return df


def risk(
        df: pd.DataFrame, from_day: str, to_day: str,
        annual=True) -> pd.Series:
    df = df.interpolate(method='backfill', axis=0)  # TODO change to lambda
    df_daily = daily_return(df, from_day, to_day)
    df_daily = df_daily - df_daily.mean(axis=0)
    r = np.sqrt(df_daily.var(axis=0))
    r.name = 'risk'
    if annual:
        n_year = (date.fromisoformat(to_day) - date.fromisoformat(
            from_day)).days / 365.
        r = r / annual
    return r


def returns(
        df: pd.DataFrame, from_day: str, to_day: str,
        annual=True) -> pd.Series:
    # day0 = max(pd.Timestamp(date.fromisoformat(from_day)),
    #     df.index[0])
    df = df.interpolate(method='backfill', axis=0) #TODO change to lambda
    r = (df.loc[to_day] - df.loc[from_day]) / df.loc[from_day] * 100
    r.name = 'return'
    if annual:
        n_year = (date.fromisoformat(to_day) - date.fromisoformat(
            from_day)).days / 365.
        r = r.abs().pow(1. / n_year) * np.sign(r)
    return r


def profit(
        df: pd.DataFrame, from_day: str, to_day: str) -> pd.DataFrame:
    df_profit = (df.interpolate(method='backfill', axis=0) # TODO use lambda
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
        df: pd.DataFrame, from_day: str, to_day: str) -> pd.DataFrame:
    ret = returns(df, from_day, to_day)
    ris = risk(df, from_day, to_day)
    df = pd.concat([ris, ret], axis=1, join='inner')
    df_ticker = us_tickers()
    df_ticker.index.name = 'Symbol'
    df.index.name = 'Symbol'
    df = df.join(df_ticker, on='Symbol', how='left')
    # df = df[~df[['risk', 'return', 'Market Cap']].isna().any(axis=1)]
    return df


def efficient_frontier(
        df_daily, profits, target_profits, from_day, to_day,
        annual=True):
    def total_risk(weights):
        df = df_daily[from_day:to_day] - df_daily[from_day:to_day].mean(axis=0)
        # print(df.head())
        return np.sqrt(
            np.sum(
                (df.to_numpy() * weights.reshape(1, -1)),
                axis=1)
            .var())

    if len(df_daily.columns) == 1:
        return None

    if annual:
        n_year = (date.fromisoformat(to_day) - date.fromisoformat(
            from_day)).days / 365.
        profits_ann = np.abs(profits) ** (1. / n_year) * np.sign(profits)
    else:
        profits_ann = profits

    n = len(profits_ann)
    C1 = profits_ann.reshape(1, -1)
    C2 = np.ones((1, n), dtype='float')
    lc1 = None
    lc2 = LinearConstraint(C2, 1., 1.)
    bounds = [(0., 1.) for i in range(n)]
    w0 = np.ones(n, dtype='float') / n

    weights = np.zeros((len(target_profits), n), dtype='float')
    total_risks = np.zeros(len(target_profits), dtype='float')
    max_it = 3
    for i, target_profit in enumerate(target_profits):
        lc1 = LinearConstraint(C1, target_profit, target_profit)
        res = minimize(
            total_risk, w0,
            bounds=bounds, constraints=(lc1, lc2),
            options={'maxiter': max_it}
            # True if x[1].nit >= max_it else False
        )
        # print(res)
        weights[i] = res.x
        total_risks[i] = total_risk(weights[i])
    total_profits = np.dot(weights, profits_ann)

    if annual:
        total_risks = total_risks ** (1. / n_year)

    return weights, total_risks, total_profits


if __name__ == '__main__':
    df = stock_prices()
    df_filt = df[['AAC', 'AAPL', 'AA']]
    df_capm = capm(df_filt, '2020-04-09', '2021-04-01')
    df_profit = profit(df_filt, '2020-04-09', '2021-04-01')
    df_daily = daily_return(df_filt, '2020-04-09', '2021-04-01')
    profits = df_profit.iloc[-1].to_numpy()

    print(df.info())
    # print(df.describe())

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