import pandas as pd
from capm import stock_prices
import os
import pickle
from pkg_resources import resource_stream

if __name__ == '__main__':
    # df = stock_prices()
    # print(df.head())
    # df_2016 = df['2016':]
    # print(df_2016.info())
    # df.to_parquet('stock_closes_2016.pq')

    # tickers = stock_prices().columns.tolist()
    # with open('tickers.pkl', 'wb') as f:
    #     pickle.dump(tickers, f)

    tickers = pickle.load(resource_stream('resources', 'tickers.pkl'))
    print(len(tickers))