import pandas as pd
from capm import stock_prices
# from guppy import hpy
import os

if __name__ == '__main__':
    # h = hpy()
    df = stock_prices()
    print(df.info())
    df_2016 = df['2016':]
    print(df_2016.info())
    df_2016.to_parquet('stock_closes_2016.pq')
    # h.heap()