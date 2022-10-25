import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers.format_btc_data import fix_format
import datetime as dt
import time
import sys


def _load(ticker, timeframe):
    if ticker == 'BTCUSDT':
        fix_format(ticker, timeframe[:2])

        data = pd.read_csv(f'{ticker}_{timeframe}.csv')
        data = data.loc[:, :'Volume']
        data['Change'] = data['Close'] - data['Open']


    else:
        data = pd.read_csv(f'{ticker}_{timeframe}.csv')
        data = data.sort_index(axis=0, ascending=False)
        data = data.loc[:, :'Volume']

    # print(data.isna().values.sum(), 'NaN values found')
    data.dropna(axis=0, inplace=True)

    data.set_index(pd.DatetimeIndex(data['Time']), drop=True, inplace=True)
    data.drop('Time', axis=1, inplace=True)
    data.rename(columns={'Last': 'Close'}, inplace=True)

    return data