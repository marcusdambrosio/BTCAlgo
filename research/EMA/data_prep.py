import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from helpers.format_btc_data import fix_format
from helpers.indicators import EMA
import datetime as dt
import time
import sys
import os
from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer


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



def prep_data(ticker, timeframe, short, long):
    if f'{ticker}_{timeframe}.csv' not in os.listdir('../../data'):
        data = _load(ticker, timeframe)
    else:
        data = pd.read_csv(f'../../data/{ticker}_{timeframe}.csv')


    data = data.iloc[-150000:, :].reset_index()
    data['Change'] = data['Close'] - data['Open']

    closes = data['Close']
    data['shortEMA'] = EMA(closes, short)
    data['longEMA'] = EMA(closes, long)

    cross = []

    for i, item in enumerate(data['shortEMA']):
        if item >= data.longEMA[i]:
            if i == 0:
                under = False

            over = True

            if over and under:
                cross.append('up')
                under = False
            else:
                cross.append(0)


        else:
            if i == 0:
                over = False

            under = True

            if over and under:
                cross.append('down')
                over = False

            else:
                cross.append(0)

    data['cross'] = cross

    splits = []
    start = 0

    for i, item in enumerate(data['cross']):

        if item == 0:
            continue

        else:
            splits.append(data.iloc[start:i])
            start = i



    master = pd.DataFrame(columns = ['cross_val', 'time', 'max_win',  'max_loss', 'cross_change', 'volume', 'direction'])

    del splits[0]

    for split in splits:
        ind = split.index[0]
        end_ind = split.index[-1]

        if split.cross[ind] == 'up':
            current = {'cross_val': split.Close[ind],
                        'time' : split.Time[ind],
                        'max_win' : (np.max(split['High']) - split.Close[ind]),
                        'max_loss': (np.min(split['Low']) - split.Close[ind]),
                        'cross_change' : split.Change[ind],
                        'volume' : split.Volume[ind],
                        'direction' : 'up'}
            master = master.append(current, ignore_index= True)

        else:
            current = {'cross_val' : split.Close[ind],
                        'time': split.Time[ind],
                        'max_win': (split.Close[ind] - np.min(split['Low'])),
                        'max_loss': (split.Close[ind] - np.max(split['High'])),
                        'cross_change': split.Change[ind],
                        'volume': split.Volume[ind],
                        'direction': 'down'}
            master = master.append(current, ignore_index= True)

        end_val = split.Close[end_ind]

    next_cross = master['cross_val'].shift(-1).tolist()
    next_cross[-1] = end_val
    master['next_cross'] = next_cross

    return master, splits



































