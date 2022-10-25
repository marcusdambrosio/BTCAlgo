import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_all_times(freq = '30'):
    all_times = pd.date_range('00:00:00', '23:59:59', freq= freq + 'min')
    all_times = [str(time) for time in all_times]
    all_timesstr = [time[11:] for time in all_times]
    all_timesdt = [dt.time(int(c[:2]), int(c[3:5])) for c in all_timesstr]

    return all_timesstr, all_timesdt

def get_ema_params(ticker, timeframe, freq = '30min'):
    data = pd.read_csv(f'research/EMA/{ticker}_{timeframe}_{freq}TODopti.csv')

    expected = data['expected']
    bottom = expected.min()
    top = expected.max()
    diff = top - bottom

    e = 1.1
    while (1/(1+e**-diff)) < .999:
        e += .05

    w = (1/(1+e**-(expected-bottom))/.5)
    data['weight'] = w

    ema_dict = {}

    for row in data.iterrows():
        ind = row[0]
        row = row[1]
        ema_dict[row['TOD']] = [row['short'], row['long'], row['take_prof'], row['weight']]

    return ema_dict
#
# def get_position_sizing(ticker, timeframe, freq = '30min'):
#     data = pd.read_csv(f'{ticker}_{timeframe}_{freq}TODopti.csv')
#     data = data[data['expected'] > 0]
#     expected = data['expected']
#     bottom = expected.min()
#     top = expected.max()
#     diff = top - bottom
#
#     e = 1.1
#     while (1/(1+e**-diff)) < .999:
#         e += .05
#
#     w = (1/(1+e**-(expected-bottom))/.5)
#     weights = {}
#     for  i, TOD in enumerate(data['TOD']):
#         weights[TOD] = w[i]
#
#     return weights
#
#

