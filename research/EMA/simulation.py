import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from research.load_data import _load
from data_prep import prep_data
import datetime as dt
import time
import sys
from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer



def simulate(ticker, timeframe, short, long, take_prof, stop_loss = None):
    master, splits = prep_data(ticker, timeframe, short, long)

    pnl = 0
    num = [0, 0]
    buys = []
    sells = []

    for row in master.iterrows():
        ind = row[0]
        row = row[1]

        if row['direction'] == 'up':
            buys.append(row['cross_val'])
        else:
            sells.append(row['cross_val'])

        if row['max_win'] >=  take_prof * row['cross_val']:
            pnl += take_prof - .5
            num[0] += 1

            if row['direction'] == 'up':
                sells.append((1 + take_prof) * row['cross_val'])

            else:
                buys.append((1 - take_prof) * row['cross_val'])

        elif stop_loss != None and row['max_loss'] >= stop_loss:
            pnl -= stop_loss - .5
            num[1] += 1

            if row['direction'] == 'up':
                sells.append((1 - stop_loss) * row['cross_val'])

            else:
                buys.append((1 + stop_loss) * row['cross_val'])

        else:
            if row['direction'] == 'up':
                pnl += row['next_cross'] - row['cross_val']
                sells.append(row['next_cross'])

            else:
                pnl += row['cross_val'] - row['next_cross']
                buys.append(row['next_cross'])

            num[1] += 1

    print(f'Tested {ticker} {timeframe} | short = {short} | long = {long} | take prof = {take_prof} | stop loss = {stop_loss}')
    print(f'PNL is {pnl} with {np.sum(num)} trades and a {round(num[0]/np.sum(num)*100, 1) } win %')
    return pnl, num, buys , sells

simulate('NQU20' , '5min', 9,30,15,10)

def time_specific_simulate(data, short, long, take_prof):
    master = data
    pnl = 0
    num = [0, 0]
    buys = []
    sells = []

    for row in master.iterrows():
        ind = row[0]
        row = row[1]

        if row['direction'] == 'up':
            buys.append(row['cross_val'])
        else:
            sells.append(row['cross_val'])

        if row['max_win'] >= take_prof * row['cross_val']:
            pnl += take_prof
            num[0] += 1

            if row['direction'] == 'up':
                sells.append((1 + take_prof) * row['cross_val'])

            else:
                buys.append((1 - take_prof) * row['cross_val'])


        else:
            if row['direction'] == 'up':
                pnl += row['next_cross'] - row['cross_val']
                sells.append(row['next_cross'])

            else:
                pnl += row['cross_val'] - row['next_cross']
                buys.append(row['next_cross'])

            num[1] += 1

    print(f'Tested {ticker} {timeframe} | short = {short} | long = {long} | take_prof = {take_prof}')
    print(f'PNL is {pnl} with {np.sum(num)} trades and a {round(num[0]/np.sum(num)*100, 1) } win %')
    return pnl, num, buys , sells
