import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optimization import find_best_pnl
from simulation import simulate
from research.load_data import _load
from indicators import EMA
import datetime as dt
import time
import sys


def find_expected_value(ticker, timeframe):
    best_pnls = find_best_pnl('NQU20')

    master = pd.DataFrame(columns=['pair', 'take_prof', 'winpct', 'expected'])

    for row in best_pnls.iterrows():
        ind = row[0]
        row = row[1]
        pair = row['pair']

        take_prof = int(row['take_prof'])


        pnl, num, buys, sells = simulate(ticker, timeframe, pair[0], pair[1], take_prof)
        returns = np.array(sells) - np.array(buys)
        returns_losses = [c for c in returns if np.abs(c) < take_prof]

        e_val = row['winpct'] * take_prof + np.mean(returns_losses) * ( 1 -row['winpct'])

        master = master.append({'pair' : pair,
                                'take_prof' : take_prof,
                                'winpct' : row['winpct'],
                                'expected' : float(e_val)}, ignore_index=True)

    master.to_csv(f'{ticker}_{timeframe}_e_vals.csv')
    return master


def position_sizing(ticker, timeframe, numsteps = 10, exponential_scaler = .5):
    e_vals = pd.read_csv(f'{ticker}_{timeframe}_e_vals.csv')

    max_data = e_vals[e_vals['expected'] == np.max(e_vals['expected'])]
    min_data = e_vals[e_vals['expected'] == np.min(e_vals['expected'])]

    max_eval = float(max_data['expected'])
    min_eval = float(min_data['expected'])

    xspan = np.linspace(min_eval, max_eval, numsteps).reshape(-1 ,1)
    unscaled_positions = xspan ** exponential_scaler

    sig_Graph = 1 / (1 + e ** (-xspan))
    plt.plot(xspan, sig_Graph)
    plt.show()
    scaled_positions = unscaled_positions / unscaled_positions[-1]
