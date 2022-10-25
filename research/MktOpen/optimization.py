import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from research.load_data import _load
from indicators import EMA
import datetime as dt
import time
import sys



def mktopen_optimization(ticker, timeframe, prof_range = [], loss_range = []):
    master = pd.DataFrame(columns = ['prof', 'stop', 'pnl'])
    # try:
    for prof in prof_range:
        for stop in loss_range:
            if stop >= 0:
                stop = -stop

            currdata = mktopen_sim(ticker, timeframe, prof, stop)
            pnl = np.sum(currdata['pnl'])
            master = master.append({'prof' : prof,
                                    'stop' : stop,
                                    'pnl' : pnl}, ignore_index=True)
    # except:
    #     print(f'{prof} and {stop} pair is broken')
    #

    max_strat = master[master['pnl'] == np.max(master['pnl'])]
    max_prof , max_stop = max_strat.prof.values[0], max_strat.stop.values[0]


    mktopen_sim(ticker, timeframe, max_prof, max_stop, graph = True)