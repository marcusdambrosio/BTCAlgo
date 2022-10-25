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


@jit

def best_entry(ticker, timeframe, short, long):
    master, splits = prep_data(ticker, timeframe, short, long)

    for i, split in enumerate(splits):
        if len(split) < 5:
            splits[i] = split.iloc[:len(split), :]

        else:
            splits[i] = split.iloc[:5, :]

    best_entry = []

    for i, split in enumerate(splits):
        split_low = split['Low']
        split_high = split['High']
        entry = master['cross_val']

        if master.direction[i] == 'up':
            best_entry.append((np.min(split_low) - entry[i]))

        else:
            best_entry.append((entry[i] - np.max(split_high)))

    return best_entry


def best_entry_sim(ticker, timeframe, short, long, take_prof=8, best_entry=-1):
    master, splits = prep_data(ticker, timeframe, short, long)

    pnl = 0
    num = [0, 0]
    buys = []
    sells = []
    better_entry = False

    for i, split in enumerate(splits):
        if len(split) < 5:
            splits[i] = split.iloc[:len(split), :]

        else:
            splits[i] = split.iloc[:5, :]

    for i, split in enumerate(splits):
        split_low = split['Low']
        split_high = split['High']
        entry = master['cross_val']

        if master.direction[i] == 'up':

            if (np.min(split_low) - entry[i]) <= best_entry:
                buys.append(entry[i] + best_entry)
                pnl -= (entry[i] + best_entry)
                # potentially remove this depending on whether take_prof will be based on cross val or entry
                entry[i] = entry[i] + best_entry

            else:
                continue
                buys.append(entry[i])
                pnl -= entry[i]

        else:

            if (entry[i] - np.max(split_high)) <= best_entry:
                sells.append(entry[i] - best_entry)
                pnl += (entry[i] - best_entry)
                # potentially remove this depending on whether take_prof will be based on cross val or entry
                entry[i] = entry[i] - best_entry

            else:
                continue
                sells.append(entry[i])
                pnl += entry[i]

        if master.max_win[i] >= take_prof:
            # pnl += entry[i] + take_prof - best_entry
            num[0] += 1

            if master.direction[i] == 'up':
                pnl += (entry[i] + take_prof - best_entry)
                sells.append(entry[i] + take_prof - best_entry)

            else:
                pnl -= (entry[i] - take_prof + best_entry)
                buys.append(entry[i] - take_prof + best_entry)


        else:
            num[1] += 1

            if master.direction[i] == 'up':
                pnl += master.next_cross[i]
                sells.append(master.next_cross[i])

            else:
                pnl -= master.next_cross[i]
                buys.append(master.next_cross[i])

    return pnl, num, buys, sells
