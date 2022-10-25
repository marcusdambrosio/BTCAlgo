import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from research.load_data import _load
from indicators import EMA
import datetime as dt
import time
import sys



def mktopen_avganalysis(ticker, timeframe, filtering_quantiles = [0, .9]):
    time_dict = mktopen_data(ticker, timeframe, filtering_quantiles)

    allmins_data = pd.DataFrame(columns = ['time', 'avghigh', 'avglow', 'opportunity', 'loss', 'drawdown', 'avgvol'])

    minrange = np.arange(30, 51, int(timeframe[0]))
    minrange = minrange[1:]

    fulltimes = ['08:' + str(c) for c in minrange]

    for time in fulltimes:
        currdata =  time_dict[time]
        allmins_data = allmins_data.append({'time' : time,
                                            'avghigh' : np.mean(currdata['nexthigh'] - currdata['nextopen']),
                                            'avglow' : np.mean(currdata['nextlow'] - currdata['nextopen']),
                                            'opportunity' : np.mean((currdata['opp_dir'])),
                                            'loss' : np.mean(currdata['loss']),
                                            'drawdown' : np.mean(currdata['drawdown']),
                                            'Volume' : np.mean(currdata['Volume'])}, ignore_index = True)


    fig, ax = plt.subplots(2, 2, sharex = True)
    ax[0, 0].bar(fulltimes, allmins_data['avghigh'], color = 'green', label = 'Avg Next High')
    ax[0, 0].bar(fulltimes, allmins_data['avglow'], color = 'red' , label = 'Avg Next Low')
    ax[0, 0].grid(axis = 'y')
    ax[0, 0].legend()
    ax[0, 0].set_title('Next Min Extrema')

    ax[1, 0].bar(fulltimes,allmins_data['opportunity'], color = 'green', label = 'Opportunity')
    ax[1, 0].bar(fulltimes,allmins_data['loss'], color = 'red', label = 'Loss')
    ax[1, 0].grid(axis = 'y')
    ax[1, 0].set_title('Avg Opportunity and Loss')

    ax[0, 1].bar(fulltimes, allmins_data['Volume'])
    ax[0, 1].grid(axis='y')
    ax[0, 1].set_title('Avg Volume')

    ax[1, 1].bar(fulltimes, allmins_data['drawdown'], color = 'red')
    ax[1, 1].grid(axis='y')
    ax[1, 1].set_title('Avg Max Drawdown')
    plt.show()

    return allmins_data

def mktopen_sim(ticker, timeframe, take_prof, stop_loss, graph = False):
    time_dict = mktopen_data(ticker, timeframe)
    times = time_dict.keys()
    full_sim_dict = {}

    for key in times:
        full_sim_dict[key] = pd.DataFrame(columns = ['outcome', 'drawdown', 'Volume'])

    for key in times:
        currdata = time_dict[key]

        for row in currdata.iterrows():
            ind = row[0]
            row = row[1]


            if row['opp_dir'] >= take_prof:
                outc = take_prof
            elif row['drawdown'] <= stop_loss:
                outc = stop_loss - .5
            else:
                outc = row['loss']

            full_sim_dict[key] = full_sim_dict[key].append({'outcome' : outc,
                                                  'drawdown' : row['drawdown'],
                                                  'Volume' : row['Volume']}, ignore_index = True)


    master = pd.DataFrame(columns = ['pnl', 'winpct', 'windrawdown', 'losedrawdown'])

    for key in times:
        currdata = full_sim_dict[key]
        winners = currdata[currdata['outcome'] == take_prof]
        losers = currdata[currdata['outcome'] != take_prof]
        win_drawdown = np.mean(winners['drawdown'])
        loss_drawdown = np.mean(losers['drawdown'])

        pnl = np.sum(currdata['outcome'])
        winpct = len(winners) / len(currdata)

        master = master.append({'pnl' : pnl,
                                'winpct' : winpct,
                                'windrawdown' : win_drawdown,
                                'lossdrawdown' : loss_drawdown}, ignore_index=True)

    if graph:
        fig, ax = plt.subplots(2, 2, sharex = True)
        ax[0, 0].bar(times, master['pnl'], color = 'green')
        ax[0, 0].grid(axis = 'y')
        ax[0, 0].set_title('PNL')

        ax[1, 0].bar(times, master['windrawdown'], color = 'green', width = .5, label = 'Winners')
        ax[1, 0].bar(times, master['lossdrawdown'], color = 'red', width = .25, label = 'Losers')
        ax[1, 0].grid(axis = 'y')
        ax[1, 0].legend()
        ax[1, 0].set_title('Drawdown')

        ax[0, 1].bar(times, master['winpct'])
        ax[0, 1].grid(axis='y')
        ax[0, 1].set_title('Win Pct')

        plt.suptitle(f'Take prof at {take_prof} and stop loss at {stop_loss}')
        plt.show()

    return master
