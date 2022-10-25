import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from research.load_data import _load
import datetime as dt
import time
import sys
from numba import jit, cuda
import numpy as np
from timeit import default_timer as timer


@jit

def volume_change_analysis(ticker ,timeframe, short, long, take_prof):
    master, splits = prep_data(ticker, timeframe, short, long)

    fig, ax = plt.subplots(2)

    ax[0].scatter(master['volume'], master['max_win'])
    ax[0].axhline(take_prof)
    ax[0].set_xlabel('Volume')
    ax[0].grid()

    ax[1].scatter(np.abs(master['cross_change']), master['max_win'])
    ax[1].axhline(take_prof)
    ax[1].set_xlabel('Change during cross')
    ax[1].grid()

    plt.ylabel('max_win')
    plt.show()


def length_of_trade(ticker, timeframe, short, long, take_prof):
    master, splits = prep_data(ticker, timeframe, short, long)
    maxwins = master['max_win']
    winlengths = []
    losslengths = []

    for i, split in enumerate(splits):
        if maxwins[i] < take_prof:
            losslengths.append(len(split))


        else:
            counter = 0
            if master.direction[i] == 'up':
                while (split.High[counter] - master.cross_val[i]) < take_prof:
                    counter += 1

            else:
                while (master.cross_val[i] - split.Low[counter]) < take_prof:
                    counter += 1

            winlengths.append(counter + 1)


    '''STATISTICAL FILTERING'''
    # winlengths = pd.Series(winlengths)
    # losslengths = pd.Series(losslengths)
    # winlengths = winlengths[winlengths.between(0, winlengths.quantile(.95))]
    # losslengths = losslengths[losslengths.between(0, losslengths.quantile(.95))]


    avgwinlength = round(np.average(winlengths), 1)
    avglosslength = round(np.average(losslengths), 1)

    fig, ax = plt.subplots(2, sharex = True)

    ax[0].hist(winlengths, bins = 50, color = 'Green', label = f'Average Win Time: {avgwinlength} {timeframe} candlesticks or {int(timeframe[0] ) *avgwinlength} minutes')
    ax[0].grid()
    ax[0].legend()

    ax[1].hist(losslengths, bins = 50, color = 'Red', label = f'Average Loss: {avglosslength} {timeframe} candlesticks or {int(timeframe[0] ) *avglosslength} minutes')
    ax[1].grid()
    ax[1].legend()

    plt.xlabel('Trade Length (in candlesticks)')
    plt.ylabel('Occurences')

    plt.show()


def winloss_analysis(ticker, timeframe, short, long):
    master, splits = prep_data(ticker, timeframe, short, long)
    maxwins = master['max_win']
    maxlosses = master['max_loss']
    wins, losses = [], []

    print('With full data, the average max_win is ', np.average(maxwins))
    print('With full data, the average max_loss is ', np.average(maxlosses))

    filteredwins = maxwins[maxwins.between(0, maxwins.quantile(.9))]
    filteredlosses = maxlosses[maxlosses.between(0, maxlosses.quantile(.9))]

    print('With top 10% filtered out, the average max_win is ', np.average(filteredwins))



    newtime = []

    for item in master['time']:
        t = str(item)
        newtime.append(int(t[11:13]))
    plt.scatter(newtime, master['max_win'], color = 'green')
    plt.scatter(newtime, master['max_loss'], color = 'red')
    plt.grid()
    plt.show()



def time_of_day_analysis(ticker, timeframe, short, long, take_prof, TODfreq = '30min'):
    master, splits = prep_data(ticker, timeframe, short, long)
    all_times = pd.date_range('00:00:00', '23:59:59', freq = TODfreq)

    all_times = [str(time) for time in all_times]
    all_times = [time[11:-3] for time in all_times]
    all_times_str = all_times


    for i, t in enumerate(all_times):
        if t[0] == '0':
            all_times[i] = dt.time(int(t[1]), int(t[3:]))

        else:
            all_times[i] = dt.time(int(t[:2]), int(t[3:]))

    new_master = pd.DataFrame(columns=['cross_val', 'time', 'max_win', 'max_loss', 'cross_change', 'volume', 'direction', 'TOD'])
    ##MAKE NEW TIME OF DAYS
    TOD = []
    for t in master['time']:
        t = t.strftime('%H:%M')

        if t[0] == '0':
            t = dt.time(int(t[1]), int(t[3:]))
        else:
            t = dt.time(int(t[:2]), int(t[3:]))

        TOD.append(t)

    master['TOD'] = TOD

    min_splits = {}
    for t_str in all_times:
        min_splits[t_str] = pd.DataFrame()

    for row in master.iterrows():
        ind = row[0]
        row = row[1]
        curr_tod = row['TOD']

        if curr_tod >= all_times[-1]:
            min_splits[dt.time(23, 30)] = min_splits[dt.time(23, 30)].append(row)
        else:
            for i, times in enumerate(all_times):
                if times < curr_tod < all_times[i+1]:
                    min_splits[times] = min_splits[times].append(row)
                    break


    master_TOD = pd.DataFrame()

    for i, item in enumerate(all_times):

        curr_df = min_splits[item]

        if not len(curr_df):
            master_TOD = master_TOD.append({'TOD' : item,
                                            'max_win' : 0,
                                            'max_loss' : 0,
                                            'winpct' : 0,
                                            'dpoints' : len(curr_df)}, ignore_index = True)
            continue

        wincount = len([c for c in curr_df['max_win'].tolist() if c >= take_prof + .25])
        master_TOD = master_TOD.append({'TOD' : item,
                                        'max_win' : np.mean(curr_df['max_win']),
                                        'max_loss' : np.mean(curr_df['max_loss']),
                                        'winpct' : wincount/len(curr_df),
                                        'dpoints' : len(curr_df)}, ignore_index = True)




    master_TOD.set_index(master_TOD['TOD'], inplace = True)

    fig, ax = plt.subplots(2, sharex = True)
    xind = master_TOD['TOD'].astype(str)

    #win loss plot
    ax[0].plot(xind, master_TOD['max_win'], color = 'green', label = 'Max win')
    ax[0].plot(xind, master_TOD['max_loss'], color = 'red', label = 'Max loss')
    ax[0].set_ylabel('Trade value')

    #win pct plot
    ax[1].plot(xind, master_TOD['winpct'], color = 'blue', label = 'winpct')

    #dpoint addition
    for i, item in enumerate(master_TOD['dpoints']):
        ax[1].annotate(item, (xind[i], master_TOD.winpct[i]))

    plt.xlabel('Time Of Day')
    plt.xticks(rotation=45)
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    plt.show()

    return master_TOD


def analyze_all_options(ticker, timeframe):
    data = pd.read_csv(f'{ticker}_{timeframe}_optimizeData.csv')
    short_periods = data['short']
    long_periods = data['long']
    pnl = data['pnl']
    winning = data['winning']
    losing = data['losing']
    winpct = winning / (winning + losing)
    takeprof = data['take prof']

    fig, ax = plt.subplots(2, sharex=True)

    ax[0].scatter(takeprof, pnl)
    ax[0].grid()
    ax[0].set_ylabel('PNL')

    ax[1].scatter(takeprof, winpct)
    ax[1].grid()
    ax[1].set_ylabel('Win Percentage')

    plt.xlabel('Take Prof Value')
    plt.show()

def analyze_optimization(ticker, timeframe):
    data = pd.read_csv(f'{ticker}_{timeframe}_optimizeData.csv')
    maxpnl = np.max(data['pnl'])
    ind_of_max = data[data['pnl'] == maxpnl].index
    maxinfo = data.iloc[ind_of_max, :]
    short = int(maxinfo['short'])
    long = int(maxinfo['long'])

    same_emas_ind = data[data['short'] == short].index
    same_emas = data.iloc[same_emas_ind, :]
    same_emas_ind = same_emas[same_emas['long'] == long].index
    same_emas = data.iloc[same_emas_ind, :]

    pnl = same_emas['pnl']
    winning = same_emas['winning']
    losing = same_emas['losing']
    takeprof = same_emas['take prof']
    winpct = winning / (winning + losing)
    winpct_aspct = winpct * 100


    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(takeprof, pnl)
    ax[0].scatter(takeprof, pnl)
    ax[0].grid()
    ax[0].set_ylabel('PNL')

    ax[1].plot(takeprof, winpct_aspct)
    ax[1].scatter(takeprof, winpct_aspct)
    ax[1].grid()
    ax[1].set_ylabel('Win Percentage')

    plt.xlabel('Take Prof Value')
    plt.show()

def TOD_profitability(ticker, timeframe, TODfreq = '30min', short = False, long = False):
    if short and long:
        data = opti_df.to_csv(f'{ticker}_{timeframe}_{short}_{long}_{TODfreq}TODopti.csv')

    else:
        data = pd.read_csv(f'{ticker}_{timeframe}_{TODfreq}TODopti.csv')

    pnl = data['pnl']
    TOD = data['TOD']
    fig, ax = plt.subplots(2, sharey = True)
    ax[0].bar(TOD, pnl)
    ax[0].tick_params(axis = 'x', rotation = 45)
    ax[0].grid(axis = 'y')
    ax[0].set_ylabel('PNL')

    sorted = data.sort_values('pnl', ascending = False)
    ax[1].bar(np.arange(len(sorted['pnl'])), sorted['pnl'])
    ax[1].grid(axis = 'y')
    ax[1].set_ylabel('PNL')

    plt.suptitle(f'PNL Analysis for {ticker} {timeframe}, freq = {TODfreq}')
    plt.show()



def full_TOD_analysis(ticker, timeframe, TODfreq = '30min', short = False, long = False):
    if short and long:
        data = pd.read_csv(f'{ticker}_{timeframe}_{short}_{long}_{TODfreq}TODopti.csv')

    else:
        data = pd.read_csv(f'{ticker}_{timeframe}_{TODfreq}TODopti.csv')

    TOD = data['TOD']
    fig, ax = plt.subplots(3, 2)
    plt.subplots_adjust(top = .95, bottom = .05, left = .05, right = .95, hspace = .3)

    ax[0,0].bar(TOD, data['avgwin'], color = 'green' , label = 'Max Win')
    ax[0,0].bar(TOD, data['avgloss'], color = 'red' , label = 'Max Loss')
    ax[0,0].grid(axis = 'y')
    ax[0,0].legend()
    ax[0,0].tick_params(axis = 'x', rotation = 45)
    ax[0,0].set_title('Wins and Losses')

    ax[0,1].bar(TOD, data['numtrades'])
    ax[0,1].grid(axis = 'y')
    ax[0,1].tick_params(axis = 'x', rotation = 45)
    ax[0,1].set_title('Number of Trades since 6/10/2020')

    ax[1,0].bar(TOD, data['drawdown'], color = 'red')
    ax[1,0].grid(axis='y')
    ax[1,0].tick_params(axis='x', rotation=45)
    ax[1,0].set_title('Max Drawdown')

    ax[1,1].bar(TOD, data['winpct'] * 100)
    ax[1,1].grid(axis='y')
    ax[1,1].tick_params(axis='x', rotation=45)
    ax[1,1].set_title('Win Percentage')

    ax[2, 1].bar(TOD, data['expected'])
    ax[2, 1].grid(axis='y')
    ax[2, 1].tick_params(axis='x', rotation=45)
    ax[2, 1].set_title('Expected Value')

    ax[2, 0].bar(TOD, data['pnl'], color = 'green', label = f'Total: {data.pnl.sum()}')
    ax[2, 0].grid(axis='y')
    ax[2, 0].tick_params(axis='x', rotation=45)
    ax[2, 0].set_title('PNL')
    ax[2, 0].legend()


    plt.suptitle(f'Full Analysis for {ticker} {timeframe}, freq = {TODfreq}')
    plt.show()


full_TOD_analysis('BTCUSDT', '5min')