import pandas as pd
import sys


def fix_format(ticker, timeframe, shortened = False):
    data = pd.read_csv(f'{ticker}_{timeframe}.csv')
    data.rename(columns = {'timestamp' : 'Time',
                           'open' :'Open',
                           'high' : 'High',
                           'low' : 'Low',
                           'close' : 'Close',
                           'volume' : 'Volume'}, inplace = True)

    newdts = []

    for i ,item in enumerate(data['Time']):
        yr = item[:4]
        mo = item[5:7]
        day = item[8:10]
        t = item[11:16]

        if mo[0] == '0':
            mo = mo[-1]
        if day[0] == '0':
            day = day[-1]
        if t[0] == '0':
            t = t[1:]


        newdt = mo + '/' + day + '/' + yr + ' ' + t
        newdts.append(newdt)

    data['Time'] = newdts

    if shortened:
        bars = int(-250000 / int(timeframe[0]))
        data = data.iloc[bars:, :]
        data.to_csv(f'{ticker}_{timeframe[0]}min_short.csv')

    else:
        data.to_csv(f'{ticker}_{timeframe[0]}min.csv')

