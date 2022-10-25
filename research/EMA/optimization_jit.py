@jit
def make_period_pairs(short_range=[9, 14, 20, 30, 50], long_range=[20, 30, 50, 100]):
    pairs = []

    for short in short_range:
        for long in long_range:
            if long <= short:
                continue
            else:
                pairs.append([short, long])

    return pairs


@jit
def find_best_pair(ticker, timeframe):
    master = pd.DataFrame(columns=['pair', 'pnl', 'num trades'])
    pairs = make_period_pairs()
    for pair in pairs:
        pnl, num = simulate(ticker, timeframe, pair[0], pair[1])
        master = master.append({'pair': pair, 'pnl': pnl, 'num trades': num}, ignore_index=True)

    max = np.max(master['pnl'])

    ind_of_max = (master[master['pnl'] == max].index)
    print(master.iloc[ind_of_max, :])


@jit
def optimize_strategy(ticker, timeframe, short_range=[9, 14, 20, 30, 50], long_range=[20, 30, 50, 100],
                      prof_range=[]):
    master = pd.DataFrame(columns=['short', 'long', 'pnl', 'winning', 'losing', 'take prof'])
    pairs = make_period_pairs(short_range, long_range)

    for take_prof in prof_range:

        for pair in pairs:
            pnl, num, buys, sells = simulate(ticker, timeframe, pair[0], pair[1], take_prof)
            master = master.append(
                {'short': pair[0], 'long': pair[1], 'pnl': pnl, 'winning': num[0], 'losing': num[1],
                 'take prof': take_prof}, ignore_index=True)

    master.to_csv(f'{ticker}_{timeframe}_optimizeData.csv')
    maxpnl = np.max(master['pnl'])
    ind_of_max = master[master['pnl'] == maxpnl].index
    max = master.iloc[ind_of_max, :]
    print(f'The optimal strategy for {ticker} {timeframe} is:', max)

    return master


@jit
def time_specific_master_optimization(data, time, short_range=[9, 14, 20, 30, 50], long_range=[20, 30, 50, 100],
                                      prof_range=[]):
    master = pd.DataFrame(columns=['time', 'short', 'long', 'pnl', 'winning', 'losing', 'take prof'])
    pairs = make_period_pairs(short_range, long_range)

    for timeframe in timeframes:

        for take_prof in prof_range:

            for pair in pairs:

                try:
                    pnl, num, buys, sells = time_specific_simulate(data, pair[0], pair[1], take_prof)
                    master = master.append({'time': time, 'short': pair[0], 'long': pair[1], 'pnl': pnl,
                                            'winning': num[0], 'losing': num[1], 'take prof': take_prof},
                                           ignore_index=True)

                except:
                    print(f'{ticker} or {timeframe} data not found.')

    maxpnl = np.max(master['pnl'])
    ind_of_max = master[master['pnl'] == maxpnl].index
    max = master.iloc[ind_of_max, :]
    print(f'The optimal strategy for {time} and {ticker} is:', max)

    return master


@jit
def find_maxpnl(ticker):
    data = pd.read_csv(f'{ticker}_allCombinations.csv')
    maxpnl = np.max(data['pnl'])
    ind_of_max = data[data['pnl'] == maxpnl].index
    maxinfo = data.iloc[ind_of_max, :]
    print(maxinfo)

    return maxinfo


#
# for time in [1,5,15,30]:
#     for long in [20,50]:
#         pnl, num, buys, sells = simulate('ESU20', f'{time}min',  9, long)
#         print(f'pnl for {time} min {long} is {pnl} with {num} trades')
#


@jit
def configure_prof_range(timeframe, price):
    time_to_pct = {'1min': .0015,
                   '5min': .0025,
                   '15min': .0031,
                   '30min': .0035,
                   '60min': .04}

    if timeframe[-1] != 'n':
        timeframe = timeframe[:4]

    pct_prof = time_to_pct[timeframe]
    prof_mid = price * pct_prof
    prof_start = np.floor(prof_mid / 2)
    prof_end = np.floor(prof_mid * 1.5)
    prof_step = np.floor((prof_end - prof_start) / 8)
    prof_range = np.arange(prof_start, prof_end + 1, 1)

    return prof_range


@jit
def master_optimization(ticker, timeframes=['1min', '5min', '15min', '30min', '60min'], short_range=[9, 14, 20, 30, 50],
                        long_range=[20, 30, 50, 100], prof_range=[]):
    data = pd.read_csv(f'{ticker}_{timeframes[0]}.csv')

    example_price = data.Open[0]
    master = pd.DataFrame(columns=['timeframe', 'short', 'long', 'pnl', 'winning', 'losing', 'take prof'])
    pairs = make_period_pairs(short_range, long_range)

    for timeframe in timeframes:

        for take_prof in prof_range:

            for pair in pairs:

                try:
                    pnl, num, buys, sells = simulate(ticker, timeframe, pair[0], pair[1], take_prof)
                    master = master.append({'timeframe': timeframe, 'short': pair[0], 'long': pair[1], 'pnl': pnl,
                                            'winning': num[0], 'losing': num[1], 'take prof': take_prof},
                                           ignore_index=True)

                except:
                    print(f'{ticker} or {timeframe} data not found.')

    master.to_csv(f'{ticker}_allCombinations.csv')
    maxpnl = np.max(master['pnl'])
    ind_of_max = master[master['pnl'] == maxpnl].index
    max = master.iloc[ind_of_max, :]
    print(f'The optimal strategy for {ticker} is:', max)

    return master


@jit
def find_best_pnl(ticker):
    data = pd.read_csv(f'{ticker}_allCombinations.csv')
    pairs = []

    for row in data.iterrows():
        ind = row[0]
        row = row[1]
        pairs.append((row['short'], row['long']))

    data['pair'] = pairs

    unique_pairs = pd.Series(pairs).unique()

    master = pd.DataFrame(columns=['pair', 'pnl', 'winpct', 'numtrades', 'take_prof'])
    for pair in unique_pairs:
        pair_data = data[data['pair'] == pair]
        max_pnl = pair_data[pair_data['pnl'] == np.max(pair_data['pnl'])]

        winpct = max_pnl['winning'] / (max_pnl['winning'] + max_pnl['losing'])
        numtrades = max_pnl['winning'] + max_pnl['losing']
        master = master.append({'pair': pair,
                                'pnl': max_pnl['pnl'],
                                'winpct': winpct,
                                'numtrades': numtrades,
                                'take_prof': max_pnl['take prof']}, ignore_index=True)

    master.to_csv(f'{ticker}_best_pnl_options.csv')
    return master


@jit
def find_returns(master, take_prof):
    returns = pd.DataFrame(columns=['type', 'val'])
    for row in master.iterrows():
        ind = row[0]
        row = row[1]

        if row['max_win'] > row['cross_val'] * take_prof:
            returns = returns.append({'type': 1,
                                      'val': row['cross_val'] * (take_prof - .075 / 100)}, ignore_index=True)

        else:
            r = (row['cross_val'] - row['next_cross']) if row['direction'] == 'down' else (
                    row['next_cross'] - row['cross_val'])
            r -= row['cross_val'] * (.075 / 100)
            returns = returns.append({'type': -1,
                                      'val': r}, ignore_index=True)

    master['returns_type'] = returns['type']
    master['returns_val'] = returns['val']

    return master


global all_times30
global all_times60
global all_timesdt30
global all_timesdt60

all_times30 = ['00:00:00', '00:30:00', '01:00:00', '01:30:00', '02:00:00', '02:30:00', '03:00:00', '03:30:00',
               '04:00:00',
               '04:30:00', '05:00:00', '05:30:00', '06:00:00', '06:30:00', '07:00:00', '07:30:00', '08:00:00',
               '08:30:00',
               '09:00:00', '09:30:00', '10:00:00', '10:30:00', '11:00:00', '11:30:00', '12:00:00', '12:30:00',
               '13:00:00',
               '13:30:00', '14:00:00', '14:30:00', '15:00:00', '15:30:00', '16:00:00', '16:30:00', '17:00:00',
               '17:30:00',
               '18:00:00', '18:30:00', '19:00:00', '19:30:00', '20:00:00', '20:30:00', '21:00:00', '21:30:00',
               '22:00:00',
               '22:30:00', '23:00:00', '23:30:00']
all_timesdt30 = [dt.time(int(c[:2]), int(c[3:5])) for c in all_times30]

all_times60 = ['00:00:00', '01:00:00', '02:00:00', '03:00:00', '04:00:00', '05:00:00', '06:00:00', '07:00:00',
               '08:00:00',
               '09:00:00', '10:00:00', '11:00:00', '12:00:00', '13:00:00', '14:00:00', '15:00:00', '16:00:00',
               '17:00:00',
               '18:00:00', '19:00:00', '20:00:00', '21:00:00', '22:00:00', '23:00:00']
all_timesdt60 = [dt.time(int(c[:2]), int(c[3:5])) for c in all_times60]


@jit(nopython=True)
def TOD_optimization(ticker, timeframe, TODfreq='30min', short_range=[9, 14, 20, 30, 50], long_range=[20, 30, 50, 100],
                     prof_range=[]):
    pairs = make_period_pairs(short_range, long_range)

    if TODfreq == '30min':
        all_times = all_times30
        all_timesdt = all_timesdt30
    else:
        all_times = all_times60
        all_timesdt = all_timesdt60

    all_timesdt = [dt.time(int(c[:2]), int(c[3:5])) for c in all_times]

    master_df = pd.DataFrame(columns=all_times)
    master_df_index = []
    timeframe_dict = {}

    for t in all_times:
        timeframe_dict[t] = pd.DataFrame()

    for pair in pairs:
        pdata, splits = prep_data(ticker, timeframe, pair[0], pair[1])

        TOD = []
        for t in pdata['time']:
            t = t[-5:]

            if t[0] == '0':
                t = dt.time(int(t[1]), int(t[3:]))
            else:
                t = dt.time(int(t[:2]), int(t[3:]))

            if t >= all_timesdt[-1]:
                t = all_timesdt[-1]

            else:
                for i, times in enumerate(all_timesdt):
                    if times < t < all_timesdt[i + 1]:
                        t = times

            TOD.append(t)

        pdata['TOD'] = TOD

        for take_prof in prof_range:
            print('started ' + pair + ', ' + take_prof + '...')

            pdata = find_returns(pdata, take_prof)

            for tod in all_timesdt:

                curr_tod = pdata[pdata['TOD'] == tod]

                if not len(curr_tod):
                    wins, losses, numtrades, winpct, max_drawdown = [0], [0], 0, 0, 0

                else:
                    wins = curr_tod[curr_tod['returns_type'] == 1].returns_val.tolist()
                    losses = curr_tod[curr_tod['returns_type'] == -1].returns_val.tolist()

                    numtrades = len(curr_tod)
                    winpct = len(wins) / len(curr_tod)
                    max_drawdown = np.min(curr_tod['max_loss'])

                avgwin = np.mean(wins) if len(wins) else 0
                avgloss = np.mean(losses) if len(losses) else 0
                timeframe_dict[str(tod)] = timeframe_dict[str(tod)].append({'pnl': np.sum(curr_tod['returns_val']),
                                                                            'short': pair[0],
                                                                            'long': pair[1],
                                                                            'take_prof': take_prof,
                                                                            'avgwin': avgwin,
                                                                            'avgloss': avgloss,
                                                                            'numtrades': numtrades,
                                                                            'winpct': winpct,
                                                                            'drawdown': max_drawdown},
                                                                           ignore_index=True)

    master_df['labels'] = master_df_index
    opti_df = pd.DataFrame(
        columns=['TOD', 'pnl', 'short', 'long', 'take_prof', 'avgwin', 'avgloss', 'numtrades', 'winpct', 'drawdown',
                 'expected'])

    # for i, col in enumerate(master_df.columns):
    #     colmax = np.max(master_df[col]
    #     opti_df[master_df_index[i]] = colmax

    for tod in all_times:
        curr_df = timeframe_dict[tod]
        maxpnl = np.max(curr_df['pnl'])

        if maxpnl == 0:
            print(tod + 'not valid')
            continue

        maxind = curr_df[curr_df['pnl'] == maxpnl].index

        if len(maxind) > 1:
            newcurr_df = curr_df.loc[maxind, :].sort_values('drawdown', ascending=True)
            maxind = newcurr_df.index[0]

        short = curr_df.loc[maxind, 'short']
        long = curr_df.loc[maxind, 'long']
        take_prof = curr_df.loc[maxind, 'take_prof']
        avgwin = curr_df.loc[maxind, 'avgwin']
        avgloss = curr_df.loc[maxind, 'avgloss']
        numtrades = curr_df.loc[maxind, 'numtrades']
        winpct = curr_df.loc[maxind, 'winpct']
        drawdown = curr_df.loc[maxind, 'drawdown']
        expected = avgwin * winpct + avgloss * (1 - winpct)

        opti_df = opti_df.append({'TOD': str(tod),
                                  'pnl': float(maxpnl),
                                  'short': float(short),
                                  'long': float(long),
                                  'take_prof': float(take_prof),
                                  'avgwin': float(avgwin),
                                  'avgloss': float(avgloss),
                                  'numtrades': float(numtrades),
                                  'winpct': float(winpct),
                                  'drawdown': float(drawdown),
                                  'expected': float(expected)}, ignore_index=True)

    # if len(short_range) == 1 and len(long_range) == 1:
    #     opti_df.to_csv(f'{ticker}_{timeframe}_{short_range[0]}_{long_range[0]}_{TODfreq}TODopti.csv')
    #
    # else:
    #     opti_df.to_csv(f'{ticker}_{timeframe}_{TODfreq}TODopti.csv')


start = timer()
TOD_optimization('BTCUSDT', '5min', '30min', [9], [20], [20 / 10000])
print('using CPU', timer() - start)

start = timer()
TOD_optimization('BTCUSDT', '5min', '30min', [9], [20], [20 / 10000])
print('using GPU', timer() - start)
#
#
# TOD_optimization('BTCUSDT', '5min', '30min', [9,14,20,30,50], [20,30,50,100], np.linspace(6/10000, 30/10000, num = 20))