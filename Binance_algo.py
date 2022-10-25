from binance.client import Client
from binance.websockets import BinanceSocketManager
from binance.enums import *
import urllib3
import datetime as dt
from helpers.algo_helpers import get_all_times, get_ema_params
import pandas as pd
import numpy as np
from helpers import config
from helpers.mailer import send_email
import sys
import time
from functools import partial
urllib3.disable_warnings()


# client = Client(config.API_KEY, config.API_SECRET, tld = 'us') # {'verify' : False, 'timeout' : 20}
# try:
#     past_klines = pd.read_csv('BTCUSDT_5min_klines.csv')
#
# except:
#
#     past_klines = pd.DataFrame({'time':0, 'open':0, 'high':0, 'low':0, 'close':0}, index = [0])
#


class MyAlgo(Client):

    def __init__(self, tickers_timeframes, TOD_freq):
        Client.__init__(self, config.API_KEY, config.API_SECRET, tld = 'us') # {'verify' : False, 'timeout' : 20}
        self.tickers = []
        self.timeframes = {}
        self.current_price = {}
        self.all_ema_params = {}
        self.curr_ema_params = {}
        self.TOD = dt.datetime.today().strftime('%H:%M')
        self.all_timesstr, self.all_timesdt = get_all_times(TOD_freq)
        self.master_ema_df = pd.DataFrame(columns = ['ticker', 'timeframe', 'short', 'long'])
        self.side = {}
        self.positions = {}
        self.orderId_positions = {}
        self.dir = {}
        self.position_scalers = {}
        self.base_position = {}
        self.klines = {}
        self.prof_reference = {}
        self.signal = {}
        self.update_dir = False

        myTradeSocket = BinanceSocketManager(self)

        for pair in tickers_timeframes:
            if pair[0] not in self.tickers:
                self.tickers.append(pair[0])
            if pair[0] not in self.timeframes.keys():
                self.timeframes[pair[0]] = [pair[1]]
            else:
                self.timeframes[pair[0]].append(pair[1])

        for ticker in self.tickers:
            tradeSocket = myTradeSocket.start_trade_socket(ticker, partial(self.store_price, ticker=ticker))
            for timeframe in self.timeframes[ticker]:
                # if dt.datetime.fromtimestamp(past_klines.iloc[-1, 0]).strftime('%H:%M') == dt.datetime.today().strftime('%H:%M'):
                #     self.klines['BTCUSDT_5min'] = past_klines
                #     print('pulled ', len(past_klines), ' klines from storage' )
                #
                # else:
                #     self.klines[ticker + '_' + timeframe] = pd.DataFrame(columns = ['time', 'open', 'high', 'low', 'close'])
                self.signal[ticker + '_' + timeframe] = None
                self.klines[ticker + '_' + timeframe] = self.get_hist_klines(ticker, timeframe, 100)
                self.positions[ticker + '_' + timeframe] = 0
                self.prof_reference[ticker + '_' + timeframe] = 0
                curr_ema_dict = get_ema_params(ticker,timeframe)
                for TOD in self.all_timesstr:
                    self.all_ema_params[f'{ticker}_{timeframe}_{TOD}'] = curr_ema_dict[TOD]
                myTradeSocket = BinanceSocketManager(self)

                klineSocket = myTradeSocket.start_kline_socket(ticker, callback=partial(self.store_kline_data,
                                                                                        ticker=ticker,
                                                                                        timeframe=timeframe),
                                                               interval=timeframe[:timeframe.index('i')])

                myTradeSocket.start()

    def store_kline_data(self, msg, ticker, timeframe):
        candle = msg['k']
        if candle['x'] == True:
            self.klines[ticker + '_' + timeframe] = self.klines[ticker + '_' + timeframe].append({'time' : candle['t']/1000,
                                                                                                  'open' : float(candle['o']),
                                                                                                  'high' : float(candle['h']),
                                                                                                  'low' : float(candle['l']),
                                                                                                  'close' : float(candle['c'])}, ignore_index = True)
            self.prof_reference[ticker +'_'+ timeframe] = float(candle['c'])

            if len(self.klines[ticker + '_' + timeframe]) > 100:
                self.klines[ticker + '_' + timeframe] = self.klines[ticker + '_' + timeframe].iloc[1:, :]

    def store_price(self, msg, ticker):
        self.current_price[ticker] = msg['p']

    #
    # def start_trade_websocket(self, tickers):
    #     myTradeSocket = BinanceSocketManager(self)
    #     for ticker in tickers:
    #         tradeSocket = myTradeSocket.start_trade_socket(ticker, partial(self.store_price, ticker = ticker))
    #         for timeframe in self.timeframes[ticker]:
    #             klineSocket = myTradeSocket.start_kline_socket(ticker, callback = partial(self.store_kline_data, ticker = ticker, timeframe = timeframe), interval = timeframe[:timeframe.index('i')])
    #
    #             myTradeSocket.start()


    def createOrder(self, ticker, side, type = 'MARKET', quantity = 5, lim_price = False, margin =  False, timeframe = '5min'):
        quantity = 15/float(self.prof_reference[ticker + '_' + timeframe]) * quantity
        quantity = float("{:0.0{}f}".format(quantity, 6))

        if side == 'BUY':
            take_prof = self.prof_reference[ticker + '_' + timeframe] * (1 + self.master_ema_df.loc[ticker + '_' + timeframe, 'take_prof'])
        else:
            take_prof = self.prof_reference[ticker + '_' + timeframe] * (1 - self.master_ema_df.loc[ticker + '_' + timeframe, 'take_prof'])


        take_prof = float(round(take_prof))

        if margin:
            if type == 'LIMIT':
                order = self.create_margin_order(symbol=ticker,
                                                    side=side,
                                                    type=type,
                                                    timeInForce = TIME_IN_FORCE_GTC,
                                                    quantity=quantity,
                                                    price = lim_price)

            elif type == 'MARKET':
                order = self.create_margin_order(symbol=ticker,
                                                    side=side,
                                                    type=type,
                                                    quantity=quantity)

            prof_order = self.create_margin_order(symbol=ticker,
                                                     side=side,
                                                     type='TAKE_PROFIT_LIMIT',
                                                     timeInForce=TIME_IN_FORCE_GTC,
                                                     price=take_prof,
                                                     stopPrice = take_prof,
                                                     quantity=quantity)


        else:
            if type == 'LIMIT':
                order = self.create_order(symbol=ticker,
                                           side=side,
                                           type=type,
                                           timeInForce=TIME_IN_FORCE_GTC,
                                           quantity=quantity,
                                           price=lim_price)
            elif type == 'MARKET':
                order = self.create_order(symbol=ticker,
                                           side=side,
                                           type=type,
                                           quantity=quantity)
            print(order)
            prof_order = self.create_order(symbol=ticker,
                                              side=side,
                                              type='TAKE_PROFIT',
                                              # timeInForce=TIME_IN_FORCE_GTC,
                                              stopPrice=take_prof,
                                              quantity = quantity)
        return order, prof_order



    def cancel_all_orders(self):
        my_open_orders = self.get_open_orders()
        cancellations = []
        for order in my_open_orders:
            cancellations.append(self.cancel_order(my_open_orders[ticker], my_open_orders['orderId']))
        return cancellations


    def get_hist_klines(self, ticker, timeframe, periods):
        interval_dict = {'min' : 'm',
                         'hr' : 'h'}
        time_back_dict = {'m' : 'minutes',
                          'h' : 'hours'}

        period_multiplier = ''
        interval_type = ''
        for element in timeframe:
            try:
                period_multiplier += str(int(element))
            except:
                interval_type += element

        timeback = str(int(period_multiplier) * 100) + ' ' + time_back_dict[interval_dict[interval_type]] + ' ago UTC'
        klines = self.get_historical_klines(ticker, period_multiplier + interval_dict[interval_type], timeback)
        kline_data = pd.DataFrame(columns = ['time', 'open','high','low','close'])
        for kline in klines:
            kline_data = kline_data.append({'time': kline[0] / 1000,
                                             'open': float(kline[1]),
                                             'high': float(kline[2]),
                                             'low': float(kline[3]),
                                             'close': float(kline[4])}, ignore_index = True)
        return kline_data



    def current_ema_params(self):
        for i, t in enumerate(self.all_timesdt):
            if (dt.datetime.today() + dt.timedelta(minutes = 60)).time() >= self.all_timesdt[-1]:
                newTOD = self.all_timesstr[-1]
                break

            elif t <= (dt.datetime.today() + dt.timedelta(minutes = 60)).time() < self.all_timesdt[i + 1]:
                newTOD = self.all_timesstr[i]
                break

        if newTOD != self.TOD:
            self.TOD = newTOD
            self.update_dir = True
            for ticker in self.tickers:
                for timeframe in self.timeframes[ticker]:
                    self.curr_ema_params[f'{ticker}_{timeframe}'] = self.all_ema_params[f'{ticker}_{timeframe}_{self.TOD}']




    def update_emas(self):
        self.master_ema_df = pd.DataFrame(columns=['ticker', 'timeframe', 'short', 'long'])
        # if not len(self.curr_ema_params):
        self.current_ema_params()

        for ticker in self.tickers:
            for timeframe in self.timeframes[ticker]:
                short, long, take_prof, pos_weight = self.curr_ema_params[f'{ticker}_{timeframe}']
                print(short,long, self.TOD)
                closes = self.klines[ticker +'_'+ timeframe].close
                s_ema = closes[-int(short):].mean()
                l_ema = closes[-int(long):].mean()
                print(s_ema, l_ema)

                if self.update_dir:
                    new_dir = 1 if s_ema >= l_ema else -1
                    if len(self.dir):
                        if new_dir < self.dir[ticker + '_' + timeframe]:
                            self.signal[ticker + '_' + timeframe] = 'SELL'
                        elif new_dir > self.dir[ticker + '_' + timeframe]:
                            self.signal[ticker + '_' + timeframe] = 'BUY'


                    self.dir[ticker + '_' + timeframe] = new_dir
                    print(self.signal[ticker + '_' + timeframe])
                    self.update_dir = False

                self.master_ema_df = self.master_ema_df.append({'ticker': ticker,
                                                                'timeframe' : timeframe,
                                                                'short' : s_ema,
                                                                'long' : l_ema,
                                                                'take_prof' : float(take_prof),
                                                                'pos_weight' : pos_weight}, ignore_index = True)

        self.master_ema_df.set_index(self.master_ema_df['ticker'] + '_' + self.master_ema_df['timeframe'], inplace = True)



    def trade_decisions(self):
        if not len(self.master_ema_df):
            print('EMA df initialized')


        if not len(self.dir.keys()):
            for row in self.master_ema_df.iterrows():
                ind = row[0]
                row = row[1]
                self.dir[row['ticker'] + '_' + row['timeframe']] = 1 if row['short'] >= row['long'] else -1
                print('Direction initialized')


        for row in self.master_ema_df.iterrows():
            ind = row[0]
            row = row[1]
            ticker = row['ticker']
            timeframe = row['timeframe']
            curr_dir = 1 if row['short'] >= row['long'] else -1
            print(self.signal[row['ticker'] + '_' + row['timeframe']])
            # self.signal[row['ticker'] + '_' + row['timeframe']] = 'BUY'
            if self.signal[row['ticker'] + '_' + row['timeframe']] != None:
                print('ABOUT TO TRADE')
                side = self.signal[ticker + '_' + timeframe]
                # if f'{ticker}_{timeframe}' not in self.base_position.keys():
                #     quantity = row['pos_weight']
                # else:
                #     quantity = self.base_position[f'{ticker}_{timeframe}'] * row['pos_weight']
                quantity = row['pos_weight']
                if self.positions[ticker + '_' + timeframe] != 0:
                    quantity += self.positions[ticker + '_' + timeframe]

                order = self.createOrder(ticker = ticker, side = side, type = 'MARKET', quantity = float(quantity))
                send_email(message = order, subject=f'{side} ORDER FOR {quantity} EXECUTED')
                print(f'{side} ORDER FOR {quantity} EXECUTED')
                self.side[ticker + '_' + timeframe] = curr_dir
                self.positions[ticker + '_' + timeframe] = quantity if curr_dir > 0 else  -quantity
                self.avg_cost[ticker + '_' + timeframe] = self.current_price[ticker]
                self.orderId_positions[str(order)] = quantity if curr_dir > 0 else -quantity
                self.signal[row['ticker'] + '_' + row['timeframe']] = None



    def close(self):
        for ticker in self.tickers:
            for timeframe in self.timeframes[ticker]:

                if self.positions[ticker + '_' + timeframe] < 0:
                    if self.avg_cost[ticker] - self.current_price[ticker] > self.master_ema_df[ticker + '_' + timeframe].take_prof:
                        quantity = -self.position[ticker + '_' + timeframe]
                        order = self.create_order(ticker, 'BUY', 'MARKET', quantity = quantity)
                        self.positions[row[ticker] + '_' + timeframe] = 0
                        self.avg_cost[row[ticker] + '_' + timeframe] = 0
                        self.orderId_positions[str(order)] = quantity

                elif self.positions[ticker + '_' + timeframe]  > 0:
                    if self.current_price[ticker] - self.avg_cost[ticker] > self.master_ema_df[ticker + '_' + timeframe].take_prof:
                        quantity = self.position[ticker + '_' + timeframe]
                        order = self.create_order(ticker, 'SELL', 'MARKET', quantity=quantity)
                        self.positions[row[ticker] + '_' + timeframe] = 0
                        self.avg_cost[row[ticker] + '_' + timeframe] = 0
                        self.orderId_positions[str(order)] = -quantity


app = MyAlgo([['BTCUSDT', '5min']], '30')


time.sleep(1)
print('Algo started...')

update = True
while True:
    if int(dt.datetime.today().strftime('%M'))%5 ==0:
        if update:
            app.update_emas()

            print(len(app.klines['BTCUSDT_5min']))
            for key, df in app.klines.items():
                df.to_csv(f'{key}_klines.csv')
            update = False
            app.trade_decisions()

    else:
        update = True

