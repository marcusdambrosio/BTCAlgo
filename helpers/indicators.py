import pandas as pd
import numpy as np


def SMA(data, period):
    return data.rolling(window = period, min_periods = 1).mean()

def EMA(data,period):
    return data.ewm(span = period, min_periods = 1).mean()

def MACD(data, short_period = 12, long_period = 26):
    return data.ewm(span = short_period, min_periods = 1).mean() - data.ewm(span = long_period, min_periods = 1).mean()

def Stochastic(high, low, close):
    highs = high.rolling(window = 14, min_periods = 1).max()
    lows = low.rolling(window = 14, min_periods = 1).max()
    return (close - lows) / (highs - lows) * 100

def RSI(close, period = 14):
    delta = close.diff()

    dUp, dDown = delta.copy(), delta.copy()

    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(period).mean()
    RolDown = dDown.rolling(period).mean().abs()

    RolUp = RolUp.reindex_like(delta, method='ffill')
    RolDown = RolDown.reindex_like(delta, method='ffill')

    RS = RolUp / RolDown
    rsi = 100.0 - (100.0 / (1.0 + RS))
    return rsi

def ichimoku(high, low, close, short_period=9, mid_period=26, long_period=52, offset=26):
    period1_high = high.rolling(short_period).max()
    period1_low = low.rolling(short_period).max()

    tenkan_sen = (period1_high + period1_low) / 2

    period2_high = high.rolling(mid_period).max()
    period2_low = low.rolling(mid_period).max()

    kijun_sen = (period2_high + period2_low) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(offset)

    period3_high = high.rolling(long_period).max()
    period3_low = high.rolling(long_period).max()

    senkou_span_b = ((period3_high + period3_low) / 2).shift(offset)

    chikou_span = close.shift(-offset)


    return [tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span]
