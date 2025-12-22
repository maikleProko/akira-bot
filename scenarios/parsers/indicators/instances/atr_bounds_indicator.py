import numpy as np
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator

class AtrBoundsIndicator(Indicator):
    def __init__(self, history_market_parser: HistoryMarketParser, period=3, multiplier=2.5):
        super().__init__(history_market_parser)
        self.period = period
        self.multiplier = multiplier
        self.bounds = {}

    def get_atr_bounds(self, df):
        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()

        previous_close = np.full_like(close, np.nan)
        previous_close[1:] = close[:-1]

        tr = np.maximum(high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)))
        tr[0] = high[0] - low[0]

        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i - 1] * (self.period - 1) + tr[i]) / self.period

        scaled_atr = atr * self.multiplier
        upper = close + scaled_atr
        lower = close - scaled_atr

        return upper, lower

    def run(self, start_time=None, end_time=None):
        df = self.history_market_parser.df
        upper, lower = self.get_atr_bounds(df)
        self.bounds = {
            'upper': upper[-1],
            'lower': lower[-1],
        }


