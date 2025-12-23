import numpy as np
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator


class KamaIndicator(Indicator):
    def __init__(
        self,
        history_market_parser: HistoryMarketParser,
        length: int = 14,
        fast_length: int = 2,
        slow_length: int = 30
    ):
        super().__init__(history_market_parser)
        self.length = length
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.value = None
        self.is_bearish_kamas = []

    def get_kama(self, df):
        close = df['close'].to_numpy(dtype=float)
        n = len(close)

        # --- Momentum ---
        mom = np.zeros(n)
        mom[self.length:] = np.abs(close[self.length:] - close[:-self.length])

        # --- Volatility ---
        diff = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.zeros(n)
        for i in range(self.length, n):
            volatility[i] = diff[i - self.length + 1:i + 1].sum()

        # --- Efficiency Ratio ---
        er = np.zeros(n)
        nonzero = volatility != 0
        er[nonzero] = mom[nonzero] / volatility[nonzero]

        # --- Alphas ---
        fast_alpha = 2 / (self.fast_length + 1)
        slow_alpha = 2 / (self.slow_length + 1)
        alpha = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

        # --- KAMA ---
        kama = np.zeros(n)
        kama[0] = close[0]

        for i in range(1, n):
            kama[i] = alpha[i] * close[i] + (1 - alpha[i]) * kama[i - 1]

        return kama

    def run(self, start_time=None, end_time=None):
        df = self.history_market_parser.df
        kama = self.get_kama(df)

        self.is_bearish_kamas = []

        for i in range(-20, 0):
            self.is_bearish_kamas.append(df['close'][len(df['close'])-1+i] > kama[i])
