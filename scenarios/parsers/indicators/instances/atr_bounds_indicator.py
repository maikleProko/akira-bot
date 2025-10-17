from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator
import pandas as pd
import numpy as np


class AtrBoundsIndicator(Indicator):

    def __init__(self, history_market_parser: HistoryMarketParser, atr_period: int = 3, atr_multiplier: float = 2.5):
        super().__init__(history_market_parser)
        self.bounds = {}
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    @staticmethod
    def get_atr_compute(dataframe: pd.DataFrame, atr_period: int = 3, atr_multiplier: float = 2.5):
        try:
            # Ensure the DataFrame has the required columns
            if not all(col in dataframe.columns for col in ['high', 'low', 'close']):
                raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")

            # Calculate True Range (TR)
            high = dataframe['high']
            low = dataframe['low']
            close = dataframe['close']
            close_shifted = close.shift(1)  # Previous close for TR calculation

            # True Range = max(high - low, abs(high - close[1]), abs(low - close[1]))
            tr = pd.DataFrame({
                'tr1': high - low,
                'tr2': (high - close_shifted).abs(),
                'tr3': (low - close_shifted).abs()
            }).max(axis=1)

            # Calculate ATR using RMA (TradingView's ATR uses RMA)
            alpha = 1 / atr_period
            atr = np.zeros(len(tr))
            atr[0] = tr.iloc[0] if not pd.isna(tr.iloc[0]) else 0

            # RMA: atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
            for i in range(1, len(tr)):
                if not pd.isna(tr.iloc[i]):
                    atr[i] = alpha * tr.iloc[i] + (1 - alpha) * atr[i - 1]
                else:
                    atr[i] = atr[i - 1]  # Carry forward last valid ATR if TR is NaN

            # Convert ATR to pandas Series for calculations
            atr = pd.Series(atr, index=dataframe.index)

            # Calculate scaled ATR
            scaled_atr = atr * atr_multiplier

            # Calculate upper and lower ATR bands using close as the source
            upper = close + scaled_atr
            lower = close - scaled_atr

            # Convert to lists and handle NaN values
            upper = upper.fillna(0).tolist()
            lower = lower.fillna(0).tolist()

            return upper, lower
        except Exception as e:
            print(f"Error in get_atr_compute: {e}")
            return [], []

    def run(self, start_time=None, end_time=None):
        df = self.history_market_parser.df
        upper, lower = self.get_atr_compute(df, self.atr_period, self.atr_multiplier)
        self.bounds = {
            'upper': float(upper[-1]) if upper else 0,
            'lower': float(lower[-1]) if lower else 0,
        }


