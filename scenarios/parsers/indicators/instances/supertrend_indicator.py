import numpy as np
import pandas as pd
from datetime import datetime
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator


class SupertrendIndicator(Indicator):
    """
    Индикатор Supertrend.
    Адаптирован под исторический и реал-тайм режимы.
    """

    def __init__(self, history_market_parser: HistoryMarketParser, period=10, multiplier=3.0):
        super().__init__(history_market_parser)
        self.period = period
        self.multiplier = multiplier

        # Текущие значения (на "последнюю" свечу)
        self.values = {
            'supertrend': np.nan,
            'direction': np.nan,   # 1 = bullish, -1 = bearish
            'upper_band': np.nan,
            'lower_band': np.nan,
        }

        self.values2 = {
            'supertrend': np.nan,
            'direction': np.nan,
            'upper_band': np.nan,
            'lower_band': np.nan,
        }

        # Полные исторические серии (для исторического режима)
        self._historical_supertrend = None   # pd.Series
        self._historical_direction = None    # pd.Series
        self._historical_upper_band = None   # pd.Series
        self._historical_lower_band = None   # pd.Series

    def get_supertrend(self, df: pd.DataFrame):
        """
        Вычисляет Supertrend.
        Возвращает 4 np.array той же длины, что и df:
        - supertrend
        - direction (1 = bullish, -1 = bearish)
        - final_upperband
        - final_lowerband
        """
        if df.empty or len(df) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        high = df['high'].to_numpy(dtype=float)
        low = df['low'].to_numpy(dtype=float)
        close = df['close'].to_numpy(dtype=float)

        previous_close = np.full_like(close, np.nan)
        previous_close[1:] = close[:-1]

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - previous_close),
                np.abs(low - previous_close)
            )
        )
        tr[0] = high[0] - low[0]

        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i - 1] * (self.period - 1) + tr[i]) / self.period

        hl2 = (high + low) / 2.0
        basic_upperband = hl2 + self.multiplier * atr
        basic_lowerband = hl2 - self.multiplier * atr

        final_upperband = np.copy(basic_upperband)
        final_lowerband = np.copy(basic_lowerband)

        for i in range(1, len(df)):
            if basic_upperband[i] < final_upperband[i - 1] or close[i - 1] > final_upperband[i - 1]:
                final_upperband[i] = basic_upperband[i]
            else:
                final_upperband[i] = final_upperband[i - 1]

            if basic_lowerband[i] > final_lowerband[i - 1] or close[i - 1] < final_lowerband[i - 1]:
                final_lowerband[i] = basic_lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i - 1]

        supertrend = np.full(len(df), np.nan)
        direction = np.full(len(df), np.nan)

        if len(df) > 0:
            if close[0] <= final_upperband[0]:
                supertrend[0] = final_upperband[0]
                direction[0] = -1
            else:
                supertrend[0] = final_lowerband[0]
                direction[0] = 1

        for i in range(1, len(df)):
            if supertrend[i - 1] == final_upperband[i - 1]:
                if close[i] <= final_upperband[i]:
                    supertrend[i] = final_upperband[i]
                    direction[i] = -1
                else:
                    supertrend[i] = final_lowerband[i]
                    direction[i] = 1
            else:
                if close[i] >= final_lowerband[i]:
                    supertrend[i] = final_lowerband[i]
                    direction[i] = 1
                else:
                    supertrend[i] = final_upperband[i]
                    direction[i] = -1

        return supertrend, direction, final_upperband, final_lowerband

    def prepare(self, start_time=None, end_time=None):
        """
        Подготовка для исторического режима:
        Вычисляем Supertrend на всём history_df один раз.
        """
        df = self.history_market_parser.history_df

        if df is None or df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            empty_idx = pd.Index([])
            self._historical_supertrend = pd.Series([], index=empty_idx)
            self._historical_direction = pd.Series([], index=empty_idx)
            self._historical_upper_band = pd.Series([], index=empty_idx)
            self._historical_lower_band = pd.Series([], index=empty_idx)
            return

        supertrend_np, direction_np, upper_band_np, lower_band_np = self.get_supertrend(df)

        self._historical_supertrend = pd.Series(supertrend_np, index=df.index)
        self._historical_direction = pd.Series(direction_np, index=df.index)
        self._historical_upper_band = pd.Series(upper_band_np, index=df.index)
        self._historical_lower_band = pd.Series(lower_band_np, index=df.index)

    def run_historical(self, start_time: datetime, current_time: datetime):
        """
        Для исторического режима: берём значения на последней свече <= current_time.
        """
        if (
            self._historical_supertrend is None or self._historical_supertrend.empty or
            self._historical_direction is None or self._historical_direction.empty or
            self._historical_upper_band is None or self._historical_upper_band.empty or
            self._historical_lower_band is None or self._historical_lower_band.empty
        ):
            self.values = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }
            self.values2 = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }
            return

        time_col = pd.to_datetime(self.history_market_parser.history_df['time'])
        mask = time_col <= current_time

        if not mask.any():
            self.values = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }
            self.values2 = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }
            return

        last_idx = time_col[mask].index[-1]

        self.values = {
            'supertrend': float(self._historical_supertrend.loc[last_idx]),
            'direction': float(self._historical_direction.loc[last_idx]),
            'upper_band': float(self._historical_upper_band.loc[last_idx]),
            'lower_band': float(self._historical_lower_band.loc[last_idx]),
        }

        if mask.sum() > 1:
            last_idx2 = time_col[mask].index[-2]
            self.values2 = {
                'supertrend': float(self._historical_supertrend.loc[last_idx2]),
                'direction': float(self._historical_direction.loc[last_idx2]),
                'upper_band': float(self._historical_upper_band.loc[last_idx2]),
                'lower_band': float(self._historical_lower_band.loc[last_idx2]),
            }
        else:
            self.values2 = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }

    def run_realtime(self):
        """
        Для реального времени: вычисляем на текущем df и берём последние значения.
        """
        df = self.history_market_parser.df

        if df is None or df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            self.values = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }
            self.values2 = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }
            return

        supertrend_np, direction_np, upper_band_np, lower_band_np = self.get_supertrend(df)

        if len(supertrend_np) > 0:
            self.values = {
                'supertrend': float(supertrend_np[-1]),
                'direction': float(direction_np[-1]),
                'upper_band': float(upper_band_np[-1]),
                'lower_band': float(lower_band_np[-1]),
            }
        else:
            self.values = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }

        if len(supertrend_np) > 1:
            self.values2 = {
                'supertrend': float(supertrend_np[-2]),
                'direction': float(direction_np[-2]),
                'upper_band': float(upper_band_np[-2]),
                'lower_band': float(lower_band_np[-2]),
            }
        else:
            self.values2 = {
                'supertrend': np.nan,
                'direction': np.nan,
                'upper_band': np.nan,
                'lower_band': np.nan,
            }