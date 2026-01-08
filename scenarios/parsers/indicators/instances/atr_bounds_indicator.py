import numpy as np
import pandas as pd
from datetime import datetime
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator


class AtrBoundsIndicator(Indicator):
    """
    Индикатор ATR-based bounds (верхняя и нижняя границы на основе ATR).
    Логика расчёта ATR полностью сохранена из оригинального кода.
    Адаптирован под исторический и реал-тайм режимы.
    """

    def __init__(self, history_market_parser: HistoryMarketParser, period=3, multiplier=2.5):
        super().__init__(history_market_parser)
        self.period = period
        self.multiplier = multiplier

        # Текущие значения (на "последнюю" свечу)
        self.bounds = {
            'upper': np.nan,
            'lower': np.nan,
        }

        self.bounds2 = {
            'upper': np.nan,
            'lower': np.nan,
        }

        # Полные исторические серии (для исторического режима)
        self._historical_upper = None  # pd.Series
        self._historical_lower = None  # pd.Series

    def get_atr_bounds(self, df: pd.DataFrame):
        """
        Вычисляет ATR и границы точно так же, как в оригинальном коде.
        Возвращает два np.array: upper и lower той же длины, что и df.
        """
        if df.empty or len(df) == 0:
            return np.array([]), np.array([])

        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()

        previous_close = np.full_like(close, np.nan)
        previous_close[1:] = close[:-1]

        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - previous_close),
                                   np.abs(low - previous_close)))
        tr[0] = high[0] - low[0]  # первая свеча — просто high - low

        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i - 1] * (self.period - 1) + tr[i]) / self.period

        scaled_atr = atr * self.multiplier
        upper = close + scaled_atr
        lower = close - scaled_atr

        return upper, lower

    def prepare(self, start_time=None, end_time=None):
        """
        Подготовка для исторического режима:
        Вычисляем ATR bounds на всём history_df один раз.
        """
        df = self.history_market_parser.history_df

        if df is None or df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            empty_idx = pd.Index([])
            self._historical_upper = pd.Series([], index=empty_idx)
            self._historical_lower = pd.Series([], index=empty_idx)
            return

        upper_np, lower_np = self.get_atr_bounds(df)

        self._historical_upper = pd.Series(upper_np, index=df.index)
        self._historical_lower = pd.Series(lower_np, index=df.index)

    def run_historical(self, start_time: datetime, current_time: datetime):
        """
        Для исторического режима: берём значения upper/lower на последней свече <= current_time.
        """
        if (self._historical_upper is None or self._historical_upper.empty or
                self._historical_lower is None or self._historical_lower.empty):
            self.bounds = {'upper': np.nan, 'lower': np.nan}
            self.bounds2 = {'upper': np.nan, 'lower': np.nan}
            return

        time_col = pd.to_datetime(self.history_market_parser.history_df['time'])
        mask = time_col <= current_time

        if not mask.any():
            self.bounds = {'upper': np.nan, 'lower': np.nan}
            self.bounds2 = {'upper': np.nan, 'lower': np.nan}
            return

        last_idx = time_col[mask].index[-1]
        last_idx2 = time_col[mask].index[-2]

        self.bounds = {
            'upper': float(self._historical_upper.loc[last_idx]),
            'lower': float(self._historical_lower.loc[last_idx]),
        }

        self.bounds2 = {
            'upper': float(self._historical_upper.loc[last_idx2]),
            'lower': float(self._historical_lower.loc[last_idx2]),
        }

    def run_realtime(self):
        """
        Для реального времени: вычисляем на текущем df и берём последние значения.
        """
        df = self.history_market_parser.df

        if df is None or df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            self.bounds = {'upper': np.nan, 'lower': np.nan}
            return

        upper_np, lower_np = self.get_atr_bounds(df)

        if len(upper_np) > 0 and len(lower_np) > 0:
            self.bounds = {
                'upper': float(upper_np[-1]),
                'lower': float(lower_np[-1]),
            }
        else:
            self.bounds = {'upper': np.nan, 'lower': np.nan}

        if len(upper_np) > 1 and len(lower_np) > 1:
            self.bounds2 = {
                'upper': float(upper_np[-2]),
                'lower': float(lower_np[-2]),
            }
        else:
            self.bounds2 = {'upper': np.nan, 'lower': np.nan}