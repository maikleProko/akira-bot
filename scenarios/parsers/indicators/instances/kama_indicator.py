import numpy as np
import pandas as pd
from datetime import datetime
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator


class KamaIndicator(Indicator):
    """
    Индикатор на основе Kaufman's Adaptive Moving Average (KAMA).
    Определяет текущий тренд по направлению KAMA:
        - 'BULLISH' — если KAMA растёт
        - 'BEARISH' — если KAMA падает

    Дополнительно предоставляет тренд на предыдущих двух свечах (trend2 и trend3).
    Логика расчёта полностью соответствует PineScript.
    """

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

        # Текущие значения на "последней" свече
        self.kama_value = np.nan      # Текущее значение KAMA
        self.trend = "BEARISH"        # Тренд текущей свечи
        self.trend2 = "BEARISH"       # Тренд предыдущей свечи
        self.trend3 = "BEARISH"       # Тренд предпредыдущей свечи

        # Полные исторические серии для исторического режима
        self._historical_kama = None   # pd.Series — значение KAMA
        self._historical_trend = None  # pd.Series — тренд по всем свечам

    def _compute_kama(self, close_series: pd.Series):
        """
        Вычисляет KAMA и тренд по серии close.
        Возвращает два pd.Series: kama_values и trend ('BULLISH'/'BEARISH')
        """
        if close_series.empty or len(close_series) < self.length:
            empty_idx = close_series.index[:0]
            return pd.Series([], index=empty_idx), pd.Series([], index=empty_idx)

        src = close_series.copy()

        # --- Efficiency Ratio (ER) ---
        change = src.diff()
        mom = (src - src.shift(self.length)).abs()
        volatility = change.abs().rolling(self.length).sum()

        mom_np = mom.to_numpy()
        vol_np = volatility.to_numpy()
        er_np = np.where(vol_np != 0, mom_np / vol_np, 0.0)

        # --- Smoothing constants ---
        fast_alpha = 2 / (self.fast_length + 1)
        slow_alpha = 2 / (self.slow_length + 1)
        alpha_np = (er_np * (fast_alpha - slow_alpha) + slow_alpha) ** 2

        # --- Рекурсивный расчёт KAMA ---
        kama = np.zeros(len(src))
        kama[0] = src.iloc[0]

        for i in range(1, len(src)):
            if np.isnan(alpha_np[i]):
                kama[i] = kama[i - 1]
            else:
                kama[i] = alpha_np[i] * src.iloc[i] + (1 - alpha_np[i]) * kama[i - 1]

        kama_series = pd.Series(kama, index=src.index)

        # --- Определение тренда ---
        kama_shifted = kama_series.shift(1)
        trend_values = np.where(kama_series > kama_shifted, "BULLISH", "BEARISH")
        if len(trend_values) > 0:
            trend_values[0] = "BEARISH"  # Первая свеча — BEARISH

        trend_series = pd.Series(trend_values, index=src.index)

        return kama_series, trend_series

    def prepare(self, start_time=None, end_time=None):
        """
        Подготовка для исторического режима: вычисляем KAMA и тренд на всём history_df.
        """
        df = self.history_market_parser.history_df

        if df is None or df.empty or 'close' not in df.columns:
            empty_idx = pd.Index([])
            self._historical_kama = pd.Series([], index=empty_idx)
            self._historical_trend = pd.Series([], index=empty_idx)
            return

        close_series = df['close']
        self._historical_kama, self._historical_trend = self._compute_kama(close_series)

    def run_historical(self, start_time: datetime, current_time: datetime):
        """
        Для исторического режима: берём значения на последней свече <= current_time
        и на двух предыдущих (если они есть).
        """
        if (self._historical_kama is None or self._historical_kama.empty or
                self._historical_trend is None or self._historical_trend.empty):
            self.kama_value = np.nan
            self.trend = "BEARISH"
            self.trend2 = "BEARISH"
            self.trend3 = "BEARISH"
            return

        time_col = pd.to_datetime(self.history_market_parser.history_df['time'])
        mask = time_col <= current_time

        if not mask.any():
            self.kama_value = np.nan
            self.trend = "BEARISH"
            self.trend2 = "BEARISH"
            self.trend3 = "BEARISH"
            return

        valid_indices = time_col[mask].index

        last_idx = valid_indices[-1]
        self.kama_value = float(self._historical_kama.loc[last_idx])
        self.trend = str(self._historical_trend.loc[last_idx])

        # Предыдущая свеча
        if len(valid_indices) >= 2:
            self.trend2 = str(self._historical_trend.loc[valid_indices[-2]])
        else:
            self.trend2 = "BEARISH"

        # Предпредыдущая свеча
        if len(valid_indices) >= 3:
            self.trend3 = str(self._historical_trend.loc[valid_indices[-3]])
        else:
            self.trend3 = "BEARISH"

    def run_realtime(self):
        """
        Для реального времени: вычисляем на текущем df и берём последние значения.
        """
        df = self.history_market_parser.df

        if df is None or df.empty or 'close' not in df.columns:
            self.kama_value = np.nan
            self.trend = "BEARISH"
            self.trend2 = "BEARISH"
            self.trend3 = "BEARISH"
            return

        close_series = df['close']
        kama_series, trend_series = self._compute_kama(close_series)

        if kama_series.empty:
            self.kama_value = np.nan
            self.trend = "BEARISH"
            self.trend2 = "BEARISH"
            self.trend3 = "BEARISH"
            return

        indices = kama_series.index
        last_idx = indices[-1]
        self.kama_value = float(kama_series.loc[last_idx])
        self.trend = str(trend_series.loc[last_idx])

        if len(indices) >= 2:
            self.trend2 = str(trend_series.loc[indices[-2]])
        else:
            self.trend2 = "BEARISH"

        if len(indices) >= 3:
            self.trend3 = str(trend_series.loc[indices[-3]])
        else:
            self.trend3 = "BEARISH"