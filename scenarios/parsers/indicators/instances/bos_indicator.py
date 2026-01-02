import numpy as np
import pandas as pd
from datetime import datetime
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator

class BosIndicator(Indicator):
    """
    Детектор bullish BOS (break of structure на покупку) по логике LuxAlgo SMC
    для swing-структуры (internal = false).
    Логика полностью сохранена из оригинального кода — ничего не менялось в расчётах.
    Адаптировано под исторический и реал-тайм режимы через prepare / run_historical / run_realtime.
    """
    _BEARISH_LEG = 0
    _BULLISH_LEG = 1
    _BULLISH = "BULLISH"
    _BEARISH = "BEARISH"

    def __init__(self, history_market_parser: HistoryMarketParser):
        super().__init__(history_market_parser)
        self.swing_size = 5
        # Результат на "текущий" момент (последняя свеча)
        self.is_now_BOS = False # Был ли bullish BOS на последней свече
        self.bos_cross_price = None # Уровень, который пробили при BOS
        # Хранилища для исторического режима: значения по каждой строке df
        self._historical_is_bos = None # pd.Series[bool] — был ли BOS на этой свече
        self._historical_cross_price = None # pd.Series[float] — цена пробоя (если был BOS)
        self._historical_trend_bias = None # pd.Series[str] — bias на конец свечи

    def _compute_full_history(self, df: pd.DataFrame):
        """
        Вычисляет всю структуру свингов и BOS бар за баром.
        Возвращает три серии с индексами df:
            - is_bos: True только на свече, где произошёл bullish BOS
            - cross_price: уровень пробоя (swing_high_current) или NaN
            - trend_bias: финальный bias после обработки свечи
        """
        if df is None or df.empty:
            return pd.Series(), pd.Series(), pd.Series()
        n = len(df)
        size = int(self.swing_size)
        if size < 1 or n <= size + 1:
            empty = pd.Series([False] * n, index=df.index)
            empty_price = pd.Series([np.nan] * n, index=df.index)
            empty_bias = pd.Series([self._BEARISH] * n, index=df.index)
            return empty, empty_price, empty_bias
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        # Состояние
        swing_high_current = None
        swing_high_last = None
        swing_high_crossed = False
        swing_high_index = None
        swing_low_current = None
        swing_low_last = None
        swing_low_crossed = False
        swing_low_index = None
        leg_prev = self._BEARISH_LEG
        trend_bias = self._BEARISH
        # Результаты по свечам
        is_bos_list = [False] * n
        cross_price_list = [np.nan] * n
        trend_bias_list = [self._BEARISH] * n
        for i in range(n):
            # 1) Обновление свингов (leg + getCurrentStructure)
            if i >= size:
                pivot_idx = i - size
                if pivot_idx + 1 <= i:
                    window_high = highs[pivot_idx + 1: i + 1]
                    window_low = lows[pivot_idx + 1: i + 1]
                    new_leg_high = highs[pivot_idx] > window_high.max() if window_high.size > 0 else False
                    new_leg_low = lows[pivot_idx] < window_low.min() if window_low.size > 0 else False
                else:
                    new_leg_high = new_leg_low = False
                leg_val = leg_prev
                if new_leg_high:
                    leg_val = self._BEARISH_LEG
                elif new_leg_low:
                    leg_val = self._BULLISH_LEG
                change = leg_val - leg_prev
                new_pivot = change != 0
                pivot_low = change == +1 # bullish leg start
                pivot_high = change == -1 # bearish leg start
                if new_pivot:
                    if pivot_low:
                        swing_low_last = swing_low_current
                        swing_low_current = lows[pivot_idx]
                        swing_low_crossed = False
                        swing_low_index = pivot_idx
                    else:
                        swing_high_last = swing_high_current
                        swing_high_current = highs[pivot_idx]
                        swing_high_crossed = False
                        swing_high_index = pivot_idx
                leg_prev = leg_val
            # 2) Проверка пробоев (displayStructure)
            bos_this_bar = False
            bos_price_this_bar = np.nan
            # Bullish: пробой swingHigh вверх
            if (
                swing_high_current is not None
                and i > 0
                and not swing_high_crossed
            ):
                level = swing_high_current
                prev_close = closes[i - 1]
                cur_close = closes[i]
                crossed_up = (prev_close < level) and (cur_close > level)
                if crossed_up:
                    tag = "CHOCH" if trend_bias == self._BEARISH else "BOS"
                    if tag == "BOS":
                        bos_this_bar = True
                        bos_price_this_bar = level
                    swing_high_crossed = True
                    trend_bias = self._BULLISH
            # Bearish: пробой swingLow вниз (для корректного обновления bias)
            if (
                swing_low_current is not None
                and i > 0
                and not swing_low_crossed
            ):
                level = swing_low_current
                prev_close = closes[i - 1]
                cur_close = closes[i]
                crossed_down = (prev_close > level) and (cur_close < level)
                if crossed_down:
                    tag = "CHOCH" if trend_bias == self._BULLISH else "BOS"
                    swing_low_crossed = True
                    trend_bias = self._BEARISH
            # Сохраняем результаты для текущей свечи
            is_bos_list[i] = bos_this_bar
            cross_price_list[i] = bos_price_this_bar if bos_this_bar else np.nan
            trend_bias_list[i] = trend_bias
        # Преобразуем в Series с тем же индексом, что и df
        is_bos_series = pd.Series(is_bos_list, index=df.index, dtype=bool)
        cross_price_series = pd.Series(cross_price_list, index=df.index)
        bias_series = pd.Series(trend_bias_list, index=df.index)
        return is_bos_series, cross_price_series, bias_series

    def prepare(self, start_time=None, end_time=None):
        """
        Подготовка для исторического режима: вычисляем BOS на всём history_df один раз.
        """
        df = self.history_market_parser.history_df
        if df is None or df.empty or 'close' not in df.columns:
            empty_idx = pd.Index([])
            self._historical_is_bos = pd.Series([], index=empty_idx, dtype=bool)
            self._historical_cross_price = pd.Series([], index=empty_idx)
            self._historical_trend_bias = pd.Series([], index=empty_idx)
            return
        self._historical_is_bos, self._historical_cross_price, self._historical_trend_bias = \
            self._compute_full_history(df)

    def run_historical(self, start_time: datetime, current_time: datetime):
        """
        Для исторического режима: берём значения на последней свече, где время <= current_time.
        """
        if (self._historical_is_bos is None or self._historical_is_bos.empty):
            self.is_now_BOS = False
            self.bos_cross_price = None
            return
        time_col = pd.to_datetime(self.history_market_parser.history_df['time'])
        mask = time_col <= current_time
        if not mask.any():
            self.is_now_BOS = False
            self.bos_cross_price = None
            return
        last_idx = time_col[mask].index[-1]
        self.is_now_BOS = bool(self._historical_is_bos.loc[last_idx])
        self.bos_cross_price = float(self._historical_cross_price.loc[last_idx]) \
            if not np.isnan(self._historical_cross_price.loc[last_idx]) else None

    def run_realtime(self):
        """
        Для реального времени: полный прогон по текущему df.
        """
        df = self.history_market_parser.df
        if df is None or df.empty or 'close' not in df.columns:
            self.is_now_BOS = False
            self.bos_cross_price = None
            return
        is_bos_series, cross_price_series, _ = self._compute_full_history(df)
        # Берём значения с последней строки
        last_idx = df.index[-1]
        self.is_now_BOS = bool(is_bos_series.loc[last_idx])
        self.bos_cross_price = float(cross_price_series.loc[last_idx]) \
            if not np.isnan(cross_price_series.loc[last_idx]) else None