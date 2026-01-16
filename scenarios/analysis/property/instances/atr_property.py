from scenarios.analysis.property.abstracts.property import Property
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
import numpy as np
import pandas as pd


class AtrProperty(Property):
    def __init__(self, minutes):
        super().__init__()
        self.minutes = minutes
        self.history_market_parser: HistoryBinanceParser = None
        self.atr_indicator: AtrBoundsIndicator = None

    def prepare(self, deal):
        self.history_market_parser = HistoryBinanceParser('BTC', 'USDT', self.minutes, 1000, 'generating')
        self.atr_indicator = AtrBoundsIndicator(self.history_market_parser, period=3, multiplier=2.5)
        self.market_processes = [
            self.history_market_parser,
            self.atr_indicator
        ]

    def _get_atr_series(self):
        """
        Вычисляет серию значений ATR на основе DataFrame парсера.
        Копирует логику расчета ATR из индикатора.
        """
        df = self.history_market_parser.df
        if df is None or df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            return pd.Series()

        high = df['high'].to_numpy()
        low = df['low'].to_numpy()
        close = df['close'].to_numpy()

        previous_close = np.full_like(close, np.nan)
        previous_close[1:] = close[:-1]

        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - previous_close),
                                   np.abs(low - previous_close)))
        tr[0] = high[0] - low[0]

        atr = np.zeros_like(tr)
        atr[0] = tr[0]
        period = self.atr_indicator.period
        for i in range(1, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        return pd.Series(atr, index=df.index)

    def _get_atr_value(self) -> float:
        """
        Возвращает текущее значение ATR (на последней свече).
        """
        atr_series = self._get_atr_series()
        if atr_series.empty:
            return np.nan
        return float(atr_series.iloc[-1])

    def _get_atr_percent_of_price(self) -> float:
        """
        Возвращает ATR / цена (close последней свечи).
        """
        atr = self._get_atr_value()
        if np.isnan(atr):
            return np.nan
        df = self.history_market_parser.df
        if df.empty:
            return np.nan
        close = float(df['close'].iloc[-1])
        if close == 0:
            return np.nan
        return atr / close

    def _get_atr_slope(self, n: int = 5) -> float:
        """
        Вычисляет изменение (наклон) ATR за последние N баров.
        Slope = (atr[-1] - atr[-N]) / (N - 1)
        Возвращает 0.0, если недостаточно данных.
        """
        atr_series = self._get_atr_series()
        if len(atr_series) < n:
            return 0.0
        atr_last = atr_series.iloc[-1]
        atr_prev = atr_series.iloc[-n]
        slope = (atr_last - atr_prev) / (n - 1)
        return float(slope)

    def _get_current_range_to_atr(self) -> float:
        """
        Вычисляет диапазон текущей свечи / ATR.
        Диапазон = high[-1] - low[-1]
        """
        df = self.history_market_parser.df
        if df is None or df.empty:
            return np.nan
        high = float(df['high'].iloc[-1])
        low = float(df['low'].iloc[-1])
        atr = self._get_atr_value()
        if np.isnan(atr) or atr == 0:
            return np.nan
        return (high - low) / atr

    def _get_volatility_regime(self) -> str:
        """
        Определяет режим волатильности на основе ATR / price.
        - low: < 0.005
        - normal: 0.005 - 0.015
        - high: > 0.015
        Пороги выбраны ориентировочно для BTC/USDT на разных таймфреймах.
        """
        percent = self._get_atr_percent_of_price()
        if np.isnan(percent):
            return 'unknown'
        if percent < 0.005:
            return 'low'
        elif percent < 0.015:
            return 'normal'
        else:
            return 'high'

    def analyze(self, deal):
        deal[f'a{str(self.minutes)}_v'] = float(self._get_atr_value())
        deal[f'a{str(self.minutes)}_p'] = int(self._get_atr_percent_of_price())
        deal[f'a{str(self.minutes)}_sl'] = float(self._get_atr_slope(n=5))
        deal[f'a{str(self.minutes)}_r'] = float(self._get_current_range_to_atr())
        deal[f'a{str(self.minutes)}_vr'] = str(self._get_volatility_regime())
        return deal