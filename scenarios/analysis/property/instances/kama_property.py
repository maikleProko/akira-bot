from scenarios.analysis.property.abstracts.property import Property
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
import numpy as np


class KamaProperty(Property):
    def __init__(self, minutes: int):
        super().__init__()
        self.minutes = minutes
        self.history_market_parser: HistoryBinanceParser = None
        self.kama_indicator: KamaIndicator = None

    def prepare(self, deal):
        self.history_market_parser = HistoryBinanceParser(
            'BTC', 'USDT', self.minutes, 1000, mode='generating'
        )
        self.kama_indicator = KamaIndicator(
            self.history_market_parser, 7, 2, 30
        )
        self.market_processes = [
            self.history_market_parser,
            self.kama_indicator
        ]

    # ──────────────────────────────────────────────────────────────
    # Наклон KAMA за последние N баров (линейная регрессия)
    # ──────────────────────────────────────────────────────────────
    def _get_kama_slope(self, lookback: int = 8) -> float:
        """
        Возвращает наклон KAMA за последние lookback баров (в пунктах за бар)
        Положительный → растёт, отрицательный → падает
        """
        if not hasattr(self.kama_indicator, 'kama') or len(self.kama_indicator.kama) < lookback:
            return 0.0

        # Берём последние lookback значений KAMA
        recent_kama = np.array(self.kama_indicator.kama[-lookback:])

        # x — просто индексы 0,1,2,...,lookback-1
        x = np.arange(len(recent_kama))

        # Линейная регрессия: slope = cov(x,y) / var(x)
        slope, _ = np.polyfit(x, recent_kama, deg=1)

        return round(slope, 4)   # можно оставить больше знаков, если нужно

    # ──────────────────────────────────────────────────────────────
    # Общее направление KAMA (True = вверх, False = вниз)
    # ──────────────────────────────────────────────────────────────
    def _get_kama_direction(self, lookback: int = 20) -> bool:
        """
        Считаем, что KAMA идёт вверх, если:
        • последнее значение выше, чем lookback баров назад
        • ИЛИ большинство последних изменений положительные
        """
        if not hasattr(self.kama_indicator, 'kama') or len(self.kama_indicator.kama) < lookback:
            return False

        recent = np.array(self.kama_indicator.kama[-lookback:])

        # Вариант 1 — просто сравниваем начало и конец окна
        if recent[-1] > recent[0]:
            return True

        # Вариант 2 — считаем, сколько раз было повышение (более надёжно)
        diffs = np.diff(recent)
        positive_changes = (diffs > 0).sum()

        # Если больше половины изменений положительные → считаем направление вверх
        return positive_changes > (len(diffs) / 2)

    def analyze(self, deal):
        deal[f'k{self.minutes}_t'] = str(self.kama_indicator.trend)          # выше/ниже текущей цены
        deal[f'k{self.minutes}_sl'] = float(self._get_kama_slope(lookback=8))   # наклон за 8 баров
        deal[f'k{self.minutes}_d']  = bool(self._get_kama_direction(lookback=20))  # общее направление

        return deal