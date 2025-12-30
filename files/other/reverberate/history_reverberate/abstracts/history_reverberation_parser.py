from abc import ABC, abstractmethod
from datetime import datetime
from time import sleep
from typing import List, Dict, Optional

from files.other.reverberate.history_reverberate.constants import *
from files.other.reverberate.history_reverberate.services.aggregation_service import AggregationService
from files.other.reverberate.history_reverberate.services.reverberation_service import ReverberationService
from files.other.reverberate.history_reverberate.services.time_service import TimeService


# ====== Абстрактный мета‑класс (объединяет сервисы) ======
class HistoryReverberationParser(ABC):
    def __init__(self):
        self.time_service = TimeService()
        self.aggregation_service = AggregationService()
        self.reverberation_service = ReverberationService()
        self.results = []

    def reverberate_in_minute(self, minute_start, window_minute_starts, symbol, multi_window_minutes, max_window_minutes):
        try:
            print('[' + str(minute_start) + '/' + str(window_minute_starts[len(window_minute_starts)-1]) + ']')
            element = self._reverberate_for_minute_with_multi_windows(minute_start, symbol, multi_window_minutes, max_window_minutes)
            self.results.append(element)
        except Exception as e:
            print('------------')
            print('error: ' + str(e))
            print('retrying...')
            sleep(2)
            self.reverberate_in_minute(minute_start, window_minute_starts, symbol, multi_window_minutes,max_window_minutes)

    def reverberate_in_period(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
        symbol: str,
        multi_window_minutes: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Возвращает список ревербераций для каждой минуты в [start_datetime, end_datetime].
        Каждая запись содержит:
          - базовую минутную реверберацию (reverberation_50, reverberation_75)
          - дополнительные поля reverberation_75_{N}_minutes для каждого N в multi_window_minutes
        """
        if multi_window_minutes is None:
            multi_window_minutes = DEFAULT_MULTI_WINDOW_MINUTES
        window_minute_starts = self.time_service.minute_range(start_datetime, end_datetime)
        max_window_minutes = max(multi_window_minutes)
        self.results: List[Dict] = []
        for minute_start in window_minute_starts:
            self.reverberate_in_minute(minute_start, window_minute_starts, symbol, multi_window_minutes, max_window_minutes)
        return self.results

    def _reverberate_for_minute_with_multi_windows(
        self,
        minute_start: datetime,
        symbol: str,
        multi_window_minutes: List[int],
        max_window_minutes: int
    ) -> Dict:
        minute_start_milliseconds = self.time_service.to_milliseconds(minute_start)
        window_end_milliseconds = minute_start_milliseconds + max_window_minutes * 60 * 1000 - 1
        trades = self.fetch_trades(symbol, minute_start_milliseconds, window_end_milliseconds)
        # базовая минутная реверберация (1 минута)
        one_minute_seconds = 60
        ratios_one_minute = self.aggregation_service.per_second_buy_ratios(trades, minute_start_milliseconds, one_minute_seconds)
        base_metrics = self.reverberation_service.compute_window_reverberation(ratios_one_minute, minute_start_milliseconds, one_minute_seconds)
        # дополнительные метрики для каждого окна
        for window_minutes in sorted(set(multi_window_minutes)):
            if window_minutes == 1:
                # уже посчитано
                base_value = base_metrics["reverberation_75"]
                base_metrics[f"reverberation_75_{window_minutes}_minutes"] = base_value
                continue
            window_seconds = window_minutes * 60
            ratios = self.aggregation_service.per_second_buy_ratios(trades, minute_start_milliseconds, window_seconds)
            metrics = self.reverberation_service.compute_window_reverberation(ratios, minute_start_milliseconds, window_seconds)
            base_metrics[f"reverberation_75_{window_minutes}_minutes"] = metrics["reverberation_75"]
        return base_metrics

    @abstractmethod
    def fetch_trades(self, symbol: str, start_milliseconds: int, end_milliseconds: int) -> List[Dict]:
        """
        Должен вернуть список сделок в унифицированном формате:
        [
          {'time': <milliseconds>, 'quantity': <float>, 'is_buyer_maker': <bool>},
          ...
        ]
        """
        raise NotImplementedError





