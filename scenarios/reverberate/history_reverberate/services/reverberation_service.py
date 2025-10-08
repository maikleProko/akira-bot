from typing import List, Dict
from scenarios.reverberate.history_reverberate.services.time_service import TimeService


# ====== Сервис расчёта реверберации ======
class ReverberationService:
    @staticmethod
    def compute_window_reverberation(ratios: List[float], window_start_milliseconds: int, window_seconds: int) -> Dict:
        """
        ratios: список долей buy по секундам длиной window_seconds.
        Возвращает нормализованные метрики reverberation_50/75.
        """
        count_ge_50 = sum(1 for ratio in ratios if ratio >= 0.5)
        count_ge_75 = sum(1 for ratio in ratios if ratio >= 0.75)
        return {
            "date": TimeService.from_milliseconds(window_start_milliseconds).isoformat(),
            "reverberation_50": count_ge_50 / float(window_seconds),
            "reverberation_75": count_ge_75 / float(window_seconds),
        }