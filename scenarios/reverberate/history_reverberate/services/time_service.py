from datetime import datetime, timedelta, timezone
from typing import List, Union


# ====== Утилиты времени ======
class TimeService:
    @staticmethod
    def to_milliseconds(datetime_object: datetime) -> int:
        return int(datetime_object.replace(tzinfo=timezone.utc).timestamp() * 1000)

    @staticmethod
    def from_milliseconds(milliseconds: int) -> datetime:
        return datetime.fromtimestamp(milliseconds / 1000.0, tz=timezone.utc)

    @staticmethod
    def minute_range(start_datetime: datetime, end_datetime: datetime) -> List[datetime]:
        start_minute = start_datetime.replace(second=0, microsecond=0, tzinfo=timezone.utc)
        end_minute = end_datetime.replace(second=0, microsecond=0, tzinfo=timezone.utc)
        minutes: List[datetime] = []
        current = start_minute
        while current <= end_minute:
            minutes.append(current)
            current = current + timedelta(minutes=1)
        return minutes

    @staticmethod
    def window_range(start_datetime: datetime, end_datetime: datetime,
                     step_time: Union[int, timedelta]) -> List[datetime]:
        """
        Возвращает список стартовых меток для окон длиной step_time.
        step_time может быть целым числом (минуты) или timedelta.
        Старт выравнивается по секундам (сек=0, микросек=0) для предсказуемости.
        """
        if isinstance(step_time, int):
            step_delta = timedelta(minutes=step_time)
        else:
            step_delta = step_time
        start_aligned = start_datetime.replace(second=0, microsecond=0, tzinfo=timezone.utc)
        end_aligned = end_datetime.replace(second=0, microsecond=0, tzinfo=timezone.utc)
        out: List[datetime] = []
        cur = start_aligned
        while cur <= end_aligned:
            out.append(cur)
            cur = cur + step_delta
        return out