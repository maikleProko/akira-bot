import pandas as pd
from utils.core.functions import MarketProcess
import csv
from datetime import datetime, timedelta
from typing import Any
from typing import List, Optional


class HistoryMarketParser(MarketProcess):
    """
    Базовый парсер для работы с историей цены.
    Парсер формирует pandas.DataFrame и сохраняет его в self.df.

    Таймфрейм задаётся параметром timeframe (количество минут на одну свечу).
    Например:
        timeframe=1  → 1-минутные свечи ("1m")
        timeframe=5  → 5-минутные свечи ("5m")
        timeframe=15 → 15-минутные свечи ("15m")
    """

    def __init__(
        self,
        symbol1: str = 'BTC',
        symbol2: str = 'USDT',
        timeframe: int = 1,        # новое: количество минут на свечу
        candles: int = 1000        # новое имя: сколько свечей загружать в реал-тайм режиме
    ):
        if timeframe <= 0:
            raise ValueError("timeframe должен быть положительным целым числом")

        self.slash_symbol = symbol1 + '/' + symbol2
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.timeframe = timeframe                  # минуты на одну свечу
        self.candles = candles                      # количество свечей для реал-тайм загрузки
        self.interval_str = f"{timeframe}m"         # строка интервала для API, напр. "5m"

        self.df = None
        self.history_df = None

        self.init()

    def init(self):
        pass

    def normalize_slash_symbol(self, slash_symbol: str) -> str:
        clean = slash_symbol.replace("/", "").replace("-", "").replace("_", "")
        return clean.upper()

    def adjust_timestamp(self, ts_ms: int) -> datetime:
        return datetime.utcfromtimestamp(ts_ms / 1000.0) + timedelta(hours=3)

    def save_csv(self, filepath: str, headers: List[str], rows: List[List[Any]]) -> None:
        """Сохраняет CSV-файл с указанными заголовками и строками."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def get_df(
        self,
        slash_symbol: str,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Должен быть реализован в подклассах.
        interval теперь передаётся как строка вида "1m", "5m" и т.д.
        """
        raise NotImplementedError("get_df should be implemented in subclasses.")

    def prepare(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        """
        Загружает исторические данные за указанный диапазон (опционально).
        """
        if start_time is not None and end_time is not None:
            self.history_df = self.get_df(
                self.slash_symbol,
                interval=self.interval_str,
                start_time=start_time,
                end_time=end_time
            )
        # Если диапазон не указан — просто очищаем history_df
        elif start_time is None and end_time is None:
            self.history_df = pd.DataFrame()
        else:
            raise ValueError("Для prepare нужно указать либо оба параметра (start_time и end_time), либо ни одного.")

    def run_realtime(self):
        """
        Загружает последние self.candles свечей с текущим таймфреймом, aligned к кратным интервалам относительно часа.
        """
        end_time = datetime.now()

        # Находим начало текущей (или последней завершённой) свечи, aligned к timeframe
        minutes = end_time.minute
        floor_min = (minutes // self.timeframe) * self.timeframe
        current_candle_start = end_time.replace(minute=floor_min, second=0, microsecond=0)

        # Вычисляем start_time для первых свечей назад
        start_time = current_candle_start - timedelta(minutes=(self.candles - 1) * self.timeframe)

        self.df = self.get_df(
            self.slash_symbol,
            interval=self.interval_str,
            start_time=start_time,
            end_time=end_time,
            limit=self.candles  # многие API позволяют указать limit, используем его
        )

    def run_historical(self, start_time: datetime, current_time: datetime):
        """
        Вырезает нужный диапазон из заранее загруженного history_df.
        """
        if self.history_df is None or self.history_df.empty:
            raise ValueError("history_df не загружен. Сначала вызовите prepare().")

        mask = (pd.to_datetime(self.history_df['time']) >= start_time) & \
               (pd.to_datetime(self.history_df['time']) <= current_time)

        self.df = self.history_df[mask].copy()