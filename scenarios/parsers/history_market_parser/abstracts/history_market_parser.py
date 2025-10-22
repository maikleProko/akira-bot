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
    """
    def __init__(self, platform_name: str, symbol1: str = 'BTC', symbol2: str = 'USDT', minutes: int = 1000):
        self.slash_symbol = symbol1 + '/' + symbol2
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.minutes = minutes
        self.platform = platform_name
        self.df = None
        self.history_df = None

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

    def get_df(self,
                       slash_symbol: str,
                       interval: str = "1m",
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 1000) -> pd.DataFrame:
        raise NotImplementedError("Parse should be implemented in subclasses.")

    def prepare(self, start_time=None, end_time=None):
        if start_time is not None and end_time is not None:
            self.history_df = self.get_df(self.slash_symbol, interval="1m", start_time=start_time, end_time=end_time)

    def run_realtime(self):
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=self.minutes)
        self.df = self.get_df(self.slash_symbol, interval="1m", start_time=start_time, end_time=end_time)

    def run_historical(self, start_time, current_time):
        if not self.history_df.empty:
            mask = (pd.to_datetime(self.history_df['time']) >= start_time) & \
                   (pd.to_datetime(self.history_df['time']) <= current_time)
            self.df = self.history_df[mask].copy()
        else:
            raise ValueError("history_df не загружен или не указаны start_time/current_time")


