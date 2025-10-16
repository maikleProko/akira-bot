import pandas as pd

from utils.functions import MarketProcess
import csv
from datetime import datetime, timedelta
from typing import List, Any
from typing import List, Optional

class HistoryMarketParser(MarketProcess):
    """
    Базовый парсер для работы с историей цены.
    Парсер формирует pandas.DataFrame и сохраняет его в self.df.
    """
    def __init__(self, platform_name: str):
        self.platform = platform_name
        self.df = None

    def normalize_slash_symbol(self, slash_symbol: str) -> str:
        clean = slash_symbol.replace("/", "").replace("-", "").replace("_", "")
        return clean.upper()

    def adjust_timestamp(self, ts_ms: int) -> datetime:
        return datetime.utcfromtimestamp(ts_ms / 1000.0) + timedelta(hours=3)

    def prepare(self):
        self.df = None

    def run(self, start_time=None, current_time=None, end_time=None):
        raise NotImplementedError("Run should be implemented in subclasses.")

    def save_csv(self, filepath: str, headers: List[str], rows: List[List[Any]]) -> None:
        """Сохраняет CSV-файл с указанными заголовками и строками."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def parse(self,
                       slash_symbol: str,
                       interval: str = "1m",
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 1000) -> pd.DataFrame:
        raise NotImplementedError("Parse should be implemented in subclasses.")

