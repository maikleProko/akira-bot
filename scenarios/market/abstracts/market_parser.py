import os
import csv
import requests
from datetime import datetime, timedelta
from typing import List, Optional, Any

from utils.functions import MarketProcess


class MarketParser(MarketProcess):
    """
    Базовый парсер для сохранения истории цены торговой пары.
    Сохраняет CSV в files/market_history/{platform}/{SYMBOL}_{YYYYMMDD_HHMM}.csv
    Временные метки подгоняются +3 часа.
    """
    BASE_DIR = "files/market_history"

    def __init__(self, platform_name: str):
        self.platform = platform_name

    def normalize_pair(self, pair: str) -> str:
        """Преобразует пару в формат BTCUSDT (без разделителей, в верхнем регистре)."""
        clean = pair.replace("/", "").replace("-", "").replace("_", "")
        return clean.upper()

    def _ensure_dir(self, path: str) -> None:
        """Создаёт директорию, если её нет."""
        os.makedirs(path, exist_ok=True)

    def build_filepath(self, symbol: str, dt: datetime) -> str:
        """Формирует путь к файлу и создаёт директорию."""
        folder = os.path.join(self.BASE_DIR, self.platform)
        self._ensure_dir(folder)
        fname = f"{symbol}_{dt.strftime('%Y%m%d_%H%M')}.csv"
        return os.path.join(folder, fname)

    def adjust_timestamp(self, ts_ms: int) -> datetime:
        """Переводит миллисекунды UTC в datetime и прибавляет +3 часа."""
        return datetime.utcfromtimestamp(ts_ms / 1000.0) + timedelta(hours=3)

    def save_csv(self, filepath: str, headers: List[str], rows: List[List[Any]]) -> None:
        """Сохраняет CSV-файл с указанными заголовками и строками."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)