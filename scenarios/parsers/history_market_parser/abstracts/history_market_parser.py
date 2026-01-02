import pandas as pd
import csv
import os
from datetime import datetime, timedelta
from typing import Any, List, Optional
from utils.core.functions import MarketProcess


class HistoryMarketParser(MarketProcess):
    """
    Базовый парсер для работы с историей цены.
    Теперь поддерживает два режима:
        - "generating": загружает данные с API и сохраняет в CSV
        - "loading": читает уже существующий CSV (быстро, для повторных бэктекстов)
    """

    def __init__(
        self,
        symbol1: str = 'BTC',
        symbol2: str = 'USDT',
        timeframe: int = 1,
        candles: int = 1000,
        mode: str = "generating"  # новое: "generating" или "loading"
    ):
        if timeframe <= 0:
            raise ValueError("timeframe должен быть положительным целым числом")

        if mode not in {"generating", "loading"}:
            raise ValueError("mode должен быть 'generating' или 'loading'")

        self.slash_symbol = symbol1 + '/' + symbol2
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.timeframe = timeframe
        self.candles = candles
        self.interval_str = f"{timeframe}m"
        self.mode = mode  # <-- новая переменная

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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

    def load_csv(self, filepath: str) -> pd.DataFrame:
        """Загружает CSV и возвращает DataFrame"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV файл не найден: {filepath}. Запустите в режиме 'generating' хотя бы раз.")
        df = pd.read_csv(filepath)
        # Убедимся, что колонка time — datetime
        df['time'] = pd.to_datetime(df['time'])
        return df

    def get_df(
        self,
        slash_symbol: str,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        raise NotImplementedError("get_df should be implemented in subclasses.")

    def prepare(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        if self.mode == "loading":
            # Формируем путь к файлу, как если бы мы сохраняли
            symbol = self.normalize_slash_symbol(self.slash_symbol)
            start_str = start_time.strftime("%Y%m%d") if start_time else "begin"
            end_str = end_time.strftime("%Y%m%d") if end_time else "end"
            filename = f'files/history_data/binance_{symbol}_{self.interval_str}_{start_str}_to_{end_str}.csv'
            self.history_df = self.load_csv(filename)
            print(f"[HistoryParser] Загружен CSV (loading mode): {filename}")
        else:
            # generating mode — старое поведение
            if start_time is not None and end_time is not None:
                self.history_df = self.get_df(
                    self.slash_symbol,
                    interval=self.interval_str,
                    start_time=start_time,
                    end_time=end_time
                )
            elif start_time is None and end_time is None:
                self.history_df = pd.DataFrame()
            else:
                raise ValueError("Для prepare нужно указать либо оба параметра, либо ни одного.")

    def run_realtime(self):
        if self.mode == "loading":
            # В реал-тайм режиме loading не имеет смысла — всегда генерируем свежие данные
            print("[HistoryParser] Реал-тайм: принудительно generating mode")
        # Всегда загружаем свежие данные с API
        end_time = datetime.now()
        minutes = end_time.minute
        floor_min = (minutes // self.timeframe) * self.timeframe
        current_candle_start = end_time.replace(minute=floor_min, second=0, microsecond=0)
        start_time = current_candle_start - timedelta(minutes=(self.candles - 1) * self.timeframe)

        self.df = self.get_df(
            self.slash_symbol,
            interval=self.interval_str,
            start_time=start_time,
            end_time=end_time,
            limit=self.candles
        )

    def run_historical(self, start_time: datetime, current_time: datetime):
        if self.history_df is None or self.history_df.empty:
            raise ValueError("history_df не загружен. Сначала вызовите prepare().")

        mask = (pd.to_datetime(self.history_df['time']) >= start_time) & \
               (pd.to_datetime(self.history_df['time']) <= current_time)

        self.df = self.history_df[mask].copy()