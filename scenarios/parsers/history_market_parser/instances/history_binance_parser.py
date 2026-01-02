import pandas as pd
import requests
import os
from datetime import datetime, timedelta
from typing import List, Optional
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser


class HistoryBinanceParser(HistoryMarketParser):
    """
    Парсер Binance с поддержкой mode="generating"/"loading"
    """
    API_URL = "https://api.binance.com/api/v3/klines"

    INTERVAL_DELTAS = {
        "1m": timedelta(minutes=1),
        "3m": timedelta(minutes=3),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
        "6h": timedelta(hours=6),
        "8h": timedelta(hours=8),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
        "3d": timedelta(days=3),
        "1w": timedelta(weeks=1),
        "1M": timedelta(days=31),
    }

    def init(self):
        self.platform = 'binance'
        self.headers = ["time", "open", "high", "low", "close", "volume"]

    def _to_ms(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

    def _ms_to_dt(self, ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000)

    def fetch_klines(self, symbol: str, interval: str, start_time=None, end_time=None, limit=None) -> List[List]:
        # (тот же код, что был раньше — без изменений)
        all_klines: List[List] = []

        if limit is not None:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if start_time:
                params["startTime"] = self._to_ms(start_time)
            if end_time:
                params["endTime"] = self._to_ms(end_time)
            resp = requests.get(self.API_URL, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()

        current_start = start_time
        while True:
            params = {"symbol": symbol, "interval": interval, "limit": 1000}
            if current_start:
                params["startTime"] = self._to_ms(current_start)
            if end_time:
                params["endTime"] = self._to_ms(end_time)

            resp = requests.get(self.API_URL, params=params, timeout=10)
            resp.raise_for_status()
            klines = resp.json()
            if not klines:
                break
            all_klines.extend(klines)

            last_open_ms = int(klines[-1][0])
            last_open_dt = self._ms_to_dt(last_open_ms)
            delta = self.INTERVAL_DELTAS.get(interval)
            if delta is None:
                raise ValueError(f"Неизвестный interval: {interval}")
            next_start = last_open_dt + delta

            if end_time and next_start >= end_time:
                break
            if len(klines) < 1000:
                break
            current_start = next_start

        return all_klines

    def _klines_to_rows(self, klines: List[List]):
        rows = []
        for k in klines:
            t = self.adjust_timestamp(int(k[0])).strftime("%Y-%m-%d %H:%M:%S")
            rows.append([t, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
        return rows

    def get_df(
        self,
        slash_symbol: str,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:

        symbol = self.normalize_slash_symbol(slash_symbol)

        # Определяем путь к файлу
        if limit:
            filename = f'files/history_data/last{limit}_candles_binance_{symbol}_{self.interval_str}.csv'
        else:
            start_str = start_time.strftime("%Y%m%d") if start_time else "begin"
            end_str = end_time.strftime("%Y%m%d") if end_time else "end"
            filename = f'files/history_data/binance_{symbol}_{self.interval_str}_{start_str}_to_{end_str}.csv'

        # Если mode=loading и файл существует — читаем его
        if self.mode == "loading" and os.path.exists(filename):
            print(f"[HistoryBinanceParser] Загружен существующий CSV: {filename}")
            return self.load_csv(filename)

        # Иначе — generating: скачиваем с API и сохраняем
        klines = self.fetch_klines(
            symbol=symbol,
            interval=self.interval_str,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        rows = self._klines_to_rows(klines)
        df = pd.DataFrame(rows, columns=self.headers)

        print(f"[HistoryBinanceParser] Данные получены с API и сохранены: {filename}")
        self.save_csv(filename, self.headers, rows)

        return df