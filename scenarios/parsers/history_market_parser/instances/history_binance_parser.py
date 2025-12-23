import pandas as pd
import requests
from datetime import datetime
from typing import List, Optional
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser


class HistoryBinanceParser(HistoryMarketParser):
    """
    Парсер для Binance. Получает kline (candlestick) данные и формирует DataFrame:
    time(+3h), open, high, low, close, volume
    """
    API_URL = "https://api.binance.com/api/v3/klines"

    def init(self):
        self.platform = 'binance'
        self.headers = ["time", "open", "high", "low", "close", "volume"]

    def _to_ms(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

    def fetch_klines(self,
                     symbol: str,
                     interval: str,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 1000) -> List[List]:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = self._to_ms(start_time)
        if end_time:
            params["endTime"] = self._to_ms(end_time)
        resp = requests.get(self.API_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _klines_to_rows(self, klines: List[List]):
        rows = []
        for k in klines:
            t = self.adjust_timestamp(int(k[0])).strftime("%Y-%m-%d %H:%M:%S")
            rows.append([t, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
        return rows

    def get_df(self,
               slash_symbol: str,
               interval: str = "1m",  # этот параметр теперь игнорируется — используется self.interval_str
               start_time: Optional[datetime] = None,
               end_time: Optional[datetime] = None,
               limit: int = 1000) -> pd.DataFrame:
        """
        Основной метод: получает данные за промежуток и формирует DataFrame.
        Интервал берётся из self.interval_str (задан при инициализации через timeframe).
        """
        symbol = self.normalize_slash_symbol(slash_symbol)

        # Используем интервал из родительского класса (например, "5m", "15m" и т.д.)
        klines = self.fetch_klines(
            symbol=symbol,
            interval=self.interval_str,  # <-- ключевое изменение
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )

        rows = self._klines_to_rows(klines)
        df = pd.DataFrame(rows, columns=self.headers)

        # Сохранение с указанием таймфрейма для удобства
        filename = f'files/history_data/last{limit}_candles_binance_{symbol}_{self.interval_str}.csv'
        self.save_csv(filename, self.headers, rows)

        return df