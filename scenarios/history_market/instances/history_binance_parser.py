import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Optional
from scenarios.history_market.abstracts.history_market_parser import HistoryMarketParser


class HistoryBinanceParser(HistoryMarketParser):
    """
    Парсер для Binance. Получает kline (candlestick) данные и формирует DataFrame:
    time(+3h), open, high, low, close, volume
    """
    API_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, symbol1: str = 'BTC', symbol2: str = 'USDT', minutes: int = 1000):
        super().__init__("binance")
        self.slash_symbol = symbol1 + '/' + symbol2
        self.minutes = minutes
        self.headers = ["time", "open", "high", "low", "close", "volume"]

    def _to_ms(self, dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

    def fetch_klines(self,
                     symbol: str,
                     interval: str = "1m",
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

    def parse(self,
                       slash_symbol: str,
                       interval: str = "1m",
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 1000) -> pd.DataFrame:
        """
        Основной метод: получает данные за промежуток и формирует DataFrame.
        Результат сохраняется в self.df и возвращается.
        """
        symbol = self.normalize_slash_symbol(slash_symbol)
        klines = self.fetch_klines(symbol, interval, start_time, end_time, limit)
        rows = self._klines_to_rows(klines)
        self.df = pd.DataFrame(rows, columns=self.headers)
        self.save_csv(f'files/history_data/last1000_current_history_data_binance_{symbol}.csv', self.headers, rows)
        return self.df

    def prepare(self):
        self.df = None

    def run(self, start_time=None, current_time=None, end_time=None):
        if start_time is None and current_time is None and end_time is None:
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=self.minutes)

        self.parse(self.slash_symbol, interval="1m", start_time=start_time, end_time=end_time)
        return self.df


