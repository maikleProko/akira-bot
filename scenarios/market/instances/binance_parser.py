import requests
from datetime import datetime, timedelta
from typing import List, Optional
from scenarios.market.abstracts.market_parser import MarketParser


class BinanceParser(MarketParser):
    """
    Парсер для Binance. Получает kline (candlestick) данные и сохраняет CSV:
    time(+3h), open, high, low, close, volume
    """
    API_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self):
        super().__init__("binance")

    def _to_ms(self, dt: datetime) -> int:
        """Конвертирует datetime в миллисекунды (UTC)."""
        return int(dt.timestamp() * 1000)

    def fetch_klines(self,
                     symbol: str,
                     interval: str = "1m",
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 1000) -> List[List]:
        """Запрашивает список kline от Binance (возвращает сырой JSON-список)."""
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = self._to_ms(start_time)
        if end_time:
            params["endTime"] = self._to_ms(end_time)
        resp = requests.get(self.API_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _klines_to_rows(self, klines: List[List]) -> List[List]:
        """Преобразует kline в строки [time_iso, open, high, low, close, volume]."""
        rows = []
        for k in klines:
            t = self.adjust_timestamp(int(k[0]))
            rows.append([t.strftime("%Y-%m-%d %H:%M:%S"), k[1], k[2], k[3], k[4], k[5]])
        return rows

    def parse_and_save(self,
                       pair: str,
                       interval: str = "1m",
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 1000) -> str:
        """
        Основной метод: получает данные за промежуток и сохраняет CSV.
        Возвращает путь к сохранённому файлу.
        """
        symbol = self.normalize_pair(pair)
        klines = self.fetch_klines(symbol, interval, start_time, end_time, limit)
        rows = self._klines_to_rows(klines)
        ref_dt = (end_time or datetime.utcnow()) + timedelta(hours=3)
        filepath = self.build_filepath(symbol, ref_dt)
        headers = ["time(+3h)", "open", "high", "low", "close", "volume"]
        self.save_csv(filepath, headers, rows)
        return filepath