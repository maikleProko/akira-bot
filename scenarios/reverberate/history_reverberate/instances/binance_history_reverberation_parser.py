import requests
from typing import List, Dict
from scenarios.reverberate.history_reverberate.abstracts.history_reverberation_parser import HistoryReverberationParser
from scenarios.reverberate.history_reverberate.constants import *


# ====== Реализация для Binance ======
class BinanceHistoryReverberationParser(HistoryReverberationParser):
    def fetch_trades(self, symbol: str, start_milliseconds: int, end_milliseconds: int) -> List[Dict]:
        url = BINANCE_BASE_URL + BINANCE_AGGREGATE_TRADES_ENDPOINT
        params = {
            "symbol": symbol.upper(),
            "startTime": start_milliseconds,
            "endTime": end_milliseconds,
            "limit": MAXIMUM_BATCH_SIZE,
        }
        all_trades: List[Dict] = []
        last_start_milliseconds = start_milliseconds
        while True:
            params["startTime"] = last_start_milliseconds
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            for item in batch:
                all_trades.append({
                    "time": int(item.get("T", 0)),
                    "quantity": float(item.get("q", 0.0)),
                    "is_buyer_maker": bool(item.get("m", False)),
                })
            if len(batch) < MAXIMUM_BATCH_SIZE:
                break
            last_start_milliseconds = int(batch[-1].get("T", last_start_milliseconds)) + 1
            if last_start_milliseconds > end_milliseconds:
                break

        result = [t for t in all_trades if start_milliseconds <= t["time"] <= end_milliseconds]
        return result