import requests
from typing import List, Dict
from scenarios.other.reverberate.history_reverberate.abstracts.history_reverberation_parser import HistoryReverberationParser
from scenarios.other.reverberate.history_reverberate.constants import *


# ====== Реализация для Bybit (пример преобразования) ======
class BybitHistoryReverberationParser(HistoryReverberationParser):
    def fetch_trades(self, symbol: str, start_milliseconds: int, end_milliseconds: int) -> List[Dict]:
        url = BYBIT_BASE_URL + "/v2/public/trading-records"
        params = {"symbol": symbol.upper(), "limit": 200}
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        all_trades: List[Dict] = []
        for item in payload.get("result", []):
            # Пример соответствия полей — адаптируйте под фактический ответ Bybit
            all_trades.append({
                "time": int(item.get("trade_time_ms", 0)),
                "quantity": float(item.get("qty") or item.get("size") or 0.0),
                "is_buyer_maker": bool(item.get("isBuyerMaker", False)),
            })
        return [trade for trade in all_trades if start_milliseconds <= trade["time"] <= end_milliseconds]