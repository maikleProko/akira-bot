from typing import List, Dict


# ====== Сервис агрегации по секундам ======
class AggregationService:
    @staticmethod
    def per_second_buy_ratios(trades: List[Dict], window_start_milliseconds: int, window_seconds: int) -> List[float]:
        """
        trades: [{'time': ms, 'quantity': float, 'is_buyer_maker': bool}, ...]
        Возвращает список из window_seconds долей buy для каждой секунды окна.
        """
        start_second = window_start_milliseconds // 1000
        buckets = {start_second + i: {"buy_volume": 0.0, "total_volume": 0.0} for i in range(window_seconds)}
        for trade in trades:
            trade_second = int(trade["time"]) // 1000
            if trade_second < start_second or trade_second >= start_second + window_seconds:
                continue
            quantity = float(trade.get("quantity", 0.0))
            is_buyer_maker_flag = trade.get("is_buyer_maker", False)
            is_buy_trade = (is_buyer_maker_flag is False)
            buckets[trade_second]["total_volume"] += quantity
            if is_buy_trade:
                buckets[trade_second]["buy_volume"] += quantity
        ratios: List[float] = []
        for i in range(window_seconds):
            sec = start_second + i
            total = buckets[sec]["total_volume"]
            buy = buckets[sec]["buy_volume"]
            ratios.append((buy / total) if total > 0 else 0.0)
        return ratios