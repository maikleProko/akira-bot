from typing import List, Tuple, Type
from datetime import datetime, timezone, timedelta
from scenarios.reverberate.history_reverberate.constants import *
from scenarios.reverberate.history_reverberate.abstracts.history_reverberation_parser import HistoryReverberationParser
from scenarios.reverberate.history_reverberate.instances.binance_history_reverberation_parser import BinanceHistoryReverberationParser
from scenarios.reverberate.history_reverberate.instances.bybit_history_reverberation_parser import BybitHistoryReverberationParser

# ====== Регистрация парсеров и фабрика ======
PARSER_REGISTRY: List[Tuple[str, Type[HistoryReverberationParser]]] = [
    ("binance", BinanceHistoryReverberationParser),
    ("bybit", BybitHistoryReverberationParser),
    # ....
]

def get_parser(exchange_name: str) -> HistoryReverberationParser:
    lookup_key = (exchange_name or "").strip().lower()
    for registered_name, parser_class in PARSER_REGISTRY:
        if registered_name.lower() == lookup_key:
            return parser_class()
    raise ValueError(f"Unknown exchange: {exchange_name}")


def history_reverberate():
    parser = get_parser("binance")
    current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
    start_time = current_time - timedelta(minutes=50)
    end_time = current_time - timedelta(minutes=1)
    reverberation_results = parser.reverberate_in_period(
        start_datetime=start_time,
        end_datetime=end_time,
        symbol="BTCUSDT",
        multi_window_minutes=DEFAULT_MULTI_WINDOW_MINUTES
    )
    for item in reverberation_results:
        print(item)