import json
from typing import List, Tuple, Type
from datetime import datetime, timedelta
from files.other.reverberate.history_reverberate.constants import *
from files.other.reverberate.history_reverberate.abstracts.history_reverberation_parser import HistoryReverberationParser
from files.other.reverberate.history_reverberate.instances.binance_history_reverberation_parser import BinanceHistoryReverberationParser
from files.other.reverberate.history_reverberate.instances.bybit_history_reverberation_parser import BybitHistoryReverberationParser

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


def history_reverberate_with_prepared_data(platform_name, start_time, end_time, symbol):
    parser = get_parser(platform_name)
    reverberation_results = parser.reverberate_in_period(
        start_datetime=start_time,
        end_datetime=end_time,
        symbol=symbol,
        multi_window_minutes=DEFAULT_MULTI_WINDOW_MINUTES
    )
    return reverberation_results

def get_time_from_string(datetime_string):
    return datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S")-timedelta(hours=3)


def save_to_file(array, platform_name, symbol):
    # сохранить в файл
    out_path = f'files/reverberation/reverberation_data_{symbol}_{platform_name}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(array, f, ensure_ascii=False, indent=2)

    print(f'reverberation {symbol} from {platform_name} result saved to {out_path}')


def history_reverberate(platform_name, start_time_string, end_time_string, symbol):
    start_time = get_time_from_string(start_time_string)
    end_time = get_time_from_string(end_time_string)
    results = history_reverberate_with_prepared_data(platform_name, start_time, end_time, symbol)
    save_to_file(results, platform_name, symbol)