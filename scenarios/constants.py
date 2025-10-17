from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.parsers.orderbook_parser.beautifulsoup_orderbook_parse.instances.beautifulsoup_coinglass_parser import \
    BeautifulsoupCoinglassParser
from scenarios.parsers.orderbook_parser.orderbook_parser import OrderbookParser
from scenarios.strategies.historical_strategies.level_strategy.level_strategy import LevelStrategy
from scenarios.strategies.historical_strategies.test_strategy.test_strategy import TestStrategy

#SYMBOLS
symbol1 = 'BTC'
symbol2 = 'USDT'


#FOR HISTORICAL TRADING
realtime = False
start_time_string='2025/10/16 09:25'
end_time_string='2025/10/16 18:45'


#PROCESSES (PARSERS)
history_market_parser = HistoryBinanceParser(symbol1, symbol2)
orderbook_parser = OrderbookParser(BeautifulsoupCoinglassParser(symbol1, symbol2))
nwe_bounds_indicator = NweBoundsIndicator(history_market_parser)
atr_bounds_indicator = AtrBoundsIndicator(history_market_parser)


#PROCESSES (STRATEGIES)
level_strategy = LevelStrategy(
    history_market_parser=history_market_parser,
    orderbook_parser=orderbook_parser,
    nwe_bounds_indicator=nwe_bounds_indicator,
    atr_bounds_indicator=atr_bounds_indicator
)

#MARKET PROCESSES
MARKET_PROCESSES = [
    history_market_parser,
    orderbook_parser,
    atr_bounds_indicator,
    nwe_bounds_indicator,
    level_strategy
]