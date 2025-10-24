from scenarios.parsers.arbitrage_parser.instances.binance_arbitrage_parser import BinanceArbitrageParser
from scenarios.parsers.arbitrage_parser.instances.binance_smart_arbitrage_buyer import BinanceSmartArbitrageBuyer
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
'''
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.parsers.orderbook_parser.beautifulsoup_orderbook_parse.instances.beautifulsoup_coinglass_parser import \
    BeautifulsoupCoinglassParser
from scenarios.parsers.orderbook_parser.orderbook_parser import OrderbookParser
from scenarios.strategies.historical_strategies.level_strategy.reverb_level_strategy import ReverbLevelStrategy
from scenarios.strategies.historical_strategies.level_strategy.simple_level_strategy import SimpleLevelStrategy
from scenarios.strategies.historical_strategies.level_strategy.smart_level_strategy import SmartLevelStrategy
'''
from scenarios.strategies.historical_strategies.simple_corridor_strategy.simple_corridor_strategy import \
    SimpleCorridorStrategy
'''
from scenarios.strategies.historical_strategies.test_strategy.test_strategy import TestStrategy
'''

#SYMBOLS
symbol1 = 'BTC'
symbol2 = 'USDT'


#FOR HISTORICAL TRADING
realtime = True
start_time_string='2025/10/21 06:30'
end_time_string='2025/10/22 15:55'


#PROCESSES (PARSERS)
history_market_parser = HistoryBinanceParser(symbol1, symbol2)
'''
orderbook_parser = OrderbookParser(BeautifulsoupCoinglassParser(symbol1, symbol2))
nwe_bounds_indicator = NweBoundsIndicator(history_market_parser)
atr_bounds_indicator = AtrBoundsIndicator(history_market_parser)
'''

#PROCESSES (STRATEGIES)
'''
level_strategy = SmartLevelStrategy(
    history_market_parser=history_market_parser,
    orderbook_parser=orderbook_parser,
    nwe_bounds_indicator=nwe_bounds_indicator,
    atr_bounds_indicator=atr_bounds_indicator
)
'''
simple_corridor_strategy = SimpleCorridorStrategy(
    history_market_parser=history_market_parser,
)

arbitrage_parser = BinanceSmartArbitrageBuyer(2000)

#MARKET PROCESSES
MARKET_PROCESSES = [
    arbitrage_parser
]