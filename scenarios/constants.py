from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator

from scenarios.strategies.historical_strategies.simple_corridor_strategy.simple_corridor_strategy import \
    SimpleCorridorStrategy
from scenarios.strategies.historical_strategies.smart_money_strategy.choch_strategy import \
    CHoCHStrategy

#SYMBOLS
symbol1 = 'BTC'
symbol2 = 'USDT'


#FOR HISTORICAL TRADING
realtime = False
start_time_string='2025/12/22 10:00'
end_time_string='2025/12/22 15:00'


#PROCESSES (PARSERS)
history_market_parser = HistoryBinanceParser(symbol1, symbol2, 525600)
nwe_bounds_indicator = NweBoundsIndicator(history_market_parser)
atr_bounds_indicator = AtrBoundsIndicator(history_market_parser)
choch_indicator = CHoCHIndicator(history_market_parser)

#PROCESSES (STRATEGIES)
choch_strategy = CHoCHStrategy(
    history_market_parser=history_market_parser,
    choch_indicator=choch_indicator
)


#MARKET PROCESSES
MARKET_PROCESSES = [
    history_market_parser,
    choch_indicator,
    choch_strategy
]