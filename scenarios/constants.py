from scenarios.strategies.factor_analyzer import FactorAnalyzer
from scenarios.history_market.instances.history_binance_parser import HistoryBinanceParser
from scenarios.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.orderbook_patterns.beautifulsoup_orderbook_parse.instances.beautifulsoup_coinglass_parser import \
    BeautifulsoupCoinglassParser
from scenarios.orderbook_patterns.orderbook_parser import OrderbookParser
from scenarios.strategies.obies_straregy import ObiesStrategy
from scenarios.strategies.obl_strategy import OblStrategy
from scenarios.strategies.olis_strategy import OlisStrategy
from scenarios.strategies.smart_scalp_strategy import SmartScalpStrategy

symbol1 = 'BTC'
symbol2 = 'USDT'

history_market_parser = HistoryBinanceParser(symbol1, symbol2)
orderbook_parser = OrderbookParser(BeautifulsoupCoinglassParser(symbol1, symbol2))
nwe_bounds_indicator = NweBoundsIndicator(history_market_parser)
atr_bounds_indicator = AtrBoundsIndicator(history_market_parser)
factor_analyzer = FactorAnalyzer(
    history_market_parser=history_market_parser,
    orderbook_parser=orderbook_parser,
    nwe_bounds_indicator=nwe_bounds_indicator,
    atr_bounds_indicator=atr_bounds_indicator
)

olis_strategy = OlisStrategy(
    history_market_parser=history_market_parser,
    orderbook_parser=orderbook_parser,
    nwe_bounds_indicator=nwe_bounds_indicator,
    atr_bounds_indicator=atr_bounds_indicator
)

obies_strategy = ObiesStrategy(
    history_market_parser=history_market_parser,
    orderbook_parser=orderbook_parser,
    nwe_bounds_indicator=nwe_bounds_indicator,
    atr_bounds_indicator=atr_bounds_indicator
)

smart_scalp_strategy = SmartScalpStrategy(
    history_market_parser=history_market_parser,
    orderbook_parser=orderbook_parser,
    nwe_bounds_indicator=nwe_bounds_indicator,
    atr_bounds_indicator=atr_bounds_indicator
)

obl_strategy = OblStrategy(
    history_market_parser=history_market_parser,
    orderbook_parser=orderbook_parser,
    nwe_bounds_indicator=nwe_bounds_indicator,
    atr_bounds_indicator=atr_bounds_indicator
)

MARKET_PROCESSES = [
    history_market_parser,
    orderbook_parser,
    nwe_bounds_indicator,
    atr_bounds_indicator,
    obl_strategy,
    obies_strategy,
    smart_scalp_strategy,
    olis_strategy
]