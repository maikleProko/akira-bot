from scenarios.analyzers.factor_analyzer import FactorAnalyzer
from scenarios.history_market.instances.history_binance_parser import HistoryBinanceParser
from scenarios.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.orderbook_patterns.beautifulsoup_orderbook_parse.instances.beautifulsoup_coinglass_parser import \
    BeautifulsoupCoinglassParser
from scenarios.orderbook_patterns.orderbook_parser import OrderbookParser
from scenarios.reverberate.reverberate_parser import ReverberateParser

history_market_parser = HistoryBinanceParser()
#reverberate_parser = ReverberateParser()
orderbook_parser = OrderbookParser(BeautifulsoupCoinglassParser())
nwe_bounds_indicator = NweBoundsIndicator(history_market_parser)
atr_bounds_indicator = AtrBoundsIndicator(history_market_parser)
factor_analyzer = FactorAnalyzer(
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
    factor_analyzer
]