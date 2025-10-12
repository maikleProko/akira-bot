from scenarios.market.instances.binance_parser import BinanceParser
from scenarios.orderbook_patterns.orderbook_analyzer import OrderbookAnalyzer
from trash.analyze_orderbook import analyze_orderbook
from utils.decorators import periodic

MARKET_PROCESSES = [
    OrderbookAnalyzer(),
    BinanceParser()
]


class MainProcessor:

    @periodic(second=0)
    def run(self):
        for market_process in MARKET_PROCESSES:
            market_process.run()

    def prepare(self):
        for market_process in MARKET_PROCESSES:
            market_process.prepare()

    def __init__(self):
        self.prepare()
        self.run()
