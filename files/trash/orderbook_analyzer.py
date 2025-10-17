from files.trash.estimate_orderbook.estimate_orderbook import do_strategy
from scenarios.parsers.orderbook_parser.orderbook_parser import OrderbookParser
from utils.core.functions import MarketProcess


class OrderbookAnalyzer(MarketProcess):
    def __init__(self, orderbook_parser=OrderbookParser()):
        self.orderbook_parser = orderbook_parser

    def prepare(self, start_time=None, end_time=None):
        pass

    def run_realtime(self):
        do_strategy(self.orderbook_parser.parser, self.orderbook_parser.orderbook)