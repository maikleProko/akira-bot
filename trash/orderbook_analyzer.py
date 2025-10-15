import json
import os
from datetime import datetime
from scenarios.orderbook_patterns.estimate_orderbook.estimate_orderbook import do_strategy
from scenarios.orderbook_patterns.orderbook_parser import OrderbookParser
from scenarios.orderbook_patterns.selenium_orderbook_parse.instances.selenium_coinglass_parser import SeleniumCoinglassParser
from utils.decorators import periodic
from utils.functions import MarketProcess


class OrderbookAnalyzer(MarketProcess):
    def __init__(self, orderbook_parser=OrderbookParser()):
        self.orderbook_parser = orderbook_parser

    def prepare(self):
        pass

    def run(self):
        do_strategy(self.orderbook_parser.parser, self.orderbook_parser.orderbook)