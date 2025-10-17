from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from utils.core.functions import MarketProcess


class Indicator(MarketProcess):

    def __init__(self, history_market_parser: HistoryMarketParser):
        self.history_market_parser = history_market_parser

    def prepare(self, start_time=None, end_time=None):
        pass

    def run_realtime(self):
        pass

