from scenarios.history_market.abstracts.history_market_parser import HistoryMarketParser
from utils.functions import MarketProcess


class Indicator(MarketProcess):

    def __init__(self, history_market_parser: HistoryMarketParser):
        self.history_market_parser = history_market_parser

    def prepare(self):
        pass

    def run(self):
        pass

