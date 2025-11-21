from datetime import datetime

from scenarios.parsers.arbitrage_parser.rolling_arbitrage_parser.core.abstract_arbitrage_parser import \
    AbstractArbitrageParser
from utils.core.functions import MarketProcess


class RollingArbitrageParser(AbstractArbitrageParser):
    def __init__(self, api_key='', api_secret=''):
        self.session = None
        self.api_url = None
        self.init(api_key, api_secret)

    def run_realtime(self):
        pass
