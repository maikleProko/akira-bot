import numpy as np
import pandas as pd
from scenarios.analysis.property.abstracts.property import Property
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.orderblock_indicator import OrderblockIndicator

class SMCOrderblokProperty(Property):
    def __init__(self, minutes):
        super().__init__()
        self.minutes = minutes
        self.history_market_parser: HistoryBinanceParser = None
        self.orderblock_indicator: OrderblockIndicator = None

    def prepare(self, deal):
        self.history_market_parser = HistoryBinanceParser('BTC', 'USDT', self.minutes, 1000, 'generating')
        self.orderblock_indicator = OrderblockIndicator(self.history_market_parser)
        self.market_processes = [
            self.history_market_parser,
            self.orderblock_indicator
        ]

    def analyze(self, deal):
        prefix = f'o{str(self.minutes)}_'
        deal[prefix + 'lbuo'] = str(self.orderblock_indicator.last_bull_orderblock)
        deal[prefix + 'lbeo'] = str(self.orderblock_indicator.last_bear_orderblock)
        return deal