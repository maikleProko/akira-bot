from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from utils.core.functions import MarketProcess


class CHoCHStrategy(MarketProcess):
    def __init__(
        self,
        history_market_parser: HistoryMarketParser,
        choch_indicator: CHoCHIndicator
    ):
        self.history_market_parser=history_market_parser
        self.choch_indicator=choch_indicator

    def run_historical(self, start_time, current_time):
        pass
        print(str(current_time) + ' ' + str(self.choch_indicator.is_now_CHoCH))



