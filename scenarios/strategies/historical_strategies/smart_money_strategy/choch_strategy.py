from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.strategies.strategy import Strategy


class CHoCHStrategy(Strategy):
    def __init__(
        self,
        history_market_parser_1m: HistoryMarketParser,
        history_market_parser_15m: HistoryMarketParser,
        kama_indicator: KamaIndicator,
        choch_indicator: CHoCHIndicator
    ):
        super().__init__()
        self.history_market_parser_1m=history_market_parser_1m
        self.history_market_parser_15m=history_market_parser_15m
        self.choch_indicator=choch_indicator
        self.kama_indicator=kama_indicator

    def run_historical(self, start_time, current_time):

        if self.kama_indicator.is_bearish_kamas[-1] and self.kama_indicator.is_bearish_kamas[-2] and self.choch_indicator.is_now_CHoCH:
            self.is_accepted_by_strategy = True





