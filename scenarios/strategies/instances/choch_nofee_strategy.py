from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class CHoCHNoFeeStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            kama_indicator: KamaIndicator,
            choch_indicator: CHoCHIndicator,
            nwe_bounds_indicator: NweBoundsIndicator,
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.kama_indicator = kama_indicator
        self.choch_indicator = choch_indicator
        self.nwe_bounds_indicator = nwe_bounds_indicator

    def run_historical(self, start_time, current_time):
        if (
                self.choch_indicator.is_now_CHoCH and self.choch_indicator.choch_cross_price <=
                self.history_market_parser_1m.df['close'].iloc[-1]
        ) and \
                self.kama_indicator.trend == "BULLISH":
            self.is_accepted_by_strategy = True
        else:
            self.is_accepted_by_strategy = False

    def run_realtime(self):
        print('---')
        print(self.choch_indicator.is_now_CHoCH)
        print(self.history_market_parser_1m.df['close'].iloc[-1])
        print(self.kama_indicator.trend)
        if (
                self.choch_indicator.is_now_CHoCH and self.choch_indicator.choch_cross_price <=
                self.history_market_parser_1m.df['close'].iloc[-1]
        ) and \
                self.kama_indicator.trend == "BULLISH":
            print('accepted')
            self.is_accepted_by_strategy = True
        else:
            self.is_accepted_by_strategy = False
