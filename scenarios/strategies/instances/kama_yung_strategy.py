from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class KamaYungStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            history_market_parser_15m: HistoryMarketParser,
            nwe_bounds_indicator: NweBoundsIndicator,
            kama_indicator_120m: KamaIndicator,
            kama_indicator_15m: KamaIndicator,
            kama_indicator_1m: KamaIndicator,
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.history_market_parser_15m = history_market_parser_15m
        self.kama_indicator_120m = kama_indicator_120m
        self.kama_indicator_15m = kama_indicator_15m
        self.kama_indicator_1m = kama_indicator_1m
        self.nwe_bounds_indicator = nwe_bounds_indicator

    def is_under_nwe(self):
        return self.history_market_parser_1m.df['close'].iloc[-1] < self.nwe_bounds_indicator.bounds['out']

    def is_big_trend_kama(self, kama_indicator: KamaIndicator):
        return kama_indicator.trend == "BULLISH" and kama_indicator.trend2 == "BULLISH" and kama_indicator.trend3 == "BULLISH"

    def is_medium_trend_kama(self, kama_indicator: KamaIndicator):
        return kama_indicator.trend == "BULLISH" and kama_indicator.trend2 == "BULLISH"

    def is_small_trend_kama(self, kama_indicator: KamaIndicator):
        return kama_indicator.trend == "BULLISH"

    def is_correct_kamas(self):
        return self.is_small_trend_kama(self.kama_indicator_1m) and self.is_small_trend_kama(self.kama_indicator_15m) and self.is_small_trend_kama(self.kama_indicator_120m)



    def run_historical(self, start_time, current_time):
        if self.is_under_nwe() and self.is_correct_kamas():
            self.is_accepted_by_strategy = True
        else:
            self.is_accepted_by_strategy = False
