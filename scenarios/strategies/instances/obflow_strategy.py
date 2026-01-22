from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.orderblock_indicator import OrderblockIndicator
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class ObFlowStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            orderblock_indicator: OrderblockIndicator,
            atr_bounds_indicator: AtrBoundsIndicator,
            kama_indicator_240m: KamaIndicator = None,
            kama_indicator_60m: KamaIndicator = None,
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.orderblock_indicator = orderblock_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.kama_indicator_240m = kama_indicator_240m
        self.kama_indicator_60m = kama_indicator_60m
        self.take_profit = None

    def is_under_atr_and_under_ob_collided(self):
        return self.atr_bounds_indicator.bounds2['lower'] \
               < self.orderblock_indicator.last_bull_orderblock['y2'] < \
               self.atr_bounds_indicator.bounds['lower']

    def is_both_obs_existed(self):
        return self.orderblock_indicator.last_bear_orderblock and self.orderblock_indicator.last_bear_orderblock

    def is_kama240_correct(self):
        return self.kama_indicator_240m.trend == "BULLISH" and self.kama_indicator_240m.trend2 == "BULLISH" and self.kama_indicator_240m.trend3 == "BULLISH"

    def is_kama60_correct(self):
        return self.kama_indicator_60m.trend == "BULLISH" and self.kama_indicator_60m.trend2 == "BULLISH" and self.kama_indicator_60m.trend3 == "BULLISH"

    def run_historical(self, start_time, current_time):
        if self.is_both_obs_existed() and self.is_under_atr_and_under_ob_collided():
            print(f"[OrderblockStrategy] ACCEPTED: {current_time}")
            self.take_profit = self.orderblock_indicator.last_bear_orderblock['y1']
            self.is_accepted_by_strategy = True
        else:
            self.is_accepted_by_strategy = False
