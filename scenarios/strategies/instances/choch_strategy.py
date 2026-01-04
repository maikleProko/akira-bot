from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.OrderblockIndicator import OrderblockIndicator
from scenarios.parsers.indicators.instances.bos_indicator import BosIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class CHoCHStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            history_market_parser_15m: HistoryMarketParser,
            kama_indicator_60m: KamaIndicator,
            kama_indicator_15m: KamaIndicator,
            kama_indicator_1m: KamaIndicator,
            bos_indicator: BosIndicator,
            choch_indicator: CHoCHIndicator,
            nwe_bounds_indicator: NweBoundsIndicator,
            orderblock_indicator: OrderblockIndicator
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.history_market_parser_15m = history_market_parser_15m
        self.kama_indicator_60m = kama_indicator_60m
        self.kama_indicator_15m = kama_indicator_15m
        self.kama_indicator_1m = kama_indicator_1m
        self.choch_indicator = choch_indicator
        self.bos_indicator = bos_indicator
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.orderblock_indicator = orderblock_indicator

    def calculate_av_ob(self):
        try:
            av_ob1 = self.orderblock_indicator.last_bull_orderblock['y1'] + \
                     self.orderblock_indicator.last_bull_orderblock['y2']
            av_ob2 = self.orderblock_indicator.last_bear_orderblock['y1'] + \
                     self.orderblock_indicator.last_bear_orderblock['y2']
            return (av_ob1 + av_ob2) / 2
        except:
            return 0

    def run_historical(self, start_time, current_time):
        if (
                self.choch_indicator.is_now_CHoCH and self.choch_indicator.choch_cross_price <=
                self.history_market_parser_1m.df['close'].iloc[-1]

                or

                self.bos_indicator.is_now_BOS and self.bos_indicator.bos_cross_price <=
                self.history_market_parser_1m.df['close'].iloc[-1]
        ) and \
                self.kama_indicator_15m.trend == "BULLISH" and \
                self.kama_indicator_15m.trend2 == "BULLISH" and \
                self.kama_indicator_15m.trend3 == "BULLISH" and \
                self.kama_indicator_60m.trend == "BULLISH" and \
                self.kama_indicator_60m.trend2 == "BULLISH":
            self.is_accepted_by_strategy = True
        else:
            self.is_accepted_by_strategy = False
