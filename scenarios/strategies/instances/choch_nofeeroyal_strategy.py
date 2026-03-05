from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.bos_indicator import BosIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class CHoCHNoFeeRoyalStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            kama_indicator_30m: KamaIndicator,
            kama_indicator_240m: KamaIndicator,
            choch_indicator_15m: CHoCHIndicator,
            choch_indicator_2h: CHoCHIndicator,
            bos_indicator_2h: BosIndicator,
            nwe_bounds_indicator: NweBoundsIndicator,
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.kama_indicator_30m = kama_indicator_30m
        self.kama_indicator_240m = kama_indicator_240m
        self.choch_indicator_15m = choch_indicator_15m
        self.choch_indicator_2h = choch_indicator_2h
        self.bos_indicator_2h = bos_indicator_2h
        self.nwe_bounds_indicator = nwe_bounds_indicator

    def run_historical(self, start_time, current_time):
        if ((
            self.choch_indicator_15m.is_now_CHoCH and self.choch_indicator_15m.choch_cross_price <=
            self.history_market_parser_1m.df['close'].iloc[-1]
        ) or (
            self.choch_indicator_2h.is_now_CHoCH and self.choch_indicator_2h.choch_cross_price <=
            self.history_market_parser_1m.df['close'].iloc[-1]
        ) or (
            self.bos_indicator_2h.is_now_BOS and self.bos_indicator_2h.bos_cross_price <=
            self.history_market_parser_1m.df['close'].iloc[-1]
        )) and \
                self.kama_indicator_240m.trend == "BULLISH" and self.kama_indicator_30m.trend == "BULLISH":
            self.is_accepted_by_strategy = True
        else:
            self.is_accepted_by_strategy = False

    def run_realtime(self):
        if ((
            self.choch_indicator_15m.is_now_CHoCH and self.choch_indicator_15m.choch_cross_price <=
            self.history_market_parser_1m.df['close'].iloc[-1]
        ) or (
            self.choch_indicator_2h.is_now_CHoCH and self.choch_indicator_2h.choch_cross_price <=
            self.history_market_parser_1m.df['close'].iloc[-1]
        ) or (
            self.bos_indicator_2h.is_now_BOS and self.bos_indicator_2h.bos_cross_price <=
            self.history_market_parser_1m.df['close'].iloc[-1]
        )) and \
                self.kama_indicator_240m.trend == "BULLISH" and self.kama_indicator_30m.trend == "BULLISH":
            self.is_accepted_by_strategy = True
        else:
            self.is_accepted_by_strategy = False
