from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.orderblock_indicator import OrderblockIndicator
from scenarios.parsers.indicators.instances.bos_indicator import BosIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class CHoCHPercentStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            history_market_parser_15m: HistoryMarketParser,
            kama_indicator_60m: KamaIndicator,
            kama_indicator_15m: KamaIndicator,
            choch_indicator: CHoCHIndicator,
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.history_market_parser_15m = history_market_parser_15m
        self.kama_indicator_60m = kama_indicator_60m
        self.kama_indicator_15m = kama_indicator_15m
        self.choch_indicator = choch_indicator


    def run_historical(self, start_time, current_time):
        if (
                self.choch_indicator.is_now_CHoCH and self.choch_indicator.choch_cross_price <=
                self.history_market_parser_1m.df['close'].iloc[-1]
        ) and \
                self.kama_indicator_15m.trend == "BULLISH" and \
                self.kama_indicator_15m.trend2 == "BULLISH" and \
                self.kama_indicator_60m.trend == "BULLISH" and \
                self.kama_indicator_60m.trend2 == "BULLISH" and \
                self.kama_indicator_60m.trend3 == "BULLISH":
            self.is_accepted_by_strategy = True
        else:
            self.is_accepted_by_strategy = False
