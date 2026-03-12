from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.bos_indicator import BosIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class CHoCHKimmiStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            kama_indicator_15m: KamaIndicator,
            kama_indicator_120m: KamaIndicator,
            choch_indicator_1m: CHoCHIndicator,
            bos_indicator_1m: BosIndicator,
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.kama_indicator_15m = kama_indicator_15m
        self.kama_indicator_120m = kama_indicator_120m
        self.choch_indicator_1m = choch_indicator_1m
        self.bos_indicator_1m = bos_indicator_1m

    def _check_signal(self):
        kama_15m_bullish = self.kama_indicator_15m.trend == "BULLISH"
        kama_120m_bullish = self.kama_indicator_120m.trend == "BULLISH"
        has_choch = self.choch_indicator_1m.is_now_CHoCH
        has_bos = self.bos_indicator_1m.is_now_BOS
        self.is_accepted_by_strategy = kama_15m_bullish and kama_120m_bullish and (has_choch or has_bos)

    def run_historical(self, start_time, current_time):
        self._check_signal()

    def run_realtime(self):
        self._check_signal()
