from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.supertrend_indicator import SupertrendIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class CHoCHChippiStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            kama_indicator_120m: KamaIndicator,
            supertrend_indicator_15m: SupertrendIndicator,
            supertrend_indicator_1m: SupertrendIndicator,
            choch_indicator_1m: CHoCHIndicator,
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.kama_indicator_120m = kama_indicator_120m
        self.supertrend_indicator_15m = supertrend_indicator_15m
        self.supertrend_indicator_1m = supertrend_indicator_1m
        self.choch_indicator_1m = choch_indicator_1m

    def _check_signal(self):
        kama_120m_bullish = self.kama_indicator_120m.trend == "BULLISH"
        supertrend_15m_bullish = self.supertrend_indicator_15m.values.get('direction') == 1
        supertrend_1m_bullish = self.supertrend_indicator_1m.values.get('direction') == 1
        has_choch = self.choch_indicator_1m.is_now_CHoCH or self.choch_indicator_1m.is_prev_CHoCH
        self.is_accepted_by_strategy = (
            kama_120m_bullish
            and supertrend_15m_bullish
            and supertrend_1m_bullish
            and has_choch
        )

    def run_historical(self, start_time, current_time):
        self._check_signal()

    def run_realtime(self):
        self._check_signal()
