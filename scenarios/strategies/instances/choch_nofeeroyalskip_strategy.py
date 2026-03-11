from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.bos_indicator import BosIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class CHoCHNoFeeRoyalSkipStrategy(Strategy):
    def __init__(
            self,
            history_market_parser_1m: HistoryMarketParser,
            kama_indicator_240m: KamaIndicator,
            choch_indicator_15m: CHoCHIndicator,
            choch_indicator_2h: CHoCHIndicator,
            bos_indicator_2h: BosIndicator,
            nwe_bounds_indicator: NweBoundsIndicator,
    ):
        super().__init__()
        self.history_market_parser_1m = history_market_parser_1m
        self.kama_indicator_240m = kama_indicator_240m
        self.choch_indicator_15m = choch_indicator_15m
        self.choch_indicator_2h = choch_indicator_2h
        self.bos_indicator_2h = bos_indicator_2h
        self.nwe_bounds_indicator = nwe_bounds_indicator

    def _is_signal(self) -> bool:
        close = self.history_market_parser_1m.df['close'].iloc[-1]
        choch_15m_ok = (
            self.choch_indicator_15m.is_now_CHoCH
            and self.choch_indicator_15m.choch_cross_price <= close
        )
        choch_2h_ok = (
            self.choch_indicator_2h.is_now_CHoCH
            and self.choch_indicator_2h.choch_cross_price <= close
        )
        bos_2h_ok = (
            self.bos_indicator_2h.is_now_BOS
            and self.bos_indicator_2h.bos_cross_price <= close
        )
        return (
            (choch_15m_ok or choch_2h_ok or bos_2h_ok)
            and self.kama_indicator_240m.trend == "BULLISH"
        )

    def run_historical(self, start_time, current_time):
        self.is_accepted_by_strategy = self._is_signal()

    def run_realtime(self):
        self.is_accepted_by_strategy = self._is_signal()
