from abc import ABC, abstractmethod
from datetime import datetime

from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.strategies.strategy import Strategy
from utils.core.functions import MarketProcess


class RegulatorTPSL(MarketProcess, ABC):
    def __init__(
        self,
        history_market_parser: HistoryMarketParser,
        strategy: Strategy,
        risk_amount: float = 0.01,
        rr_ratio: float = 1.5
    ):
        self.history_market_parser = history_market_parser
        self.strategy = strategy
        self.risk_amount = risk_amount
        self.rr_ratio = rr_ratio
        self.take_profit = None
        self.stop_loss = None
        self.symbol1_prepared_converted_amount = None
        self.is_accepted_by_regulator = False

    @abstractmethod
    def calculate_tpsl(self):
        """Calculate TP, SL, and amount"""
        pass

    def _tick(self, current_time: datetime):
        if self.strategy.is_accepted_by_strategy == True and self.is_accepted_by_regulator == False:
            try:
                self.calculate_tpsl()
            except Exception as e:
                print(e)

    def run_realtime(self):
        self._tick(datetime.now())

    def run_historical(self, start_time: datetime, current_time: datetime):
        self._tick(current_time)

    def run(self, start_time=None, current_time=None):
        if start_time is None and current_time is None:
            self.run_realtime()
        else:
            self.run_historical(start_time, current_time)