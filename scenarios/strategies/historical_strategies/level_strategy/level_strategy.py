from asyncio import sleep
from datetime import timedelta

from scenarios.strategies.historical_strategies.level_strategy.services.orderbook_depth_evaluator import \
    OrderbookDepthEvaluator
from utils.core.functions import MarketProcess


class LevelStrategy(MarketProcess):
    def __init__(
        self,
        history_market_parser,
        orderbook_parser,
        nwe_bounds_indicator,
        atr_bounds_indicator,
        evaluator_cls=OrderbookDepthEvaluator,
        evaluator_kwargs=None
    ):
        self.history_market = history_market_parser
        self.orderbook = orderbook_parser
        self.nwe = nwe_bounds_indicator
        self.atr = atr_bounds_indicator
        self.symbol = history_market_parser.slash_symbol
        self.evaluator = evaluator_cls(**(evaluator_kwargs or {}))

    def process(self, start_time, end_time):
        print('[LevelStrategy] Start analyzing')
        levels = self.evaluator.define_depth_level(start_time, end_time)
        print(levels)
        print('[LevelStrategy] Analysis complete')
        return levels

    def prepare(self, start_time=None, end_time=None):
        self.process(start_time, end_time)

    def run_historical(self, start_time, current_time):
        self.process(current_time-timedelta(hours=4), current_time)

