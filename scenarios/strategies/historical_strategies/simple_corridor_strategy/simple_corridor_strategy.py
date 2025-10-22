from utils.core.functions import MarketProcess

from utils.machine_learning.lstm_acsending_channel import CorridorModelPredictor


class SimpleCorridorStrategy(MarketProcess):
    def __init__(
        self,
        history_market_parser,
    ):
        self.history_market = history_market_parser
        self.symbol = history_market_parser.slash_symbol

    def prepare(self, start_time=None, end_time=None):
        pass

    def run_historical(self, start_time, current_time):
        predictor = CorridorModelPredictor('lstm_v1', self.history_market.symbol1 + self.history_market.symbol2)
        predictor.predict(self.history_market.df)