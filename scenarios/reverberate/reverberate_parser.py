from scenarios.reverberate.history_reverberate.history_reverberate import *
from utils.functions import MarketProcess


class ReverberateParser(MarketProcess):
    def __init__(self, platform_name = 'binance', symbol = 'BTCUSDT', delta_minutes = 5):
        self.platform_name = platform_name
        self.symbol = symbol
        self.parser = None
        self.delta_minutes = delta_minutes
        self.reverberations = []

    def prepare(self):
        self.parser = get_parser(self.platform_name)

    def run(self):
        datetime_now = datetime.now() - timedelta(hours=3)
        self.reverberations = self.parser.reverberate_in_period(
            start_datetime = datetime_now - timedelta(minutes=self.delta_minutes),
            end_datetime= datetime_now,
            symbol=self.symbol,
            multi_window_minutes=DEFAULT_MULTI_WINDOW_MINUTES
        )
        save_to_file(self.reverberations, self.platform_name, self.symbol)
