from utils.core.functions import MarketProcess


class Strategy(MarketProcess):
    def __init__(self):
        self.is_accepted_by_strategy = False

    def run_historical(self, start_time, current_time):
        self.is_accepted_by_strategy = True

    def run_realtime(self):
        self.is_accepted_by_strategy = True

