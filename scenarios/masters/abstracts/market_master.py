from utils.core.functions import MarketProcess


class MarketMaster(MarketProcess):
    def __init__(self):
        self.market_processes: list[MarketProcess] = []

    def finalize(self):
        for market_process in self.market_processes:
            market_process.finalize()

    def prepare(self, start_time=None, end_time=None):
        for market_process in self.market_processes:
            market_process.prepare(start_time, end_time)

    def run(self, start_time=None, end_time=None):
        for market_process in self.market_processes:
            market_process.run(start_time, end_time)