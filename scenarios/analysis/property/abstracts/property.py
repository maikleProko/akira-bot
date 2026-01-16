from utils.core.functions import MarketProcess


class Property:
    def __init__(self):
        self.market_processes: list[MarketProcess] = [

        ]

    def run_market_processes(self):
        for market_process in self.market_processes:
            market_process.run_realtime()

    def build_market_processes(self, deal):
        self.prepare(deal)
        self.run_market_processes()
        return self.analyze(deal)

    def prepare(self, deal):
        pass

    def analyze(self, deal):
        return deal