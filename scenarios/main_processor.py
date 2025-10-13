from datetime import datetime

from scenarios.constants import MARKET_PROCESSES
from utils.decorators import periodic, every_second


class MainProcessor:

    @periodic()
    def run(self):
        for market_process in MARKET_PROCESSES:
            market_process.run()

    def prepare(self):
        print('Preparing...')
        for market_process in MARKET_PROCESSES:
            market_process.prepare()

    def __init__(self):
        self.prepare()
        self.run()
