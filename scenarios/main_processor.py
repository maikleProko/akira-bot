from datetime import datetime, timedelta

from scenarios.constants import MARKET_PROCESSES
from utils.decorators import periodic, every_second


class MainProcessor:

    def run(self):
        pass

    def prepare(self):
        print('Preparing...')
        for market_process in MARKET_PROCESSES:
            market_process.prepare()

    def __init__(self):
        self.prepare()
        self.run()


class RealtimeProcessor(MainProcessor):

    @periodic()
    def run(self, start_time=None, current_time=None, end_time=None):
        for market_process in MARKET_PROCESSES:
            market_process.run()


class HistoricalProcessor(MainProcessor):
    def __init__(self, start_time_string, end_time_string, minutes_interval=1):
        super().__init__()
        self.start_time_string = start_time_string
        self.end_time_string = end_time_string
        self.minutes_interval = minutes_interval

    def run(self, start_time=None, current_time=None, end_time=None):
        start_time = datetime.strptime(self.start_time_string, "%Y/%m/%d %H:%M")
        end_time = datetime.strptime(self.end_time_string, "%Y/%m/%d %H:%M")
        current_time = start_time
        while current_time <= end_time:
            for market_process in MARKET_PROCESSES:
                market_process.run(start_time, current_time, end_time)
            current_time += timedelta(minutes=self.minutes_interval)
