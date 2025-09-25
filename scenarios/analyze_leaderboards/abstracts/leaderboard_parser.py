from core.selenium.SeleniumProcessor import SeleniumProcessor


class LeaderboardParser(SeleniumProcessor):
    def __init__(self):
        if not self.file_path:
            self.file_path = '/files/common_traders.json'
        self.traders = []
        self.trades = []
        super().__init__()