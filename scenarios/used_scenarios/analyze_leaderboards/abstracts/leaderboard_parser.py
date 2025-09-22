from core.selenium.SeleniumProcessor import SeleniumProcessor


class LeaderboardParser(SeleniumProcessor):
    def __init__(self):
        self.traders = []
        self.trades = []
        super().__init__()