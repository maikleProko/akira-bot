from scenarios.other.analyze_leaderboards.abstracts.leaderboard_parser import LeaderboardParser


class BinancePassiveParser(LeaderboardParser):
    def __init__(self, file_path: str = 'files/binance_traders.json'):
        self.file_path = file_path
        super().__init__()

    def run_realtime(self):
        self._load_traders()
