from files.other.analyze_leaderboards.abstracts.leaderboard_parser import LeaderboardParser

LEADERBOARD_URL = ''

class BybitParser(LeaderboardParser):
    def __init__(self, file_path: str = 'files/bybit_traders.json'):
        self.file_path = file_path
        super().__init__()


    def go_leaderboard(self) -> None:
        self.go_no_check(LEADERBOARD_URL)

    def get_traders_all(self):
        pass

    def parse_all_trades(self):
        pass
