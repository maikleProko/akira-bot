from scenarios.analyze_leaderboards.abstracts.leaderboard_parser import LeaderboardParser
from typing import List, Dict, Any

class BinancePassiveParser(LeaderboardParser):
    def __init__(self, file_path: str = 'files/binance_traders.json'):
        self.file_path = file_path
        self.traders: List[Dict[str, Any]] = []
        self.root_is_dict = False
        self.per_trader_list_key = 'trades'   # will be autodetected on load
        super().__init__()

    def run(self):
        self._load_traders()
