from files.other.analyze_leaderboards.abstracts.leaderboard_analyzer import LeaderboardAnalyzer
from files.other.analyze_leaderboards.instances.binance_parser import BinanceParser


def analyze_leaderboards():
    binance_parser = BinanceParser()
    leaderboard_analyzer = LeaderboardAnalyzer(binance_parser.traders)
    print(leaderboard_analyzer.process())
