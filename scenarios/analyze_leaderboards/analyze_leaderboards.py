from scenarios.analyze_leaderboards.abstracts.leaderboard_analyzer import LeaderboardAnalyzer
from scenarios.analyze_leaderboards.instances.binance_parser import BinanceParser
from scenarios.analyze_leaderboards.instances.binance_passive_parser import BinancePassiveParser


def analyze_leaderboards():
    binance_parser = BinancePassiveParser()
    leaderboard_analyzer = LeaderboardAnalyzer(binance_parser.traders)
    print(leaderboard_analyzer.process())
    print(leaderboard_analyzer)
