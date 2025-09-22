from scenarios.used_scenarios.analyze_leaderboards.analyze_leaderboards import analyze_leaderboards
from utils.decorators import doing_periodical_per_1_minute


#@doing_periodical_per_1_minute
def run():
    analyze_leaderboards()


class MainProcessor:
    def __init__(self):
        run()