from scenarios.reverberate.reverberate import reverberate


#@doing_periodical_per_1_minute
def run():
    reverberate()
    #analyze_leaderboards()


class MainProcessor:
    def __init__(self):
        run()