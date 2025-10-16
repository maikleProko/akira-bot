from scenarios.reverberate.selenium_reverberate.abstracts.reverberation_parser import ReverberationParser


class ReverberationController:
    def __init__(self, reverberation_parsers: list[ReverberationParser]):
        self.reverberation_parsers = reverberation_parsers
        self.run()

    def run(self, start_time=None, current_time=None, end_time=None):
        for reverberation_parser in self.reverberation_parsers:
            reverberation_parser.go_page()

        while 1:
            for reverberation_parser in self.reverberation_parsers:
                reverberation_parser.generate_particles()