from scenarios.reverberate.abstracts.reverberation_parser import ReverberationParser


class ReverberationController:
    def __init__(self, reverberation_parsers: list[ReverberationParser]):
        self.reverberation_parsers = reverberation_parsers
        self.run()

    def run(self):
        for reverberation_parser in self.reverberation_parsers:
            reverberation_parser.go_page()

        while 1:
            for reverberation_parser in self.reverberation_parsers:
                reverberation_parser.generate_particles()