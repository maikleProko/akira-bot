from scenarios.reverberate.selenium_reverberate.abstracts.reverberation_controller import ReverberationController
from scenarios.reverberate.selenium_reverberate.instances.binance_reverberation_parser import BinanceReverberationParser


def selenium_reverberate():
    reverberation_controller = ReverberationController([
        BinanceReverberationParser()
    ])
