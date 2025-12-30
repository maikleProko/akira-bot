from files.other.reverberate.selenium_reverberate.abstracts.reverberation_controller import ReverberationController
from files.other.reverberate.selenium_reverberate.instances.binance_reverberation_parser import BinanceReverberationParser


def selenium_reverberate():
    reverberation_controller = ReverberationController([
        BinanceReverberationParser()
    ])
