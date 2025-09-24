from scenarios.reverberate.abstracts.reverberation_controller import ReverberationController
from scenarios.reverberate.instances.binance_reverberation_parser import BinanceReverberationParser


def reverberate():
    reverberation_controller = ReverberationController([
        BinanceReverberationParser()
    ])
