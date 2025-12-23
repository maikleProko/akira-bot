from utils.core.functions import MarketProcess


class Strategy(MarketProcess):
    def __init__(self):
        self.is_entry_to_deal = False