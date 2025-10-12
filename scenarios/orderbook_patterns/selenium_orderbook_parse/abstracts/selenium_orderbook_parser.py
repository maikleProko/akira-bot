from core.selenium.SeleniumProcessor import SeleniumProcessor


class SeleniumOrderbookParser(SeleniumProcessor):
    def __init__(self, symbol1='BTC', symbol2='USDT'):
        self.symbol1 = symbol1
        self.symbol2 = symbol2

        super().__init__()


