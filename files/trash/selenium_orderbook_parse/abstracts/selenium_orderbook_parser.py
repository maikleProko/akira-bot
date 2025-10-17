from utils.selenium.SeleniumProcessor import SeleniumProcessor
from scenarios.parsers.orderbook_parser.abstracts.depth_parser import DepthParser


class SeleniumOrderbookParser(SeleniumProcessor, DepthParser):
    def __init__(self, symbol1='BTC', symbol2='USDT'):
        super().__init__()
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.platform_name = 'selenium'


    def go_page(self):
        """Переход к странице"""
        pass

    def parse_bids(self):
        """Парсинг заявок на покупку"""
        pass

    def parse_asks(self):
        """Парсинг заявок на продажу"""
        pass

    def parse_orderbook(self):
        """Парсинг полного ордербука"""
        pass

    def parse_current_price(self):
        """Парсинг текущей цены"""
        pass

