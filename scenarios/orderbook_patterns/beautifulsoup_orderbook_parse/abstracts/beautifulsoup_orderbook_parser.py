# scenarios/orderbook_patterns/beautifulsoup_orderbook_parse/abstracts/beautifulsoup_orderbook_parser.py
from scenarios.orderbook_patterns.abstracts.depth_parser import DepthParser


class BeautifulsoupOrderbookParser(DepthParser):
    def __init__(self, symbol1='BTC', symbol2='USDT'):
        super().__init__()
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.parse_type = 'beautifulsoup'


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

    def close(self):
        """Закрытие ресурсов"""
        pass