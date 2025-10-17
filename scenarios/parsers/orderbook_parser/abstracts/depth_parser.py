


class DepthParser:
    def __init__(self, symbol1='BTC', symbol2='USDT'):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.parse_type = ''
        self.platform_name = ''


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