from selenium.webdriver.common.by import By

from scenarios.reverberate.selenium_reverberate.abstracts.reverberation_parser import *


class BinanceReverberationParser(ReverberationParser):
    def __init__(self):
        self.url = 'https://www.binance.com/en/trade/BTC_USDT?type=spot'
        self.market = 'binance'
        super().__init__()

    def is_have_particles(self):
        try:
            float(clean_string(self.get_element(By.CLASS_NAME, 'compare-percent-buy').text))
            float(clean_string(self.get_element(By.CLASS_NAME, 'compare-percent-sell').text))
            return True
        except:
            return False

    def get_particles(self):
        return {
            'buy_value': float(clean_string(self.driver.find_element(By.CLASS_NAME, 'compare-percent-buy').text)),
            'sell_value': float(clean_string(self.driver.find_element(By.CLASS_NAME, 'compare-percent-sell').text))
        }
