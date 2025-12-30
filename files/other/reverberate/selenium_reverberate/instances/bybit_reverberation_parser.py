from selenium.webdriver.common.by import By

from files.other.reverberate.selenium_reverberate.abstracts.reverberation_parser import *


class BybitReverberationParser(ReverberationParser):
    def __init__(self):
        self.url = 'https://www.bybit.com/en/trade/spot/BTC/USDT'
        self.market = 'bybit'
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
