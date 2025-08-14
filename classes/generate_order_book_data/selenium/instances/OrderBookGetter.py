from asyncio import sleep

from classes.generate_order_book_data.selenium.abstracts.SeleniumProcessor import SeleniumProcessor
from classes.generate_order_book_data.selenium.constants.OrderBookGetterConstants import *


class OrderBookGetter(SeleniumProcessor):
    def get_order_book_screen(self):
        self.go_no_check(CHART_URL)
        sleep(7)
        self.screen_selector(PANE_SELECTOR, PANE_FILENAME, 300)
        self.screen_selector(AXIS_SELECTOR, AXIS_FILENAME)

    def run(self):
        self.get_order_book_screen()
