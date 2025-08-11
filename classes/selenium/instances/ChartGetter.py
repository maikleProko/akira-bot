from asyncio import sleep

from classes.selenium.abstracts.SeleniumProcessor import SeleniumProcessor
from classes.selenium.constants.ChartGetterConstants import *


class ChartGetter(SeleniumProcessor):
    def get_chart_screen(self):
        self.go_no_check(CHART_URL)
        sleep(7)
        self.screen_selector(PANE_SELECTOR, PANE_FILENAME, 300)
        self.screen_selector(AXIS_SELECTOR, AXIS_FILENAME)

    def run(self):
        self.get_chart_screen()
