from asyncio import sleep
from datetime import datetime

from PIL import Image
from selenium.webdriver import Keys
from selenium.webdriver.common.by import By

from core.selenium.SeleniumProcessor import SeleniumProcessor
from scenarios.unused_scenarios.generate_order_book_data.constants.OrderBookGetterConstants import *


class OrderBookGetter(SeleniumProcessor):

    def get_one_order_book_screen(self):
        screen = self.get_screen(By.CSS_SELECTOR, FULL_SELECTOR)
        self.tap(Keys.ARROW_DOWN)
        return screen

    def get_cycle_order_book_screen(self):
        screens = []
        for i in range(0, 6):
            screens.append(self.get_one_order_book_screen())

        max_width = max(screen.width for screen in screens)

        total_height = sum(screen.height for screen in screens)
        final_screen = Image.new('RGB', (max_width, total_height))

        y_position = 0
        for screen in screens:
            x_offset = (max_width - screen.width) // 2

            final_screen.paste(screen, (x_offset, y_position))
            y_position += screen.height

        return final_screen

    def get_screen_1(self):
        print('1')
        self.click(By.XPATH, CHANGER_TIME_XPATH)
        print('2')
        self.tap(Keys.NUMPAD1)
        print('3')
        self.tap(Keys.ENTER)
        print('4')
        return self.get_cycle_order_book_screen()

    def get_screen_5(self):
        self.click(By.XPATH, CHANGER_TIME_XPATH)
        self.tap(Keys.NUMPAD5)
        self.tap(Keys.ENTER)
        return self.get_cycle_order_book_screen()

    def get_merged_screen(self):
        screen_1 = self.get_screen_1()
        screen_5 = self.get_screen_5()
        total_width = screen_1.width + screen_5.width
        max_height = max(screen_1.height, screen_5.height)
        merged_screen = Image.new('RGB', (total_width, max_height))
        merged_screen.paste(screen_1, (0, 0))
        merged_screen.paste(screen_5, (screen_1.width, 0))
        return merged_screen

    def get_order_book_screen(self):
        self.go_no_check(CHART_URL)
        sleep(7)

        now = datetime.now()
        self.click(By.CSS_SELECTOR, LIST_ELEMENT_SELECTOR)
        img = self.get_merged_screen()
        img.save('images_panes/pane' + now.strftime('%Y-%m-%d_%H-%M') + '.png')

    def run(self):
        self.get_order_book_screen()
