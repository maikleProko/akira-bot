from selenium.common import ElementClickInterceptedException, WebDriverException
from selenium.webdriver import ActionChains

from scenarios.used_scenarios.analyze_leaderboards.abstracts.leaderboard_parser import LeaderboardParser
from selenium.webdriver.common.by import By
from time import sleep

LEADERBOARD_URL = 'https://www.binance.com/ru/copy-trading'
TRADERS_ELEMENT = 'mt-[24px] grid gap-[16px] grid-cols-1 md:grid-cols-2 lg:grid-cols-3 w-full mb-[26px]'
TRADER = 'flex flex-col gap-[16px] card-outline p-[16px] md:p-[24px] hover:border-PrimaryYellow cursor-pointer no-underline'
A_TRADER_ELEMENT = 'bn-balink f-link no-underline inline-flex flex-col gap-[16px] text-PrimaryText'
PAGES_ELEMENT = 'bn-pagination-items'


class BinanceParser(LeaderboardParser):
    def parse_all_trades(self):
        pass

    def get_trades_for_trader(self):
        pass

    def parse_trader(self, trader, i):
        self.go_no_check(trader.url)
        sleep(3)


    def parse_traders(self):
        for trader, i in self.traders:
            self.parse_trader(trader, i)

    # Попробуем несколько стратегий клика последовательно
    def try_click(self, el):
        # 1) прокрутить в центр и клик через ActionChains
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            sleep(0.3)
            ActionChains(self.driver).move_to_element(el).click().perform()
            return
        except ElementClickInterceptedException:
            pass
        except WebDriverException:
            pass

        # 2) скрыть футер (если он есть) и кликнуть через JS
        try:
            self.driver.execute_script(
                "var f = document.getElementById('trade-status-footer-v2'); if (f) { f.style.display='none'; }"
            )
            sleep(0.2)
            self.driver.execute_script("arguments[0].click();", el)
            return
        except Exception:
            pass

        # 3) финальная попытка: просто JS click без скрытия
        try:
            self.driver.execute_script("arguments[0].click();", el)
            return
        except Exception as e:
            raise ElementClickInterceptedException(f"Не удалось кликнуть по элементу: {e}")

    def get_traders_from_page(self, page):
        pages_element = self.get_element(By.CLASS_NAME, PAGES_ELEMENT)
        page_element = self.find_element_by_text(pages_element, page)
        self.try_click(page_element)
        sleep(2)

        traders_element = self.get_element(By.CLASS_NAME, TRADERS_ELEMENT)
        traders = traders_element.find_elements(By.CLASS_NAME, TRADER)

        for trader in traders:
            a_trader_element = self.get_inner_element(trader, By.CLASS_NAME, A_TRADER_ELEMENT)
            trader_url = a_trader_element.get_attribute('href')

            trader = {
                'url': trader_url,
                'trades': [

                ]
            }

            self.traders.append(trader)



    def get_traders_all(self):
        pages_element = self.get_element(By.CLASS_NAME, PAGES_ELEMENT)
        last_page = int(pages_element.find_element(By.CSS_SELECTOR, ":last-child").text)
        print('number pages ' + str(last_page))

        for i in range(1, last_page):
            self.get_traders_from_page(i)

    def go_leaderboard(self):
        self.go_no_check(LEADERBOARD_URL)
        sleep(7)

    def run(self):
        self.go_leaderboard()
        self.get_traders_all()
        self.parse_all_trades()

