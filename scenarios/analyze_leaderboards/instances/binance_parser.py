import json
import logging
from time import sleep
from typing import List, Dict, Any

from selenium.common import ElementClickInterceptedException, WebDriverException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from datetime import datetime


from scenarios.analyze_leaderboards.abstracts.leaderboard_parser import LeaderboardParser

LEADERBOARD_URL = 'https://www.binance.com/ru/copy-trading'
TRADERS_XPATH = '/html/body/div[3]/div[2]/div/div[3]/div[2]'
TRADER_CARD_CLASS = 'card-outline'
A_TRADER_ELEMENT = 'bn-balink'
PAGES_ELEMENT = 'bn-pagination-items'
TRADES_XPATH = '/html/body/div[3]/div[2]/div/div[4]/div[2]/div[2]/div[2]/div/div[1]/div/div/div[2]/table/tbody'
TRADE_CARD_CLASS = 'bn-web-table-row'

logger = logging.getLogger(__name__)


class BinanceParser(LeaderboardParser):
    def __init__(self, file_path: str = 'files/binance_traders.json'):
        self.file_path = file_path
        self.traders: List[Dict[str, Any]] = []
        self.root_is_dict = False
        self.per_trader_list_key = 'trades'   # will be autodetected on load
        super().__init__()

    # ----- public flow -----
    def run(self) -> None:
        self.go_leaderboard()
        #self.get_traders_all()
        self.parse_all_trades()

    def go_leaderboard(self) -> None:
        self.go_no_check(LEADERBOARD_URL)

    # ----- traders collection -----
    def get_traders_all(self) -> None:
        pages = self._element(By.CLASS_NAME, PAGES_ELEMENT)
        last_page = int(pages.find_element(By.CSS_SELECTOR, ":last-child").text)
        logger.info("Pages count: %d", last_page)
        for p in range(1, last_page):
            self.get_traders_from_page(p)

    def get_traders_from_page(self, page: int) -> None:
        pages = self._element(By.CLASS_NAME, PAGES_ELEMENT)
        page_el = self.find_element_by_text(pages, page)
        self._click(page_el); sleep(2)
        cards = self._element(By.XPATH, TRADERS_XPATH).find_elements(By.CLASS_NAME, TRADER_CARD_CLASS)
        for card in cards:
            a_el = self.get_inner_element(card, By.CLASS_NAME, A_TRADER_ELEMENT)
            url = a_el.get_attribute('href')
            self.traders.append({'url': url, self.per_trader_list_key: []})
        self._save_traders()

    # ----- trades parsing -----
    def parse_all_trades(self) -> None:
        self._load_traders()
        for i, trader in enumerate(self.traders):
            self.parse_trader(trader, i)

    def parse_trader(self, trader: Dict[str, Any], index: int) -> None:
        self.go(trader['url'])
        self.click_history()
        pages_element = self._element(By.CLASS_NAME, PAGES_ELEMENT)
        pages = pages_element.find_elements(By.CLASS_NAME, 'bn-pagination-item')
        print(pages)
        print(len(pages))
        last_page = int(pages[-1].text)
        logger.info("Trades pages count: %d", last_page)
        for p in range(1, last_page):
            self.get_trades_from_page(index, p)


    def get_trades_from_page(self, trader_index, page):
        pages = self._element(By.CLASS_NAME, PAGES_ELEMENT)
        page_el = self.find_element_by_text(pages, page)
        self._click(page_el); sleep(2)
        self.get_trades_for_trader(trader_index)

    def click_history(self):
        self._click(self.get_element_using_text('История позиций'))
        sleep(0.5)

    def get_trades_for_trader(self, index: int) -> None:
        table = self._element(By.XPATH, TRADES_XPATH)
        rows = table.find_elements(By.CLASS_NAME, TRADE_CARD_CLASS)
        parsed = [self._parse_trade_row(r) for r in rows]
        print(parsed)
        self.traders[index].setdefault(self.per_trader_list_key, []).extend(parsed)
        self._save_traders()

    # ----- small helpers -----
    def get_value_of_param(self, param_name, row):
        param_name_element = self.get_element_in_element_using_text(row, param_name)
        print(param_name_element.text)
        parent = self.get_parent_element(param_name_element)
        return parent.find_element(By.XPATH, "//*[@class='t-caption2 text-PrimaryText']").text

    def get_trade_params(self, row):
        return self.get_parent_element(self.get_parent_element(self.get_element_in_element_using_text(row, 'Открыто')))

    def _parse_trade_row(self, row) -> Dict[str, Any]:
        return {
            'open-date': datetime.strptime(self.get_value_of_param('Открыто', row), "%Y-%m-%d %H:%M:%S"),
            'close-date': datetime.strptime(self.get_value_of_param('Закрыто', row), "%Y-%m-%d %H:%M:%S"),
            'type': row.find_elements(By.CLASS_NAME, 'bn-bubble-content')[1].text
        }

    def _load_traders(self) -> None:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self.traders = []; return
        if isinstance(data, dict) and 'traders' in data:
            self.root_is_dict = True; self.traders = data['traders']
        elif isinstance(data, list):
            self.traders = data
        else:
            self.traders = []
        if self.traders and isinstance(self.traders[0], dict):
            for k in ('trades', 'traders'):
                if k in self.traders[0]:
                    self.per_trader_list_key = k; break

    def _save_traders(self) -> None:
        payload = {'traders': self.traders} if self.root_is_dict else self.traders
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False, default=str)

    def _element(self, by: By, selector: str):
        return self.get_element(by, selector)

    def _click(self, el) -> None:
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            sleep(0.2); ActionChains(self.driver).move_to_element(el).click().perform(); return
        except (ElementClickInterceptedException, WebDriverException):
            pass
        try:
            self.driver.execute_script(
                "var f = document.getElementById('trade-status-footer-v2'); if (f) { f.style.display='none'; }"
            )
            sleep(0.1); self.driver.execute_script("arguments[0].click();", el); return
        except Exception:
            pass
        try:
            self.driver.execute_script("arguments[0].click();", el); return
        except Exception as e:
            logger.exception("Click failed"); raise ElementClickInterceptedException(f"Не удалось кликнуть: {e}")