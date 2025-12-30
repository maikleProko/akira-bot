import logging
from time import sleep
from typing import Dict, Any

from selenium.webdriver.common.by import By
from datetime import datetime
import re
from decimal import Decimal


def extract_number(s, as_decimal=False):
    if s is None:
        raise ValueError("input is None")
    # нормализуем некоторые символы минуса
    s = s.strip().replace('−', '-')
    # учитываем запись в скобках как отрицательное значение: "(480.99)" -> -480.99
    negative_by_parens = bool(re.search(r'\(\s*[-+]?\d', s))
    # найдём первое вхождение числа: опциональный знак, цифра, группы цифр (с запятыми/пробелами), дробная часть
    m = re.search(r'[-+]?\d[\d\s,]*\.?\d*', s)
    if not m:
        raise ValueError("No numeric value found in input")
    num = m.group(0)
    # удалить пробелы и разделители тысяч
    num = num.replace(' ', '').replace(',', '')
    if negative_by_parens and not num.startswith('-'):
        num = '-' + num
    return Decimal(num) if as_decimal else float(Decimal(num))

from files.other.analyze_leaderboards.abstracts.leaderboard_parser import LeaderboardParser

LEADERBOARD_URL = 'https://www.binance.com/ru/copy-trading'
TRADERS_XPATH = '/html/body/div[3]/div[2]/div/div[3]/div[2]'
TRADER_CARD_CLASS = 'card-outline'
A_TRADER_ELEMENT = 'bn-balink'
PAGES_ELEMENT = 'bn-pagination-items'
TRADES_XPATH = '/html/body/div[3]/div[2]/div/div[4]/div[2]/div[2]/div[2]/div/div[1]/div/div/div[2]/table/tbody'
TRADE_CARD_CLASS = 'bn-web-table-row'
TRADER_WINRATE_XPATH = '/html/body/div[3]/div[2]/div/div[3]/div[2]/div[1]/div/div[6]/div[2]'

logger = logging.getLogger(__name__)


class BinanceParser(LeaderboardParser):
    def __init__(self, file_path: str = 'files/binance_traders.json'):
        self.file_path = file_path
        super().__init__()

    # ----- public flow -----

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
        i = 0
        while i < len(self.traders):
            try:
                keep = self.parse_trader(self.traders[i], i)
                if not keep:
                    # удаляем текущего трейдера и НЕ увеличиваем i (следующий элемент смещается на текущую позицию)
                    self.traders.pop(i)
                else:
                    i += 1
            except Exception as e:
                print('error for parse trader ' + str(i) + ', continue: ' + str(e))
                i += 1

    def parse_trader(self, trader: Dict[str, Any], index: int) -> bool:
        self.go(trader['url'])
        self.click_history()
        pages_element = self._element(By.CLASS_NAME, PAGES_ELEMENT)
        pages = pages_element.find_elements(By.CLASS_NAME, 'bn-pagination-item')
        last_page = int(pages[-1].text)
        winrate = extract_number(self.get_element(By.XPATH, TRADER_WINRATE_XPATH).text) * 0.01

        if winrate < 0.8:
            print(f'trader {index} removed (winrate {winrate})')
            return False  # сигнализируем удаление
        # иначе сохраняем и парсим
        self.traders[index]['winrate'] = winrate
        if last_page > 12:
            last_page = 12
        for p in range(1, last_page + 1):
            self.get_trades_from_page(index, p)
        print(f'trader {index} parsed')
        return True


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
        parsed = []

        for row in rows:
            try:
                parsed.append(self._parse_trade_row(row))
            except Exception as e:
                print('error for parse trade: ' + str(e))

        self.traders[index].setdefault(self.per_trader_list_key, []).extend(parsed)
        print('trades of page ' + str(index) + ' parsed')
        self._save_traders()

    # ----- small helpers -----

    def get_value_of_trader_param(self, param_name, row):
        param_name_element = self.get_element_in_element_using_text(row, param_name)
        parent = self.get_parent_element(param_name_element)
        return parent.find_element(By.XPATH, ".//*[@class='t-subtitle2']").text

    def get_value_of_trade_param(self, param_name, row):
        param_name_element = self.get_element_in_element_using_text(row, param_name)
        parent = self.get_parent_element(param_name_element)
        return parent.find_element(By.XPATH,".//*[contains(concat('t-caption2 ', normalize-space(@class), ' '), ' text-PrimaryText ')]").text

    def get_datetime_value_of_param(self, param_name, row):
        value = self.get_value_of_trade_param(param_name, row)
        if value == '--':
            return datetime.now()
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")

    def get_type(self, row):
        text = row.find_elements(By.CLASS_NAME, 'bn-bubble-content')[1].text

        if text == "Изолированная\nШорт":
            text = 'isolated_short'

        if text == "Изолированная\nЛонг":
            text = 'isolated_long'

        if text == "Кросс\nШорт":
            text = 'cross_short'

        if text == "Кросс\nЛонг":
            text = 'cross_long'

        return text



    def _parse_trade_row(self, row) -> Dict[str, Any]:
        trade_params = self.get_trade_params(row)
        return {
            'open_date': self.get_datetime_value_of_param('Открыто', trade_params),
            'close_date': self.get_datetime_value_of_param('Закрыто', trade_params),
            'symbol': row.find_elements(By.CLASS_NAME, 't-subtitle1')[0].text,
            'type': self.get_type(row),
            'pnl': extract_number(self.get_value_of_trade_param('PnL после закрытия позиций', row)),
            'is_profit': extract_number(self.get_value_of_trade_param('PnL после закрытия позиций', row)) >= 0
        }


    def get_trade_params(self, row):
        return self.get_parent_element(self.get_parent_element(self.get_element_in_element_using_text(row, 'Открыто')))
