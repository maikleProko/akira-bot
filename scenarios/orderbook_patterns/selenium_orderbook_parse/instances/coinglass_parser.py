import re
from selenium.webdriver.common.by import By
from scenarios.orderbook_patterns.selenium_orderbook_parse.abstracts.selenium_orderbook_parser import (
    SeleniumOrderbookParser
)
import os
import json
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

COINGLASS_URL = 'https://www.coinglass.com/ru/mergev2/'

DEAL_PRICE_CLASS = 'obv2-item-price'
DEAL_AMOUNT_CLASS = 'obv2-item-amount'
DEAL_TOTAL_CLASS = 'obv2-item-total'
DEAL_TOTAL_SUBCLASS = 'MuiBox-root'


class CoinglassParser(SeleniumOrderbookParser):

    def parse_current_price(self):
        # Оставил заглушку — можно реализовать аналогично parse_deals, если есть подходящий селектор
        return 1

    def _normalize_number(self, s):
        """
        Приводит строку вроде "1,234.56" или "1 234,56" или "1 234" к int/float/None.
        Попытка сохранить разумное поведение для запятой как десятичного разделителя.
        """
        if s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        s = s.replace('\xa0', '').replace(' ', '').replace(' ', '')  # убрать NBSP и пробелы
        # Заменим запятую на точку (если использовалась как десятичный разделитель)
        s = s.replace(',', '.')
        # Оставим только цифры, точку и минус
        s = re.sub(r'[^0-9\.\-]', '', s)
        if not s or s == '-' or s == '.':
            return None
        # Если несколько точек — оставим последнюю как разделитель
        if s.count('.') > 1:
            parts = s.split('.')
            s = ''.join(parts[:-1]) + '.' + parts[-1]
        try:
            if '.' in s:
                return float(s)
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return None

    def parse_deals(self, css_selector):
        """
        Возвращает список словарей {'price': int_or_none, 'amount': float_or_none, 'total': str_or_none}
        css_selector: пример '.obv2-item.asks' или '.obv2-item.bids'
        """
        # JS: собрать текстовые значения для каждого элемента — вернёт простой список диктов (не WebElement)
        script = """
        const selector = arguments[0];
        const priceSel = arguments[1];
        const amountSel = arguments[2];
        const totalSel = arguments[3];
        const totalSub = arguments[4];
        return Array.from(document.querySelectorAll(selector)).map(el => {
            const priceEl = el.querySelector(priceSel);
            const amountEl = el.querySelector(amountSel);
            const totalContainer = el.querySelector(totalSel);
            let totalText = null;
            if (totalContainer) {
                const sub = totalContainer.querySelector('.' + totalSub);
                totalText = (sub ? sub : totalContainer).textContent;
            }
            return {
                price: priceEl ? priceEl.textContent.trim() : null,
                amount: amountEl ? amountEl.textContent.trim() : null,
                total: totalText ? totalText.trim() : null
            };
        });
        """
        raw = self.driver.execute_script(
            script,
            css_selector,
            f'.{DEAL_PRICE_CLASS}',
            f'.{DEAL_AMOUNT_CLASS}',
            f'.{DEAL_TOTAL_CLASS}',
            DEAL_TOTAL_SUBCLASS
        )

        deals = []
        for item in raw:
            price_raw = item.get('price')
            amount_raw = item.get('amount')
            total_raw = item.get('total')

            price = self._normalize_number(price_raw)
            # в оригинале вы приводили price к int — если нужно целое, оставьте int; здесь уже нормализовано
            if isinstance(price, float) and price.is_integer():
                price = int(price)

            amount = self._normalize_number(amount_raw)
            # total оставляем строкой (можно тоже нормализовать, если нужно число)
            total = self._normalize_number(total_raw)
            deals.append({
                'price': price,
                'amount': amount,
                'total': total_raw
            })
        return deals

    def parse_bids(self):
        return self.parse_deals('.obv2-item.bids')

    def parse_orderbook(self):
        return {
            'bids': self.parse_bids(),
            'asks': self.parse_deals('.obv2-item.asks')
        }

    def go_coinglass(self):
        # Перейти без ожидания (ваш метод). После навигации лучше подождать элемент, который гарантирует,
        # что страница нужная и JS подгрузился:
        self.go_no_wait(f'{COINGLASS_URL}{self.symbol1}-{self.symbol2}')
        # Ждём элемент-контейнер (в вашем коде был ant-row) — get_element предположительно блокирующий
        self.get_element(By.CLASS_NAME, 'ant-row')

    def run(self):
        pass
