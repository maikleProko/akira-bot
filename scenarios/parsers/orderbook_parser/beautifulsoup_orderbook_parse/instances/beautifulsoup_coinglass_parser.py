import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import re
from scenarios.parsers.orderbook_parser.beautifulsoup_orderbook_parse.abstracts.beautifulsoup_orderbook_parser import (
    BeautifulsoupOrderbookParser
)
from typing import Optional, Dict, List, Any

COINGLASS_URL = 'https://www.coinglass.com/ru/mergev2/'

DEAL_PRICE_CLASS = 'obv2-item-price'
DEAL_AMOUNT_CLASS = 'obv2-item-amount'
DEAL_TOTAL_CLASS = 'obv2-item-total'
DEAL_TOTAL_SUBCLASS = 'MuiBox-root'


class BeautifulsoupCoinglassParser(BeautifulsoupOrderbookParser):
    def __init__(self, symbol1='BTC', symbol2='USDT', timeout=30000, viewport_height=12000, zoom=0.25):
        super().__init__()
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.platform_name = 'coinglass'
        self.timeout = timeout
        self.viewport_height = viewport_height  # Увеличили высоту для большего количества строк
        self.zoom = zoom  # Уменьшаем зум, чтобы влезло больше
        self.playwright = None
        self.browser = None
        self.page = None
        self.soup = None
        self._loop = None
        self._initialized = False
        self.timer = 60
        self._refresh_counter = 0  # Счетчик для обновления страницы раз в минуту (60 вызовов по 1 сек)
        self._init_loop()

    def _init_loop(self):
        """Инициализация единого event loop для экземпляра"""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def _get_event_loop(self):
        """Получаем event loop (всегда один и тот же)"""
        return self._loop

    async def _init_browser(self):
        """Инициализация Playwright браузера"""
        if self._initialized:
            return

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()

        # Настройка User-Agent и viewport
        await self.page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        await self.page.set_viewport_size({"width": 1920, "height": self.viewport_height})
        await self.page.evaluate(f"document.body.style.zoom = '{self.zoom}'")  # Уменьшаем зум
        self._initialized = True

    def _normalize_number(self, s: Optional[str]) -> Optional[float]:
        """Нормализация числовых значений (синхронная версия)"""
        if not s:
            return None
        s = str(s).strip()
        if not s:
            return None
        s = s.replace('\xa0', '').replace(' ', '').replace(' ', '')
        s = s.replace(',', '.')
        s = re.sub(r'[^0-9\.\-]', '', s)
        if not s or s == '-' or s == '.':
            return None
        if s.count('.') > 1:
            parts = s.split('.')
            s = ''.join(parts[:-1]) + '.' + parts[-1]
        try:
            if '.' in s:
                return float(s)
            return int(s)
        except:
            return None

    def go_page(self):
        """Синхронный wrapper для go_page_async (инициализация и первая загрузка)"""
        loop = self._get_event_loop()
        loop.run_until_complete(self._go_page_async())

    async def _go_page_async(self):
        """Асинхронная реализация go_page"""
        await self._init_browser()

        url = f'{COINGLASS_URL}{self.symbol1}-{self.symbol2}'

        await self.page.goto(url, wait_until='networkidle', timeout=self.timeout)

        # Ждем появления элементов ордербука или другого индикатора загрузки
        try:
            await self.page.wait_for_selector('.obv2-item, .ant-row', timeout=15000)
        except Exception as e:
            # Ждем дополнительно
            await asyncio.sleep(3)

        # Симулируем скролл для подгрузки больше строк (адаптируйте селектор контейнера по inspect)
        try:
            await self.page.evaluate("""
                const containers = document.querySelectorAll('.obv2-container, .ant-row, .scroll-container');  // Возможные селекторы контейнера
                containers.forEach(container => {
                    if (container) {
                        container.scrollTop = container.scrollHeight;
                    }
                });
            """)
            await asyncio.sleep(2)  # Ждем подгрузки после скролла
        except Exception as e:
            print(f'[OrderbookParser({self.parse_type}_{self.platform_name}_parser)] Ошибка скролла: {e}')

        # Получаем HTML после рендеринга JS
        html_content = await self.page.content()
        self.soup = BeautifulSoup(html_content, 'html.parser')

        # Проверяем наличие контента
        ant_row = self.soup.find('div', class_='ant-row')
        obv_items = self.soup.select('.obv2-item')

        if not ant_row:
            print(f'[OrderbookParser({self.parse_type}_{self.platform_name}_parser)] Предупреждение: ant-row не найден для {self.symbol1}-{self.symbol2}')

    def refresh_soup(self):
        """Метод для обновления soup без полной перезагрузки страницы (для цикличного run())"""
        loop = self._get_event_loop()
        loop.run_until_complete(self._refresh_soup_async())

    async def _refresh_soup_async(self):
        """Асинхронное обновление soup"""
        if not self._initialized:
            await self._go_page_async()  # Если не инициализировано, делаем полный go
            return

        self._refresh_counter += 1

        if self._refresh_counter >= 5:  # Обновляем страницу раз в минуту (60 секунд)
            await self.page.reload(wait_until='networkidle', timeout=self.timeout)
            self._refresh_counter = 0
            # Ждем появления элементов после reload
            try:
                await self.page.wait_for_selector('.obv2-item, .ant-row', timeout=15000)
            except Exception as e:
                await asyncio.sleep(3)
        else:
            # Для динамических обновлений без reload просто ждем немного
            await asyncio.sleep(1)  # Небольшая пауза для потенциального JS-обновления

        # Всегда повторяем скролл, на случай если reload или изменения сбросили его
        try:
            await self.page.evaluate("""
                const containers = document.querySelectorAll('.obv2-container, .ant-row, .scroll-container');
                containers.forEach(container => {
                    if (container) {
                        container.scrollTop = container.scrollHeight;
                    }
                });
            """)
            await asyncio.sleep(2)
        except:
            pass

        # Всегда обновляем soup из текущего рендера страницы (учитывает динамические изменения)
        html_content = await self.page.content()
        self.soup = BeautifulSoup(html_content, 'html.parser')

    def parse_deals(self, css_selector: str) -> List[Dict[str, Any]]:
        """Синхронный парсинг сделок"""
        if not self.soup:
            raise ValueError("Сначала вызовите go_page() для загрузки страницы")

        deal_elements = self.soup.select(css_selector)
        deals = []

        # Убрали [:20] — парсим все доступные элементы
        for element in deal_elements:
            price_el = element.select_one(f'.{DEAL_PRICE_CLASS}')
            amount_el = element.select_one(f'.{DEAL_AMOUNT_CLASS}')
            total_container = element.select_one(f'.{DEAL_TOTAL_CLASS}')

            price_raw = price_el.get_text(strip=True) if price_el else None
            amount_raw = amount_el.get_text(strip=True) if amount_el else None
            total_raw = None

            if total_container:
                total_sub = total_container.select_one(f'.{DEAL_TOTAL_SUBCLASS}')
                total_raw = (total_sub.get_text(strip=True) if total_sub
                             else total_container.get_text(strip=True))

            price = self._normalize_number(price_raw)
            if isinstance(price, float) and price.is_integer():
                price = int(price)

            amount = self._normalize_number(amount_raw)
            total = self._normalize_number(total_raw)

            deals.append({
                'price': price,
                'amount': amount,
                'total': total
            })

        return deals

    def parse_bids(self):
        """Парсинг заявок на покупку"""
        return self.parse_deals('.obv2-item.bids')

    def parse_asks(self):
        """Парсинг заявок на продажу"""
        return self.parse_deals('.obv2-item.asks')

    def parse_orderbook(self):
        """Парсинг полного ордербука"""
        self.go_page()

        if not self.soup:
            raise ValueError("Сначала вызовите go_page()")
        return {
            'bids': self.parse_bids(),
            'asks': self.parse_asks()
        }

    def parse_current_price(self):
        """Парсинг текущей цены"""
        if not self.soup:
            raise ValueError("Сначала вызовите go_page()")

        # Попробуем разные селекторы для цены
        selectors = [
            '[data-testid="current-price"]',
            '.current-price',
            '.price-value',
            '.obv2-current-price',
            '.cg-price',
            'h1, h2, h3',  # Заголовки
            '.MuiTypography-h4, .MuiTypography-h3'  # Material-UI типографика
        ]

        for selector in selectors:
            elements = self.soup.select(selector)
            for el in elements[:5]:  # Проверяем первые 5
                text = el.get_text(strip=True)
                if text and len(text) > 2:  # Игнорируем слишком короткие тексты
                    price = self._normalize_number(text)
                    if price and price > 100:  # Примерная проверка на разумную цену BTC
                        return price

        print(f'[OrderbookParser({self.parse_type}_{self.platform_name}_parser)] Цена не найдена')
        return None

    def close(self):
        """Синхронный wrapper для закрытия"""
        loop = self._get_event_loop()
        loop.run_until_complete(self._close_async())
        if not loop.is_running():
            loop.close()

    async def _close_async(self):
        """Асинхронное закрытие"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self._initialized = False
        print(f'[OrderbookParser({self.parse_type}_{self.platform_name}_parser)] Браузер закрыт')

    def __enter__(self):
        self.go_page()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()