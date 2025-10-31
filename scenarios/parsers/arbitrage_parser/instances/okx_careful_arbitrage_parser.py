import time

import requests
from okx import MarketData as okxMarket, Trade as okxTrade, Account as okxAccount

from scenarios.parsers.arbitrage_parser.abstracts.careful_arbitrage_parser import CarefulArbitrageParser


class OkxCarefulArbitrageParser(CarefulArbitrageParser):
    OKX_API = "https://www.okx.com"

    def init_clients(self, api_key, api_secret, api_passphrase):
        self.market_client = okxMarket.MarketAPI(api_key=api_key, api_secret_key=api_secret, passphrase=api_passphrase, flag='0')
        self.trade_client = okxTrade.TradeAPI(api_key=api_key, api_secret_key=api_secret, passphrase=api_passphrase, flag='0')
        self.user_client = okxAccount.AccountAPI(api_key=api_key, api_secret_key=api_secret, passphrase=api_passphrase, flag='0')

    def test_clients(self):
        try:
            accounts = self.user_client.get_account_balance()
            if accounts['code'] != '0':
                raise Exception(f"Ошибка авторизации API: {accounts['msg']}")
            if not accounts['data']:
                raise Exception("Тестовый запрос API для балансов вернул пустой список (возможно, неверные ключи)")
            self.log_message(f"Авторизация успешна. Список балансов: {accounts}")
        except Exception as e:
            self.log_message(f"Ошибка при тестировании клиентов: {str(e)}")
            raise  # Поднимаем ошибку, чтобы прервать выполнение, если авторизация не удалась

    def test_ticker(self):
        ticker_response = requests.get(f"{self.OKX_API}/api/v5/market/ticker?instId=ETH-USDT", timeout=10)
        ticker_response.raise_for_status()
        ticker = ticker_response.json()['data'][0]
        if not ticker or 'bidPx' not in ticker or 'askPx' not in ticker:
            raise Exception("Недопустимый ответ тикера")
        self.log_message(f"Тикер для ETH-USDT: {ticker}")

    def fetch_symbols(self):
        print("Запрашиваю данные с OKX... (может занять пару секунд)")
        url = self.OKX_API + "/api/v5/public/instruments?instType=SPOT"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()['data']
        except Exception as e:
            self.log_message(f"Ошибка при получении символов: {str(e)}")
            return []

    def fetch_tickers(self):
        if self.production and self.market_client:
            all_tickers = self.market_client.get_tickers(instType='SPOT')
            return all_tickers['data']
        else:
            response = requests.get(f"{self.OKX_API}/api/v5/market/tickers?instType=SPOT", timeout=10)
            response.raise_for_status()
            return response.json()['data']

    def check_balance(self, asset):
        if not self.production:
            return float('inf')
        try:
            balance = self.user_client.get_account_balance(ccy=asset)
            if balance['code'] != '0':
                raise Exception(f"Ошибка API: {balance['msg']}")
            if balance['data'] and balance['data'][0]['details']:
                available = float(balance['data'][0]['details'][0]['availBal'])
                self.log_message(f"Баланс {asset}: {available:.8f}")
                return available
            self.log_message(f"Баланс {asset}: 0.0 (детали не найдены)")
            return 0.0
        except KeyError as e:
            self.log_message(f"KeyError при проверке баланса для {asset}: {str(e)}. Полный ответ: {balance}")
            return 0.0
        except Exception as e:
            self.log_message(f"Ошибка при проверке баланса для {asset}: {str(e)}")
            return 0.0

    def fetch_ticker_price(self, symbol):
        try:
            response = requests.get(f"{self.OKX_API}/api/v5/market/ticker?instId={symbol}", timeout=5)
            response.raise_for_status()
            data = response.json()['data'][0]
            if not data or 'bidPx' not in data or 'askPx' not in data:
                return None
            return {'sell': float(data['bidPx']), 'buy': float(data['askPx'])}
        except Exception as e:
            return None

    def create_order_params(self, symbol, direction, ordType, adjusted_amount):
        side = 'sell' if direction == 'sell' else 'buy'
        params = {
            'instId': symbol,
            'tdMode': 'cash',
            'side': side,
            'ordType': ordType,
            'sz': str(adjusted_amount)
        }
        if direction == 'sell':
            params['tgtCcy'] = 'base_ccy'  # Не обязательно, default, но для ясности
        else:
            params['tgtCcy'] = 'quote_ccy'  # Для buy: sz - сумма quote-валюты
        return params

    def place_order(self, order_params):
        order = self.trade_client.place_order(**order_params)
        if order['code'] != '0':
            raise Exception(f"Ошибка размещения ордера: {order['msg']}")
        return order['data'][0]['ordId']

    def monitor_order(self, symbol, order_id, direction, amount, from_asset, to_asset, expected_price, fee_rate):
        start_time = time.time()
        while time.time() - start_time < 5:
            order_details = self.trade_client.get_order(instId=symbol, ordId=order_id)
            if order_details is None or not order_details['data']:
                self.log_message(f"Ошибка: get_order для {order_id} вернул None")
                return None, False
            details = order_details['data'][0]
            if details['state'] == 'filled':
                execution_time = time.time() - start_time
                filled_amount = float(details['accFillSz'])
                avg_price = float(details['avgPx'])
                dealt_funds = filled_amount * avg_price
                actual_price = avg_price  # Исправление: для обоих направлений actual_price = avgPx (цена сделки)
                # Точный расчет net amount с использованием 'fee' из details
                fee = float(details['fee'])  # fee negative, paid amount
                fee_ccy = details['feeCcy']
                if direction == 'sell':
                    actual_amount = dealt_funds + fee if fee_ccy == to_asset else dealt_funds  # fee deducted from received
                else:
                    actual_amount = filled_amount + fee if fee_ccy == to_asset else filled_amount
                self.log_message(f"Транзакция {direction} {amount:.8f} {from_asset} -> {to_asset} завершена за {execution_time:.2f} сек.")
                self.log_message(f"Ожидаемая цена: {expected_price:.8f}, Фактическая средняя цена: {actual_price:.8f}")
                self.log_message(f"Ожидаемое количество: {(amount * expected_price * (1 - fee_rate) if direction == 'sell' else (amount / expected_price) * (1 - fee_rate)):.8f} {to_asset}, Фактическое количество: {actual_amount:.8f} {to_asset}")
                return actual_amount, True
            time.sleep(0.1)
        self.log_message(f"Транзакция {direction} {amount:.8f} {from_asset} -> {to_asset} не завершилась за 5 секунд, отменяется.")
        self.trade_client.cancel_order(instId=symbol, ordId=order_id)
        return None, False

    def fetch_current_prices(self):
        try:
            if self.production and self.market_client:
                all_tickers = self.market_client.get_tickers(instType='SPOT')
                tickers = all_tickers['data']
            else:
                response = requests.get(f"{self.OKX_API}/api/v5/market/tickers?instType=SPOT", timeout=10)
                response.raise_for_status()
                tickers = response.json()['data']
            return {t['instId']: {'sell': float(t['bidPx']), 'buy': float(t['askPx'])} for t in tickers if 'bidPx' in t and 'askPx' in t}
        except Exception as e:
            self.log_message(f"Ошибка при получении всех тикеров для проверки: {str(e)}")
            return None