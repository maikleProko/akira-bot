import json
import time
from decimal import Decimal, ROUND_CEILING
import requests
from pybit.unified_trading import HTTP
from scenarios.parsers.arbitrage_parser.core.careful_arbitrage_parser import CarefulArbitrageParser
from scenarios.parsers.arbitrage_parser.core.utils.exchange_client import ExchangeClient


class BybitExchangeClient(ExchangeClient):
    BYBIT_API = "https://api.bybit.com"

    def __init__(self, logger):
        self.session = None
        self.logger = logger
        self.restricted_pairs = set()

    def init_clients(self, api_key, api_secret, api_passphrase):
        self.session = HTTP(api_key=api_key, api_secret=api_secret)

    def test_clients(self):
        if self.session is None:
            raise Exception("Session not initialized")
        balance = self.session.get_wallet_balance(accountType="UNIFIED")
        if balance['retCode'] != 0:
            raise Exception(f"Ошибка авторизации API: {balance['retMsg']}")
        if not balance['result']:
            raise Exception("Тестовый запрос API для балансов вернул пустой список (возможно, неверные ключи)")
        return balance

    def test_ticker(self):
        ticker_response = requests.get(f"{self.BYBIT_API}/v5/market/tickers?category=spot&symbol=ETHUSDT", timeout=10)
        ticker_response.raise_for_status()
        data = ticker_response.json()
        if data['retCode'] != 0 or not data['result']['list']:
            raise Exception("Недопустимый ответ тикера")
        ticker = data['result']['list'][0]
        if not ticker or 'bid1Price' not in ticker or 'ask1Price' not in ticker:
            raise Exception("Недопустимый ответ тикера")
        return ticker

    def fetch_available_coins(self):
        raise NotImplementedError

    def normalize_symbols(self, symbols):
        normalized = []
        for item in symbols:
            if item['status'] != 'Trading':
                continue
            norm = {
                'instId': item['symbol'],
                'baseCcy': item['baseCoin'],
                'quoteCcy': item['quoteCoin'],
                'state': 'live',
                'minSz': item['lotSizeFilter']['minOrderQty'],
                'lotSz': item['lotSizeFilter']['basePrecision'],
                'tickSz': item['priceFilter']['tickSize']
            }
            normalized.append(norm)
        return normalized

    def fetch_symbols(self):
        print("Запрашиваю данные с Bybit... (может занять пару секунд)")
        if self.session:
            response = self.session.get_instruments_info(category="spot")
            if response['retCode'] != 0:
                raise Exception(f"Ошибка при получении символов: {response['retMsg']}")
            symbols = response['result']['list']
        else:
            response = requests.get(f"{self.BYBIT_API}/v5/market/instruments-info?category=spot", timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['retCode'] != 0:
                raise Exception(f"Ошибка при получении символов: {data['retMsg']}")
            symbols = data['result']['list']
        return self.normalize_symbols(symbols)

    def normalize_tickers(self, tickers):
        normalized = []
        for t in tickers:
            norm = {
                'instId': t['symbol'],
                'bidPx': t['bid1Price'],
                'askPx': t['ask1Price']
            }
            normalized.append(norm)
        return normalized

    def fetch_market_tickers(self):
        if self.session:
            response = self.session.get_tickers(category="spot")
            if response['retCode'] != 0:
                raise Exception(f"Ошибка при получении тикеров: {response['retMsg']}")
            return self.normalize_tickers(response['result']['list'])
        return None

    def fetch_public_tickers(self):
        response = requests.get(f"{self.BYBIT_API}/v5/market/tickers?category=spot", timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['retCode'] != 0:
            raise Exception(f"Ошибка при получении тикеров: {data['retMsg']}")
        return self.normalize_tickers(data['result']['list'])

    def fetch_tickers(self):
        tickers = self.fetch_market_tickers()
        if tickers is not None:
            return tickers
        return self.fetch_public_tickers()

    def check_balance(self, asset):
        if self.session is None:
            raise Exception("Session not initialized for private endpoint")
        balance = self.session.get_wallet_balance(accountType="UNIFIED", coin=asset)
        if balance['retCode'] != 0:
            raise Exception(f"Ошибка API: {balance['retMsg']}")
        if balance['result']['balances']:
            return float(balance['result']['balances'][0]['availableBalance'])
        return 0.0

    def fetch_ticker_price(self, symbol):
        response = requests.get(f"{self.BYBIT_API}/v5/market/tickers?category=spot&symbol={symbol}", timeout=5)
        response.raise_for_status()
        data = response.json()
        if data['retCode'] != 0 or not data['result']['list']:
            return None
        ticker_data = data['result']['list'][0]
        return {'sell': float(ticker_data['bid1Price']), 'buy': float(ticker_data['ask1Price'])}

    async def fetch_ticker_async(self, session, symbol):
        url = f"{self.BYBIT_API}/v5/market/tickers?category=spot&symbol={symbol}"
        async with session.get(url) as resp:
            data = await resp.json()
            if data['retCode'] != 0 or not data['result']['list']:
                return None
            ticker_data = data['result']['list'][0]
            return {'name': symbol, 'sell': float(ticker_data['bid1Price']), 'buy': float(ticker_data['ask1Price'])}




    def set_common_params(self, params, symbol, direction, ordType, adjusted_amount):
        side = 'Sell' if direction == 'sell' else 'Buy'
        params['category'] = 'spot'
        params['symbol'] = symbol
        params['side'] = side
        params['orderType'] = 'Market' if ordType == 'market' else 'Limit'

    def set_currency_params(self, params, direction, ordType, adjusted_amount):
        if ordType == 'market':
            if direction == 'sell':
                params['qty'] = str(adjusted_amount)
            else:
                params['quoteCoinQty'] = str(adjusted_amount)
        else:
            params['qty'] = str(adjusted_amount)

    def create_order_params(self, symbol, direction, ordType, adjusted_amount):
        params = {}
        self.set_common_params(params, symbol, direction, ordType, adjusted_amount)
        self.set_currency_params(params, direction, ordType, adjusted_amount)
        return params

    def place_order(self, order_params):
        if self.session is None:
            raise Exception("Session not initialized for private endpoint")
        order = self.session.place_order(**order_params)
        if order['retCode'] != 0:
            raise Exception(f"Ошибка размещения ордера: code={order['retCode']}, msg={order['retMsg']}, data={order.get('result', 'N/A')}")
        return order['result']['orderId']

    def cancel_order(self, symbol, order_id):
        if self.session is None:
            raise Exception("Session not initialized for private endpoint")
        self.session.cancel_order(category="spot", symbol=symbol, orderId=order_id)

    def get_order_details(self, symbol, order_id):
        if self.session is None:
            raise Exception("Session not initialized for private endpoint")
        return self.session.get_order_history(category="spot", symbol=symbol, orderId=order_id)

    def fetch_market_prices(self):
        if self.session:
            response = self.session.get_tickers(category="spot")
            if response['retCode'] != 0:
                raise Exception(f"Ошибка при получении тикеров: {response['retMsg']}")
            return response['result']['list']
        return None

    def fetch_public_prices(self):
        response = requests.get(f"{self.BYBIT_API}/v5/market/tickers?category=spot", timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['retCode'] != 0:
            raise Exception(f"Ошибка при получении тикеров: {data['retMsg']}")
        return data['result']['list']


    def build_prices_dict(self, tickers):
        return {t['symbol']: {'sell': float(t['bid1Price']), 'buy': float(t['ask1Price'])} for t in tickers if 'bid1Price' in t and 'ask1Price' in t and t['bid1Price'] != '' and t['ask1Price'] != ''}

    def fetch_current_prices(self):
        tickers = self.fetch_market_prices()
        if tickers is not None:
            return self.build_prices_dict(tickers)
        tickers = self.fetch_public_prices()
        return self.build_prices_dict(tickers)

    def get_size_decimals(self, entry):
        base_min_size_dec = Decimal(str(entry.get('base_min_size', '0.0001')))
        quote_min_size_dec = Decimal(str(entry.get('quote_min_size', '5.0')))
        base_increment_dec = Decimal(str(entry.get('base_increment', '1e-8')))
        quote_increment_dec = Decimal(str(entry.get('quote_increment', '0.01')))
        return base_min_size_dec, quote_min_size_dec, base_increment_dec, quote_increment_dec

    def get_ticker_decimals(self, ticker):
        bid_dec = Decimal(str(ticker['sell']))
        ask_dec = Decimal(str(ticker['buy']))
        return bid_dec, ask_dec

    def prepare_sell_order(self, base_min_size_dec, base_increment_dec, ask_dec):
        ratio_dec = base_min_size_dec / base_increment_dec
        ceil_ratio_dec = ratio_dec.to_integral_value(rounding=ROUND_CEILING)
        adjusted_min_size_dec = ceil_ratio_dec * base_increment_dec
        bad_price_dec = ask_dec * Decimal('1.1')
        return str(adjusted_min_size_dec), str(bad_price_dec)

    def prepare_buy_order(self, quote_min_size_dec, quote_increment_dec, bid_dec):
        min_quote_dec = quote_min_size_dec
        ratio_dec = min_quote_dec / quote_increment_dec
        ceil_ratio_dec = ratio_dec.to_integral_value(rounding=ROUND_CEILING)
        adjusted_min_size_dec = ceil_ratio_dec * quote_increment_dec
        bad_price_dec = bid_dec * Decimal('0.9')
        return str(adjusted_min_size_dec), str(bad_price_dec)

    def attempt_check_pair(self, sym, direction, price_map, entry, sizes, ticker):
        base_min_size_dec, quote_min_size_dec, base_increment_dec, quote_increment_dec = sizes
        bid_dec, ask_dec = self.get_ticker_decimals(ticker)
        if direction == 'sell':
            sz, px = self.prepare_sell_order(base_min_size_dec, base_increment_dec, ask_dec)
        else:
            sz, px = self.prepare_buy_order(quote_min_size_dec, quote_increment_dec, bid_dec)
        order_params = self.create_order_params(sym, direction, 'limit', sz)
        order_params['price'] = px
        order_id = self.place_order(order_params)
        self.cancel_order(sym, order_id)
        return True

    def handle_check_error(self, e, sym):
        error_str = str(e)
        if 'compliance' in error_str.lower():
            self.logger.log_message(f"Пара {sym} заблокирована по compliance")
            self.restricted_pairs.add(sym)
            return False
        return None

    def check_pair_available(self, sym, direction, price_map):
        if sym in self.restricted_pairs:
            return False
        for attempt in range(3):
            try:
                entry = price_map.get(sym, {})
                sizes = self.get_size_decimals(entry)
                ticker = self.fetch_ticker_price(sym)
                if ticker is None:
                    raise Exception("Failed to fetch ticker for bad_price")
                return self.attempt_check_pair(sym, direction, price_map, entry, sizes, ticker)
            except Exception as e:
                result = self.handle_check_error(e, sym)
                if result is not None:
                    return result
                self.logger.log_message(f"Проверка пары {sym} на доступность покупки успешна.")
                return True
        self.logger.log_message(f"Проверка пары {sym} провалилась после 3 попыток. Считаем недоступной.")
        return False

    def check_order_filled(self, details, direction, to_asset):
        filled_amount = float(details.get('cumExecQty', 0))
        avg_price = float(details.get('avgPrice', 0))
        dealt_funds = filled_amount * avg_price
        fee = float(details.get('cumExecFee', 0))
        fee_ccy = details.get('feeAsset', '')
        if direction == 'sell':
            return dealt_funds - fee if fee_ccy == to_asset else dealt_funds
        return filled_amount - fee if fee_ccy == to_asset else filled_amount

    def log_successful_trade(self, execution_time, direction, amount, from_asset, to_asset, expected_price, actual_price, expected_amount, actual_amount, fee_rate):
        self.logger.log_message(f"Транзакция {direction} {amount:.8f} {from_asset} -> {to_asset} завершена за {execution_time:.2f} сек.")
        self.logger.log_message(f"Ожидаемая цена: {expected_price:.8f}, Фактическая средняя цена: {actual_price:.8f}")
        self.logger.log_message(f"Ожидаемое количество: {expected_amount:.8f} {to_asset}, Фактическое количество: {actual_amount:.8f} {to_asset}")

    def calculate_expected_amount(self, direction, amount, expected_price, fee_rate, to_asset):
        if direction == 'sell':
            return amount * expected_price * (1 - fee_rate)
        return (amount / expected_price) * (1 - fee_rate)

    def monitor_order_loop(self, symbol, order_id, direction, amount, from_asset, to_asset, expected_price, fee_rate, start_time):
        while time.time() - start_time < 5:
            order_details = self.get_order_details(symbol, order_id)
            if order_details['retCode'] != 0 or not order_details['result']:
                self.logger.log_message(f"Ошибка: get_order для {order_id} вернул ошибку")
                return None, False
            details = order_details['result'][0]
            if details['orderStatus'] == 'Filled':
                execution_time = time.time() - start_time
                avg_price = float(details['avgPrice'])
                actual_amount = self.check_order_filled(details, direction, to_asset)
                expected_amount = self.calculate_expected_amount(direction, amount, expected_price, fee_rate, to_asset)
                self.log_successful_trade(execution_time, direction, amount, from_asset, to_asset, expected_price, avg_price, expected_amount, actual_amount, fee_rate)
                return actual_amount, True
            time.sleep(0.1)
        self.logger.log_message(f"Транзакция {direction} {amount:.8f} {from_asset} -> {to_asset} не завершилась за 5 секунд, отменяется.")
        self.cancel_order(symbol, order_id)
        return None, False

    def monitor_order(self, symbol, order_id, direction, amount, from_asset, to_asset, expected_price, fee_rate):
        start_time = time.time()
        return self.monitor_order_loop(symbol, order_id, direction, amount, from_asset, to_asset, expected_price, fee_rate, start_time)


class BybitCarefulArbitrageParser(CarefulArbitrageParser):
    def create_exchange_client(self, logger):
        return BybitExchangeClient(logger)