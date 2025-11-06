import time
from decimal import Decimal, getcontext, ROUND_CEILING
from scenarios.parsers.arbitrage_parser.core.careful_arbitrage_parser import CarefulArbitrageParser
from scenarios.parsers.arbitrage_parser.core.utils.exchange_client import ExchangeClient
getcontext().prec = 28
import requests
from okx import MarketData as okxMarket, Trade as okxTrade, Account as okxAccount



class OkxExchangeClient(ExchangeClient):
    OKX_API = "https://www.okx.com"

    def __init__(self, logger):
        self.market_client = None
        self.trade_client = None
        self.user_client = None
        self.logger = logger
        self.restricted_pairs = set()

    def init_clients(self, api_key, api_secret, api_passphrase):
        self.market_client = okxMarket.MarketAPI(api_key=api_key, api_secret_key=api_secret, passphrase=api_passphrase, flag='0')
        self.trade_client = okxTrade.TradeAPI(api_key=api_key, api_secret_key=api_secret, passphrase=api_passphrase, flag='0')
        self.user_client = okxAccount.AccountAPI(api_key=api_key, api_secret_key=api_secret, passphrase=api_passphrase, flag='0')

    def test_clients(self):
        accounts = self.user_client.get_account_balance()
        if accounts['code'] != '0':
            raise Exception(f"Ошибка авторизации API: {accounts['msg']}")
        if not accounts['data']:
            raise Exception("Тестовый запрос API для балансов вернул пустой список (возможно, неверные ключи)")
        return accounts

    def test_ticker(self):
        ticker_response = requests.get(f"{self.OKX_API}/api/v5/market/ticker?instId=ETH-USDT", timeout=10)
        ticker_response.raise_for_status()
        ticker = ticker_response.json()['data'][0]
        if not ticker or 'bidPx' not in ticker or 'askPx' not in ticker:
            raise Exception("Недопустимый ответ тикера")
        return ticker

    def fetch_available_coins(self):
        raise NotImplementedError

    def fetch_symbols(self):
        print("Запрашиваю данные с OKX... (может занять пару секунд)")
        url = self.OKX_API + "/api/v5/public/instruments?instType=SPOT"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['data']

    def fetch_market_tickers(self):
        if self.market_client:
            all_tickers = self.market_client.get_tickers(instType='SPOT')
            return all_tickers['data']
        return None

    def fetch_public_tickers(self):
        response = requests.get(f"{self.OKX_API}/api/v5/market/tickers?instType=SPOT", timeout=10)
        response.raise_for_status()
        return response.json()['data']

    def fetch_tickers(self):
        tickers = self.fetch_market_tickers()
        if tickers is not None:
            return tickers
        return self.fetch_public_tickers()

    def check_balance(self, asset):
        balance = self.user_client.get_account_balance(ccy=asset)
        if balance['code'] != '0':
            raise Exception(f"Ошибка API: {balance['msg']}")
        if balance['data'] and balance['data'][0]['details']:
            return float(balance['data'][0]['details'][0]['availBal'])
        return 0.0

    def fetch_ticker_price(self, symbol):
        response = requests.get(f"{self.OKX_API}/api/v5/market/ticker?instId={symbol}", timeout=5)
        response.raise_for_status()
        data = response.json()['data'][0]
        if not data or 'bidPx' not in data or 'askPx' not in data:
            return None
        return {'sell': float(data['bidPx']), 'buy': float(data['askPx'])}



    def set_common_params(self, params, symbol, direction, ordType, adjusted_amount):
        side = 'sell' if direction == 'sell' else 'buy'
        params['instId'] = symbol
        params['tdMode'] = 'cash'
        params['side'] = side
        params['ordType'] = ordType
        params['sz'] = str(adjusted_amount)

    def set_currency_params(self, params, direction):
        if direction == 'sell':
            params['tgtCcy'] = 'base_ccy'
        else:
            params['tgtCcy'] = 'quote_ccy'

    def create_order_params(self, symbol, direction, ordType, adjusted_amount):
        params = {}
        self.set_common_params(params, symbol, direction, ordType, adjusted_amount)
        self.set_currency_params(params, direction)
        return params

    def place_order(self, order_params):
        order = self.trade_client.place_order(**order_params)
        if order['code'] != '0':
            raise Exception(f"Ошибка размещения ордера: code={order['code']}, msg={order['msg']}, data={order.get('data', 'N/A')}")
        if not order['data']:
            raise Exception("Ордер размещён, но data пусто — возможно, batch issue.")
        return order['data'][0]['ordId']

    def cancel_order(self, symbol, order_id):
        self.trade_client.cancel_order(instId=symbol, ordId=order_id)

    def get_order_details(self, symbol, order_id):
        return self.trade_client.get_order(instId=symbol, ordId=order_id)

    def fetch_market_prices(self):
        if self.market_client:
            all_tickers = self.market_client.get_tickers(instType='SPOT')
            tickers = all_tickers['data']
            return tickers
        return None

    def fetch_public_prices(self):
        response = requests.get(f"{self.OKX_API}/api/v5/market/tickers?instType=SPOT", timeout=10)
        response.raise_for_status()
        return response.json()['data']

    def get_nullable_float(self, thing):
        try:
            return float(thing)
        except:
            return 0

    def build_prices_dict(self, tickers):
        return {t['instId']: {'sell': self.get_nullable_float(t['bidPx']), 'buy': self.get_nullable_float(t['askPx'])} for t in tickers if 'bidPx' in t and 'askPx' in t and t['bidPx'] != '' and t['askPx'] != ''}

    def fetch_current_prices(self):
        print('1')
        tickers = self.fetch_market_prices()
        print('2')
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
        order_params['px'] = px
        order_id = self.place_order(order_params)
        self.cancel_order(sym, order_id)
        return True

    def handle_check_error(self, e, sym):
        error_str = str(e)
        if '51155' in error_str or 'compliance' in error_str.lower():
            self.logger.log_message(f"Пара {sym} заблокирована по compliance (ошибка 51155)")
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
                self.logger.log_message(f"Проверка пары {sym} на доступность покупки успешна. Retry...")
                return True
        self.logger.log_message(f"Проверка пары {sym} провалилась после 3 попыток. Считаем недоступной.")
        return False

    def check_order_filled(self, details, direction, to_asset):
        filled_amount = float(details['accFillSz'])
        avg_price = float(details['avgPx'])
        dealt_funds = filled_amount * avg_price
        fee = float(details['fee'])
        fee_ccy = details['feeCcy']
        if direction == 'sell':
            return dealt_funds + fee if fee_ccy == to_asset else dealt_funds
        return filled_amount + fee if fee_ccy == to_asset else filled_amount

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
            if order_details is None or not order_details['data']:
                self.logger.log_message(f"Ошибка: get_order для {order_id} вернул None")
                return None, False
            details = order_details['data'][0]
            if details['state'] == 'filled':
                execution_time = time.time() - start_time
                avg_price = float(details['avgPx'])
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


class OkxCarefulArbitrageParser(CarefulArbitrageParser):
    def create_exchange_client(self, logger):
        return OkxExchangeClient(logger)