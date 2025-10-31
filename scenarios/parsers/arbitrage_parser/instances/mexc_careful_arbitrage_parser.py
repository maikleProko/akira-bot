import time
import requests
import hmac
import hashlib

from scenarios.parsers.arbitrage_parser.abstracts.careful_arbitrage_parser import CarefulArbitrageParser



class MexcCarefulArbitrageParser(CarefulArbitrageParser):
    MEXC_API = "https://api.mexc.com"

    def _signed_request(self, method, endpoint, params=None):
        if params is None:
            params = {}
        timestamp = int(time.time() * 1000)
        params['timestamp'] = timestamp
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        params['signature'] = signature
        headers = {
            'X-MEXC-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
        url = f"{self.MEXC_API}{endpoint}"
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            elif method == 'POST':
                response = requests.post(url, headers=headers, params=params, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, params=params, timeout=10)
            else:
                raise ValueError("Unsupported method")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.log_message(f"Ошибка в signed_request для {endpoint}: {str(e)}")
            return {'code': -1, 'msg': str(e)}

    def init_clients(self, api_key, api_secret, api_passphrase):
        self.api_key = api_key
        self.api_secret = api_secret
        # api_passphrase not used in MEXC

    def test_clients(self):
        try:
            account = self._signed_request('GET', '/api/v3/account')
            if 'balances' not in account:
                raise Exception(f"Ошибка авторизации API: {account.get('msg', 'Unknown error')}")
            self.log_message("Клиенты API успешно инициализированы и авторизованы")
        except Exception as e:
            self.log_message(f"Ошибка при тестировании клиентов: {str(e)}")
            raise

    def test_ticker(self):
        try:
            response = requests.get(f"{self.MEXC_API}/api/v3/ticker/bookTicker?symbol=ETHUSDT", timeout=10)
            response.raise_for_status()
            ticker = response.json()
            if 'bidPrice' not in ticker or 'askPrice' not in ticker:
                raise Exception("Недопустимый ответ тикера")
            self.log_message(f"Тикер для ETHUSDT: {ticker}")
        except Exception as e:
            self.log_message(f"Ошибка при тестировании тикера: {str(e)}")
            raise

    def fetch_symbols(self):
        print("Запрашиваю данные с MEXC... (может занять пару секунд)")
        url = self.MEXC_API + "/api/v3/exchangeInfo"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            symbols = r.json()['symbols']
            result = []
            for s in symbols:
                if s['status'] != 'ENABLED':
                    continue
                base = s['baseAsset']
                quote = s['quoteAsset']
                instId = s['symbol']
                lot_filter = next((f for f in s['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                price_filter = next((f for f in s['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                if not lot_filter or not price_filter:
                    continue
                minSz = float(lot_filter['minQty'])
                lotSz = float(lot_filter['stepSize'])
                tickSz = float(price_filter['tickSize'])
                result.append({
                    'instId': instId,
                    'state': 'live',
                    'baseCcy': base,
                    'quoteCcy': quote,
                    'minSz': minSz,
                    'lotSz': lotSz,
                    'tickSz': tickSz
                })
            return result
        except Exception as e:
            self.log_message(f"Ошибка при получении символов: {str(e)}")
            return []

    def fetch_tickers(self):
        url = self.MEXC_API + "/api/v3/ticker/bookTicker"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            tickers = response.json()
            normalized = []
            for t in tickers:
                normalized.append({
                    'instId': t['symbol'],
                    'bidPx': t.get('bidPrice', '0'),
                    'askPx': t.get('askPrice', '0')
                })
            return normalized
        except Exception as e:
            self.log_message(f"Ошибка при получении тикеров: {str(e)}")
            return []

    def check_balance(self, asset):
        if not self.production:
            return float('inf')
        try:
            account = self._signed_request('GET', '/api/v3/account')
            if 'balances' not in account:
                raise Exception(f"Ошибка API: {account.get('msg', 'Unknown error')}")
            for b in account['balances']:
                if b['asset'] == asset:
                    available = float(b['free'])
                    self.log_message(f"Баланс {asset}: {available:.8f}")
                    return available
            self.log_message(f"Баланс {asset}: 0.0 (не найден)")
            return 0.0
        except Exception as e:
            self.log_message(f"Ошибка при проверке баланса для {asset}: {str(e)}")
            return 0.0

    def fetch_ticker_price(self, symbol):
        try:
            url = f"{self.MEXC_API}/api/v3/ticker/bookTicker?symbol={symbol}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            if 'bidPrice' not in data or 'askPrice' not in data:
                return None
            return {'sell': float(data['bidPrice']), 'buy': float(data['askPrice'])}
        except Exception as e:
            self.log_message(f"Ошибка при получении тикера для {symbol}: {str(e)}")
            return None

    def create_order_params(self, symbol, direction, ordType, adjusted_amount):
        side = 'SELL' if direction == 'sell' else 'BUY'
        params = {
            'symbol': symbol,
            'side': side,
            'type': 'MARKET',
        }
        if direction == 'sell':
            params['quantity'] = str(adjusted_amount)
        else:
            params['quoteOrderQty'] = str(adjusted_amount)
        return params

    def place_order(self, order_params):
        order = self._signed_request('POST', '/api/v3/order', order_params)
        if 'orderId' not in order:
            raise Exception(f"Ошибка размещения ордера: {order.get('msg', 'Unknown error')}")
        return order['orderId']

    def monitor_order(self, symbol, order_id, direction, amount, from_asset, to_asset, expected_price, fee_rate):
        start_time = time.time()
        while time.time() - start_time < 5:
            params = {'symbol': symbol, 'orderId': order_id}
            order_details = self._signed_request('GET', '/api/v3/order', params)
            if 'status' not in order_details:
                self.log_message(f"Ошибка: get_order для {order_id} вернул {order_details}")
                return None, False
            if order_details['status'] == 'FILLED':
                execution_time = time.time() - start_time
                filled_amount = float(order_details['executedQty'])
                cummulative_quote = float(order_details['cummulativeQuoteQty'])
                avg_price = cummulative_quote / filled_amount if filled_amount > 0 else 0.0
                # Fetch fees from myTrades
                trades_params = {'symbol': symbol, 'orderId': order_id}
                trades = self._signed_request('GET', '/api/v3/myTrades', trades_params)
                fee = 0.0
                fee_ccy = None
                for tr in trades:
                    fee += float(tr['commission'])
                    fee_ccy = tr['commissionAsset']
                if direction == 'sell':
                    actual_amount = cummulative_quote - fee if fee_ccy == to_asset else cummulative_quote
                else:
                    actual_amount = filled_amount - fee if fee_ccy == to_asset else filled_amount
                self.log_message(f"Транзакция {direction} {amount:.8f} {from_asset} -> {to_asset} завершена за {execution_time:.2f} сек.")
                self.log_message(f"Ожидаемая цена: {expected_price:.8f}, Фактическая средняя цена: {avg_price:.8f}")
                expected_new_amt = amount * expected_price * (1 - fee_rate) if direction == 'sell' else (amount / expected_price) * (1 - fee_rate)
                self.log_message(f"Ожидаемое количество: {expected_new_amt:.8f} {to_asset}, Фактическое количество: {actual_amount:.8f} {to_asset}")
                return actual_amount, True
            time.sleep(0.1)
        self.log_message(f"Транзакция {direction} {amount:.8f} {from_asset} -> {to_asset} не завершилась за 5 секунд, отменяется.")
        cancel_params = {'symbol': symbol, 'orderId': order_id}
        self._signed_request('DELETE', '/api/v3/order', cancel_params)
        return None, False

    def fetch_current_prices(self):
        try:
            url = f"{self.MEXC_API}/api/v3/ticker/bookTicker"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            tickers = response.json()
            return {t['symbol']: {'sell': float(t['bidPrice']), 'buy': float(t['askPrice'])} for t in tickers}
        except Exception as e:
            self.log_message(f"Ошибка при получении всех тикеров для проверки: {str(e)}")
            return None