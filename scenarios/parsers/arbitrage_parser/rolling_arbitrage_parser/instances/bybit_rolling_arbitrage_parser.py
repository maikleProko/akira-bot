# bybit_rolling_arbitrage_parser.py
import requests
from pybit.unified_trading import HTTP
from scenarios.parsers.arbitrage_parser.rolling_arbitrage_parser.core.rolling_arbitrage_parser import \
    RollingArbitrageParser
from utils.core.functions import log
class BybitRollingArbitrageParser(RollingArbitrageParser):
    def init(self, api_key, api_secret, api_passphrase):
        self.session = HTTP(api_key=api_key, api_secret=api_secret)
        self.api_url = "https://api.bybit.com"
        self.symbol_price_url = f"{self.api_url}/v5/market/tickers?category=spot&symbol="
    def get_objects(self, response):
        if response['retCode'] != 0:
            raise Exception(f"Ошибка: {response['retMsg']}")
        return response['result']['list']
    def get_symbols(self):
        log('Запрашиваем пары с Bybit...')
        return self.get_objects(self.session.get_instruments_info(category="spot"))
    def get_tickers(self):
        log('Запрашиваем тикеры с Bybit...')
        return self.get_objects(self.session.get_tickers(category="spot"))
    def get_balance(self, asset):
        balance = self.session.get_wallet_balance(accountType="UNIFIED", coin=asset)
        if balance['retCode'] != 0:
            raise Exception(f"Ошибка API: {balance['retMsg']}")
        coins = balance['result']['list'][0]['coin']
        for c in coins:
            if c['coin'] == asset:
                return float(c['availableBalance'])
        return 0.0

    def get_fees(self):
        log('Запрашиваем комиссии с Bybit...')
        return self.get_objects(self.session.get_fee_rates(category="spot"))
    def get_symbol_price(self, data, symbol):
        if data['retCode'] != 0 or not data['result']['list']:
            return None
        ticker_data = data['result']['list'][0]
        return {'name': symbol, 'sell': float(ticker_data['bid1Price']), 'buy': float(ticker_data['ask1Price'])}
    def sync_get_symbol_price(self, symbol):
        response = requests.get(f"{self.symbol_price_url}{symbol}", timeout=5)
        response.raise_for_status()
        return self.get_symbol_price(response.json(), symbol)
    async def async_get_symbol_price(self, session, symbol):
        async with session.get(f"{self.symbol_price_url}{symbol}") as response:
            return self.get_symbol_price(await response.json(), symbol)
    def place_order(self, **order_params):
        order = self.session.place_order(**{
            'category': 'spot',
            **order_params
        })
        if order['retCode'] != 0:
            raise Exception(
                f"Ошибка размещения ордера: code={order['retCode']}, msg={order['retMsg']}, data={order.get('result', 'N/A')}")
        return order['result']['orderId']
    def get_order_history(self, **params):
        return self.session.get_order_history(**params)