from collections import defaultdict

from utils.core.functions import MarketProcess


class AbstractArbitrageParser(MarketProcess):

    def init(self,api_key, api_secret):
        raise NotImplementedError

    def run_realtime(self):
        raise NotImplementedError

    def get_objects(self, response):
        raise NotImplementedError

    def get_symbols(self):
        raise NotImplementedError

    def get_tickers(self):
        raise NotImplementedError

    def get_balance(self, asset):
        raise NotImplementedError

    def place_order(self, **order_params):
        raise NotImplementedError


    ### GRAPH

    def get_quote_min_size(self, quote):
        if quote == 'USDT' or quote == 'USDC':
            return 5.0
        elif quote == 'BTC':
            return 0.0001
        elif quote == 'ETH':
            return 0.001
        else:
            return 0.01

    def process_symbol_checks(self, s):
        symbol, base, quote, base_min_size, base_increment, quote_increment = self.extract_symbol_data(s)
        if not self.check_state_live(s):
            return False
        if not self.check_ignore_assets(base, quote):
            return False
        if not self.check_available_coins(base, quote):
            return False
        if not self.check_increments_valid(base_increment, base_min_size):
            return False
        return True

    def extract_symbol_data(self, s):
        return s['symbol'], s['baseCoin'], s['quoteCoin'], float(s['lotSizeFilter']['minOrderQty']), float(s['lotSizeFilter']['basePrecision']), float(s['priceFilter']['tickSize'])

    def process_symbols(self, info, symbol_map, out_edges, symbols):
        for s in info:
            if not self.process_symbol_checks(s):
                return
            symbol, base, quote, base_min_size, base_increment, quote_increment = self.extract_symbol_data(s)
            quote_min_size = self.get_quote_min_size(quote)
            symbol_map[(base, quote)] = symbol
            out_edges[base].add(quote)
            out_edges[quote].add(base)
            symbols.append((symbol, base, quote, base_min_size, quote_min_size, base_increment, quote_increment))


    def build_graph_and_prices(self):
        info = self.get_symbols()
        symbol_map, out_edges, symbols = {}, defaultdict(set), []
        self.process_all_symbols(info, symbol_map, out_edges, symbols)
        price_map = {}
        try:
            ticker_map = self.fetch_and_map_tickers()
            self.build_price_map(symbols, ticker_map, price_map)
        except Exception as e:
            self.logger.log_message(f"Ошибка при получении всех тикеров: {str(e)}")
        return out_edges, symbol_map, price_map
        

