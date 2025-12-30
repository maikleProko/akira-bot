from collections import defaultdict


class GraphBuilder:
    def __init__(self, exchange_client, ignore, available_coins, logger):
        self.exchange_client = exchange_client
        self.ignore = ignore
        self.available_coins = available_coins
        self.logger = logger

    def drain_usdts(self, symbols):
        result = []
        for tup in symbols:
            symbol, base, quote, *rest = tup
            if quote != 'USDT' or base == 'BTC':
                result.append(tup)
        return result

    def get_usdt_usdc_min(self):
        return 5.0

    def get_btc_min(self):
        return 0.0001

    def get_eth_min(self):
        return 0.001

    def get_default_min(self):
        return 0.01

    def get_quote_min_size(self, quote):
        if quote == 'USDT' or quote == 'USDC':
            return self.get_usdt_usdc_min()
        elif quote == 'BTC':
            return self.get_btc_min()
        elif quote == 'ETH':
            return self.get_eth_min()
        else:
            return self.get_default_min()

    def check_state_live(self, s):
        return s['state'] == 'live'

    def check_ignore_assets(self, base, quote):
        return base not in self.ignore and quote not in self.ignore

    def check_available_coins(self, base, quote):
        if self.available_coins is None:
            return True
        return base in self.available_coins and quote in self.available_coins

    def check_increments_valid(self, base_increment, base_min_size):
        return base_increment < 1.0 and base_min_size < 1.0

    def extract_symbol_data(self, s):
        symbol = s['instId']
        base = s['baseCcy']
        quote = s['quoteCcy']
        base_min_size = float(s['minSz'])
        base_increment = float(s['lotSz'])
        quote_increment = float(s['tickSz'])
        return symbol, base, quote, base_min_size, base_increment, quote_increment

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

    def add_to_maps(self, s, symbol_map, out_edges, symbols):
        symbol, base, quote, base_min_size, base_increment, quote_increment = self.extract_symbol_data(s)
        quote_min_size = self.get_quote_min_size(quote)
        symbol_map[(base, quote)] = symbol
        out_edges[base].add(quote)
        out_edges[quote].add(base)
        symbols.append((symbol, base, quote, base_min_size, quote_min_size, base_increment, quote_increment))

    def process_symbol(self, s, symbol_map, out_edges, symbols):
        if not self.process_symbol_checks(s):
            return
        self.add_to_maps(s, symbol_map, out_edges, symbols)

    def get_ticker_entry(self, ticker):
        bid = float(ticker['bidPx'])
        ask = float(ticker['askPx'])
        return bid, ask

    def check_prices_positive(self, bid, ask):
        return bid > 0 and ask > 0

    def create_price_entry(self, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, bid, ask):
        return {
            'bid': bid,
            'ask': ask,
            'base': base,
            'quote': quote,
            'base_min_size': base_min_size,
            'quote_min_size': quote_min_size,
            'base_increment': base_increment,
            'quote_increment': quote_increment
        }

    def process_ticker_try(self, sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, ticker_map):
        ticker = ticker_map.get(sym)
        if not ticker:
            return None
        bid, ask = self.get_ticker_entry(ticker)
        if not self.check_prices_positive(bid, ask):
            return None
        return self.create_price_entry(base, quote, base_min_size, quote_min_size, base_increment, quote_increment, bid, ask)

    def process_ticker(self, sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, ticker_map):
        try:
            return self.process_ticker_try(sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, ticker_map)
        except Exception as e:
            #self.logger.log_message(f"Ошибка при обработке цены для {sym}: {str(e)}")
            return None

    def init_graph_structures(self):
        symbol_map = {}
        out_edges = defaultdict(set)
        symbols = []
        return symbol_map, out_edges, symbols

    def process_all_symbols(self, info, symbol_map, out_edges, symbols):
        for s in info:
            self.process_symbol(s, symbol_map, out_edges, symbols)

    def fetch_and_map_tickers(self):
        tickers = self.exchange_client.fetch_tickers()
        ticker_map = {t['instId']: t for t in tickers}
        return ticker_map

    def build_price_map(self, symbols, ticker_map, price_map):
        symbols = self.drain_usdts(symbols)
        for sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment in symbols:
            entry = self.process_ticker(sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, ticker_map)
            if entry:
                price_map[sym] = entry

    def build_graph_and_prices(self):
        info = self.exchange_client.fetch_symbols()
        symbol_map, out_edges, symbols = self.init_graph_structures()
        self.process_all_symbols(info, symbol_map, out_edges, symbols)
        price_map = {}
        try:
            ticker_map = self.fetch_and_map_tickers()
            self.build_price_map(symbols, ticker_map, price_map)
        except Exception as e:
            self.logger.log_message(f"Ошибка при получении всех тикеров: {str(e)}")
        return out_edges, symbol_map, price_map