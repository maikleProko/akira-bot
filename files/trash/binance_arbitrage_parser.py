from collections import defaultdict
import requests
from files.trash.arbitrage_parser import ArbitrageParser


class BinanceArbitrageParser(ArbitrageParser):
    BINANCE_API = "https://api.binance.com"

    def fetch_exchange_info(self):
        print("Запрашиваю данные с Binance... (может занять пару секунд)")
        url = self.BINANCE_API + "/api/v3/exchangeInfo"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    def fetch_book_tickers(self):
        url = self.BINANCE_API + "/api/v3/ticker/bookTicker"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()

    def build_graph_and_prices(self):
        """
        Возвращает:
          - out_edges: dict asset -> set(assets reachable за 1 сделку)
          - symbol_map: dict of (base,quote) -> symbol string
          - price_map: dict symbol -> {'bid': float, 'ask': float, 'base':..., 'quote':...}
        """
        info = self.fetch_exchange_info()
        tickers = self.fetch_book_tickers()

        ticker_map = {t['symbol']: t for t in tickers}

        price_map = {}
        symbol_map = {}
        out_edges = defaultdict(set)

        for s in info['symbols']:
            symbol = s['symbol']
            status = s.get('status', '')
            if status != 'TRADING':
                continue
            base = s['baseAsset']
            quote = s['quoteAsset']
            t = ticker_map.get(symbol)
            if not t:
                continue
            try:
                bid = float(t['bidPrice'])
                ask = float(t['askPrice'])
            except Exception:
                continue
            if bid <= 0 or ask <= 0:
                continue

            price_map[symbol] = {'bid': bid, 'ask': ask, 'base': base, 'quote': quote}
            symbol_map[(base, quote)] = symbol
            out_edges[base].add(quote)
            out_edges[quote].add(base)

        return out_edges, symbol_map, price_map