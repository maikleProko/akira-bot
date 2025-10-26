from collections import defaultdict
import requests
from scenarios.parsers.arbitrage_parser.abstracts.arbitrage_parser import ArbitrageParser


class BybitArbitrageParser(ArbitrageParser):
    BYBIT_API = "https://api.bybit.com"

    def fetch_instruments_info(self):
        print("Запрашиваю данные с Bybit... (может занять пару секунд)")
        url = self.BYBIT_API + "/v5/market/instruments-info?category=spot"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['result']['list']

    def fetch_tickers(self):
        url = self.BYBIT_API + "/v5/market/tickers?category=spot"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['result']['list']

    def build_graph_and_prices(self):
        """
        Возвращает:
          - out_edges: dict asset -> set(assets reachable за 1 сделку)
          - symbol_map: dict of (base,quote) -> symbol string
          - price_map: dict symbol -> {'bid': float, 'ask': float, 'base':..., 'quote':...}
        """
        info = self.fetch_instruments_info()
        tickers = self.fetch_tickers()

        ticker_map = {t['symbol']: t for t in tickers}

        price_map = {}
        symbol_map = {}
        out_edges = defaultdict(set)

        for s in info:
            symbol = s['symbol']
            status = s['status']
            if status != 'Trading':
                continue
            base = s['baseCoin']
            quote = s['quoteCoin']
            t = ticker_map.get(symbol)
            if not t:
                continue
            try:
                bid = float(t['bid1Price'])
                ask = float(t['ask1Price'])
            except Exception:
                continue
            if bid <= 0 or ask <= 0:
                continue

            price_map[symbol] = {'bid': bid, 'ask': ask, 'base': base, 'quote': quote}
            symbol_map[(base, quote)] = symbol
            out_edges[base].add(quote)
            out_edges[quote].add(base)

        return out_edges, symbol_map, price_map