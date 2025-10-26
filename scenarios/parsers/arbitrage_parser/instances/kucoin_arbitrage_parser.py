from collections import defaultdict
import requests
from scenarios.parsers.arbitrage_parser.abstracts.arbitrage_parser import ArbitrageParser


class KuCoinArbitrageParser(ArbitrageParser):
    KUCOIN_API = "https://api.kucoin.com"

    def fetch_symbols(self):
        print("Запрашиваю данные с KuCoin... (может занять пару секунд)")
        url = self.KUCOIN_API + "/api/v2/symbols"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['data']

    def fetch_all_tickers(self):
        url = self.KUCOIN_API + "/api/v1/market/allTickers"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()['data']['ticker']

    def build_graph_and_prices(self):
        """
        Возвращает:
          - out_edges: dict asset -> set(assets reachable за 1 сделку)
          - symbol_map: dict of (base,quote) -> symbol string
          - price_map: dict symbol -> {'bid': float, 'ask': float, 'base':..., 'quote':...}
        """
        info = self.fetch_symbols()
        tickers = self.fetch_all_tickers()

        ticker_map = {t['symbol']: t for t in tickers}

        price_map = {}
        symbol_map = {}
        out_edges = defaultdict(set)

        for s in info:
            symbol = s['symbol']
            if not s.get('enableTrading', False):
                continue
            base = s['baseCurrency']
            quote = s['quoteCurrency']
            t = ticker_map.get(symbol)
            if not t:
                continue
            try:
                bid = float(t['sell'])
                ask = float(t['buy'])
            except Exception:
                continue
            if bid <= 0 or ask <= 0:
                continue

            price_map[symbol] = {'bid': bid, 'ask': ask, 'base': base, 'quote': quote}
            symbol_map[(base, quote)] = symbol
            out_edges[base].add(quote)
            out_edges[quote].add(base)

        return out_edges, symbol_map, price_map