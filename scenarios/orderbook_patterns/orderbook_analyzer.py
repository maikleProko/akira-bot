import json
import os
from datetime import datetime
from scenarios.orderbook_patterns.estimate_orderbook.estimate_orderbook import do_strategy
from scenarios.orderbook_patterns.selenium_orderbook_parse.instances.coinglass_parser import CoinglassParser
from utils.decorators import periodic
from utils.functions import MarketProcess


class OrderbookAnalyzer(MarketProcess):
    def __init__(self, parser=CoinglassParser()):
        self.parser = parser
        self.orderbook = {}

    def prepare(self):
        self.parser.go_coinglass()

    def run(self):
        self.orderbook = self.parser.parse_orderbook()
        do_strategy(self.parser, self.orderbook)

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        out_dir = os.path.join('files', 'orderbook')
        os.makedirs(out_dir, exist_ok=True)
        fname = f'orderbook_coinglass_{self.parser.symbol1}-{self.parser.symbol2}-{ts}.json'
        tmp_path = os.path.join(out_dir, fname + '.tmp')
        final_path = os.path.join(out_dir, fname)

        # сериализация (если в orderbook есть non-serializable объекты — поправьте parse_orderbook)
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.orderbook, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, final_path)
        print(f'[CoinglassParser] Saved orderbook to {final_path}')