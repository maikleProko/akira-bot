import json
import os
from datetime import datetime
from scenarios.orderbook_patterns.abstracts.depth_parser import DepthParser
from utils.functions import MarketProcess
import json

class OrderbookParser(MarketProcess):
    def __init__(self, parser: DepthParser):
        self.parser = parser
        self.orderbook = {}

    def prepare(self):
        print(f"[OrderbookParser({self.parser.parse_type}_{self.parser.platform_name}_parser)] Подготовка {self.parser.parse_type}_{self.parser.platform_name}_parser...")
        self.parser.go_page()  # Теперь работает синхронно

    def run(self, start_time=None, current_time=None, end_time=None):
        if current_time is None:
            try:
                self.orderbook = self.parser.parse_orderbook()

                ts = datetime.now().strftime('%Y:%m:%d_%H:%M')
                out_dir = os.path.join('files', 'orderbook', f'{self.parser.parse_type}', f'{self.parser.platform_name}')
                os.makedirs(out_dir, exist_ok=True)
                fname = f'{self.parser.parse_type}_{self.parser.platform_name}_orderbook_{self.parser.symbol1}-{self.parser.symbol2}-{ts}.json'
                tmp_path = os.path.join(out_dir, fname + '.tmp')
                final_path = os.path.join(out_dir, fname)

                # сериализация (если в orderbook есть non-serializable объекты — поправьте parse_orderbook)
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.orderbook, f, ensure_ascii=False, indent=2)
                os.replace(tmp_path, final_path)
                print(f'[OrderbookParser({self.parser.parse_type}_{self.parser.platform_name}_parser)] Saved orderbook to {final_path}')
            except Exception as e:
                print(f"[OrderbookParser({self.parser.parse_type}_{self.parser.platform_name}_parser)] Ошибка парсинга: {e}")
                self.orderbook = {'bids': [], 'asks': []}

        else:
            current_time_string = current_time.strftime('%Y:%m:%d_%H:%M')
            in_dir = os.path.join('files', 'orderbook', f'{self.parser.parse_type}', f'{self.parser.platform_name}')
            fname = f'{self.parser.parse_type}_{self.parser.platform_name}_orderbook_{self.parser.symbol1}-{self.parser.symbol2}-{current_time_string}.json'
            final_path = os.path.join(in_dir, fname)

            with open(final_path, 'r') as file:
                self.orderbook = json.load(file)

