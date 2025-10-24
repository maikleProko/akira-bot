import json
import requests
from collections import defaultdict, deque

from scenarios.parsers.arbitrage_parser.abstracts.arbitrage_parser import ArbitrageParser
from utils.core.functions import MarketProcess
from datetime import datetime
import time
import threading

class BinanceSmartArbitrageParser(ArbitrageParser):
    BINANCE_API = "https://api.binance.com"

    def __init__(self):
        super().__init__()
        self.path_counts = defaultdict(lambda: deque())
        self.active_monitors = {}
        self.fee_rate = 0.001
        self.min_profit = 0.00001
        self.start_amount = 1.0
        self.max_cycle_len = 4

    def fetch_exchange_info(self):
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

    def run_realtime(self):
        print("Запрашиваю данные с Binance... (может занять пару секунд)")
        ops = self.find_arbitrage_cycles(fee_rate=self.fee_rate, min_profit=self.min_profit, start_amount=self.start_amount,
                                    max_cycles=200000, max_cycle_len=self.max_cycle_len)

        if not ops:
            print("Арбитражных возможностей не найдено (по заданному порогу).")
        else:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            with open('files/decisions/decisions_ArbitrageParser.txt', 'a', encoding='utf-8') as f:
                f.write(f"Time: {current_time}\n")
                for o in ops:
                    f.write(f"Path: {' -> '.join(o['path'])}\n")
                    f.write(f"Profit: {o['profit_perc']*100:.4f}%\n")
                    f.write(f"Start Amount: {o['start_amount']}{o['start_asset']}\n")
                    f.write(f"End Amount: {o['end_amount']:.8f}{o['start_asset']}\n")
                    f.write("Calculation steps:\n")
                    amt = o['start_amount']
                    for t in o['trades']:
                        frm, to, sym, dirc, new_amt, price = t
                        if dirc == 'sell':
                            f.write(f"Selling {amt:.8f} {frm} for {to} using pair {sym} at bid price {price:.8f} ({to} per {frm})\n")
                            f.write(f"Amount received: {amt:.8f} * {price:.8f} * (1 - {self.fee_rate}) = {new_amt:.8f} {to}\n")
                        else:
                            f.write(f"Buying {new_amt:.8f} {to} with {amt:.8f} {frm} using pair {sym} at ask price {price:.8f} ({frm} per {to})\n")
                            f.write(f"Amount bought: ({amt:.8f} / {price:.8f}) * (1 - {self.fee_rate}) = {new_amt:.8f} {to}\n")
                        amt = new_amt
                    f.write("----\n")

            for o in ops[:50]:
                print("Путь:", " -> ".join(o['path']), f"Начало {o['start_amount']}{o['start_asset']} -> Конец {o['end_amount']:.8f}{o['start_asset']}", f"Прибыль {o['profit_perc']*100:.4f}%")
                print("Calculation steps:")
                amt = o['start_amount']
                for t in o['trades']:
                    frm, to, sym, dirc, new_amt, price = t
                    if dirc == 'sell':
                        print(f"  Selling {amt:.8f} {frm} for {to} using pair {sym} at bid price {price:.8f} ({to} per {frm})")
                        print(f"  Amount received: {amt:.8f} * {price:.8f} * (1 - {self.fee_rate}) = {new_amt:.8f} {to}")
                    else:
                        print(f"  Buying {new_amt:.8f} {to} with {amt:.8f} {frm} using pair {sym} at ask price {price:.8f} ({frm} per {to})")
                        print(f"  Amount bought: ({amt:.8f} / {price:.8f}) * (1 - {self.fee_rate}) = {new_amt:.8f} {to}")
                    amt = new_amt
                print("----")

            # Track paths
            current_timestamp = time.time()
            for o in ops:
                cycle_nodes = o['path'][:-1]  # remove closing element
                norm_path = self.normalize_cycle(cycle_nodes)
                self.path_counts[norm_path].append(current_timestamp)
                # Clean old entries (>5 minutes)
                while self.path_counts[norm_path] and self.path_counts[norm_path][0] < current_timestamp - 300:
                    self.path_counts[norm_path].popleft()
                # Check if >=10 in last 5 min and no active monitor
                if len(self.path_counts[norm_path]) >= 10 and norm_path not in self.active_monitors:
                    thread = threading.Thread(target=self.monitor_path, args=(norm_path,))
                    thread.start()
                    self.active_monitors[norm_path] = thread

    def monitor_path(self, path_tuple):
        last_arbitrage_time = time.time()
        while True:
            # Fetch current prices
            _, symbol_map, price_map = self.build_graph_and_prices()
            # Simulate the path
            cycle_nodes = list(path_tuple)
            trades = []
            amt = self.start_amount
            valid = True
            for i in range(len(cycle_nodes)):
                frm = cycle_nodes[i]
                to = cycle_nodes[(i + 1) % len(cycle_nodes)]
                new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, self.fee_rate)
                if new_amt is None:
                    valid = False
                    break
                trades.append((frm, to, sym, direction, new_amt, price))
                amt = new_amt
            if valid:
                profit_perc = (amt - self.start_amount) / self.start_amount
                if profit_perc > self.min_profit:
                    # Arbitrage found, update time
                    last_arbitrage_time = time.time()
                    # Log to file
                    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    path_str = ' -> '.join(cycle_nodes + [cycle_nodes[0]])
                    with open('files/decisions/decisions_ArbitrageParser.txt', 'a', encoding='utf-8') as f:
                        f.write(f"Monitor Time: {current_time_str}\n")
                        f.write(f"Path: {path_str}\n")
                        f.write(f"Profit: {profit_perc*100:.4f}%\n")
                        f.write(f"Start Amount: {self.start_amount}{cycle_nodes[0]}\n")
                        f.write(f"End Amount: {amt:.8f}{cycle_nodes[0]}\n")
                        f.write("Calculation steps:\n")
                        sim_amt = self.start_amount
                        for t in trades:
                            frm, to, sym, dirc, new_amt, price = t
                            if dirc == 'sell':
                                f.write(f"Selling {sim_amt:.8f} {frm} for {to} using pair {sym} at bid price {price:.8f} ({to} per {frm})\n")
                                f.write(f"Amount received: {sim_amt:.8f} * {price:.8f} * (1 - {self.fee_rate}) = {new_amt:.8f} {to}\n")
                            else:
                                f.write(f"Buying {new_amt:.8f} {to} with {sim_amt:.8f} {frm} using pair {sym} at ask price {price:.8f} ({frm} per {to})\n")
                                f.write(f"Amount bought: ({sim_amt:.8f} / {price:.8f}) * (1 - {self.fee_rate}) = {new_amt:.8f} {to}\n")
                            sim_amt = new_amt
                        f.write("----\n")

            # Check if 10 minutes without arbitrage
            if time.time() - last_arbitrage_time > 600:
                break
            time.sleep(0.5)
        # Remove from active monitors
        del self.active_monitors[path_tuple]