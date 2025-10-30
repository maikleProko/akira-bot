import requests
from collections import defaultdict, deque

from files.trash.arbitrage_parser import ArbitrageParser
from datetime import datetime
import time
import threading

class BinanceSmartArbitrageBuyer(ArbitrageParser):
    BINANCE_API = "https://api.binance.com"

    def __init__(self, initial_capital=2000):
        super().__init__()
        self.initial_capital = initial_capital
        self.path_counts = defaultdict(lambda: deque())
        self.active_monitors = {}
        self.fee_rate = 0.001
        self.min_profit = 0.00001
        self.start_amount = 1.0
        self.max_cycle_len = 4
        self.global_balances = {}  # To store cumulative balances from monitors for future use

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
                # Clean old entries (>2 minutes)
                while self.path_counts[norm_path] and self.path_counts[norm_path][0] < current_timestamp - 120:
                    self.path_counts[norm_path].popleft()
                # Check if >=4 in last 2 min and no active monitor
                if len(self.path_counts[norm_path]) >= 4 and norm_path not in self.active_monitors:
                    thread = threading.Thread(target=self.monitor_path, args=(norm_path,))
                    thread.start()
                    self.active_monitors[norm_path] = thread

    def get_total_value_usdt(self, balances, symbol_map, price_map, fee=0.0):
        total = 0.0
        for asset, amt in balances.items():
            if asset == 'USDT':
                total += amt
            else:
                val, _, _, _ = self.convert_amount(amt, asset, 'USDT', symbol_map, price_map, fee)
                if val is not None:
                    total += val
        return total

    def monitor_path(self, path_tuple):
        # Fetch initial prices
        _, symbol_map, price_map = self.build_graph_and_prices()

        cycle_assets = list(path_tuple)
        N = len(cycle_assets)
        portion_usdt = self.initial_capital / N
        balances = {}

        for asset in cycle_assets:
            if asset == 'USDT':
                balances['USDT'] = portion_usdt
            else:
                new_amt, sym, direction, price = self.convert_amount(portion_usdt, 'USDT', asset, symbol_map, price_map, self.fee_rate)
                if new_amt is None:
                    print(f"Cannot initialize position for {asset}")
                    return
                balances[asset] = new_amt

        original_initial_value = self.get_total_value_usdt(balances, symbol_map, price_map, 0.0)
        initial_value = original_initial_value
        last_arbitrage_time = time.time()

        # Log initialization
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        path_str = ' -> '.join(cycle_assets + [cycle_assets[0]])
        with open('files/decisions/decisions_SmartArbitrageBuyer.txt', 'a', encoding='utf-8') as f:
            f.write(f"Monitor Started: {current_time_str}\n")
            f.write(f"Path: {path_str}\n")
            f.write(f"Initial Capital: {self.initial_capital} USDT\n")
            f.write(f"Initial Balances: {balances}\n")
            f.write(f"Initial Total Value: {initial_value:.8f} USDT\n")
            f.write("----\n")

        while True:
            # Fetch current prices
            _, symbol_map, price_map = self.build_graph_and_prices()

            # Simulate cycle for unit to check if arbitrage
            start_asset = cycle_assets[0]
            amt = self.start_amount
            valid = True
            trades_sim = []  # To store simulated trades for checking
            for i in range(N):
                frm = cycle_assets[i]
                to = cycle_assets[(i + 1) % N]
                new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, self.fee_rate)
                if new_amt is None:
                    valid = False
                    break
                trades_sim.append((frm, to, sym, direction, new_amt, price, amt))
                amt = new_amt

            if valid:
                profit_perc = (amt - self.start_amount) / self.start_amount
                # Additional check for break-even: ensure profit_perc > 0 (beyond min_profit)
                if profit_perc > self.min_profit and profit_perc > 0:
                    # Arbitrage opportunity: perform roll
                    new_balances = {}
                    trades = []
                    for i in range(N):
                        frm = cycle_assets[i]
                        to = cycle_assets[(i + 1) % N]
                        amt = balances.get(frm, 0.0)
                        if amt > 0:
                            new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, self.fee_rate)
                            if new_amt is not None:
                                new_balances[to] = new_balances.get(to, 0.0) + new_amt
                                trades.append((frm, to, sym, direction, new_amt, price, amt))
                            else:
                                valid = False
                                break

                    if valid:
                        balances = new_balances
                        last_arbitrage_time = time.time()
                        current_value = self.get_total_value_usdt(balances, symbol_map, price_map, 0.0)
                        profit = current_value - initial_value  # or previous, but cumulative
                        initial_value = current_value  # update for next

                        # Log
                        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        with open('files/decisions/decisions_SmartArbitrageBuyer.txt', 'a', encoding='utf-8') as f:
                            f.write(f"Arbitrage Executed: {current_time_str}\n")
                            f.write(f"Path: {path_str}\n")
                            f.write(f"Profit Perc: {profit_perc*100:.4f}%\n")
                            f.write("Trades:\n")
                            for t in trades:
                                frm, to, sym, dirc, new_amt, price, orig_amt = t
                                if dirc == 'sell':
                                    f.write(f"Selling {orig_amt:.8f} {frm} for {to} using pair {sym} at bid price {price:.8f} ({to} per {frm})\n")
                                    f.write(f"Amount received: {orig_amt:.8f} * {price:.8f} * (1 - {self.fee_rate}) = {new_amt:.8f} {to}\n")
                                else:
                                    f.write(f"Buying {new_amt:.8f} {to} with {orig_amt:.8f} {frm} using pair {sym} at ask price {price:.8f} ({frm} per {to})\n")
                                    f.write(f"Amount bought: ({orig_amt:.8f} / {price:.8f}) * (1 - {self.fee_rate}) = {new_amt:.8f} {to}\n")
                            f.write(f"New Balances: {balances}\n")
                            f.write(f"Current Total Value: {current_value:.8f} USDT\n")
                            f.write(f"Profit this cycle: {profit:.8f} USDT\n")
                            f.write("----\n")

            # Check if 1 minute without arbitrage
            if time.time() - last_arbitrage_time > 60:
                current_value_with_fees = self.get_total_value_usdt(balances, symbol_map, price_map, self.fee_rate)
                if current_value_with_fees >= original_initial_value:
                    break
                # else continue monitoring

            time.sleep(0.5)

        # Final log
        with open('files/decisions/decisions_SmartArbitrageBuyer.txt', 'a', encoding='utf-8') as f:
            f.write(f"Monitor Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write(f"Path: {path_str}\n")
            f.write(f"Final Balances: {balances}\n")
            f.write(f"Final Total Value: {self.get_total_value_usdt(balances, symbol_map, price_map, 0.0):.8f} USDT\n")
            f.write("----\n")

        # Accumulate results to global_balances for future use
        for asset, amt in balances.items():
            self.global_balances[asset] = self.global_balances.get(asset, 0.0) + amt

        # Remove from active monitors
        del self.active_monitors[path_tuple]