from collections import defaultdict
import requests
from scenarios.parsers.arbitrage_parser.abstracts.arbitrage_parser import ArbitrageParser
from datetime import datetime, timedelta
import json
from functools import wraps
import math
import time
from kucoin.client import Market, Trade, User

class CarefulKuCoinArbitrageParser(ArbitrageParser):
    KUCOIN_API = "https://api.kucoin.com"

    def __init__(self, deposit=1000.0, production=False, api_key=None, api_secret=None, api_passphrase=None):
        self.deposit = deposit
        self.production = production
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.consecutive_same = 0
        self.possible = True
        self.prev_paths = None
        self.market_client = None
        self.trade_client = None
        self.user_client = None
        if self.production:
            if not api_key or not api_secret:
                self.log_message("Error: API key and secret are required for production mode")
                self.production = False
            else:
                try:
                    self.market_client = Market(key=api_key, secret=api_secret, passphrase=api_passphrase)
                    self.trade_client = Trade(key=api_key, secret=api_secret, passphrase=api_passphrase)
                    self.user_client = User(key=api_key, secret=api_secret, passphrase=api_passphrase)
                    self.log_message("Successfully initialized KuCoin API clients")
                except Exception as e:
                    self.log_message(f"Failed to initialize KuCoin API clients: {str(e)}")
                    self.production = False

    def log_message(self, message):
        """Логирует сообщение в консоль и в final_decisions.txt"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"{timestamp}: {message}")
        with open('files/decisions/final_decisions.txt', 'a', encoding='utf-8') as f:
            f.write(f"{timestamp}: {message}\n")

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
          - price_map: dict symbol -> {'bid': float, 'ask': float, 'base':..., 'quote':..., 'base_min_size':..., 'quote_min_size':..., 'base_increment':..., 'quote_increment':...}
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

            base_min_size = float(s['baseMinSize'])
            quote_min_size = float(s['quoteMinSize'])
            base_increment = float(s['baseIncrement'])
            quote_increment = float(s['quoteIncrement'])

            price_map[symbol] = {
                'bid': bid,
                'ask': ask,
                'base': base,
                'quote': quote,
                'base_min_size': base_min_size,
                'quote_min_size': quote_min_size,
                'base_increment': base_increment,
                'quote_increment': quote_increment
            }
            symbol_map[(base, quote)] = symbol
            out_edges[base].add(quote)
            out_edges[quote].add(base)

        return out_edges, symbol_map, price_map

    def check_balance(self, asset):
        """Проверяет баланс указанного актива"""
        try:
            accounts = self.user_client.get_account_list()
            for account in accounts:
                if account['currency'] == asset and account['type'] == 'trade':
                    available = float(account['available'])
                    return available
            return 0.0
        except Exception as e:
            self.log_message(f"Ошибка при проверке баланса для {asset}: {str(e)}")
            return 0.0

    def execute_trade(self, from_asset, to_asset, amount, symbol, direction, expected_price, fee_rate):
        """Выполняет реальную сделку и возвращает фактический результат"""
        start_time = time.time()
        try:
            # Place market order
            side = 'sell' if direction == 'sell' else 'buy'
            order = self.trade_client.create_market_order(symbol=symbol, side=side, size=amount if direction == 'sell' else None, funds=amount if direction == 'buy' else None)
            order_id = order['orderId']

            # Monitor order status
            while time.time() - start_time < 5:
                order_details = self.trade_client.get_order_details(order_id)
                if order_details['isActive'] is False:
                    execution_time = time.time() - start_time
                    # Fetch actual executed amount
                    filled_amount = float(order_details['dealSize'])
                    dealt_funds = float(order_details['dealFunds'])
                    actual_price = dealt_funds / filled_amount if direction == 'sell' else filled_amount / dealt_funds
                    actual_amount = dealt_funds * (1 - fee_rate) if direction == 'sell' else filled_amount * (1 - fee_rate)
                    self.log_message(f"Транзакция {direction} {amount:.8f} {from_asset} -> {to_asset} завершена за {execution_time:.2f} сек.")
                    self.log_message(f"Ожидаемая цена: {expected_price:.8f}, Фактическая средняя цена: {actual_price:.8f}")
                    self.log_message(f"Ожидаемое количество: {(amount * expected_price * (1 - fee_rate) if direction == 'sell' else (amount / expected_price) * (1 - fee_rate)):.8f} {to_asset}, "
                                     f"Фактическое количество: {actual_amount:.8f} {to_asset}")
                    return actual_amount, True
                time.sleep(0.1)

            self.log_message(f"Транзакция {direction} {amount:.8f} {from_asset} -> {to_asset} не завершилась за 5 секунд, отменяется.")
            return None, False
        except Exception as e:
            self.log_message(f"Ошибка при выполнении транзакции {direction} {amount:.8f} {from_asset} -> {to_asset}: {str(e)}")
            return None, False

    def run_realtime(self):
        fee = 0.001  # 0.1% комиссия
        min_profit = 0.000001  # 0.1% порог
        start = 1.0
        max_len = 4  # ищем циклы длины 3 и 4

        ops = self.find_arbitrage_cycles(fee_rate=fee, min_profit=min_profit, start_amount=start,
                                         max_cycles=200000, max_cycle_len=max_len)

        # Находим наиболее прибыльный путь с start_asset == 'USDT'
        usdt_ops = [o for o in ops if o['start_asset'] == 'USDT']
        current_best_path = None
        if usdt_ops:
            best_op = max(usdt_ops, key=lambda x: x['profit_perc'])
            current_best_path = tuple(best_op['path'])

        # Проверяем, совпадает ли лучший USDT-путь с предыдущим
        if current_best_path and current_best_path == self.prev_paths:
            self.consecutive_same += 1
        else:
            self.consecutive_same = 1
        self.prev_paths = current_best_path

        # Стандартное поведение (логирование в консоль)
        if not ops:
            self.log_message("Арбитражных возможностей не найдено (по заданному порогу).")
        else:
            for o in ops[:50]:
                self.log_message(f"Путь: {' -> '.join(o['path'])}")
                self.log_message(f"Начало {o['start_amount']}{o['start_asset']} -> Конец {o['end_amount']:.8f}{o['start_asset']}")
                self.log_message(f"Прибыль {o['profit_perc'] * 100:.4f}%")
                self.log_message("Calculation steps:")
                amt = o['start_amount']
                for t in o['trades']:
                    frm, to, sym, dirc, new_amt, price = t
                    if dirc == 'sell':
                        self.log_message(f"  Selling {amt:.8f} {frm} for {to} using pair {sym} at bid price {price:.8f} ({to} per {frm})")
                        self.log_message(f"  Amount received: {amt:.8f} * {price:.8f} * (1 - {fee}) = {new_amt:.8f} {to}")
                    else:
                        self.log_message(f"  Buying {new_amt:.8f} {to} with {amt:.8f} {frm} using pair {sym} at ask price {price:.8f} ({frm} per {to})")
                        self.log_message(f"  Amount bought: ({amt:.8f} / {price:.8f}) * (1 - {fee}) = {new_amt:.8f} {to}")
                    amt = new_amt
                self.log_message("----")

        # Если 5 раз подряд один и тот же лучший USDT-путь, входим в режим активного трейдера
        self.log_message(f"CHECKING: consecutive_same = {self.consecutive_same}")
        if self.consecutive_same >= 5 and usdt_ops and self.possible:
            self.possible = False
            self.log_message("Входим в режим активного трейдера")
            # Выбираем лучшую возможность
            best_op = max(usdt_ops, key=lambda x: x['profit_perc'])
            initial_deposit = self.deposit
            amt = self.deposit
            trades = []
            valid = True
            out_edges, symbol_map, price_map = self.build_graph_and_prices()  # Перезагружаем свежие цены

            if self.production:
                # Проверяем баланс USDT
                available_usdt = self.check_balance('USDT')
                if available_usdt < self.deposit:
                    self.log_message(f"Недостаточно USDT на балансе: доступно {available_usdt:.8f}, требуется {self.deposit:.8f}")
                    valid = False

            if valid:
                for i in range(len(best_op['path']) - 1):
                    frm = best_op['path'][i]
                    to = best_op['path'][i + 1]
                    new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, fee)
                    if new_amt is None:
                        self.log_message(f"Транзакция {frm} -> {to} невозможна из-за ограничений минимального размера или точности")
                        valid = False
                        break

                    # Проверяем прибыльность перед выполнением
                    try:
                        current_ticker = self.market_client.get_ticker(symbol=sym) if self.production else {'buy': price_map[sym]['ask'], 'sell': price_map[sym]['bid']}
                        current_price = float(current_ticker['sell']) if direction == 'sell' else float(current_ticker['buy'])
                    except Exception as e:
                        self.log_message(f"Ошибка при получении текущих цен для {sym}: {str(e)}")
                        valid = False
                        break
                    expected_new_amt = amt * current_price * (1 - fee) if direction == 'sell' else (amt / current_price) * (1 - fee)
                    if expected_new_amt < amt:
                        self.log_message(f"Транзакция {frm} -> {to} убыточна: ожидается {expected_new_amt:.8f} {to}, текущий объем {amt:.8f} {frm}")
                        valid = False
                        break

                    if self.production:
                        # Проверяем баланс текущего актива
                        available = self.check_balance(frm)
                        if available < amt:
                            self.log_message(f"Недостаточно {frm} на балансе: доступно {available:.8f}, требуется {amt:.8f}")
                            valid = False
                            break

                        # Выполняем реальную сделку
                        actual_amount, success = self.execute_trade(frm, to, amt, sym, direction, price, fee)
                        if not success:
                            valid = False
                            break
                        new_amt = actual_amount
                    else:
                        # Симуляция
                        new_amt = expected_new_amt

                    trades.append((frm, to, sym, direction, new_amt, current_price))
                    amt = new_amt

            if valid:
                # Сохраняем результаты
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                with open('files/decisions/final_decisions.txt', 'a', encoding='utf-8') as f:
                    f.write(f"Time: {current_time}\n")
                    f.write(f"Path: {' -> '.join(best_op['path'])}\n")
                    f.write(f"Profit: {(amt / initial_deposit - 1) * 100:.4f}%\n")
                    f.write(f"Start Amount: {initial_deposit}{best_op['start_asset']}\n")
                    f.write(f"End Amount: {amt:.8f}{best_op['start_asset']}\n")
                    f.write("Calculation steps:\n")
                    step_amt = initial_deposit
                    for t in trades:
                        frm, to, sym, dirc, new_amt, price = t
                        if dirc == 'sell':
                            f.write(f"Selling {step_amt:.8f} {frm} for {to} using pair {sym} at price {price:.8f} ({to} per {frm})\n")
                            f.write(f"Amount received: {step_amt:.8f} * {price:.8f} * (1 - {fee}) = {new_amt:.8f} {to}\n")
                        else:
                            f.write(f"Buying {new_amt:.8f} {to} with {step_amt:.8f} {frm} using pair {sym} at price {price:.8f} ({frm} per {to})\n")
                            f.write(f"Amount bought: ({step_amt:.8f} / {price:.8f}) * (1 - {fee}) = {new_amt:.8f} {to}\n")
                        step_amt = new_amt
                    f.write("----\n")
                self.log_message(f"Арбитраж завершен: начальная сумма {initial_deposit:.8f} USDT, конечная сумма {amt:.8f} USDT")
                # Обновляем депозит
                self.deposit = amt
            else:
                self.log_message("Арбитраж не выполнен из-за ошибок в транзакциях или убыточности")

            # Сбрасываем счетчик после попытки
            self.consecutive_same = 0

    def find_arbitrage_cycles(self, fee_rate=0.001, min_profit=0.0001, start_amount=1.0,
                              max_cycles=200000, max_cycle_len=4):
        """
        Ищет циклы длины 3..max_cycle_len.
        """
        if max_cycle_len < 3:
            raise ValueError("max_cycle_len must be >= 3")

        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        opportunities = []
        checked = 0
        seen_cycles = set()

        assets = list(out_edges.keys())

        stop_flag = False

        def dfs(start, current_path):
            nonlocal checked, stop_flag

            if stop_flag:
                return

            current = current_path[-1]

            if len(current_path) >= 3 and start in out_edges[current]:
                cycle_nodes = current_path[:]
                norm = self.normalize_cycle(cycle_nodes)
                if norm not in seen_cycles:
                    seen_cycles.add(norm)
                    trades = []
                    amt = start_amount
                    valid = True
                    for i in range(len(cycle_nodes)):
                        frm = cycle_nodes[i]
                        to = cycle_nodes[(i + 1) % len(cycle_nodes)]
                        new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, fee_rate)
                        if new_amt is None:
                            valid = False
                            break
                        trades.append((frm, to, sym, direction, new_amt, price))
                        amt = new_amt
                    if valid:
                        profit = amt - start_amount
                        profit_perc = profit / start_amount
                        if profit_perc > 1.0:
                            valid = False
                        if valid and profit_perc > min_profit:
                            path_with_return = list(cycle_nodes) + [cycle_nodes[0]]
                            opportunities.append({
                                'path': path_with_return,
                                'start_asset': cycle_nodes[0],
                                'start_amount': start_amount,
                                'end_amount': amt,
                                'profit': profit,
                                'profit_perc': profit_perc,
                                'trades': trades
                            })

                checked += 1
                if checked > max_cycles:
                    stop_flag = True
                    return

            if len(current_path) >= max_cycle_len:
                return

            for nb in out_edges[current]:
                if stop_flag:
                    return
                if nb == start:
                    continue
                if nb in current_path:
                    continue
                checked += 1
                if checked > max_cycles:
                    stop_flag = True
                    return
                current_path.append(nb)
                dfs(start, current_path)
                current_path.pop()

        for a in assets:
            if stop_flag:
                break
            if len(out_edges[a]) < 1:
                continue
            dfs(a, [a])

        opportunities.sort(key=lambda x: x['profit_perc'], reverse=True)
        return opportunities

    def convert_amount(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        """
        Преобразует amount из from_asset в to_asset используя доступные пары.
        """
        sym_direct = symbol_map.get((from_asset, to_asset))
        if sym_direct:
            p = price_map.get(sym_direct)
            if not p:
                return None, None, None, None
            bid = p['bid']
            if amount < p['base_min_size']:
                return None, None, None, None
            new_amount = amount * bid * (1 - fee_rate)
            if new_amount < p['quote_min_size']:
                return None, None, None, None
            inc = p['quote_increment']
            new_amount = math.floor(new_amount / inc) * inc
            if new_amount <= 0:
                return None, None, None, None
            return new_amount, sym_direct, 'sell', bid

        sym_rev = symbol_map.get((to_asset, from_asset))
        if sym_rev:
            p = price_map.get(sym_rev)
            if not p:
                return None, None, None, None
            ask = p['ask']
            if amount < p['quote_min_size']:
                return None, None, None, None
            quantity = amount / ask
            if quantity < p['base_min_size']:
                return None, None, None, None
            new_amount = quantity * (1 - fee_rate)
            inc = p['base_increment']
            new_amount = math.floor(new_amount / inc) * inc
            if new_amount <= 0:
                return None, None, None, None
            return new_amount, sym_rev, 'buy', ask

        return None, None, None, None

    def normalize_cycle(self, nodes):
        """
        Нормализует цикл nodes (без повторного стартового элемента) для удаления ротационных дубликатов.
        """
        if not nodes:
            return tuple()
        n = len(nodes)
        rotations = []
        for i in range(n):
            rot = tuple(nodes[i:] + nodes[:i])
            rotations.append(rot)
        return min(rotations)

    def save_to_file(self, array, file_name):
        out_path = f'files/{file_name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(array, f, ensure_ascii=False, indent=2)
        self.log_message(f"fast reverberation result saved to {out_path}")