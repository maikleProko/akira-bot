# careful_arbitrage_parser.py

from collections import defaultdict
from datetime import datetime
import json
import math
import time
from decimal import Decimal, getcontext, ROUND_FLOOR
getcontext().prec = 28  # Точность для крипто

from files.trash.arbitrage_parser import ArbitrageParser as MarketProcess

class CarefulArbitrageParser(MarketProcess):
    def __init__(self, deposit=0.0001, production=True, api_key=None, api_secret=None, api_passphrase=None, strict=False, strict_coin='USDT', fee_rate=0.001, min_profit=0.005, use_all_balance=True, max_profit = 100.0):
        self.deposit = deposit
        self.production = production
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.consecutive_same = []
        self.fee_rate = fee_rate
        self.min_profit = min_profit
        self.possible = True
        self.prev_paths = None
        self.market_client = None
        self.trade_client = None
        self.user_client = None
        self.max_profit=max_profit
        self.ignore = []
        self.check_liquidity = False
        self.strict = strict
        self.strict_coin = strict_coin
        self.use_all_balance = use_all_balance
        self.restricted_pairs = set()  # Кэш заблокированных пар
        self.ignored_symbols = frozenset([
        ])
        if self.production:
            if not api_key or not api_secret:
                self.log_message("Ошибка: API-ключ и секрет необходимы для продакшен-режима")
                self.production = False
            else:
                try:
                    self.init_clients(api_key, api_secret, api_passphrase)
                    self.test_clients()
                    self.test_ticker()
                    self.log_message("Клиенты API успешно инициализированы")
                except Exception as e:
                    self.log_message(f"Не удалось инициализировать клиенты API: {str(e)}")
                    self.production = False

    def init_clients(self, api_key, api_secret, api_passphrase):
        raise NotImplementedError

    def test_clients(self):
        raise NotImplementedError

    def test_ticker(self):
        raise NotImplementedError

    def log_message(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"{timestamp}: {message}")
        with open('files/decisions/final_decisions.txt', 'a', encoding='utf-8') as f:
            f.write(f"{timestamp}: {message}\n")

    def fetch_symbols(self):
        raise NotImplementedError

    def process_symbol(self, s, symbol_map, out_edges, symbols):
        symbol = s['instId']
        if s['state'] != 'live':
            return
        base = s['baseCcy']
        quote = s['quoteCcy']
        if base in self.ignore or quote in self.ignore:
            return
        base_min_size = float(s['minSz'])
        base_increment = float(s['lotSz'])
        quote_increment = float(s['tickSz'])
        quote_min_size = self.get_quote_min_size(quote)
        if base_increment >= 1.0 or base_min_size >= 1.0:
            return
        symbol_map[(base, quote)] = symbol
        out_edges[base].add(quote)
        out_edges[quote].add(base)
        symbols.append((symbol, base, quote, base_min_size, quote_min_size, base_increment, quote_increment))

    def get_quote_min_size(self, quote):
        if quote == 'USDT' or quote == 'USDC':
            return 5.0
        elif quote == 'BTC':
            return 0.0001
        elif quote == 'ETH':
            return 0.001
        else:
            return 0.01

    def fetch_tickers(self):
        raise NotImplementedError

    def process_ticker(self, sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, ticker_map):
        ticker = ticker_map.get(sym)
        if not ticker:
            return None
        try:
            bid = float(ticker['bidPx'])
            ask = float(ticker['askPx'])
            if bid <= 0 or ask <= 0:
                return None
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
        except Exception as e:
            self.log_message(f"Ошибка при обработке цены для {sym}: {str(e)}")
            return None

    def build_graph_and_prices(self):
        info = self.fetch_symbols()
        symbol_map = {}
        out_edges = defaultdict(set)
        symbols = []
        for s in info:
            self.process_symbol(s, symbol_map, out_edges, symbols)
        price_map = {}
        try:
            tickers = self.fetch_tickers()
            ticker_map = {t['instId']: t for t in tickers}
            for sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment in symbols:
                entry = self.process_ticker(sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, ticker_map)
                if entry:
                    price_map[sym] = entry
        except Exception as e:
            self.log_message(f"Ошибка при получении всех тикеров: {str(e)}")
        return out_edges, symbol_map, price_map

    def check_balance(self, asset):
        raise NotImplementedError

    def fetch_ticker_price(self, symbol):
        raise NotImplementedError

    def validate_symbol(self, symbol, price_map):
        if price_map and symbol not in price_map:
            self.log_message(f"Ошибка: Символ {symbol} не найден в price_map")
            return False
        return True

    def get_size_constraints(self, entry):
        return (
            Decimal(str(entry.get('base_min_size', 0.0))),
            Decimal(str(entry.get('quote_min_size', 0.0))),
            Decimal(str(entry.get('base_increment', 0.00000001))),
            Decimal(str(entry.get('quote_increment', 0.00000001)))
        )

    def get_constraints(self, symbol, price_map, expected_price):
        if price_map:
            entry = price_map.get(symbol, {})
        else:
            entry = {}
        sizes = self.get_size_constraints(entry)
        ask = Decimal(str(entry.get('ask', expected_price)))
        return sizes + (ask,)

    def get_cycle_constraints(self, sym, price_map):
        entry = price_map.get(sym, {})
        return self.get_size_constraints(entry)

    def adjust_balance(self, from_asset, amount):
        available = Decimal(str(self.check_balance(from_asset)))
        amount_dec = Decimal(str(amount))
        if available < amount_dec:
            self.log_message(f"Недостаточно {from_asset} на балансе: доступно {available}, требуется {amount_dec}. Используем весь доступный баланс.")
            amount_dec = available
            if amount_dec == 0:
                self.log_message(f"Баланс {from_asset} равен нулю, транзакция невозможна.")
                return 0, False
        return float(amount_dec), True

    def adjust_sell_amount(self, amount, base_min_size, base_increment, from_asset, symbol):
        amount_dec = Decimal(str(amount))
        base_min_size_dec = Decimal(str(base_min_size))
        base_increment_dec = Decimal(str(base_increment))
        adjusted_amount_dec = (amount_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
        if adjusted_amount_dec < base_min_size_dec:
            self.log_message(f"Ошибка: Сумма {adjusted_amount_dec} {from_asset} меньше минимального размера {base_min_size_dec} для {symbol}")
            return 0, False
        if adjusted_amount_dec <= 0:
            self.log_message(f"Ошибка: После округления сумма {adjusted_amount_dec} {from_asset} равна нулю для {symbol}")
            return 0, False
        self.log_message(f"Скорректированная сумма для продажи: {adjusted_amount_dec} {from_asset} (base_increment: {base_increment_dec})")
        return float(adjusted_amount_dec), True

    def get_buy_ticker(self, symbol):
        ticker = self.fetch_ticker_price(symbol)
        if ticker is None or 'buy' not in ticker:
            self.log_message(f"Ошибка: Не удалось получить текущую цену для {symbol}")
            return None
        return float(ticker['buy'])

    def adjust_buy_funds(self, amount, available, quote_min_size, from_asset, symbol):
        amount_dec = Decimal(str(amount))
        available_dec = Decimal(str(available))
        quote_min_size_dec = Decimal(str(quote_min_size))
        funds_dec = min(amount_dec, available_dec)
        if funds_dec < quote_min_size_dec:
            self.log_message(f"Ошибка: Сумма {funds_dec} {from_asset} меньше минимального размера {quote_min_size_dec} для {symbol}")
            return 0, False
        return float(funds_dec), True

    def adjust_buy_amount(self, funds, current_ask_price, base_min_size, base_increment, quote_increment, to_asset, symbol, available, from_asset):
        funds_dec = Decimal(str(funds))
        current_ask_price_dec = Decimal(str(current_ask_price))
        base_min_size_dec = Decimal(str(base_min_size))
        base_increment_dec = Decimal(str(base_increment))
        quote_increment_dec = Decimal(str(quote_increment))
        available_dec = Decimal(str(available))

        max_base_amount_dec = funds_dec / current_ask_price_dec
        adjusted_base_amount_dec = (max_base_amount_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
        if adjusted_base_amount_dec < base_min_size_dec:
            self.log_message(f"Ошибка: Скорректированное количество {adjusted_base_amount_dec} {to_asset} меньше минимального размера {base_min_size_dec} для {symbol}")
            return 0, False
        adjusted_amount_dec = adjusted_base_amount_dec * current_ask_price_dec
        adjusted_amount_dec = (adjusted_amount_dec / quote_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * quote_increment_dec
        if adjusted_amount_dec <= 0:
            self.log_message(f"Ошибка: После округления сумма {adjusted_amount_dec} {from_asset} равна нулю для {symbol}")
            return 0, False
        if adjusted_amount_dec > available_dec:
            self.log_message(f"Ошибка: Скорректированная сумма {adjusted_amount_dec} {from_asset} превышает доступный баланс {available_dec}")
            return 0, False
        self.log_message(f"Скорректированная сумма для покупки: {adjusted_amount_dec} {from_asset} (quote_increment: {quote_increment_dec}, expected {adjusted_base_amount_dec} {to_asset} at price {current_ask_price_dec})")
        return float(adjusted_amount_dec), True

    def create_order_params(self, symbol, direction, ordType, adjusted_amount):
        raise NotImplementedError

    def place_order(self, order_params):
        raise NotImplementedError

    def monitor_order(self, symbol, order_id, direction, amount, from_asset, to_asset, expected_price, fee_rate):
        raise NotImplementedError

    def execute_trade(self, from_asset, to_asset, amount, symbol, direction, expected_price, fee_rate, price_map=None):
        if not self.validate_symbol(symbol, price_map):
            return None, False
        base_min_size, quote_min_size, base_increment, quote_increment, ask_price = self.get_constraints(symbol, price_map, expected_price)
        if not self.production:
            ticker = self.fetch_ticker_price(symbol)
            if ticker is None or 'buy' not in ticker or 'sell' not in ticker:
                self.log_message(f"Симуляция: недопустимый ответ get_ticker для {symbol}")
                return None, False
            current_price = float(ticker['sell']) if direction == 'sell' else float(ticker['buy'])
            new_amount = amount * current_price * (1 - fee_rate) if direction == 'sell' else (amount / current_price) * (1 - fee_rate)
            self.log_message(f"Симуляция транзакции {direction} {amount:.8f} {from_asset} -> {to_asset} при цене {current_price:.8f}")
            self.log_message(f"Ожидаемое количество: {new_amount:.8f} {to_asset}")
            return new_amount, True
        amount, success = self.adjust_balance(from_asset, amount)
        if not success:
            return None, False
        start_time = time.time()
        try:
            adjusted_amount = amount
            if direction == 'sell':
                adjusted_amount, success = self.adjust_sell_amount(amount, base_min_size, base_increment, from_asset, symbol)
                if not success:
                    return None, False
            else:
                current_ask_price = self.get_buy_ticker(symbol)
                if current_ask_price is None:
                    return None, False
                funds, success = self.adjust_buy_funds(amount, amount, quote_min_size, from_asset, symbol)
                if not success:
                    return None, False
                adjusted_amount, success = self.adjust_buy_amount(funds, current_ask_price, base_min_size, base_increment, quote_increment, to_asset, symbol, amount, from_asset)
                if not success:
                    return None, False
            order_params = self.create_order_params(symbol, direction, 'market', adjusted_amount)
            order_id = self.place_order(order_params)
            self.log_message(f"Размещен ордер {order_id} для {direction} {adjusted_amount:.8f} {from_asset} -> {to_asset}")
            return self.monitor_order(symbol, order_id, direction, adjusted_amount, from_asset, to_asset, expected_price, fee_rate)
        except Exception as e:
            self.log_message(f"Ошибка при выполнении транзакции {direction} {amount:.8f} {from_asset} -> {to_asset}: {str(e)}")
            self.log_message(f"Ограничения для {symbol}: base_min_size={base_min_size:.8f}, quote_min_size={quote_min_size:.8f}, base_increment={base_increment:.8f}, quote_increment={quote_increment:.8f}")
            return None, False

    def process_cycle_trade(self, amt, frm, to, symbol_map, price_map, fee_rate):
        sym = symbol_map.get((frm, to)) or symbol_map.get((to, frm))
        if not sym:
            return None, None, None, None
        direction = 'sell' if symbol_map.get((frm, to)) else 'buy'
        price = price_map[sym]['bid'] if direction == 'sell' else price_map[sym]['ask']
        return amt, sym, direction, price

    def validate_trade_constraints(self, amt, direction, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate, frm, to, sym):
        amt_dec = Decimal(str(amt))
        price_dec = Decimal(str(price))
        base_min_size_dec = Decimal(str(base_min_size))
        quote_min_size_dec = Decimal(str(quote_min_size))
        base_increment_dec = Decimal(str(base_increment))
        quote_increment_dec = Decimal(str(quote_increment))
        fee_dec = Decimal(str(fee_rate))

        if direction == 'sell':
            adjusted_amount_dec = (amt_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
            if adjusted_amount_dec < base_min_size_dec:
                return None
            new_amt_dec = adjusted_amount_dec * price_dec * (Decimal('1') - fee_dec)
            if new_amt_dec < quote_min_size_dec:
                return None
            new_amt_dec = (new_amt_dec / quote_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * quote_increment_dec
            if new_amt_dec <= 0:
                return None
            return float(new_amt_dec)
        else:
            max_base_amount_dec = amt_dec / price_dec
            adjusted_base_amount_dec = (max_base_amount_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
            if adjusted_base_amount_dec < base_min_size_dec:
                return None
            adjusted_amount_dec = adjusted_base_amount_dec * price_dec
            adjusted_amount_dec = (adjusted_amount_dec / quote_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * quote_increment_dec
            if adjusted_amount_dec < quote_min_size_dec:
                return None
            new_amt_dec = adjusted_base_amount_dec * (Decimal('1') - fee_dec)
            return float(new_amt_dec)


    def find_arbitrage_cycles(self, fee_rate=0.001, min_profit=0.0001, start_amount=1.0,
                              max_cycles=2000000, max_cycle_len=4, max_profit=100.0):
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
                        # Проверка на игнорируемые соседние пары
                        ignore_cycle = False
                        for i in range(len(cycle_nodes)):
                            frm = cycle_nodes[i]
                            to = cycle_nodes[(i + 1) % len(cycle_nodes)]
                            if (frm, to) in self.ignored_symbols or (to, frm) in self.ignored_symbols:
                                ignore_cycle = True
                                break
                        if not ignore_cycle:
                            profit = amt - start_amount
                            profit_perc = profit / start_amount
                            if min_profit < profit_perc < max_profit:
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

    def convert_direct(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
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
        return None, None, None, None

    def convert_reverse(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
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

    def convert_amount(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        direct = self.convert_direct(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)
        if direct[0] is not None:
            return direct
        return self.convert_reverse(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)

    def normalize_cycle(self, nodes):
        if not nodes:
            return tuple()
        n = len(nodes)
        rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(n)]
        return min(rotations)

    def save_to_file(self, array, file_name):
        out_path = f'files/{file_name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(array, f, ensure_ascii=False, indent=2)
        self.log_message(f"Результаты сохранены в {out_path}")

    def find_valid_ops(self, ops, strict):
        if strict:
            return [o for o in ops if o['start_asset'] == self.strict_coin and o['path'][0] == self.strict_coin and o['path'][-1] == self.strict_coin]
        return ops

    def check_pair_available(self, sym, direction, price_map):
        if sym in self.restricted_pairs:
            return False
        if not self.production:
            return True  # В симуляции все пары ок
        for attempt in range(3):  # Retry 3 times
            try:
                entry = price_map.get(sym, {})
                base_min_size = entry.get('base_min_size', 0.0001)
                quote_min_size = entry.get('quote_min_size', 5.0)
                base_increment = entry.get('base_increment', 1e-8)
                quote_increment = entry.get('quote_increment', 0.01)
                ticker = self.fetch_ticker_price(sym)
                if ticker is None:
                    raise Exception("Failed to fetch ticker for bad_price")
                bid = float(ticker['sell'])
                ask = float(ticker['buy'])
                if direction == 'sell':
                    adjusted_min_size = math.ceil(base_min_size / base_increment) * base_increment
                    bad_price = bid * 0.9  # Below market, won't fill
                    order_params = self.create_order_params(sym, direction, 'limit', adjusted_min_size)
                else:
                    min_quote = quote_min_size
                    adjusted_min_size = math.ceil(min_quote / quote_increment) * quote_increment
                    bad_price = ask * 1.1  # Above market, won't fill
                    order_params = self.create_order_params(sym, direction, 'limit', adjusted_min_size)
                order_params['px'] = str(bad_price)
                order_id = self.place_order(order_params)
                self.trade_client.cancel_order(instId=sym, ordId=order_id)
                return True
            except Exception as e:
                error_str = str(e)
                if '51155' in error_str or 'compliance' in error_str.lower():
                    self.log_message(f"Пара {sym} заблокирована по compliance (ошибка 51155)")
                    self.restricted_pairs.add(sym)
                    return False
                else:
                    self.log_message(f"Ошибка проверки пары {sym} (попытка {attempt+1}): {error_str}. Retry...")
                    time.sleep(1)  # Wait before retry
        self.log_message(f"Проверка пары {sym} провалилась после 3 попыток. Считаем недоступной.")
        return False

    def get_best_op(self, valid_ops):
        if valid_ops:
            if self.strict:
                valid_ops = [op for op in valid_ops if op['start_asset'] == self.strict_coin]
            return max(valid_ops, key=lambda x: x['profit_perc'])
        return None

    def check_op_warm(self, op, price_map):
        restricted = False
        for t in op['trades']:
            sym = t[2]  # sym из trades
            if not self.check_pair_available(sym, t[3], price_map):  # direction = t[3]
                restricted = True
                break
        if not restricted:
            return op
        else:
            return None

    def get_norm_path(self, best_op):
        if best_op:
            return self.normalize_cycle(best_op['path'][:-1])
        return None

    def update_consecutive(self, ops):
        current_norms = set()
        for op in ops:
            norm = self.normalize_cycle(op['path'][:-1])
            current_norms.add(norm)
        self.consecutive_same = [d for d in self.consecutive_same if d['path'] in current_norms]
        for norm in current_norms:
            existing = False
            for d in self.consecutive_same:
                if d['path'] == norm:
                    d['cons_value'] += 1
                    existing = True
                    break
            if not existing:
                self.consecutive_same.append({'path': norm, 'cons_value': 1})

    def get_cons_for_op(self, op):
        norm = self.normalize_cycle(op['path'][:-1])
        for d in self.consecutive_same:
            if d['path'] == norm:
                return d['cons_value']
        return 0

    def get_consecutive_for_best(self, best_op):
        if best_op:
            norm = self.get_norm_path(best_op)
            for d in self.consecutive_same:
                if d['path'] == norm:
                    return d['cons_value']
        return 0

    def get_sell_strings(self, amt, frm, to, sym, price, fee_rate, new_amt):
        line1 = f" Selling {amt:.8f} {frm} for {to} using pair {sym} at bid price {price:.8f} ({to} per {frm})"
        line2 = f" Amount received: {amt:.8f} * {price:.8f} * (1 - {fee_rate}) = {new_amt:.8f} {to}"
        return line1, line2

    def get_buy_strings(self, amt, frm, to, sym, price, fee_rate, new_amt):
        line1 = f" Buying {new_amt:.8f} {to} with {amt:.8f} {frm} using pair {sym} at ask price {price:.8f} ({frm} per {to})"
        line2 = f" Amount bought: ({amt:.8f} / {price:.8f}) * (1 - {fee_rate}) = {new_amt:.8f} {to}"
        return line1, line2

    def get_trade_strings(self, amt, t, fee_rate):
        frm, to, sym, dirc, new_amt, price = t
        if dirc == 'sell':
            return self.get_sell_strings(amt, frm, to, sym, price, fee_rate, new_amt)
        else:
            return self.get_buy_strings(amt, frm, to, sym, price, fee_rate, new_amt)

    def print_op(self, o, fee_rate):
        if o is None:
            return
        print("Путь:", " -> ".join(o['path']), f"Начало {o['start_amount']}{o['start_asset']} -> Конец {o['end_amount']:.8f}{o['start_asset']}", f"Прибыль {o['profit_perc']*100:.4f}%")
        print("Calculation steps:")
        amt = o['start_amount']
        for t in o['trades']:
            line1, line2 = self.get_trade_strings(amt, t, fee_rate)
            print(line1)
            print(line2)
            amt = t[4]
        print("----")

    def print_ops(self, ops):
        for o in ops[:50]:
            self.print_op(o, self.fee_rate)

    def check_enter_trade_mode(self, high_cons_ops, possible):
        return bool(high_cons_ops) and possible

    def init_trade_vars(self, best_op):
        initial_deposit = self.deposit
        amt = self.deposit
        trades = []
        valid = True
        return initial_deposit, amt, trades, valid

    def adjust_start_balance(self, start_asset):
        if not self.production:
            return self.deposit if not self.use_all_balance else float('inf'), True
        available_start = Decimal(str(self.check_balance(start_asset)))
        if available_start == 0:
            self.log_message(f"Баланс {start_asset} равен нулю, арбитраж невозможен.")
            return 0, False
        if self.use_all_balance:
            amt = available_start
            self.log_message(f"Используем весь доступный баланс {start_asset}: {amt}")
        else:
            amt = min(available_start, Decimal(str(self.deposit)))
            if amt < Decimal(str(self.deposit)):
                self.log_message(f"Недостаточно {start_asset} на балансе: доступно {available_start}, требуется {self.deposit}. Используем {amt}.")
        return float(amt), True

    def fetch_current_prices(self):
        raise NotImplementedError

    def get_current_price(self, sym, direction, current_prices, price_map):
        current_ticker = current_prices.get(sym)
        if current_ticker is None:
            self.log_message(f"Ошибка: Недопустимый ответ API для {sym}, использую price_map")
            return price_map[sym]['bid'] if direction == 'sell' else price_map[sym]['ask']
        return current_ticker['sell'] if direction == 'sell' else current_ticker['buy']

    def validate_sim_trade(self, cycle_amt, direction, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee, frm, sym, to):
        cycle_amt_dec = Decimal(str(cycle_amt))
        current_price_dec = Decimal(str(current_price))
        base_min_size_dec = Decimal(str(base_min_size))
        quote_min_size_dec = Decimal(str(quote_min_size))
        base_increment_dec = Decimal(str(base_increment))
        quote_increment_dec = Decimal(str(quote_increment))
        fee_dec = Decimal(str(fee))
        if direction == 'sell':
            adjusted_amount_dec = (cycle_amt_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
            if adjusted_amount_dec < base_min_size_dec:
                self.log_message(f"Симуляция: Сумма {adjusted_amount_dec} {frm} меньше минимального размера {base_min_size_dec} для {sym}")
                return None
            expected_new_amt_dec = adjusted_amount_dec * current_price_dec * (Decimal('1') - fee_dec)
            if expected_new_amt_dec < quote_min_size_dec:
                self.log_message(f"Симуляция: Ожидаемое количество {expected_new_amt_dec} {to} меньше минимального размера {quote_min_size_dec} для {sym}")
                return None
            return float(expected_new_amt_dec)
        else:
            max_base_amount_dec = cycle_amt_dec / current_price_dec
            adjusted_base_amount_dec = (max_base_amount_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
            if adjusted_base_amount_dec < base_min_size_dec:
                self.log_message(f"Симуляция: Скорректированное количество {adjusted_base_amount_dec} {to} меньше минимального размера {base_min_size_dec} для {sym}")
                return None
            adjusted_amount_dec = adjusted_base_amount_dec * current_price_dec
            adjusted_amount_dec = (adjusted_amount_dec / quote_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * quote_increment_dec
            if adjusted_amount_dec < quote_min_size_dec:
                self.log_message(f"Симуляция: Скорректированная сумма {adjusted_amount_dec} {frm} меньше минимального размера {quote_min_size_dec} для {sym}")
                return None
            return float(adjusted_base_amount_dec * (Decimal('1') - fee_dec))

    def simulate_cycle_loop(self, best_op, amt, symbol_map, price_map, fee, current_prices):
        cycle_amt = amt
        is_profitable = True
        simulated_trades = []
        for i in range(len(best_op['path']) - 1):
            frm = best_op['path'][i]
            to = best_op['path'][i + 1]
            new_amt, sym, direction, price = self.convert_amount(cycle_amt, frm, to, symbol_map, price_map, fee)
            if new_amt is None:
                self.log_message(f"Симуляция: Транзакция {frm} -> {to} невозможна из-за ограничений минимального размера или точности")
                return None, False, None
            current_price = self.get_current_price(sym, direction, current_prices, price_map)
            base_min_size, quote_min_size, base_increment, quote_increment = self.get_cycle_constraints(sym, price_map)
            expected_new_amt = self.validate_sim_trade(cycle_amt, direction, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee, frm, sym, to)
            if expected_new_amt is None:
                return None, False, None
            simulated_trades.append((frm, to, sym, direction, expected_new_amt, current_price))
            cycle_amt = expected_new_amt
        return cycle_amt, is_profitable, simulated_trades

    def simulate_cycle(self, best_op, amt, symbol_map, price_map, fee, min_profit, max_profit):
        current_prices = self.fetch_current_prices()
        if current_prices is None:
            return None, False, None, False
        cycle_amt, is_profitable, simulated_trades = self.simulate_cycle_loop(best_op, amt, symbol_map, price_map, fee, current_prices)
        if cycle_amt is None:
            return None, False, None, False
        profit_perc = (cycle_amt / amt - 1) * 100
        if not is_profitable or min_profit * 100 >= profit_perc >= max_profit * 100:
            self.log_message(f"Цикл {' -> '.join(best_op['path'])} отклонён: прогнозируемая прибыль {profit_perc:.4f}% меньше/выше диапазона или убыточна")
            return None, False, None, False
        self.log_message(f"Симуляция цикла успешна: прогнозируемая прибыль {profit_perc:.4f}%")
        self.log_simulated_steps(simulated_trades, fee, amt)
        return simulated_trades, is_profitable, cycle_amt, True

    def log_simulated_steps(self, simulated_trades, fee, initial_deposit):
        self.log_message("Симулированные шаги:")
        step_amt = initial_deposit
        for t in simulated_trades:
            line1, line2 = self.get_trade_strings(step_amt, t, fee)
            self.log_message(line1)
            self.log_message(line2)
            step_amt = t[4]
        self.log_message("----")

    def execute_cycle_loop(self, best_op, amt, trades, valid, symbol_map, price_map, fee):
        self.log_path_and_profit(best_op)
        self.log_calc_steps(best_op, fee)
        is_profitable = True
        for i in range(len(best_op['path']) - 1):
            frm = best_op['path'][i]
            to = best_op['path'][i + 1]
            new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, fee)
            if new_amt is None:
                self.log_message(f"Транзакция {frm} -> {to} невозможна из-за ограничений минимального размера или точности")
                valid = False
                break
            current_ticker = self.fetch_ticker_price(sym)
            self.log_message(f"Raw ticker response для {sym}: {current_ticker}")
            if current_ticker is None or 'buy' not in current_ticker or 'sell' not in current_ticker:
                self.log_message(f"Ошибка: Недопустимый ответ API для {sym}, пропускаем транзакцию")
                valid = False
                break
            current_price = float(current_ticker['sell']) if direction == 'sell' else float(current_ticker['buy'])
            expected_new_amt = amt * current_price * (1 - fee) if direction == 'sell' else (amt / current_price) * (1 - fee)
            if self.production:
                available = self.check_balance(frm)
                if available < amt:
                    self.log_message(f"Недостаточно {frm} на балансе: доступно {available:.8f}, требуется {amt:.8f}. Используем весь доступный баланс.")
                    amt = available
                    if amt == 0:
                        self.log_message(f"Баланс {frm} равен нулю, транзакция невозможна.")
                        valid = False
                        break
                    new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, fee)
                    if new_amt is None:
                        self.log_message(f"Транзакция {frm} -> {to} невозможна с доступным балансом {amt:.8f} из-за ограничений минимального размера или точности")
                        valid = False
                        break
                actual_amount, success = self.execute_trade(frm, to, amt, sym, direction, price, fee, price_map)
                if not success:
                    valid = False
                    break
                new_amt = actual_amount
            else:
                new_amt = expected_new_amt
            trades.append((frm, to, sym, direction, new_amt, current_price))
            amt = new_amt
            time.sleep(0.8)
        return amt, trades, valid, is_profitable

    def execute_cycle(self, best_op, initial_deposit, amt, trades, valid, symbol_map, price_map, fee):
        amt, trades, valid, is_profitable = self.execute_cycle_loop(best_op, amt, trades, valid, symbol_map, price_map, fee)
        return amt, trades, valid, is_profitable

    def log_path_and_profit(self, best_op):
        self.log_message(f"Путь: {' -> '.join(best_op['path'])}")
        self.log_message(f"Начало {best_op['start_amount']}{best_op['start_asset']} -> Конец {best_op['end_amount']:.8f}{best_op['start_asset']}")
        self.log_message(f"Прибыль {best_op['profit_perc'] * 100:.4f}%")

    def log_calc_steps(self, best_op, fee):
        self.log_message("Calculation steps:")
        step_amt = best_op['start_amount']
        for t in best_op['trades']:
            line1, line2 = self.get_trade_strings(step_amt, t, fee)
            self.log_message(line1)
            self.log_message(line2)
            step_amt = t[4]
        self.log_message("----")

    def get_sell_write(self, step_amt, frm, to, sym, price, fee, new_amt, f):
        f.write(f"Selling {step_amt:.8f} {frm} for {to} using pair {sym} at price {price:.8f} ({to} per {frm})\n")
        f.write(f"Amount received: {step_amt:.8f} * {price:.8f} * (1 - {fee}) = {new_amt:.8f} {to}\n")

    def get_buy_write(self, step_amt, frm, to, sym, price, fee, new_amt, f):
        f.write(f"Buying {new_amt:.8f} {to} with {step_amt:.8f} {frm} using pair {sym} at price {price:.8f} ({frm} per {to})\n")
        f.write(f"Amount bought: ({step_amt:.8f} / {price:.8f}) * (1 - {fee}) = {new_amt:.8f} {to}\n")

    def write_trade_step(self, step_amt, t, fee, f):
        frm, to, sym, dirc, new_amt, price = t
        if dirc == 'sell':
            self.get_sell_write(step_amt, frm, to, sym, price, fee, new_amt, f)
        else:
            self.get_buy_write(step_amt, frm, to, sym, price, fee, new_amt, f)

    def save_trade_results(self, best_op, amt, initial_deposit, trades, is_profitable, fee):
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        with open('files/decisions/final_decisions.txt', 'a', encoding='utf-8') as f:
            f.write(f"Time: {current_time}\n")
            f.write(f"Path: {' -> '.join(best_op['path'])}\n")
            profit_perc = (amt / initial_deposit - 1) * 100
            f.write(f"Profit: {profit_perc:.4f}%{' (убыточный цикл)' if not is_profitable else ''}\n")
            f.write(f"Start Amount: {initial_deposit}{best_op['start_asset']}\n")
            f.write(f"End Amount: {amt:.8f}{best_op['start_asset']}\n")
            f.write("Calculation steps:\n")
            step_amt = initial_deposit
            for t in trades:
                self.write_trade_step(step_amt, t, fee, f)
                step_amt = t[4]
            f.write("----\n")
        self.log_message(f"Арбитраж завершен: начальная сумма {initial_deposit:.8f} {best_op['start_asset']}, конечная сумма {amt:.8f} {best_op['start_asset']}{' (убыточный цикл)' if not is_profitable else ''}")
        self.deposit = amt

    def run_realtime_init(self, strict):
        fee = self.fee_rate
        min_profit = self.min_profit
        start = 1.0
        max_len = 6
        ops = self.find_arbitrage_cycles(fee_rate=fee, min_profit=min_profit, start_amount=start, max_cycles=20000000000, max_cycle_len=max_len, max_profit=self.max_profit)
        self.update_consecutive(ops)
        valid_ops = self.find_valid_ops(ops, strict)
        best_op = self.get_best_op(valid_ops)
        return ops, valid_ops, best_op

    def run_realtime_print(self, ops):
        if not ops:
            self.log_message("Арбитражных возможностей не найдено (по заданному порогу).")
        else:
            #self.print_ops(ops)
            pass

    def run_realtime_trade(self, selected_op):
        self.possible = False
        self.log_message("Входим в режим активного трейдера")
        initial_deposit, amt, trades, valid = self.init_trade_vars(selected_op)
        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        start_asset = selected_op['start_asset']
        amt, valid = self.adjust_start_balance(start_asset)
        if not valid:
            return
        self.log_message(f"Проверка прибыльности цикла: {' -> '.join(selected_op['path'])}")
        simulated_trades, is_profitable, cycle_amt, valid = self.simulate_cycle(selected_op, amt, symbol_map, price_map, self.fee_rate, min_profit=self.min_profit, max_profit=self.max_profit)
        if not valid:
            return
        amt, trades, valid, is_profitable = self.execute_cycle(selected_op, initial_deposit, amt, trades, valid, symbol_map, price_map, self.fee_rate)
        if valid:
            self.save_trade_results(selected_op, amt, initial_deposit, trades, is_profitable, self.fee_rate)
        else:
            self.log_message("Арбитраж не выполнен из-за ошибок в транзакциях или убыточности цикла")
        norm = self.get_norm_path(selected_op)
        self.consecutive_same = [d for d in self.consecutive_same if d['path'] != norm]

    def run_realtime(self):
        ops, valid_ops, best_op = self.run_realtime_init(self.strict)
        if best_op is not None:
            print('best op: ')
            self.print_op(best_op, self.fee_rate)
        else:
            self.log_message("Лучшая возможность арбитража не найдена.")
        self.run_realtime_print(ops)
        consecutive_same = self.get_consecutive_for_best(best_op)
        self.log_message(f"CHECKING: consecutive_same = {consecutive_same}")
        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        if consecutive_same > 2 and self.possible:
            best_op = self.check_op_warm(best_op, price_map)
            if best_op:
                self.run_realtime_trade(best_op)