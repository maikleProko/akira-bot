from collections import defaultdict
import requests
from files.trash.arbitrage_parser import ArbitrageParser
from datetime import datetime
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from kucoin.client import Market, Trade, User


class CarefulKuCoinArbitrageParser(ArbitrageParser):
    KUCOIN_API = "https://api.kucoin.com"

    def __init__(self, deposit=1.0, production=True, api_key=None, api_secret=None, api_passphrase=None):
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
        self.ignore = ['A', 'BULLA']  # Игнорируемые активы
        self.check_liquidity = False  # Опциональная проверка ликвидности
        if self.production:
            if not api_key or not api_secret:
                self.log_message("Ошибка: API-ключ и секрет необходимы для продакшен-режима")
                self.production = False
            else:
                try:
                    self.market_client = Market(key=api_key, secret=api_secret, passphrase=api_passphrase)
                    self.trade_client = Trade(key=api_key, secret=api_secret, passphrase=api_passphrase)
                    self.user_client = User(key=api_key, secret=api_secret, passphrase=api_passphrase)
                    accounts = self.user_client.get_account_list()
                    if accounts is None:
                        raise Exception("Тестовый запрос API для аккаунтов вернул None")
                    self.log_message(f"Список аккаунтов: {accounts}")
                    ticker_response = requests.get(f"{self.KUCOIN_API}/api/v1/market/orderbook/level1?symbol=ETH-USDT",
                                                   timeout=10)
                    ticker_response.raise_for_status()
                    ticker = ticker_response.json()['data']
                    if not ticker or 'bestBid' not in ticker or 'bestAsk' not in ticker:
                        raise Exception("Недопустимый ответ тикера")
                    self.log_message(f"Тикер для ETH-USDT: {ticker}")
                    self.log_message("Клиенты KuCoin API успешно инициализированы")
                except Exception as e:
                    self.log_message(f"Не удалось инициализировать клиенты KuCoin API: {str(e)}")
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
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return r.json()['data']
        except Exception as e:
            self.log_message(f"Ошибка при получении символов: {str(e)}")
            return []

    def build_graph_and_prices(self):
        """
        Возвращает:
          - out_edges: dict asset -> set(assets reachable за 1 сделку)
          - symbol_map: dict of (base,quote) -> symbol string
          - price_map: dict symbol -> {'bid': float, 'ask': float, 'base':..., 'quote':..., 'base_min_size':..., 'quote_min_size':..., 'base_increment':..., 'quote_increment':...}
        """
        info = self.fetch_symbols()

        symbol_map = {}
        out_edges = defaultdict(set)
        symbols = []

        for s in info:
            symbol = s['symbol']
            if not s.get('enableTrading', False):
                continue
            base = s['baseCurrency']
            quote = s['quoteCurrency']
            # Пропускаем пары с игнорируемыми активами
            if base in self.ignore or quote in self.ignore:
                continue
            base_min_size = float(s['baseMinSize'])
            quote_min_size = float(s['quoteMinSize'])
            base_increment = float(s['baseIncrement'])
            quote_increment = float(s['quoteIncrement'])

            # Фильтр для пар с нереалистичными ограничениями
            if base_increment >= 1.0 or base_min_size >= 1.0:
                continue

            symbol_map[(base, quote)] = symbol
            out_edges[base].add(quote)
            out_edges[quote].add(base)
            symbols.append((symbol, base, quote, base_min_size, quote_min_size, base_increment, quote_increment))

        # Получаем все цены одним запросом
        price_map = {}
        try:
            if self.production and self.market_client:
                all_tickers = self.market_client.get_all_tickers()
                tickers = all_tickers['ticker']
            else:
                response = requests.get(f"{self.KUCOIN_API}/api/v1/market/allTickers", timeout=10)
                response.raise_for_status()
                tickers = response.json()['data']['ticker']

            ticker_map = {t['symbol']: t for t in tickers}

            for sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment in symbols:
                ticker = ticker_map.get(sym)
                if not ticker:
                    continue
                try:
                    bid = float(ticker['sell'])  # bestBid
                    ask = float(ticker['buy'])  # bestAsk
                    if bid <= 0 or ask <= 0:
                        continue
                    price_map[sym] = {
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
        except Exception as e:
            self.log_message(f"Ошибка при получении всех тикеров: {str(e)}")

        return out_edges, symbol_map, price_map

    def check_balance(self, asset):
        """Проверяет баланс указанного актива"""
        if not self.production:
            return float('inf')  # В режиме симуляции считаем баланс бесконечным
        try:
            accounts = self.user_client.get_account_list()
            if accounts is None:
                self.log_message(f"Ошибка при проверке баланса для {asset}: API вернул None")
                return 0.0
            for account in accounts:
                if account['currency'] == asset and account['type'] == 'trade':
                    available = float(account['available'])
                    self.log_message(f"Баланс {asset}: {available:.8f}")
                    return available
            self.log_message(f"Баланс {asset}: 0.0 (аккаунт не найден)")
            return 0.0
        except Exception as e:
            self.log_message(f"Ошибка при проверке баланса для {asset}: {str(e)}")
            return 0.0

    def fetch_ticker_price(self, symbol):
        """Получает текущие цены (bid/ask) для символа через REST API"""
        start_time = time.time()
        try:
            response = requests.get(f"{self.KUCOIN_API}/api/v1/market/orderbook/level1?symbol={symbol}", timeout=5)
            response.raise_for_status()
            data = response.json()['data']
            if not data or 'bestBid' not in data or 'bestAsk' not in data:
                return None
            return {'sell': float(data['bestBid']), 'buy': float(data['bestAsk'])}
        except Exception as e:
            return None

    def execute_trade(self, from_asset, to_asset, amount, symbol, direction, expected_price, fee_rate, price_map=None):
        """Выполняет реальную сделку и возвращает фактический результат"""
        # Validate symbol exists in price_map
        if price_map and symbol not in price_map:
            self.log_message(f"Ошибка: Символ {symbol} не найден в price_map")
            return None, False

        # Get symbol constraints and current price
        if price_map:
            constraints = price_map.get(symbol, {})
            base_min_size = constraints.get('base_min_size', 0.0)
            quote_min_size = constraints.get('quote_min_size', 0.0)
            base_increment = constraints.get('base_increment', 0.00000001)
            quote_increment = constraints.get('quote_increment', 0.00000001)
            ask_price = constraints.get('ask', expected_price)
        else:
            base_min_size = quote_min_size = 0.0
            base_increment = quote_increment = 0.00000001
            ask_price = expected_price

        if not self.production:
            # Simulate trade
            ticker = self.fetch_ticker_price(symbol) if self.production else {'buy': expected_price,
                                                                              'sell': expected_price}
            if ticker is None or 'buy' not in ticker or 'sell' not in ticker:
                self.log_message(f"Симуляция: недопустимый ответ get_ticker для {symbol}")
                return None, False
            current_price = float(ticker['sell']) if direction == 'sell' else float(ticker['buy'])
            new_amount = amount * current_price * (1 - fee_rate) if direction == 'sell' else (
                                                                                                         amount / current_price) * (
                                                                                                         1 - fee_rate)
            self.log_message(
                f"Симуляция транзакции {direction} {amount:.8f} {from_asset} -> {to_asset} при цене {current_price:.8f}")
            self.log_message(f"Ожидаемое количество: {new_amount:.8f} {to_asset}")
            return new_amount, True

        # Check balance and adjust amount if necessary
        available = self.check_balance(from_asset)
        if available < amount:
            self.log_message(
                f"Недостаточно {from_asset} на балансе: доступно {available:.8f}, требуется {amount:.8f}. Используем весь доступный баланс.")
            amount = available
            if amount == 0:
                self.log_message(f"Баланс {from_asset} равен нулю, транзакция невозможна.")
                return None, False

        start_time = time.time()
        try:
            # Adjust amount based on direction and constraints
            adjusted_amount = amount
            if direction == 'sell':
                # For sell orders, adjust size (base currency) to base_increment
                if adjusted_amount < base_min_size:
                    self.log_message(
                        f"Ошибка: Сумма {adjusted_amount:.8f} {from_asset} меньше минимального размера {base_min_size:.8f} для {symbol}")
                    return None, False
                adjusted_amount = math.floor(adjusted_amount / base_increment) * base_increment
                if adjusted_amount <= 0:
                    self.log_message(
                        f"Ошибка: После округления сумма {adjusted_amount:.8f} {from_asset} равна нулю для {symbol}")
                    return None, False
                self.log_message(
                    f"Скорректированная сумма для продажи: {adjusted_amount:.8f} {from_asset} (base_increment: {base_increment:.8f})")
            else:
                # For buy orders, calculate max base amount affordable within balance and constraints
                ticker = self.fetch_ticker_price(symbol)
                if ticker is None or 'buy' not in ticker:
                    self.log_message(f"Ошибка: Не удалось получить текущую цену для {symbol}")
                    return None, False
                current_ask_price = float(ticker['buy'])
                funds = min(amount, available)  # Use minimum of requested amount and available balance
                if funds < quote_min_size:
                    self.log_message(
                        f"Ошибка: Сумма {funds:.8f} {from_asset} меньше минимального размера {quote_min_size:.8f} для {symbol}")
                    return None, False
                # Calculate max base amount affordable
                max_base_amount = funds / current_ask_price
                if max_base_amount < base_min_size:
                    self.log_message(
                        f"Ошибка: Максимальное количество {max_base_amount:.8f} {to_asset} меньше минимального размера {base_min_size:.8f} для {symbol}")
                    return None, False
                # Adjust base amount to base_increment
                adjusted_base_amount = math.floor(max_base_amount / base_increment) * base_increment
                if adjusted_base_amount < base_min_size:
                    self.log_message(
                        f"Ошибка: Скорректированное количество {adjusted_base_amount:.8f} {to_asset} меньше минимального размера {base_min_size:.8f} для {symbol}")
                    return None, False
                # Recalculate funds to match adjusted base amount
                adjusted_amount = adjusted_base_amount * current_ask_price
                adjusted_amount = math.floor(adjusted_amount / quote_increment) * quote_increment
                if adjusted_amount <= 0:
                    self.log_message(
                        f"Ошибка: После округления сумма {adjusted_amount:.8f} {from_asset} равна нулю для {symbol}")
                    return None, False
                if adjusted_amount > available:
                    self.log_message(
                        f"Ошибка: Скорректированная сумма {adjusted_amount:.8f} {from_asset} превышает доступный баланс {available:.8f}")
                    return None, False
                self.log_message(
                    f"Скорректированная сумма для покупки: {adjusted_amount:.8f} {from_asset} (quote_increment: {quote_increment:.8f}, expected {adjusted_base_amount:.8f} {to_asset} at price {current_ask_price:.8f})")

            # Place market order
            side = 'sell' if direction == 'sell' else 'buy'
            order = self.trade_client.create_market_order(
                symbol=symbol,
                side=side,
                size=adjusted_amount if direction == 'sell' else None,
                funds=adjusted_amount if direction == 'buy' else None
            )
            order_id = order['orderId']
            self.log_message(
                f"Размещен ордер {order_id} для {direction} {adjusted_amount:.8f} {from_asset} -> {to_asset}")

            # Monitor order status
            while time.time() - start_time < 5:
                order_details = self.trade_client.get_order_details(order_id)
                if order_details is None:
                    self.log_message(f"Ошибка: get_order_details для {order_id} вернул None")
                    return None, False
                if not order_details['isActive']:
                    execution_time = time.time() - start_time
                    filled_amount = float(order_details['dealSize'])
                    dealt_funds = float(order_details['dealFunds'])
                    actual_price = dealt_funds / filled_amount if direction == 'sell' else filled_amount / dealt_funds
                    actual_amount = dealt_funds * (1 - fee_rate) if direction == 'sell' else filled_amount * (
                                1 - fee_rate)
                    self.log_message(
                        f"Транзакция {direction} {adjusted_amount:.8f} {from_asset} -> {to_asset} завершена за {execution_time:.2f} сек.")
                    self.log_message(
                        f"Ожидаемая цена: {expected_price:.8f}, Фактическая средняя цена: {actual_price:.8f}")
                    self.log_message(
                        f"Ожидаемое количество: {(amount * expected_price * (1 - fee_rate) if direction == 'sell' else (amount / expected_price) * (1 - fee_rate)):.8f} {to_asset}, "
                        f"Фактическое количество: {actual_amount:.8f} {to_asset}")
                    return actual_amount, True
                time.sleep(0.1)

            self.log_message(
                f"Транзакция {direction} {adjusted_amount:.8f} {from_asset} -> {to_asset} не завершилась за 5 секунд, отменяется.")
            return None, False
        except Exception as e:
            self.log_message(
                f"Ошибка при выполнении транзакции {direction} {amount:.8f} {from_asset} -> {to_asset}: {str(e)}")
            self.log_message(
                f"Ограничения для {symbol}: base_min_size={base_min_size:.8f}, quote_min_size={quote_min_size:.8f}, "
                f"base_increment={base_increment:.8f}, quote_increment={quote_increment:.8f}")
            return None, False

    def find_arbitrage_cycles(self, fee_rate=0.001, min_profit=0.0001, start_amount=1.0,
                              max_cycles=200000, max_cycle_len=4):
        """
        Ищет циклы длины 3..max_cycle_len.
        """
        if max_cycle_len < 3:
            raise ValueError("max_cycle_len должен быть >= 3")

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
                        sym = symbol_map.get((frm, to)) or symbol_map.get((to, frm))
                        if not sym:
                            valid = False
                            break
                        direction = 'sell' if symbol_map.get((frm, to)) else 'buy'
                        price = price_map[sym]['bid'] if direction == 'sell' else price_map[sym]['ask']
                        # Проверяем ограничения
                        constraints = price_map.get(sym, {})
                        base_min_size = constraints.get('base_min_size', 0.0)
                        quote_min_size = constraints.get('quote_min_size', 0.0)
                        base_increment = constraints.get('base_increment', 0.00000001)
                        quote_increment = constraints.get('quote_increment', 0.00000001)
                        if direction == 'sell':
                            adjusted_amount = math.floor(amt / base_increment) * base_increment
                            if adjusted_amount < base_min_size:
                                valid = False
                                break
                            new_amt = adjusted_amount * price * (1 - fee_rate)
                            if new_amt < quote_min_size:
                                valid = False
                                break
                        else:
                            max_base_amount = amt / price
                            adjusted_base_amount = math.floor(max_base_amount / base_increment) * base_increment
                            if adjusted_base_amount < base_min_size:
                                valid = False
                                break
                            adjusted_amount = adjusted_base_amount * price
                            adjusted_amount = math.floor(adjusted_amount / quote_increment) * quote_increment
                            if adjusted_amount < quote_min_size:
                                valid = False
                                break
                            new_amt = adjusted_base_amount * (1 - fee_rate)
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
        self.log_message(f"Результаты сохранены в {out_path}")

    def run_realtime(self, strict=True):
        fee = 0.001  # 0.1% комиссия
        min_profit = 0.000001  # 0.1% порог
        start = 1.0
        max_len = 5  # ищем циклы длины 3 и 4

        ops = self.find_arbitrage_cycles(fee_rate=fee, min_profit=min_profit, start_amount=start,
                                         max_cycles=200000, max_cycle_len=max_len)

        # Находим наиболее прибыльный путь
        if strict:
            valid_ops = [o for o in ops if
                         o['start_asset'] == 'USDT' and o['path'][0] == 'USDT' and o['path'][-1] == 'USDT']
        else:
            valid_ops = ops
        current_best_path = None
        if valid_ops:
            best_op = max(valid_ops, key=lambda x: x['profit_perc'])
            current_best_path = tuple(best_op['path'])

        # Проверяем, совпадает ли лучший путь с предыдущим
        if current_best_path and current_best_path == self.prev_paths:
            self.consecutive_same += 1
        else:
            self.consecutive_same = 1
        self.prev_paths = current_best_path

        # Стандартное поведение (логирование в консоль)
        if not ops:
            self.log_message("Арбитражных возможностей не найдено (по заданному порогу).")
        else:
            if valid_ops and best_op:
                print("Путь:", " -> ".join(best_op['path']),
                      f"Начало {best_op['start_amount']}{best_op['start_asset']} -> Конец {best_op['end_amount']:.8f}{best_op['start_asset']}",
                      f"Прибыль {best_op['profit_perc'] * 100:.4f}%")
                print("Calculation steps:")
                amt = best_op['start_amount']
                for t in best_op['trades']:
                    frm, to, sym, dirc, new_amt, price = t
                    if dirc == 'sell':
                        print(
                            f"  Selling {amt:.8f} {frm} for {to} using pair {sym} at bid price {price:.8f} ({to} per {frm})")
                        print(f"  Amount received: {amt:.8f} * {price:.8f} * (1 - {fee}) = {new_amt:.8f} {to}")
                    else:
                        print(
                            f"  Buying {new_amt:.8f} {to} with {amt:.8f} {frm} using pair {sym} at ask price {price:.8f} ({frm} per {to})")
                        print(f"  Amount bought: ({amt:.8f} / {price:.8f}) * (1 - {fee}) = {new_amt:.8f} {to}")
                    amt = new_amt
                print("----")

        # Если 5 раз подряд один и тот же лучший путь, входим в режим активного трейдера
        self.log_message(f"CHECKING: consecutive_same = {self.consecutive_same}")
        if self.consecutive_same >= 2 and valid_ops and self.possible:
            # self.possible = False
            self.log_message("Входим в режим активного трейдера")
            # Выбираем лучшую возможность
            best_op = max(valid_ops, key=lambda x: x['profit_perc'])
            initial_deposit = self.deposit
            amt = self.deposit
            trades = []
            valid = True
            out_edges, symbol_map, price_map = self.build_graph_and_prices()  # Перезагружаем свежие цены

            # Проверяем баланс начального актива
            start_asset = best_op['start_asset']
            if self.production:
                available_start = self.check_balance(start_asset)
                if available_start < self.deposit:
                    self.log_message(
                        f"Недостаточно {start_asset} на балансе: доступно {available_start:.8f}, требуется {self.deposit:.8f}. Используем весь доступный баланс.")
                    amt = available_start
                    if amt == 0:
                        self.log_message(f"Баланс {start_asset} равен нулю, арбитраж невозможен.")
                        valid = False

            # Проверка прибыльности цикла с актуальными ценами
            if valid:
                self.log_message(f"Проверка прибыльности цикла: {' -> '.join(best_op['path'])}")
                cycle_amt = amt
                is_profitable = True
                simulated_trades = []
                # Параллельно запрашиваем цены для всех пар в пути
                syms = [symbol_map.get((best_op['path'][i], best_op['path'][i + 1])) or symbol_map.get(
                    (best_op['path'][i + 1], best_op['path'][i])) for i in range(len(best_op['path']) - 1)]
                current_prices = {}
                with ThreadPoolExecutor(max_workers=30) as executor:
                    future_to_sym = {executor.submit(self.fetch_ticker_price, sym): sym for sym in syms if sym}
                    for future in as_completed(future_to_sym):
                        sym = future_to_sym[future]
                        try:
                            current_ticker = future.result()
                            if current_ticker:
                                current_prices[sym] = current_ticker
                        except Exception as e:
                            self.log_message(f"Ошибка при параллельном получении цены для {sym}: {str(e)}")
                            valid = False
                            break

                if not valid:
                    self.log_message("Арбитраж не выполнен из-за ошибок в получении цен")
                    return

                for i in range(len(best_op['path']) - 1):
                    frm = best_op['path'][i]
                    to = best_op['path'][i + 1]
                    new_amt, sym, direction, price = self.convert_amount(cycle_amt, frm, to, symbol_map, price_map, fee)
                    if new_amt is None:
                        self.log_message(
                            f"Симуляция: Транзакция {frm} -> {to} невозможна из-за ограничений минимального размера или точности")
                        valid = False
                        break
                    # Получаем актуальную цену из параллельного запроса
                    current_ticker = current_prices.get(sym)
                    if current_ticker is None or 'buy' not in current_ticker or 'sell' not in current_ticker:
                        self.log_message(f"Ошибка: Недопустимый ответ API для {sym}, использую price_map")
                        current_price = price_map[sym]['bid'] if direction == 'sell' else price_map[sym]['ask']
                    else:
                        current_price = float(current_ticker['sell']) if direction == 'sell' else float(
                            current_ticker['buy'])
                    # Проверяем ограничения
                    constraints = price_map.get(sym, {})
                    base_min_size = constraints.get('base_min_size', 0.0)
                    quote_min_size = constraints.get('quote_min_size', 0.0)
                    base_increment = constraints.get('base_increment', 0.00000001)
                    quote_increment = constraints.get('quote_increment', 0.00000001)
                    if direction == 'sell':
                        adjusted_amount = math.floor(cycle_amt / base_increment) * base_increment
                        if adjusted_amount < base_min_size:
                            self.log_message(
                                f"Симуляция: Сумма {adjusted_amount:.8f} {frm} меньше минимального размера {base_min_size:.8f} для {sym}")
                            valid = False
                            break
                        expected_new_amt = adjusted_amount * current_price * (1 - fee)
                        if expected_new_amt < quote_min_size:
                            self.log_message(
                                f"Симуляция: Ожидаемое количество {expected_new_amt:.8f} {to} меньше минимального размера {quote_min_size:.8f} для {sym}")
                            valid = False
                            break
                    else:
                        max_base_amount = cycle_amt / current_price
                        adjusted_base_amount = math.floor(max_base_amount / base_increment) * base_increment
                        if adjusted_base_amount < base_min_size:
                            self.log_message(
                                f"Симуляция: Скорректированное количество {adjusted_base_amount:.8f} {to} меньше минимального размера {base_min_size:.8f} для {sym}")
                            valid = False
                            break
                        adjusted_amount = adjusted_base_amount * current_price
                        adjusted_amount = math.floor(adjusted_amount / quote_increment) * quote_increment
                        if adjusted_amount < quote_min_size:
                            self.log_message(
                                f"Симуляция: Скорректированная сумма {adjusted_amount:.8f} {frm} меньше минимального размера {quote_min_size:.8f} для {sym}")
                            valid = False
                            break
                        expected_new_amt = adjusted_base_amount * (1 - fee)
                    simulated_trades.append((frm, to, sym, direction, expected_new_amt, current_price))
                    cycle_amt = expected_new_amt
                    if cycle_amt < amt:
                        self.log_message(
                            f"Симуляция: Транзакция {frm} -> {to} убыточна: ожидается {cycle_amt:.8f} {to}, текущий объем {amt:.8f} {frm}")
                        is_profitable = False
                    amt = cycle_amt

                # Проверяем итоговую прибыль цикла
                profit_perc = (cycle_amt / initial_deposit - 1) * 100
                if not is_profitable or profit_perc <= min_profit * 100:
                    self.log_message(
                        f"Цикл {' -> '.join(best_op['path'])} отклонён: прогнозируемая прибыль {profit_perc:.4f}% меньше порога {min_profit * 100:.4f}% или убыточна")
                    valid = False
                else:
                    self.log_message(f"Симуляция цикла успешна: прогнозируемая прибыль {profit_perc:.4f}%")
                    self.log_message("Симулированные шаги:")
                    step_amt = initial_deposit
                    for t in simulated_trades:
                        frm, to, sym, dirc, new_amt, price = t
                        if dirc == 'sell':
                            self.log_message(
                                f"  Selling {step_amt:.8f} {frm} for {to} using pair {sym} at price {price:.8f} ({to} per {frm})")
                            self.log_message(
                                f"  Amount received: {step_amt:.8f} * {price:.8f} * (1 - {fee}) = {new_amt:.8f} {to}")
                        else:
                            self.log_message(
                                f"  Buying {new_amt:.8f} {to} with {step_amt:.8f} {frm} using pair {sym} at price {price:.8f} ({frm} per {to})")
                            self.log_message(
                                f"  Amount bought: ({step_amt:.8f} / {price:.8f}) * (1 - {fee}) = {new_amt:.8f} {to}")
                        step_amt = new_amt
                    self.log_message("----")

            # Выполняем цикл, если он прибыльный
            if valid:
                amt = initial_deposit
                for i in range(len(best_op['path']) - 1):
                    frm = best_op['path'][i]
                    to = best_op['path'][i + 1]
                    new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, fee)
                    if new_amt is None:
                        self.log_message(
                            f"Транзакция {frm} -> {to} невозможна из-за ограничений минимального размера или точности")
                        valid = False
                        break

                    self.log_message(f"Путь: {' -> '.join(best_op['path'])}")
                    self.log_message(
                        f"Начало {best_op['start_amount']}{best_op['start_asset']} -> Конец {best_op['end_amount']:.8f}{best_op['start_asset']}")
                    self.log_message(f"Прибыль {best_op['profit_perc'] * 100:.4f}%")
                    self.log_message("Calculation steps:")
                    step_amt = best_op['start_amount']
                    for t in best_op['trades']:
                        frm_t, to_t, sym_t, dirc, new_amt_t, price_t = t
                        if dirc == 'sell':
                            self.log_message(
                                f"  Selling {step_amt:.8f} {frm_t} for {to_t} using pair {sym_t} at bid price {price_t:.8f} ({to_t} per {frm_t})")
                            self.log_message(
                                f"  Amount received: {step_amt:.8f} * {price_t:.8f} * (1 - {fee}) = {new_amt_t:.8f} {to_t}")
                        else:
                            self.log_message(
                                f"  Buying {new_amt_t:.8f} {to_t} with {step_amt:.8f} {frm_t} using pair {sym_t} at ask price {price_t:.8f} ({frm_t} per {to_t})")
                            self.log_message(
                                f"  Amount bought: ({step_amt:.8f} / {price_t:.8f}) * (1 - {fee}) = {new_amt_t:.8f} {to_t}")
                        step_amt = new_amt_t
                    self.log_message("----")

                    # Проверяем прибыльность перед выполнением
                    current_ticker = self.fetch_ticker_price(sym) if self.production else {'buy': price_map[sym]['ask'],
                                                                                           'sell': price_map[sym][
                                                                                               'bid']}
                    self.log_message(f"Raw ticker response для {sym}: {current_ticker}")
                    if current_ticker is None or 'buy' not in current_ticker or 'sell' not in current_ticker:
                        self.log_message(f"Ошибка: Недопустимый ответ API для {sym}, использую price_map")
                        current_price = price_map[sym]['bid'] if direction == 'sell' else price_map[sym]['ask']
                    else:
                        current_price = float(current_ticker['sell']) if direction == 'sell' else float(
                            current_ticker['buy'])
                    expected_new_amt = amt * current_price * (1 - fee) if direction == 'sell' else (
                                                                                                           amt / current_price) * (
                                                                                                           1 - fee)
                    if expected_new_amt < amt:
                        self.log_message(
                            f"Транзакция {frm} -> {to} убыточна: ожидается {expected_new_amt:.8f} {to}, текущий объем {amt:.8f} {frm}")
                        is_profitable = False

                    if self.production:
                        # Проверяем баланс текущего актива
                        available = self.check_balance(frm)
                        if available < amt:
                            self.log_message(
                                f"Недостаточно {frm} на балансе: доступно {available:.8f}, требуется {amt:.8f}. Используем весь доступный баланс.")
                            amt = available
                            if amt == 0:
                                self.log_message(f"Баланс {frm} равен нулю, транзакция невозможна.")
                                valid = False
                                break
                            # Перепроверяем ограничения с новым amt
                            new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map,
                                                                                 fee)
                            if new_amt is None:
                                self.log_message(
                                    f"Транзакция {frm} -> {to} невозможна с доступным балансом {amt:.8f} из-за ограничений минимального размера или точности")
                                valid = False
                                break

                        # Выполняем реальную сделку
                        actual_amount, success = self.execute_trade(frm, to, amt, sym, direction, price, fee, price_map)
                        if not success:
                            valid = False
                            break
                        new_amt = actual_amount
                    else:
                        # Симуляция
                        new_amt = expected_new_amt

                    trades.append((frm, to, sym, direction, new_amt, current_price))
                    amt = new_amt
                    time.sleep(0.1)  # Avoid rate limits

            if valid:
                # Сохраняем результаты
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
                        frm, to, sym, dirc, new_amt, price = t
                        if dirc == 'sell':
                            f.write(
                                f"Selling {step_amt:.8f} {frm} for {to} using pair {sym} at price {price:.8f} ({to} per {frm})\n")
                            f.write(
                                f"Amount received: {step_amt:.8f} * {price:.8f} * (1 - {fee}) = {new_amt:.8f} {to}\n")
                        else:
                            f.write(
                                f"Buying {new_amt:.8f} {to} with {step_amt:.8f} {frm} using pair {sym} at price {price:.8f} ({frm} per {to})\n")
                            f.write(
                                f"Amount bought: ({step_amt:.8f} / {price:.8f}) * (1 - {fee}) = {new_amt:.8f} {to}\n")
                        step_amt = new_amt
                    f.write("----\n")
                self.log_message(
                    f"Арбитраж завершен: начальная сумма {initial_deposit:.8f} {best_op['start_asset']}, конечная сумма {amt:.8f} {best_op['start_asset']}{' (убыточный цикл)' if not is_profitable else ''}")
                # Обновляем депозит
                self.deposit = amt
            else:
                self.log_message("Арбитраж не выполнен из-за ошибок в транзакциях или убыточности цикла")

            # Сбрасываем счетчик после попытки
            self.consecutive_same = 0