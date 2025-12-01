# abstract_arbitrage_parser.py
from collections import defaultdict
from typing import Dict, Tuple, Optional, List
import aiohttp
from colorama import Fore, Style, init
import asyncio
import time
from decimal import Decimal, ROUND_DOWN
init(autoreset=True) # Автоматический сброс цвета
from utils.core.logger import Logger
from utils.core.functions import MarketProcess
class AbstractArbitrageParser(MarketProcess):
    def __init__(
            self,
            ignore=None,
            available_coins=None,
            strict=True,
            strict_coin='BTC',
            possible=False,
            fee_rate=0.001,
            min_profit=0.006,
            max_profit=0.1,
            max_cycles=20000000000,
            max_cycle_len=6,
            api_key='',
            api_secret='',
            api_passphrase='-1',
            production=False,
            only_once=False,
            deposit=1000.0,
            is_real_fee=False,
            slippage_limit=0.003,
            max_retries=3,
            believe_score=1,
            is_testing_only_once=False,
            is_run_once=False,
    ):
        self.ignore = ignore or []
        self.available_coins = available_coins or []
        self.strict = strict
        self.strict_coin = strict_coin
        self.possible = possible
        self.fee_rate = fee_rate
        self.min_profit = min_profit
        self.max_profit = max_profit
        self.max_cycles = max_cycles
        self.max_cycle_len = max_cycle_len
        self.production = production
        self.only_once = only_once
        self.deposit = deposit
        self.is_real_fee = is_real_fee
        self.slippage_limit = slippage_limit
        self.max_retries = max_retries
        self.believe_score = believe_score
        self.is_testing_only_once = is_testing_only_once
        self.is_run_once = is_run_once
        self.logger = Logger()
        # Структуры для графа и цен
        self.out_edges: Dict[str, set] = defaultdict(set)
        self.symbol_map: Dict[Tuple[str, str], str] = {}
        self.price_map: Dict[str, Dict] = {} # Теперь с timestamp: {'bid':, 'ask':, ..., 'timestamp': time.time()}
        self.fee_map: Dict[str, float] = {}  # symbol -> fee_rate (реальная или фиксированная)
        self.lot_size_map: Dict[str, Dict] = {}  # symbol -> {'minOrderQty': Decimal, 'maxMktOrderQty': Decimal, 'qtyStep': Decimal, 'minNotionalValue': Decimal}
        # Кешируем тикеры
        self.tickers_cache: Dict[str, Dict] = {}
        self.consecutive_same = []
        self.start_amount = 0.0
        self.previous_path: Optional[Tuple] = None
        self.consecutive_count: int = 0
        self.init(api_key, api_secret, api_passphrase)
        self._build_symbol_map() # Строим symbol_map разово в init
    def _build_symbol_map(self):
        """
        Строим symbol_map и out_edges разово в init, без цен.
        Также заполняем fee_map в зависимости от is_real_fee и lot_size_map.
        """
        self.out_edges.clear()
        self.symbol_map.clear()
        self.fee_map.clear()
        self.lot_size_map.clear()
        symbols_info = self.get_symbols()
        # Получаем fees если is_real_fee
        if self.is_real_fee:
            fees = self.get_fees()
            fee_dict = {f.get('symbol'): float(f.get('takerFeeRate', self.fee_rate)) for f in fees}  # Предполагаем taker fee для market orders
        else:
            fee_dict = {}
        for s in symbols_info:
            symbol = s.get('instId') or s.get('symbol')
            base = s.get('baseCcy') or self._split_symbol(symbol)[0]
            quote = s.get('quoteCcy') or self._split_symbol(symbol)[1]
            if not base or not quote:
                continue
            if base in self.ignore or quote in self.ignore:
                continue
            if self.available_coins and (base not in self.available_coins or quote not in self.available_coins):
                continue
            self.symbol_map[(base, quote)] = symbol
            self.out_edges[base].add(quote)
            self.out_edges[quote].add(base)
            # Заполняем fee_map
            self.fee_map[symbol] = fee_dict.get(symbol, self.fee_rate)
            # Заполняем lot_size_map
            lot_filter = s.get('lotSizeFilter', {})
            self.lot_size_map[symbol] = {
                'minOrderQty': Decimal(lot_filter.get('minOrderQty', '0')),
                'maxMktOrderQty': Decimal(lot_filter.get('maxMktOrderQty', '0')),
                'qtyStep': Decimal(lot_filter.get('qtyStep', '0')),
                'minNotionalValue': Decimal(s.get('minNotionalValue', '0'))
            }
    # ==================== ОСНОВНОЙ ЦИКЛ РЕАЛТАЙМ ====================
    def run_realtime(self):
        """
        Основной метод реального времени.
        1. Получаем актуальные цены и строим граф
        2. Ищем арбитражные циклы с помощью DFS
        3. Если найден профитный путь — визуализируем расчёт, запускаем торговлю (реализация в наследниках)
        """
        while True:
            try:
                # Шаг 1: Получаем все цены с биржи и строим структуры (для поиска циклов нужно все)
                self._fetch_prices()
                # Шаг 2: Ищем арбитражные циклы
                best_path = self._find_arbitrage_paths()
                if best_path:
                    norm_path = self._normalize_path(best_path['path'])
                    if norm_path == self.previous_path:
                        self.consecutive_count += 1
                    else:
                        self.consecutive_count = 1
                        self.previous_path = norm_path
                    self.logger.print_message(f"Найден арбитражный путь: {' -> '.join(best_path['path'])} "
                                            f"| Прибыль: {best_path['profit']:.4%} | Consecutive: {self.consecutive_count}/{self.believe_score}")
                    if self.consecutive_count >= self.believe_score:
                        # Визуализация расчёта прибыли
                        self.visualize_cycle_precise(best_path, self.deposit)
                        # Запускаем торговлю всегда
                        asyncio.run(self.start_trade(best_path))
                        self.consecutive_count = 0
                        self.previous_path = None
                        if self.is_testing_only_once or self.is_run_once:
                            break  # Завершаем после одного запуска
                else:
                    self.logger.print_message("Арбитражных возможностей не найдено")
                    self.consecutive_count = 0
                    self.previous_path = None
            except Exception as e:
                self.logger.print_message(f"Ошибка в run_realtime: {e}")
                self.consecutive_count = 0
                self.previous_path = None
            if self.only_once:
                break
            time.sleep(1)  # Задержка между итерациями
    def _normalize_path(self, path: List[str]) -> Tuple:
        """
        Нормализует путь для сравнения, аналогично normalize_cycle.
        """
        if not path:
            return tuple()
        nodes = path[:-1]  # Убираем замыкание
        n = len(nodes)
        rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(n)]
        return min(rotations)
    # ==================== 1. ПАРСИНГ ЦЕН ====================
    def _fetch_prices(self):
        self.price_map.clear()
        self.tickers_cache.clear()
        # Получаем тикеры (для поиска циклов - все)
        raw_tickers = self.get_tickers()
        current_time = time.time()
        for ticker in raw_tickers:
            symbol = ticker.get('instId') or ticker.get('symbol')
            bid = float(ticker.get('bidPx') or ticker.get('bid1Price') or 0)
            ask = float(ticker.get('askPx') or ticker.get('ask1Price') or 0)
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            # Находим base и quote по symbol
            base_quote = next(((b, q) for (b, q), sym in self.symbol_map.items() if sym == symbol), None)
            if not base_quote:
                continue
            base, quote = base_quote
            self.price_map[symbol] = {
                'bid': bid,
                'ask': ask,
                'base': base,
                'quote': quote,
                'timestamp': current_time
            }
            self.tickers_cache[symbol] = ticker
    def _fetch_specific_prices(self, path: List[str]):
        """
        Обновляет price_map только для пар в пути (цикле).
        В test: использует кэш, без API.
        В prod: запрашивает API, если цена старая (>5 сек).
        """
        N = len(path)
        current_time = time.time()
        for i in range(N):
            from_coin = path[i]
            to_coin = path[(i + 1) % N]
            sym_direct = self.symbol_map.get((from_coin, to_coin))
            sym_rev = self.symbol_map.get((to_coin, from_coin))
            symbol = sym_direct or sym_rev
            if not symbol:
                self.logger.print_message(f"No symbol for {from_coin}-{to_coin}")
                continue
            if symbol in self.price_map and current_time - self.price_map[symbol].get('timestamp', 0) < 5:
                continue # Кэш свежий, пропускаем
            # Всегда запрашиваем свежий тикер, даже в тесте
            ticker = self.sync_get_symbol_price(symbol)
            if not ticker:
                self.logger.print_message(f"Failed to fetch price for {symbol}")
                continue
            bid = ticker['sell']
            ask = ticker['buy']
            if bid <= 0 or ask <= 0 or ask <= bid:
                continue
            base, quote = next(((b, q) for (b, q), sym in self.symbol_map.items() if sym == symbol), (None, None))
            if not base or not quote:
                continue
            self.price_map[symbol] = {
                'bid': bid,
                'ask': ask,
                'base': base,
                'quote': quote,
                'timestamp': current_time
            }
            self.tickers_cache[symbol] = {'symbol': symbol, 'bid1Price': bid, 'ask1Price': ask}
    async def _async_fetch_specific_prices(self, path: List[str]):
        """
        Асинхронная версия _fetch_specific_prices.
        В test: использует кэш асинхронно.
        """
        N = len(path)
        current_time = time.time()
        tasks = []
        symbols_to_fetch = []
        async with aiohttp.ClientSession() as session:
            for i in range(N):
                from_coin = path[i]
                to_coin = path[(i + 1) % N]
                sym_direct = self.symbol_map.get((from_coin, to_coin))
                sym_rev = self.symbol_map.get((to_coin, from_coin))
                symbol = sym_direct or sym_rev
                if not symbol:
                    continue
                if symbol in self.price_map and current_time - self.price_map[symbol].get('timestamp', 0) < 5:
                    continue
                # Всегда асинхронный fetch, даже в тесте
                symbols_to_fetch.append(symbol)
                tasks.append(self.async_get_symbol_price(session, symbol))
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, ticker in enumerate(results):
                    if isinstance(ticker, Exception) or not ticker:
                        continue
                    symbol = symbols_to_fetch[idx]
                    bid = ticker['sell']
                    ask = ticker['buy']
                    if bid <= 0 or ask <= 0 or ask <= bid:
                        continue
                    base, quote = next(((b, q) for (b, q), sym in self.symbol_map.items() if sym == symbol), (None, None))
                    if not base or not quote:
                        continue
                    self.price_map[symbol] = {
                        'bid': bid,
                        'ask': ask,
                        'base': base,
                        'quote': quote,
                        'timestamp': current_time
                    }
                    self.tickers_cache[symbol] = {'symbol': symbol, 'bid1Price': bid, 'ask1Price': ask}
    def _split_symbol(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Разбивает символ на base и quote.
        Поддерживает форматы: BTCUSDT, BTC/USDT, BTC-USDT и т.д.
        """
        symbol = symbol.replace('/', '').replace('-', '').upper()
        for sep in ['USDT', 'USDC', 'USD', 'BTC', 'ETH', 'BNB', 'EUR']:
            if symbol.endswith(sep):
                base = symbol[:-len(sep)]
                quote = sep
                return base, quote
        return None, None
    # ==================== 2. ПОИСК АРБИТРАЖА ====================
    def _find_arbitrage_paths(self) -> Optional[dict]:
        """
        Поиск арбитражных циклов с помощью DFS.
        Упрощенная версия без валидаций размеров для скорости.
        """
        if not self.out_edges:
            return None
        assets = list(self.out_edges.keys())
        # Если strict, ограничиваем стартовые монеты
        if self.strict and self.strict_coin in self.out_edges:
            start_assets = [self.strict_coin]
            self.logger.print_message(f"Strict mode: ищем только циклы через {self.strict_coin}")
        else:
            start_assets = assets
            self.logger.print_message(f"Свободный поиск по всем {len(assets)} монетам")
        opportunities = []
        checked = 0
        seen_cycles = set()
        stop_flag = False
        def normalize_cycle(nodes: List[str]) -> Tuple:
            if not nodes:
                return tuple()
            n = len(nodes)
            rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(n)]
            return min(rotations)
        def dfs(start: str, current_path: List[str]):
            nonlocal checked, stop_flag
            if stop_flag:
                return
            current = current_path[-1]
            if len(current_path) >= 3 and start in self.out_edges[current]:
                cycle_nodes = current_path[:]
                norm = normalize_cycle(cycle_nodes)
                if norm not in seen_cycles:
                    seen_cycles.add(norm)
                    # Упрощенный расчет профита без валидаций
                    multiplier = 1.0
                    valid = True
                    for i in range(len(cycle_nodes)):
                        frm = cycle_nodes[i]
                        to = cycle_nodes[(i + 1) % len(cycle_nodes)]
                        rate = self._get_rate(frm, to)
                        if rate is None:
                            valid = False
                            break
                        multiplier *= rate
                    if valid and multiplier > 1.0:
                        profit = multiplier - 1
                        if self.min_profit < profit < self.max_profit:
                            full_path = cycle_nodes + [cycle_nodes[0]]
                            opportunities.append({
                                'path': full_path,
                                'multiplier': multiplier,
                                'profit': profit,
                                'length': len(cycle_nodes)
                            })
                            if profit > 0.05:
                                stop_flag = True
                                return
                checked += 1
                if checked > self.max_cycles:
                    stop_flag = True
                    return
            if len(current_path) >= self.max_cycle_len:
                return
            for nb in self.out_edges[current]:
                if stop_flag:
                    return
                if nb == start or nb in current_path:
                    continue
                checked += 1
                if checked > self.max_cycles:
                    stop_flag = True
                    return
                current_path.append(nb)
                dfs(start, current_path)
                current_path.pop()
        for a in start_assets:
            if stop_flag:
                break
            if len(self.out_edges[a]) < 1:
                continue
            dfs(a, [a])
        if not opportunities:
            return None
        opportunities.sort(key=lambda x: x['profit'], reverse=True)
        return opportunities[0] # Возвращаем лучший
    def _get_rate(self, from_asset: str, to_asset: str) -> Optional[float]:
        """
        Получает rate с учетом fee, без валидаций размеров.
        """
        sym_direct = self.symbol_map.get((from_asset, to_asset))
        sym_rev = self.symbol_map.get((to_asset, from_asset))
        symbol = sym_direct or sym_rev
        if not symbol:
            return None
        p = self.price_map.get(symbol)
        if not p:
            return None
        fee = self.fee_map.get(symbol, self.fee_rate)
        if sym_direct:
            bid = p['bid']
            return bid * (1 - fee)
        else:
            ask = p['ask']
            return (1 / ask) * (1 - fee)
    def _get_raw_rate(self, from_asset: str, to_asset: str) -> Optional[float]:
        """
        Получает сырой rate без учета fee.
        """
        sym_direct = self.symbol_map.get((from_asset, to_asset))
        if sym_direct:
            p = self.price_map.get(sym_direct)
            if not p:
                return None
            bid = p['bid']
            return bid
        sym_rev = self.symbol_map.get((to_asset, from_asset))
        if sym_rev:
            p = self.price_map.get(sym_rev)
            if not p:
                return None
            ask = p['ask']
            return 1 / ask
        return None
    def _check_slippage(self, symbol: str) -> bool:
        p = self.price_map.get(symbol)
        if not p:
            return False
        bid = p['bid']
        ask = p['ask']
        if bid <= 0 or ask <= 0:
            return False
        slippage = (ask - bid) / bid
        return slippage <= self.slippage_limit
    def visualize_cycle_precise(self, cycle_data: dict, start_amount: float = 1000.0):
        """
        Максимально точная визуализация с учётом реального порядка сделок и комиссий.
        Использует реальные цены из price_map.
        """
        if not cycle_data:
            return
        path = cycle_data['path'][:-1] # убираем замыкающую монету
        self.logger.log_message(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 70}")
        self.logger.log_message(f"{Fore.CYAN}{Style.BRIGHT} ТОЧНЫЙ РАСЧЁТ АРБИТРАЖНОГО ЦИКЛА")
        self.logger.log_message(f"{'=' * 70}")
        current_amount = start_amount
        current_coin = path[0]
        total_fees = 0.0
        steps = []
        for i in range(len(path)):
            from_coin = path[i]
            to_coin = path[(i + 1) % len(path)]
            rate = self._get_raw_rate(from_coin, to_coin) # Сырой rate без fee
            if rate is None:
                self.logger.log_message(f"Ошибка: не удалось получить rate для {from_coin} -> {to_coin}")
                return
            sym_direct = self.symbol_map.get((from_coin, to_coin))
            symbol = sym_direct if sym_direct else self.symbol_map.get((to_coin, from_coin))
            is_reverse = sym_direct is None
            action = f"Покупаем {to_coin} за {from_coin}" if is_reverse else f"Продаём {from_coin} за {to_coin}"
            amount_before_fee = current_amount
            fee_rate = self.fee_map.get(symbol, self.fee_rate)
            fee = amount_before_fee * fee_rate
            total_fees += fee
            amount_after_fee = amount_before_fee - fee
            amount_received = amount_after_fee * rate
            steps.append({
                'step': i + 1,
                'action': action,
                'symbol': symbol,
                'rate': rate,
                'amount_in': amount_before_fee,
                'fee': fee,
                'amount_out': amount_received,
                'coin_out': to_coin
            })
            current_amount = amount_received
            current_coin = to_coin
        profit = (current_amount - start_amount) / start_amount
        self.logger.log_message(f"Старт: {start_amount:.4f} {path[0]}")
        self.logger.log_message(f"{'—' * 70}")
        for s in steps:
            color = Fore.MAGENTA if s['step'] == len(steps) else Fore.WHITE
            self.logger.log_message(f"{color}Шаг {s['step']}: {s['action']}")
            self.logger.log_message(f" {s['symbol']:>12} | Курс: {s['rate']:.10f}")
            self.logger.log_message(
                f" Ввод: {s['amount_in']:.6f} → Комиссия: {s['fee']:.6f} → Вывод: {s['amount_out']:.6f} {s['coin_out']}")
        self.logger.log_message(f"{'—' * 70}")
        self.logger.log_message(f"{Fore.GREEN}{Style.BRIGHT}ФИНИШ: {current_amount:.4f} {path[0]}")
        self.logger.log_message(f"{Fore.GREEN}{Style.BRIGHT}ПРИБЫЛЬ: {current_amount - start_amount:+,.4f} ({profit:+.5%})")
        self.logger.log_message(f"{Fore.RED}Комиссии: {total_fees:,.4f} ({total_fees / start_amount:.3%} от старта)")
        self.logger.log_message(f"{'=' * 70}\n")
    # ==================== АБСТРАКТНЫЕ МЕТОДЫ ====================
    def init(self, api_key, api_secret, api_passphrase):
        raise NotImplementedError
    def get_symbols(self):
        raise NotImplementedError
    def get_tickers(self):
        raise NotImplementedError
    def get_balance(self, asset):
        raise NotImplementedError
    def get_fees(self):
        raise NotImplementedError("get_fees должен быть реализован в наследнике")
    def place_order(self, **order_params):
        raise NotImplementedError
    def get_order_history(self, **params):
        raise NotImplementedError
    def sync_get_symbol_price(self, symbol):
        raise NotImplementedError
    async def async_get_symbol_price(self, session, symbol):
        raise NotImplementedError
    async def start_trade(self, path_data: dict):
        """
        Этот метод будет переопределяться в наследниках.
        """
        raise NotImplementedError("start_trade должен быть реализован в наследнике")
    # ==================== ТОРГОВЫЕ HELPERЫ ====================
    def _round_qty(self, qty: float, qty_step: Decimal, min_qty: Decimal) -> float:
        print(qty)
        print(qty_step)
        print(min_qty)
        d_qty = Decimal(str(qty))
        d_step = qty_step
        if d_step == 0:
            return float((d_qty // min_qty) * min_qty)
        rounded = (d_qty // d_step) * d_step
        if rounded < min_qty:
            return 0.0

        return float(rounded.quantize(d_step, rounding=ROUND_DOWN))
    def _get_trade_params(self, from_coin: str, to_coin: str, amount: float) -> Optional[dict]:
        """
        Определяет параметры для place_order: symbol, side, qty.
        Для market ордера на spot.
        """
        sym_direct = self.symbol_map.get((from_coin, to_coin))
        p = None
        is_sell = True
        if sym_direct:
            symbol = sym_direct
            p = self.price_map.get(symbol)
            if not p:
                return None
            side = 'Sell'
            rate = p['bid']
            qty = amount  # base amount
            notional = qty * rate
            is_sell = True
        else:
            sym_rev = self.symbol_map.get((to_coin, from_coin))
            if sym_rev:
                symbol = sym_rev
                p = self.price_map.get(symbol)
                if not p:
                    return None
                side = 'Buy'
                rate = p['ask']
                qty = amount / rate  # quote amount / ask to get base qty
                notional = amount
                is_sell = False
            else:
                return None
        if not self._check_slippage(symbol):
            self.logger.print_message(f"Slippage too high for {symbol}")
            return None
        lot_info = self.lot_size_map.get(symbol, {})
        min_qty = lot_info.get('minOrderQty', Decimal('0'))
        max_mkt_qty = lot_info.get('maxMktOrderQty', Decimal('inf'))
        qty_step = lot_info.get('qtyStep', Decimal('1'))
        min_notional = lot_info.get('minNotionalValue', Decimal('0'))
        if notional < float(min_notional):
            self.logger.print_message(f"Notional too low for {symbol}: {notional} < {min_notional}")
            return None
        qty_rounded = self._round_qty(qty, qty_step, min_qty)
        '''        
        if qty_rounded <= 0 or qty_rounded < float(min_qty) or qty_rounded > float(max_mkt_qty):
        self.logger.print_message(f"Invalid qty for {symbol}: {qty_rounded}")
        return None'''
        return {'symbol': symbol, 'side': side, 'qty': qty_rounded}
    async def async_place_order(self, **order_params):
        """
        Асинхронная версия place_order.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.place_order(**order_params))
    async def _async_fetch_prices(self):
        """
        Асинхронная версия _fetch_prices.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._fetch_prices)
    async def _check_order_filled(self, order_id: str, symbol: str) -> Tuple[float, float]:
        """
        Проверяет статус ордера и возвращает filled_qty и avg_price.
        """
        for attempt in range(5):
            history = self.get_order_history(category='spot', orderId=order_id, symbol=symbol)
            if history and history['retCode'] == 0 and history['result']['list']:
                order = history['result']['list'][0]
                status = order['orderStatus']
                filled_qty = float(order['cumExecQty'])
                avg_price = float(order['avgPrice']) if order['avgPrice'] else 0.0
                if status == 'Filled':
                    return filled_qty, avg_price
                elif status in ['Cancelled', 'Rejected']:
                    raise Exception(f"Order {order_id} failed: {status}")
            await asyncio.sleep(0.5)
        raise Exception(f"Order {order_id} not filled in time")
    def perform_trade(self, from_coin: str, to_coin: str, amount: float, balances: Optional[Dict[str, float]] = None) -> float:
        """
        Выполняет сделку: симуляция в test, реальный ордер в prod.
        """
        sym_direct = self.symbol_map.get((from_coin, to_coin))
        sym_rev = self.symbol_map.get((to_coin, from_coin))
        symbol = sym_direct or sym_rev
        if not symbol:
            self.logger.print_message(f"No symbol for {from_coin}-{to_coin}")
            return 0.0
        self._fetch_specific_prices([from_coin, to_coin]) # Обновление цен всегда
        rate = self._get_raw_rate(from_coin, to_coin)
        if rate is None:
            self.logger.print_message(f"Cannot get rate for {from_coin} -> {to_coin}")
            return 0.0
        fee_rate = self.fee_map.get(symbol, self.fee_rate)
        fee = amount * fee_rate
        amount_after_fee = amount - fee
        amount_received = amount_after_fee * rate
        if self.production:
            params = self._get_trade_params(from_coin, to_coin, amount)
            if params is None:
                self.logger.print_message(f"Cannot get trade params for {from_coin} -> {to_coin}")
                return 0.0
            for retry in range(self.max_retries):
                try:
                    order_id = self.place_order(category='spot', orderType='Market', **params)
                    filled_qty, avg_price = asyncio.run(self._check_order_filled(order_id, symbol))
                    if filled_qty < params['qty'] * 0.95:  # Partial fill tolerance
                        self.logger.print_message(f"Partial fill for {order_id}: {filled_qty}/{params['qty']}")
                    # Calculate actual received
                    if params['side'] == 'Sell':
                        actual_received = filled_qty * avg_price * (1 - fee_rate)
                    else:
                        actual_received = filled_qty * (1 - fee_rate)
                    # Update balances cache if provided
                    if balances is not None:
                        balances[from_coin] = self.get_balance(from_coin)  # Refresh
                        balances[to_coin] = self.get_balance(to_coin)
                    self.logger.print_message(f"Placed order {order_id} for {from_coin} -> {to_coin}: {amount} -> {actual_received}")
                    return actual_received
                except Exception as e:
                    self.logger.print_message(f"Error placing order (retry {retry+1}): {e}")
                    if retry == self.max_retries - 1:
                        return 0.0
                    time.sleep(1)
            return 0.0
        else:
            if balances is not None:
                balances[from_coin] -= amount
                balances[to_coin] += amount_received
            self.logger.print_message(f"Simulated trade {from_coin} -> {to_coin}: {amount} -> {amount_received}")
            return amount_received
    async def async_perform_trade(self, from_coin: str, to_coin: str, amount: float, balances: Optional[Dict[str, float]] = None) -> float:
        """
        Асинхронная версия perform_trade.
        """
        sym_direct = self.symbol_map.get((from_coin, to_coin))
        sym_rev = self.symbol_map.get((to_coin, from_coin))
        symbol = sym_direct or sym_rev
        if not symbol:
            self.logger.print_message(f"No symbol for {from_coin}-{to_coin}")
            return 0.0
        await self._async_fetch_specific_prices([from_coin, to_coin]) # Обновление цен всегда
        rate = self._get_raw_rate(from_coin, to_coin)
        if rate is None:
            self.logger.print_message(f"Cannot get rate for {from_coin} -> {to_coin}")
            return 0.0
        fee_rate = self.fee_map.get(symbol, self.fee_rate)
        fee = amount * fee_rate
        amount_after_fee = amount - fee
        amount_received = amount_after_fee * rate
        if self.production:
            params = self._get_trade_params(from_coin, to_coin, amount)
            if params is None:
                self.logger.print_message(f"Cannot get trade params for {from_coin} -> {to_coin}")
                return 0.0
            for retry in range(self.max_retries):
                try:
                    order_id = await self.async_place_order(category='spot', orderType='Market', **params)
                    filled_qty, avg_price = await self._check_order_filled(order_id, symbol)
                    if filled_qty < params['qty'] * 0.95:
                        self.logger.print_message(f"Partial fill for {order_id}: {filled_qty}/{params['qty']}")
                    if params['side'] == 'Sell':
                        actual_received = filled_qty * avg_price * (1 - fee_rate)
                    else:
                        actual_received = filled_qty * (1 - fee_rate)
                    if balances is not None:
                        balances[from_coin] = self.get_balance(from_coin)
                        balances[to_coin] = self.get_balance(to_coin)
                    self.logger.print_message(f"Placed async order {order_id} for {from_coin} -> {to_coin}: {amount} -> {actual_received}")
                    return actual_received
                except Exception as e:
                    self.logger.print_message(f"Error placing async order (retry {retry+1}): {e}")
                    if retry == self.max_retries - 1:
                        return 0.0
                    await asyncio.sleep(1)
            return 0.0
        else:
            if balances is not None:
                balances[from_coin] -= amount
                balances[to_coin] += amount_received
            self.logger.print_message(f"Simulated async trade {from_coin} -> {to_coin}: {amount} -> {amount_received}")
            return amount_received