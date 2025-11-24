from collections import defaultdict

from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.utils.logger import Logger
from scenarios.parsers.arbitrage_parser.rolling_arbitrage_parser.core.patterns.direct_conversion_pattern import \
    DirectConversionPattern
from scenarios.parsers.arbitrage_parser.rolling_arbitrage_parser.core.patterns.reverse_conversion_pattern import \
    ReverseConversionPattern
from utils.core.functions import MarketProcess


class AbstractArbitrageParser(MarketProcess):

    def __init__(self):
        self.ignore = None
        self.available_coins = None
        self.exchange_client = None
        self.logger = Logger()
        self.direct_strategy = DirectConversionPattern(self.logger)
        self.reverse_strategy = ReverseConversionPattern(self.logger)
        self.strict = True
        self.strict_coin = ''
        self.consecutive_same = []
        self.possible = False
        self.fee_rate = 0.001
        self.min_profit = 0.006
        self.max_profit = 0.1
        self.max_cycles = 20000000000
        self.max_cycle_len = 6

    def init(self,api_key, api_secret):
        raise NotImplementedError

    def run_realtime(self):
        raise NotImplementedError

    def get_objects(self, response):
        raise NotImplementedError

    def get_symbols(self):
        raise NotImplementedError

    def get_tickers(self):
        raise NotImplementedError

    def get_balance(self, asset):
        raise NotImplementedError

    def place_order(self, **order_params):
        raise NotImplementedError


    ### 1. GRAPH

    def _get_quote_min_size(self, quote):
        if quote == 'USDT' or quote == 'USDC':
            return 5.0
        elif quote == 'BTC':
            return 0.0001
        elif quote == 'ETH':
            return 0.001
        else:
            return 0.01

    def _process_symbol_checks(self, s):
        symbol, base, quote, base_min_size, base_increment, quote_increment = self._extract_symbol_data(s)
        if not s['state'] == 'live': # check_state_live
            return False
        if not base not in self.ignore and quote not in self.ignore: # check_ignore_assets
            return False
        if not self.available_coins is None or (base in self.available_coins and quote in self.available_coins): # check_available_coins
            return False
        if not base_increment < 1.0 and base_min_size < 1.0: # check_increments_valid
            return False
        return True

    def _extract_symbol_data(self, s):
        return s['symbol'], s['baseCoin'], s['quoteCoin'], float(s['lotSizeFilter']['minOrderQty']), float(s['lotSizeFilter']['basePrecision']), float(s['priceFilter']['tickSize'])

    def _process_symbols(self, info, symbol_map, out_edges, symbols):
        for s in info:
            if not self._process_symbol_checks(s):
                return
            symbol, base, quote, base_min_size, base_increment, quote_increment = self._extract_symbol_data(s)
            quote_min_size = self._get_quote_min_size(quote)
            symbol_map[(base, quote)] = symbol
            out_edges[base].add(quote)
            out_edges[quote].add(base)
            symbols.append((symbol, base, quote, base_min_size, quote_min_size, base_increment, quote_increment))

    def _fetch_and_map_tickers(self):
        tickers = self.exchange_client.fetch_tickers()
        ticker_map = {t['instId']: t for t in tickers}
        return ticker_map

    def _process_ticker_try(self, sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, ticker_map):
        ticker = ticker_map.get(sym)
        if not ticker:
            return None
        bid = float(ticker['bidPx'])
        ask = float(ticker['askPx'])
        if not bid > 0 and ask > 0:
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

    def _build_price_map(self, symbols, ticker_map, price_map):

        symbols = []
        for tup in symbols:
            symbol, base, quote, *rest = tup
            if quote != 'USDT' or base == 'BTC':
                symbols.append(tup)

        for sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment in symbols:
            try:
                entry = self._process_ticker_try(sym, base, quote, base_min_size, quote_min_size, base_increment, quote_increment, ticker_map)
                price_map[sym] = entry
            except:
                pass


    def build_graph_and_prices(self):
        info = self.get_symbols()
        symbol_map, out_edges, symbols = {}, defaultdict(set), []
        self._process_symbols(info, symbol_map, out_edges, symbols)
        price_map = {}
        try:
            ticker_map = self._fetch_and_map_tickers()
            self._build_price_map(symbols, ticker_map, price_map)
        except Exception as e:
            self.logger.log_message(f"Ошибка при получении всех тикеров: {str(e)}")
        return out_edges, symbol_map, price_map

    # 2. FIND CYCLES

    def _convert_amount(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        direct = self.direct_strategy.convert(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)
        if direct[0] is not None:
            return direct
        return self.reverse_strategy.convert(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)

    def _normalize_cycle(self, nodes):
        if not nodes:
            return tuple()
        n = len(nodes)
        rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(n)]
        return min(rotations)

    def _normalize_and_check_seen(self, cycle_nodes, seen_cycles):
        norm = self._normalize_cycle(cycle_nodes)
        if norm in seen_cycles:
            return True, None
        seen_cycles.add(norm)
        return False, norm

    def _process_cycle_trade(self, amt, frm, to, symbol_map, price_map, fee_rate, trades):
        new_amt, sym, direction, price = self._convert_amount(amt, frm, to, symbol_map, price_map, fee_rate)
        if new_amt is None:
            return None, False
        trades.append((frm, to, sym, direction, new_amt, price))
        return new_amt, True

    def _process_cycle_trades(self, cycle_nodes, start_amount, symbol_map, price_map, fee_rate):
        amt = start_amount
        trades = []
        valid = True
        for i in range(len(cycle_nodes)):
            frm = cycle_nodes[i]
            to = cycle_nodes[(i + 1) % len(cycle_nodes)]
            amt, valid = self._process_cycle_trade(amt, frm, to, symbol_map, price_map, fee_rate, trades)
            if not valid:
                break
        return amt, trades, valid

    def _check_ignore_cycle(self, cycle_nodes, ignored_symbols):
        for i in range(len(cycle_nodes)):
            frm = cycle_nodes[i]
            to = cycle_nodes[(i + 1) % len(cycle_nodes)]
            if (frm, to) in ignored_symbols or (to, frm) in ignored_symbols:
                return True
        return False

    def _add_opportunity_if_valid(self, valid, ignore_cycle, min_profit, max_profit, opportunities, cycle_nodes, start_amount, amt, trades):
        if not valid or ignore_cycle:
            return False
        profit = amt - start_amount
        profit_perc = profit / start_amount
        if not min_profit < profit_perc < max_profit:
            return False
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
        return True

    def _increment_checked(self, checked, max_cycles, stop_flag):
        checked += 1
        if checked > max_cycles:
            stop_flag = True
        return checked, stop_flag

    def _process_cycle_if_new(self, current_path, seen_cycles, start_amount, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities):
        cycle_nodes = current_path[:]
        norm = self._normalize_cycle(cycle_nodes)
        if norm in seen_cycles:
            return True, None
        seen_cycles.add(norm)
        seen, norm = False, norm
        if seen:
            return False
        amt, trades, valid = self._process_cycle_trades(cycle_nodes, start_amount, symbol_map, price_map, fee_rate)
        ignore_cycle = self._check_ignore_cycle(cycle_nodes, ignored_symbols)
        return self._add_opportunity_if_valid(valid, ignore_cycle, min_profit, max_profit, opportunities, cycle_nodes, start_amount, amt, trades)

    def _dfs(self, start, current_path, out_edges, seen_cycles, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities, checked, max_cycles, max_cycle_len, stop_flag):
        current = current_path[-1]
        if stop_flag:
            return checked, stop_flag
        if len(current_path) >= 3 and start in out_edges[current]:
            stop_flag = self._process_cycle_if_new(current_path, seen_cycles, self.start_amount, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities)
            checked, stop_flag = self._increment_checked(checked, max_cycles, stop_flag)
            if stop_flag:
                return checked, stop_flag
        if len(current_path) >= max_cycle_len or stop_flag:
            return checked, stop_flag
        for nb in out_edges[current]:
            if stop_flag:
                return checked, stop_flag
            if nb == start or nb in current_path:
                continue
            checked, stop_flag = self._increment_checked(checked, max_cycles, stop_flag)
            if stop_flag:
                return checked, stop_flag
            current_path.append(nb)
            checked, stop_flag = self._dfs(start, current_path, out_edges, seen_cycles, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities, checked, max_cycles, max_cycle_len, stop_flag)
            current_path.pop()
        return checked, stop_flag


    def find_arbitrage_cycles(self, out_edges, symbol_map, price_map, fee_rate, min_profit, start_amount, max_cycles,
                              max_cycle_len, max_profit, ignored_symbols):
        self.start_amount = start_amount
        if max_cycle_len < 3:
            raise ValueError("max_cycle_len must be >= 3")

        opportunities = []
        checked = 0
        seen_cycles = set()
        stop_flag = False

        assets = list(out_edges.keys())
        for a in assets:
            if stop_flag:
                break
            if len(out_edges[a]) < 1:
                continue
            checked, stop_flag = self._dfs(a, [a], out_edges, seen_cycles, symbol_map, price_map, fee_rate,
                                          ignored_symbols, min_profit, max_profit, opportunities, checked, max_cycles,
                                          max_cycle_len, stop_flag)
        opportunities = opportunities.sort(key=lambda x: x['profit_perc'], reverse=True)
        return opportunities
        

    # 3. BEST OP

    def _update_consecutive(self, ops):
        current_norms = set()
        for op in ops:
            norm = self._normalize_cycle(op['path'][:-1])
            current_norms.add(norm)

        self.consecutive_same = [d for d in self.consecutive_same if d['path'] in current_norms]
        for norm in current_norms:
            existing = False
            for d in self.consecutive_same:
                if d['path'] == norm:
                    d['cons_value'] += 1
                    existing = True
            if not existing:
                self.consecutive_same.append({'path': norm, 'cons_value': 1})


    def _find_valid_ops(self, ops, strict):
        if strict:
            return [o for o in ops if o['start_asset'] == self.strict_coin and o['path'][0] == self.strict_coin and o['path'][-1] == self.strict_coin]
        return ops

    def _get_best_op(self, valid_ops, strict):
        if valid_ops:
            if strict:
                valid_ops = [op for op in valid_ops if op['start_asset'] == self.strict_coin]
            return max(valid_ops, key=lambda x: x['profit_perc'])
        return None


    def calculate_best_op(self, cycles_ops, strict):
        self._update_consecutive(cycles_ops)
        valid_ops = self._find_valid_ops(cycles_ops, strict)
        #best_op = self._get_best_op(valid_ops, strict)
        return cycles_ops, valid_ops, valid_ops[0]

    # RUNTIME TRADE

    def start_trade(self, best_op, out_edges, symbol_map, price_map):
        raise NotImplementedError


    def run_realtime(self):
        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        ops = self.find_arbitrage_cycles(out_edges, symbol_map, price_map, self.fee_rate, self.min_profit, self.start_amount, self.max_cycles, self.max_cycle_len, self.max_profit, [])
        best_op = self.calculate_best_op(ops, self.strict)
        if len(self.consecutive_same) > 0  and self.possible:
            self.start_trade(best_op, out_edges, symbol_map, price_map)




