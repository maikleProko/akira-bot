from scenarios.parsers.arbitrage_parser.core.cycle_executors.patterns.direct_conversion_pattern import \
    DirectConversionPattern
from scenarios.parsers.arbitrage_parser.core.cycle_executors.patterns.reverse_conversion_pattern import \
    ReverseConversionPattern


class CycleFinder:
    def __init__(self, logger):
        self.logger = logger
        self.direct_strategy = DirectConversionPattern(logger)
        self.reverse_strategy = ReverseConversionPattern(logger)

    def try_direct_convert(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        return self.direct_strategy.convert(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)

    def try_reverse_convert(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        return self.reverse_strategy.convert(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)

    def convert_amount(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        direct = self.try_direct_convert(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)
        if direct[0] is not None:
            return direct
        return self.try_reverse_convert(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)

    def normalize_cycle(self, nodes):
        if not nodes:
            return tuple()
        n = len(nodes)
        rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(n)]
        return min(rotations)

    def check_cycle_len(self, current_path):
        return len(current_path) >= 3

    def check_start_neighbor(self, start, current, out_edges):
        return start in out_edges[current]

    def normalize_and_check_seen(self, cycle_nodes, seen_cycles):
        norm = self.normalize_cycle(cycle_nodes)
        if norm in seen_cycles:
            return True, None
        seen_cycles.add(norm)
        return False, norm

    def init_cycle_trade_vars(self, start_amount):
        amt = start_amount
        trades = []
        valid = True
        return amt, trades, valid

    def process_cycle_trade(self, amt, frm, to, symbol_map, price_map, fee_rate, trades):
        new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, fee_rate)
        if new_amt is None:
            return None, False
        trades.append((frm, to, sym, direction, new_amt, price))
        return new_amt, True

    def process_cycle_trades(self, cycle_nodes, start_amount, symbol_map, price_map, fee_rate):
        amt, trades, valid = self.init_cycle_trade_vars(start_amount)
        for i in range(len(cycle_nodes)):
            frm = cycle_nodes[i]
            to = cycle_nodes[(i + 1) % len(cycle_nodes)]
            amt, valid = self.process_cycle_trade(amt, frm, to, symbol_map, price_map, fee_rate, trades)
            if not valid:
                break
        return amt, trades, valid

    def check_ignore_cycle(self, cycle_nodes, ignored_symbols):
        for i in range(len(cycle_nodes)):
            frm = cycle_nodes[i]
            to = cycle_nodes[(i + 1) % len(cycle_nodes)]
            if (frm, to) in ignored_symbols or (to, frm) in ignored_symbols:
                return True
        return False

    def calculate_profit(self, amt, start_amount):
        profit = amt - start_amount
        profit_perc = profit / start_amount
        return profit, profit_perc

    def check_profit_range(self, profit_perc, min_profit, max_profit):
        return min_profit < profit_perc < max_profit

    def create_opportunity(self, cycle_nodes, start_amount, amt, profit, profit_perc, trades):
        path_with_return = list(cycle_nodes) + [cycle_nodes[0]]
        return {
            'path': path_with_return,
            'start_asset': cycle_nodes[0],
            'start_amount': start_amount,
            'end_amount': amt,
            'profit': profit,
            'profit_perc': profit_perc,
            'trades': trades
        }

    def add_opportunity_if_valid(self, valid, ignore_cycle, min_profit, max_profit, opportunities, cycle_nodes, start_amount, amt, trades):
        if not valid or ignore_cycle:
            return
        profit, profit_perc = self.calculate_profit(amt, start_amount)
        if not self.check_profit_range(profit_perc, min_profit, max_profit):
            return
        opp = self.create_opportunity(cycle_nodes, start_amount, amt, profit, profit_perc, trades)
        opportunities.append(opp)

    def process_cycle_if_new(self, current_path, seen_cycles, start_amount, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities):
        cycle_nodes = current_path[:]
        seen, norm = self.normalize_and_check_seen(cycle_nodes, seen_cycles)
        if seen:
            return
        amt, trades, valid = self.process_cycle_trades(cycle_nodes, start_amount, symbol_map, price_map, fee_rate)
        ignore_cycle = self.check_ignore_cycle(cycle_nodes, ignored_symbols)
        self.add_opportunity_if_valid(valid, ignore_cycle, min_profit, max_profit, opportunities, cycle_nodes, start_amount, amt, trades)

    def increment_checked(self, checked, max_cycles, stop_flag):
        checked += 1
        if checked > max_cycles:
            stop_flag = True
        return checked, stop_flag

    def dfs_base_checks(self, current_path, max_cycle_len, stop_flag):
        if len(current_path) >= max_cycle_len:
            return True
        if stop_flag:
            return True
        return False

    def dfs_neighbor_checks(self, nb, start, current_path):
        if nb == start:
            return True
        if nb in current_path:
            return True
        return False

    def dfs(self, start, current_path, out_edges, seen_cycles, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities, checked, max_cycles, max_cycle_len, stop_flag):
        current = current_path[-1]
        if stop_flag:
            return checked, stop_flag
        if self.check_cycle_len(current_path) and self.check_start_neighbor(start, current, out_edges):
            self.process_cycle_if_new(current_path, seen_cycles, self.start_amount, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities)
            checked, stop_flag = self.increment_checked(checked, max_cycles, stop_flag)
            if stop_flag:
                return checked, stop_flag
        if self.dfs_base_checks(current_path, max_cycle_len, stop_flag):
            return checked, stop_flag
        for nb in out_edges[current]:
            if stop_flag:
                return checked, stop_flag
            if self.dfs_neighbor_checks(nb, start, current_path):
                continue
            checked, stop_flag = self.increment_checked(checked, max_cycles, stop_flag)
            if stop_flag:
                return checked, stop_flag
            current_path.append(nb)
            checked, stop_flag = self.dfs(start, current_path, out_edges, seen_cycles, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities, checked, max_cycles, max_cycle_len, stop_flag)
            current_path.pop()
        return checked, stop_flag

    def init_dfs_vars(self):
        opportunities = []
        checked = 0
        seen_cycles = set()
        stop_flag = False
        return opportunities, checked, seen_cycles, stop_flag

    def validate_max_cycle_len(self, max_cycle_len):
        if max_cycle_len < 3:
            raise ValueError("max_cycle_len must be >= 3")

    def get_assets(self, out_edges):
        return list(out_edges.keys())

    def sort_opportunities(self, opportunities):
        opportunities.sort(key=lambda x: x['profit_perc'], reverse=True)
        return opportunities

    def find_arbitrage_cycles(self, out_edges, symbol_map, price_map, fee_rate, min_profit, start_amount, max_cycles, max_cycle_len, max_profit, ignored_symbols):
        self.start_amount = start_amount
        self.validate_max_cycle_len(max_cycle_len)
        opportunities, checked, seen_cycles, stop_flag = self.init_dfs_vars()
        assets = self.get_assets(out_edges)
        for a in assets:
            if stop_flag:
                break
            if len(out_edges[a]) < 1:
                continue
            checked, stop_flag = self.dfs(a, [a], out_edges, seen_cycles, symbol_map, price_map, fee_rate, ignored_symbols, min_profit, max_profit, opportunities, checked, max_cycles, max_cycle_len, stop_flag)
        opportunities = self.sort_opportunities(opportunities)
        return opportunities