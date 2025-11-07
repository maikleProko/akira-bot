import json
from abc import abstractmethod

from scenarios.parsers.arbitrage_parser.core.cycle_executors.cycle_executor import CycleExecutor
from scenarios.parsers.arbitrage_parser.core.cycle_executors.cycle_finder import CycleFinder
from scenarios.parsers.arbitrage_parser.core.cycle_executors.graph_builder import GraphBuilder
from scenarios.parsers.arbitrage_parser.core.trade_executors.trade_executor import TradeExecutor
from scenarios.parsers.arbitrage_parser.core.trade_executors.trade_validator import TradeValidator
from scenarios.parsers.arbitrage_parser.core.utils.logger import Logger
from utils.core.functions import MarketProcess


class CarefulArbitrageParser(MarketProcess):
    def __init__(self, deposit=0.0001, production=True, api_key=None, api_secret=None, api_passphrase=None, strict=False, strict_coin='USDT', fee_rate=0.001, min_profit=0.005, use_all_balance=False, max_profit = 100.0, ignore=None, only_once = True, abusing_only_once = True):
        if ignore is None:
            ignore = []
        self.deposit = deposit
        self.production = production
        self.strict = strict
        self.strict_coin = strict_coin
        self.fee_rate = fee_rate
        self.min_profit = min_profit
        self.max_profit = max_profit
        self.use_all_balance = use_all_balance
        self.ignore = ignore
        self.consecutive_same = []
        self.possible = True
        self.ignored_symbols = frozenset([])
        self.logger = Logger()
        self.only_once = only_once
        self.exchange_client = self.create_exchange_client(self.logger)
        self.graph_builder = GraphBuilder(self.exchange_client, self.ignore, None, self.logger)
        self.cycle_finder = CycleFinder(self.logger)
        self.trade_validator = TradeValidator(self.exchange_client)
        self.trade_executor = TradeExecutor(self.exchange_client, self.trade_validator, self.logger, self.production)
        self.cycle_executor = CycleExecutor(self.cycle_finder, self.trade_executor, self.trade_validator, self.exchange_client, self.logger, self.production, self.use_all_balance, self.deposit, self.fee_rate, self.min_profit, self.max_profit, abusing_only_once)
        self.has_api = False
        self.available_coins = None
        self.init_api(api_key, api_secret, api_passphrase)
        if self.has_api:
            self.fetch_available_coins_safe()

    @abstractmethod
    def create_exchange_client(self, logger):
        pass

    def init_clients_safe(self, api_key, api_secret, api_passphrase):
        self.exchange_client.init_clients(api_key, api_secret, api_passphrase)
        self.exchange_client.test_clients()
        self.exchange_client.test_ticker()
        self.logger.log_message("Клиенты API успешно инициализированы")
        self.has_api = True

    def init_api(self, api_key, api_secret, api_passphrase):
        if api_key and api_secret:
            try:
                self.init_clients_safe(api_key, api_secret, api_passphrase)
            except Exception as e:
                self.logger.log_message(f"Не удалось инициализировать клиенты API: {str(e)}")
                self.production = False
        else:
            self.logger.log_message("API keys not provided, running in simulation mode without API")
            self.production = False

    def fetch_available_coins_safe(self):
        try:
            self.available_coins = self.exchange_client.fetch_available_coins()
            self.graph_builder.available_coins = self.available_coins
        except Exception as e:
            self.logger.log_message(f"Failed to fetch available coins: {str(e)}")
            self.available_coins = None

    def build_graph_and_prices(self):
        return self.graph_builder.build_graph_and_prices()

    def find_arbitrage_cycles(self, fee_rate=0.001, min_profit=0.0001, start_amount=1.0, max_cycles=2000000, max_cycle_len=4, max_profit=100.0):
        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        return self.cycle_finder.find_arbitrage_cycles(out_edges, symbol_map, price_map, fee_rate or self.fee_rate, min_profit or self.min_profit, start_amount, max_cycles, max_cycle_len, max_profit or self.max_profit, self.ignored_symbols)

    def check_pair_available(self, sym, direction, price_map):
        return self.exchange_client.check_pair_available(sym, direction, price_map)

    def filter_strict_ops(self, valid_ops):
        return [op for op in valid_ops if op['start_asset'] == self.strict_coin]

    def get_best_op(self, valid_ops):
        if valid_ops:
            if self.strict:
                valid_ops = self.filter_strict_ops(valid_ops)
            return max(valid_ops, key=lambda x: x['profit_perc'])
        return None

    def check_restricted(self, op, price_map):
        for t in op['trades']:
            sym = t[2]
            if not self.check_pair_available(sym, t[3], price_map):
                return True
        return False

    def check_op_warm(self, op, price_map):
        restricted = self.check_restricted(op, price_map)
        if not restricted:
            return op
        else:
            return None

    def get_norm_path(self, best_op):
        if best_op:
            return self.cycle_finder.normalize_cycle(best_op['path'][:-1])
        return None

    def get_current_norms(self, ops):
        current_norms = set()
        for op in ops:
            norm = self.cycle_finder.normalize_cycle(op['path'][:-1])
            current_norms.add(norm)
        return current_norms

    def filter_consecutive(self, current_norms):
        self.consecutive_same = [d for d in self.consecutive_same if d['path'] in current_norms]

    def update_existing_cons(self, norm):
        for d in self.consecutive_same:
            if d['path'] == norm:
                d['cons_value'] += 1
                return True
        return False

    def add_new_cons(self, norm):
        self.consecutive_same.append({'path': norm, 'cons_value': 1})

    def update_consecutive(self, ops):
        current_norms = self.get_current_norms(ops)
        self.filter_consecutive(current_norms)
        for norm in current_norms:
            existing = self.update_existing_cons(norm)
            if not existing:
                self.add_new_cons(norm)

    def get_cons_for_op(self, op):
        norm = self.cycle_finder.normalize_cycle(op['path'][:-1])
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

    def print_op_path(self, o):
        print("Путь:", " -> ".join(o['path']), f"Начало {o['start_amount']}{o['start_asset']} -> Конец {o['end_amount']:.8f}{o['start_asset']}", f"Прибыль {o['profit_perc']*100:.4f}%")

    def print_calc_header(self):
        print("Calculation steps:")

    def print_trade_steps(self, o, fee_rate):
        amt = o['start_amount']
        for t in o['trades']:
            line1, line2 = self.cycle_executor.get_trade_strings(amt, t, fee_rate)
            print(line1)
            print(line2)
            amt = t[4]
        print("----")

    def fill_local_balances(self, best_op):
        self.exchange_client.local_balances = {}
        if best_op and best_op['trades']:
            for t in best_op['trades']:
                self.exchange_client.local_balances[t[0]] = self.exchange_client.check_balance(t[0])

    def print_op(self, o, fee_rate):
        if o is None:
            return
        print(o)
        self.print_op_path(o)
        self.print_calc_header()
        self.print_trade_steps(o, fee_rate)

    def print_ops(self, ops):
        for o in ops[:50]:
            self.print_op(o, self.fee_rate)

    def filter_valid_ops(self, ops):
        return [o for o in ops if o['start_asset'] == self.strict_coin and o['path'][0] == self.strict_coin and o['path'][-1] == self.strict_coin]

    def find_valid_ops(self, ops, strict):
        if strict:
            return self.filter_valid_ops(ops)
        return ops

    def save_to_file(self, array, file_name):
        out_path = f'files/{file_name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(array, f, ensure_ascii=False, indent=2)
        self.logger.log_message(f"Результаты сохранены в {out_path}")

    def init_trade_vars(self, best_op):
        initial_deposit = self.deposit
        amt = self.deposit
        trades = []
        valid = True
        return initial_deposit, amt, trades, valid

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
            self.logger.log_message("Арбитражных возможностей не найдено (по заданному порогу).")
        else:
            pass

    def prepare_trade(self, selected_op, out_edges, symbol_map, price_map, start_asset):
        if self.only_once:
            self.possible = False
        self.logger.log_message("Входим в режим активного трейдинга")
        initial_deposit, amt, trades, valid = self.init_trade_vars(selected_op)
        amt, valid = self.cycle_executor.adjust_start_balance(start_asset)
        if not valid:
            return'''
        self.logger.log_message(f"Проверка прибыльности цикла: {' -> '.join(selected_op['path'])}")
        simulated_trades, is_profitable, cycle_amt, valid = self.cycle_executor.simulate_cycle(selected_op, amt, symbol_map, price_map, self.fee_rate)
        if not valid:
            return'''
        return initial_deposit, amt, trades, valid, True

    def execute_and_save_trade(self, selected_op, initial_deposit, amt, trades, valid, is_profitable, symbol_map, price_map):
        amt, trades, valid, is_profitable = self.cycle_executor.execute_cycle(selected_op, initial_deposit, amt, trades, valid, symbol_map, price_map, self.fee_rate)
        if valid:
            self.deposit = self.cycle_executor.save_trade_results(selected_op, amt, initial_deposit, trades, is_profitable, self.fee_rate)
        else:
            self.logger.log_message("Арбитраж не выполнен из-за ошибок в транзакциях или убыточности цикла")

    def reset_consecutive(self, norm):
        self.consecutive_same = [d for d in self.consecutive_same if d['path'] != norm]

    def run_realtime_trade(self, selected_op):
        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        start_asset = selected_op['start_asset']
        initial_deposit, amt, trades, valid, is_profitable = self.prepare_trade(selected_op, out_edges, symbol_map, price_map, start_asset)
        if not valid:
            return
        self.execute_and_save_trade(selected_op, initial_deposit, amt, trades, valid, is_profitable, symbol_map, price_map)
        norm = self.get_norm_path(selected_op)
        self.reset_consecutive(norm)

    def log_best_op(self, best_op):
        if best_op is not None:
            print('best op: ')
            self.trade_validator.best_op = best_op
            self.print_op(best_op, self.fee_rate)
            self.fill_local_balances(best_op)
        else:
            self.logger.log_message("Лучшая возможность арбитража не найдена.")

    def update_best_op(self, best_op):
        if best_op is not None:
            print('best op: ')
            self.trade_validator.best_op = best_op
            self.fill_local_balances(best_op)

    def check_consecutive_threshold(self, consecutive_same):
        return consecutive_same > 0

    def run_realtime(self):
        ops, valid_ops, best_op = self.run_realtime_init(self.strict)
        self.update_best_op(best_op)
        self.run_realtime_print(ops)
        consecutive_same = self.get_consecutive_for_best(best_op)
        #self.logger.log_message(f"CHECKING: consecutive_same = {consecutive_same}")
        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        if self.check_consecutive_threshold(consecutive_same) and self.possible:
            best_op = self.check_op_warm(best_op, price_map)
            if best_op:
                self.run_realtime_trade(best_op)