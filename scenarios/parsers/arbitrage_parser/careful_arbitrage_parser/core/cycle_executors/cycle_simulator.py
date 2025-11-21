from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.cycle_executors.cycle_price_fetcher import CyclePriceFetcher
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.cycle_executors.cycle_logger import CycleLogger


class CycleSimulator:
    def __init__(self, cycle_finder, trade_validator, logger, min_profit, max_profit, exchange_client):
        self.cycle_finder = cycle_finder
        self.trade_validator = trade_validator
        self.logger = logger
        self.min_profit = min_profit
        self.max_profit = max_profit
        self.exchange_client = exchange_client

    def simulate_cycle_loop(self, best_op, amt, symbol_map, price_map, fee, current_prices):
        cycle_amt = amt
        is_profitable = True
        simulated_trades = []
        for i in range(len(best_op['path']) - 1):
            frm = best_op['path'][i]
            to = best_op['path'][i + 1]
            new_amt, sym, direction, price = self.cycle_finder.convert_amount(cycle_amt, frm, to, symbol_map, price_map, fee)
            if new_amt is None:
                self.logger.log_message(f"Симуляция: Транзакция {frm} -> {to} невозможна из-за ограничений минимального размера или точности")
                #return None, False, None
            current_price = CyclePriceFetcher(self.exchange_client, self.logger).get_current_price(sym, direction, current_prices, price_map)
            base_min_size, quote_min_size, base_increment, quote_increment = self.trade_validator.get_cycle_constraints(sym, price_map)
            expected_new_amt = self.trade_validator.validate_sim_trade(cycle_amt, direction, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee, frm, sym, to, self.logger.log_message)
            if expected_new_amt is None:
                return None, False, None
            simulated_trades.append((frm, to, sym, direction, expected_new_amt, current_price))
            cycle_amt = expected_new_amt
        return cycle_amt, is_profitable, simulated_trades

    def check_profit_valid(self, profit_perc):
        return self.min_profit * 100 < profit_perc < self.max_profit * 100

    def log_profit_rejection(self, profit_perc, best_op):
        self.logger.log_message(f"Цикл {' -> '.join(best_op['path'])} отклонён: прогнозируемая прибыль {profit_perc:.4f}% меньше/выше диапазона или убыточна")

    def log_sim_success(self, profit_perc):
        self.logger.log_message(f"Симуляция цикла успешна: прогнозируемая прибыль {profit_perc:.4f}%")

    def simulate_cycle(self, best_op, amt, symbol_map, price_map, fee):
        current_prices = CyclePriceFetcher(self.exchange_client, self.logger).exchange_client.fetch_current_prices()
        if current_prices is None:
            return None, False, None, False
        cycle_amt, is_profitable, simulated_trades = self.simulate_cycle_loop(best_op, amt, symbol_map, price_map, fee, current_prices)
        if cycle_amt is None:
            return None, False, None, False
        profit_perc = (cycle_amt / amt - 1) * 100
        #if not is_profitable or not self.check_profit_valid(profit_perc):
            #self.log_profit_rejection(profit_perc, best_op)
            #return None, False, None, False
        self.log_sim_success(profit_perc)
        CycleLogger(self.logger).log_simulated_steps(simulated_trades, fee, amt)
        return simulated_trades, is_profitable, cycle_amt, True