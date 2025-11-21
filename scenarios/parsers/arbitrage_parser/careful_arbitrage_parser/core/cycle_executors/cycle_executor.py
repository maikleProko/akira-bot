import asyncio
import time
import aiohttp
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.cycle_executors.cycle_price_fetcher import CyclePriceFetcher
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.cycle_executors.cycle_simulator import CycleSimulator
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.cycle_executors.cycle_start_balance_adjuster import \
    CycleStartBalanceAdjuster
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.cycle_executors.cycle_trade_result_saver import CycleTradeResultSaver
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.cycle_executors.cycle_logger import CycleLogger


class CycleExecutor:
    def __init__(self, cycle_finder, trade_executor, trade_validator, exchange_client, logger, production, use_all_balance, deposit, fee_rate, min_profit, max_profit, only_once):
        self.cycle_finder = cycle_finder
        self.trade_executor = trade_executor
        self.trade_validator = trade_validator
        self.exchange_client = exchange_client
        self.logger = logger
        self.production = production
        self.use_all_balance = use_all_balance
        self.deposit = deposit
        self.fee_rate = fee_rate
        self.min_profit = min_profit
        self.max_profit = max_profit
        self.only_once = only_once
        self.balance_adjuster = CycleStartBalanceAdjuster(exchange_client, logger, production, use_all_balance, deposit)
        self.simulator = CycleSimulator(cycle_finder, trade_validator, logger, min_profit, max_profit, exchange_client)
        self.cycle_logger = CycleLogger(logger)
        self.result_saver = CycleTradeResultSaver(logger)

    def adjust_start_balance(self, start_asset):
        return self.balance_adjuster.adjust_start_balance(start_asset)

    def get_current_price(self, sym, direction, current_prices, price_map):
        return CyclePriceFetcher(self.exchange_client, self.logger).get_current_price(sym, direction, current_prices, price_map)

    def simulate_cycle(self, best_op, amt, symbol_map, price_map, fee):
        return self.simulator.simulate_cycle(best_op, amt, symbol_map, price_map, fee)

    def log_path_and_profit(self, best_op):
        self.cycle_logger.log_path_and_profit(best_op)

    def log_calc_steps(self, best_op, fee):
        self.cycle_logger.log_calc_steps(best_op, fee)

    def get_trade_strings(self, amt, t, fee_rate):
        return self.cycle_logger.get_trade_strings(amt, t, fee_rate)

    def process_cycle_trade(self, amt, frm, to, symbol_map, price_map, fee):
        return self.cycle_finder.convert_amount(amt, frm, to, symbol_map, price_map, fee)

    def fetch_ticker_and_price(self, sym, direction):
        current_ticker = self.exchange_client.fetch_ticker_price(sym)
        #self.logger.log_message(f"{sym}: {current_ticker}")
        if current_ticker is None or 'buy' not in current_ticker or 'sell' not in current_ticker:
            self.logger.log_message(f"Ошибка: Недопустимый ответ API для {sym}, пропускаем транзакцию")
            return None, None
        current_price = float(current_ticker['sell']) if direction == 'sell' else float(current_ticker['buy'])
        return current_ticker, current_price

    def calculate_expected_new_amt(self, amt, current_price, fee, direction):
        if direction == 'sell':
            return amt * current_price * (1 - fee)
        return (amt / current_price) * (1 - fee)

    def adjust_for_insufficient_balance(self, frm, amt, fee, symbol_map, price_map, to):
        available = self.exchange_client.check_balance(frm)
        if available < amt:
            #self.logger.log_message(f"Недостаточно {frm} на балансе: доступно {available:.8f}, требуется {amt:.8f}. Используем весь доступный баланс.")
            amt = available
            if amt == 0:
                self.logger.log_message(f"Баланс {frm} равен нулю, транзакция невозможна.")
                return None, False
            new_amt, sym, direction, price = self.process_cycle_trade(amt, frm, to, symbol_map, price_map, fee)
            if new_amt is None:
                self.logger.log_message(f"Транзакция {frm} -> {to} невозможна с доступным балансом {amt:.8f} из-за ограничений минимального размера или точности")
                return None, False
            return amt, True
        return amt, True

    def execute_trade_step(self, frm, to, amt, sym, direction, price, fee, price_map, trades, symbol_map):

        if self.production:
            amt, success = self.adjust_for_insufficient_balance(frm, amt, fee, symbol_map, price_map, to)
            if not success:
                return None, False
            actual_amount, success, current_price = self.trade_executor.execute_trade(frm, to, amt, sym, direction, price, fee, price_map)
            if not success:
                return None, False
            new_amt = actual_amount
        else:
            _, current_price = self.fetch_ticker_and_price(sym, direction)
            new_amt = self.calculate_expected_new_amt(amt, current_price, fee, direction)
        trades.append((frm, to, sym, direction, new_amt, current_price))
        return new_amt, True

    def execute_cycle_loop(self, best_op, amt, trades, valid, symbol_map, price_map, fee):
        #self.log_path_and_profit(best_op)
        #self.log_calc_steps(best_op, fee)
        is_profitable = True
        for i in range(len(best_op['path']) - 1):
            frm = best_op['path'][i]
            to = best_op['path'][i + 1]
            new_amt, sym, direction, price = self.process_cycle_trade(amt, frm, to, symbol_map, price_map, fee)
            if new_amt is None:
                self.logger.log_message(f"Транзакция {frm} -> {to} невозможна из-за ограничений минимального размера или точности")
                valid = False
                break
            _, current_price = self.fetch_ticker_and_price(sym, direction)
            if current_price is None:
                valid = False
                break
            amt, valid = self.execute_trade_step(frm, to, amt, sym, direction, price, fee, price_map, trades, symbol_map)
            if not valid:
                break
            time.sleep(0.8)
        return amt, trades, valid, is_profitable

    def execute_cycle(self, best_op, initial_deposit, amt, trades, valid, symbol_map, price_map, fee):
        self.deposit = initial_deposit
        amt, trades, valid, is_profitable = self.abuse(best_op, amt, trades, valid, symbol_map, price_map, fee)
        return amt, trades, valid, is_profitable

    ###

    def save_trade_results(self, best_op, amt, initial_deposit, trades, is_profitable, fee):
        return self.result_saver.save_trade_results(best_op, amt, initial_deposit, trades, is_profitable, fee)



    async def async_all_ticker_and_price(self, op):
        symbols = [trade[2] for trade in op['trades']]
        async with aiohttp.ClientSession() as session:
            tasks = [self.exchange_client.fetch_ticker_async(session, sym) for sym in symbols]
            results = await asyncio.gather(*tasks)
        return [res for res in results if res is not None]


    def fast_estimate(self, op_info, op):
        op_info_dict = {d['name']: d for d in op_info}
        amt = op['start_amount']
        for trade in op['trades']:
            sym = trade[2]
            direction = trade[3]
            prices = op_info_dict.get(sym)
            if prices is None:
                return 0.0
            price = prices['sell'] if direction == 'sell' else prices['buy']
            if direction == 'sell':
                amt = amt * price * (1 - self.fee_rate)
            else:
                amt = (amt / price) * (1 - self.fee_rate)
        profit_perc = (amt / op['start_amount'] - 1) * 100
        return profit_perc

    def abuse(self, best_op, amt, trades, valid, symbol_map, price_map, fee):
        start_time = time.time()
        is_profitable = None
        print('ABUZING MODE BEGIN')
        while time.time() - start_time < 300:
            prices = asyncio.run(self.async_all_ticker_and_price(best_op))
            estimate = self.fast_estimate(prices, best_op)
            if estimate > self.min_profit:
                amt, trades, valid, is_profitable = self.execute_cycle_loop(best_op, amt, trades, valid, symbol_map, price_map, fee)

                if valid:
                    self.deposit = self.save_trade_results(best_op, amt, self.deposit, trades, is_profitable, self.fee_rate)
                else:
                    self.logger.log_message("Арбитраж не выполнен из-за ошибок в транзакциях или убыточности цикла")

                if self.only_once:
                    break
            else:
                time.sleep(0.5)
        print('ABUZING MODE END')
        return amt, trades, valid, is_profitable



