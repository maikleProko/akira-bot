
class CycleLogger:
    def __init__(self, logger):
        self.logger = logger

    def get_sell_strings(self, step_amt, t, fee_rate):
        frm, to, sym, dirc, new_amt, price = t
        line1 = f" Продажа {step_amt:.8f} {frm} для {to} с использованием пары {sym} по bid цене {price:.8f} ({to} per {frm})"
        line2 = f" Получено: {step_amt:.8f} * {price:.8f} * (1 - {fee_rate}) = {new_amt:.8f} {to}"
        return line1, line2

    def get_buy_strings(self, step_amt, t, fee_rate):
        frm, to, sym, dirc, new_amt, price = t
        line1 = f" Покупка {new_amt:.8f} {to} с {step_amt:.8f} {frm} с использованием пары {sym} по ask цене {price:.8f} ({frm} per {to})"
        line2 = f" Получено: ({step_amt:.8f} / {price:.8f}) * (1 - {fee_rate}) = {new_amt:.8f} {to}"
        return line1, line2

    def get_trade_strings(self, step_amt, t, fee_rate):
        dirc = t[3]
        if dirc == 'sell':
            return self.get_sell_strings(step_amt, t, fee_rate)
        else:
            return self.get_buy_strings(step_amt, t, fee_rate)

    def log_simulated_steps(self, simulated_trades, fee, initial_deposit):
        self.logger.log_message("Симулированные шаги:")
        step_amt = initial_deposit
        for t in simulated_trades:
            line1, line2 = self.get_trade_strings(step_amt, t, fee)
            self.logger.log_message(line1)
            self.logger.log_message(line2)
            step_amt = t[4]
        self.logger.log_message("----")

    def log_path_and_profit(self, best_op):
        self.logger.log_message(f"Путь: {' -> '.join(best_op['path'])}")
        self.logger.log_message(f"Начало {best_op['start_amount']}{best_op['start_asset']} -> Конец {best_op['end_amount']:.8f}{best_op['start_asset']}")
        self.logger.log_message(f"Прибыль {best_op['profit_perc'] * 100:.4f}%")

    def log_calc_steps(self, best_op, fee):
        self.logger.log_message("Calculation steps:")
        step_amt = best_op['start_amount']
        for t in best_op['trades']:
            line1, line2 = self.get_trade_strings(step_amt, t, fee)
            self.logger.log_message(line1)
            self.logger.log_message(line2)
            step_amt = t[4]
        self.logger.log_message("----")
