from datetime import datetime


class CycleTradeResultSaver:
    def __init__(self, logger):
        self.logger = logger

    def get_current_time(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def write_header(self, f, current_time, best_op, profit_perc, is_profitable):
        f.write(f"Time: {current_time}\n")
        f.write(f"Path: {' -> '.join(best_op['path'])}\n")
        f.write(f"Profit: {profit_perc:.4f}%{' (убыточный цикл)' if not is_profitable else ''}\n")
        f.write(f"Start Amount: {best_op['start_amount']}{best_op['start_asset']}\n")
        f.write(f"End Amount: {best_op['end_amount']:.8f}{best_op['start_asset']}\n")

    def write_steps_header(self, f):
        f.write("Calculation steps:\n")

    def write_trade_step_sell(self, f, step_amt, t, fee):
        frm, to, sym, _, new_amt, price = t
        f.write(f" Продажа {step_amt:.8f} {frm} для {to} с использованием пары {sym} по bid цене {price:.8f} ({to} per {frm})")
        f.write(f" Получено от покупки: {step_amt:.8f} * {price:.8f} * (1 - {fee}) = {new_amt:.8f} {to}")

    def write_trade_step_buy(self, f, step_amt, t, fee):
        frm, to, sym, _, new_amt, price = t
        f.write(f" Покупка {new_amt:.8f} {to} с {step_amt:.8f} {frm} с использованием пары {sym} по ask цене {price:.8f} ({frm} per {to})")
        f.write(f" Получено: ({step_amt:.8f} / {price:.8f}) * (1 - {fee}) = {new_amt:.8f} {to}")

    def write_trade_step(self, step_amt, t, fee, f):
        dirc = t[3]
        if dirc == 'sell':
            self.write_trade_step_sell(f, step_amt, t, fee)
        else:
            self.write_trade_step_buy(f, step_amt, t, fee)

    def write_trade_results(self, best_op, amt, initial_deposit, trades, is_profitable, fee):
        current_time = self.get_current_time()
        with open('files/decisions/final_decisions.txt', 'a', encoding='utf-8') as f:
            profit_perc = (amt / initial_deposit - 1) * 100
            self.write_header(f, current_time, best_op, profit_perc, is_profitable)
            self.write_steps_header(f)
            step_amt = initial_deposit
            for t in trades:
                self.write_trade_step(step_amt, t, fee, f)
                step_amt = t[4]
            f.write("----\n")
        self.logger.log_message(f"Арбитраж завершен: начальная сумма {initial_deposit:.8f} {best_op['start_asset']}, конечная сумма {amt:.8f} {best_op['start_asset']}{' (убыточный цикл)' if not is_profitable else ''}")

    def save_trade_results(self, best_op, amt, initial_deposit, trades, is_profitable, fee):
        self.write_trade_results(best_op, amt, initial_deposit, trades, is_profitable, fee)
