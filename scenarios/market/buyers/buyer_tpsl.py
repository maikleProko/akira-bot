from datetime import datetime
import pandas as pd
import os

from triton.language import condition

from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from utils.core.functions import MarketProcess


class BuyerTPSL(MarketProcess):
    def __init__(
            self,
            history_market_parser: HistoryMarketParser,
            regulator_tpsl: RegulatorTPSL,
            symbol1='BTC',
            symbol2='USDT',
            symbol1_amount: float = 0.0,
            balance_usdt: BalanceUSDT = None,
            fee: float = 0.001,
            is_take_profit_for_close = False
    ):
        self.history_market_parser = history_market_parser
        self.regulator_tpsl = regulator_tpsl
        self.in_position = False
        self.entry_price = None
        self.entry_time = None
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.symbol1_amount = symbol1_amount
        self.balance_usdt = balance_usdt
        self.fee = fee
        self.trades = []
        self.current_timestamp = ""
        self.saved_timestamp = ""
        self.is_realtime_triggered = 0
        self.realtime = False
        self.is_take_profit_for_close = is_take_profit_for_close
        log_date = datetime.now().strftime("%Y%m%d")
        log_dir = "files/decisions"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"decisions_{log_date}.txt")
        self._log(f"\n=== НОВАЯ СЕССИЯ | {symbol1}/{symbol2} | Баланс: {balance_usdt.amount:.2f} USDT ===\n")

    def prepocess_realtime(self):
        self.realtime = True

    def _log(self, message: str):
        entry = f"[{self.current_timestamp}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        print(entry.strip())

    def _fix_deal(self, reason):
        with open(f'files/decisions/deals.txt', "a", encoding="utf-8") as f:
            f.write("{")
            f.write(f"\"date\": \"{self.entry_time}\", \"result\": \"{reason}\"")
            f.write("},\n")

    def prepare(self, start_time: datetime = None, end_time: datetime = None):
        pass

    def run_realtime(self):
        self._tick(datetime.now())

    def run_historical(self, start_time: datetime, current_time: datetime):
        self._tick(current_time)

    def _tick(self, current_time: datetime):
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return
        last_row = self.history_market_parser.df.iloc[-1]
        current_price = last_row['close']
        self.current_timestamp = pd.to_datetime(last_row['time'])
        if not self.in_position:
            if self.regulator_tpsl.is_accepted_by_regulator:
                self._signal_open_position(current_price, self.current_timestamp)
        else:
            self._check_exit_conditions(last_row, current_price, self.current_timestamp)

    def _signal_open_position(self, price: float, timestamp: datetime):
        amount_to_buy = self.regulator_tpsl.symbol1_prepared_converted_amount
        cost = amount_to_buy * price
        fee_entry = cost * self.fee
        total_cost = cost + fee_entry
        if self.balance_usdt.amount < total_cost:
            self.regulator_tpsl.is_accepted_by_regulator = False
            return
        self._log(
            f"SIGNAL BUY OPEN @ ~{price:.2f} | amount: {amount_to_buy:.6f} {self.symbol1} | cost: ~{total_cost:.2f} | "
            f"TP: {self.regulator_tpsl.take_profit:.2f} | SL: {self.regulator_tpsl.stop_loss:.2f}")
        if not self.realtime:
            self.balance_usdt.amount -= total_cost
            self.symbol1_amount += amount_to_buy
            self.entry_price = price
            self.entry_time = timestamp
            self.trades.append({
                "type": "BUY",
                "time": timestamp,
                "price": price,
                "amount": amount_to_buy,
                "fee": fee_entry
            })
        self.in_position = True
        self.entry_price = price  # Temporary for realtime
        self.entry_time = timestamp
        self.is_realtime_triggered = 1

    def _check_exit_conditions(self, last_row, current_price: float, timestamp: datetime):
        tp = self.regulator_tpsl.take_profit
        sl = self.regulator_tpsl.stop_loss
        if last_row['low'] <= sl:
            self._signal_close_position(sl, timestamp, "SL")
            return

        if self.is_take_profit_for_close:
            condition = (current_price >= tp)
            bought_price = current_price
        else:
            condition = (last_row['high'] >= tp)
            bought_price = tp

        if condition:
            if self.realtime:
                if self.is_realtime_triggered == 1:
                    self.saved_timestamp = self.current_timestamp

                self.is_realtime_triggered = 2

                if self.saved_timestamp != self.current_timestamp and self.is_realtime_triggered == 2:
                    self.is_realtime_triggered = 3

                if self.is_realtime_triggered > 2:
                    self._signal_close_position(bought_price, timestamp, "TP")
            else:
                self._signal_close_position(bought_price, timestamp, "TP")

    def _signal_close_position(self, price: float, timestamp: datetime, reason: str):
        amount = self.symbol1_amount  # Use current holding
        self.regulator_tpsl.is_accepted_by_regulator = False
        self._log(f"SIGNAL CLOSE {reason} @ ~{price:.2f} | amount: {amount:.6f} {self.symbol1}")
        if not self.realtime:
            proceeds_gross = amount * price
            fee_exit = proceeds_gross * self.fee
            proceeds_net = proceeds_gross - fee_exit
            self.balance_usdt.amount += proceeds_net
            self.symbol1_amount -= amount
            entry_cost = self.entry_price * amount + self.entry_price * amount * self.fee
            net_pnl = proceeds_net - entry_cost
            self._log(f"SIM CLOSE {reason} @ {price:.2f} | net PnL: {net_pnl:+.2f} USDT | "
                      f"balance: {self.balance_usdt.amount:.2f} USDT | {self.symbol1}: {self.symbol1_amount:.6f}")
            self._fix_deal(reason)
            self.trades.append({
                "type": "SELL",
                "reason": reason,
                "time": timestamp,
                "price": price,
                "amount": amount,
                "pnl": net_pnl,
                "fee": fee_exit
            })
        self.in_position = False
        self.entry_price = None
        self.entry_time = None
        self.is_realtime_triggered = 0

    def finalize(self):
        if not self.in_position:
            return
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return
        last_row = self.history_market_parser.df.iloc[-1]
        final_price = last_row['close']
        final_timestamp = pd.to_datetime(last_row['time'])
        self._signal_close_position(final_price, final_timestamp, "FINAL_CLOSE")

    # New public methods for broker to update actuals
    def update_actual_open(self, actual_price: float, actual_amount: float, actual_fee: float, timestamp: datetime):
        self.entry_price = actual_price
        self.entry_time = timestamp
        self.symbol1_amount = actual_amount
        cost = actual_amount * actual_price + actual_fee
        self.balance_usdt.amount -= cost  # Update if not already synced
        self._log(f"ACTUAL BUY OPEN @ {actual_price:.2f} | amount: {actual_amount:.6f} | fee: {actual_fee:.2f}")
        self.trades.append({
            "type": "BUY",
            "time": timestamp,
            "price": actual_price,
            "amount": actual_amount,
            "fee": actual_fee
        })

    def update_actual_close(self, actual_price: float, actual_amount: float, actual_fee: float, reason: str,
                            timestamp: datetime):
        proceeds_net = actual_amount * actual_price - actual_fee
        entry_cost = self.entry_price * actual_amount + (
                    self.entry_price * actual_amount * self.fee)  # Approx entry fee
        net_pnl = proceeds_net - entry_cost
        self.balance_usdt.amount += proceeds_net
        self.symbol1_amount -= actual_amount
        self._log(
            f"ACTUAL CLOSE {reason} @ {actual_price:.2f} | net PnL: {net_pnl:+.2f} USDT | fee: {actual_fee:.2f} | "
            f"balance: {self.balance_usdt.amount:.2f} USDT | {self.symbol1}: {self.symbol1_amount:.6f}")
        self._fix_deal(reason)
        self.trades.append({
            "type": "SELL",
            "reason": reason,
            "time": timestamp,
            "price": actual_price,
            "amount": actual_amount,
            "pnl": net_pnl,
            "fee": actual_fee
        })

    def sync_balances(self, usdt_amount: float, symbol1_amount: float):
        self.balance_usdt.amount = usdt_amount
        self.symbol1_amount = symbol1_amount
        # Update in_position based on real holdings
        if self.symbol1_amount > 0.000001:
            if not self.in_position:
                self.in_position = True
                self._log("SYNC: Detected open position on exchange.")
        else:
            if self.in_position:
                self.in_position = False
                self.entry_price = None
                self.entry_time = None
                self.is_realtime_triggered = 0
                self._log("SYNC: Detected closed position on exchange.")

