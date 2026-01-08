# BuyerTPSL.py
from datetime import datetime
import pandas as pd
import os

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

        log_date = datetime.now().strftime("%Y%m%d")
        log_dir = "files/decisions"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"decisions_{log_date}.txt")
        self._log(f"\n=== НОВАЯ СЕССИЯ | {symbol1}/{symbol2} | Баланс: {balance_usdt.amount:.2f} USDT ===\n")

    def _log(self, message: str):
        entry = f"[{self.current_timestamp}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        print(entry.strip())

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
                self._open_position(current_price, self.current_timestamp)
        else:
            self._check_exit_conditions(last_row, current_price, self.current_timestamp)

    def _open_position(self, price: float, timestamp: datetime):
        amount_to_buy = self.regulator_tpsl.symbol1_prepared_converted_amount
        cost = amount_to_buy * price
        fee_cost = cost * self.fee

        if self.balance_usdt.amount < (cost):
            self.regulator_tpsl.is_accepted_by_regulator = False
            return

        self.balance_usdt.amount -= (cost)
        self.symbol1_amount += amount_to_buy

        self.in_position = True
        self.entry_price = price
        self.entry_time = timestamp

        self._log(f"BUY OPEN @ {price:.2f} | amount: {amount_to_buy:.6f} {self.symbol1} | cost: {cost} | "
                  f"TP: {self.regulator_tpsl.take_profit:.2f} | SL: {self.regulator_tpsl.stop_loss:.2f}")

        self.trades.append({
            "type": "BUY",
            "time": timestamp,
            "price": price,
            "amount": amount_to_buy,
            "fee": fee_cost
        })

    def _check_exit_conditions(self, last_row, current_price: float, timestamp: datetime):
        tp = self.regulator_tpsl.take_profit
        sl = self.regulator_tpsl.stop_loss

        if last_row['low'] <= sl:
            self._close_position(sl, timestamp, "SL")
            return

        if last_row['high'] >= tp:
            self._close_position(tp, timestamp, "TP")

    def _close_position(self, price: float, timestamp: datetime, reason: str):
        amount_to_sell = self.regulator_tpsl.symbol1_prepared_converted_amount
        self.regulator_tpsl.is_accepted_by_regulator = False

        proceeds = amount_to_sell * price
        fee_cost = proceeds * self.fee

        self.balance_usdt.amount += (proceeds - fee_cost)
        self.symbol1_amount -= amount_to_sell

        # Правильный чистый PnL
        entry_cost_with_fee = (self.entry_price * amount_to_sell) + (self.entry_price * amount_to_sell * self.fee)
        exit_proceeds_net = proceeds - fee_cost
        net_pnl = exit_proceeds_net - entry_cost_with_fee

        self._log(f"CLOSE {reason} @ {price:.2f} | net PnL: {net_pnl:+.2f} USDT | "
                  f"balance: {self.balance_usdt.amount:.2f} USDT | {self.symbol1}: {self.symbol1_amount:.6f}")

        self.trades.append({
            "type": "SELL",
            "reason": reason,
            "time": timestamp,
            "price": price,
            "amount": amount_to_sell,
            "pnl": net_pnl,
            "fee": fee_cost
        })

        self.in_position = False
        self.entry_price = None
        self.entry_time = None

    def finalize(self):
        if not self.in_position:
            return

        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return

        last_row = self.history_market_parser.df.iloc[-1]
        final_price = last_row['close']
        final_timestamp = pd.to_datetime(last_row['time'])

        self._close_position(final_price, final_timestamp, "FINAL_CLOSE")