# BuyerKAMA.py
from datetime import datetime
import pandas as pd
import os
from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from utils.core.functions import MarketProcess

class BuyerKAMA(MarketProcess):
    def __init__(
            self,
            history_market_parser: HistoryMarketParser,
            kama_indicator: KamaIndicator,
            kama_indicator_other: KamaIndicator,
            kama_indicator_other2: KamaIndicator,
            symbol1='BTC',
            symbol2='USDT',
            symbol1_amount: float = 0.0,
            balance_usdt: BalanceUSDT = None,
            fee: float = 0.001,
            investment_usdt: float = 1000.0  # сколько USDT инвестировать в каждую покупку (без учета fee)
    ):
        self.history_market_parser = history_market_parser
        self.kama_indicator = kama_indicator
        self.kama_indicator_other = kama_indicator
        self.kama_indicator_other2 = kama_indicator
        self.in_position = False
        self.entry_price = None
        self.entry_time = None
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.symbol1_amount = symbol1_amount
        self.balance_usdt = balance_usdt
        self.fee = fee
        self.investment_usdt = investment_usdt
        self.amount_in_position = 0.0
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

        # Обновляем индикатор KAMA для текущего времени
        self.kama_indicator.run_historical(None, current_time)

        if not self.in_position:
            if self.kama_indicator.trend == "BULLISH" and self.kama_indicator.trend2 == "BULLISH" and self.kama_indicator.trend3 == "BULLISH" and self.kama_indicator_other.trend == "BULLISH" and self.kama_indicator_other2.trend == "BULLISH":
                self._open_position(current_price, self.current_timestamp)
        else:
            if self.kama_indicator.trend == "BEARISH":
                self._close_position(current_price, self.current_timestamp, "BEARISH")

    def _open_position(self, price: float, timestamp: datetime):
        cost = self.investment_usdt
        amount_to_buy = cost / price
        fee_entry = cost * self.fee
        total_cost = cost + fee_entry

        if self.balance_usdt.amount < total_cost:
            return

        self.balance_usdt.amount -= total_cost
        self.symbol1_amount += amount_to_buy
        self.amount_in_position = amount_to_buy
        self.in_position = True
        self.entry_price = price
        self.entry_time = timestamp

        self._log(f"BUY OPEN @ {price:.2f} | amount: {amount_to_buy:.6f} {self.symbol1} | cost: {total_cost:.2f}")

        self.trades.append({
            "type": "BUY",
            "time": timestamp,
            "price": price,
            "amount": amount_to_buy,
            "fee": fee_entry
        })

    def _close_position(self, price: float, timestamp: datetime, reason: str):
        amount = self.amount_in_position
        proceeds_gross = amount * price
        fee_exit = proceeds_gross * self.fee
        proceeds_net = proceeds_gross - fee_exit
        self.balance_usdt.amount += proceeds_net
        self.symbol1_amount -= amount
        self.amount_in_position = 0.0

        # Самый честный и понятный способ посчитать net PnL
        entry_cost = self.entry_price * amount + self.entry_price * amount * self.fee
        net_pnl = proceeds_net - entry_cost

        self._log(f"CLOSE {reason} @ {price:.2f} | net PnL: {net_pnl:+.2f} USDT | "
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

    def finalize(self):
        if not self.in_position:
            return
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return
        last_row = self.history_market_parser.df.iloc[-1]
        final_price = last_row['close']
        final_timestamp = pd.to_datetime(last_row['time'])
        self._close_position(final_price, final_timestamp, "FINAL_CLOSE")