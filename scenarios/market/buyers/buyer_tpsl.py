from datetime import datetime
import pandas as pd

from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from utils.core.functions import MarketProcess


class BuyerTPSL(MarketProcess):
    """
    Класс, который реализует мониторинг позиции BUY с TP/SL.
    Входит в позицию при regulator.is_accepted_by_regulator == True,
    затем отслеживает достижение take-profit или stop-loss.
    При завершении исторического теста — принудительно закрывает позицию.
    """
    def __init__(
            self,
            history_market_parser: HistoryMarketParser,
            regulator_tpsl: RegulatorTPSL,
            symbol1='BTC',
            symbol2='USDT',
            symbol1_amount: float = 0.0,
            balance_usdt: BalanceUSDT = None,
            fee: float = 0.001,  # 0.1%
    ):
        self.history_market_parser = history_market_parser
        self.regulator_tpsl = regulator_tpsl

        # Состояние позиции
        self.in_position = False
        self.entry_price = None
        self.entry_time = None

        # Балансы и параметры
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.symbol1_amount = symbol1_amount
        self.balance_usdt = balance_usdt
        self.fee = fee

        # Статистика сделок
        self.trades = []

    def prepare(self, start_time: datetime = None, end_time: datetime = None):
        """Подготовка — ничего особенного"""
        pass

    def run_realtime(self):
        self._tick(datetime.now())

    def run_historical(self, start_time: datetime, current_time: datetime):
        self._tick(current_time)

    def _tick(self, current_time: datetime):
        """Основная логика на каждом тике"""
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return

        last_row = self.history_market_parser.df.iloc[-1]
        current_price = last_row['close']
        current_timestamp = pd.to_datetime(last_row['time'])

        if not self.in_position:
            if self.regulator_tpsl.is_accepted_by_regulator:
                self._open_position(current_price, current_timestamp)
        else:
            self._check_exit_conditions(last_row, current_price, current_timestamp)

    def _open_position(self, price: float, timestamp: datetime):
        amount_to_buy = self.regulator_tpsl.symbol1_prepared_converted_amount
        cost = amount_to_buy * price
        fee_cost = cost * self.fee

        if self.balance_usdt.amount < (cost + fee_cost):
            # print(f"[{timestamp}] Недостаточно USDT для входа")
            return

        self.balance_usdt.amount -= (cost + fee_cost)
        self.symbol1_amount += amount_to_buy

        self.in_position = True
        self.entry_price = price
        self.entry_time = timestamp

        print(f"[{timestamp}] BUY OPEN @ {price:.2f} | "
              f"amount: {amount_to_buy:.6f} {self.symbol1} | "
              f"TP: {self.regulator_tpsl.take_profit:.2f} | "
              f"SL: {self.regulator_tpsl.stop_loss:.2f}")

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

        # Сначала SL по low (логика 2)
        if last_row['low'] <= sl:
            self._close_position(sl, timestamp, "SL")
            return

        # Затем TP по close (логика 1)
        if current_price >= tp:
            self._close_position(current_price, timestamp, "TP")

    def _close_position(self, price: float, timestamp: datetime, reason: str):
        """Закрытие позиции по указанной цене и причине"""
        amount_to_sell = self.regulator_tpsl.symbol1_prepared_converted_amount
        self.regulator_tpsl.is_accepted_by_regulator = False

        proceeds = amount_to_sell * price
        fee_cost = proceeds * self.fee

        # Исправлено: вычитаем комиссию при продаже
        self.balance_usdt.amount += (proceeds - fee_cost)
        self.symbol1_amount -= amount_to_sell

        total_fees = 0  # можно посчитать из входа + выхода
        if self.trades:
            entry_trade = [t for t in self.trades if t["type"] == "BUY"][-1]
            total_fees = entry_trade.get("fee", 0) + fee_cost

        pnl = (price - self.entry_price) * amount_to_sell - total_fees

        print(f"[{timestamp}] CLOSE {reason} @ {price:.2f} | "
              f"pnl: {pnl:+.2f} USDT | "
              f"balance: {self.balance_usdt.amount:.2f} USDT | "
              f"{self.symbol1}: {self.symbol1_amount:.6f}")

        self.trades.append({
            "type": "SELL",
            "reason": reason,
            "time": timestamp,
            "price": price,
            "amount": amount_to_sell,
            "pnl": pnl,
            "fee": fee_cost
        })

        # Сброс состояния
        self.in_position = False
        self.entry_price = None
        self.entry_time = None

    def finalize(self):
        """
        Вызывается в конце исторического бэктекста.
        Если позиция осталась открытой — принудительно закрываем по последней цене close.
        Это гарантирует корректный подсчёт итогового баланса и PnL.
        """
        if not self.in_position:
            return  # Нечего закрывать

        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            print("[finalize_position] Нет данных для закрытия позиции!")
            return

        last_row = self.history_market_parser.df.iloc[-1]
        final_price = last_row['close']
        final_timestamp = pd.to_datetime(last_row['time'])

        print(f"\n[END OF BACKTEST] Принудительное закрытие открытой позиции по рыночной цене")
        self._close_position(final_price, final_timestamp, "FINAL_CLOSE")

        print(f"[BACKTEST FINISHED] Итоговый баланс: {self.balance_usdt.amount:.2f} USDT {self.symbol1}: {self.symbol1_amount:.6f}")