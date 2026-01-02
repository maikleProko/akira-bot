import os
from datetime import datetime
import pandas as pd

from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from utils.core.functions import MarketProcess

class BuyerTPSL(MarketProcess):
    """
    Класс, который реализует мониторинг позиции BUY с TP/SL.
    Входит в позицию при strategy.is_accepted_by_strategy == True,
    затем отслеживает достижение take-profit или stop-loss.
    Тейк-профит по логике 1 (на close), стоп-лосс по логике 2 (на low, закрытие по sl цене).
    """
    def __init__(
            self,
            history_market_parser: HistoryMarketParser,  # обычно 1m таймфрейм
            regulator_tpsl: RegulatorTPSL,  # объект с .take_profit и .stop_loss (в цене)
            symbol1='BTC',
            symbol2='USDT',
            symbol1_amount: float = 0.0,  # всего BTC (или другого base)
            balance_usdt: BalanceUSDT = None,
            fee: float = 0.001,  # комиссия 0.04%
    ):
        self.history_market_parser = history_market_parser
        self.regulator_tpsl = regulator_tpsl
        # Состояние позиции
        self.in_position = False
        self.entry_price = None
        self.entry_time = None
        # Балансы
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.symbol1_amount = symbol1_amount  # BTC
        self.balance_usdt = balance_usdt
        self.fee = fee
        # Статистика (опционально, для отладки/логов)
        self.trades = []
        log_date = datetime.now().strftime("%Y%m%d")
        log_dir = "files/decisions"
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"decisions_{log_date}.txt")
        self._log(f"\n=== НОВАЯ СЕССИЯ БЭКТЕСТА / РЕАЛ-ТАЙМ ===\n"
                  f"Пара: {symbol1}/{symbol2} | Начальный баланс: {balance_usdt.amount:.2f} USDT\n")

    def _log(self, message: str):
        """Записывает сообщение в лог-файл с временной меткой"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        # Также выводим в консоль (как было)
        print(log_entry.strip())

    def prepare(self, start_time: datetime = None, end_time: datetime = None):
        """Подготовка — обычно ничего особенного не требуется"""
        pass

    def run_realtime(self):
        """Реал-тайм версия (вызывается периодически)"""
        self._tick(datetime.now())

    def run_historical(self, start_time: datetime, current_time: datetime):
        """Историческая версия — вызывается на каждом тике"""
        self._tick(current_time)

    def _tick(self, current_time: datetime):
        """
        Основная логика на каждом тике (1m обычно)
        """
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return
        # Берём последнюю свечу
        last_row = self.history_market_parser.df.iloc[-1]
        current_price = last_row['close']
        current_timestamp = pd.to_datetime(last_row['time'])
        # 1. Проверяем сигнал на вход, если ещё не в позиции
        if not self.in_position:
            if self.regulator_tpsl.is_accepted_by_regulator:
                self._open_position(current_price, current_timestamp)
        else:
            # 2. Уже в позиции → проверяем TP/SL
            self._check_exit_conditions(last_row, current_price, current_timestamp)

    def _open_position(self, price: float, timestamp: datetime):
        """Открываем BUY позицию"""
        amount_to_buy = self.regulator_tpsl.symbol1_prepared_converted_amount
        # Сколько USDT уйдёт на покупку + комиссия
        cost = amount_to_buy * price
        fee_cost = cost * self.fee
        if self.balance_usdt.amount < (cost + fee_cost):
            #print(f"[{timestamp}] Недостаточно {self.symbol2} для открытия позиции")
            return
        # Выполняем покупку
        self.balance_usdt.amount -= (cost + fee_cost)
        self.symbol1_amount += amount_to_buy
        self.in_position = True
        self.entry_price = price
        self.entry_time = timestamp
        self._log(f"[{timestamp}] BUY OPEN @ {price:.2f} | "
              f"amount: {amount_to_buy:.6f} {self.symbol1} | "
              f"cost: {cost:.2f} + {fee_cost:.2f}(fee) = {(cost + fee_cost)} | "
              f"take-profit: {self.regulator_tpsl.take_profit} | "
              f"stop-loss: {self.regulator_tpsl.stop_loss}"
        )
        # Можно сохранить информацию о сделке
        self.trades.append({
            "type": "BUY",
            "time": timestamp,
            "price": price,
            "amount": amount_to_buy,
            "fee": fee_cost
        })

    def _check_exit_conditions(self, last_row, current_price: float, timestamp: datetime):
        """Проверяем условия выхода по TP или SL"""
        tp = self.regulator_tpsl.take_profit
        sl = self.regulator_tpsl.stop_loss
        # Сначала проверяем SL по логике 2 (на low)
        hit_sl = last_row['low'] <= sl
        if hit_sl:
            self._close_position(sl, timestamp, "SL")
            return  # Выходим, если SL сработал
        # Затем проверяем TP по логике 1 (на close)
        hit_tp = current_price >= tp
        if hit_tp:
            self._close_position(current_price, timestamp, "TP")

    def _close_position(self, price: float, timestamp: datetime, reason: str):
        """Закрываем позицию — продаём весь купленный объём"""
        amount_to_sell = self.regulator_tpsl.symbol1_prepared_converted_amount
        self.regulator_tpsl.is_accepted_by_regulator = False
        proceeds = amount_to_sell * price
        fee_cost = proceeds * self.fee
        # Получаем USDT обратно
        self.balance_usdt.amount += (proceeds) # ATTENTION!!!!
        self.symbol1_amount -= amount_to_sell
        pnl = (price - self.entry_price) * amount_to_sell - fee_cost  # комиссии на вход и выход
        self._log(f"[{timestamp}] CLOSE {reason} @ {price:.2f} | "
              f"pnl: {pnl:.2f} USDT | "
              f"balance USDT: {self.balance_usdt.amount:.2f} | BTC: {self.symbol1_amount:.6f}")
        # Сохраняем информацию о закрытии
        self.trades.append({
            "type": "SELL",
            "reason": reason,
            "time": timestamp,
            "price": price,
            "amount": amount_to_sell,
            "pnl": pnl,
            "fee": fee_cost
        })
        # Сбрасываем состояние позиции
        self.in_position = False
        self.entry_price = None
        self.entry_time = None

    def finalize(self):
        """
        Вызывается в конце исторического бэктекста.
        Если позиция осталась открытой — принудительно закрываем.
        """
        if not self.in_position:
            self._log("[END OF BACKTEST] Позиций нет — ничего не закрываем.")
            return

        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            self._log("[finalize] Нет данных для закрытия позиции!")
            return

        last_row = self.history_market_parser.df.iloc[-1]
        final_price = last_row['close']
        final_timestamp = pd.to_datetime(last_row['time'])

        self._log(f"[END OF BACKTEST] Принудительное закрытие открытой позиции по цене {final_price:.2f}")

        self._close_position(final_price, final_timestamp, "FINAL_CLOSE")

        self._log(f"[BACKTEST FINISHED] Итоговый баланс: {self.balance_usdt.amount:.2f} USDT | "
                  f"{self.symbol1}: {self.symbol1_amount:.6f}")