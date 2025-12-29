from datetime import datetime
import pandas as pd
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
            symbol1_all: float = 0.0,  # всего BTC (или другого base)
            symbol2_all: float = 10000.0,  # всего USDT (или quote)
            fee: float = 0.0004,  # комиссия 0.04%
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
        self.symbol1_all = symbol1_all  # BTC
        self.symbol2_all = symbol2_all  # USDT
        self.fee = fee
        # Статистика (опционально, для отладки/логов)
        self.trades = []

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
        if self.symbol2_all < (cost + fee_cost):
            print(f"[{timestamp}] Недостаточно {self.symbol2} для открытия позиции")
            return
        # Выполняем покупку
        self.symbol2_all -= (cost + fee_cost)
        self.symbol1_all += amount_to_buy
        self.in_position = True
        self.entry_price = price
        self.entry_time = timestamp
        print(f"[{timestamp}] BUY OPEN @ {price:.2f} | "
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
        self.symbol2_all += (proceeds - fee_cost)
        self.symbol1_all -= amount_to_sell
        pnl = (price - self.entry_price) * amount_to_sell - fee_cost * 2  # комиссии на вход и выход
        print(f"[{timestamp}] CLOSE {reason} @ {price:.2f} | "
              f"pnl: {pnl:.2f} USDT | "
              f"balance USDT: {self.symbol2_all:.2f} | BTC: {self.symbol1_all:.6f}")
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