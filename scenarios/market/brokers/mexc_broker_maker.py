# MEXCBrokerMaker.py
import ccxt
from datetime import datetime
from typing import Optional

from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from utils.core.functions import MarketProcess


class MEXCBrokerMaker(MarketProcess):
    def __init__(
            self,
            buyer: BuyerTPSL,
            api_key: str,
            api_secret: str,
            test_mode: bool = False
    ):
        self.buyer = buyer
        self.symbol = f"{buyer.symbol1}{buyer.symbol2}"
        self.exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        if test_mode:
            self.exchange.set_sandbox_mode(True)
        self.active = False
        self.prev_in_position = False
        self.log_file = self.buyer.log_file
        self.current_buy_order_id: Optional[str] = None
        self.current_tp_order_id: Optional[str] = None

    def prepocess_realtime(self):
        self.active = True

    def prepare(self, start_time: datetime = None, end_time: datetime = None):
        if not self.active:
            return
        try:
            self.exchange.fetch_balance()
            self._log("MEXCBrokerMaker: Соединение с MEXC установлено успешно.")
            self._cancel_all_open_orders()
        except Exception as e:
            self._log(f"MEXCBrokerMaker: Ошибка соединения с MEXC: {str(e)}")

    def run(self, start_time: datetime = None, current_time: datetime = None):
        if not self.active:
            return
        if start_time is not None or current_time is not None:
            return
        self._tick()

    def _tick(self):
        current_in_position = self._sync_balances_and_position()

        self._check_orders()

        if self.buyer.in_position and not self.prev_in_position:
            if not current_in_position and not self.current_buy_order_id:
                self._place_buy_limit()

        if not self.buyer.in_position and self.prev_in_position:
            if current_in_position:
                self._execute_sell()  # Market sell, так как сигнал на выход (SL или manual)

        if self.buyer.in_position and not current_in_position:
            self.buyer.in_position = False
            self._log("MEXCBrokerMaker: Позиция не найдена на бирже, синхронизируем buyer.in_position = False")

        if not self.buyer.in_position and current_in_position:
            self._execute_sell()
            self._log("MEXCBrokerMaker: Несанкционированная позиция, выполняем продажу")

        self.prev_in_position = self.buyer.in_position

    def _sync_balances_and_position(self) -> bool:
        try:
            balance = self.exchange.fetch_balance()
            usdt_amount = balance.get(self.buyer.symbol2, {}).get('free', 0.0)
            symbol1_amount = balance.get(self.buyer.symbol1, {}).get('free', 0.0)
            self.buyer.sync_balances(usdt_amount, symbol1_amount)
            return symbol1_amount > 0.000001
        except Exception as e:
            self._log(f"MEXCBrokerMaker: Ошибка синхронизации: {str(e)}")
            return False

    def _check_orders(self):
        if self.current_buy_order_id:
            try:
                order = self.exchange.fetch_order(self.current_buy_order_id, self.symbol)
                if order['status'] == 'closed':
                    if order['filled'] > 0:
                        self._handle_buy_filled(order)
                    else:
                        self.current_buy_order_id = None
                        self._log("MEXCBrokerMaker: Buy ордер закрыт без филла")
                elif order['status'] in ('canceled', 'rejected'):
                    self.current_buy_order_id = None
                    self._log("MEXCBrokerMaker: Buy ордер отменен/отклонен")
            except Exception as e:
                self._log(f"MEXCBrokerMaker: Ошибка проверки buy: {str(e)}")

        if self.current_tp_order_id:
            try:
                order = self.exchange.fetch_order(self.current_tp_order_id, self.symbol)
                if order['status'] == 'closed':
                    if order['filled'] > 0:
                        self._handle_sell_filled(order, "TP")
                    else:
                        self.current_tp_order_id = None
                        self._log("MEXCBrokerMaker: TP ордер закрыт без филла")
                elif order['status'] in ('canceled', 'rejected'):
                    self.current_tp_order_id = None
                    self._log("MEXCBrokerMaker: TP ордер отменен/отклонен")
            except Exception as e:
                self._log(f"MEXCBrokerMaker: Ошибка проверки TP: {str(e)}")

    def _place_buy_limit(self):
        amount = self.buyer.regulator_tpsl.symbol1_prepared_converted_amount
        if amount <= 0:
            return
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['bid']
            price = self.exchange.price_to_precision(self.symbol, price)
            amount = self.exchange.amount_to_precision(self.symbol, amount)
            order = self.exchange.create_limit_buy_order(self.symbol, amount, price)
            self.current_buy_order_id = order['id']
            self._log(f"MEXCBrokerMaker: Limit BUY размещен @ {price:.2f} | amount: {amount:.6f}")
        except Exception as e:
            self._log(f"MEXCBrokerMaker: Limit BUY ошибка: {str(e)}")
            self.buyer.in_position = False

    def _place_tp_limit(self):
        amount = self.buyer.symbol1_amount
        if amount <= 0:
            return
        try:
            tp_price = self.buyer.regulator_tpsl.take_profit
            tp_price = self.exchange.price_to_precision(self.symbol, tp_price)
            amount = self.exchange.amount_to_precision(self.symbol, amount)
            order = self.exchange.create_limit_sell_order(self.symbol, amount, tp_price)
            self.current_tp_order_id = order['id']
            self._log(f"MEXCBrokerMaker: Limit SELL (TP) размещен @ {tp_price:.2f} | amount: {amount:.6f}")
        except Exception as e:
            self._log(f"MEXCBrokerMaker: Limit TP ошибка: {str(e)}")

    def _handle_buy_filled(self, order):
        filled_amount = order['filled']
        actual_price = order['average']
        actual_fee = order.get('fee', {}).get('cost', 0.0)
        timestamp = datetime.fromtimestamp(order['timestamp'] / 1000)
        self.buyer.update_actual_open(actual_price, filled_amount, actual_fee, timestamp)
        self.current_buy_order_id = None
        self._log(f"MEXCBrokerMaker: BUY executed @ {actual_price:.2f} | filled: {filled_amount:.6f} | fee: {actual_fee:.2f}")
        self._place_tp_limit()  # Автоматический TP с биржи

    def _handle_sell_filled(self, order, reason: str):
        filled_amount = order['filled']
        actual_price = order['average']
        actual_fee = order.get('fee', {}).get('cost', 0.0)
        timestamp = datetime.fromtimestamp(order['timestamp'] / 1000)
        self.buyer.update_actual_close(actual_price, filled_amount, actual_fee, reason, timestamp)
        self.current_tp_order_id = None
        self._log(f"MEXCBrokerMaker: SELL ({reason}) executed @ {actual_price:.2f} | filled: {filled_amount:.6f} | fee: {actual_fee:.2f}")
        self._cancel_all_open_orders()

    def _execute_sell(self):
        amount = self.buyer.symbol1_amount
        if amount <= 0:
            return
        try:
            order = self.exchange.create_market_sell_order(self.symbol, amount)
            filled_order = self.exchange.fetch_order(order['id'], self.symbol)
            actual_amount = filled_order['filled']
            actual_price = filled_order['average']
            actual_fee = filled_order.get('fee', {}).get('cost', 0.0)
            timestamp = datetime.fromtimestamp(filled_order['timestamp'] / 1000)
            last_price = self.buyer.history_market_parser.df.iloc[-1]['close'] if self.buyer.history_market_parser.df is not None else actual_price
            reason = "TP" if last_price >= self.buyer.regulator_tpsl.take_profit else "SL" if last_price <= self.buyer.regulator_tpsl.stop_loss else "UNKNOWN"
            self.buyer.update_actual_close(actual_price, actual_amount, actual_fee, reason, timestamp)
            self._log(f"MEXCBrokerMaker: Market SELL executed @ {actual_price:.2f} | amount: {actual_amount:.6f} | fee: {actual_fee:.2f} | reason: {reason}")
            self._cancel_all_open_orders()
        except Exception as e:
            self._log(f"MEXCBrokerMaker: SELL error: {str(e)}")

    def _cancel_all_open_orders(self):
        try:
            self.exchange.cancel_all_orders(self.symbol)
            self.current_buy_order_id = None
            self.current_tp_order_id = None
        except Exception as e:
            self._log(f"MEXCBrokerMaker: Cancel error: {str(e)}")

    def finalize(self):
        if not self.active:
            return
        try:
            self._cancel_all_open_orders()
            if self._sync_balances_and_position():
                self._execute_sell()
        except Exception as e:
            self._log(f"MEXCBrokerMaker: Finalize error: {str(e)}")

    def _log(self, message: str):
        entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        print(entry.strip())