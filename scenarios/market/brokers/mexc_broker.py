
# MEXCBroker.py
import ccxt
from datetime import datetime

from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from utils.core.functions import MarketProcess


class MEXCBroker(MarketProcess):
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

    def prepocess_realtime(self):
        self.active = True

    def prepare(self, start_time: datetime = None, end_time: datetime = None):
        if not self.active:
            return
        try:
            self.exchange.fetch_balance()
            self._log("MEXCBroker: Соединение с MEXC установлено успешно.")
        except Exception as e:
            self._log(f"MEXCBroker: Ошибка соединения с MEXC: {str(e)}")

    def run(self, start_time: datetime = None, current_time: datetime = None):
        if not self.active:
            return
        if start_time is not None or current_time is not None:
            return
        self._tick()

    def _tick(self):
        # First, sync real balances to buyer
        try:
            balance = self.exchange.fetch_balance()
            usdt_amount = balance.get(self.buyer.symbol2, {}).get('free', 0.0)
            symbol1_amount = balance.get(self.buyer.symbol1, {}).get('free', 0.0)
            self.buyer.sync_balances(usdt_amount, symbol1_amount)
            current_in_position = symbol1_amount > 0.000001
        except Exception as e:
            return

        # React to changes in buyer's in_position (after its run)
        if self.buyer.in_position and not self.prev_in_position:
            if not current_in_position:
                self._execute_buy()

        if not self.buyer.in_position and self.prev_in_position:
            if current_in_position:
                self._execute_sell()

        # Handle discrepancies (though sync should handle most)
        if self.buyer.in_position and not current_in_position:
            self.buyer.sync_balances(usdt_amount, symbol1_amount)  # Force sync again

        if not self.buyer.in_position and current_in_position:
            self._execute_sell()

        self.prev_in_position = self.buyer.in_position

    def _execute_buy(self):
        amount = self.buyer.regulator_tpsl.symbol1_prepared_converted_amount
        if amount <= 0:
            return
        try:
            order = self.exchange.create_market_buy_order(self.symbol, amount)
            filled_order = self.exchange.fetch_order(order['id'], self.symbol)
            actual_amount = filled_order['filled']
            actual_price = filled_order['average']
            actual_fee = filled_order.get('fee', {}).get('cost', 0.0)
            timestamp = datetime.fromtimestamp(filled_order['timestamp'] / 1000)
            self.buyer.update_actual_open(actual_price, actual_amount, actual_fee, timestamp)
            self._log(
                f"MEXCBroker: BUY executed @ {actual_price:.2f} | amount: {actual_amount:.6f} | fee: {actual_fee:.2f}")
        except Exception as e:
            self._log(f"MEXCBroker: BUY error: {str(e)}")
            self.buyer.in_position = False

    def _execute_sell(self):
        amount = self.buyer.symbol1_amount  # Use real amount from sync
        if amount <= 0:
            return
        try:
            order = self.exchange.create_market_sell_order(self.symbol, amount)
            filled_order = self.exchange.fetch_order(order['id'], self.symbol)
            actual_amount = filled_order['filled']
            actual_price = filled_order['average']
            actual_fee = filled_order.get('fee', {}).get('cost', 0.0)
            timestamp = datetime.fromtimestamp(filled_order['timestamp'] / 1000)
            # Determine reason based on price
            last_price = self.buyer.history_market_parser.df.iloc[-1][
                'close'] if self.buyer.history_market_parser.df is not None else actual_price
            reason = "TP" if last_price >= self.buyer.regulator_tpsl.take_profit else "SL" if last_price <= self.buyer.regulator_tpsl.stop_loss else "UNKNOWN"
            self.buyer.update_actual_close(actual_price, actual_amount, actual_fee, reason, timestamp)
            self._log(
                f"MEXCBroker: SELL executed @ {actual_price:.2f} | amount: {actual_amount:.6f} | fee: {actual_fee:.2f}")
        except Exception as e:
            self._log(f"MEXCBroker: SELL error: {str(e)}")

    def _log(self, message: str):
        entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        print(entry.strip())

    def finalize(self):
        if not self.active:
            return
        try:
            balance = self.exchange.fetch_balance()
            symbol1_amount = balance.get(self.buyer.symbol1, {}).get('free', 0.0)
            if symbol1_amount > 0.000001:
                self._execute_sell()
        except Exception as e:
            self._log(f"MEXCBroker: Finalize error: {str(e)}")