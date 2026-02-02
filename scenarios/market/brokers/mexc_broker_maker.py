# MEXCBrokerMakerSimple.py
import ccxt
from datetime import datetime

from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from utils.core.functions import MarketProcess


class MEXCBrokerMakerSimple(MarketProcess):
    def __init__(
            self,
            buyer: BuyerTPSL,
            api_key: str,
            api_secret: str,
            test_mode: bool = False,
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

        self.current_open_order_id = None       # buy лимитка на вход
        self.current_close_order_id = None      # sell лимитка на TP/SL

    def prepocess_realtime(self):
        self.active = True

    def prepare(self, start_time=None, end_time=None):
        if not self.active:
            return
        try:
            self.exchange.fetch_balance()
            self._log("MEXC Maker: подключение ок")
            self._cancel_all_open_orders()   # чистим на старте
        except Exception as e:
            self._log(f"MEXC Maker: ошибка подключения → {e}")

    def run(self, start_time=None, current_time=None):
        if not self.active:
            return

        try:
            self._sync_position()
            self._check_orders_status()
            self._react_to_signals()
        except Exception as e:
            self._log(f"run error: {e}")

    def _sync_position(self):
        bal = self.exchange.fetch_balance()
        usdt_free = bal.get(self.buyer.symbol2, {}).get('free', 0.0)
        coin_free = bal.get(self.buyer.symbol1, {}).get('free', 0.0)
        self.buyer.sync_balances(usdt_free, coin_free)
        return coin_free > 0.000001   # считаем, что в позиции

    def _check_orders_status(self):
        # проверяем buy-ордер (вход)
        if self.current_open_order_id:
            try:
                o = self.exchange.fetch_order(self.current_open_order_id, self.symbol)
                if o['status'] in ('closed', 'filled'):
                    self._on_buy_filled(o)
                elif o['status'] in ('canceled', 'expired'):
                    self.current_open_order_id = None
            except Exception:
                pass  # пропускаем, попробуем в следующий тик

        # проверяем sell-ордер (выход)
        if self.current_close_order_id:
            try:
                o = self.exchange.fetch_order(self.current_close_order_id, self.symbol)
                if o['status'] in ('closed', 'filled'):
                    self._on_sell_filled(o)
                elif o['status'] in ('canceled', 'expired'):
                    self.current_close_order_id = None
            except Exception:
                pass

    def _react_to_signals(self):
        now_in_position = self.buyer.symbol1_amount > 0.000001

        # ----------------------------------------------------------------
        # Сигнал на ВХОД
        # ----------------------------------------------------------------
        if self.buyer.in_position and not self.prev_in_position:
            if now_in_position:
                self._log("уже в позиции по балансу → пропускаем buy")
            elif self.current_open_order_id:
                self._log("buy лимитка уже висит → пропускаем")
            else:
                self._place_limit_buy()

        # ----------------------------------------------------------------
        # Сигнал на ВЫХОД
        # ----------------------------------------------------------------
        if not self.buyer.in_position and self.prev_in_position:
            if not now_in_position:
                self._log("уже не в позиции → пропускаем sell")
            else:
                # закрываем по TP или SL в зависимости от того, что сработало
                self._place_limit_sell_by_tpsl()

        self.prev_in_position = self.buyer.in_position

    def _place_limit_buy(self):
        try:
            amount = self.buyer.regulator_tpsl.symbol1_prepared_converted_amount
            if amount <= 0:
                self._log("нулевой объём на покупку → игнор")
                return

            # Берём последнюю известную цену (можно заменить на fetch_ticker)
            price = self.buyer.history_market_parser.df.iloc[-1]['close'] if self.buyer.history_market_parser.df is not None else None
            if price is None:
                ticker = self.exchange.fetch_ticker(self.symbol)
                price = ticker['last']

            price = self.exchange.price_to_precision(self.symbol, price)
            amount = self.exchange.amount_to_precision(self.symbol, amount)

            order = self.exchange.create_limit_buy_order(self.symbol, amount, price)
            self.current_open_order_id = order['id']
            self._log(f"BUY LIMIT поставлен @ {price}  qty={amount}")

        except Exception as e:
            self._log(f"BUY LIMIT ошибка: {e}")
            self.current_open_order_id = None

    def _place_limit_sell_by_tpsl(self):
        try:
            amount = self.buyer.symbol1_amount
            if amount <= 0.000001:
                self._log("нет монет на продажу → игнор")
                return

            # Определяем, по какому уровню закрываемся
            if self.buyer.regulator_tpsl.stop_loss_triggered():
                price = self.buyer.regulator_tpsl.stop_loss
                reason = "SL"
            else:
                price = self.buyer.regulator_tpsl.take_profit
                reason = "TP"

            price = self.exchange.price_to_precision(self.symbol, price)
            amount = self.exchange.amount_to_precision(self.symbol, amount)

            # отменяем старый, если был
            self._cancel_close_order()

            order = self.exchange.create_limit_sell_order(self.symbol, amount, price)
            self.current_close_order_id = order['id']
            self._log(f"SELL LIMIT ({reason}) поставлен @ {price}  qty={amount}")

        except Exception as e:
            self._log(f"SELL LIMIT ошибка: {e}")
            # аварийное закрытие рыночно
            try:
                self.exchange.create_market_sell_order(self.symbol, amount)
                self._log("аварийная рыночная продажа")
            except:
                pass

    def _on_buy_filled(self, order):
        price = order['average'] or order['price']
        filled = order['filled']
        fee = order.get('fee', {}).get('cost', 0.0)
        ts = datetime.fromtimestamp(order['timestamp'] / 1000)

        self.buyer.update_actual_open(price, filled, fee, ts)
        self.current_open_order_id = None
        self._log(f"BUY LIMIT исполнен @ {price:.6f}  filled={filled:.6f}")

        # после входа сразу ставим лимитку на выход
        self._place_limit_sell_by_tpsl()

    def _on_sell_filled(self, order):
        price = order['average'] or order['price']
        filled = order['filled']
        fee = order.get('fee', {}).get('cost', 0.0)
        ts = datetime.fromtimestamp(order['timestamp'] / 1000)

        reason = "TP" if price >= self.buyer.regulator_tpsl.take_profit else "SL"
        self.buyer.update_actual_close(price, filled, fee, reason, ts)
        self.current_close_order_id = None
        self._log(f"SELL LIMIT исполнен @ {price:.6f}  filled={filled:.6f}  reason={reason}")

    def _cancel_close_order(self):
        if self.current_close_order_id:
            try:
                self.exchange.cancel_order(self.current_close_order_id, self.symbol)
            except:
                pass
            self.current_close_order_id = None

    def _cancel_all_open_orders(self):
        try:
            self.exchange.cancel_all_orders(self.symbol)
            self.current_open_order_id = None
            self.current_close_order_id = None
        except:
            pass

    def finalize(self):
        if not self.active:
            return
        try:
            self._cancel_all_open_orders()
            if self.buyer.symbol1_amount > 0.000001:
                self.exchange.create_market_sell_order(
                    self.symbol,
                    self.buyer.symbol1_amount
                )
                self._log("finalize → принудительная рыночная продажа")
        except Exception as e:
            self._log(f"finalize error: {e}")

    def _log(self, msg: str):
        entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        print(entry.strip())