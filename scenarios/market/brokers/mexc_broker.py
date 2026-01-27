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
            test_mode: bool = False,
            slippage_percent: float = 0.4,          # макс допустимое проскальзывание при входе (%)
            maker_offset_percent: float = 0.08,     # смещение для maker-ордеров (вниз при buy, вверх при sell)
            tp_maker_offset_percent: float = 0.05,  # небольшой offset для TP limit (выше take_profit для sell)
    ):
        super().__init__()
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

        # Настраиваемые параметры
        self.slippage_percent = slippage_percent
        self.maker_offset_percent = maker_offset_percent
        self.tp_maker_offset_percent = tp_maker_offset_percent

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

    def run_realtime(self):
        self._tick()

    def _tick(self):
        try:
            balance = self.exchange.fetch_balance()
            usdt_amount = balance.get(self.buyer.symbol2, {}).get('free', 0.0)
            symbol1_amount = balance.get(self.buyer.symbol1, {}).get('free', 0.0)
            self.buyer.sync_balances(usdt_amount, symbol1_amount)
            current_in_position = symbol1_amount > 0.000001
        except Exception as e:
            self._log(f"MEXCBroker: Balance error: {str(e)}")
            return

        # Реакция на сигнал BUY от стратегии
        if self.buyer.in_position and not self.prev_in_position:
            if not current_in_position:
                self._execute_limit_buy_with_tpsl()

        # Если стратегия вышла, но позиция ещё есть → fallback market sell
        if not self.buyer.in_position and self.prev_in_position:
            if current_in_position:
                self._execute_market_sell_fallback()

        # Если биржа закрыла позицию (TP/SL сработали), но стратегия ещё думает что в позиции
        if self.buyer.in_position and not current_in_position:
            self.buyer.in_position = False
            self._log("Позиция закрыта биржей (вероятно TP/SL) → синхронизировали состояние")

        # Отмена висячих ордеров, если нет позиции
        if not current_in_position:
            try:
                self.exchange.cancel_all_orders(self.symbol)
            except Exception:
                pass

        self.prev_in_position = self.buyer.in_position

    def _execute_limit_buy_with_tpsl(self):
        """Limit buy (maker) + attached conditional TP/SL"""
        requested_amount = self.buyer.regulator_tpsl.symbol1_prepared_converted_amount
        if requested_amount <= 0:
            self._log("Нулевое количество для покупки → пропуск")
            return

        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            ask = ticker['ask'] or ticker['last']
            current_price = ticker['last']

            # Maker-friendly цена входа
            entry_limit_price = ask * (1 - self.maker_offset_percent / 100)
            entry_limit_price = self.exchange.price_to_precision(self.symbol, entry_limit_price)

            amount = self.exchange.amount_to_precision(self.symbol, requested_amount)

            tp = self.buyer.regulator_tpsl.take_profit
            sl = self.buyer.regulator_tpsl.stop_loss

            # TP limit price — чуть выше для повышения шанса maker-исполнения
            tp_limit_price = tp * (1 + self.tp_maker_offset_percent / 100)
            tp_limit_price = self.exchange.price_to_precision(self.symbol, tp_limit_price)

            # Параметры для attached TP/SL (насколько ccxt/MEXC позволяет)
            params = {
                'takeProfitPrice': str(tp),                # trigger TP
                'takeProfitLimitPrice': str(tp_limit_price),
                'takeProfitType': 'LIMIT',
                'stopLossPrice': str(sl),                  # trigger SL
                'stopLossType': 'MARKET',
                'triggerType': 'LAST_PRICE',
                # Если MEXC требует других полей — добавь здесь
            }

            order = self.exchange.create_limit_buy_order(
                symbol=self.symbol,
                amount=amount,
                price=entry_limit_price,
                params=params
            )

            self._log(
                f"LIMIT BUY placed @ {entry_limit_price:.6f} "
                f"(ask ~{ask:.2f}) | amount: {amount} | "
                f"with TP@{tp_limit_price:.2f} / SL@{sl:.2f}"
            )

        except Exception as e:
            self._log(f"Ошибка размещения BUY+TP/SL: {str(e)}")
            self.buyer.in_position = False

    def _execute_market_sell_fallback(self):
        """Fallback: market sell если позиция осталась, но стратегия вышла"""
        amount = self.buyer.symbol1_amount
        if amount <= 0.000001:
            return

        try:
            amount_str = self.exchange.amount_to_precision(self.symbol, amount)
            order = self.exchange.create_market_sell_order(self.symbol, amount_str)
            filled = self.exchange.fetch_order(order['id'], self.symbol)
            price = filled.get('average', 0)
            self._log(f"Fallback MARKET SELL @ ~{price:.2f} | amount: {amount_str}")
        except Exception as e:
            self._log(f"Fallback SELL error: {str(e)}")

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
                self._execute_market_sell_fallback()
            self.exchange.cancel_all_orders(self.symbol)
            self._log("Finalize: позиция закрыта (если была), ордера отменены.")
        except Exception as e:
            self._log(f"Finalize error: {str(e)}")