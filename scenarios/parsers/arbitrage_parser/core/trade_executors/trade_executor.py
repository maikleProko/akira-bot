import asyncio
import json
import time
import traceback

import websockets


class TradeExecutor:
    def __init__(self, exchange_client, trade_validator, logger, production):
        self.exchange_client = exchange_client
        self.trade_validator = trade_validator
        self.logger = logger
        self.production = production

    def get_buy_ticker(self, symbol):
        ticker = self.exchange_client.fetch_ticker_price(symbol)
        if ticker is None or 'buy' not in ticker:
            self.logger.log_message(f"Ошибка: Не удалось получить текущую цену для {symbol}")
            return None
        return float(ticker['buy'])

    def simulate_trade(self, symbol, direction, amount, fee_rate, from_asset, to_asset):
        ticker = self.exchange_client.fetch_ticker_price(symbol)
        if ticker is None or 'buy' not in ticker or 'sell' not in ticker:
            self.logger.log_message(f"Симуляция: недопустимый ответ get_ticker для {symbol}")
            return None, False
        current_price = float(ticker['sell']) if direction == 'sell' else float(ticker['buy'])
        new_amount = amount * current_price * (1 - fee_rate) if direction == 'sell' else (amount / current_price) * (1 - fee_rate)
        self.logger.log_message(f"Симуляция транзакции {direction} {amount:.8f} {from_asset} -> {to_asset} при цене {current_price:.8f}")
        self.logger.log_message(f"Ожидаемое количество: {new_amount:.8f} {to_asset}")
        return new_amount, True

    def adjust_for_sell(self, amount, direction, base_min_size, base_increment, from_asset, symbol):
        if direction == 'sell':
            adjusted_amount, success = self.trade_validator.adjust_sell_amount(self.logger.log_message, amount, base_min_size, base_increment, from_asset, symbol)
            #if not success:
                #return None, False
            return adjusted_amount, True
        return amount, True

    def adjust_for_buy(self, amount, direction, quote_min_size, from_asset, symbol, current_ask_price, base_min_size, base_increment, quote_increment, to_asset):
        if direction != 'sell':
            funds, success = self.trade_validator.adjust_buy_funds(self.logger.log_message, amount, amount, quote_min_size, from_asset, symbol)
            #if not success:
                #return None, False
            adjusted_amount, success = self.trade_validator.adjust_buy_amount(self.logger.log_message, funds, current_ask_price, base_min_size, base_increment, quote_increment, to_asset, symbol, amount, from_asset)
            #if not success:
                #return None, False
            return adjusted_amount, True
        return amount, True

    def execute_trade_prod(self, from_asset, to_asset, amount, symbol, direction, expected_price, fee_rate, price_map, base_min_size, quote_min_size, base_increment, quote_increment, ask_price):
        start_time = time.time()
        try:
            adjusted_amount, success = self.adjust_for_sell(amount, direction, base_min_size, base_increment, from_asset, symbol)
            if not success:
                return None, False
            if direction != 'sell':
                current_ask_price = self.get_buy_ticker(symbol)
                if current_ask_price is None:
                    return None, False
                adjusted_amount, success = self.adjust_for_buy(amount, direction, quote_min_size, from_asset, symbol, current_ask_price, base_min_size, base_increment, quote_increment, to_asset)
                if not success:
                    return None, False
            order_params = self.exchange_client.create_order_params(symbol, direction, 'market', adjusted_amount)
            order_id = self.exchange_client.place_order(order_params)
            self.logger.log_message(f"Размещен ордер {order_id} для {direction} {adjusted_amount:.8f} {from_asset} -> {to_asset}")
            return self.exchange_client.monitor_order(symbol, order_id, direction, adjusted_amount, from_asset, to_asset, expected_price, fee_rate)
        except Exception as e:
            # Полный стек в текстовом виде
            full_trace = traceback.format_exc()

            # Информация о последнем фрейме traceback: файл, строка, функция, код
            tb_list = traceback.extract_tb(e.__traceback__)
            if tb_list:
                last_frame = tb_list[-1]
                file_name = last_frame.filename
                line_no = last_frame.lineno
                func_name = last_frame.name
                code_line = (last_frame.line or '').strip()
                location_info = f"{file_name}:{line_no} in {func_name} -> {code_line}"
            else:
                location_info = "Traceback frames not available"

            # Логирование с деталями
            self.logger.log_message(f"Ошибка при выполнении транзакции {direction} {amount:.8f} {from_asset} -> {to_asset}: {str(e)}")
            self.logger.log_message(f"Место ошибки: {location_info}")
            self.logger.log_message("Полный стек вызовов:\n" + full_trace)
            self.logger.log_message(f"Ограничения для {symbol}: base_min_size={base_min_size:.8f}, quote_min_size={quote_min_size:.8f}, base_increment={base_increment:.8f}, quote_increment={quote_increment:.8f}")
            return None, False

    def execute_trade(self, from_asset, to_asset, amount, symbol, direction, expected_price, fee_rate, price_map):
        if not self.trade_validator.validate_symbol(symbol, price_map):
            return None, False
        base_min_size, quote_min_size, base_increment, quote_increment, ask_price = self.trade_validator.get_constraints(symbol, price_map, expected_price)
        if not self.production:
            return self.simulate_trade(symbol, direction, amount, fee_rate, from_asset, to_asset)
        amount, success = self.trade_validator.adjust_balance(self.exchange_client.check_balance, self.logger.log_message, from_asset, amount, self.production)
        if not success:
            return None, False
        return self.execute_trade_prod(from_asset, to_asset, amount, symbol, direction, expected_price, fee_rate, price_map, base_min_size, quote_min_size, base_increment, quote_increment, ask_price)
