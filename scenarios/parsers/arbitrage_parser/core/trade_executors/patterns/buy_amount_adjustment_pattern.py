from time import sleep
from decimal import ROUND_FLOOR, Decimal

from scenarios.parsers.arbitrage_parser.core.utils.exchange_client import ExchangeClient
from scenarios.parsers.arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class BuyAmountAdjustmentPattern(ValidationPattern):
    def __init__(self, logger):
        self.logger = logger

    def calculate_max_base(self, funds_dec, current_ask_price_dec):
        return funds_dec / current_ask_price_dec

    def floor_adjust_base(self, max_base_amount_dec, base_increment_dec):
        return (max_base_amount_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * Decimal(base_increment_dec)

    def check_min_base(self, adjusted_base_amount_dec, base_min_size_dec):
        return adjusted_base_amount_dec >= base_min_size_dec

    def log_min_base_error(self, adjusted_base_amount_dec, base_min_size_dec, to_asset, symbol):
        self.logger.log_message(f"Ошибка: Скорректированное количество {adjusted_base_amount_dec} {to_asset} меньше минимального размера {base_min_size_dec} для {symbol}")

    def calculate_adjusted_amount(self, adjusted_base_amount_dec, current_ask_price_dec):
        return Decimal(adjusted_base_amount_dec) * Decimal(current_ask_price_dec)

    def floor_adjust_amount(self, adjusted_amount_dec, quote_increment_dec):
        return (adjusted_amount_dec / quote_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * Decimal(quote_increment_dec)

    def check_positive_amount(self, adjusted_amount_dec):
        return adjusted_amount_dec > 0

    def log_zero_amount_error(self, adjusted_amount_dec, from_asset, symbol):
        self.logger.log_message(f"Ошибка: После округления сумма {adjusted_amount_dec} {from_asset} равна нулю для {symbol}")

    def check_available_balance(self, adjusted_amount_dec, available_dec):
        return adjusted_amount_dec <= available_dec

    def log_exceed_balance_error(self, adjusted_amount_dec, available_dec, from_asset):
        self.logger.log_message(f"Ошибка: Скорректированная сумма {adjusted_amount_dec} {from_asset} превышает доступный баланс {available_dec}")

    def log_adjusted_buy(self, adjusted_amount_dec, from_asset, quote_increment_dec, adjusted_base_amount_dec, to_asset, current_ask_price_dec, adjusted_amount_mode):
        self.logger.log_message(f"Скорректированная сумма для покупки: {adjusted_amount_dec} {from_asset} (quote_increment: {quote_increment_dec}, expected {adjusted_base_amount_dec} {to_asset} at price {current_ask_price_dec}), mode: {str(adjusted_amount_mode)}")

    def adjust_buy_amount_step0(self, funds, current_ask_price, base_min_size, base_increment, to_asset, symbol):
        funds_dec = Decimal(str(funds))
        current_ask_price_dec = Decimal(str(current_ask_price))
        max_base_amount_dec = self.calculate_max_base(funds_dec, current_ask_price_dec)

        return max_base_amount_dec, True

    def adjust_buy_amount_step1(self, funds, current_ask_price, base_min_size, base_increment, to_asset, symbol):
        funds_dec = Decimal(str(funds))
        current_ask_price_dec = Decimal(str(current_ask_price))
        base_min_size_dec = Decimal(str(base_min_size))
        base_increment_dec = Decimal(str(base_increment))
        max_base_amount_dec = self.calculate_max_base(funds_dec, current_ask_price_dec)
        adjusted_base_amount_dec = self.floor_adjust_base(max_base_amount_dec, base_increment_dec)
        if not self.check_min_base(adjusted_base_amount_dec, base_min_size_dec):
            self.log_min_base_error(adjusted_base_amount_dec, base_min_size_dec, to_asset, symbol)
            return 0, False
        return adjusted_base_amount_dec, True

    def adjust_buy_amount_step2(self, adjusted_base_amount_dec, current_ask_price_dec, quote_increment, from_asset, symbol, available):
        quote_increment_dec = Decimal(str(quote_increment))
        available_dec = Decimal(str(available))
        adjusted_amount_dec = self.calculate_adjusted_amount(adjusted_base_amount_dec, current_ask_price_dec)
        adjusted_amount_dec = self.floor_adjust_amount(adjusted_amount_dec, quote_increment_dec)
        if adjusted_amount_dec <= 0:
            self.log_zero_amount_error(adjusted_amount_dec, from_asset, symbol)
            return 0, False
        if not self.check_available_balance(adjusted_amount_dec, available_dec):
            self.log_exceed_balance_error(adjusted_amount_dec, available_dec, from_asset)
            return 0, False
        return adjusted_amount_dec, True

    def adjust_buy_amount(self, funds, current_ask_price, base_min_size, base_increment, quote_increment, to_asset, symbol, available, from_asset, adjusted_amount_mode, exchange_client, best_op):
        self.wait_valid_buy_price(exchange_client, best_op, symbol)
        adjusted_base_amount_dec, success = self.adjust_buy_amount_step1(funds, current_ask_price, base_min_size, base_increment, to_asset, symbol)
        quote_increment_dec = Decimal(str(quote_increment))
        if not success:
            return 0, False
        adjusted_amount_dec, success = self.adjust_buy_amount_step2(adjusted_base_amount_dec, current_ask_price, quote_increment, from_asset, symbol, available)
        if not success:
            return 0, False
        self.log_adjusted_buy(adjusted_amount_dec, from_asset, quote_increment_dec, adjusted_base_amount_dec, to_asset, current_ask_price, adjusted_amount_mode)
        return float(adjusted_amount_dec), True

    def get_required_price(self, trades, symbol):
        for t in trades:
            if t[2] == symbol:
                return t[-1]
        return None

    def wait_valid_buy_price(self, exchange_client: ExchangeClient, best_op, symbol):
        actual_price = float(exchange_client.fetch_ticker_price(symbol)['buy'])
        required_price = self.get_required_price(best_op['trades'], symbol)

        while actual_price < required_price:
            print(f"actual: {str(actual_price)} and required: {str(required_price)}")
            actual_price = float(exchange_client.fetch_ticker_price(symbol)['buy'])
            required_price = self.get_required_price(best_op['trades'], symbol)
            sleep(0.01)

    def validate(self, funds, current_ask_price, base_min_size, base_increment, quote_increment, to_asset, symbol, available, from_asset, adjusted_amount_mode, exchange_client, best_op):
        return self.adjust_buy_amount(funds, current_ask_price, base_min_size, base_increment, quote_increment, to_asset, symbol, available, from_asset, adjusted_amount_mode, exchange_client, best_op)