from decimal import ROUND_FLOOR, Decimal, ROUND_DOWN
from time import sleep

from scenarios.parsers.arbitrage_parser.core.utils.exchange_client import ExchangeClient
from scenarios.parsers.arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class SellAmountAdjustmentPattern(ValidationPattern):
    def __init__(self, logger):
        self.logger = logger

    def simple_floor(self, f1, f2):
        d1 = Decimal(f1)
        d2 = Decimal(f2)
        if d2 <= 0:
            raise ValueError("f2 must be > 0")
        exp = d2.as_tuple().exponent
        if exp >= 0:
            raise ValueError("f2 must be 10**(-n), for example 0.01")
        # проверка, что f2 = exactly 10**(-n)
        if d2 != Decimal(1).scaleb(exp):
            raise ValueError("f2 должен быть точно 10**(-n) (например '0.01', '0.001')")
        return d1.quantize(d2, rounding=ROUND_DOWN)

    def floor_adjust_amount(self, amount_dec, base_increment_dec):
        return (amount_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * Decimal(base_increment_dec)

    def check_min_size(self, adjusted_amount_dec, base_min_size_dec):
        return adjusted_amount_dec >= base_min_size_dec

    def log_min_size_error(self, adjusted_amount_dec, base_min_size_dec, from_asset, symbol):
        self.logger.log_message(f"Ошибка: Сумма {adjusted_amount_dec} {from_asset} меньше минимального размера {base_min_size_dec} для {symbol}")

    def check_positive_amount(self, adjusted_amount_dec):
        return adjusted_amount_dec > 0

    def log_zero_amount_error(self, adjusted_amount_dec, from_asset, symbol):
        self.logger.log_message(f"Ошибка: После округления сумма {adjusted_amount_dec} {from_asset} равна нулю для {symbol}")

    def log_adjusted_amount(self, adjusted_amount_dec, from_asset, base_increment_dec, adjusted_amount_mode):
        self.logger.log_message(f"Скорректированная сумма для продажи: {adjusted_amount_dec} {from_asset} (base_increment: {base_increment_dec}), mode: {str(adjusted_amount_mode)}")

    def adjust_sell_amount(self, amount, base_min_size, base_increment, from_asset, symbol, adjusted_amount_mode, exchange_client, best_op):
        self.wait_valid_sell_price(exchange_client, best_op, symbol)
        adjusted_amount_dec = 0
        amount_dec = Decimal(str(amount))
        base_min_size_dec = Decimal(str(base_min_size))
        base_increment_dec = Decimal(str(base_increment))
        if adjusted_amount_mode == 0:
            adjusted_amount_dec = amount_dec

        if adjusted_amount_mode == 1:
            adjusted_amount_dec = self.simple_floor(amount_dec, base_increment)

        if adjusted_amount_mode > 1:
            adjusted_amount_dec = self.floor_adjust_amount(amount_dec, base_increment_dec)

            if not self.check_min_size(adjusted_amount_dec, base_min_size_dec):
                self.log_min_size_error(adjusted_amount_dec, base_min_size_dec, from_asset, symbol)
                return 0, False
        if not self.check_positive_amount(adjusted_amount_dec):
            self.log_zero_amount_error(adjusted_amount_dec, from_asset, symbol)
            return 0, False
        #self.log_adjusted_amount(adjusted_amount_dec, from_asset, base_increment_dec, adjusted_amount_mode)
        return Decimal(adjusted_amount_dec), True


    def get_required_price(self, trades, symbol):
        for t in trades:
            if t[2] == symbol:
                return t[-1]
        return None

    def wait_valid_sell_price(self, exchange_client: ExchangeClient, best_op, symbol):
        actual_price = float(exchange_client.fetch_ticker_price(symbol)['sell'])
        required_price = self.get_required_price(best_op['trades'], symbol)

        while actual_price > required_price * 0.99975:
            #print(f"actual: {str(actual_price)} and required: {str(required_price)}")
            actual_price = float(exchange_client.fetch_ticker_price(symbol)['sell'])
            required_price = self.get_required_price(best_op['trades'], symbol)
            sleep(0.01)

    def validate(self, amount, base_min_size, base_increment, from_asset, symbol, adjusted_amount_mode, exchange_client, best_op):
        return self.adjust_sell_amount(amount, base_min_size, base_increment, from_asset, symbol, adjusted_amount_mode, exchange_client, best_op)