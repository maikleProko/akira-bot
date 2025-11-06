from decimal import ROUND_FLOOR, Decimal

from scenarios.parsers.arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class SellAmountAdjustmentPattern(ValidationPattern):
    def __init__(self, logger):
        self.logger = logger

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

    def log_adjusted_amount(self, adjusted_amount_dec, from_asset, base_increment_dec):
        self.logger.log_message(f"Скорректированная сумма для продажи: {adjusted_amount_dec} {from_asset} (base_increment: {base_increment_dec})")

    def adjust_sell_amount(self, amount, base_min_size, base_increment, from_asset, symbol):
        amount_dec = Decimal(str(amount))
        base_min_size_dec = Decimal(str(base_min_size))
        base_increment_dec = Decimal(str(base_increment))
        adjusted_amount_dec = self.floor_adjust_amount(amount_dec, base_increment_dec)
        if not self.check_min_size(adjusted_amount_dec, base_min_size_dec):
            self.log_min_size_error(adjusted_amount_dec, base_min_size_dec, from_asset, symbol)
            return 0, False
        if not self.check_positive_amount(adjusted_amount_dec):
            self.log_zero_amount_error(adjusted_amount_dec, from_asset, symbol)
            return 0, False
        self.log_adjusted_amount(adjusted_amount_dec, from_asset, base_increment_dec)
        return Decimal(adjusted_amount_dec), True

    def validate(self, amount, base_min_size, base_increment, from_asset, symbol):
        return self.adjust_sell_amount(amount, base_min_size, base_increment, from_asset, symbol)