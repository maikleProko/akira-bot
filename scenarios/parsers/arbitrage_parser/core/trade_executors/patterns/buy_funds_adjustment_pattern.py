from decimal import Decimal

from scenarios.parsers.arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class BuyFundsAdjustmentPattern(ValidationPattern):
    def __init__(self, logger):
        self.logger = logger

    def min_funds(self, amount_dec, available_dec):
        return min(amount_dec, available_dec)

    def check_min_size(self, funds_dec, quote_min_size_dec):
        return funds_dec >= quote_min_size_dec

    def log_min_size_error(self, funds_dec, quote_min_size_dec, from_asset, symbol):
        self.logger.log_message(f"Ошибка: Сумма {funds_dec} {from_asset} меньше минимального размера {quote_min_size_dec} для {symbol}")

    def adjust_buy_funds(self, amount, available, quote_min_size, from_asset, symbol):
        amount_dec = Decimal(str(amount))
        available_dec = Decimal(str(available))
        quote_min_size_dec = Decimal(str(quote_min_size))
        funds_dec = self.min_funds(amount_dec, available_dec)
        if not self.check_min_size(funds_dec, quote_min_size_dec):
            self.log_min_size_error(funds_dec, quote_min_size_dec, from_asset, symbol)
            return 0, False
        return float(funds_dec), True

    def validate(self, amount, available, quote_min_size, from_asset, symbol):
        return self.adjust_buy_funds(amount, available, quote_min_size, from_asset, symbol)