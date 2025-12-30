from decimal import Decimal

from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class BalanceAdjustmentPattern(ValidationPattern):
    def __init__(self, logger):
        self.logger = logger

    def check_available_balance(self, available, amount_dec):
        return available >= amount_dec

    def log_insufficient_balance(self, from_asset, available, amount_dec):
        self.logger.log_message(f"Недостаточно {from_asset} на балансе: доступно {available}, требуется {amount_dec}. Используем весь доступный баланс.")

    def log_zero_balance(self, from_asset):
        self.logger.log_message(f"Баланс {from_asset} равен нулю, транзакция невозможна.")

    def adjust_balance_non_prod(self, amount):
        return amount, True

    def adjust_balance_prod(self, check_balance, from_asset, amount):
        available = Decimal(str(check_balance(from_asset)))
        amount_dec = Decimal(str(amount))
        if not self.check_available_balance(available, amount_dec):
            self.log_insufficient_balance(from_asset, available, amount_dec)
            amount_dec = available
            if amount_dec == 0:
                self.log_zero_balance(from_asset)
                return 0, False
        return float(amount_dec), True

    def adjust_balance(self, check_balance, from_asset, amount, production):
        if not production:
            return self.adjust_balance_non_prod(amount)
        return self.adjust_balance_prod(check_balance, from_asset, amount)

    def validate(self, check_balance, from_asset, amount, production):
        return self.adjust_balance(check_balance, from_asset, amount, production)