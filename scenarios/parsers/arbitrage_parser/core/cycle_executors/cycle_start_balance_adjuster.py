from decimal import Decimal


class CycleStartBalanceAdjuster:
    def __init__(self, exchange_client, logger, production, use_all_balance, deposit):
        self.exchange_client = exchange_client
        self.logger = logger
        self.production = production
        self.use_all_balance = use_all_balance
        self.deposit = deposit

    def adjust_non_prod(self):
        return self.deposit, True

    def check_zero_balance(self, available_start, start_asset):
        if available_start == 0:
            self.logger.log_message(f"Баланс {start_asset} равен нулю, арбитраж невозможен.")
            return 0, False
        return None, True

    def use_all(self, available_start, start_asset):
        amt = available_start
        self.logger.log_message(f"Используем весь доступный баланс {start_asset}: {amt}")
        return float(amt)

    def use_min_deposit(self, available_start, start_asset):
        amt = min(available_start, Decimal(str(self.deposit)))
        if amt < Decimal(str(self.deposit)):
            self.logger.log_message(f"Недостаточно {start_asset} на балансе: доступно {available_start}, требуется {self.deposit}. Используем {amt}.")
        return float(amt)

    def adjust_prod(self, start_asset):
        available_start = Decimal(str(self.exchange_client.check_balance(start_asset)))
        _, success = self.check_zero_balance(available_start, start_asset)
        if not success:
            return 0, False
        if self.use_all_balance:
            return self.use_all(available_start, start_asset), True
        return self.use_min_deposit(available_start, start_asset), True

    def adjust_start_balance(self, start_asset):
        if not self.production:
            return self.adjust_non_prod()
        return self.adjust_prod(start_asset)