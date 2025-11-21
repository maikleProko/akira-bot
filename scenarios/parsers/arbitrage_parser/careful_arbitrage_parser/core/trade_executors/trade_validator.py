from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.trade_executors.patterns.balance_adjustment_pattern import \
    BalanceAdjustmentPattern
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.trade_executors.patterns.buy_amount_adjustment_pattern import \
    BuyAmountAdjustmentPattern
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.trade_executors.patterns.buy_funds_adjustment_pattern import \
    BuyFundsAdjustmentPattern
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.trade_executors.patterns.constraint_validation_pattern import \
    ConstraintValidationPattern
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.trade_executors.patterns.sell_amount_adjustment_pattern import \
    SellAmountAdjustmentPattern
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.trade_executors.patterns.sim_trade_validation_pattern import \
    SimTradeValidationPattern
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.trade_executors.patterns.symbol_validation_pattern import \
    SymbolValidationPattern
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.trade_executors.patterns.trade_constraint_validation_pattern import \
    TradeConstraintValidationPattern
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.utils.logger import Logger


class TradeValidator:
    def __init__(self, exchange_client):
        self.symbol_strategy = SymbolValidationPattern()
        self.constraint_strategy = ConstraintValidationPattern()
        self.trade_constraint_strategy = TradeConstraintValidationPattern()
        self.sim_trade_strategy = SimTradeValidationPattern(Logger())
        self.best_op = None
        self.exchange_client = exchange_client

    def validate_symbol(self, symbol, price_map):
        return self.symbol_strategy.validate(symbol, price_map)

    def get_size_constraints(self, entry):
        return self.constraint_strategy.get_size_constraints(entry)

    def get_constraints(self, symbol, price_map, expected_price):
        return self.constraint_strategy.validate(symbol, price_map, expected_price)

    def get_cycle_constraints(self, sym, price_map):
        return self.constraint_strategy.validate(sym, price_map, cycle=True)

    def adjust_balance(self, check_balance, log_message, from_asset, amount, production):
        strategy = BalanceAdjustmentPattern(Logger())
        return strategy.validate(check_balance, from_asset, amount, production)

    def adjust_sell_amount(self, log_message, amount, base_min_size, base_increment, from_asset, symbol, adjusted_amount_mode):
        strategy = SellAmountAdjustmentPattern(Logger())
        return strategy.validate(amount, base_min_size, base_increment, from_asset, symbol, adjusted_amount_mode, self.exchange_client, self.best_op)

    def adjust_buy_funds(self, log_message, amount, available, quote_min_size, from_asset, symbol):
        strategy = BuyFundsAdjustmentPattern(Logger())
        return strategy.validate(amount, available, quote_min_size, from_asset, symbol)

    def adjust_buy_amount(self, log_message, funds, current_ask_price, base_min_size, base_increment, quote_increment, to_asset, symbol, available, from_asset, adjusted_amount_mode):
        strategy = BuyAmountAdjustmentPattern(Logger())
        return strategy.validate(funds, current_ask_price, base_min_size, base_increment, quote_increment, to_asset, symbol, available, from_asset, adjusted_amount_mode, self.exchange_client, self.best_op)

    def validate_trade_constraints(self, amt, direction, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate, frm, to, sym):
        return self.trade_constraint_strategy.validate(amt, direction, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate, frm, to, sym)

    def validate_sim_trade(self, cycle_amt, direction, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee, frm, sym, to, log_message):
        return self.sim_trade_strategy.validate(cycle_amt, direction, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee, frm, sym, to)