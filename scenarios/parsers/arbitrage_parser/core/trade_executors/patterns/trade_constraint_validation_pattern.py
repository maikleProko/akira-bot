from decimal import Decimal, ROUND_FLOOR

from scenarios.parsers.arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class TradeConstraintValidationPattern(ValidationPattern):
    def to_decimal(self, value):
        return Decimal(str(value))

    def init_decimals_sell(self, amt, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate):
        amt_dec = self.to_decimal(amt)
        price_dec = self.to_decimal(price)
        base_min_size_dec = self.to_decimal(base_min_size)
        quote_min_size_dec = self.to_decimal(quote_min_size)
        base_increment_dec = self.to_decimal(base_increment)
        quote_increment_dec = self.to_decimal(quote_increment)
        fee_dec = self.to_decimal(fee_rate)
        return amt_dec, price_dec, base_min_size_dec, quote_min_size_dec, base_increment_dec, quote_increment_dec, fee_dec

    def adjust_sell_amount(self, amt_dec, base_increment_dec, base_min_size_dec):
        adjusted_amount_dec = (amt_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
        if adjusted_amount_dec < base_min_size_dec:
            return None
        return adjusted_amount_dec

    def calculate_new_amt_sell(self, adjusted_amount_dec, price_dec, fee_dec, quote_min_size_dec):
        new_amt_dec = adjusted_amount_dec * price_dec * (Decimal('1') - fee_dec)
        if new_amt_dec < quote_min_size_dec:
            return None
        return new_amt_dec

    def floor_new_amt_sell(self, new_amt_dec, quote_increment_dec):
        new_amt_dec = (new_amt_dec / quote_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * quote_increment_dec
        if new_amt_dec <= 0:
            return None
        return new_amt_dec

    def validate_trade_sell(self, amt_dec, price_dec, base_min_size_dec, quote_min_size_dec, base_increment_dec, quote_increment_dec, fee_dec):
        adjusted_amount_dec = self.adjust_sell_amount(amt_dec, base_increment_dec, base_min_size_dec)
        if adjusted_amount_dec is None:
            return None
        new_amt_dec = self.calculate_new_amt_sell(adjusted_amount_dec, price_dec, fee_dec, quote_min_size_dec)
        if new_amt_dec is None:
            return None
        new_amt_dec = self.floor_new_amt_sell(new_amt_dec, quote_increment_dec)
        if new_amt_dec is None:
            return None
        return float(new_amt_dec)

    def init_decimals_buy(self, amt, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate):
        return self.init_decimals_sell(amt, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate)

    def calculate_max_base_buy(self, amt_dec, price_dec):
        return amt_dec / price_dec

    def adjust_base_buy(self, max_base_amount_dec, base_increment_dec, base_min_size_dec):
        adjusted_base_amount_dec = (max_base_amount_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
        if adjusted_base_amount_dec < base_min_size_dec:
            return None
        return adjusted_base_amount_dec

    def calculate_adjusted_amount_buy(self, adjusted_base_amount_dec, price_dec):
        return adjusted_base_amount_dec * price_dec

    def floor_adjusted_buy(self, adjusted_amount_dec, quote_increment_dec, quote_min_size_dec):
        adjusted_amount_dec = (adjusted_amount_dec / quote_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * quote_increment_dec
        if adjusted_amount_dec < quote_min_size_dec:
            return None
        return adjusted_amount_dec

    def calculate_new_amt_buy(self, adjusted_base_amount_dec, fee_dec):
        return adjusted_base_amount_dec * (Decimal('1') - fee_dec)

    def validate_trade_buy(self, amt_dec, price_dec, base_min_size_dec, quote_min_size_dec, base_increment_dec, quote_increment_dec, fee_dec):
        max_base_amount_dec = self.calculate_max_base_buy(amt_dec, price_dec)
        adjusted_base_amount_dec = self.adjust_base_buy(max_base_amount_dec, base_increment_dec, base_min_size_dec)
        if adjusted_base_amount_dec is None:
            return None
        adjusted_amount_dec = self.calculate_adjusted_amount_buy(adjusted_base_amount_dec, price_dec)
        adjusted_amount_dec = self.floor_adjusted_buy(adjusted_amount_dec, quote_increment_dec, quote_min_size_dec)
        if adjusted_amount_dec is None:
            return None
        new_amt_dec = self.calculate_new_amt_buy(adjusted_base_amount_dec, fee_dec)
        return float(new_amt_dec)

    def validate_trade_constraints(self, amt, direction, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate, frm, to, sym):
        if direction == 'sell':
            decimals = self.init_decimals_sell(amt, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate)
            return self.validate_trade_sell(*decimals)
        else:
            decimals = self.init_decimals_buy(amt, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate)
            return self.validate_trade_buy(*decimals)

    def validate(self, amt, direction, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate, frm, to, sym):
        return self.validate_trade_constraints(amt, direction, price, base_min_size, quote_min_size, base_increment, quote_increment, fee_rate, frm, to, sym)