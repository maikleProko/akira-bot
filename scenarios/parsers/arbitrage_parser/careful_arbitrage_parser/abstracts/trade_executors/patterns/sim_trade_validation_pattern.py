from decimal import Decimal, ROUND_FLOOR

from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class SimTradeValidationPattern(ValidationPattern):
    def __init__(self, logger):
        self.logger = logger

    def to_decimal(self, value):
        return Decimal(str(value))

    def init_sim_decimals(self, cycle_amt, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee):
        cycle_amt_dec = self.to_decimal(cycle_amt)
        current_price_dec = self.to_decimal(current_price)
        base_min_size_dec = self.to_decimal(base_min_size)
        quote_min_size_dec = self.to_decimal(quote_min_size)
        base_increment_dec = self.to_decimal(base_increment)
        quote_increment_dec = self.to_decimal(quote_increment)
        fee_dec = self.to_decimal(fee)
        return cycle_amt_dec, current_price_dec, base_min_size_dec, quote_min_size_dec, base_increment_dec, quote_increment_dec, fee_dec

    def adjust_sim_sell(self, cycle_amt_dec, base_increment_dec, base_min_size_dec, frm, sym):
        adjusted_amount_dec = (cycle_amt_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
        if adjusted_amount_dec < base_min_size_dec:
            self.logger.log_message(f"Симуляция: Сумма {adjusted_amount_dec} {frm} меньше минимального размера {base_min_size_dec} для {sym}")
            return None
        return adjusted_amount_dec

    def calculate_expected_sell(self, adjusted_amount_dec, current_price_dec, fee_dec, quote_min_size_dec, to, sym):
        expected_new_amt_dec = adjusted_amount_dec * current_price_dec * (Decimal('1') - fee_dec)
        if expected_new_amt_dec < quote_min_size_dec:
            self.logger.log_message(f"Симуляция: Ожидаемое количество {expected_new_amt_dec} {to} меньше минимального размера {quote_min_size_dec} для {sym}")
            return None
        return expected_new_amt_dec

    def validate_sim_sell(self, cycle_amt_dec, current_price_dec, base_min_size_dec, quote_min_size_dec, base_increment_dec, fee_dec, frm, sym, to):
        adjusted_amount_dec = self.adjust_sim_sell(cycle_amt_dec, base_increment_dec, base_min_size_dec, frm, sym)
        if adjusted_amount_dec is None:
            return None
        return self.calculate_expected_sell(adjusted_amount_dec, current_price_dec, fee_dec, quote_min_size_dec, to, sym)

    def calculate_max_base_sim_buy(self, cycle_amt_dec, current_price_dec):
        return cycle_amt_dec / current_price_dec

    def adjust_base_sim_buy(self, max_base_amount_dec, base_increment_dec, base_min_size_dec, to, sym):
        adjusted_base_amount_dec = (max_base_amount_dec / base_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * base_increment_dec
        if adjusted_base_amount_dec < base_min_size_dec:
            #self.logger.log_message(f"Симуляция: Скорректированное количество {adjusted_base_amount_dec} {to} меньше минимального размера {base_min_size_dec} для {sym}")
            return None
        return adjusted_base_amount_dec

    def calculate_adjusted_sim_buy(self, adjusted_base_amount_dec, current_price_dec):
        return adjusted_base_amount_dec * current_price_dec

    def floor_adjusted_sim_buy(self, adjusted_amount_dec, quote_increment_dec, quote_min_size_dec, frm, sym):
        adjusted_amount_dec = (adjusted_amount_dec / quote_increment_dec).to_integral_value(rounding=ROUND_FLOOR) * quote_increment_dec
        if adjusted_amount_dec < quote_min_size_dec:
            #self.logger.log_message(f"Симуляция: Скорректированная сумма {adjusted_amount_dec} {frm} меньше минимального размера {quote_min_size_dec} для {sym}")
            return None
        return adjusted_amount_dec

    def calculate_expected_buy(self, adjusted_base_amount_dec, fee_dec):
        return adjusted_base_amount_dec * (Decimal('1') - fee_dec)

    def validate_sim_buy(self, cycle_amt_dec, current_price_dec, base_min_size_dec, quote_min_size_dec, base_increment_dec, quote_increment_dec, fee_dec, to, sym, frm):
        max_base_amount_dec = self.calculate_max_base_sim_buy(cycle_amt_dec, current_price_dec)
        adjusted_base_amount_dec = self.adjust_base_sim_buy(max_base_amount_dec, base_increment_dec, base_min_size_dec, to, sym)
        if adjusted_base_amount_dec is None:
            return None
        adjusted_amount_dec = self.calculate_adjusted_sim_buy(adjusted_base_amount_dec, current_price_dec)
        adjusted_amount_dec = self.floor_adjusted_sim_buy(adjusted_amount_dec, quote_increment_dec, quote_min_size_dec, frm, sym)
        if adjusted_amount_dec is None:
            return None
        return self.calculate_expected_buy(adjusted_base_amount_dec, fee_dec)

    def validate_sim_trade(self, cycle_amt, direction, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee, frm, sym, to):
        decimals = self.init_sim_decimals(cycle_amt, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee)
        if direction == 'sell':
            expected_new_amt_dec = self.validate_sim_sell(*decimals[:-1], frm, sym, to)
        else:
            expected_new_amt_dec = self.validate_sim_buy(*decimals, to, sym, frm)
        if expected_new_amt_dec is None:
            return None
        return float(expected_new_amt_dec)

    def validate(self, cycle_amt, direction, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee, frm, sym, to):
        return self.validate_sim_trade(cycle_amt, direction, current_price, base_min_size, quote_min_size, base_increment, quote_increment, fee, frm, sym, to)