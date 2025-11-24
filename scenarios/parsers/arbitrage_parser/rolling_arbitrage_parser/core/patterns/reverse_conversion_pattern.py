import math
from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.utils.patterns.conversion_pattern import ConversionPattern


class ReverseConversionPattern(ConversionPattern):
    def __init__(self, logger):
        self.logger = logger

    def get_reverse_symbol(self, to_asset, from_asset, symbol_map):
        return symbol_map.get((to_asset, from_asset))

    def get_price_entry(self, sym_rev, price_map):
        return price_map.get(sym_rev)

    def check_min_quote(self, amount, p):
        return amount >= p['quote_min_size']

    def calculate_quantity(self, amount, ask):
        return amount / ask

    def check_min_base(self, quantity, p):
        return quantity >= p['base_min_size']

    def calculate_new_amount(self, quantity, fee_rate):
        return quantity * (1 - fee_rate)

    def floor_increment(self, new_amount, inc):
        return math.floor(new_amount / inc) * inc

    def check_positive_amount(self, new_amount):
        return new_amount > 0

    def convert_reverse(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        sym_rev = self.get_reverse_symbol(to_asset, from_asset, symbol_map)
        if not sym_rev:
            return None, None, None, None
        p = self.get_price_entry(sym_rev, price_map)
        if not p:
            return None, None, None, None
        ask = p['ask']
        if not self.check_min_quote(amount, p):
            return None, None, None, None
        quantity = self.calculate_quantity(amount, ask)
        if not self.check_min_base(quantity, p):
            return None, None, None, None
        new_amount = self.calculate_new_amount(quantity, fee_rate)
        inc = p['base_increment']
        new_amount = self.floor_increment(new_amount, inc)
        if not self.check_positive_amount(new_amount):
            return None, None, None, None
        return new_amount, sym_rev, 'buy', ask

    def convert(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        return self.convert_reverse(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)