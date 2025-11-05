import math

from scenarios.parsers.arbitrage_parser.core.utils.patterns.conversion_pattern import ConversionPattern


class DirectConversionPattern(ConversionPattern):
    def __init__(self, logger):
        self.logger = logger

    def get_direct_symbol(self, from_asset, to_asset, symbol_map):
        return symbol_map.get((from_asset, to_asset))

    def get_price_entry(self, sym_direct, price_map):
        return price_map.get(sym_direct)

    def check_min_base(self, amount, p):
        return amount >= p['base_min_size']

    def calculate_new_amount(self, amount, bid, fee_rate):
        return amount * bid * (1 - fee_rate)

    def check_min_quote(self, new_amount, p):
        return new_amount >= p['quote_min_size']

    def floor_increment(self, new_amount, inc):
        return math.floor(new_amount / inc) * inc

    def check_positive_amount(self, new_amount):
        return new_amount > 0

    def convert_direct(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        sym_direct = self.get_direct_symbol(from_asset, to_asset, symbol_map)
        if not sym_direct:
            return None, None, None, None
        p = self.get_price_entry(sym_direct, price_map)
        if not p:
            return None, None, None, None
        bid = p['bid']
        if not self.check_min_base(amount, p):
            return None, None, None, None
        new_amount = self.calculate_new_amount(amount, bid, fee_rate)
        if not self.check_min_quote(new_amount, p):
            return None, None, None, None
        inc = p['quote_increment']
        new_amount = self.floor_increment(new_amount, inc)
        if not self.check_positive_amount(new_amount):
            return None, None, None, None
        return new_amount, sym_direct, 'sell', bid

    def convert(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        return self.convert_direct(amount, from_asset, to_asset, symbol_map, price_map, fee_rate)