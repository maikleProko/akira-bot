from decimal import Decimal

from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class ConstraintValidationPattern(ValidationPattern):
    def get_size_constraints(self, entry):
        return (
            Decimal(str(entry.get('base_min_size', 0.0))),
            Decimal(str(entry.get('quote_min_size', 0.0))),
            Decimal(str(entry.get('base_increment', 0.00000001))),
            Decimal(str(entry.get('quote_increment', 0.00000001)))
        )

    def get_constraints(self, symbol, price_map, expected_price):
        entry = price_map.get(symbol, {})
        sizes = self.get_size_constraints(entry)
        ask = Decimal(str(entry.get('ask', expected_price)))
        return sizes + (ask,)

    def get_cycle_constraints(self, sym, price_map):
        entry = price_map.get(sym, {})
        return self.get_size_constraints(entry)

    def validate(self, symbol, price_map, expected_price=None, cycle=False):
        if cycle:
            return self.get_cycle_constraints(symbol, price_map)
        return self.get_constraints(symbol, price_map, expected_price)
