from scenarios.parsers.arbitrage_parser.careful_arbitrage_parser.core.utils.patterns.validation_pattern import ValidationPattern


class SymbolValidationPattern(ValidationPattern):
    def validate_symbol(self, symbol, price_map):
        if price_map and symbol not in price_map:
            return False
        return True

    def validate(self, symbol, price_map):
        return self.validate_symbol(symbol, price_map)
