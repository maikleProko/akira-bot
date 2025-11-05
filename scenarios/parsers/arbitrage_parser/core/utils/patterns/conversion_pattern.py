from abc import ABC, abstractmethod


class ConversionPattern(ABC):
    @abstractmethod
    def convert(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        pass
