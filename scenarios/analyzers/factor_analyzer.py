from datetime import datetime
from scenarios.history_market.abstracts.history_market_parser import HistoryMarketParser
from scenarios.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.orderbook_patterns.orderbook_parser import OrderbookParser
from scenarios.reverberate.reverberate_parser import ReverberateParser
from utils.functions import MarketProcess


class FactorAnalyzer(MarketProcess):
    def __init__(self, history_market_parser: HistoryMarketParser, orderbook_parser: OrderbookParser, reverberate_parser: ReverberateParser, atr_bounds_indicator: AtrBoundsIndicator, nwe_bounds_indicator: NweBoundsIndicator):
        self.history_market_parser = history_market_parser
        self.orderbook_parser = orderbook_parser
        self.reverberate_parser = reverberate_parser
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.nwe_bounds = []
        self.atr_bounds = []
        self.avg_atr = -1
        self.prepare()

    def prepare(self):
        print(f"{datetime.now()}: FactorAnalyzer preparing...")
        # Инициализация, если нужно (например, загрузка начальных данных или индикаторов)
        # Предполагаем, что NWE и ATR будут переданы внешне
        pass

    def run(self):
        # Получаем текущие данные
        current_price = self._get_current_price()
        orderbook = self.orderbook_parser.orderbook
        reverberations = self.reverberate_parser.reverberations[-2:]  # Последние 2 реверберации (1-2 мин)
        self.nwe_bounds = self.nwe_bounds_indicator.bounds  # Предполагается {upper: float, lower: float}
        self.atr_bounds = self.atr_bounds_indicator.bounds  # Предполагается {upper: float, lower: float}
        self.avg_atr = (self.atr_bounds_indicator.bounds['lower'] + self.atr_bounds_indicator.bounds['upper'])/2     # Средний ATR для фильтра волатильности

        nwe_lower = self.nwe_bounds['lower'] if self.nwe_bounds else None
        nwe_upper = self.nwe_bounds['upper'] if self.nwe_bounds else None
        atr_lower = self.atr_bounds['lower'] if self.atr_bounds else None
        atr_upper = self.atr_bounds['upper'] if self.atr_bounds else None
        current_atr = self._calculate_current_atr()

        if not (current_price and orderbook and reverberations and nwe_lower and atr_lower):
            print(f"{datetime.now()}: Missing data for analysis")
            return

        # Фильтр волатильности: Пропускаем, если ATR слишком низкий
        if self.avg_atr and current_atr < 0.8 * self.avg_atr:
            return

        # Расчёт факторов для long
        long_conditions = self._check_long_conditions(current_price, nwe_lower, atr_lower, orderbook, reverberations)
        # Расчёт факторов для short
        short_conditions = self._check_short_conditions(current_price, nwe_upper, atr_upper, orderbook, reverberations)

        # Вывод сигнала, если условия выполнены
        if long_conditions:
            print(f"{datetime.now()}: Enter LONG trade at price {current_price}")
        if short_conditions:
            print(f"{datetime.now()}: Enter SHORT trade at price {current_price}")

    def _get_current_price(self):
        """Получить текущую цену из history_market_parser.df (последняя close)"""
        if hasattr(self.history_market_parser, 'df') and not self.history_market_parser.df.empty:
            return float(self.history_market_parser.df.iloc[-1]['close'])
        return None

    def _calculate_current_atr(self):
        """Рассчитать текущий ATR (примерная реализация, если не передан)"""
        if hasattr(self.history_market_parser, 'df') and len(self.history_market_parser.df) >= 14:
            df = self.history_market_parser.df[-14:]
            high_low = df['high'].astype(float) - df['low'].astype(float)
            high_close = abs(df['high'].astype(float) - df['close'].astype(float).shift(1))
            low_close = abs(df['low'].astype(float) - df['close'].astype(float).shift(1))
            true_range = high_low.combine(high_close, max).combine(low_close, max)
            current_atr = true_range.mean()
            self.avg_atr = current_atr if not self.avg_atr else 0.9 * self.avg_atr + 0.1 * current_atr
            return current_atr
        return None

    def _check_long_conditions(self, price, nwe_lower, atr_lower, orderbook, reverberations):
        """Проверка условий для входа в LONG"""
        # Условие 1: Цена у зоны поддержки
        if price > min(nwe_lower, atr_lower):
            return False

        # Условие 2: Реверберация (buy pressure)
        latest_reverb = reverberations[-1] if len(reverberations) >= 1 else None
        prev_reverb = reverberations[-2] if len(reverberations) >= 2 else None
        if not latest_reverb or not prev_reverb:
            return False
        if (latest_reverb.get('reverberation_75_1_minutes', 0) <= 0.4 or
            prev_reverb.get('reverberation_50_2_minutes', 0) <= 0.5):
            return False

        # Условие 3: Order book дисбаланс
        bid_volume = sum([bid['amount'] for bid in orderbook['bids'][:10]])
        ask_volume = sum([ask['amount'] for ask in orderbook['asks'][:10]])
        bid_ask_ratio = bid_volume / ask_volume if ask_volume > 0 else float('inf')
        avg_depth = self._calculate_avg_depth(orderbook)
        current_bid_depth = bid_volume / avg_depth if avg_depth > 0 else 0
        if bid_ask_ratio <= 1.5 or current_bid_depth <= 0.5:
            return False

        return True

    def _check_short_conditions(self, price, nwe_upper, atr_upper, orderbook, reverberations):
        """Проверка условий для входа в SHORT"""
        # Условие 1: Цена у зоны сопротивления
        if price < max(nwe_upper, atr_upper):
            return False

        # Условие 2: Реверберация (sell pressure)
        latest_reverb = reverberations[-1] if len(reverberations) >= 1 else None
        prev_reverb = reverberations[-2] if len(reverberations) >= 2 else None
        if not latest_reverb or not prev_reverb:
            return False
        if (latest_reverb.get('sell_reverberation_75_1_minutes', 0) <= 0.4 or
            prev_reverb.get('sell_reverberation_50_2_minutes', 0) <= 0.5):
            return False

        # Условие 3: Order book дисбаланс
        bid_volume = sum([bid['amount'] for bid in orderbook['bids'][:10]])
        ask_volume = sum([ask['amount'] for ask in orderbook['asks'][:10]])
        ask_bid_ratio = ask_volume / bid_volume if bid_volume > 0 else float('inf')
        avg_depth = self._calculate_avg_depth(orderbook)
        current_ask_depth = ask_volume / avg_depth if avg_depth > 0 else 0
        if ask_bid_ratio <= 1.5 or current_ask_depth <= 0.5:
            return False

        return True

    def _calculate_avg_depth(self, orderbook):
        """Рассчитать среднюю глубину order book за последние 5 мин (примерно)"""
        # Предполагаем, что depth сохраняется где-то в orderbook_parser (иначе нужна история)
        bid_volume = sum([bid['amount'] for bid in orderbook['bids'][:10]])
        ask_volume = sum([ask['amount'] for ask in orderbook['asks'][:10]])
        return (bid_volume + ask_volume) / 2

    def set_indicators(self, nwe_bounds: dict, atr_bounds: dict):
        """Установить значения индикаторов NWE и ATR"""
        self.nwe_bounds = nwe_bounds
        self.atr_bounds = atr_bounds