# RegulatorNWEATRNotStrict.py
from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class RegulatorNWEATRNotStrict(RegulatorTPSL):
    """
    Регулятор:
    - SL: чистый убыток РОВНО -risk_usdt USDT после всех комиссий
    - TP: расстояние от entry_price до TP = расстояние от entry_price до SL * ratio (коэффициент, напр. 1.5)
    - Опционально: если ожидаемая чистая прибыль на TP < min_profit_usdt, то отвергаем сделку
    """
    def __init__(
        self,
        history_market_parser: HistoryMarketParser,
        nwe_bounds_indicator: NweBoundsIndicator,
        atr_bounds_indicator: AtrBoundsIndicator,
        strategy: Strategy,
        risk_usdt: float = 10.0,           # ровно столько теряем чистыми на SL
        ratio: float = 1.5,                # коэффициент для TP относительно SL (TP distance = SL distance * ratio)
        min_profit_usdt: float = 12.0,      # если >0, минимум столько чистыми на TP, иначе отвергаем
        fee_rate: float = 0.001,           # комиссия за сторону (0.1%)
        min_amount_step: float = 0.00001
    ):
        super().__init__(history_market_parser, strategy, risk_usdt, None)
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.fee_rate = fee_rate
        self.min_amount_step = min_amount_step
        self.risk_usdt = risk_usdt
        self.ratio = ratio
        self.min_profit_usdt = min_profit_usdt

    def calculate_tpsl(self):
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return

        entry_price = self.history_market_parser.df['close'].iloc[-1]

        # SL — самая нижняя граница из индикаторов
        atr_lower = self.atr_bounds_indicator.bounds.get('lower', entry_price)
        nwe_lower = self.nwe_bounds_indicator.bounds.get('lower', entry_price)
        sl_price = min(atr_lower, nwe_lower)

        if sl_price >= entry_price or self.nwe_bounds_indicator.candle_count < 500 or sl_price <= 0:
            return

        fee = self.fee_rate
        sl_distance = entry_price - sl_price

        # Чистый убыток на 1 единицу base при SL
        net_loss_per_unit = sl_distance + fee * (entry_price + sl_price)

        if net_loss_per_unit <= 0:
            return

        # Размер позиции для РОВНО -risk_usdt на SL
        amount = self.risk_usdt / net_loss_per_unit

        # Округление вниз
        if self.min_amount_step > 0:
            amount = (amount // self.min_amount_step) * self.min_amount_step

        if amount <= 0:
            return

        # TP: расстояние TP = расстояние SL * ratio
        tp_distance = sl_distance * self.ratio
        tp_price = entry_price + tp_distance

        # Если min_profit_usdt > 0, проверяем ожидаемую чистую прибыль на TP
        if self.min_profit_usdt > 0:
            # net_pnl_on_tp = amount * (tp_price - entry_price) - fee * amount * (entry_price + tp_price)
            net_pnl_on_tp = amount * tp_distance - fee * amount * (entry_price + tp_price)
            if net_pnl_on_tp < self.min_profit_usdt:
                return  # Отвергаем, если меньше минимума

        # Сохраняем
        self.stop_loss = sl_price
        self.take_profit = tp_price
        self.symbol1_prepared_converted_amount = amount
        self.is_accepted_by_regulator = True