# RegulatorNWEATR.py
from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class RegulatorNWEATR(RegulatorTPSL):
    """
    Регулятор:
    - SL: чистый убыток РОВНО -10 USDT (или risk_usdt) после всех комиссий
    - TP: чистая прибыль МИНИМУМ +15 USDT (или min_profit_usdt), а чаще больше — по close свечи
    """
    def __init__(
        self,
        history_market_parser: HistoryMarketParser,
        nwe_bounds_indicator: NweBoundsIndicator,
        atr_bounds_indicator: AtrBoundsIndicator,
        strategy: Strategy,
        risk_usdt: float = 30.0,           # ровно столько теряем чистыми на SL
        min_profit_usdt: float = 50.0,      # минимум столько зарабатываем чистыми на TP
        fee_rate: float = 0.001,            # комиссия за сторону (0.1%)
        min_amount_step: float = 0.00001
    ):
        super().__init__(history_market_parser, strategy, risk_usdt, None)
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.fee_rate = fee_rate
        self.min_amount_step = min_amount_step
        self.risk_usdt = risk_usdt
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

        # Теперь TP: расстояние, чтобы чистая прибыль была МИНИМУМ min_profit_usdt
        # net_profit = amount * (tp_price - entry_price) - amount * fee * (entry_price + tp_price)
        # tp_price - entry_price = (min_profit_usdt / amount + fee * (entry_price + tp_price)) / (1 - fee)
        # Упрощаем и решаем уравнение
        approx_tp_distance = (self.min_profit_usdt / amount + fee * entry_price * 2) / (1 - fee)
        tp_price = entry_price + approx_tp_distance

        # Сохраняем
        self.stop_loss = sl_price
        self.take_profit = tp_price
        self.symbol1_prepared_converted_amount = amount
        self.is_accepted_by_regulator = True