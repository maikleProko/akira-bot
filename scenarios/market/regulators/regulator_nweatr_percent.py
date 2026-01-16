from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class RegulatorNWEATRPercent(RegulatorTPSL):
    """
    Регулятор для мейкер-стратегии (комиссия только при закрытии позиции):
    - SL: чистый убыток РОВНО -risk_percent % от депозита после комиссии
    - TP: чистая прибыль МИНИМУМ +profit_percent % от депозита после комиссии
    """

    def __init__(
            self,
            history_market_parser: HistoryMarketParser,
            nwe_bounds_indicator: NweBoundsIndicator,
            atr_bounds_indicator: AtrBoundsIndicator,
            strategy: Strategy,
            balance_usdt: BalanceUSDT,  # размер депозита в USDT
            risk_percent: float = 1.0,  # % депозита под риск (SL)
            profit_percent: float = 1.5,  # минимальный % депозита на TP
            maker_fee_rate: float = 0.001,  # комиссия мейкера за сторону (пример: 0.02%)
            min_amount_step: float = 0.00001
    ):
        risk_usdt = balance_usdt.amount * (risk_percent / 100)
        min_profit_usdt = balance_usdt.amount * (profit_percent / 100)

        super().__init__(history_market_parser, strategy, risk_usdt, None)

        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.fee_rate = maker_fee_rate
        self.min_amount_step = min_amount_step
        self.risk_usdt = risk_usdt
        self.min_profit_usdt = min_profit_usdt
        self.balance_usdt = balance_usdt
        self.risk_percent = risk_percent
        self.profit_percent = profit_percent

    def calculate_tpsl(self):
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return

        entry_price = self.history_market_parser.df['close'].iloc[-1]

        # SL — самая нижняя граница из индикаторов
        atr_lower = self.atr_bounds_indicator.bounds.get('lower', entry_price)
        nwe_lower = self.nwe_bounds_indicator.bounds.get('lower', entry_price)
        sl_price = min(atr_lower, nwe_lower)

        if (sl_price >= entry_price or
                self.nwe_bounds_indicator.candle_count < 500 or
                sl_price <= 0):
            return

        fee = self.fee_rate
        sl_distance = entry_price - sl_price

        # ────────────────────────────────────────────────────────────────
        # Для мейкера: комиссия только при закрытии → fee × exit_price
        # ────────────────────────────────────────────────────────────────

        # Чистый убыток на 1 единицу base при срабатывании SL
        # = падение цены + комиссия при закрытии
        net_loss_per_unit = sl_distance + fee * sl_price

        if net_loss_per_unit <= 0:
            return

        # Размер позиции → чтобы чистый убыток был ровно risk_usdt
        amount = self.risk_usdt / net_loss_per_unit

        # Округление вниз до допустимого шага
        if self.min_amount_step > 0:
            amount = (amount // self.min_amount_step) * self.min_amount_step

        if amount <= 0:
            return

        # ────────────────────────────────────────────────────────────────
        # Take Profit — минимальная чистая прибыль = min_profit_usdt
        # net_profit = amount × (tp_price - entry) - fee × amount × tp_price
        #
        # amount × (tp_price - entry - fee × tp_price) = min_profit_usdt
        # amount × (tp_price × (1 - fee) - entry) = min_profit_usdt
        # tp_price × (1 - fee) = entry + min_profit_usdt / amount
        # tp_price = (entry + min_profit_usdt / amount) / (1 - fee)
        # ────────────────────────────────────────────────────────────────

        tp_price = (entry_price + self.min_profit_usdt / amount) / (1 - fee)

        # Сохраняем результаты
        self.stop_loss = sl_price
        self.take_profit = tp_price
        self.symbol1_prepared_converted_amount = amount
        self.is_accepted_by_regulator = True