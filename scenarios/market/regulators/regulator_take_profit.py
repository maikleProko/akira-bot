# RegulatorTakeProfit.py
from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.strategies.abstracts.strategy import Strategy
class RegulatorTakeProfit(RegulatorTPSL):
    """
    FIXED PnL REGULATOR
    Якорь = TP индикатора
    TP → +50
    SL → -30
    Модель комиссии: как в RegulatorNWEATR
    """
    def __init__(
        self,
        history_market_parser,
        take_profit_indicator,
        strategy,
        risk_usdt=30.0,
        min_profit_usdt=50.0,
        fee_rate=0.001,
        min_amount_step=0.00001
    ):
        super().__init__(history_market_parser, strategy, risk_usdt, None)
        self.take_profit_indicator = take_profit_indicator
        self.risk_usdt = risk_usdt
        self.min_profit_usdt = min_profit_usdt
        self.fee_rate = fee_rate
        self.min_amount_step = min_amount_step
    def calculate_tpsl(self):
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return
        entry = self.history_market_parser.df['close'].iloc[-1]
        tp = self.take_profit_indicator.take_profit
        if tp is None or tp <= entry:
            return
        fee = self.fee_rate
        # ---------- amount ИЗ TP (якорь) ----------
        net_profit_per_unit = (tp - entry) - fee * (entry + tp)
        if net_profit_per_unit <= 0:
            return
        amount = self.min_profit_usdt / net_profit_per_unit
        if self.min_amount_step > 0:
            amount = (amount // self.min_amount_step) * self.min_amount_step
        if amount <= 0:
            return
        # ---------- SL ИЗ -30 ----------
        sl = (
            entry * (1 + self.fee_rate)
            - self.risk_usdt / amount
        ) / (1 - self.fee_rate)
        if sl <= 0 or sl >= entry:
            return # контракт невозможен
        # ---------- ACCEPT ----------
        self.take_profit = tp
        self.stop_loss = sl
        self.symbol1_prepared_converted_amount = amount
        self.is_accepted_by_regulator = True