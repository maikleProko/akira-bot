from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy


class RegulatorNWEATRMinimize(RegulatorTPSL):
    """
    Самый точный расчёт размера позиции с учётом комиссий на вход и выход.
    Чистый профит при достижении TP = rr_ratio × risk_amount
    """
    def __init__(
        self,
        history_market_parser_1m: HistoryMarketParser,
        history_market_parser_15m: HistoryMarketParser,
        nwe_bounds_indicator: NweBoundsIndicator,
        atr_bounds_indicator: AtrBoundsIndicator,
        strategy: Strategy,
        risk_amount: float = 10,       # сколько USDT рискуем на сделку (чистый убыток при SL)
        rr_ratio: float = 1.5,           # желаемое соотношение чистой прибыли к риску
        fee_rate: float = 0.0005,        # комиссия за одну сторону (0.04%)
        min_amount_step: float = 0.00001 # минимальный шаг размера позиции (для округления)
    ):
        super().__init__(history_market_parser_1m, strategy, risk_amount, rr_ratio)
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.fee_rate = fee_rate
        self.min_amount_step = min_amount_step
        self.history_market_parser=history_market_parser_1m

    def calculate_tpsl(self):
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return

        # Текущая цена входа (обычно последняя закрытая свеча)
        entry_price = self.history_market_parser.df['close'].iloc[-1]

        # Консервативный стоп-лосс — самая нижняя граница из двух индикаторов
        atr_lower = self.atr_bounds_indicator.bounds.get('lower', entry_price)
        nwe_lower = self.nwe_bounds_indicator.bounds.get('lower', entry_price)
        sl_price = min(atr_lower, nwe_lower)

        if sl_price >= entry_price or self.nwe_bounds_indicator.candle_count < 500:
            return

        # Расстояние до стоп-лосса в цене
        sl_distance = entry_price - sl_price

        # Коэффициент комиссии
        fee = self.fee_rate
        fee_entry = entry_price * fee
        fee_exit_sl = sl_price * fee

        # Чистый убыток при SL на 1 единицу позиции
        loss_per_unit = sl_distance + fee_entry + fee_exit_sl

        # Размер позиции по риску (чтобы net_loss == risk_amount)
        amount = self.risk_amount / loss_per_unit

        # Округление вниз до шага (консервативно, чтобы не превысить риск)
        if self.min_amount_step > 0:
            amount = (amount // self.min_amount_step) * self.min_amount_step

        # Если после округления amount слишком мал, пропустить
        if amount <= 0:
            return

        # Желаемая чистая прибыль при TP
        desired_net_reward = self.rr_ratio * self.risk_amount

        # Точный расчёт расстояния до TP для достижения desired_net_reward
        # Формула: d_tp = (desired / amount + 2 * fee * entry_price) / (1 - fee)
        d_tp = (desired_net_reward / amount + 2 * fee * entry_price) / (1 - fee)

        # Обеспечиваем минимальное геометрическое соотношение (d_tp / sl_distance >= min_rr)
        min_rr = 1.5  # Минимальное соотношение, как указано в запросе
        min_d_tp = min_rr * sl_distance
        d_tp = max(d_tp, min_d_tp)

        tp_price = entry_price + d_tp

        # Сохраняем результаты
        self.stop_loss = sl_price
        self.take_profit = tp_price
        self.symbol1_prepared_converted_amount = amount
        self.is_accepted_by_regulator = True

        # Для отладки (можно закомментировать)
        # fee_exit_tp = tp_price * fee
        # gross_profit = amount * d_tp
        # total_fees = amount * (fee_entry + fee_exit_tp)
        # net_profit = gross_profit - total_fees
        # print(f"Amount: {amount:.6f} | Net P/L at TP: {net_profit:.2f} | Target: {desired_net_reward:.2f}")