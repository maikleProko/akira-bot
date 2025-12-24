from scenarios.market.regulators.regulator_tpsl import RegulatorTPSL
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.strategy import Strategy


class RegulatorNWEATR(RegulatorTPSL):
    """
    Самый точный расчёт размера позиции с учётом комиссий на вход и выход.
    Чистый профит при достижении TP = rr_ratio × risk_amount
    """
    def __init__(
        self,
        history_market_parser: HistoryMarketParser,
        nwe_bounds_indicator: NweBoundsIndicator,
        atr_bounds_indicator: AtrBoundsIndicator,
        strategy: Strategy,
        risk_amount: float = 0.01,       # сколько USDT рискуем на сделку (чистый убыток при SL)
        rr_ratio: float = 1.8,           # желаемое соотношение чистой прибыли к риску
        fee_rate: float = 0.0004,        # комиссия за одну сторону (0.04%)
        min_amount_step: float = 0.00001 # минимальный шаг размера позиции (для округления)
    ):
        super().__init__(history_market_parser, strategy, risk_amount, rr_ratio)
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.fee_rate = fee_rate
        self.min_amount_step = min_amount_step

    def calculate_tpsl(self):
        if self.history_market_parser.df is None or self.history_market_parser.df.empty:
            return

        # Текущая цена входа (обычно последняя закрытая свеча)
        entry_price = self.history_market_parser.df['close'].iloc[-1]

        # Консервативный стоп-лосс — самая нижняя граница из двух индикаторов
        atr_lower = self.atr_bounds_indicator.bounds.get('lower', entry_price)
        nwe_lower = self.nwe_bounds_indicator.bounds.get('lower', entry_price)
        sl_price = min(atr_lower, nwe_lower)

        if sl_price >= entry_price:
            return


        # Расстояние до стоп-лосса в цене
        sl_distance = entry_price - sl_price

        # Желаемое чистое соотношение
        desired_net_reward = self.rr_ratio * self.risk_amount

        # Тейк-профит в цене (пока без учёта комиссии — будет скорректирован позже)
        tp_distance_rough = self.rr_ratio * sl_distance
        tp_price_rough = entry_price + tp_distance_rough

        # ───────────────────────────────────────────────────────────────
        # Самый точный расчёт размера позиции
        # Пусть x = размер позиции в base-активе (BTC, ETH и т.п.)
        #
        # При SL:
        #   Чистый убыток = x × sl_distance + x × entry_price × fee + x × sl_price × fee
        #                 = x × (sl_distance + entry_price·fee + sl_price·fee)
        #   Должно быть ≈ risk_amount (по модулю)
        #
        # При TP:
        #   Чистая прибыль = x × tp_distance - x × entry_price × fee - x × tp_price × fee
        #                  = x × (tp_distance - entry_price·fee - tp_price·fee)
        #   Должно быть = desired_net_reward
        # ───────────────────────────────────────────────────────────────

        # Коэффициент комиссии на вход + выход
        fee_entry = entry_price * self.fee_rate
        fee_exit_sl = sl_price * self.fee_rate
        fee_exit_tp = tp_price_rough * self.fee_rate

        # Чистый убыток при SL на 1 единицу позиции
        loss_per_unit = sl_distance + fee_entry + fee_exit_sl

        # Размер позиции по риску (без учёта TP)
        amount_by_risk = self.risk_amount / loss_per_unit

        # ───────────────────────────────────────────────────────────────
        # Корректировка TP и amount, чтобы чистая прибыль была точно desired_net_reward
        # ───────────────────────────────────────────────────────────────

        # Итеративное уточнение (1-2 итерации достаточно)
        amount = amount_by_risk
        for _ in range(3):  # обычно хватает 2-х итераций
            tp_price = entry_price + self.rr_ratio * sl_distance  # базовый TP
            fee_exit_tp = tp_price * self.fee_rate

            # Чистая прибыль на 1 единицу позиции при TP
            profit_per_unit = (tp_price - entry_price) - fee_entry - fee_exit_tp

            # Корректируем размер позиции
            if profit_per_unit > 0:
                amount = desired_net_reward / profit_per_unit
            else:
                # крайне редкая ситуация (очень высокий TP и комиссии)
                amount = amount_by_risk
                break

            # Округляем до шага минимального лота
            amount = round(amount / self.min_amount_step) * self.min_amount_step

        # Финальный TP (уже с учётом последнего amount, но на практике разница минимальна)
        tp_distance = self.rr_ratio * sl_distance
        tp_price = entry_price + tp_distance

        # Сохраняем результаты
        self.stop_loss = sl_price
        self.take_profit = tp_price
        self.symbol1_prepared_converted_amount = amount
        self.is_accepted_by_regulator = True

        # Для отладки (можно закомментировать)
        # gross_profit = amount * (tp_price - entry_price)
        # total_fees = amount * (fee_entry + fee_exit_tp)
        # net_profit = gross_profit - total_fees
        # print(f"Amount: {amount:.6f} | Net P/L at TP: {net_profit:.2f} | Target: {desired_net_reward:.2f}")