import numpy as np
import pandas as pd

from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator

class CHoCHIndicator(Indicator):
    """
    Детектор bullish CHoCH (смена структуры на покупку) по логике LuxAlgo SMC
    для swing-структуры (internal = false).

    Идея строго следует PineScript-фрагментам:

        - leg(size)
        - getCurrentStructure(size, equalHighLow=false, internal=false)
        - displayStructure(internal=false)

    Условие bullish CHoCH (swing):

        pivot p_ivot    = swingHigh
        trend t_rend    = swingTrend

        if ta.crossover(close, p_ivot.currentLevel) and not p_ivot.crossed
            string tag = t_rend.bias == BEARISH ? CHOCH : BOS
            ...
            t_rend.bias := BULLISH

    Здесь:
    - при проходе всей истории бар за баром обновляем swingHigh / swingLow
      по leg(size) + getCurrentStructure()
    - отслеживаем swing-тренд (bias) и флаг p_ivot.crossed
    - на последней свече проверяем, случился ли bullish CHOCH (tag == "CHOCH").
    """

    # leg() в Pine: "it can be 0 (bearish) or 1 (bullish)"
    _BEARISH_LEG = 0
    _BULLISH_LEG = 1

    # Bias тренда (аналог t_rend.bias в Pine)
    _BULLISH = "BULLISH"
    _BEARISH = "BEARISH"

    def __init__(self, history_market_parser):
        self.history_market_parser = history_market_parser
        self.is_now_CHoCH = False

        # Аналог swingStructureSizeInput (size в Pine)
        # Можно переопределить снаружи: indicator.swing_size = <значение из TV>
        self.swing_size = 5

    def run(self, start_time=None, end_time=None):
        """
        На каждом вызове:
        - берём весь df из history_market_parser.df (с уже добавленной последней свечой),
        - бар за баром эмулируем PineScript:
            * leg(size) + getCurrentStructure(size, false, false)
            * displayStructure(false) — только часть с bullish CHOCH/BOS
        - на последней свече истории выставляем:
            self.is_now_CHoCH = True, если там был bullish CHOCH (swing)
        """

        df = self.history_market_parser.df

        # По умолчанию считаем, что сигнала на последней свече нет
        self.is_now_CHoCH = False

        if df is None or df.empty:
            return

        # При желании можно учитывать start_time / end_time, но по задаче
        # просто работаем по всей истории df
        n = len(df)
        size = int(self.swing_size)

        # Недостаточно данных для построения хотя бы одного свинга
        if size < 1 or n <= size + 1:
            return

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()

        # ------------------------- Состояние swing‑pivotов и тренда -------------------------

        # pivot swingHigh
        swing_high_current = None
        swing_high_last = None
        swing_high_crossed = False  # p_ivot.crossed
        swing_high_index = None     # p_ivot.barIndex

        # pivot swingLow
        swing_low_current = None
        swing_low_last = None
        swing_low_crossed = False
        swing_low_index = None

        # Аналог leg() с var leg; здесь leg_prev — его сохранённое состояние
        leg_prev = self._BEARISH_LEG

        # Аналог swingTrend.bias.
        # Инициализируем как BEARISH, чтобы первый пробой хая дал CHOCH.
        trend_bias = self._BEARISH

        # ------------------------- Прогон истории бар за баром -------------------------

        for i in range(n):
            # 1) Обновление leg() и swing‑pivotов (getCurrentStructure)

            if i >= size:
                # В Pine:
                #   newLegHigh = high[size] > ta.highest(size)
                #   newLegLow  = low[size]  < ta.lowest(size)
                #
                # high[size] / low[size] — это бар i-size.
                # ta.highest/lowest(size) — экстремум по последним `size` барам,
                # т.е. окну (i-size+1 .. i). Таким образом, pivot — бар i-size,
                # а окно — бары после него.
                pivot_idx = i - size

                if pivot_idx + 1 <= i:
                    window_high = highs[pivot_idx + 1 : i + 1]
                    window_low = lows[pivot_idx + 1 : i + 1]

                    if window_high.size == 0 or window_low.size == 0:
                        new_leg_high = False
                        new_leg_low = False
                    else:
                        new_leg_high = highs[pivot_idx] > window_high.max()
                        new_leg_low = lows[pivot_idx] < window_low.min()
                else:
                    new_leg_high = False
                    new_leg_low = False

                # leg():
                #   var leg = 0
                #   if newLegHigh -> leg := BEARISH_LEG
                #   else if newLegLow -> leg := BULLISH_LEG
                leg_val = leg_prev
                if new_leg_high:
                    leg_val = self._BEARISH_LEG
                elif new_leg_low:
                    leg_val = self._BULLISH_LEG

                # startOfNewLeg / startOfBullishLeg / startOfBearishLeg
                change = leg_val - leg_prev
                new_pivot = change != 0
                pivot_low = change == +1   # startOfBullishLeg
                pivot_high = change == -1  # startOfBearishLeg

                # getCurrentStructure(size, equalHighLow=false, internal=false)
                if new_pivot:
                    if pivot_low:
                        # Обновляем swingLow pivot
                        swing_low_last = swing_low_current
                        swing_low_current = lows[pivot_idx]
                        swing_low_crossed = False
                        swing_low_index = pivot_idx
                        # trailing / equalHighLow / визуализация здесь опускаются, на CHOCH не влияют
                    else:
                        # Обновляем swingHigh pivot
                        swing_high_last = swing_high_current
                        swing_high_current = highs[pivot_idx]
                        swing_high_crossed = False
                        swing_high_index = pivot_idx

                leg_prev = leg_val

            # 2) displayStructure(false) — только логика CHOCH / BOS по swingHigh / swingLow

            # Bullish часть: пробой swingHigh вверх
            # В Pine:
            #   p_ivot = swingHigh
            #   if ta.crossover(close, p_ivot.currentLevel) and not p_ivot.crossed
            #       string tag = t_rend.bias == BEARISH ? CHOCH : BOS
            #       ...
            #       t_rend.bias := BULLISH
            if (
                swing_high_current is not None
                and i > 0
                and not swing_high_crossed
            ):
                level = swing_high_current
                prev_close = closes[i - 1]
                cur_close = closes[i]

                # ta.crossover(close, level)  ~  close[1] < level and close > level
                crossed_up = (prev_close < level) and (cur_close > level)

                if crossed_up:
                    tag = "CHOCH" if trend_bias == self._BEARISH else "BOS"

                    # Нас интересует bullish CHOCH на последней свече истории
                    if tag == "CHOCH" and i == n - 1:
                        self.is_now_CHoCH = True

                    # Обновляем состояние pivot и тренда, как в Pine
                    swing_high_crossed = True
                    trend_bias = self._BULLISH

            # Bearish часть: пробой swingLow вниз (нужна, чтобы корректно менять trend_bias)
            # В Pine:
            #   p_ivot = swingLow
            #   if ta.crossunder(close,p_ivot.currentLevel) and not p_ivot.crossed
            #       string tag = t_rend.bias == BULLISH ? CHOCH : BOS
            #       ...
            #       t_rend.bias := BEARISH
            if (
                swing_low_current is not None
                and i > 0
                and not swing_low_crossed
            ):
                level = swing_low_current
                prev_close = closes[i - 1]
                cur_close = closes[i]

                # ta.crossunder(close, level)  ~  close[1] > level and close < level
                crossed_down = (prev_close > level) and (cur_close < level)

                if crossed_down:
                    tag = "CHOCH" if trend_bias == self._BULLISH else "BOS"

                    # bearish CHOCH для флага is_now_CHoCH не нужен
                    swing_low_crossed = True
                    trend_bias = self._BEARISH