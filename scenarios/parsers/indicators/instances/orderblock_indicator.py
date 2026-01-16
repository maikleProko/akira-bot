import numpy as np
import pandas as pd
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator

class OrderblockIndicator(Indicator):
    """
    Детектор order blocks по логике LuxAlgo SMC
    для swing-структуры (internal = false).
    Логика полностью сохранена из оригинального кода — ничего не менялось в расчётах.
    Адаптировано под исторический и реал-тайм режимы через run.
    """
    _BEARISH_LEG = 0
    _BULLISH_LEG = 1
    _BULLISH = 1
    _BEARISH = -1

    def __init__(self, history_market_parser: HistoryMarketParser):
        super().__init__(history_market_parser)
        self.swing_size = 50  # from swingsLengthInput default
        self.atr_length = 200
        self.filter_type = "Atr"  # default
        self.mitigation_type = "High/Low"  # default
        self.max_order_blocks = 5  # swingOrderBlocksSizeInput default
        # Результат на "текущий" момент (на конец df)
        self.bull_orderblocks = []  # list of {'y1': float, 'y2': float}
        self.bear_orderblocks = []  # list of {'y1': float, 'y2': float}

        self.last_bull_orderblock = None
        self.last_bear_orderblock = None

    def run(self, start_time=None, current_time=None):
        """
        Вычисляет зоны на данных до current_time (historical) или на полном df (realtime).
        Зоны стираются и пересчитываются заново каждый раз.
        """
        try:
            sub_df = self.history_market_parser.df.copy()

            if sub_df.empty or 'close' not in sub_df.columns:
                self.bull_orderblocks = []
                self.bear_orderblocks = []
                return

            self.bull_orderblocks, self.bear_orderblocks = self._compute_zones(sub_df)
        except:
            print('!')
            self.bull_orderblocks, self.bear_orderblocks = [], []
            self.last_bull_orderblock = None
            self.last_bear_orderblock = None


    def run_realtime(self):
        self.run()


    def _compute_zones(self, df: pd.DataFrame):
        df = df.reset_index(drop=True)
        n = len(df)
        size = int(self.swing_size)
        if size < 1 or n <= size + 1:
            return [], []

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        # opens = df["open"].to_numpy()  # not used here

        # Compute tr
        tr = np.maximum(highs - lows, np.abs(highs - np.roll(closes, 1)))
        tr = np.maximum(tr, np.abs(lows - np.roll(closes, 1)))
        tr[0] = highs[0] - lows[0]

        # volatility_measure
        if self.filter_type == "Atr":
            vol_measure = np.zeros(n)
            vol_measure[0] = tr[0]
            alpha = 1.0 / self.atr_length
            for i in range(1, n):
                vol_measure[i] = alpha * tr[i] + (1 - alpha) * vol_measure[i - 1]
        else:  # RANGE
            cum_tr = np.cumsum(tr)
            indices = np.arange(1, n + 1)
            vol_measure = cum_tr / indices

        # high_volatility_bar
        high_vol = (highs - lows) >= 2 * vol_measure

        # parsed_highs and parsed_lows
        parsed_highs = np.where(high_vol, lows, highs)
        parsed_lows = np.where(high_vol, highs, lows)

        # mitigation sources
        if self.mitigation_type == "Close":
            bear_mit_src = closes
            bull_mit_src = closes
        else:  # HIGHLOW
            bear_mit_src = highs
            bull_mit_src = lows

        # Состояние
        swing_high_current = None
        swing_high_crossed = False
        swing_high_pivot_idx = None
        swing_low_current = None
        swing_low_crossed = False
        swing_low_pivot_idx = None
        leg_prev = self._BEARISH_LEG
        trend_bias = self._BEARISH
        order_blocks = []  # list of dict{'high': float, 'low': float, 'bias': int}

        for i in range(n):
            # 1) Обновление свингов (leg + getCurrentStructure)
            if i >= size:
                pivot_idx = i - size
                window_high = highs[pivot_idx + 1 : i + 1]
                window_low = lows[pivot_idx + 1 : i + 1]
                new_leg_high = highs[pivot_idx] > np.max(window_high) if len(window_high) > 0 else False
                new_leg_low = lows[pivot_idx] < np.min(window_low) if len(window_low) > 0 else False
                leg_val = leg_prev
                if new_leg_high:
                    leg_val = self._BEARISH_LEG
                elif new_leg_low:
                    leg_val = self._BULLISH_LEG
                change = leg_val - leg_prev
                new_pivot = change != 0
                pivot_low = change == +1  # bullish leg start
                pivot_high = change == -1  # bearish leg start
                if new_pivot:
                    if pivot_low:
                        swing_low_current = lows[pivot_idx]
                        swing_low_crossed = False
                        swing_low_pivot_idx = pivot_idx
                    else:
                        swing_high_current = highs[pivot_idx]
                        swing_high_crossed = False
                        swing_high_pivot_idx = pivot_idx
                leg_prev = leg_val

            # 2) Проверка пробоев (displayStructure, internal=false)
            # Bullish: пробой swingHigh вверх -> store BULLISH OB
            if swing_high_current is not None and i > 0 and not swing_high_crossed:
                level = swing_high_current
                prev_close = closes[i - 1]
                cur_close = closes[i]
                crossed_up = (prev_close <= level) and (cur_close > level)
                if crossed_up:
                    # tag = "CHOCH" if trend_bias == self._BEARISH else "BOS"  # not needed
                    swing_high_crossed = True
                    trend_bias = self._BULLISH
                    # storeOrdeBlock(internal=false, bias=BULLISH)
                    bias = self._BULLISH
                    a_rray = parsed_lows[swing_high_pivot_idx : i + 1]
                    if len(a_rray) > 0:
                        min_idx = np.argmin(a_rray)
                        parsed_index = swing_high_pivot_idx + min_idx
                        ob_high = parsed_highs[parsed_index]
                        ob_low = parsed_lows[parsed_index]
                        ob = {'high': ob_high, 'low': ob_low, 'bias': bias}
                        if len(order_blocks) >= 100:
                            order_blocks.pop()
                        order_blocks.insert(0, ob)

            # Bearish: пробой swingLow вниз -> store BEARISH OB
            if swing_low_current is not None and i > 0 and not swing_low_crossed:
                level = swing_low_current
                prev_close = closes[i - 1]
                cur_close = closes[i]
                crossed_down = (prev_close >= level) and (cur_close < level)
                if crossed_down:
                    # tag = "CHOCH" if trend_bias == self._BULLISH else "BOS"
                    swing_low_crossed = True
                    trend_bias = self._BEARISH
                    # storeOrdeBlock(internal=false, bias=BEARISH)
                    bias = self._BEARISH
                    a_rray = parsed_highs[swing_low_pivot_idx : i + 1]
                    if len(a_rray) > 0:
                        max_idx = np.argmax(a_rray)
                        parsed_index = swing_low_pivot_idx + max_idx
                        ob_high = parsed_highs[parsed_index]
                        ob_low = parsed_lows[parsed_index]
                        ob = {'high': ob_high, 'low': ob_low, 'bias': bias}
                        if len(order_blocks) >= 100:
                            order_blocks.pop()
                        order_blocks.insert(0, ob)

            # 3) deleteOrderBlocks(internal=false)
            to_remove = []
            for j in range(len(order_blocks)):
                ob = order_blocks[j]
                crossed = False
                if bear_mit_src[i] > ob['high'] and ob['bias'] == self._BEARISH:
                    crossed = True
                elif bull_mit_src[i] < ob['low'] and ob['bias'] == self._BULLISH:
                    crossed = True
                if crossed:
                    to_remove.append(j)
            for j in sorted(to_remove, reverse=True):
                del order_blocks[j]

        # Финальные видимые зоны (новейшие max_order_blocks)
        order_blocks = order_blocks[:self.max_order_blocks]
        bull_orderblocks = []
        bear_orderblocks = []
        self.last_bull_orderblock = None
        self.last_bear_orderblock = None
        for ob in order_blocks:
            y1 = min(ob['low'], ob['high'])
            y2 = max(ob['low'], ob['high'])
            zone = {'y1': y1, 'y2': y2}
            if ob['bias'] == self._BULLISH:
                bull_orderblocks.append(zone)
                if not self.last_bull_orderblock:
                    self.last_bull_orderblock = zone
            else:
                bear_orderblocks.append(zone)
                if not self.last_bear_orderblock:
                    self.last_bear_orderblock = zone
        return bull_orderblocks, bear_orderblocks