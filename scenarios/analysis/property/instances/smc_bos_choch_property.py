import numpy as np
import pandas as pd
from scenarios.analysis.property.abstracts.property import Property
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.bos_indicator import BosIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator

class SMCBosChochProperty(Property):
    _BEARISH_LEG = 0
    _BULLISH_LEG = 1
    _BULLISH = "BULLISH"
    _BEARISH = "BEARISH"

    def __init__(self, minutes):
        super().__init__()
        self.minutes = minutes
        self.history_market_parser: HistoryBinanceParser = None
        self.bos_indicator: BosIndicator = None
        self.choch_indicator: CHoCHIndicator = None

    def prepare(self, deal):
        self.history_market_parser = HistoryBinanceParser('BTC', 'USDT', self.minutes, 1000, 'generating')
        self.bos_indicator = BosIndicator(self.history_market_parser)
        self.choch_indicator = CHoCHIndicator(self.history_market_parser)
        self.market_processes = [
            self.history_market_parser,
            self.bos_indicator,
            self.choch_indicator
        ]

    def _compute_smc_structure(self):
        df = self.history_market_parser.df
        if df is None or df.empty or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
            return None

        n = len(df)
        size = 5  # swing_size from indicators
        if size < 1 or n <= size + 1:
            return None

        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()

        # Состояние
        swing_high_current = None
        swing_high_last = None
        swing_high_crossed = False
        swing_high_index = None
        swing_low_current = None
        swing_low_last = None
        swing_low_crossed = False
        swing_low_index = None
        leg_prev = self._BEARISH_LEG
        trend_bias = self._BEARISH

        # Результаты
        is_bos_list = [False] * n
        bos_price_list = [np.nan] * n
        is_choch_list = [False] * n
        choch_price_list = [np.nan] * n
        trend_bias_list = [self._BEARISH] * n
        swing_high_list = [np.nan] * n
        swing_low_list = [np.nan] * n

        for i in range(n):
            # Обновление свингов
            if i >= size:
                pivot_idx = i - size
                if pivot_idx + 1 <= i:
                    window_high = highs[pivot_idx + 1: i + 1]
                    window_low = lows[pivot_idx + 1: i + 1]
                    new_leg_high = highs[pivot_idx] > window_high.max() if window_high.size > 0 else False
                    new_leg_low = lows[pivot_idx] < window_low.min() if window_low.size > 0 else False
                else:
                    new_leg_high = new_leg_low = False

                leg_val = leg_prev
                if new_leg_high:
                    leg_val = self._BEARISH_LEG
                elif new_leg_low:
                    leg_val = self._BULLISH_LEG

                change = leg_val - leg_prev
                new_pivot = change != 0
                pivot_low = change == +1
                pivot_high = change == -1

                if new_pivot:
                    if pivot_low:
                        swing_low_last = swing_low_current
                        swing_low_current = lows[pivot_idx]
                        swing_low_crossed = False
                        swing_low_index = pivot_idx
                        swing_low_list[pivot_idx] = swing_low_current
                    else:
                        swing_high_last = swing_high_current
                        swing_high_current = highs[pivot_idx]
                        swing_high_crossed = False
                        swing_high_index = pivot_idx
                        swing_high_list[pivot_idx] = swing_high_current

                leg_prev = leg_val

            # Проверка пробоев
            bos_this_bar = False
            bos_price_this_bar = np.nan
            choch_this_bar = False
            choch_price_this_bar = np.nan

            # Bullish crossed_up
            if swing_high_current is not None and i > 0 and not swing_high_crossed:
                level = swing_high_current
                prev_close = closes[i - 1]
                cur_close = closes[i]
                crossed_up = (prev_close < level) and (cur_close > level)
                if crossed_up:
                    tag = "CHOCH" if trend_bias == self._BEARISH else "BOS"
                    if tag == "BOS":
                        bos_this_bar = True
                        bos_price_this_bar = level
                    if tag == "CHOCH":
                        choch_this_bar = True
                        choch_price_this_bar = level
                    swing_high_crossed = True
                    trend_bias = self._BULLISH

            # Bearish crossed_down
            if swing_low_current is not None and i > 0 and not swing_low_crossed:
                level = swing_low_current
                prev_close = closes[i - 1]
                cur_close = closes[i]
                crossed_down = (prev_close > level) and (cur_close < level)
                if crossed_down:
                    tag = "CHOCH" if trend_bias == self._BULLISH else "BOS"
                    swing_low_crossed = True
                    trend_bias = self._BEARISH

            is_bos_list[i] = bos_this_bar
            bos_price_list[i] = bos_price_this_bar
            is_choch_list[i] = choch_this_bar
            choch_price_list[i] = choch_price_this_bar
            trend_bias_list[i] = trend_bias

        is_bos_series = pd.Series(is_bos_list, index=df.index, dtype=bool)
        bos_price_series = pd.Series(bos_price_list, index=df.index)
        is_choch_series = pd.Series(is_choch_list, index=df.index, dtype=bool)
        choch_price_series = pd.Series(choch_price_list, index=df.index)
        bias_series = pd.Series(trend_bias_list, index=df.index)
        swing_high_series = pd.Series(swing_high_list, index=df.index)
        swing_low_series = pd.Series(swing_low_list, index=df.index)

        return is_bos_series, bos_price_series, is_choch_series, choch_price_series, bias_series, swing_high_series, swing_low_series

    def _bars_since_last_event(self, event_series: pd.Series) -> int:
        if event_series.empty:
            return -1
        try:
            last_idx = event_series[event_series].index.max()
            if pd.isna(last_idx):
                return -1
            return len(event_series) - 1 - last_idx
        except:
            return -1

    def _had_event_recent(self, event_series: pd.Series, n: int = 10) -> bool:
        if event_series.empty or len(event_series) < n:
            return False
        recent = event_series.iloc[-n:]
        return recent.any()

    def _get_market_structure_state(self) -> str:
        structure = self._compute_smc_structure()
        if structure is None:
            return 'range'
        is_bos, _, is_choch, _, bias, _, _ = structure
        current_bias = bias.iloc[-1]
        bars_since_bos = self._bars_since_last_event(is_bos)
        bars_since_choch = self._bars_since_last_event(is_choch)
        bars_since_change = min(bars_since_bos, bars_since_choch) if min(bars_since_bos, bars_since_choch) >= 0 else -1
        if bars_since_change == -1 or bars_since_change > 20:
            return 'range'
        if current_bias == self._BULLISH:
            return 'trend'
        elif current_bias == self._BEARISH:
            return 'trend'  # assuming 'trend' means any trend, but user said trend / range
        return 'range'

    def _get_higher_highs_count(self, n: int = 20) -> int:
        structure = self._compute_smc_structure()
        if structure is None:
            return 0
        _, _, _, _, _, sh, _ = structure
        swing_highs = sh.dropna()
        if len(swing_highs) < 2:
            return 0
        # Get swings in last n bars
        recent_sh = swing_highs[swing_highs.index >= len(sh) - n]
        if len(recent_sh) < 2:
            return 0
        diffs = recent_sh.diff()
        return int((diffs > 0).sum())

    def _get_higher_lows_count(self, n: int = 20) -> int:
        structure = self._compute_smc_structure()
        if structure is None:
            return 0
        _, _, _, _, _, _, sl = structure
        swing_lows = sl.dropna()
        if len(swing_lows) < 2:
            return 0
        recent_sl = swing_lows[swing_lows.index >= len(sl) - n]
        if len(recent_sl) < 2:
            return 0
        diffs = recent_sl.diff()
        return int((diffs > 0).sum())

    def analyze(self, deal):
        structure = self._compute_smc_structure()
        prefix = f's{str(self.minutes)}_'
        if structure is None:
            deal[prefix + 'chd'] = False
            deal[prefix + 'bch'] = -1
            deal[prefix + 'br'] = False
            deal[prefix + 'bb'] = -1
            deal[prefix + 'mss'] = 'range'
            deal[prefix + 'hhc'] = 0
            deal[prefix + 'hlc'] = 0
        else:
            is_bos, _, is_choch, _, _, _, _ = structure
            deal[prefix + 'chd'] = bool(is_choch.iloc[-1])
            deal[prefix + 'bch'] = int(self._bars_since_last_event(is_choch))
            deal[prefix + 'br'] = bool(self._had_event_recent(is_bos, n=10))
            deal[prefix + 'bb'] = int(self._bars_since_last_event(is_bos))
            deal[prefix + 'mss'] = str(self._get_market_structure_state())
            deal[prefix + 'hhc'] = int(self._get_higher_highs_count(n=20))
            deal[prefix + 'hlc'] = int(self._get_higher_lows_count(n=20))
        return deal