import os
import json
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from collections import namedtuple
from math import isnan
import pywt  # Для wavelet decomposition


# -------------------------
# Небольшие утилиты / индикаторы с wavelet
# -------------------------
def compute_indicators(df: pd.DataFrame, atr_period: int = 14, wavelet_level: int = 3) -> pd.DataFrame:
    """
    Вычисляет технические индикаторы и wavelet decomposition для DataFrame с OHLCV данными.
    """
    df = df.copy().reset_index(drop=True)

    # ATR
    df['hl'] = df['high'] - df['low']
    df['hc'] = (df['high'] - df['close'].shift(1)).abs()
    df['lc'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    df['atr'] = df['tr'].ewm(span=atr_period, adjust=False, min_periods=1).mean()

    # SMA
    df['sma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma50'] = df['close'].rolling(50, min_periods=1).mean()

    # Momentum
    df['ret1'] = df['close'].pct_change().fillna(0)
    df['roc5'] = df['close'].pct_change(5).fillna(0)

    # RSI
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14, adjust=False, min_periods=1).mean()
    roll_down = down.ewm(span=14, adjust=False, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - 100 / (1 + rs)

    # Volume
    df['vol20'] = df['volume'].rolling(20, min_periods=1).mean()

    # Wavelet decomposition (на close)
    coeffs = pywt.wavedec(df['close'].values, 'db4', level=wavelet_level)
    df['wavelet_approx'] = np.pad(coeffs[0], (0, len(df) - len(coeffs[0])), 'constant')  # Тренд
    details = np.sum([np.pad(c, (0, len(df) - len(c)), 'constant') for c in coeffs[1:]], axis=0)
    df['wavelet_detail'] = details  # Шум/детали

    # Очистка
    df.drop(columns=['hl', 'hc', 'lc', 'tr'], inplace=True, errors='ignore')

    # Winsorizing для outliers
    for col in ['vol20', 'atr', 'wavelet_detail']:
        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = df[col].clip(q_low, q_high)

    return df


# ZigZag by ATR
Swing = namedtuple('Swing', ['idx', 'price', 'type'])


def zigzag_by_atr(df: pd.DataFrame, atr_col='atr', mult=0.8, min_bars=3, min_atr=1e-6) -> List[Swing]:
    closes = df['close'].values
    atrs = df[atr_col].values
    N = len(closes)
    swings = []
    if N < min_bars:
        return swings

    last_ext_price = closes[0]
    last_ext_idx = 0
    last_ext_type = None
    last_atr = 0

    for i in range(1, N):
        atr = atrs[i] if not isnan(atrs[i]) else last_atr
        atr = max(atr, min_atr)
        last_atr = atr
        thr = mult * atr
        move = closes[i] - last_ext_price

        if last_ext_type is None or last_ext_type == 'low':
            if move >= thr and (i - last_ext_idx) >= min_bars:
                last_ext_type = 'high'
                last_ext_price = closes[i]
                last_ext_idx = i
                swings.append(Swing(last_ext_idx, last_ext_price, last_ext_type))

        if last_ext_type is None or last_ext_type == 'high':
            if move <= -thr and (i - last_ext_idx) >= min_bars:
                last_ext_type = 'low'
                last_ext_price = closes[i]
                last_ext_idx = i
                swings.append(Swing(last_ext_idx, last_ext_price, last_ext_type))

    return swings


# Каналы с толерантностью
Channel = namedtuple('Channel',
                     ['start_idx', 'end_idx', 'lows_idx', 'highs_idx', 'slope_low', 'slope_high', 'r2_low', 'r2_high',
                      'widths'])


def find_ascending_channels_from_swings(swings: List[Swing], min_pairs: int = 3, r2_thresh: float = 0.2,
                                        cv_width_thresh: float = 0.45, tolerance: float = 0.05) -> List[Channel]:
    lows = [s for s in swings if s.type == 'low']
    highs = [s for s in swings if s.type == 'high']
    if len(lows) < min_pairs or len(highs) < min_pairs:
        return []

    channels = []
    for start in range(len(lows) - min_pairs + 1):
        for length in range(min_pairs, len(lows) - start + 1):
            lows_w = [lows[start + i].price for i in range(length)]
            highs_w = [highs[start + i].price for i in range(length)] if len(highs) >= start + length else None
            if highs_w is None:
                continue

            if not all(lows_w[i] >= lows_w[i - 1] * (1 - tolerance) for i in range(1, length)):
                continue
            if not all(highs_w[i] >= highs_w[i - 1] * (1 - tolerance) for i in range(1, length)):
                continue

            xs = np.arange(length)
            coeffs_low = np.polyfit(xs, lows_w, 1)
            ypred_low = np.polyval(coeffs_low, xs)
            ss_res_low = np.sum((np.array(lows_w) - ypred_low) ** 2)
            ss_tot_low = np.sum((np.array(lows_w) - np.mean(lows_w)) ** 2)
            r2_low = 1 - ss_res_low / ss_tot_low if ss_tot_low != 0 else 0.0

            coeffs_high = np.polyfit(xs, highs_w, 1)
            ypred_high = np.polyval(coeffs_high, xs)
            ss_res_high = np.sum((np.array(highs_w) - ypred_high) ** 2)
            ss_tot_high = np.sum((np.array(highs_w) - np.mean(highs_w)) ** 2)
            r2_high = 1 - ss_res_high / ss_tot_high if ss_tot_high != 0 else 0.0

            widths = np.array(highs_w) - np.array(lows_w)
            mean_w = np.mean(widths)
            cv = np.std(widths) / mean_w if mean_w != 0 else np.inf

            if r2_low >= r2_thresh and r2_high >= r2_thresh and cv <= cv_width_thresh:
                lows_idx = [swings.index(lows[start + i]) for i in range(length)]
                highs_idx = [swings.index(highs[start + i]) for i in range(length)]
                start_idx = min(lows[start].idx, highs[start].idx)
                end_idx = max(lows[start + length - 1].idx, highs[start + length - 1].idx)
                ch = Channel(start_idx=start_idx, end_idx=end_idx,
                             lows_idx=lows_idx, highs_idx=highs_idx,
                             slope_low=coeffs_low[0], slope_high=coeffs_high[0],
                             r2_low=r2_low, r2_high=r2_high, widths=widths.tolist())
                channels.append(ch)

    channels.sort(key=lambda c: (c.r2_low + c.r2_high) / 2, reverse=True)
    return channels


# -------------------------
# Predictor class (rule-based, без DL)
# -------------------------
class CorridorModelPredictor:
    def __init__(self, model_name: str, pair: str, model_dir: str = 'files/models'):
        self.model_name = model_name
        self.pair = pair
        self.model_dir = model_dir
        self.base_path = os.path.join(self.model_dir, f"model_{self.model_name}_{self.pair}")
        os.makedirs(self.base_path, exist_ok=True)
        self.meta = {
            'atr_period': 14,
            'zz_mult': 0.8,
            'min_channel_pairs': 3,
            'r2_thresh': 0.2,
            'cv_width_thresh': 0.45,
            'tolerance': 0.05,
            'wavelet_level': 3,
            'seq_len': 256,
            'feature_cols': ['close', 'sma20', 'sma50', 'atr', 'ret1', 'roc5', 'rsi14', 'vol20', 'wavelet_approx',
                             'wavelet_detail']
        }
        # Сохраняем meta для consistency
        with open(os.path.join(self.base_path, 'meta.json'), 'w') as f:
            json.dump(self.meta, f, indent=2)
        print(f"Initialized heuristic predictor for {model_name}_{self.pair}")

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df2 = compute_indicators(df.copy(), atr_period=self.meta.get('atr_period', 14),
                                 wavelet_level=self.meta.get('wavelet_level', 3))
        df2[self.meta['feature_cols']] = df2[self.meta['feature_cols']].ffill().fillna(
            df2[self.meta['feature_cols']].median())
        return df2

    def _channel_quality_score(self, ch: Channel, pos_in_channel: float) -> float:
        # Heuristic score: average r2 * (1 - cv) * (0.5 + pos/2) if in channel, else 0
        avg_r2 = (ch.r2_low + ch.r2_high) / 2
        cv = np.std(ch.widths) / np.mean(ch.widths) if ch.widths else np.inf
        score = avg_r2 * max(0, 1 - cv / self.meta['cv_width_thresh'])
        if 0 < pos_in_channel < 1:
            score *= (0.5 + pos_in_channel / 2)  # Бонус за позицию в канале
        else:
            score = 0
        return min(1.0, max(0.0, score))

    def predict(self, day_df: pd.DataFrame, time_string: str, try_windows: Optional[List[int]] = None,
                aggregate: str = 'weighted_mean', heuristic_boost: float = 0.2) -> Dict[str, Any]:
        df = day_df.drop_duplicates().sort_index().reset_index(drop=True).copy()
        df_feat = self._make_features(df)
        n = len(df_feat)
        seq = self.meta['seq_len']
        if try_windows is None:
            try_windows = [seq // (2 ** i) for i in range(5) if seq // (2 ** i) >= 16]  # 256,128,64,32,16

        probs = []
        details = []
        for w in sorted(try_windows, reverse=True):
            if n < w:
                continue
            last_window = df_feat.iloc[-w:].reset_index(drop=True)
            swings = zigzag_by_atr(last_window, atr_col='atr', mult=self.meta.get('zz_mult', 0.8))
            channels = find_ascending_channels_from_swings(swings, min_pairs=self.meta.get('min_channel_pairs', 3),
                                                           r2_thresh=self.meta.get('r2_thresh', 0.2),
                                                           cv_width_thresh=self.meta.get('cv_width_thresh', 0.45),
                                                           tolerance=self.meta.get('tolerance', 0.05))
            prob = 0.0
            if channels:
                ch = channels[0]
                lows = [swings[i].price for i in ch.lows_idx]
                highs = [swings[i].price for i in ch.highs_idx]
                if len(lows) >= 2 and len(highs) >= 2:
                    xs = np.arange(len(lows))
                    coeff_low = np.polyfit(xs, lows, 1)
                    coeff_high = np.polyfit(xs, highs, 1)
                    x_now = xs[-1]
                    low_line = coeff_low[0] * x_now + coeff_low[1]
                    high_line = coeff_high[0] * x_now + coeff_high[1]
                    last_price = last_window['close'].iloc[-1]
                    width = high_line - low_line if high_line - low_line != 0 else 1e-9
                    pos_in_channel = (last_price - low_line) / width
                    in_channel = 0 <= pos_in_channel <= 1
                    if in_channel:
                        prob = self._channel_quality_score(ch, pos_in_channel)
            probs.append(prob)
            details.append({'window': w, 'prob': prob})

        if not probs:
            return {'probability_pct': 0.0, 'raw_probs': [], 'chosen_window': None,
                    'explanation': 'Not enough data.'}

        if aggregate == 'max':
            p = max(probs)
        elif aggregate == 'mean':
            p = np.mean(probs)
        elif aggregate == 'weighted_mean':
            weights = [d['window'] for d in details]
            p = np.average(probs, weights=weights)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")

        best_detail = max(details, key=lambda d: d['prob'])
        chosen_w = best_detail['window']
        last_window = df_feat.iloc[-chosen_w:].reset_index(drop=True)
        swings = zigzag_by_atr(last_window, atr_col='atr', mult=self.meta.get('zz_mult', 0.8))
        channels = find_ascending_channels_from_swings(swings, min_pairs=self.meta.get('min_channel_pairs', 3),
                                                       r2_thresh=self.meta.get('r2_thresh', 0.2),
                                                       cv_width_thresh=self.meta.get('cv_width_thresh', 0.45),
                                                       tolerance=self.meta.get('tolerance', 0.05))
        in_channel = False
        pos_in_channel = None
        channel_info = None
        if channels:
            ch = channels[0]
            lows = [swings[i].price for i in ch.lows_idx]
            highs = [swings[i].price for i in ch.highs_idx]
            if len(lows) >= 2 and len(highs) >= 2:
                xs = np.arange(len(lows))
                coeff_low = np.polyfit(xs, lows, 1)
                coeff_high = np.polyfit(xs, highs, 1)
                x_now = xs[-1]
                low_line = coeff_low[0] * x_now + coeff_low[1]
                high_line = coeff_high[0] * x_now + coeff_high[1]
                last_price = last_window['close'].iloc[-1]
                width = high_line - low_line if high_line - low_line != 0 else 1e-9
                pos_in_channel = (last_price - low_line) / width
                in_channel = 0 <= pos_in_channel <= 1
                channel_info = {'low_line': low_line, 'high_line': high_line, 'pos': pos_in_channel,
                                'r2_low': ch.r2_low, 'r2_high': ch.r2_high, 'width_mean': np.mean(ch.widths)}

        if in_channel:
            p = min(1.0, p + heuristic_boost * (1 - p))

        ret = {
            'probability_pct': p * 100.0,
            'raw_probs': details,
            'chosen_window': best_detail,
            'in_channel_heuristic': in_channel,
            'pos_in_channel': pos_in_channel,
            'channel_info': channel_info,
            'explanation': f"Aggregated over {len(details)} windows with '{aggregate}'. Boost: {heuristic_boost if in_channel else 0}"
        }

        with open(os.path.join(self.base_path+'/logs/', f'predictions_{time_string}.json'), 'w') as f:
            json.dump(ret, f, indent=2)
        return ret