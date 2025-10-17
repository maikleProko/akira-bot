import os
import glob
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple
import math
import numpy as np

class OrderbookDepthEstimator:
    def __init__(
            self,
            base_path: str = "files/orderbook/beautifulsoup/",
            filename_regex: str = r'beautifulsoup_coinglass_orderbook_BTC-USDT-(\d{4}:\d{2}:\d{2})_(\d{2}:\d{2})\.json$',
            W_price_bins: int = 3,
            min_width_bins: int = 3,
            las_weights: Dict[str, float] = None,
            las_thresh_factor: float = 0.7
    ):
        """
        Инициализация сервиса. Параметры можно тонко настраивать.
        """
        self.base_path = base_path
        self.file_pattern = os.path.join(base_path, "beautifulsoup_coinglass_orderbook_BTC-USDT-*.json")
        self.filename_regex = re.compile(filename_regex)
        self.W_price_bins = max(1, int(W_price_bins))
        self.min_width_bins = max(1, int(min_width_bins))
        # default LAS weights
        self.las_weights = las_weights or {
            'w_avg': 0.35,
            'w_persist': 0.25,
            'w_replen': 0.2,
            'w_growth': 0.1,
            'w_removed': 0.1,
            'w_trade_through': 0.15
        }
        self.las_thresh_factor = float(las_thresh_factor)

        # placeholders filled during run
        self.snapshots = []  # list of {'ts': datetime, 'bids': [(p,a),...], 'asks': [(p,a),...]}
        self.all_prices = []
        self.all_amounts = []
        self.tick = 1.0
        self.bin_size = None
        self.bin_edges = None
        self.bin_centers = None
        self.n_bins = 0
        self.min_price = None
        self.max_price = None
        self.best_prices = []

    # ========== Public API ==========
    def define_depth_level(self, start_time: datetime, current_time: datetime) -> List[Dict]:
        """
        Главная функция — возвращает список уровневых диапазонов:
        [{'side': 'bid'|'ask', 'lower': int_price, 'upper': int_price, 'score': float}, ...]
        """
        files = self._find_files(start_time, current_time)

        print(files)
        if not files:
            return []

        self._load_snapshots(files)
        if not self.snapshots:
            return []

        self._determine_tick_and_binning()
        bids_matrix, asks_matrix = self._build_matrices()
        if bids_matrix.size == 0 and asks_matrix.size == 0:
            return []

        bids_metrics = self._compute_metrics_for_side(bids_matrix, side='bids')
        asks_metrics = self._compute_metrics_for_side(asks_matrix, side='asks')

        bids_las, _ = self._build_las(bids_metrics)
        asks_las, _ = self._build_las(asks_metrics)

        bid_levels = self._select_levels_from_las(bids_las, min_width_bins=self.min_width_bins,
                                                  las_thresh_factor=self.las_thresh_factor)
        ask_levels = self._select_levels_from_las(asks_las, min_width_bins=self.min_width_bins,
                                                  las_thresh_factor=self.las_thresh_factor)

        result = []
        for (i, j, score) in bid_levels:
            lower, upper = self._bin_range_to_price(i, j)
            result.append(
                {'side': 'bid', 'lower': int(round(lower)), 'upper': int(round(upper)), 'score': float(score)})
        for (i, j, score) in ask_levels:
            lower, upper = self._bin_range_to_price(i, j)
            result.append(
                {'side': 'ask', 'lower': int(round(lower)), 'upper': int(round(upper)), 'score': float(score)})

        # sort by score descending
        result.sort(key=lambda x: x['score'], reverse=True)
        print(result)
        return result

    # ========== File discovery and loading ==========
    def _find_files(self, start_time: datetime, current_time: datetime) -> List[Tuple[datetime, str]]:
        file_list = glob.glob(self.file_pattern)
        time_files = []
        for f in file_list:
            m = self.filename_regex.search(f)
            if not m:
                continue
            date_part = m.group(1)
            time_part = m.group(2)
            try:
                ts = datetime.strptime(f"{date_part}_{time_part}", "%Y:%m:%d_%H:%M")
            except Exception:
                continue
            if start_time <= ts <= current_time:
                time_files.append((ts, f))
        time_files.sort(key=lambda x: x[0])
        return time_files

    def _load_snapshots(self, time_files: List[Tuple[datetime, str]]):
        self.snapshots = []
        self.all_prices = []
        self.all_amounts = []
        self.best_prices = []

        for ts, fname in time_files:
            try:
                with open(fname, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
            except Exception:
                continue
            bids = data.get("bids", []) or []
            asks = data.get("asks", []) or []

            bids_clean = [(float(item['price']), float(item.get('amount', 0.0))) for item in bids if
                          'price' in item]
            asks_clean = [(float(item['price']), float(item.get('amount', 0.0))) for item in asks if
                          'price' in item]

            for p, a in bids_clean + asks_clean:
                # don't include negative or zero prices
                if p is None:
                    continue
                self.all_prices.append(p)
                self.all_amounts.append(max(0.0, float(a)))

            self.snapshots.append({'ts': ts, 'bids': bids_clean, 'asks': asks_clean})

            # best prices robustly
            best_bid = max([p for p, _ in bids_clean]) if bids_clean else None
            best_ask = min([p for p, _ in asks_clean]) if asks_clean else None
            self.best_prices.append((best_bid, best_ask))

    # ========== Tick and binning ==========
    def _determine_tick_and_binning(self):
        # determine tick: minimal positive diff between unique prices
        if not self.all_prices:
            self.tick = 1.0
            self.bin_size = float(self.W_price_bins) * self.tick
            self.min_price = 0.0
            self.max_price = 0.0
            self.n_bins = 1
            self.bin_edges = [0.0, self.bin_size]
            self.bin_centers = [self.bin_size * 0.5]
            return

        unique_prices = np.array(sorted(set(self.all_prices)))
        diffs = np.diff(unique_prices)
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            self.tick = 1.0
        else:
            self.tick = float(max(1.0, np.min(diffs)))

        self.bin_size = float(self.W_price_bins) * self.tick
        self.min_price = math.floor(min(self.all_prices) / self.bin_size) * self.bin_size
        self.max_price = math.ceil(max(self.all_prices) / self.bin_size) * self.bin_size
        span = max(0.0, self.max_price - self.min_price)
        self.n_bins = int(max(1, math.ceil(span / self.bin_size)))
        # ensure at least 1 bin and edges aligned
        self.bin_edges = [self.min_price + i * self.bin_size for i in range(self.n_bins + 1)]
        self.bin_centers = [self.min_price + (i + 0.5) * self.bin_size for i in range(self.n_bins)]

    def _price_to_bin_idx(self, p: float) -> int:
        if self.n_bins <= 0:
            return 0
        idx = int(math.floor((p - self.min_price) / self.bin_size))
        if idx < 0:
            return 0
        if idx >= self.n_bins:
            return self.n_bins - 1
        return idx

    # ========== Build time x bins matrices ==========
    def _build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        T = len(self.snapshots)
        n = max(1, self.n_bins)
        bids_matrix = np.zeros((T, n), dtype=float)
        asks_matrix = np.zeros((T, n), dtype=float)

        for ti, snap in enumerate(self.snapshots):
            for p, amt in snap['bids']:
                idx = self._price_to_bin_idx(p)
                bids_matrix[ti, idx] += float(amt)
            for p, amt in snap['asks']:
                idx = self._price_to_bin_idx(p)
                asks_matrix[ti, idx] += float(amt)
        return bids_matrix, asks_matrix

    # ========== Metrics computation ==========
    def _compute_metrics_for_side(self, matrix: np.ndarray, side: str = 'bids') -> Dict[str, np.ndarray]:
        """
        Возвращает словарь метрик для каждого бина.
        """
        T = matrix.shape[0]
        eps = 1e-9
        if len(self.all_amounts) > 0:
            D_min = float(np.percentile(self.all_amounts, 50))  # median
        else:
            D_min = 1.0

        if T <= 1:
            deltas = np.zeros_like(matrix)
        else:
            deltas = np.diff(matrix, axis=0)  # shape (T-1, n_bins)

        avg_depth = matrix.mean(axis=0)
        max_depth = matrix.max(axis=0)
        persistency = np.mean(matrix >= D_min, axis=0)

        total_added = np.sum(np.maximum(deltas, 0.0), axis=0) if T > 1 else np.zeros(matrix.shape[1])
        total_removed = np.sum(np.maximum(-deltas, 0.0), axis=0) if T > 1 else np.zeros(matrix.shape[1])

        # growth rate
        half = max(1, T // 4)
        last_avg = matrix[-half:, :].mean(axis=0) if T >= half else matrix.mean(axis=0)
        first_avg = matrix[:half, :].mean(axis=0) if T >= half else matrix.mean(axis=0)
        growth_rate = (last_avg - first_avg) / (first_avg + eps)

        # replenishment scores and absorption events
        recovery_window = max(1, min(5, T - 1)) if T > 1 else 1
        n_bins_local = matrix.shape[1]
        replenishment_scores = np.zeros(n_bins_local, dtype=float)
        absorption_events = np.zeros(n_bins_local, dtype=float)
        for b in range(n_bins_local):
            depths = matrix[:, b]
            local_max = np.max(depths) if depths.size > 0 else 0.0
            if local_max <= 0:
                replenishment_scores[b] = 0.0
                absorption_events[b] = 0.0
                continue
            threshold_drop = 0.5 * local_max
            recovery_target = 0.7 * local_max
            events = 0
            recovered = 0
            for i in range(1, T):
                if depths[i - 1] >= threshold_drop and depths[i] < threshold_drop:
                    events += 1
                    recovered_flag = False
                    for j in range(i, min(T, i + recovery_window + 1)):
                        if depths[j] >= recovery_target:
                            recovered_flag = True
                            break
                    if recovered_flag:
                        recovered += 1
            absorption_events[b] = events
            replenishment_scores[b] = (recovered / events) if events > 0 else 0.0

        # trade-through proxy (midprice crossing)
        midprices = []
        for (bb, ba) in self.best_prices:
            if bb is None or ba is None:
                midprices.append(None)
            else:
                midprices.append(0.5 * (bb + ba))
        trade_through_counts = np.zeros(n_bins_local, dtype=float)
        for b in range(n_bins_local):
            center = self.bin_centers[b]
            for i in range(1, T):
                prev_mid = midprices[i - 1]
                cur_mid = midprices[i]
                if prev_mid is None or cur_mid is None:
                    continue
                if (prev_mid - center) * (cur_mid - center) < 0:
                    trade_through_counts[b] += 1

        metrics = {
            'avg_depth': avg_depth,
            'max_depth': max_depth,
            'persistency': persistency,
            'total_added': total_added,
            'total_removed': total_removed,
            'growth_rate': growth_rate,
            'replenishment': replenishment_scores,
            'absorption_events': absorption_events,
            'trade_throughs': trade_through_counts
        }
        return metrics

    # ========== LAS construction ==========
    def _build_las(self, metrics: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # weights
        w_avg = self.las_weights.get('w_avg', 0.35)
        w_persist = self.las_weights.get('w_persist', 0.25)
        w_replen = self.las_weights.get('w_replen', 0.2)
        w_growth = self.las_weights.get('w_growth', 0.1)
        w_removed = self.las_weights.get('w_removed', 0.1)
        w_trade_through = self.las_weights.get('w_trade_through', 0.15)

        avg_n = self._normalize_arr(metrics['avg_depth'])
        persist_n = self._normalize_arr(metrics['persistency'])
        replen_n = self._normalize_arr(metrics['replenishment'])
        growth_n = self._normalize_arr(metrics['growth_rate'])
        removed_ratio = metrics['total_removed'] / (metrics['total_added'] + 1e-9)
        removed_n = self._normalize_arr(removed_ratio)
        trade_through_n = self._normalize_arr(metrics['trade_throughs'])

        las = (w_avg * avg_n +
               w_persist * persist_n +
               w_replen * replen_n +
               w_growth * growth_n -
               w_removed * removed_n -
               w_trade_through * trade_through_n)
        aux = {
            'avg_n': avg_n,
            'persist_n': persist_n,
            'replen_n': replen_n,
            'growth_n': growth_n,
            'removed_n': removed_n,
            'trade_through_n': trade_through_n
        }
        return las, aux

    # ========== Candidate selection ==========
    def _select_levels_from_las(self, las_arr: np.ndarray, min_width_bins: int = 3,
                                las_thresh_factor: float = 1.0) -> List[Tuple[int, int, float]]:
        if las_arr is None or len(las_arr) == 0:
            return []
        mean = float(np.nanmean(las_arr))
        std = float(np.nanstd(las_arr))
        threshold = mean + las_thresh_factor * std
        candidates = las_arr >= threshold

        levels = []
        i = 0
        N = len(candidates)
        while i < N:
            if not candidates[i]:
                i += 1
                continue
            j = i
            while j + 1 < N and candidates[j + 1]:
                j += 1
            width = j - i + 1
            if width >= min_width_bins:
                seg_score = float(np.nanmean(las_arr[i:j + 1]))
                levels.append((i, j, seg_score))
            i = j + 1
        return levels

    # ========== Utilities ==========
    def _bin_range_to_price(self, i: int, j: int) -> Tuple[float, float]:
        lower = self.bin_edges[i] if i < len(self.bin_edges) else self.min_price
        upper = self.bin_edges[j + 1] if (j + 1) < len(self.bin_edges) else self.max_price
        return lower, upper

    def _normalize_arr(self, arr) -> np.ndarray:
        a = np.array(arr, dtype=float)
        # handle NaN
        a = np.nan_to_num(a, nan=0.0)
        mn = np.min(a)
        mx = np.max(a)
        if mx - mn < 1e-12:
            if mx == 0:
                return np.zeros_like(a)
            return np.clip(a / (mx + 1e-12), 0.0, 1.0)
        return (a - mn) / (mx - mn)