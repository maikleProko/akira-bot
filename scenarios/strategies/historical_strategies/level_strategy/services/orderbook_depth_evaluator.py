import os
import glob
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple
import math
import numpy as np

class OrderbookDepthEvaluator:
    def __init__(
        self,
        base_path: str = "files/orderbook/beautifulsoup/coinglass/",
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
        self.snapshots = []
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

    def define_depth_level(self, start_time: datetime, current_time: datetime) -> List[Dict]:
        """
        Главная функция — возвращает список уровневых диапазонов поддержки:
        [{'side': 'bid', 'lower': int_price, 'upper': int_price, 'score': float}, ...]
        """
        files = self._find_files(start_time, current_time)
        if not files:
            return []

        self._load_snapshots(files)
        if not self.snapshots:
            return []

        self._determine_tick_and_binning()
        bids_matrix, _ = self._build_matrices()
        if bids_matrix.size == 0:
            return []

        bids_metrics = self._compute_metrics_for_side(bids_matrix, side='bids')
        bids_las, _ = self._build_las(bids_metrics)

        bid_levels = self._select_levels_from_las(bids_las, min_width_bins=self.min_width_bins, las_thresh_factor=self.las_thresh_factor)

        result = []
        for (i, j, score) in bid_levels:
            lower, upper = self._bin_range_to_price(i, j)
            result.append({'side': 'bid', 'lower': int(round(lower)), 'upper': int(round(upper)), 'score': float(score)})

        # sort by score descending
        result.sort(key=lambda x: x['score'], reverse=True)
        return result

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

            bids_clean = [(float(item['price']), float(item.get('amount', 0.0))) for item in bids if 'price' in item]
            asks_clean = [(float(item['price']), float(item.get('amount', 0.0))) for item in asks if 'price' in item]

            for p, a in bids_clean + asks_clean:
                if p is None:
                    continue
                self.all_prices.append(p)
                self.all_amounts.append(max(0.0, float(a)))

            self.snapshots.append({'ts': ts, 'bids': bids_clean, 'asks': asks_clean})

            best_bid = max([p for p, _ in bids_clean]) if bids_clean else None
            best_ask = min([p for p, _ in asks_clean]) if asks_clean else None
            self.best_prices.append((best_bid, best_ask))

    def _determine_tick_and_binning(self):
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
        self.tick = float(max(1.0, np.min(diffs))) if len(diffs) > 0 else 1.0

        self.bin_size = float(self.W_price_bins) * self.tick
        self.min_price = math.floor(min(self.all_prices) / self.bin_size) * self.bin_size
        self.max_price = math.ceil(max(self.all_prices) / self.bin_size) * self.bin_size
        span = max(0.0, self.max_price - self.min_price)
        self.n_bins = int(max(1, math.ceil(span / self.bin_size)))
        self.bin_edges = [self.min_price + i * self.bin_size for i in range(self.n_bins + 1)]
        self.bin_centers = [self.min_price + (i + 0.5) * self.bin_size for i in range(self.n_bins)]

    def _build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        T = len(self.snapshots)
        n = max(1, self.n_bins)
        bids_matrix = np.zeros((T, n), dtype=float)

        for ti, snap in enumerate(self.snapshots):
            for p, amt in snap['bids']:
                idx = self._price_to_bin_idx(p)
                bids_matrix[ti, idx] += float(amt)

        return bids_matrix, None  # Возвращаем только матрицу бидов

    def _compute_metrics_for_side(self, matrix: np.ndarray, side: str = 'bids') -> Dict[str, np.ndarray]:
        T = matrix.shape[0]
        eps = 1e-9
        D_min = float(np.percentile(self.all_amounts, 50)) if len(self.all_amounts) > 0 else 1.0

        if T <= 1:
            deltas = np.zeros_like(matrix)
        else:
            deltas = np.diff(matrix, axis=0)

        avg_depth = matrix.mean(axis=0)
        max_depth = matrix.max(axis=0)
        persistency = np.mean(matrix >= D_min, axis=0)

        total_added = np.sum(np.maximum(deltas, 0.0), axis=0) if T > 1 else np.zeros(matrix.shape[1])
        total_removed = np.sum(np.maximum(-deltas, 0.0), axis=0) if T > 1 else np.zeros(matrix.shape[1])

        half = max(1, T // 4)
        last_avg = matrix[-half:, :].mean(axis=0) if T >= half else matrix.mean(axis=0)
        first_avg = matrix[:half, :].mean(axis=0) if T >= half else matrix.mean(axis=0)
        growth_rate = (last_avg - first_avg) / (first_avg + eps)

        metrics = {
            'avg_depth': avg_depth,
            'max_depth': max_depth,
            'persistency': persistency,
            'total_added': total_added,
            'total_removed': total_removed,
            'growth_rate': growth_rate
        }
        return metrics

    def _build_las(self, metrics: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        w_avg = self.las_weights.get('w_avg', 0.35)
        w_persist = self.las_weights.get('w_persist', 0.25)
        w_replen = self.las_weights.get('w_replen', 0.2)
        w_growth = self.las_weights.get('w_growth', 0.1)

        avg_n = self._normalize_arr(metrics['avg_depth'])
        persist_n = self._normalize_arr(metrics['persistency'])
        growth_n = self._normalize_arr(metrics['growth_rate'])

        las = (w_avg * avg_n +
               w_persist * persist_n +
               w_growth * growth_n)
        return las, {}

    def _select_levels_from_las(self, las_arr: np.ndarray, min_width_bins: int = 3, las_thresh_factor: float = 1.0) -> List[Tuple[int, int, float]]:
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

    def _bin_range_to_price(self, i: int, j: int) -> Tuple[float, float]:
        lower = self.bin_edges[i] if i < len(self.bin_edges) else self.min_price
        upper = self.bin_edges[j + 1] if (j + 1) < len(self.bin_edges) else self.max_price
        return lower, upper

    def _normalize_arr(self, arr) -> np.ndarray:
        a = np.array(arr, dtype=float)
        a = np.nan_to_num(a, nan=0.0)
        mn = np.min(a)
        mx = np.max(a)
        if mx - mn < 1e-12:
            if mx == 0:
                return np.zeros_like(a)
            return np.clip(a / (mx + 1e-12), 0.0, 1.0)
        return (a - mn) / (mx - mn)