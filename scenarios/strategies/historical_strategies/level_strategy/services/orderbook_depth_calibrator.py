import glob
import glob
import itertools
import json
import random
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Callable, Tuple


class OrderbookDepthCalibrator:
    """
    Класс для автоматической калибровки параметров DefineOrderbookDepthService.
    Поддерживает grid search и random search, простую оценку bounce-качества найденных уровней.
    """

    def __init__(
        self,
        service_cls: Callable[..., Any],
        base_kwargs: Dict[str, Any] = None,
        files_glob: str = "files/orderbook/beautifulsoup/beautifulsoup_coinglass_orderbook_BTC-USDT-*.json",
        filename_regex: str = r'beautifulsoup_coinglass_orderbook_BTC-USDT-(\d{4}:\d{2}:\d{2})_(\d{2}:\d{2})\.json$',
        tz_aware: bool = False
    ):
        """
        service_cls: класс (или фабрика), который имеет интерфейс DefineOrderbookDepthService(...)
                     и метод define_depth_level(start_time, current_time) -> list of levels
        base_kwargs: аргументы, которые будут передаваться в конструктор сервиса для всех испытаний
        files_glob & filename_regex: где взять набор snapshot-файлов и как парсить timestamp
        """
        self.service_cls = service_cls
        self.base_kwargs = base_kwargs or {}
        self.files_glob = files_glob
        self.filename_regex = re.compile(filename_regex)
        self.tz_aware = tz_aware

        # Pre-scan files: map timestamp -> filepath, and build sorted list of timestamps
        self.time_file_list = self._scan_files()
        # Build midprice series cache: timestamp -> midprice (or None)
        self.midprice_by_ts = self._build_midprice_cache()

    # ---------- File / snapshots helpers ----------
    def _scan_files(self) -> List[Tuple[datetime, str]]:
        files = glob.glob(self.files_glob)
        tfs = []
        for f in files:
            m = self.filename_regex.search(f)
            if not m:
                continue
            date_part = m.group(1)
            time_part = m.group(2)
            try:
                ts = datetime.strptime(f"{date_part}_{time_part}", "%Y:%m:%d_%H:%M")
            except Exception:
                continue
            tfs.append((ts, f))
        tfs.sort(key=lambda x: x[0])
        return tfs

    def _build_midprice_cache(self) -> Dict[datetime, float]:
        """
        Для каждого snapshot вычисляет midprice = 0.5*(best_bid+best_ask) если доступны.
        Если один из best отсутствует, ставим None.
        """
        cache = {}
        for ts, fpath in self.time_file_list:
            try:
                with open(fpath, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
            except Exception:
                cache[ts] = None
                continue
            bids = data.get("bids", []) or []
            asks = data.get("asks", []) or []
            # bids and asks sorted by descending (as user said). However compute robustly:
            best_bid = None
            best_ask = None
            if bids:
                # find max price in bids (just in case)
                try:
                    best_bid = max(float(x['price']) for x in bids if 'price' in x)
                except Exception:
                    best_bid = None
            if asks:
                try:
                    best_ask = min(float(x['price']) for x in asks if 'price' in x)
                except Exception:
                    best_ask = None
            if best_bid is None or best_ask is None:
                cache[ts] = None
            else:
                cache[ts] = 0.5 * (best_bid + best_ask)
        return cache

    def _get_sorted_timestamps(self) -> List[datetime]:
        return [ts for ts, _ in self.time_file_list]

    def _get_future_midprices(self, from_ts: datetime, to_ts: datetime) -> List[Tuple[datetime, float]]:
        """
        Возвращает список (ts, midprice) для snapshots с from_ts < ts <= to_ts
        """
        out = []
        for ts in self._get_sorted_timestamps():
            if from_ts < ts <= to_ts:
                mp = self.midprice_by_ts.get(ts, None)
                if mp is not None:
                    out.append((ts, mp))
        return out

    # ---------- Evaluation / scoring ----------
    def _is_level_successful(
        self,
        level: Dict[str, Any],
        eval_end_time: datetime,
        horizon_minutes: int = 30,
        rebound_pct: float = 0.002,
        require_touch: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Проверка одного уровня: ищем в будущем (eval_end_time, eval_end_time + horizon_minutes]
        первое касание уровня (для bid: price <= upper, для ask: price >= lower).
        Если касание произошло (или если require_touch==False), проверяем, произойдёт ли rebound:
         - для bid (support): после касания min_price -> затем increase >= rebound_pct * price_from_hit
         - для ask (resistance): после касания max_price -> затем decrease >= rebound_pct * price_from_hit
        Возвращает (success_flag, info).
        """
        side = level.get('side', 'bid')
        lower = float(level['lower'])
        upper = float(level['upper'])
        # center price used for thresholds
        center = 0.5 * (lower + upper)
        horizon_end = eval_end_time + timedelta(minutes=horizon_minutes)
        future = self._get_future_midprices(eval_end_time, horizon_end)  # list of (ts, mid)
        if not future:
            return False, {'reason': 'no_future_data'}

        # find first touch event
        hit_idx = None
        hit_price = None
        for idx, (ts, mp) in enumerate(future):
            if side == 'bid':
                # touch when midprice <= upper (we consider upper edge of support)
                if mp <= upper:
                    hit_idx = idx
                    hit_price = mp
                    hit_ts = ts
                    break
            else:
                # ask: touch when midprice >= lower (price reached/above resistance)
                if mp >= lower:
                    hit_idx = idx
                    hit_price = mp
                    hit_ts = ts
                    break

        if hit_idx is None:
            # level not tested in horizon
            if require_touch:
                return False, {'reason': 'not_touched'}
            else:
                # if we don't require touch, consider the first point as baseline for rebound test
                hit_idx = 0
                hit_price = future[0][1]
                hit_ts = future[0][0]

        # Now check rebound after hit
        # For support (bid): price must increase by rebound_pct relative to hit_price
        # For resistance (ask): price must decrease by rebound_pct relative to hit_price
        success = False
        rebound_target_up = hit_price * (1.0 + rebound_pct)
        rebound_target_down = hit_price * (1.0 - rebound_pct)
        for j in range(hit_idx + 1, len(future)):
            ts_j, mp_j = future[j]
            if side == 'bid':
                if mp_j >= rebound_target_up:
                    success = True
                    rebound_ts = ts_j
                    rebound_price = mp_j
                    break
            else:
                if mp_j <= rebound_target_down:
                    success = True
                    rebound_ts = ts_j
                    rebound_price = mp_j
                    break

        info = {
            'touched': hit_idx is not None,
            'hit_price': hit_price,
            'hit_ts': hit_ts if hit_idx is not None else None,
            'success': success
        }
        if success:
            info.update({'rebound_ts': rebound_ts, 'rebound_price': rebound_price})
        return success, info

    def _evaluate_params_on_windows(
        self,
        params: Dict[str, Any],
        eval_end_times: List[datetime],
        window_minutes: int,
        horizon_minutes: int,
        rebound_pct: float,
        require_touch: bool = True,
        max_levels_per_window: int = 20
    ) -> Dict[str, Any]:
        """
        Запускает сервис с указанными params по каждому оконечному времени, собирает уровни и считает:
         - total_levels_tested, touched, successful
        Возвращает dict с метриками и деталями.
        """
        # instantiate service with provided params merged into base kwargs
        svc_kwargs = dict(self.base_kwargs)
        svc_kwargs.update(params)
        service = self.service_cls(**svc_kwargs)

        total_levels = 0
        total_touched = 0
        total_success = 0
        details = []  # optional small list of per-level results (limited)

        for end_ts in eval_end_times:
            start_ts = end_ts - timedelta(minutes=window_minutes)
            try:
                levels = service.define_depth_level(start_ts, end_ts)
            except Exception as e:
                # if service fails for some param combo, penalize
                return {'error': str(e), 'score': -1.0, 'params': params}

            if not levels:
                continue
            # optionally limit how many levels per window we evaluate to cap runtime
            for lvl in levels[:max_levels_per_window]:
                total_levels += 1
                success, info = self._is_level_successful(
                    lvl,
                    eval_end_time=end_ts,
                    horizon_minutes=horizon_minutes,
                    rebound_pct=rebound_pct,
                    require_touch=require_touch
                )
                if info.get('touched'):
                    total_touched += 1
                if success:
                    total_success += 1
                if len(details) < 200:
                    d = {'end_ts': end_ts, 'level': lvl, 'success': success, 'info': info}
                    details.append(d)

        # Compute metrics
        tests = total_levels
        touched = total_touched
        succ = total_success
        # Score: success rate among tested levels (if none tested -> score 0)
        score = float(succ) / tests if tests > 0 else 0.0
        # Coverage: fraction of levels that were touched at least once
        touch_rate = float(touched) / tests if tests > 0 else 0.0
        # You can combine score*touch_rate to penalize low coverage:
        combined = score * (0.5 + 0.5 * touch_rate)  # weighted

        return {
            'params': params,
            'total_levels': tests,
            'touched': touched,
            'successful': succ,
            'score': score,
            'touch_rate': touch_rate,
            'combined_score': combined,
            'details': details
        }

    # ---------- Search / calibration ----------
    def calibrate_grid_search(
        self,
        param_grid: Dict[str, List[Any]],
        eval_start: datetime,
        eval_end: datetime,
        window_minutes: int = 10,
        step_minutes: int = 5,
        horizon_minutes: int = 30,
        rebound_pct: float = 0.002,
        require_touch: bool = True,
        max_combinations: int = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Grid search over cartesian product of param_grid values.
        param_grid: dict param_name -> list(values)
        eval windows: from eval_start to eval_end inclusive, one end_time every step_minutes
        Возвращает словарь с лучшими параметрами и full history.
        """
        # build evaluation end_times
        all_ts = self._get_sorted_timestamps()
        if not all_ts:
            raise ValueError("No snapshots found for calibration.")
        # pick end_times lying in [eval_start, eval_end]
        eval_end_times = [ts for ts in all_ts if eval_start <= ts <= eval_end]
        # sample by step_minutes
        if step_minutes > 0:
            filtered = []
            last = None
            for ts in eval_end_times:
                if last is None or (ts - last).total_seconds() >= step_minutes * 60:
                    filtered.append(ts)
                    last = ts
            eval_end_times = filtered

        if not eval_end_times:
            raise ValueError("No evaluation end times in the provided interval.")

        # iterate grid
        keys = list(param_grid.keys())
        combos = list(itertools.product(*(param_grid[k] for k in keys)))
        if max_combinations is not None:
            combos = combos[:max_combinations]

        history = []
        best = None
        for idx, combo in enumerate(combos):
            params = {k: v for k, v in zip(keys, combo)}
            res = self._evaluate_params_on_windows(
                params=params,
                eval_end_times=eval_end_times,
                window_minutes=window_minutes,
                horizon_minutes=horizon_minutes,
                rebound_pct=rebound_pct,
                require_touch=require_touch
            )
            history.append(res)
            score = res.get('combined_score', -1.0)
            if best is None or score > best['combined_score']:
                best = res
            if verbose and idx % 10 == 0:
                print(f"[grid] tested {idx+1}/{len(combos)} combos, current best combined_score={best['combined_score']:.4f}")
        return {'best': best, 'history': history}

    def calibrate_random_search(
        self,
        param_space: Dict[str, Any],
        n_iters: int,
        eval_start: datetime,
        eval_end: datetime,
        window_minutes: int = 10,
        step_minutes: int = 5,
        horizon_minutes: int = 30,
        rebound_pct: float = 0.002,
        require_touch: bool = True,
        verbose: bool = True,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Random search. param_space: key -> either list(values) or tuple(min,max,type)
        Example param_space:
           {'W_price_bins': [1,3,5], 'min_width_bins': (1,6,int), 'las_thresh_factor': (0.3,1.5,float)}
        """
        random.seed(seed)
        # build evaluation end_times same as grid search
        all_ts = self._get_sorted_timestamps()
        if not all_ts:
            raise ValueError("No snapshots found for calibration.")
        eval_end_times = [ts for ts in all_ts if eval_start <= ts <= eval_end]
        if step_minutes > 0:
            filtered = []
            last = None
            for ts in eval_end_times:
                if last is None or (ts - last).total_seconds() >= step_minutes * 60:
                    filtered.append(ts)
                    last = ts
            eval_end_times = filtered

        if not eval_end_times:
            raise ValueError("No evaluation end times in the provided interval.")

        def sample_value(spec):
            # spec can be list or (min,max,type)
            if isinstance(spec, list) or isinstance(spec, tuple) and len(spec) > 2:
                # treat as list
                return random.choice(list(spec))
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                # ambiguous: treat as numeric range float
                a, b = spec
                return random.uniform(a, b)
            if isinstance(spec, tuple) and len(spec) == 3:
                a, b, t = spec
                if t is int:
                    return random.randint(int(a), int(b))
                elif t is float:
                    return random.uniform(float(a), float(b))
                else:
                    return random.choice([a, b])
            # fallback
            if hasattr(spec, '__iter__'):
                return random.choice(list(spec))
            return spec

        history = []
        best = None
        for it in range(n_iters):
            params = {}
            for k, spec in param_space.items():
                params[k] = sample_value(spec)
            res = self._evaluate_params_on_windows(
                params=params,
                eval_end_times=eval_end_times,
                window_minutes=window_minutes,
                horizon_minutes=horizon_minutes,
                rebound_pct=rebound_pct,
                require_touch=require_touch
            )
            history.append(res)
            score = res.get('combined_score', -1.0)
            if best is None or score > best['combined_score']:
                best = res
            if verbose and it % 10 == 0:
                print(f"[random] iter {it+1}/{n_iters}, current best combined_score={best['combined_score']:.4f}")
        return {'best': best, 'history': history}