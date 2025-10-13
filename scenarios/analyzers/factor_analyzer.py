import os
import datetime
from collections import deque

class MarketProcess:
    def prepare(self):
        pass

    def run(self):
        pass


class FactorAnalyzer(MarketProcess):
    """
    Rule-based factor analyser that decides whether to enter a scalp-long.
    Writes entry signals to console and appends each as a new line into 'files/longs.txt'.
    """

    def __init__(
        self,
        history_market_parser,
        orderbook_parser,
        nwe_bounds_indicator,
        atr_bounds_indicator,
        # configurable thresholds
        top_n_depth=5,
        min_depth_imbalance=0.15,
        min_top_bid_increase_pct=0.12,
        max_spread_pct=0.0006,
        touch_eps_rel=0.0005,
        touch_eps_abs=0.5,
        cooldown_s=3,
        min_bid_total=0.001,
        out_filepath="files/longs.txt",
    ):
        self.history_market_parser = history_market_parser
        self.orderbook_parser = orderbook_parser
        self.nwe = nwe_bounds_indicator
        self.atr = atr_bounds_indicator

        # thresholds / params
        self.top_n_depth = top_n_depth
        self.min_depth_imbalance = min_depth_imbalance
        self.min_top_bid_increase_pct = min_top_bid_increase_pct
        self.max_spread_pct = max_spread_pct
        self.touch_eps_rel = touch_eps_rel
        self.touch_eps_abs = touch_eps_abs
        self.cooldown_s = cooldown_s
        self.min_bid_total = min_bid_total

        # output file
        self.out_filepath = out_filepath

        # internal buffers for micro-features (keep last few mid prices and top sizes)
        self.prev_mids = deque(maxlen=10)
        self.prev_top_bid_amount = None
        self.prev_top_ask_amount = None
        self.prev_time = None
        self.last_signal_time = None

    def _get_bounds(self, obj):
        if obj is None:
            return None
        # try direct attribute .bounds
        try:
            b = getattr(obj, "bounds")
            if callable(b):
                b = b()
            if b:
                return b
        except Exception:
            pass
        # try method get_bounds()
        try:
            getb = getattr(obj, "get_bounds", None)
            if callable(getb):
                b = getb()
                if b:
                    return b
        except Exception:
            pass
        # try attribute .last or .current or .latest
        for attr in ("last", "current", "latest"):
            try:
                b = getattr(obj, attr)
                if callable(b):
                    b = b()
                if b:
                    return b
            except Exception:
                pass
        # try direct attributes .upper .lower
        try:
            upper = getattr(obj, "upper")
            lower = getattr(obj, "lower")
            return {"upper": upper, "lower": lower}
        except Exception:
            pass
        # fallback: if object itself is dict-like
        try:
            if isinstance(obj, dict) and "lower" in obj and "upper" in obj:
                return obj
        except Exception:
            pass
        return None

    def _log_signal(self, text):
        """
        Append text as a new line to self.out_filepath.
        Ensure directory exists. Fail silently on errors to avoid breaking main loop.
        """
        try:
            folder = os.path.dirname(self.out_filepath)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            # append line with newline
            with open(self.out_filepath, "a", encoding="utf-8") as f:
                f.write(text + "\n")
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    # if fsync not permitted, ignore
                    pass
        except Exception:
            # do not raise — logging should not break runtime
            pass

    def prepare(self):
        pass

    def run(self):
        # Ensure data availability
        try:
            df = getattr(self.history_market_parser, "df", None)
            orderbook = getattr(self.orderbook_parser, "orderbook", None)
        except Exception:
            df = None
            orderbook = None

        if df is None:
            return
        try:
            if hasattr(df, "empty") and df.empty:
                return
        except Exception:
            pass

        if orderbook is None or not orderbook.get("bids") or not orderbook.get("asks"):
            return

        # get bounds
        atr_bounds = self._get_bounds(self.atr)
        nwe_bounds = self._get_bounds(self.nwe)
        if atr_bounds is None or nwe_bounds is None:
            return

        try:
            # accept dict-like or object-like
            if isinstance(atr_bounds, dict):
                lower_atr = float(atr_bounds.get("lower"))
            else:
                lower_atr = float(atr_bounds["lower"]) if isinstance(atr_bounds, (list, tuple)) else float(getattr(atr_bounds, "lower", None))
            if isinstance(nwe_bounds, dict):
                lower_nwe = float(nwe_bounds.get("lower"))
            else:
                lower_nwe = float(nwe_bounds["lower"]) if isinstance(nwe_bounds, (list, tuple)) else float(getattr(nwe_bounds, "lower", None))
        except Exception:
            try:
                lower_atr = float(atr_bounds.lower)
                lower_nwe = float(nwe_bounds.lower)
            except Exception:
                return

        min_lower = min(lower_atr, lower_nwe)

        # obtain top-of-book
        try:
            top_bid = orderbook["bids"][0]
            top_ask = orderbook["asks"][0]
            best_bid = float(top_bid["price"])
            best_ask = float(top_ask["price"])
            top_bid_amount = float(top_bid.get("amount", top_bid.get("quantity", 0.0)))
            top_ask_amount = float(top_ask.get("amount", top_ask.get("quantity", 0.0)))
        except Exception:
            return

        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_pct = spread / mid if mid > 0 else float("inf")

        # update buffers
        now = datetime.datetime.utcnow()
        self.prev_mids.append(mid)
        prev_mid = self.prev_mids[-2] if len(self.prev_mids) >= 2 else None

        # compute top N depth sums
        bids = orderbook.get("bids", [])[: self.top_n_depth]
        asks = orderbook.get("asks", [])[: self.top_n_depth]
        sum_bid = sum(float(b.get("amount", 0.0)) for b in bids)
        sum_ask = sum(float(a.get("amount", 0.0)) for a in asks)
        total_depth = sum_bid + sum_ask if (sum_bid + sum_ask) > 0 else 1e-9
        depth_imbalance = (sum_bid - sum_ask) / total_depth

        # change of top bid size vs previous tick
        top_bid_increase_pct = 0.0
        if self.prev_top_bid_amount is not None and self.prev_top_bid_amount > 0:
            top_bid_increase_pct = (top_bid_amount - self.prev_top_bid_amount) / self.prev_top_bid_amount

        # top ask decrease percent
        top_ask_decrease_pct = 0.0
        if self.prev_top_ask_amount is not None and self.prev_top_ask_amount > 0:
            top_ask_decrease_pct = (self.prev_top_ask_amount - top_ask_amount) / self.prev_top_ask_amount

        # micro-return since prev tick
        micro_return = None
        if prev_mid is not None and prev_mid > 0:
            micro_return = (mid - prev_mid) / prev_mid

        # Distance to min lower
        distance_abs = mid - min_lower
        distance_rel = distance_abs / mid if mid > 0 else 1.0
        eps = max(self.touch_eps_rel * mid, self.touch_eps_abs)  # absolute epsilon
        touch_condition = (mid <= min_lower + eps) or (distance_rel <= self.touch_eps_rel)

        # Cooldown check
        if self.last_signal_time is not None:
            elapsed_since_last = (now - self.last_signal_time).total_seconds()
        else:
            elapsed_since_last = 1e9

        # Safety checks: spread, minimal depth
        if total_depth < self.min_bid_total:
            # illiquid snapshot
            self.prev_top_bid_amount = top_bid_amount
            self.prev_top_ask_amount = top_ask_amount
            return

        if spread_pct > self.max_spread_pct:
            # too wide spread
            self.prev_top_bid_amount = top_bid_amount
            self.prev_top_ask_amount = top_ask_amount
            return

        # Decision logic (conservative, multiple confirmations)
        buy_pressure_confirm = (
            (depth_imbalance >= self.min_depth_imbalance)
            and (top_bid_increase_pct >= self.min_top_bid_increase_pct or top_ask_decrease_pct >= self.min_top_bid_increase_pct)
        )

        # Avoid if strong negative micro momentum
        momentum_ok = True
        if micro_return is not None and micro_return < -0.003:
            momentum_ok = False

        # final condition: price touched lower bound AND microstructure confirms buying pressure AND momentum not strongly negative
        if touch_condition and buy_pressure_confirm and momentum_ok and elapsed_since_last >= self.cooldown_s:
            reason = {
                "mid": round(mid, 2),
                "min_lower": round(min_lower, 2),
                "distance_abs": round(distance_abs, 4),
                "distance_rel": round(distance_rel, 6),
                "depth_imbalance": round(depth_imbalance, 4),
                "top_bid_increase_pct": round(top_bid_increase_pct, 4),
                "top_ask_decrease_pct": round(top_ask_decrease_pct, 4),
                "spread_pct": round(spread_pct, 6),
            }
            tstr = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            msg = f"[ENTRY_SIGNAL] time={tstr} instrument=BTCUSDT mid={mid:.2f} min_lower={min_lower:.2f} reason={reason}"

            # print to console
            print(msg)

            # append to file
            self._log_signal(msg)

            # update last signal time
            self.last_signal_time = now

        # store prevs for next tick
        self.prev_top_bid_amount = top_bid_amount
        self.prev_top_ask_amount = top_ask_amount
        self.prev_time = now