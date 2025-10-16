import time
from collections import deque
from math import floor

from utils.functions import MarketProcess


class SmartScalpStrategy(MarketProcess):
    """
    SmartMoney scalper decision-maker.
    Принимает ссылки на:
      - history_market_parser (должен иметь .df DataFrame с минутными свечами)
      - orderbook_parser (должен иметь .orderbook dict)
      - nwe_bounds_indicator, atr_bounds_indicator (возвращают/отдают dict {'upper':..., 'lower':...})
    Подразумевается, что эти парсеры находятся в том же процессе и уже вызываются до вызова run() этого процесса.
    """

    def __init__(
            self,
            history_market_parser,
            orderbook_parser,
            nwe_bounds_indicator,
            atr_bounds_indicator,
            topN=5,
            oi_threshold=0.60,
            w_of_seconds=10,
            sweep_lookback_s=15,
            sweep_recover_s=8,
            sweep_depth_pct=0.003,  # 0.3%
            margin_pct=0.0025,  # 0.25% of price (max margin)
            risk_amount=10.0,  # $10 risk per trade
            take_profit_multiplier=1.5,  # TP = 1.5 * SL distance
            min_stop_distance_pct=0.0005,  # 0.05% минимальная дистанция SL
            cooldown_s=60,
            max_spread_pct=0.002,  # не входить если spread > 0.2%
            depth_limit=50
    ):
        self.history = history_market_parser
        self.ob = orderbook_parser
        self.nwe = nwe_bounds_indicator
        self.atr = atr_bounds_indicator

        # параметры
        self.topN = topN
        self.oi_threshold = oi_threshold
        self.w_of = w_of_seconds
        self.sweep_lookback_s = sweep_lookback_s
        self.sweep_recover_s = sweep_recover_s
        self.sweep_depth_pct = sweep_depth_pct
        self.margin_pct = margin_pct
        self.risk_amount = risk_amount
        self.tp_mult = take_profit_multiplier
        self.min_stop_distance_pct = min_stop_distance_pct
        self.cooldown_s = cooldown_s
        self.max_spread_pct = max_spread_pct

        # внутреннее состояние
        self.prev_orderbook = None
        self.mid_price_history = deque()  # items: (ts, mid_price)
        self.oi_history = deque()  # last w_of seconds OI values
        self.orderflow_events = deque()  # timeline of booleans or counts
        self.last_signal_ts = 0
        self.cooldown_until = 0
        self.last_generated_order = None  # хранит последний рекомендованный ордер
        self._prepared = False
        self.depth_limit = depth_limit

    def prepare(self):
        # инициализация структур, возьмём текущие снапшоты если есть
        try:
            if hasattr(self.ob, 'orderbook'):
                self.prev_orderbook = self.ob.orderbook
        except Exception:
            self.prev_orderbook = None
        self.mid_price_history.clear()
        self.oi_history.clear()
        self.orderflow_events.clear()
        self.last_signal_ts = 0
        self.cooldown_until = 0
        self.last_generated_order = None
        self._prepared = True

    def _get_bounds(self, indicator):
        """
        Пытаемся получить словарь bounds из индикатора. Порядок обращений:
        - если есть метод get_bounds(), вызовем
        - если run() возвращает dict — вызовем run()
        - если есть атрибут 'bounds'/'value'/'result' — возьмём
        """
        # prefer attribute access (indicator may have stored result from its run())

        return indicator.bounds

    def _normalize_orderbook(self, raw):
        """
        Приводим price/amount к float, сортируем явно (биды desc, аски asc),
        и пересчитываем cumulative total.
        """
        if not isinstance(raw, dict):
            return None
        bids = raw.get('bids') or []
        asks = raw.get('asks') or []

        norm_bids = []
        norm_asks = []

        for b in bids:
            # b может быть dict или list [price, amount]
            try:
                if isinstance(b, dict):
                    p = float(b.get('price', b.get('p', 0)))
                    a = float(b.get('amount', b.get('size', 0)))
                elif isinstance(b, (list, tuple)) and len(b) >= 2:
                    p = float(b[0]);
                    a = float(b[1])
                else:
                    continue
            except Exception:
                continue
            norm_bids.append({'price': p, 'amount': a})

        for a_ in asks:
            try:
                if isinstance(a_, dict):
                    p = float(a_.get('price', a_.get('p', 0)))
                    a = float(a_.get('amount', a_.get('size', 0)))
                elif isinstance(a_, (list, tuple)) and len(a_) >= 2:
                    p = float(a_[0]);
                    a = float(a_[1])
                else:
                    continue
            except Exception:
                continue
            norm_asks.append({'price': p, 'amount': a})

        # Явно сортируем: bids desc, asks asc
        norm_bids.sort(key=lambda x: x['price'], reverse=True)
        norm_asks.sort(key=lambda x: x['price'])

        # пересчитать total (кумулятивно)
        cum = 0.0
        for it in norm_bids:
            cum += it['amount']
            it['total'] = cum
        cum = 0.0
        for it in norm_asks:
            cum += it['amount']
            it['total'] = cum

        return {'bids': norm_bids, 'asks': norm_asks}

    def _aggregate_orderbook(self, orderbook, step):
        """
        Агрегируем уровни в бины шагом step (например 50 или 100).
        Возвращает агрегированный стакан с сортировкой.
        """
        if step is None or step <= 0:
            return orderbook

        def agg_side(side_list, reverse_sort=False):
            buckets = {}
            for lvl in side_list:
                p = lvl['price']
                a = lvl['amount']
                # key = округление до ближайшего бина
                key = round(p / step) * step
                buckets.setdefault(key, 0.0)
                buckets[key] += a
            items = [{'price': k, 'amount': v} for k, v in buckets.items()]
            items.sort(key=lambda x: x['price'], reverse=reverse_sort)
            # recompute cumulative total
            cum = 0.0
            for it in items:
                cum += it['amount']
                it['total'] = cum
            return items

        return {
            'bids': agg_side(orderbook.get('bids', []), reverse_sort=True),
            'asks': agg_side(orderbook.get('asks', []), reverse_sort=False)
        }

    def _get_orderbook(self):
        raw = None
        if hasattr(self.ob, 'orderbook'):
            raw = self.ob.orderbook
        else:
            try:
                val = self.ob.run()
                if isinstance(val, dict) and 'bids' in val and 'asks' in val:
                    raw = val
            except Exception:
                raw = None
        if raw is None:
            return None

        norm = self._normalize_orderbook(raw)

        # урезаем глубину если нужно (depth_limit уровней с каждой стороны)
        if self.depth_limit is not None and self.depth_limit > 0:
            norm['bids'] = norm['bids'][:self.depth_limit]
            norm['asks'] = norm['asks'][:self.depth_limit]

        # агрегация в бины при необходимости
        if getattr(self, 'bin_step', None):
            norm = self._aggregate_orderbook(norm, self.bin_step)

        return norm

    def _current_mid_price(self, orderbook):
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            if not bids or not asks:
                return None
            best_bid = bids[0]['price']
            best_ask = asks[0]['price']
            return (best_bid + best_ask) / 2.0
        except Exception:
            return None

    def _oi_topN(self, orderbook, topN):
        bids = orderbook.get('bids', [])[:topN]
        asks = orderbook.get('asks', [])[:topN]
        sum_b = sum(item.get('amount', 0.0) for item in bids)
        sum_a = sum(item.get('amount', 0.0) for item in asks)
        total = sum_b + sum_a
        if total <= 0:
            return 0.5
        return sum_b / total

    def _sum_topN(self, orderbook, topN, side='bids'):
        return sum(item.get('amount', 0.0) for item in orderbook.get(side, [])[:topN])

    def _detect_aggressive_buy(self, prev_ob, cur_ob):
        """
        Простая эвристика агрессивного buy:
          - уменьшение объемов на топ asks (asks consumed) относительно prev
          - и локальный рост mid price
        Возвращает True если наблюдается spike агрессивных покупок
        """
        if prev_ob is None or cur_ob is None:
            return False
        prev_asks_sum = self._sum_topN(prev_ob, self.topN, 'asks')
        cur_asks_sum = self._sum_topN(cur_ob, self.topN, 'asks')
        avg_asks = (prev_asks_sum + cur_asks_sum) / 2.0 if (prev_asks_sum + cur_asks_sum) > 0 else 1.0
        ask_drop = prev_asks_sum - cur_asks_sum
        # relative drop threshold
        rel_drop = ask_drop / avg_asks
        # price move
        prev_mid = self._current_mid_price(prev_ob)
        cur_mid = self._current_mid_price(cur_ob)
        price_up = False
        if prev_mid is not None and cur_mid is not None:
            price_up = (cur_mid - prev_mid) / prev_mid > 0.0002  # >0.02% quick uptick
        # decide
        if rel_drop > 0.15 and price_up:
            return True
        if ask_drop > 0.5 * avg_asks and price_up:
            return True
        return False

    def _detect_bid_pool_near(self, orderbook, stopbase, price, pool_range_pct=0.001, pool_multiplier=2.5):
        """
        Пул бидов рядом со StopBase: суммы бидов в диапазоне [stopbase - pool_range, stopbase + pool_range]
        pool_range_pct — от цены
        pool_multiplier — сколько раз больше среднего объёма topN должно быть
        """
        if orderbook is None:
            return False
        pool_range = max(price * pool_range_pct, 1.0)
        bids = orderbook.get('bids', [])
        # sum amounts within range
        pool_sum = 0.0
        for b in bids:
            p = b.get('price', 0.0)
            if (stopbase - pool_range) <= p <= (stopbase + pool_range):
                pool_sum += b.get('amount', 0.0)
        # average topN bid amount
        avg_topN = 0.0
        topN_bids = bids[:self.topN]
        if topN_bids:
            avg_topN = sum(b.get('amount', 0.0) for b in topN_bids) / len(topN_bids)
        # decide
        if avg_topN <= 0:
            return False
        if pool_sum >= pool_multiplier * avg_topN:
            return True
        return False

    def _detect_sweep(self, stopbase, current_ts):
        """
        Sweep detection:
          - ищем в mid_price_history минимум в последних sweep_lookback_s секундах,
            который был ниже stopbase на sweep_depth_pct
          - и текущая цена восстановилась выше stopbase (т.е. sweep + recovery).
        """
        if not self.mid_price_history:
            return False
        # find min in lookback
        lookback_ts = current_ts - self.sweep_lookback_s
        min_price = None
        min_ts = None
        for ts, price in self.mid_price_history:
            if ts < lookback_ts:
                continue
            if min_price is None or price < min_price:
                min_price = price
                min_ts = ts
        if min_price is None or min_ts is None:
            return False
        # depth
        # if min_price <= stopbase * (1 - sweep_depth_pct) and current price > stopbase -> sweep
        if min_price <= stopbase * (1.0 - self.sweep_depth_pct):
            # ensure recovery happened recently (min_ts to now <= sweep_recover_s)
            if (current_ts - min_ts) <= self.sweep_recover_s:
                # current mid price must be above stopbase (recovered)
                if self.mid_price_history and self.mid_price_history[-1][1] > stopbase:
                    return True
        return False

    def _compute_stopbase_and_margin(self, nwe_bounds, atr_bounds, price):
        # if any missing, return None
        if not (isinstance(nwe_bounds, dict) and isinstance(atr_bounds, dict)):
            return None, None
        nwe_lower = nwe_bounds.get('lower')
        atr_lower = atr_bounds.get('lower')
        if nwe_lower is None or atr_lower is None:
            return None, None
        stopbase = min(nwe_lower, atr_lower)
        # atr width as proxy for ATR tick
        atr_width = (atr_bounds.get('upper', atr_lower) - atr_lower)
        atr_tick = max(atr_width, price * 0.0005)  # fallback to small pct if width zero
        margin = min(0.5 * atr_tick, price * self.margin_pct)
        return stopbase, margin

    def _calc_position_size(self, entry_price, stop_price):
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            return 0, 0.0
        pos_units = floor(self.risk_amount / stop_distance)
        return pos_units, stop_distance

    def run(self, start_time=None, current_time=None, end_time=None):
        try:
            # main periodic call (expected every 1s)
            # 1) grab data
            cur_ts = int(time.time())

            orderbook = self._get_orderbook()
            # update mid price history
            mid = None
            if orderbook:
                mid = self._current_mid_price(orderbook)
                if mid is not None:
                    self.mid_price_history.append((cur_ts, mid))
            # prune old mid history beyond lookback window (a bit larger)
            min_keep = cur_ts - max(self.w_of, self.sweep_lookback_s) - 5
            while self.mid_price_history and self.mid_price_history[0][0] < min_keep:
                self.mid_price_history.popleft()

            # read bounds
            nwe_bounds = self._get_bounds(self.nwe)
            atr_bounds = self._get_bounds(self.atr)

            # basic checks
            if orderbook is None or mid is None or nwe_bounds is None or atr_bounds is None:
                # insufficient data
                print('[SmartMoney] insufficient data ')
                return

            best_bid = orderbook['bids'][0]['price'] if orderbook.get('bids') else None
            best_ask = orderbook['asks'][0]['price'] if orderbook.get('asks') else None
            if best_bid is None or best_ask is None:
                print('[SmartMoney] no best bid or ask ')
                return

            spread = best_ask - best_bid
            spread_pct = spread / mid if mid else 0.0
            if spread_pct > self.max_spread_pct:
                # слишком большой спред — фильтрация
                print('[SmartMoney] spread too large ')
                self.prev_orderbook = orderbook
                return

            # compute stopbase & margin
            stopbase, margin = self._compute_stopbase_and_margin(nwe_bounds, atr_bounds, mid)
            if stopbase is None:
                print('[SmartMoney] unable to compute stopbase ')
                self.prev_orderbook = orderbook
                return

            # compute OI
            oi = self._oi_topN(orderbook, self.topN)
            # update oi_history
            self.oi_history.append((cur_ts, oi))
            while self.oi_history and self.oi_history[0][0] < cur_ts - self.w_of - 2:
                self.oi_history.popleft()

            # orderflow detection (compare prev orderbook)
            aggressive_buy = False
            if self.prev_orderbook is not None:
                aggressive_buy = self._detect_aggressive_buy(self.prev_orderbook, orderbook)
                # push event
                self.orderflow_events.append((cur_ts, aggressive_buy))
                # prune
                while self.orderflow_events and self.orderflow_events[0][0] < cur_ts - self.w_of - 2:
                    self.orderflow_events.popleft()
            else:
                # initialize empty
                self.orderflow_events.append((cur_ts, False))

            # count aggressive buy events in window
            aggr_count = sum(1 for ts, ev in self.orderflow_events if ev and ts >= cur_ts - self.w_of)

            # distance to StopBase
            price = mid
            stop_distance = price - stopbase

            # minimal stop distance absolute and relative checks
            min_stop_dollar = max(price * self.min_stop_distance_pct, 0.5)  # at least $0.5
            if stop_distance <= min_stop_dollar:
                # stop too close — ignore signal
                print('[SmartMoney] stop too close ')
                self.prev_orderbook = orderbook
                return

            # sweep detection
            sweep = self._detect_sweep(stopbase, cur_ts)

            # bid pool detection
            bid_pool = self._detect_bid_pool_near(orderbook, stopbase, price)

            # Decide whether to open LONG (mirror for SHORT can be added similarly)
            can_long = False
            # condition 1: price near StopBase
            cond1 = price <= stopbase + margin
            # condition 2: OI bullish
            cond2 = oi >= self.oi_threshold
            # condition 3: either sweep detected OR bid pool near stopbase
            cond3 = sweep or bid_pool
            # condition 4: orderflow confirmation (>=2 aggressive buys in last w_of)
            cond4 = aggr_count >= 1  # conservative: require at least 1 aggressive event in last W_of
            # condition 5: cooldown check
            cond5 = cur_ts >= self.cooldown_until

            passed = sum([cond1, cond2, cond3, cond4, cond5])
            print(f'[SmartMoney] conds: [{str(cond1)}, {str(cond2)}, {str(cond3)}, {str(cond4)}, {str(cond5)}]')
            if passed >= 4:
                can_long = True

            if can_long:
                entry_price = best_ask  # be slightly aggressive and target ask
                sl_price = stopbase
                pos_units, stop_dist = self._calc_position_size(entry_price, sl_price)
                if pos_units <= 0:
                    # position too small due to tiny risk allocation; skip
                    print('[SmartMoney] position too small ')
                    self.prev_orderbook = orderbook
                    return
                tp_price = entry_price + self.tp_mult * (entry_price - sl_price)
                order = {
                    'ts': cur_ts,
                    'side': 'LONG',
                    'entry_price': entry_price,
                    'size': pos_units,
                    'sl': sl_price,
                    'tp': tp_price,
                    'stop_distance_$': stop_dist,
                    'risk_$': self.risk_amount,
                    'oi': oi,
                    'aggr_count': aggr_count,
                    'sweep': sweep,
                    'bid_pool': bid_pool,
                    'margin': margin,
                    'spread_pct': spread_pct,
                    'notes': 'Auto-generated by SmartMoneyScalper'
                }
                # register order / cooldown
                self.last_generated_order = order
                message = f"[{time.ctime(cur_ts)}] SIGNAL: LONG entry={entry_price:.2f} size={pos_units} sl={sl_price:.2f} tp={tp_price:.2f} stop_dist=${stop_dist:.2f} oi={oi:.2f} aggr={aggr_count} sweep={sweep} bid_pool={bid_pool} spread={spread_pct:.4f}"

                print(message)
                with open('files/decisions_SmartScalpStrategy.txt', 'a') as f:
                    f.write(message + '\n')

                self.last_signal_ts = cur_ts
                self.cooldown_until = cur_ts + self.cooldown_s
                # after generation we do not try to open multiple signals until cooldown expires
                # (external execution system can read last_generated_order and place actual order)
            # update prev orderbook
            self.prev_orderbook = orderbook
        except Exception as e:
            print(f'[SmartMoney] error: {str(e)}')

    # helper for external code to get last generated order
    def get_last_order(self):
        return self.last_generated_order

    # optional: reset cooldown (e.g., after manual intervention)
    def reset_cooldown(self):
        self.cooldown_until = 0