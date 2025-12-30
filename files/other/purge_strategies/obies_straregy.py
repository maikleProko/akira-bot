import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta

from utils.core.functions import MarketProcess


class ObiesStrategy(MarketProcess):
    """
    Стратегия скальпинга, использующая:
      - history_market_parser (минутные свечи, self.df)
      - orderbook_parser (посекундные снимки стакана, self.orderbook)
      - nwe_bounds_indicator (dict {'upper', 'lower'})
      - atr_bounds_indicator (dict {'upper', 'lower'})
    Все парсеры передаются в __init__.
    """

    def __init__(self,
                 history_market_parser,
                 orderbook_parser,
                 nwe_bounds_indicator,
                 atr_bounds_indicator,
                 *,
                 topN=5,
                 imbalance_top5_threshold=0.4,
                 imbalance_top1_threshold=0.6,
                 aggressor_ratio_threshold=0.65,
                 aggressor_window_s=5,
                 orderbook_window_s=5,
                 spread_threshold=3.0,    # в $; подстройте под BTCUSDT
                 min_top5_depth_usd=5000, # минимальная ликвидность
                 max_stop_usd=10.0,
                 tp_usd=15.0,
                 cooldown_s=30,
                 max_daily_trades=100,
                 logger=None):
        self.history_market_parser = history_market_parser
        self.orderbook_parser = orderbook_parser
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator

        # параметры стратегии
        self.topN = topN
        self.imbalance_top5_threshold = imbalance_top5_threshold
        self.imbalance_top1_threshold = imbalance_top1_threshold
        self.aggressor_ratio_threshold = aggressor_ratio_threshold
        self.aggressor_window_s = aggressor_window_s
        self.orderbook_window_s = orderbook_window_s
        self.spread_threshold = spread_threshold
        self.min_top5_depth_usd = min_top5_depth_usd
        self.max_stop_usd = max_stop_usd
        self.tp_usd = tp_usd
        self.cooldown_s = cooldown_s
        self.max_daily_trades = max_daily_trades

        # внутреннее состояние
        self.ob_history = deque()  # (timestamp, orderbook_snapshot)
        self.ob_history_ttl = max(self.aggressor_window_s, self.orderbook_window_s) + 2
        self.last_trade_time = None
        self.open_trades = []  # list of dicts: entry, sl, tp, size, side, opened_at
        self.trades_today = []
        self.daily_date = datetime.now().date()

        # логирование
        self.log = logger or logging.getLogger('ScalpingDecision')
        self.log.setLevel(logging.INFO)

        # вспомогательные
        self.stats = defaultdict(int)  # counters
        super().__init__()  # если базовый класс требует init

    def prepare(self, start_time=None, end_time=None):
        # вызывается однократно при старте
        self.log.info("ScalpingDecision prepared. Params: topN=%s, agressor_window_s=%s", self.topN, self.aggressor_window_s)

    def run_realtime(self):
        # главный цикл - вызывается каждую секунду (periodic)
        try:
            # Обновим дневную статистику
            now = datetime.now()
            if now.date() != self.daily_date:
                self.daily_date = now.date()
                self.trades_today.clear()
                print('[ObiesStrategy] Not current day')
                self.log.info("New day: reset trades_today")

            # получаем данные от парсеров - ожидается что они уже обновили свои self.* в run() ранее
            ob = getattr(self.orderbook_parser, 'orderbook', None)
            nwe = getattr(self.nwe_bounds_indicator, 'bounds', None) or getattr(self.nwe_bounds_indicator, 'last_bounds', None)
            atr = getattr(self.atr_bounds_indicator, 'bounds', None) or getattr(self.atr_bounds_indicator, 'last_bounds', None)
            df = getattr(self.history_market_parser, 'df', None)

            if ob is None or nwe is None or atr is None or df is None:
                self.log.debug("One of required inputs is missing (ob=%s nwe=%s atr=%s df=%s)", bool(ob), bool(nwe), bool(atr), bool(df))
                # попытка очистить историю и выйти
                self._push_orderbook_snapshot(ob)
                self._update_open_trades()
                print(f'[ObiesStrategy] havent all')
                return

            # сортируем/нормализуем стакан и добавляем в историю
            ob_norm = self._normalize_orderbook(ob)
            self._push_orderbook_snapshot(ob_norm)

            # вычисляем метрики
            metrics = self._compute_metrics(ob_norm)

            # рассчитываем microprice (входная цена)
            microprice = metrics.get('microprice')
            if microprice is None:
                self.log.debug("No microprice, skipping")
                print(f'[ObiesStrategy] no microprice')
                self._update_open_trades()
                return

            # lower bounds
            lwe_lower = nwe.get('lower')
            atr_lower = atr.get('lower')
            if lwe_lower is None or atr_lower is None:
                self.log.debug("Bounds missing")
                self._update_open_trades()
                print(f'[ObiesStrategy] bounds missing')
                return

            min_lower = min(lwe_lower, atr_lower)

            # проверка сигнала long (зеркально для short можно добавить, здесь реализуем long; легко расширяется)
            can_enter, reasons = self._check_long_conditions(microprice, min_lower, metrics, nwe, atr, df)

            # проверка ограничений дневных / cooldown
            if can_enter:
                if self.last_trade_time and (datetime.now() - self.last_trade_time).total_seconds() < self.cooldown_s:
                    can_enter = False
                    reasons.append("cooldown_active")
                if len(self.trades_today) >= self.max_daily_trades:
                    can_enter = False
                    reasons.append("daily_limit_reached")

            if can_enter:
                self._open_trade(side='long', entry_price=microprice, sl_distance=self.max_stop_usd, tp_distance=self.tp_usd)
                self.last_trade_time = datetime.now()
                self.trades_today.append({'time': self.last_trade_time, 'side': 'long'})
                self.stats['trades_total'] += 1
                self.log.info("Opened LONG at %.2f (min_lower=%.2f). Metrics: %s", microprice, min_lower, metrics)

                print(f'[ObiesStrategy] LONG [5/5]')
                with open('files/decisions/decisions_ObiesStrategy.txt', 'a') as f:
                    f.write(str(datetime.now()) + '\n')
            else:
                # для отладки: лог причины отказа иногда
                if self.stats['tick'] % 5 == 0:
                    print(f'[ObiesStrategy] CANCEL by conds: {str(reasons)}')
                    self.log.debug("No entry. Reasons: %s -- metrics: %s", reasons, metrics)

            # обновляем открытые позиции (закрытие по tp/sl)
            self._update_open_trades()

            self.stats['tick'] += 1

        except Exception as e:
            print('error' + str(e))
            print(f'[ObiesStrategy] error: {str(e)}')
            self.log.exception("Error in ScalpingDecision.run: %s", e)

    # ---------------------------
    # Вспомогательные методы
    # ---------------------------

    def _normalize_orderbook(self, ob):
        """
        Возвращаем нормализованную структуру:
        {'bids': [{'price': p, 'amount': a, 'total': t}, ...], 'asks': [...]}
        bids - отсортированы по убыванию цен (best bid first)
        asks - отсортированы по возрастанию цен (best ask first)
        """
        if ob is None:
            return None
        bids = ob.get('bids', [])[:]  # ожидаются dicts
        asks = ob.get('asks', [])[:]

        try:
            bids_sorted = sorted(bids, key=lambda x: float(x['price']), reverse=True)
            asks_sorted = sorted(asks, key=lambda x: float(x['price']))
        except Exception:
            # если формат нестандартный - вернуть как есть
            bids_sorted = bids
            asks_sorted = asks

        return {'bids': bids_sorted, 'asks': asks_sorted, 'timestamp': datetime.now()}

    def _push_orderbook_snapshot(self, ob_norm):
        # держим историю снимков стакана для окна агрессора
        ts = datetime.now()
        if ob_norm is None:
            # просто ограничим длину
            while len(self.ob_history) > self.ob_history_ttl:
                self.ob_history.popleft()
            return
        self.ob_history.append((ts, ob_norm))
        # удаляем старые
        cutoff = ts - timedelta(seconds=self.ob_history_ttl + 2)
        while self.ob_history and self.ob_history[0][0] < cutoff:
            self.ob_history.popleft()

    def _compute_metrics(self, ob_norm):
        """
        Вычисляем:
          - best bid/ask, topN sums, imbalance_top1/top5
          - microprice
          - spread
          - estimated buy/sell volume (aggressor) за агрессор_window_s, aggressor_ratio
          - depth change rate (delta top5)
        """
        result = {}
        bids = ob_norm['bids']
        asks = ob_norm['asks']
        if not bids or not asks:
            return result

        best_bid = float(bids[0]['price'])
        best_ask = float(asks[0]['price'])
        bid1_size = float(bids[0].get('amount', 0.0))
        ask1_size = float(asks[0].get('amount', 0.0))

        # compute topN sums
        topN = self.topN
        bid_topN_sum = sum(float(x.get('amount', 0.0)) for x in bids[:topN])
        ask_topN_sum = sum(float(x.get('amount', 0.0)) for x in asks[:topN])

        # imbalance
        denom = (bid_topN_sum + ask_topN_sum)
        imbalance_top5 = None
        if denom > 0:
            imbalance_top5 = (bid_topN_sum - ask_topN_sum) / denom

        denom1 = (bid1_size + ask1_size)
        imbalance_top1 = None
        if denom1 > 0:
            imbalance_top1 = (bid1_size - ask1_size) / denom1

        # microprice weighted by sizes at top
        microprice = None
        if (ask1_size + bid1_size) > 0:
            microprice = (best_bid * ask1_size + best_ask * bid1_size) / (ask1_size + bid1_size)

        spread = best_ask - best_bid

        # estimate topN depth in USD (approx using midprice)
        mid = (best_bid + best_ask) / 2.0
        top5_depth_usd = (bid_topN_sum + ask_topN_sum) * mid

        # compute aggressor ratio over last W seconds using ob_history differences
        buy_vol_est, sell_vol_est = self._estimate_executed_from_orderbook()
        aggressor_ratio = None
        if buy_vol_est + sell_vol_est > 0:
            aggressor_ratio = buy_vol_est / (buy_vol_est + sell_vol_est)

        # depth change rate: compare oldest and newest snapshots in ob_history
        depth_change_rate = 0.0
        if len(self.ob_history) >= 2:
            oldest = self.ob_history[0][1]
            newest = self.ob_history[-1][1]
            bid_old = sum(float(x.get('amount', 0.0)) for x in oldest['bids'][:topN])
            ask_old = sum(float(x.get('amount', 0.0)) for x in oldest['asks'][:topN])
            bid_new = sum(float(x.get('amount', 0.0)) for x in newest['bids'][:topN])
            ask_new = sum(float(x.get('amount', 0.0)) for x in newest['asks'][:topN])
            depth_change_rate = ((bid_new + ask_new) - (bid_old + ask_old)) / max(1.0, (bid_old + ask_old))

        result.update({
            'best_bid': best_bid,
            'best_ask': best_ask,
            'bid1_size': bid1_size,
            'ask1_size': ask1_size,
            'bid_topN_sum': bid_topN_sum,
            'ask_topN_sum': ask_topN_sum,
            'imbalance_top5': imbalance_top5,
            'imbalance_top1': imbalance_top1,
            'microprice': microprice,
            'spread': spread,
            'top5_depth_usd': top5_depth_usd,
            'buy_vol_est': buy_vol_est,
            'sell_vol_est': sell_vol_est,
            'aggressor_ratio': aggressor_ratio,
            'depth_change_rate': depth_change_rate,
        })
        return result

    def _estimate_executed_from_orderbook(self):
        """
        Простая эвристика оценки объёма агрессивных покупок/продаж по изменениям top1 amount
        на последовательных snapshot'ах за окно aggressor_window_s.
        buy executed ~ уменьшение ask1 при постоянной/приближающейся best_ask price.
        sell executed ~ уменьшение bid1 при постоянной/приближающейся best_bid price.
        """
        buy_vol = 0.0
        sell_vol = 0.0
        if len(self.ob_history) < 2:
            return buy_vol, sell_vol

        cutoff = datetime.now() - timedelta(seconds=self.aggressor_window_s)
        relevant = [item for item in self.ob_history if item[0] >= cutoff]
        if len(relevant) < 2:
            relevant = list(self.ob_history)[-2:]

        for i in range(1, len(relevant)):
            prev = relevant[i-1][1]
            cur = relevant[i][1]
            # best asks/bids
            prev_ask = prev['asks'][0] if prev['asks'] else None
            cur_ask = cur['asks'][0] if cur['asks'] else None
            prev_bid = prev['bids'][0] if prev['bids'] else None
            cur_bid = cur['bids'][0] if cur['bids'] else None

            # estimate buys: ask size decreased at same price (someone bought)
            if prev_ask and cur_ask:
                if float(prev_ask['price']) == float(cur_ask['price']):
                    delta = float(max(0.0, prev_ask.get('amount', 0.0) - cur_ask.get('amount', 0.0)))
                    buy_vol += delta
                else:
                    # если цена сдвинулась вниз и ask исчез — сложнее интерпретировать; пропускаем
                    pass

            # estimate sells: bid size decreased at same price (someone sold)
            if prev_bid and cur_bid:
                if float(prev_bid['price']) == float(cur_bid['price']):
                    delta = float(max(0.0, prev_bid.get('amount', 0.0) - cur_bid.get('amount', 0.0)))
                    sell_vol += delta
                else:
                    pass

        return buy_vol, sell_vol

    def _check_long_conditions(self, microprice, min_lower, metrics, nwe, atr, df):
        """
        Проверяем все условия для входа в LONG (A..E).
        Возвращает (can_enter: bool, reasons: list[str])
        """
        reasons = []
        passed_count = 0

        # 1) расстояние до min_lower <= max_stop_usd
        distance = microprice - min_lower
        if distance < 0:
            reasons.append(f"below_min_lower distance={distance:.2f}")
        elif distance > self.max_stop_usd:
            reasons.append(f"distance_to_lower_too_large {distance:.2f} > {self.max_stop_usd}")
        else:
            passed_count += 1

        # 2) imbalance (orderbook absorption)
        imb5 = metrics.get('imbalance_top5') or 0.0
        imb1 = metrics.get('imbalance_top1') or 0.0
        if imb5 >= self.imbalance_top5_threshold or imb1 >= self.imbalance_top1_threshold:
            passed_count += 1
        else:
            reasons.append(f"imbalance insufficient (top5={imb5:.2f}, top1={imb1:.2f})")

        # 3) aggressor ratio
        ar = metrics.get('aggressor_ratio')
        if ar is None:
            reasons.append("aggressor_ratio_missing")
        else:
            if ar >= self.aggressor_ratio_threshold:
                passed_count += 1
            else:
                reasons.append(f"aggressor_ratio_low {ar:.2f} < {self.aggressor_ratio_threshold}")

        # 4) spread and liquidity
        passed4 = True
        spread = metrics.get('spread', 999999)
        if spread > self.spread_threshold:
            reasons.append(f"spread_too_large {spread:.2f} > {self.spread_threshold}")
            passed4 = False
        if metrics.get('top5_depth_usd', 0.0) < self.min_top5_depth_usd:
            reasons.append(f"low_liquidity top5_depth_usd={metrics.get('top5_depth_usd'):.2f}")
            passed4 = False
        if passed4:
            passed_count += 1

        # 5) ATR/NWE context: избегаем периодов экстремальной ATR
        # Если у atr.bounds есть информация о "ширине" (upper-lower), проверим
        atr_width = None
        if atr and atr.get('upper') and atr.get('lower'):
            atr_width = abs(float(atr['upper']) - float(atr['lower']))
        if atr_width is not None and atr_width > 2.0 * self.max_stop_usd * 10:  # умножаем на 10 для BTC масштабирования; параметр можно подстроить
            reasons.append(f"atr_too_wide {atr_width:.2f}")
        else:
            passed_count += 1

        # 6) sweep detection (простой): нет резких однонаправленных выбиваний вниз в последние orderbook_window_s
        if self._detect_down_sweep():
            reasons.append("recent_down_sweep_detected")
        else:
            passed_count += 1

        # Проверяем, прошли ли хотя бы 5 из 6 условий
        if passed_count >= 5:
            # 7) cancel/spoof detection: простая эвристика (обязательное условие)
            if self._detect_spoofing():
                reasons.append("spoofing_detected")
                return False, reasons
            else:
                return True, reasons
        else:
            # 7) все равно проверяем spoofing, но поскольку уже не прошли, добавляем если detected
            if self._detect_spoofing():
                reasons.append("spoofing_detected")
            return False, reasons

    def _detect_down_sweep(self):
        """
        Простая логика: если в последние orderbook_window_s наблюдалось снижение best_bid ниже
        предыдущих на много уровней (например > 3 ценовых уровней), считаем рискованным.
        """
        if len(self.ob_history) < 2:
            return False
        relevant = [item for item in self.ob_history if item[0] >= datetime.now() - timedelta(seconds=self.orderbook_window_s)]
        if len(relevant) < 2:
            return False
        bids = [float(snap[1]['bids'][0]['price']) for snap in relevant if snap[1]['bids']]
        if not bids:
            return False
        # если best_bid упал более чем на процент_threshold от максим в окне
        max_bid = max(bids)
        min_bid = min(bids)
        if max_bid <= 0:
            return False
        pct_drop = (max_bid - min_bid) / max_bid
        if pct_drop > 0.005:  # 0.5% — порог для BTC можно настроить
            return True
        return False

    def _detect_spoofing(self):
        """
        Простейшая эвристика: если в окне появились большие объёмы на top1/top5 и затем быстро ушли без заметного исполнения,
        то считаем подозрительным. Реализуем по сравнению сумм в средней и последней snapshot.
        """
        if len(self.ob_history) < 4:
            return False
        recent = list(self.ob_history)[-4:]
        sums = []
        for ts, snap in recent:
            bid_sum = sum(float(x.get('amount', 0.0)) for x in snap['bids'][:self.topN])
            ask_sum = sum(float(x.get('amount', 0.0)) for x in snap['asks'][:self.topN])
            sums.append(bid_sum + ask_sum)
        # если сумма резко выросла (появление крупного лимита) и затем резко упала — возможное спуфинг
        if sums[1] > 3.0 * sums[0] and sums[2] < 0.5 * sums[1]:
            return True
        return False

    def _open_trade(self, side, entry_price, sl_distance, tp_distance):
        """
        Создаём запись позиции (симуляция открытие). На реальную биржу нужно подключать execution.
        Позиция содержит OCO: tp и sl.
        """
        size = self._calc_position_size(entry_price, sl_distance)
        trade = {
            'side': side,
            'entry_price': entry_price,
            'sl_price': entry_price - sl_distance if side == 'long' else entry_price + sl_distance,
            'tp_price': entry_price + tp_distance if side == 'long' else entry_price - tp_distance,
            'size': size,
            'opened_at': datetime.now(),
            'status': 'open'
        }
        self.open_trades.append(trade)
        self.log.info("Simulated open trade: %s", trade)

    def _calc_position_size(self, entry_price, sl_distance):
        """
        Позиция задаётся фиксированным риском $ = sl_amount * size -> size = risk / (sl_distance)
        Здесь риск = max_stop_usd; size в BTC = risk / (sl_distance * entry_price) (приблизительно).
        Но т.к. у нас фиксированные $ (10$), для BTC рассчитываем объём:
        size = risk_usd / (sl_distance_in_usd)  (для контрактов/деривативов можно поправить).
        """
        # простая реализация: risk USD = max_stop_usd, size in base currency = risk / sl_distance
        # Note: для реального расчёта нужно учитывать то, в каких единицах деноминирован контракт.
        if sl_distance <= 0:
            return 0.0
        size = self.max_stop_usd / sl_distance
        return size

    def _update_open_trades(self):
        """
        Проверяем открытые позиции относительно текущего microprice и закрываем по tp/sl.
        В реальном исполнении нужно отслеживать рыночные цены/транзакции.
        """
        if not self.open_trades:
            return
        # используем последний snapshot для определения текущ price
        if not self.ob_history:
            return
        current_ob = self.ob_history[-1][1]
        metrics = self._compute_metrics(current_ob)
        current_price = metrics.get('microprice') or (metrics.get('best_ask') + metrics.get('best_bid')) / 2.0

        remaining = []
        for trade in self.open_trades:
            closed = False
            if trade['side'] == 'long':
                if current_price >= trade['tp_price']:
                    self._close_trade(trade, 'tp', current_price)
                    closed = True
                elif current_price <= trade['sl_price']:
                    self._close_trade(trade, 'sl', current_price)
                    closed = True
            else:
                if current_price <= trade['tp_price']:
                    self._close_trade(trade, 'tp', current_price)
                    closed = True
                elif current_price >= trade['sl_price']:
                    self._close_trade(trade, 'sl', current_price)
                    closed = True
            if not closed:
                remaining.append(trade)
        self.open_trades = remaining

    def _close_trade(self, trade, reason, exit_price):
        trade['closed_at'] = datetime.now()
        trade['exit_price'] = exit_price
        trade['status'] = 'closed'
        trade['close_reason'] = reason
        # statistics
        pnl = 0.0
        if trade['side'] == 'long':
            pnl = (exit_price - trade['entry_price']) * trade['size']
        else:
            pnl = (trade['entry_price'] - exit_price) * trade['size']
        trade['pnl'] = pnl
        self.log.info("Closed trade: reason=%s entry=%.2f exit=%.2f size=%.6f pnl=%.4f", reason, trade['entry_price'], exit_price, trade['size'], pnl)
        # сохраняем историю
        self.stats['closed_trades'] += 1
        # Реальная торговля: здесь можно отправить запись в систему учёта/бэкап на диск.