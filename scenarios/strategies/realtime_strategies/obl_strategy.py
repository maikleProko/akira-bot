import time
from collections import deque
from datetime import datetime

class OblStrategy:
    def __init__(self,
                 history_market_parser,
                 orderbook_parser,
                 nwe_bounds_indicator,
                 atr_bounds_indicator,
                 symbol_pair='BTCUSDT',
                 params=None):
        # всё что нам нужно — ссылки на процессы/парсеры
        self.history_market = history_market_parser
        self.orderbook = orderbook_parser
        self.nwe = nwe_bounds_indicator
        self.atr = atr_bounds_indicator
        self.symbol = symbol_pair

        # Параметры стратегии (по умолчанию, поменяй при необходимости)
        default_params = {
            # стакан/агрегация
            'depth_levels': 30,  # количество уровней сверху стакана для агрегации
            'cluster_mult': 5.0,  # умножитель среднего бина для детекции кластера
            'cluster_abs_min': 0.5,  # абсолютный порог объёма кластера (BTC эквивалента)
            'cluster_price_tol': 5.0,  # tolerance по цене в USD при сравнении кластеров (бин)
            't_sustain': 4,  # сколько сек последних snapshot'ов кластер должен присутствовать
            # orderflow approx
            'w_seconds': 6,  # окно для агрессивного flow (в сек)
            'imbalance_thresh': 0.60,  # порог imbalance для подтверждения
            # фильтры
            'spread_max_ticks': 30.0,  # макс спред в USD (tweak на тикер)
            'liquidity_min': 0.8,  # минимальный суммарный объём в depth_levels (BTC эквивалент)
            'momentum_reject_secs': 2,  # окно для проверки агрессивного импульса
            'momentum_threshold_ticks': 20.0,  # если цена движется против входа быстрее этого -> reject
            # tolerances to boundary
            'boundary_tol_abs': 20.0,  # абсолютная толерантность при касании границы (в USD)
            # wait/cooldowns
            'limit_twait': 1.0,  # max время ожидания лимитного входа (сек) - для будущей робастности
            'post_loss_cooldown': 30,  # после стопа не входить в том же направлении (сек)
            'post_win_cooldown': 5,  # после успешного входа краткий cooldown (сек)
            # general
            'min_data_age': 0.0  # требование свежести данных (можно 0)
        }
        if params:
            default_params.update(params)
        self.p = default_params

        # state: истории стакана последних N секунд (для устойчивости кластеров)
        self.orderbook_history = deque(maxlen=max(10, self.p['t_sustain'] + 5))
        # для orderflow approx — хранить агрессивные события (buy/sell volumes) по сек
        self.flow_history = deque(maxlen=self.p['w_seconds'] + 5)  # элементы: dict{'buy':x,'sell':y,'ts':t}
        # хранить предыдущий snapshot для вычисления дельт
        self.prev_snapshot = None

        # active cooldown timers: {'long': ts_until_allowed, 'short': ts}
        self.cooldowns = {'long': 0.0, 'short': 0.0}

        # лог сигналов (в памяти). Можно также выбрасывать в файл.
        self.signals_log = []

    def prepare(self, start_time=None, end_time=None):
        # подготовка — ничего специфичного нужно
        print(f"[OblStrategy] prepared for {self.symbol}")

    def _safe_get_orderbook(self):
        ob = getattr(self.orderbook, 'orderbook', None)
        if not ob:
            return None
        bids = ob.get('bids', [])
        asks = ob.get('asks', [])
        if not bids or not asks:
            return None
        # приводим к ожидаемой форме: списки цен/amounts, упорядочены
        return {'bids': bids, 'asks': asks, 'ts': time.time()}

    def _agg_top_depth(self, snapshot, depth):
        # Возвращает сумму amount для топ depth уровней по bids и asks, а также средний бин-объём
        bids = snapshot['bids'][:depth]
        asks = snapshot['asks'][:depth]
        sum_bids = sum([b['amount'] for b in bids])
        sum_asks = sum([a['amount'] for a in asks])
        avg_bin = (sum([b['amount'] for b in bids]) + sum([a['amount'] for a in asks])) / max(1,
                                                                                              (len(bids) + len(asks)))
        best_bid = bids[0]['price']
        best_ask = asks[0]['price']
        return {'sum_bids': sum_bids, 'sum_asks': sum_asks, 'avg_bin': avg_bin, 'best_bid': best_bid,
                'best_ask': best_ask,
                'bids': bids, 'asks': asks}

    def _detect_clusters(self, snapshot, depth):
        """
        Находит 'кластеры' на bids/asks среди топ depth уровней по правилам:
        bin_amount > max(avg_bin * cluster_mult, cluster_abs_min)
        Возвращает список кластеров (price, amount) для bids и asks
        """
        agg = self._agg_top_depth(snapshot, depth)
        clusters = {'bids': [], 'asks': []}
        avg_bin = agg['avg_bin']
        min_threshold = max(self.p['cluster_abs_min'], avg_bin * self.p['cluster_mult'])
        # bids
        for b in agg['bids']:
            if b['amount'] >= min_threshold:
                clusters['bids'].append({'price': b['price'], 'amount': b['amount']})
        for a in agg['asks']:
            if a['amount'] >= min_threshold:
                clusters['asks'].append({'price': a['price'], 'amount': a['amount']})
        return clusters, agg

    def _cluster_sustained(self, price, side):
        """
        Проверить, есть ли кластер около price на предыдущих snapshot'ах непрерывно T_sustain сек
        Мы считаем кластер 'присутствующим' в snapshot, если в нем найден уровень в пределах cluster_price_tol USD.
        """
        needed = self.p['t_sustain']
        if needed <= 1:
            return True
        now = time.time()
        found_count = 0
        # проходим назад по history; считаем сколько последних секунд подряд есть кластер
        # note: orderbook_history содержит snapshot'ы от более свежих к старым (append справа), но deque iteration слабозависим
        # будем считать непрерывность: для каждой секунды должно быть snapshot с кластером
        # упрощение: проверим последние needed snapshots: во всех ли из них присутствует подходящий кластер
        if len(self.orderbook_history) < needed:
            return False
        # берем последние needed snapshots
        last_snaps = list(self.orderbook_history)[-needed:]
        tol = self.p['cluster_price_tol']
        for s in last_snaps:
            clusters, _ = self._detect_clusters(s, self.p['depth_levels'])
            # проверим side
            present = False
            for c in clusters.get(side, []):
                if abs(c['price'] - price) <= tol:
                    present = True
                    break
            if not present:
                return False
        return True

    def _update_flow(self, prev_snap, curr_snap):
        """
        Простая approx of taker flow между двумя snapshot'ами:
        - агрессивные покупки (taker buys) — если верхние ask уровни уменьшились (consumed)
        - агрессивные продажи (taker sells) — если верхние bid уровни уменьшились
        Возвращаем dict{'buy':vol,'sell':vol}
        """
        if not prev_snap or not curr_snap:
            return {'buy': 0.0, 'sell': 0.0}
        # используем только топ 5 уровней для агрессивности
        topN = min(6, len(prev_snap['asks']), len(curr_snap['asks']))
        buy_vol = 0.0
        for i in range(topN):
            prev_a = prev_snap['asks'][i]['amount']
            curr_a = curr_snap['asks'][i]['amount']
            diff = prev_a - curr_a
            if diff > 0:
                buy_vol += diff
        topN2 = min(6, len(prev_snap['bids']), len(curr_snap['bids']))
        sell_vol = 0.0
        for i in range(topN2):
            prev_b = prev_snap['bids'][i]['amount']
            curr_b = curr_snap['bids'][i]['amount']
            diff = prev_b - curr_b
            if diff > 0:
                sell_vol += diff
        return {'buy': buy_vol, 'sell': sell_vol}

    def _price_movement_over(self, window_secs):
        """
        Возвращает изменение mid-price за последние window_secs (используем orderbook_history snapshots).
        """
        if not self.orderbook_history:
            return 0.0
        snaps = list(self.orderbook_history)
        now = time.time()
        # ищём самый старый snapshot, не старше window_secs
        ref = None
        for s in reversed(snaps):  # reversed because we appended in order; newest at right
            if now - s.get('ts', now) >= window_secs:
                ref = s
                break
        if not ref:
            # если нет достаточно старого — берём самый старый в истории
            ref = snaps[0]
        # current mid:
        curr = snaps[-1]
        ref_mid = (ref['bids'][0]['price'] + ref['asks'][0]['price']) / 2.0
        curr_mid = (curr['bids'][0]['price'] + curr['asks'][0]['price']) / 2.0
        return curr_mid - ref_mid

    def _now(self):
        return time.time()

    def run_realtime(self):
        # основной цикл, вызывается каждую секунду
        ts_now = self._now()

        # 1) прочитать данные
        # проверяем доступность данных у парсеров/индикаторов
        hist_df = getattr(self.history_market, 'df', None)
        ob_snapshot = self._safe_get_orderbook()
        nwe_bounds = getattr(self.nwe, 'bounds', None) or getattr(self.nwe, 'last_bounds', None) or getattr(self.nwe,
                                                                                                            'value',
                                                                                                            None)
        atr_bounds = getattr(self.atr, 'bounds', None) or getattr(self.atr, 'last_bounds', None) or getattr(self.atr,
                                                                                                            'value',
                                                                                                            None)

        if (hist_df is None) or (ob_snapshot is None) or (nwe_bounds is None) or (atr_bounds is None):
            # недостаточно данных в текущий тик — выходим
            # print("[OblStrategy] waiting for data...")
            return

        # 2) проверка на свежесть (optionally)
        if self.p['min_data_age'] > 0:
            if ts_now - ob_snapshot['ts'] > self.p['min_data_age']:
                # stale data
                return


        # 3) обновляем истории стакана для устойчивости кластеров
        self.orderbook_history.append(ob_snapshot)

        # 4) вычисляем approx orderflow между двумя двумя последними snapshot'ами
        flow_delta = self._update_flow(self.prev_snapshot, ob_snapshot) if self.prev_snapshot else {'buy': 0.0,
                                                                                                    'sell': 0.0}
        # Записываем в flow_history со временем
        self.flow_history.append({'ts': ts_now, 'buy': flow_delta['buy'], 'sell': flow_delta['sell']})

        # Суммарный flow за окно
        total_buy = sum([x['buy'] for x in self.flow_history])
        total_sell = sum([x['sell'] for x in self.flow_history])
        flow_imbalance = 0.0
        if total_buy + total_sell > 0:
            flow_imbalance = (total_buy - total_sell) / (total_buy + total_sell)

        # 5) boundary calculation
        try:
            lower_boundary = min(float(nwe_bounds.get('lower')), float(atr_bounds.get('lower')))
            upper_boundary = max(float(nwe_bounds.get('upper')), float(atr_bounds.get('upper')))
        except Exception:
            # некорректные bounds
            return

        # 6) price / spread / liquidity checks
        best_bid = ob_snapshot['bids'][0]['price']
        best_ask = ob_snapshot['asks'][0]['price']
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid

        # aggregated liquidity top D
        depthD = self.p['depth_levels']
        agg = self._agg_top_depth(ob_snapshot, depthD)
        total_top_liq = agg['sum_bids'] + agg['sum_asks']

        # 7) detect clusters in current snapshot
        clusters, agg2 = self._detect_clusters(ob_snapshot, depthD)
        # for decision we want nearest sustained bid cluster below price and sustained ask cluster above
        nearest_bid_cluster = None
        for c in clusters['bids']:
            # ищем ближайший кластер ниже mid_price (или ниже best_bid)
            if c['price'] <= mid_price + 0.0001:
                # pick the one closest to mid
                if (nearest_bid_cluster is None) or (
                        abs(mid_price - c['price']) < abs(mid_price - nearest_bid_cluster['price'])):
                    nearest_bid_cluster = c
        nearest_ask_cluster = None
        for c in clusters['asks']:
            if c['price'] >= mid_price - 0.0001:
                if (nearest_ask_cluster is None) or (
                        abs(mid_price - c['price']) < abs(mid_price - nearest_ask_cluster['price'])):
                    nearest_ask_cluster = c

        # 8) cluster sustained check
        bid_cluster_sustained = False
        if nearest_bid_cluster:
            bid_cluster_sustained = self._cluster_sustained(nearest_bid_cluster['price'], 'bids')
        ask_cluster_sustained = False
        if nearest_ask_cluster:
            ask_cluster_sustained = self._cluster_sustained(nearest_ask_cluster['price'], 'asks')

        # 9) momentum check: reject if price moving strongly against direction over momentum_reject_secs
        recent_price_move = self._price_movement_over(self.p['momentum_reject_secs'])  # delta mid
        # 10) decision thresholds
        ts = ts_now
        signals = []

        # LONG decision
        # conditions summary:
        # - mid_price is at or below lower_boundary + tol
        # - there is a sustained bid_cluster at/under price (nearest_bid_cluster)
        # - flow_imbalance >= threshold OR recent aggressive buy volume > sell
        # - spread <= max
        # - liquidity >= min
        # - not in cooldown

        if mid_price <= (lower_boundary + self.p['boundary_tol_abs']):
            # cluster exist and near the price (or below)
            cond_cluster = (nearest_bid_cluster is not None) and bid_cluster_sustained
            cond_spread = (spread <= self.p['spread_max_ticks'])
            cond_liq = (total_top_liq >= self.p['liquidity_min'])
            cond_flow = (flow_imbalance >= self.p['imbalance_thresh']) or (total_buy > total_sell and total_buy > 0)
            cond_momentum = not (
                        recent_price_move < -self.p['momentum_threshold_ticks'])  # если цена резко падает — reject
            cond_cooldown = (ts >= self.cooldowns['long'])
            print(f'[OblStrategy] conds: [{str(cond_cluster)}, {str(cond_spread)}, {str(cond_liq)}, {str(cond_flow)}, {str(cond_momentum)}, {str(cond_cooldown)}]')
            passed = sum([cond_cluster, cond_spread, cond_liq, cond_flow, cond_momentum, cond_cooldown])
            if passed >= 5:
                # формируем сигнал LONG
                entry_price = best_ask  # либо mid или machine_learning; для сигнала возьмём текущую лучшую цену исполнения ask
                # стоп можем вычислить как lower_boundary - buffer
                stop_price = lower_boundary - 1.0  # buffer 1 USD; tune as needed
                take_price = entry_price + (entry_price - stop_price) * 1.5
                signal = {
                    'ts': datetime.utcfromtimestamp(ts).isoformat(),
                    'side': 'LONG',
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'take_price': take_price,
                    'reason': 'touch_lower_boundary + sustained_bid_cluster + flow_confirm',
                    'mid': mid_price,
                    'lower_boundary': lower_boundary,
                    'upper_boundary': upper_boundary,
                    'spread': spread,
                    'total_top_liq': total_top_liq,
                    'flow_imbalance': flow_imbalance,
                    'nearest_bid_cluster': nearest_bid_cluster
                }
                signals.append(signal)
                # ставим короткий cooldown на повтор
                self.cooldowns['long'] = ts + self.p['post_win_cooldown']  # если будем считать на будущее
                # логгируем
                self.signals_log.append(signal)
                message = "[OblStrategy][SIGNAL] LONG"
                print(message)
                with open('files/decisions/decisions_OblStrategy.txt', 'a') as f:
                    f.write(message + '\n')

        # SHORT decision (mirror)
        if mid_price >= (upper_boundary - self.p['boundary_tol_abs']):
            cond_cluster = (nearest_ask_cluster is not None) and ask_cluster_sustained
            cond_spread = (spread <= self.p['spread_max_ticks'])
            cond_liq = (total_top_liq >= self.p['liquidity_min'])
            cond_flow = (flow_imbalance <= -self.p['imbalance_thresh']) or (total_sell > total_buy and total_sell > 0)
            cond_momentum = not (recent_price_move > self.p['momentum_threshold_ticks'])
            cond_cooldown = (ts >= self.cooldowns['short'])
            print(f'[OblStrategy] conds: [{str(cond_cluster)}, {str(cond_spread)}, {str(cond_liq)}, {str(cond_flow)}, {str(cond_momentum)}, {str(cond_cooldown)}]')
            passed = sum([cond_cluster, cond_spread, cond_liq, cond_flow, cond_momentum, cond_cooldown])
            if passed >= 5:
                entry_price = best_bid
                stop_price = upper_boundary + 1.0
                take_price = entry_price - (stop_price - entry_price) * 1.5
                signal = {
                    'ts': datetime.utcfromtimestamp(ts).isoformat(),
                    'side': 'SHORT',
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'take_price': take_price,
                    'reason': 'touch_upper_boundary + sustained_ask_cluster + flow_confirm',
                    'mid': mid_price,
                    'lower_boundary': lower_boundary,
                    'upper_boundary': upper_boundary,
                    'spread': spread,
                    'total_top_liq': total_top_liq,
                    'flow_imbalance': flow_imbalance,
                    'nearest_ask_cluster': nearest_ask_cluster
                }
                signals.append(signal)
                self.cooldowns['short'] = ts + self.p['post_win_cooldown']
                self.signals_log.append(signal)
                print("[OblStrategy][SIGNAL] SHORT", signal)

        # optional: если захотите — можно отправлять сигналы в файл/ брокеру тут
        # например: self._emit_signal(signal)

        # update prev_snapshot
        self.prev_snapshot = ob_snapshot

        # можно вернуть signals (для тестов/юнит-тестов)
        return signals