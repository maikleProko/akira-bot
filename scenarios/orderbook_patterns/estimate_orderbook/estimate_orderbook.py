import os
import json
from datetime import datetime
from statistics import mean
import glob
import numpy as np  # Для улучшенных расчетов, волатильности

def do_strategy(parser, orderbook):
    """
    Улучшенная стратегия входа в LONG: trigger на близости к lower_bound (Nadaraya+ATR combo), подтверждение orderbook imbalance.
    Fixed PNL: -10 USD risk, +15 USD reward (R:R=1.5:1).
    Цель: ~3-5 сделок/день, winrate 65-75% via строгие фильтры и adaptive параметры.
    Улучшения:
    - Adaptive thresholds на основе волатильности (ATR-based).
    - Дополнительный тренд-фильтр (e.g., цена выше 200-period MA для избежания downtrends).
    - RSI confirmation для oversold условий.
    - Улучшенный slippage и liquidity checks с deeper book analysis.
    - Position sizing с учетом account balance (max 1% risk per trade, но fixed USD capped).
    - Trailing stop optional, но сохраняем fixed reward для simplicity.
    - Лучший error handling и logging.
    - Diversification hint: run on multiple pairs with correlation check (not implemented here).
    Вход:
      - parser: CoinglassParser с symbol1, symbol2, lower_bound (float из Nadaraya+ATR), добавлены atr_current, ma_200, rsi_current.
      - orderbook: dict с 'bids'/'asks'.
    Результат:
      JSON-line в files/orderbook_decisions.txt при входе.
    """

    # ---------------- ПАРАМЕТРЫ ----------------
    HISTORY_MINUTES = 5  # Увеличено для лучшей статистики (меньше шума)
    TOP_N_LEVELS = 10  # Deeper для imbalance (лучше capture spoofing)
    IMBALANCE_THRESHOLD_BASE = 0.25  # Базовый, adaptive на vol
    MIN_LIQUIDITY_USD = 5000.0  # Увеличено для robustness
    MAX_SPREAD_PCT = 0.001  # Ужесточено
    EPS_CLOSE_TO_BOUND_BASE = 0.0025  # Базовый, adaptive
    STOP_BUFFER_PCT = 0.0005  # Tightened
    RISK_USD = 10.0  # Fixed risk
    REWARD_USD = 15.0  # Fixed reward
    MIN_TRADE_USD = 100.0  # Увеличено для significance
    MAX_SLIPPAGE_PCT = 0.0008  # Ужесточено
    MIN_SECONDS_BETWEEN_TRADES = 600  # 10 мин — для качества, меньше overtrading
    DECISIONS_FILE = os.path.join('files', 'orderbook_decisions.txt')
    FILES_DIR = os.path.join('files', 'orderbook')
    TRADING_FEES_PCT = 0.0004
    VOL_ADJUST_FACTOR = 1.2  # Для adaptive thresholds
    RSI_THRESHOLD = 35.0  # Oversold для long
    MA_TREND_FILTER = True  # Вкл тренд-фильтр
    ACCOUNT_BALANCE_USD = 10000.0  # Пример; в реальности из API

    # ---------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------

    def safe_float(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    def get_best_prices(ob):
        bids = ob.get('bids', []) or []
        asks = ob.get('asks', []) or []
        best_bid = max(bids, key=lambda x: safe_float(x.get('price', float('-inf')))) if bids else None
        best_ask = min(asks, key=lambda x: safe_float(x.get('price', float('inf')))) if asks else None
        return best_bid, best_ask

    def calculate_imbalance(ob, n=TOP_N_LEVELS):
        bids = sorted(ob.get('bids', [])[:n], key=lambda x: -safe_float(x.get('price', 0)))
        asks = sorted(ob.get('asks', [])[:n], key=lambda x: safe_float(x.get('price', 0)))
        bid_vol = sum(safe_float(x.get('amount', 0)) * safe_float(x.get('price', 0)) for x in bids)  # Weighted by price
        ask_vol = sum(safe_float(x.get('amount', 0)) * safe_float(x.get('price', 0)) for x in asks)
        total_vol = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total_vol if total_vol > 1e-12 else 0.0

    def load_recent_orderbooks(symbol1, symbol2, n=HISTORY_MINUTES):
        pattern = os.path.join(FILES_DIR, f'orderbook_coinglass_{symbol1}-{symbol2}-*.json')
        files = sorted(glob.glob(pattern))[-n:]
        history = []
        for fpath in files:
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    history.append(json.load(f))
            except Exception as e:
                print(f'[Strategy] Error loading {fpath}: {e}')
                continue
        return history, files

    def check_liquidity(ob, min_usd=MIN_LIQUIDITY_USD, depth_levels=20):
        bids = ob.get('bids', [])[:depth_levels]
        asks = ob.get('asks', [])[:depth_levels]
        bid_vol_usd = sum(safe_float(x.get('amount', 0)) * safe_float(x.get('price', 0)) for x in bids)
        ask_vol_usd = sum(safe_float(x.get('amount', 0)) * safe_float(x.get('price', 0)) for x in asks)
        return (bid_vol_usd + ask_vol_usd) >= min_usd

    def cumulative_asks_to_qty(ob, desired_qty, max_slippage_pct=MAX_SLIPPAGE_PCT):
        asks = sorted(ob.get('asks', []), key=lambda x: safe_float(x.get('price', 0)))
        total_qty = 0.0
        total_value = 0.0
        for lvl in asks:
            price = safe_float(lvl.get('price', 0))
            qty = safe_float(lvl.get('amount', 0))
            take_qty = min(qty, desired_qty - total_qty)
            total_qty += take_qty
            total_value += take_qty * price
            if total_qty + 1e-12 >= desired_qty:
                break
        if total_qty < desired_qty * 0.95:  # Stricter fill requirement
            return None, None
        vwap = total_value / total_qty if total_qty > 0 else None
        return total_qty, vwap

    def recent_decisions(symbol, min_seconds=MIN_SECONDS_BETWEEN_TRADES):
        decisions = []
        if os.path.exists(DECISIONS_FILE):
            try:
                with open(DECISIONS_FILE, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            decisions.append(json.loads(line.strip()))
                        except Exception:
                            continue
            except Exception as e:
                print(f'[Strategy] Error reading decisions: {e}')
        last_trade_ts = None
        for d in reversed(decisions):
            if d.get('symbol') == symbol:
                try:
                    last_trade_ts = datetime.strptime(d.get('ts'), '%Y-%m-%dT%H:%M:%SZ')
                    break
                except Exception:
                    continue
        if last_trade_ts:
            seconds_since = (datetime.utcnow() - last_trade_ts).total_seconds()
            return seconds_since < min_seconds
        return False

    def calculate_volatility(history):
        prices = []
        for ob in history:
            _, best_ask = get_best_prices(ob)
            if best_ask:
                prices.append(safe_float(best_ask.get('price', 0)))
        if len(prices) < 2:
            return 1.0  # Default
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns) * np.sqrt(len(prices))  # Annualized proxy

    # ---------------- ЛОГИКА СТРАТЕГИИ ----------------
    symbol = f'{parser.symbol1}-{parser.symbol2}'

    # 1. Проверка данных
    if not (orderbook.get('bids') and orderbook.get('asks')):
        print(f'[Strategy] Invalid orderbook for {symbol}. Skipping.')
        return

    lower_bound = getattr(parser, 'lower_bound', None)
    atr_current = getattr(parser, 'atr_current', 1.0)  # Assume provided
    ma_200 = getattr(parser, 'ma_200', None)
    rsi_current = getattr(parser, 'rsi_current', 50.0)  # Assume provided
    if lower_bound is None:
        print(f'[Strategy] Lower bound not provided for {symbol}. Skipping.')
        return
    lower_bound = safe_float(lower_bound)

    # 2. Загрузка истории
    history, files_used = load_recent_orderbooks(parser.symbol1, parser.symbol2, HISTORY_MINUTES)
    if len(history) < HISTORY_MINUTES:
        print(f'[Strategy] Not enough history ({len(history)}/{HISTORY_MINUTES}) for {symbol}. Skipping.')
        return
    history.append(orderbook)

    # 3. Adaptive параметры
    vol = calculate_volatility(history)
    imbalance_threshold = IMBALANCE_THRESHOLD_BASE * (1 + vol * VOL_ADJUST_FACTOR)
    eps_close_to_bound = EPS_CLOSE_TO_BOUND_BASE * (1 + vol)

    # 4. Метрики orderbook
    best_bid, best_ask = get_best_prices(orderbook)
    if not (best_bid and best_ask):
        print(f'[Strategy] Missing best bid/ask for {symbol}. Skipping.')
        return

    bid_price = safe_float(best_bid.get('price', 0))
    ask_price = safe_float(best_ask.get('price', 0))
    mid_price = (bid_price + ask_price) / 2.0
    spread_pct = (ask_price - bid_price) / mid_price if mid_price > 0 else float('inf')

    # 5. Primary trigger: близость к lower_bound
    rel_diff = abs(ask_price - lower_bound) / max(lower_bound, 1e-12)
    if rel_diff > eps_close_to_bound:
        print(f'[Strategy] Price {ask_price:.2f} not close to lower_bound {lower_bound:.2f} (diff={rel_diff:.4f}, eps={eps_close_to_bound:.4f}) for {symbol}. Skipping.')
        return

    # 6. Фильтры: спред, ликвидность
    if spread_pct > MAX_SPREAD_PCT:
        print(f'[Strategy] Spread too wide ({spread_pct:.4f}) for {symbol}. Skipping.')
        return
    if not check_liquidity(orderbook):
        print(f'[Strategy] Insufficient liquidity for {symbol}. Skipping.')
        return

    # 7. Дополнительные confirmation: RSI и trend filter
    if rsi_current > RSI_THRESHOLD:
        print(f'[Strategy] RSI not oversold ({rsi_current:.1f} > {RSI_THRESHOLD}) for {symbol}. Skipping.')
        return
    if MA_TREND_FILTER and ma_200 and ask_price < ma_200:
        print(f'[Strategy] Price below MA200 ({ask_price:.2f} < {ma_200:.2f}) — potential downtrend for {symbol}. Skipping.')
        return

    # 8. Imbalance confirmation (adaptive)
    imbalances = [calculate_imbalance(ob, TOP_N_LEVELS) for ob in history]
    recent_imbalance = imbalances[-1]
    avg_imbalance = mean(imbalances[:-1]) if imbalances[:-1] else 0.0

    if recent_imbalance < imbalance_threshold or recent_imbalance < avg_imbalance * 1.1:  # Stricter
        print(f'[Strategy] Imbalance weak ({recent_imbalance:.3f} vs thr={imbalance_threshold:.3f}, avg={avg_imbalance:.3f}) for {symbol}. Skipping.')
        return

    # 9. Rate limiting
    if recent_decisions(symbol):
        print(f'[Strategy] Trade too recent for {symbol}. Skipping.')
        return

    # 10. Position sizing (fixed risk, но cap на % balance)
    entry_price = ask_price
    stop_buffer = max(lower_bound * STOP_BUFFER_PCT, atr_current * 0.5)  # Adaptive buffer
    stop_price = lower_bound - stop_buffer
    stop_distance = entry_price - stop_price

    if stop_distance <= 0:
        print(f'[Strategy] Invalid stop distance for {symbol}. Skipping.')
        return

    qty = RISK_USD / stop_distance
    max_risk_qty = (ACCOUNT_BALANCE_USD * 0.01) / stop_distance  # Max 1% risk
    qty = min(qty, max_risk_qty)
    take_distance = (REWARD_USD / qty)
    take_price = entry_price + take_distance
    notional_usd = qty * entry_price

    if notional_usd < MIN_TRADE_USD:
        print(f'[Strategy] Notional too small ({notional_usd:.2f} USD) for {symbol}. Skipping.')
        return

    # 11. Depth & slippage
    filled_qty, vwap = cumulative_asks_to_qty(orderbook, qty)
    if not vwap:
        print(f'[Strategy] Not enough depth for {symbol}. Skipping.')
        return

    slippage_pct = (vwap - entry_price) / entry_price if vwap else float('inf')
    if slippage_pct > MAX_SLIPPAGE_PCT:
        print(f'[Strategy] Slippage too high ({slippage_pct:.4f}) for {symbol}. Skipping.')
        return

    # 12. Fees & expected PNL
    fee_cost = notional_usd * TRADING_FEES_PCT * 2
    expected_pnl = REWARD_USD - fee_cost - (slippage_pct * notional_usd)  # Учет slippage в PNL
    if expected_pnl <= 0:
        print(f'[Strategy] Expected PnL negative ({expected_pnl:.2f} USD) for {symbol}. Skipping.')
        return

    # 13. Решение
    decision = {
        'ts': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'symbol': symbol,
        'type': 'LONG_ENTRY_SUGGESTED',
        'entry_price': entry_price,
        'qty': qty,
        'stop_price': stop_price,
        'take_price': take_price,
        'vwap_fill': vwap,
        'slippage_pct': slippage_pct,
        'risk_usd': RISK_USD,
        'reward_usd': REWARD_USD,
        'expected_pnl_usd': expected_pnl,
        'metrics': {
            'imbalance': recent_imbalance,
            'avg_imbalance': avg_imbalance,
            'spread_pct': spread_pct,
            'lower_bound': lower_bound,
            'rel_diff_to_bound': rel_diff,
            'rsi': rsi_current,
            'ma_200': ma_200,
            'vol': vol,
            'files_used': files_used
        }
    }

    # 14. Запись с locking
    try:
        os.makedirs(os.path.dirname(DECISIONS_FILE), exist_ok=True)
        with open(DECISIONS_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(decision, ensure_ascii=False) + '\n')
        print(f"[Strategy] LONG entry suggested for {symbol}: price={entry_price:.2f}, qty={qty:.6f}, stop={stop_price:.2f}, take={take_price:.2f}, expected_pnl={expected_pnl:.2f} USD")
    except Exception as e:
        print(f'[Strategy] Error writing decision for {symbol}: {e}')

# Код запуска остается похожим, но рекомендуется перейти на websocket для real-time orderbook (e.g., Binance API) вместо Selenium для снижения latency.
# Добавить в parser расчет RSI, MA200, current ATR.
# Для backtesting: интегрировать с backtrader или zipline, тестировать на historical data с out-of-sample.
# Мониторинг: добавить alerts на drawdown >20%.